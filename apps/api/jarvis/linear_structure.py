"""Optional mapping of Linear teams and projects onto Homes (structure onboarding).

The owner's choices are data on the connection (`LinearConnection.structure`). `reconcile` computes the
whole outcome (schema changes, Homes, tree, issue counts, affected imported issues): the preview runs it
without writing and apply runs the same code with writes. Sync calls `place` for the issues it touched, so
new issues inherit their mapped Home and repeated runs change nothing.

Only an imported task's record Home moves. `task.project_id` is the Linear project, a source field: it
changes only through the Linear write-back policy (`linear_sync.before_task_update`), never here.

State shape (`conn.structure`, valid only for `workspace_id`):
  status        "" (never decided) | "applied" | "skipped"
  catalog       teams, projects (name, archived, team_ids) and issue counts read from Linear by `discover`
  meanings      {"team": type_id | None, "project": type_id | None}, the owner's confirmed meaning
  rules         {"team:<id>" | "project:<id>": {action, record_id, created, name, archived}}
  pending       {fingerprint, proposal_id}: the structure proposal a preview made for its schema changes
  applied_fingerprint, applied_at, revision
"""

import hashlib
import json
import re
from copy import deepcopy
from types import SimpleNamespace
from typing import Literal
from uuid import NAMESPACE_URL, uuid4, uuid5

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import func, select

from .db import session_scope
from .domain import DomainError, advisory, emit, owned, serial
from .google_auth import unseal
from .linear_client import LinearClient, LinearFailure
from .linear_sync import connection
from .models import LinearIssue, Task, now
from .structure_models import StructureRecord

KINDS = ("team", "project")
# Links whose task is not managed here: unlinked by the owner, or no longer in the synced scope.
QUIET = {"unlinked", "out_of_scope", "unavailable"}


class Body(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Meaning(Body):
    """What a Linear team or project is in Eridani: an existing type, or a new container type."""

    type_id: str | None = Field(default=None, min_length=1, max_length=80)
    new_type: str | None = Field(default=None, min_length=1, max_length=120)

    @model_validator(mode="after")
    def one(self):
        if (self.type_id is None) == (self.new_type is None):
            raise ValueError("Choose an existing type or name a new one.")
        return self


class Rule(Body):
    source: Literal["team", "project"]
    source_id: str = Field(min_length=1, max_length=100)
    action: Literal["reuse", "create", "skip"]
    record_id: str | None = Field(default=None, max_length=36)  # reuse: the existing Home
    title: str | None = Field(default=None, min_length=1, max_length=500)  # create: defaults to the Linear name
    # create: "team" (a project goes inside its team's Home), "top", or an existing record id.
    parent: str = Field(default="team", min_length=1, max_length=36)


class Mapping(Body):
    team: Meaning | None = None
    project: Meaning | None = None
    rules: list[Rule] = Field(default_factory=list, max_length=500)


class Apply(Mapping):
    fingerprint: str = Field(min_length=64, max_length=64)


def current(conn):
    state = conn.structure or {}
    # A mapping names one Linear workspace's IDs; reconnecting to another workspace starts fresh.
    return deepcopy(state) if state.get("workspace_id") == conn.workspace_id else {}


def save(conn, state):
    conn.structure = {**state, "workspace_id": conn.workspace_id, "revision": state.get("revision", 0) + 1}


def overview(conn):
    """The connection summary's part: whether to offer the review step."""
    state = current(conn)
    status = state.get("status", "")
    return {
        "status": status,
        "offer": bool(conn.enabled and conn.team_ids and status not in {"applied", "skipped"}),
        "homes": sum(1 for r in state.get("rules", {}).values() if r.get("record_id")),
        "applied_at": state.get("applied_at"),
        "discovered_at": (state.get("catalog") or {}).get("discovered_at"),
    }


def legacy_project_id(owner, workspace, project_id):
    """The local Project row `linear_sync.local_project` keeps for a Linear project."""
    return str(uuid5(NAMESPACE_URL, "linear:" + owner + ":" + workspace + ":" + project_id))


def discover(owner):
    """Read the selected teams' projects and issue counts from Linear. Nothing else changes."""
    with session_scope() as db:
        conn = connection(db, owner)
        if not conn.team_ids:
            raise DomainError("LINEAR_SCOPE", "Select the Linear teams to sync first.", 409)
        key = unseal(conn.credentials)["api_key"]
        teams, only_mine, viewer = list(conn.team_ids), conn.only_mine, conn.viewer_id
        generation, workspace = conn.generation, conn.workspace_id
    client = LinearClient(key)
    try:
        meta = client.identity()
        if meta["organization"]["id"] != workspace:
            raise LinearFailure("workspace_changed")
        projects = client.projects()
        scope = {"team": {"id": {"in": teams}}}
        if only_mine:
            scope["assignee"] = {"id": {"eq": viewer}}
        places = client.issue_places(scope)
    except (LinearFailure, KeyError, TypeError):
        raise DomainError(
            "LINEAR_UNAVAILABLE", "Could not read your Linear teams and projects. Nothing changed; try again.", 503
        ) from None
    finally:
        client.close()
    counts = {}
    for row in places:
        cell = counts.setdefault(row["team"]["id"], {})
        project = (row.get("project") or {}).get("id") or ""
        cell[project] = cell.get(project, 0) + 1
    used = {p for cells in counts.values() for p in cells}
    catalog = {
        "teams": [{"id": t["id"], "name": t["name"], "key": t.get("key")} for t in meta["teams"] if t["id"] in teams],
        "projects": [
            {
                "id": p["id"],
                "name": p["name"],
                "archived": bool(p.get("archivedAt")),
                "team_ids": [t["id"] for t in (p.get("teams") or {}).get("nodes", [])],
            }
            for p in projects
            if p["id"] in used or {t["id"] for t in (p.get("teams") or {}).get("nodes", [])} & set(teams)
        ],
        "counts": counts,
        "team_ids": teams,
        "only_mine": only_mine,
        "discovered_at": now().isoformat(),
    }
    with session_scope() as db:
        advisory(db, f"workspace:{owner}")
        conn = connection(db, owner, generation)
        state = current(conn)
        state["catalog"] = catalog
        follow_sources(db, owner, state, sources(catalog["teams"], catalog["projects"]))
        save(conn, state)
        emit(db, owner, "linear.changed", owner)
        return view(db, owner, conn)


def sources(teams, projects):
    return {
        **{"team:" + t["id"]: {"name": t["name"], "archived": False} for t in teams},
        **{
            "project:" + p["id"]: {"name": p["name"], "archived": bool(p.get("archived") or p.get("archivedAt"))}
            for p in projects
        },
    }


def follow_sources(db, owner, state, seen):
    """Record renamed/archived sources. A Home created from a source follows a rename only while its title
    is still the old Linear name; a local title is kept. Archiving in Linear never archives a Home."""
    rules, changed = dict(state.get("rules", {})), False
    for key, rule in rules.items():
        source = seen.get(key)
        if not source or (rule.get("name"), rule.get("archived", False)) == (source["name"], source["archived"]):
            continue
        if rule.get("name") != source["name"] and rule.get("created") and rule.get("record_id"):
            rename(db, owner, rule["record_id"], rule.get("name"), source["name"])
        rules[key] = {**rule, "name": source["name"], "archived": source["archived"]}
        changed = True
    if changed:
        state["rules"] = rules
    return changed


def rename(db, owner, record_id, old, new):
    from .structure import ensure
    from .structure import mutate as structure_mutate
    from .structure_schema import RecordUpdate

    record = db.get(StructureRecord, record_id)
    if not record or record.owner_id != owner or record.archived or record.title != old:
        return
    schema = ensure(db, owner)
    try:
        structure_mutate(
            db,
            owner,
            "record.update",
            RecordUpdate(
                record_id=record.id, expected_revision=record.revision, schema_revision=schema.revision, title=new[:500]
            ),
            "linear-mapping:" + str(uuid4()),
        )
    except DomainError:
        pass  # A Home that cannot be renamed keeps its title; the mapping still holds by source ID.


def view(db, owner, conn):
    """The review step's data: catalog with counts, saved choices, types and candidate Homes."""
    from .structure import ensure

    schema = ensure(db, owner)
    state = current(conn)
    catalog = state.get("catalog")
    if catalog:
        counts = catalog["counts"]
        catalog = {
            **{k: v for k, v in catalog.items() if k != "counts"},
            "teams": [{**t, "issues": sum(counts.get(t["id"], {}).values())} for t in catalog["teams"]],
            "projects": [
                {**p, "issues": sum(cells.get(p["id"], 0) for cells in counts.values())} for p in catalog["projects"]
            ],
            "unassigned": {team: cells.get("", 0) for team, cells in counts.items()},
        }
    homes = candidate_homes(db, owner, schema)
    titles = {h["id"]: h for h in homes}
    return {
        "status": state.get("status", ""),
        "revision": state.get("revision", 0),
        "catalog": catalog,
        "meanings": state.get("meanings", {}),
        "rules": [
            {"key": key, **rule, "home": titles.get(rule.get("record_id"))}
            for key, rule in state.get("rules", {}).items()
        ],
        "types": [
            {"id": t["id"], "name": t["name"], "plural": t["plural"], "capabilities": t["capabilities"]}
            for t in schema.definition["types"]
            if not t["archived"]
        ],
        "homes": homes,
        "applied_at": state.get("applied_at"),
    }


def candidate_homes(db, owner, schema):
    """Active records that can hold work: not the imported task type, not pure notes."""
    from .structure import import_type

    try:
        task_type = import_type(schema, "task")["id"]
    except DomainError:
        task_type = None
    types = {t["id"]: t for t in schema.definition["types"]}
    rows = list(
        db.execute(
            select(StructureRecord.id, StructureRecord.title, StructureRecord.parent_id, StructureRecord.type_id,
                   StructureRecord.archived).where(StructureRecord.owner_id == owner)
        )
    )
    index = {r.id: r for r in rows}

    def path(row):
        names, seen = [], set()
        while row and row.id not in seen and len(names) < 20:
            seen.add(row.id)
            names.append(row.title)
            row = index.get(row.parent_id)
        return " / ".join(reversed(names))[-300:]

    result = []
    for row in rows:
        t = types.get(row.type_id)
        caps = set(t["capabilities"]) if t else set()
        if row.archived or not t or t["archived"] or row.type_id == task_type or caps == {"content"}:
            continue
        result.append({"id": row.id, "title": row.title, "type_id": row.type_id, "type_name": t["name"],
                       "path": path(row)})
    return sorted(result, key=lambda h: h["path"].casefold())[:500]


def slug(name, taken):
    base = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")[:60] or "linear-home"
    candidate, n = base, 2
    while candidate in taken:
        candidate, n = f"{base}-{n}", n + 1
    return candidate


def resolve_type(definition, types, meaning, kind, changes):
    if meaning.type_id:
        t = types.get(meaning.type_id)
        if not t or t["archived"]:
            raise DomainError("INVALID_TYPE", "Choose an active type for Linear " + kind + "s.")
        return t["id"]
    name = meaning.new_type
    same = next((t for t in types.values() if not t["archived"] and t["name"].casefold() == name.casefold()), None)
    if same:
        return same["id"]  # Repeating setup reuses the type it created.
    t = {
        "id": slug(name, set(types)),
        "name": name,
        "plural": name if name.endswith("s") else name + "s",
        "description": "A Home for work from Linear " + kind + "s, named during Linear setup.",
        "capabilities": [],
        "parent_types": [],
        "fields": [],
        "statuses": [],
        "archived": False,
    }
    definition["types"].append(t)
    types[t["id"]] = t
    changes.append({"kind": "new_type", "type_id": t["id"], "text": "New type “" + name + "” for Linear " + kind + "s"})
    return t["id"]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()


def reconcile(db, owner, conn, mapping, *, write=False, previewed=None):
    """The one code path for preview (write=False) and apply (write=True).

    previewed: the fingerprint the owner confirmed, remembered so a retried apply is recognised."""
    from .structure import ensure, import_type

    state = current(conn)
    catalog = state.get("catalog")
    if not catalog:
        raise DomainError("LINEAR_DISCOVER", "Read your Linear teams and projects first.", 409)
    schema = ensure(db, owner)
    definition = deepcopy(schema.definition)
    types = {t["id"]: t for t in definition["types"]}
    saved = state.get("rules", {})
    found = {("team", t["id"]): t for t in catalog["teams"]} | {("project", p["id"]): p for p in catalog["projects"]}
    given = {}
    for rule in mapping.rules:
        if (rule.source, rule.source_id) not in found:
            raise DomainError("LINEAR_SOURCE", "Your Linear teams or projects changed. Read them again.", 409)
        given[(rule.source, rule.source_id)] = rule
    changes, meanings = [], {}
    for kind in KINDS:
        if not any(r.action != "skip" and r.source == kind for r in given.values()):
            meanings[kind] = None
            continue
        meaning = getattr(mapping, kind)
        if meaning is None:
            # Never assume a team or project is a Client (or anything else): the owner names it.
            raise DomainError(
                "LINEAR_MEANING",
                "What is a Linear " + kind + " in Eridani? Choose a type or name a new one.",
                data={"source": kind},
            )
        meanings[kind] = resolve_type(definition, types, meaning, kind, changes)

    homes = {}
    for kind, sid in sorted(found, key=lambda k: KINDS.index(k[0])):
        source, key = found[(kind, sid)], kind + ":" + sid
        rule, before = given.get((kind, sid)), saved.get(key, {})
        entry = {"key": key, "source": kind, "source_id": sid, "name": source["name"],
                 "archived": bool(source.get("archived")), "action": "skip", "record_id": None,
                 "created": False, "title": None, "type_id": None, "parent": None}
        homes[key] = entry
        if rule is None or rule.action == "skip":
            continue
        if rule.action == "reuse":
            record = owned(db, StructureRecord, rule.record_id or "", owner)
            if record.archived:
                raise DomainError("LINEAR_HOME", "Choose an active Home for " + source["name"] + ".")
            entry.update(action="reuse", record_id=record.id, title=record.title, type_id=record.type_id,
                         created=bool(before.get("created") and before.get("record_id") == record.id))
            continue
        made = db.get(StructureRecord, before["record_id"]) if before.get("created") and before.get("record_id") else None
        if made and made.owner_id == owner and not made.archived:
            # Repeated setup reuses the Home it created instead of making another.
            entry.update(action="existing", record_id=made.id, title=made.title, type_id=made.type_id, created=True)
            continue
        parent = None
        if rule.parent == "team":
            team = next((homes["team:" + t] for t in source.get("team_ids", []) if
                         homes.get("team:" + t, {}).get("action", "skip") != "skip"), None) if kind == "project" else None
            parent = team and (team["record_id"] or "new:" + team["key"])
        elif rule.parent != "top":
            home = owned(db, StructureRecord, rule.parent, owner)
            if home.archived:
                raise DomainError("LINEAR_HOME", "Choose an active parent Home for " + source["name"] + ".")
            parent = home.id
        entry.update(action="create", title=rule.title or source["name"][:500], type_id=meanings[kind], parent=parent)

    task_type = import_type(SimpleNamespace(definition=definition), "task")["id"]

    def home_type(ref):
        if ref.startswith("new:"):
            return homes[ref[4:]]["type_id"]
        return db.get(StructureRecord, ref).type_id

    def allow(child, parent_type):
        t = types[child]
        if parent_type not in t["parent_types"]:
            t["parent_types"] = [*t["parent_types"], parent_type]
            changes.append({"kind": "allow_home", "type_id": child, "home_type_id": parent_type,
                            "text": t["plural"] + " can live in a " + types[parent_type]["name"]})

    for entry in homes.values():
        if entry["action"] != "skip":
            allow(task_type, entry["type_id"])
        if entry["action"] == "create" and entry["parent"]:
            allow(entry["type_id"], home_type(entry["parent"]))

    plan_key = fingerprint({
        "schema": schema.revision,
        "generation": conn.generation,
        "catalog": catalog["discovered_at"],
        "changes": changes,
        "homes": [[e[k] for k in ("key", "action", "record_id", "title", "type_id", "parent")] for e in homes.values()],
    })
    if write:
        if changes:
            raise DomainError("SCHEMA_CHANGED", "Apply the previewed structure change first.", 409)
        create(db, owner, schema, homes)
    refs = {e["key"]: e["record_id"] or "new:" + e["key"] for e in homes.values() if e["action"] != "skip"}
    names = {ref: homes[key]["title"] for key, ref in refs.items()}
    placement = place(db, owner, conn, refs, write=write, names=names)
    result = {
        "fingerprint": plan_key,
        "schema_changes": changes,
        "homes": list(homes.values()),
        "tree": tree(db, owner, homes, refs, catalog["counts"], types),
        "placement": placement,
    }
    if write:
        rules = {k: v for k, v in saved.items() if tuple(k.split(":", 1)) not in found}
        for e in homes.values():
            rules[e["key"]] = {"source": e["source"], "source_id": e["source_id"], "name": e["name"],
                               "archived": e["archived"], "record_id": e["record_id"], "created": e["created"],
                               "action": "skip" if e["action"] == "skip" else "home"}
        state.update(status="applied", meanings=meanings, rules=rules, applied_at=now().isoformat(),
                     applied_fingerprint=previewed or plan_key)
        state.pop("pending", None)
        save(conn, state)
        emit(db, owner, "linear.changed", owner)
    else:
        result["definition"] = definition
    return result


def create(db, owner, schema, homes):
    """Create the Homes the plan names (teams first, so projects can go inside them)."""
    from .structure import mutate as structure_mutate
    from .structure_schema import RecordCreate

    for entry in homes.values():
        if entry["action"] != "create":
            continue
        parent = entry["parent"]
        if parent and parent.startswith("new:"):
            parent = homes[parent[4:]]["record_id"]
        record = structure_mutate(
            db,
            owner,
            "record.create",
            RecordCreate(type_id=entry["type_id"], title=entry["title"], parent_id=parent,
                         schema_revision=schema.revision),
            "linear-mapping:" + str(uuid4()),
        )
        row = db.get(StructureRecord, record["id"])
        row.provenance = {**row.provenance, "origin": "linear_mapping",
                          "linear_source": {"kind": entry["source"], "id": entry["source_id"]}}
        entry.update(record_id=row.id, created=True, parent=parent)


def tree(db, owner, homes, refs, counts, types):
    """The resulting Homes with the Linear issues that land directly in each, and what stays unfiled."""
    landing, unfiled = {}, 0
    for team, cells in counts.items():
        for project, n in cells.items():
            ref = (refs.get("project:" + project) if project else None) or refs.get("team:" + team)
            if ref:
                landing[ref] = landing.get(ref, 0) + n
            else:
                unfiled += n
    nodes = {}
    for key, ref in refs.items():
        entry = homes[key]
        node = nodes.setdefault(ref, {
            "ref": ref, "title": entry["title"], "type_name": types[entry["type_id"]]["name"],
            "new": entry["action"] == "create", "sources": [], "issues": landing.get(ref, 0), "children": [],
            "parent": entry["parent"] if entry["action"] == "create" else (db.get(StructureRecord, ref).parent_id),
        })
        node["sources"].append({"kind": entry["source"], "name": entry["name"], "archived": entry["archived"]})
    roots = []
    for node in nodes.values():
        parent = nodes.get(node.pop("parent"))
        (parent["children"] if parent else roots).append(node)
    return {"homes": roots, "unfiled": unfiled}


def place(db, owner, conn, targets, *, write, links=None, names=None):
    """File imported issues in their mapped Home. Keeps intentional local exceptions.

    targets: {"team:<id>" | "project:<id>": Home id, or "new:<key>" for a Home a preview has not made}.
    An issue follows the mapping while its Home is the one the mapping last gave it (`linear_home` in the
    record's provenance), or, never placed before, while it is unfiled or in the legacy "· Linear" project.
    Any other Home is the owner's choice and stays. Subtasks follow their Linear parent issue instead."""
    from .hierarchy import for_task
    from .structure import ensure

    query = select(LinearIssue).where(LinearIssue.owner_id == owner, LinearIssue.workspace_id == conn.workspace_id)
    rows = list(db.scalars(query)) if links is None else list(links)
    linked = set(db.scalars(select(LinearIssue.remote_id).where(
        LinearIssue.owner_id == owner, LinearIssue.workspace_id == conn.workspace_id,
        LinearIssue.sync_state != "unlinked")))
    names = dict(names or {})
    counts = {"moved": 0, "kept": 0, "unchanged": 0, "subtasks": 0, "unmapped": 0, "blocked": 0}
    sample = []
    schema = ensure(db, owner) if write else None

    def label(ref):
        if ref is None:
            return "Not filed"
        if ref not in names:
            home = db.get(StructureRecord, ref)
            names[ref] = home.title if home else "Unknown Home"
        return names[ref]

    for link in rows:
        if link.sync_state in QUIET:
            continue
        snapshot = link.snapshot or {}
        if (snapshot.get("parent") or {}).get("id") in linked:
            counts["subtasks"] += 1
            continue
        project = (snapshot.get("project") or {}).get("id")
        target = (targets.get("project:" + project) if project else None) or targets.get(
            "team:" + ((snapshot.get("team") or {}).get("id") or ""))
        if not target:
            counts["unmapped"] += 1
            continue
        record = for_task(db, owner, link.task_id)
        task = db.get(Task, link.task_id)
        if record is None and write:
            from .structure import observe_core

            try:
                observe_core(db, owner, "task.update", serial(task), "linear-mapping:" + link.id)
            except DomainError:
                pass  # No actionable type to adopt it into: it stays unfiled and the sync goes on.
            record = for_task(db, owner, link.task_id)
            if record is None:
                counts["blocked"] += 1
                continue
        provenance = record.provenance if record else {}
        home = record.parent_id if record else None
        if "linear_home" in provenance:
            automatic = {provenance["linear_home"]}
        else:
            automatic = {None, legacy_project_id(owner, conn.workspace_id, project) if project else None}
        if home == target:
            counts["unchanged"] += 1
            if write and provenance.get("linear_home") != target:
                record.provenance = {**provenance, "linear_home": target}
            continue
        if home not in automatic:
            counts["kept"] += 1
            outcome = "kept"
        elif write and not move(db, owner, schema, record, target):
            counts["blocked"] += 1
            outcome = "blocked"
        else:
            counts["moved"] += 1
            outcome = "moved"
        if len(sample) < 100:
            sample.append({"task_id": link.task_id, "title": task.title if task else "", "outcome": outcome,
                           "from": label(home), "to": label(target)})
    return {**counts, "changes": sample}


def move(db, owner, schema, record, target):
    from .hierarchy import project_parent
    from .structure import record_type, remember_core, validate_parent

    try:
        validate_parent(db, owner, record, record_type(schema, record.type_id, archived=True), target)
    except DomainError:
        return False  # An archived or disallowed Home: the issue stays where it is.
    record.parent_id = target
    record.sort_order = (db.scalar(select(func.max(StructureRecord.sort_order)).where(
        StructureRecord.owner_id == owner, StructureRecord.parent_id == target)) or 0) + 1024
    record.provenance = {**record.provenance, "linear_home": target}
    record.revision += 1
    record.updated_at = now()
    project_parent(db, record)
    remember_core(db, record)
    emit(db, owner, "record.changed", record.id, record.revision)
    return True


def preview(db, owner, mapping):
    from .onboarding import expire_proposal
    from .structure import ensure
    from .structure import preview as preview_schema
    from .structure_schema import Definition
    from .structure_schema import Preview as SchemaPreview

    advisory(db, f"workspace:{owner}")
    conn = connection(db, owner)
    plan = reconcile(db, owner, conn, mapping)
    definition = plan.pop("definition")
    state = current(conn)
    expire_proposal(db, owner, (state.get("pending") or {}).get("proposal_id"))
    state.pop("pending", None)
    plan["proposal"] = None
    if plan["schema_changes"]:
        # Schema changes go through the guarded structure preview/apply; apply confirms this proposal.
        proposal = preview_schema(
            db,
            owner,
            SchemaPreview(expected_revision=ensure(db, owner).revision, definition=Definition.model_validate(definition)),
            "linear-mapping-preview:" + str(uuid4()),
        )
        state["pending"] = {"fingerprint": plan["fingerprint"], "proposal_id": proposal["id"]}
        plan["proposal"] = {"id": proposal["id"], "blocking_count": proposal["impact"]["blocking_count"],
                            "issues": proposal["impact"]["issues"][:20]}
    save(conn, state)
    return plan


def apply(db, owner, body):
    from .structure import apply as apply_schema
    from .structure import ensure
    from .structure_schema import Apply as SchemaApply

    advisory(db, f"workspace:{owner}")
    conn = connection(db, owner)
    state = current(conn)
    if state.get("status") == "applied" and state.get("applied_fingerprint") == body.fingerprint:
        # A retried apply whose first response was lost: already done, nothing is created twice.
        return {"applied": True, "repeated": True, "structure": overview(conn)}
    plan = reconcile(db, owner, conn, body)
    if plan["fingerprint"] != body.fingerprint:
        raise DomainError("LINEAR_STALE", "Your Linear mapping, Homes or structure changed. Review the new preview.", 409)
    if plan["schema_changes"]:
        pending = state.get("pending") or {}
        if pending.get("fingerprint") != body.fingerprint:
            raise DomainError("LINEAR_STALE", "Preview this mapping again before applying it.", 409)
        apply_schema(
            db,
            owner,
            SchemaApply(proposal_id=pending["proposal_id"], expected_revision=ensure(db, owner).revision),
            "linear-mapping-apply:" + str(uuid4()),
        )
    result = reconcile(db, owner, conn, body, write=True, previewed=body.fingerprint)
    return {"applied": True, "repeated": False, "structure": overview(conn), **result}


def skip(db, owner):
    """Import normally. A mapping applied earlier stops filing new issues; filed issues stay put."""
    from .onboarding import expire_proposal

    advisory(db, f"workspace:{owner}")
    conn = connection(db, owner)
    state = current(conn)
    expire_proposal(db, owner, (state.get("pending") or {}).get("proposal_id"))
    state.pop("pending", None)
    state["status"] = "skipped"
    save(conn, state)
    emit(db, owner, "linear.changed", owner)
    return overview(conn)


def after_sync(db, owner, conn, links, seen):
    """Called by linear_sync.process_sync with the links it touched and the source names it read."""
    state = current(conn)
    if state.get("status") != "applied":
        return
    if follow_sources(db, owner, state, seen):
        save(conn, state)
    targets = {k: r["record_id"] for k, r in state.get("rules", {}).items()
               if r.get("action") == "home" and r.get("record_id")}
    if targets and links:
        place(db, owner, conn, targets, write=True, links=links)
