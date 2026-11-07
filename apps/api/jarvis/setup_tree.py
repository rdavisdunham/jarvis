"""Set up with Dani: a proposed tree of spaces, areas, goals and projects, previewed and then applied.

Never a silent write: `setup.tree_preview` stores a proposal with a preview_hash and changes nothing;
`setup.tree_apply` applies exactly that preview, as one command through the ordinary create and update
mutations, so every change is journaled under one receipt and reverted as a group.
See docs/SETUP_WITH_DANI_PLAN.md.
"""

import hashlib
import json
from datetime import timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from .domain import Args, DomainError
from .models import Area, Goal, GoalProjectLink, Project, Space, now
from .setup_models import OrganizationProposal, SetupSession

MAX_NODES = 80
MAX_DEPTH = 4
MODELS = {"space": Space, "area": Area, "goal": Goal, "project": Project}
# Which kinds may hold which: spaces are top level; areas sit in a space; goals in a space or area;
# projects in a space, an area or under a goal (linked to it, homed with it).
PARENTS = {"space": {None}, "area": {"space"}, "goal": {"space", "area"}, "project": {"space", "area", "goal"}}
ORDER = {"space": 0, "area": 1, "goal": 2, "project": 3}


class Node(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    id: str = Field(min_length=1, max_length=64, description="Stable ID within this proposal.")
    kind: Literal["space", "area", "goal", "project"]
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    existing_id: str | None = Field(default=None, max_length=36)
    action: Literal["create", "keep", "rename", "archive"] = "create"
    archive_confirmed: bool = Field(default=False, description="Archive happens only when the owner ticks it.")
    children: list["Node"] = Field(default_factory=list, max_length=MAX_NODES)


class TreePreview(Args):
    tree: list[Node] = Field(max_length=MAX_NODES)


class TreeApply(Args):
    proposal_id: str = Field(min_length=1, max_length=36)
    preview_hash: str = Field(min_length=64, max_length=64)


COMMANDS = {"setup.tree_preview": TreePreview, "setup.tree_apply": TreeApply}


def walk(nodes, parent=None, depth=1):
    for node in nodes:
        yield node, parent, depth
        yield from walk(node.children, node, depth + 1)


def fold(text):
    return " ".join(text.split()).casefold()


def existing(db, owner, node):
    row = db.get(MODELS[node.kind], node.existing_id) if node.existing_id else None
    return row if row is not None and row.owner_id == owner else None


def match(db, owner, node, parent_row):
    """An existing record this created node should reuse instead (unique names are matched before creating)."""
    model = MODELS[node.kind]
    q = select(model).where(model.owner_id == owner)
    if node.kind == "area":
        if parent_row is None:
            return None
        q = q.where(Area.space_id == parent_row.id)
    elif node.kind == "goal":
        if parent_row is None:
            return None
        q = q.where(Goal.area_id == parent_row.id) if isinstance(parent_row, Area) else q.where(
            Goal.space_id == parent_row.id, Goal.area_id.is_(None))
    return next((row for row in db.scalars(q) if fold(row.name) == fold(node.title)), None)


def normalize(db, owner, tree):
    """Validate the tree against the owner's records and match created rows to existing ones by name.
    Returns (tree, issues); an issue blocks applying."""
    nodes = list(walk(tree))
    if len(nodes) > MAX_NODES:
        raise DomainError("INVALID_ARGUMENT", f"A proposal holds at most {MAX_NODES} rows.")
    if len({n.id for n, _, _ in nodes}) != len(nodes):
        raise DomainError("INVALID_ARGUMENT", "Each row in the proposal needs its own ID.")
    issues, rows = [], {}
    seen_projects = {}
    for node, parent, depth in nodes:
        label = node.kind.capitalize() + " " + json.dumps(node.title)
        if depth > MAX_DEPTH:
            raise DomainError("INVALID_ARGUMENT", "The proposal is nested too deeply.")
        if (parent.kind if parent else None) not in PARENTS[node.kind]:
            where = ("inside a " + parent.kind) if parent else "at the top level"
            raise DomainError("INVALID_ARGUMENT", f"{label} cannot sit {where}.")
        parent_row = rows.get(parent.id) if parent else None
        if node.existing_id:
            row = existing(db, owner, node)
            if row is None:
                raise DomainError("NOT_FOUND", f"{label} refers to a record that is not available.", 404)
            if node.action == "create":
                node.action = "rename" if row.name != node.title else "keep"
        else:
            if node.action != "create":
                raise DomainError("INVALID_ARGUMENT", f"{label} must name the existing record to {node.action}.")
            row = match(db, owner, node, parent_row)
            if row is not None:
                node.existing_id, node.action = row.id, "keep"
        if node.action == "rename" and row is not None and row.name == node.title:
            node.action = "keep"
        rows[node.id] = row
        if row is not None and row.archived and node.action != "archive":
            issues.append({"node_id": node.id, "message": f"{label} matches an archived record. Restore it first, or choose another name."})
        if node.action == "archive" and node.archive_confirmed and any(c.action != "archive" for c in node.children):
            issues.append({"node_id": node.id, "message": f"{label} still holds rows you are keeping. Move or archive them first."})
        if node.kind == "project" and node.action != "archive":
            key = fold(node.title)
            if key in seen_projects:
                issues.append({"node_id": node.id, "message": f"Two projects are called {json.dumps(node.title)}. Project names are unique."})
            seen_projects[key] = node.id
            if node.action in {"create", "rename"}:
                clash = db.scalar(select(Project).where(Project.owner_id == owner, Project.name == node.title,
                                                        Project.id != (node.existing_id or "")))
                if clash is not None:
                    issues.append({"node_id": node.id, "message": f"Another project is already called {json.dumps(node.title)}."})
        if node.kind == "space" and node.action == "rename":
            clash = db.scalar(select(Space).where(Space.owner_id == owner, Space.name == node.title, Space.id != node.existing_id))
            if clash is not None:
                issues.append({"node_id": node.id, "message": f"Another space is already called {json.dumps(node.title)}."})
    for siblings in [tree, *(n.children for n, _, _ in nodes)]:
        names = [fold(n.title) for n in siblings if n.action != "archive"]
        if len(names) != len(set(names)):
            issues.append({"node_id": None, "message": "Two rows under the same parent have the same name."})
    return tree, issues


def impact(tree, issues):
    counts = {kind: 0 for kind in MODELS}
    renamed = archived = kept = 0
    for node, _, _ in walk(tree):
        if node.action == "create":
            counts[node.kind] += 1
        elif node.action == "rename":
            renamed += 1
        elif node.action == "archive" and node.archive_confirmed:
            archived += 1
        else:
            kept += 1
    return {"create": counts, "rename": renamed, "archive": archived, "keep": kept,
            "issues": issues, "blocking_count": len(issues)}


def fingerprint(db, owner, tree):
    """Hash of the normalized tree plus the current revision of every record it names, so any later change
    to those records makes the preview stale."""
    revisions = {}
    for node, _, _ in walk(tree):
        if node.existing_id:
            row = existing(db, owner, node)
            revisions[node.kind + ":" + node.existing_id] = [row.revision, row.name, row.archived] if row else None
    payload = json.dumps({"tree": [n.model_dump(mode="json") for n in tree], "records": revisions},
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def public(row):
    return {"id": row.id, "tree": row.tree, "impact": row.impact, "preview_hash": row.preview_hash,
            "applied_at": row.applied_at.isoformat() if row.applied_at else None,
            "expires_at": row.expires_at.isoformat(), "command_id": row.command_id}


def read(db, owner, proposal_id):
    from .setup import personal
    personal(db, owner)
    row = db.get(OrganizationProposal, proposal_id)
    if not row or row.owner_id != owner:
        raise DomainError("NOT_FOUND", "That proposal is no longer available.", 404)
    return public(row)


def preview(db, owner, tree, command_id):
    from .access import actor
    from .setup import personal
    personal(db, owner)
    tree, issues = normalize(db, owner, tree)
    session = db.get(SetupSession, owner, with_for_update=True)
    previous = db.get(OrganizationProposal, session.tree_proposal_id) if session and session.tree_proposal_id else None
    if previous and not previous.applied_at:
        previous.expires_at = now()
    row = OrganizationProposal(owner_id=owner, account_id=actor(db, owner) or owner,
                               conversation_id=session.conversation_id if session else None,
                               tree=[n.model_dump(mode="json") for n in tree], impact=impact(tree, issues),
                               preview_hash=fingerprint(db, owner, tree), expires_at=now() + timedelta(days=2))
    db.add(row)
    db.flush()
    if session:
        session.tree_proposal_id = row.id
        if session.status in {"interviewing", "proposing"}:
            session.status = "reviewing"
        session.revision += 1
    return public(row)


def apply(db, owner, args, command_id):
    from .domain import mutate as domain_mutate
    from .domain import COMMANDS as ALL
    from .setup import personal
    personal(db, owner)
    row = db.get(OrganizationProposal, args.proposal_id, with_for_update=True)
    if not row or row.owner_id != owner:
        raise DomainError("NOT_FOUND", "That proposal is no longer available.", 404)
    if row.applied_at:
        raise DomainError("REVISION_CONFLICT", "This organization was already created.", 409)
    if row.expires_at <= now():
        raise DomainError("PREVIEW_REQUIRED", "This proposal expired. Preview it again before creating it.", 409)
    tree = [Node.model_validate(n) for n in row.tree]
    if args.preview_hash != row.preview_hash or fingerprint(db, owner, tree) != row.preview_hash:
        raise DomainError("PREVIEW_REQUIRED", "Your organization changed since this preview. Preview it again.", 409)
    if row.impact.get("blocking_count"):
        raise DomainError("INVALID_ARGUMENT", "Fix the highlighted rows before creating this organization.")
    ids, created, changed = {}, [], []

    def run(tool, values):
        return domain_mutate(db, owner, tool, ALL[tool].model_validate(values), command_id)

    parents = {node.id: parent for node, parent, _ in walk(tree)}
    for node, parent, _ in walk(tree):
        parent_id = ids.get(parent.id) if parent else None
        if node.action == "create":
            values = {"name": node.title, "description": node.description}
            if node.kind == "area":
                values["space_id"] = parent_id
            elif node.kind in {"goal", "project"}:
                # A project under a goal is homed where the goal is, and linked to it.
                home = parent
                while home is not None and home.kind == "goal":
                    home = parents[home.id]
                if home is not None:
                    values["area_id" if home.kind == "area" else "space_id"] = ids[home.id]
                if node.kind == "project" and parent and parent.kind == "goal":
                    values["goal_ids"] = [parent_id]
            ids[node.id] = run(node.kind + ".create", values)["id"]
            created.append({"kind": node.kind, "id": ids[node.id]})
            continue
        current = db.get(MODELS[node.kind], node.existing_id)
        ids[node.id] = current.id
        key = node.kind + "_id"
        if node.action == "rename":
            run(node.kind + ".update", {key: current.id, "expected_revision": current.revision, "name": node.title})
            changed.append({"kind": node.kind, "id": current.id})
        elif node.action == "archive" and node.archive_confirmed:
            run(node.kind + ".update", {key: current.id, "expected_revision": current.revision, "archived": True})
            changed.append({"kind": node.kind, "id": current.id})
        elif node.kind == "project" and parent and parent.kind == "goal" and node.action == "keep" and parent_id:
            linked = set(db.scalars(select(GoalProjectLink.goal_id).where(GoalProjectLink.project_id == current.id)))
            if parent_id not in linked:
                run("project.update", {key: current.id, "expected_revision": current.revision,
                                       "goal_ids": sorted(linked | {parent_id})})
                changed.append({"kind": node.kind, "id": current.id})
    row.applied_at, row.command_id = now(), command_id
    session = db.get(SetupSession, owner)
    if session and session.tree_proposal_id == row.id:
        session.revision += 1
    return {"proposal": public(row), "created": created, "changed": changed,
            "created_count": len(created), "changed_count": len(changed)}


def mutate(db, owner, tool, args, command_id):
    if tool == "setup.tree_preview":
        return preview(db, owner, args.tree, command_id)
    return apply(db, owner, args, command_id)


# Revert: the whole applied tree is undone as one group (record_contents.restore dispatches here).

def outside_links(db, model, identity, group):
    """Records outside this group that point at `identity`."""
    from .models import Base
    for table in Base.metadata.sorted_tables:
        if table.name in {"task_references", "note_embeddings", "routing_observations", "note_organizations",
                          "note_entry_sources", "search_documents", "action_changes"}:
            continue
        for column in table.columns:
            if not any(fk.column.table is model.__table__ for fk in column.foreign_keys):
                continue
            for linked in db.execute(select(table).where(column == identity)).mappings():
                others = [linked[c.name] for c in table.columns if c.foreign_keys and c.name != column.name
                          and any(fk.column.table.name in {"spaces", "areas", "goals", "projects"} for fk in c.foreign_keys)]
                own = linked.get("id") in group if "id" in linked else bool(others) and all(v in group or v is None for v in others)
                if not own:
                    return True
    return False


def undo_problem(db, owner, rows):
    from .action_history import instants, snapshot
    from .work_crypto import unseal
    if any(r.reverted_by for r in rows):
        return "Already reverted."
    group = {r.entity_id for r in rows if r.before_ciphertext is None}
    for change in rows:
        model = MODELS.get(change.entity_kind)
        current = db.get(model, change.entity_id) if model else None
        if current is None or current.owner_id != owner:
            return "These records are no longer available."
        after = instants(unseal(change.after_ciphertext)) if change.after_ciphertext else None
        now_values = instants(snapshot(current))
        if after is not None and {k: v for k, v in now_values.items() if k not in {"updated_at"}} != {
                k: v for k, v in after.items() if k not in {"updated_at"}}:
            return "Some of these records changed after they were created. Review them before reverting."
        if change.before_ciphertext is None and outside_links(db, model, current.id, group):
            return "Other records now use part of this organization. Review them before reverting."
    return None


def restore(db, owner, rows, command_id):
    """Archive what the apply created (deepest first) and restore what it renamed or archived."""
    from uuid import NAMESPACE_URL, uuid5
    from .action_history import instants
    from .domain import execute
    from .work_crypto import unseal
    reason = undo_problem(db, owner, rows)
    if reason:
        raise DomainError("REVISION_CONFLICT", reason, 409)
    plan = sorted(rows, key=lambda r: -ORDER.get(r.entity_kind, 0))
    done = 0
    for change in plan:
        current = db.get(MODELS[change.entity_kind], change.entity_id)
        key = change.entity_kind + "_id"
        if change.before_ciphertext is None:
            values = {"archived": True}
        else:
            before = instants(unseal(change.before_ciphertext))
            values = {k: before[k] for k in ("name", "archived") if k in before and before[k] != getattr(current, k)}
            if not values:
                change.reverted_by = command_id
                continue
        child = str(uuid5(NAMESPACE_URL, command_id + ":" + str(done)))
        execute(db, owner, child, change.entity_kind + ".update",
                {key: current.id, "expected_revision": current.revision, **values})
        change.reverted_by = command_id
        done += 1
    return {"restored": True, "changed_count": done}
