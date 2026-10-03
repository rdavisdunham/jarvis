"""Versioned, user-owned organization over stable task and note capabilities."""

import hashlib
import json
import math
from uuid import NAMESPACE_URL, uuid5
from datetime import date, datetime, timedelta

from sqlalchemy import select, func

from .domain import DomainError, advisory, check_revision, emit, nulls, owned, serial
from .models import (
    Area,
    Goal,
    GoalProjectLink,
    Note,
    Project,
    Schedule,
    Space,
    Task,
    now,
)
from .structure_models import (
    FieldUnderstanding,
    StructureLink,
    StructureProposal,
    StructureRecord,
    StructureSchema,
)
from .structure_schema import Definition

MEANINGS = ("backlog", "open", "in_progress", "waiting", "deferred", "completed", "cancelled")
TASK_BINDINGS = {
    "due_date",
    "due_time",
    "due_timezone",
    "planned_date",
    "priority",
    "estimate_minutes",
    "assignee",
}


BINDING_DEFAULTS = {"assignee": "owner", "priority": 0}
# Custom workflows may omit a core meaning; map to the closest remaining status.
NEAREST = {
    "backlog": ("open",),
    "open": ("backlog", "in_progress"),
    "in_progress": ("open",),
    "waiting": ("in_progress", "open"),
    "deferred": ("backlog", "open"),
    "completed": ("cancelled",),
    "cancelled": ("completed",),
}


def task_status(t, meaning, current=None):
    statuses = t["statuses"]
    if any(s["id"] == current and s["meaning"] == meaning for s in statuses):
        return current
    for candidate in (meaning, *NEAREST.get(meaning, ())):
        found = next((s["id"] for s in statuses if s["meaning"] == candidate), None)
        if found:
            return found
    if any(s["id"] == current for s in statuses):
        return current
    return statuses[0]["id"] if statuses else None


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()
    ).hexdigest()


def default_definition():
    from .field_library import DEFAULT_OPENS_AS
    statuses = [{"id": s, "name": s.replace("_", " ").capitalize(), "meaning": s} for s in MEANINGS]

    def field(binding, name, kind, description):
        return {"id": binding, "name": name, "kind": kind, "description": description, "binding": binding}

    work = [
        field("due_date", "Deadline", "date", "The actual deadline, distinct from a planned work day."),
        field("due_time", "Due time", "text", "Optional local clock time for the deadline, HH:MM."),
        field("due_timezone", "Time zone", "text", "IANA zone for the deadline, such as America/Chicago."),
        field(
            "planned_date", "Do date", "date", "The day you intend to work on this; it is not a deadline."
        ),
        field(
            "priority",
            "Priority",
            "number",
            "Task importance from 0 to 3; not permission to interrupt quiet hours.",
        ),
        field("estimate_minutes", "Estimate", "number", "Estimated effort in minutes."),
        field(
            "assignee",
            "Assigned to",
            "text",
            "The responsible person or agent; assignment does not grant access or launch work.",
        ),
    ]
    timeline = [
        field("start_date", "Start", "date", "Planned start of an initiative; not reserved busy time."),
        field("target_date", "Target", "date", "Target finish of an initiative; not a task deadline."),
    ]
    metric = [
        field("metric_baseline", "Starting value", "number", "The starting measurement of this outcome."),
        field(
            "metric_current",
            "Current value",
            "number",
            "The most recent measurement, independent of task counts.",
        ),
        field("metric_target", "Target value", "number", "The desired measured outcome."),
        field("metric_unit", "Unit", "text", "Unit of measurement, such as dollars or customers."),
    ]
    definitions = [
        (
            "space",
            "Space",
            "Spaces",
            "Broad context such as Work, Personal or School. This never controls access.",
            [],
            [],
        ),
        ("area", "Area", "Areas", "An ongoing responsibility, such as Health or Operations.", [], []),
        (
            "client",
            "Client",
            "Clients",
            "A customer organization for whom projects are performed. Projects have separate identities.",
            [],
            [],
        ),
        (
            "project",
            "Project",
            "Projects",
            "A finite initiative with an intended result, optionally belonging to a client.",
            ["work", "timeline"],
            work + timeline,
        ),
        (
            "goal",
            "Goal",
            "Goals",
            "A desired outcome measured independently of completed tasks.",
            ["timeline", "metric"],
            timeline + metric,
        ),
        (
            "task",
            "Task",
            "Tasks",
            "An actionable unit of work that may be completed and have deadlines or reminders.",
            ["work"],
            work,
        ),
        (
            "note",
            "Note",
            "Notes",
            "Authored material whose content is preserved and may link to other work.",
            ["content"],
            [],
        ),
    ]
    types = [
        {
            "id": k,
            "name": n,
            "plural": plural,
            "description": d,
            "capabilities": caps,
            "parent_types": [r[0] for r in definitions],
            "fields": fields,
            "statuses": statuses if "work" in caps else [],
            "opens_as": DEFAULT_OPENS_AS.get(k, (None, (), "auto"))[2],
        }
        for k, n, plural, d, caps, fields in definitions
    ]
    relationships = [
        {
            "id": "supports",
            "name": "Supports",
            "description": "An initiative supports an outcome without owning or completing it.",
            "source_types": ["project"],
            "target_types": ["goal"],
            "cardinality": "many_to_many",
        },
        {
            "id": "related",
            "name": "Related to",
            "description": "A contextual cross-link that does not alter home or inherited fields.",
            "source_types": [t["id"] for t in types],
            "target_types": [t["id"] for t in types],
            "cardinality": "many_to_many",
        },
    ]
    relationships.append({
        "id":"blocks","name":"Blocks","description":"An unfinished prerequisite. Advisory: explicit progress is allowed.",
        "source_types":[t["id"] for t in types if "work" in t["capabilities"]],
        "target_types":[t["id"] for t in types if "work" in t["capabilities"]],
        "cardinality":"many_to_many","behavior":"blocks",
    })
    from .field_library import upgrade
    return upgrade(Definition(types=types, relationships=relationships).model_dump(mode="json"))


def definition_entries(definition):
    for t in definition["types"]:
        yield (
            "type:" + t["id"],
            {k: t[k] for k in ("name", "description", "capabilities", "parent_types", "archived")},
        )
        for f in t["fields"]:
            yield "field:" + t["id"] + ":" + f["id"], {k: v for k, v in f.items() if k != "library_id"}
    for r in definition["relationships"]:
        yield "relationship:" + r["id"], r


def ensure(db, owner):
    schema = db.get(StructureSchema, owner)
    if schema:
        if "field_library" not in schema.definition or any("opens_as" not in t for t in schema.definition["types"]):
            from .field_library import upgrade
            schema.definition = upgrade(schema.definition)
        return schema
    advisory(db, "workspace:" + owner)
    schema = db.get(StructureSchema, owner, populate_existing=True)
    if schema:
        return schema
    schema = StructureSchema(owner_id=owner, definition=default_definition())
    db.add(schema)
    db.flush()
    for key, entry in definition_entries(schema.definition):
        db.add(
            FieldUnderstanding(
                owner_id=owner,
                definition_id=key,
                fingerprint=fingerprint(entry),
                status="ready",
                understanding={"meaning": entry["description"], "source": "default_template"},
            )
        )
    # Best-effort import preserves source rows and never invokes external write services.
    imports = []
    for kind, model in (
        ("space", Space),
        ("area", Area),
        ("goal", Goal),
        ("project", Project),
        ("task", Task),
        ("note", Note),
    ):
        for old in db.scalars(select(model).where(model.owner_id == owner)):
            values = {}
            for field in next(t["fields"] for t in schema.definition["types"] if t["id"] == kind):
                value = getattr(old, field["binding"], None)
                if value is not None:
                    values[field["id"]] = (
                        value.isoformat()
                        if hasattr(value, "isoformat")
                        else float(value)
                        if field["kind"] == "number"
                        else value
                    )
            row = StructureRecord(
                id=old.id,
                owner_id=owner,
                type_id=kind,
                title=getattr(old, "title", getattr(old, "name", "Untitled")),
                body=getattr(old, "content", getattr(old, "notes", getattr(old, "description", ""))),
                values=values,
                status_id=old.status if kind == "task" else None,
                task_id=old.id if kind == "task" else None,
                note_id=old.id if kind == "note" else None,
                sort_order=(len(imports)+1)*1024,
                legacy_kind=kind,
                archived=old.archived,
                provenance={"origin": "legacy_unknown", "automatic_learning": False},
            )
            db.add(row)
            imports.append((row, old))
    db.flush()
    ids = {r.id for r, _ in imports}
    for row, old in imports:
        parent = (
            getattr(old, "parent_goal_id", None)
            or getattr(old, "project_id", None)
            or getattr(old, "area_id", None)
            or getattr(old, "space_id", None)
        )
        if parent in ids and parent != row.id:
            row.parent_id = parent
    for link in db.scalars(select(GoalProjectLink).join(Project).where(Project.owner_id == owner)):
        if link.project_id in ids and link.goal_id in ids:
            db.add(
                StructureLink(
                    id=str(
                        uuid5(
                            NAMESPACE_URL,
                            "eridani:legacy-link:" + owner + ":" + link.project_id + ":" + link.goal_id,
                        )
                    ),
                    owner_id=owner,
                    relationship_id="supports",
                    source_id=link.project_id,
                    target_id=link.goal_id,
                )
            )
    db.flush()
    return schema


def opens_as(t, has_children=False):
    """Resolved presentation: explicit setting first; auto keeps the inferred container rule."""
    mode = t.get("opens_as", "auto")
    if mode != "auto":
        return mode
    caps = set(t["capabilities"])
    return "container" if has_children or "timeline" in caps or not caps & {"work", "content"} else "item"


def record_type(schema, type_id, *, archived=False):
    t = next((t for t in schema.definition["types"] if t["id"] == type_id), None)
    if not t or (t["archived"] and not archived):
        raise DomainError("INVALID_TYPE", "Choose an active record type from the current structure.")
    return t


def assert_schema(schema, revision):
    if schema.revision != revision:
        raise DomainError(
            "SCHEMA_CHANGED", "Your structure changed. Read its current definition and retry.", 409
        )


def schema_data(db, owner):
    schema = ensure(db, owner)
    if core_drift(db, owner):
        sync_core_records(db, owner, schema)
    from .hierarchy import adopt
    adopt(db, owner, schema)
    return {
        "revision": schema.revision,
        **schema.definition,
        "understandings": [
            serial(r)
            for r in db.scalars(select(FieldUnderstanding).where(FieldUnderstanding.owner_id == owner))
        ],
    }


def parent_chain(db, owner, parent_id, *, self_id=None):
    chain, seen = [], {self_id} if self_id else set()
    while parent_id:
        if parent_id in seen or len(chain) >= 100:
            raise DomainError("HIERARCHY_CYCLE", "Choose a parent outside this record's descendants.")
        seen.add(parent_id)
        parent = owned(db, StructureRecord, parent_id, owner)
        chain.append(parent)
        parent_id = parent.parent_id
    return chain


def validate_parent(db, owner, row, t, parent_id):
    chain = parent_chain(db, owner, parent_id, self_id=row.id if row else None)
    if chain and (chain[0].archived or chain[0].type_id not in t["parent_types"]):
        raise DomainError("INVALID_PARENT", "That type is not an allowed main home for this record.")
    return chain


def validate_values(db, owner, t, values, *, allow_archived=False):
    if len(values) > 60:
        raise DomainError("INVALID_ARGUMENT", "Too many field values.")
    fields = {f["id"]: f for f in t["fields"] if allow_archived or not f["archived"]}
    clean = {}
    for key, value in values.items():
        f = fields.get(key)
        if not f:
            raise DomainError(
                "INVALID_FIELD",
                "Read the current type definition before editing fields.",
                data={"field_id": key},
            )
        if value is None:
            clean[key] = None
            continue
        kind = f["kind"]
        valid = True
        if kind in {"text", "long_text"}:
            valid = isinstance(value, str) and len(value) <= (30000 if kind == "long_text" else 5000)
        elif kind == "number":
            valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        elif kind == "boolean":
            valid = isinstance(value, bool)
        elif kind in {"date", "datetime"}:
            try:
                if not isinstance(value, str):
                    raise ValueError()
                date.fromisoformat(value) if kind == "date" else datetime.fromisoformat(value)
            except (ValueError, TypeError):
                valid = False
        elif kind in {"select", "multiselect"}:
            items = value if kind == "multiselect" else [value]
            valid = (
                isinstance(items, list)
                and all(isinstance(v, str) and v in {o["id"] for o in f["options"]} for v in items)
                and len(items) <= 100
            )
            if valid and kind == "multiselect":
                value = list(dict.fromkeys(items))
        elif kind == "relation":
            items = value if f["multiple"] else [value]
            valid = isinstance(items, list) and len(items) <= 200 and all(isinstance(v, str) for v in items)
            if valid:
                for identity in items:
                    target = owned(db, StructureRecord, identity, owner)
                    if target.archived or target.type_id not in f["target_types"]:
                        valid = False
                        break
                if f["multiple"]:
                    value = list(dict.fromkeys(items))
        if not valid:
            raise DomainError(
                "INVALID_FIELD_VALUE", "Check the value for " + f["name"] + ".", data={"field_id": key}
            )
        clean[key] = value
    return clean


def data(db, row, schema=None):
    schema = schema or ensure(db, row.owner_id)
    t = record_type(schema, row.type_id, archived=True)
    result = serial(row)
    result["schema_revision"] = schema.revision
    from .sources import native
    result["source"] = native(row.id)
    values = dict(row.values)
    task = db.get(Task, row.task_id) if row.task_id else None
    note = db.get(Note, row.note_id) if row.note_id else None
    if task:
        from .sources import task_source
        result["source"] = task_source(db, task)
        capture=serial(task)
        result.update({k:capture.get(k) for k in ("is_quick_list","quick_list_parent_id","quick_done","quick_total")})
        result.update(title=task.title, body=task.notes, archived=task.archived, task_revision=task.revision)
        result["status_id"] = task_status(t, task.status, row.status_id)
        result["status_meaning"] = task.status
        for f in t["fields"]:
            if f.get("binding") in TASK_BINDINGS:
                v = getattr(task, f["binding"])
                values[f["id"]] = v.isoformat() if hasattr(v, "isoformat") else v
    if note:
        result.update(title=note.title, body=note.content, note_revision=note.revision)
    inherited = {}
    chain = parent_chain(db, row.owner_id, row.parent_id, self_id=row.id)
    for f in t["fields"]:
        if f["inherit"] and f["id"] not in values:
            for parent in chain:
                if f["kind"] == "relation" and parent.type_id in f["target_types"]:
                    values[f["id"]] = [parent.id] if f["multiple"] else parent.id
                    inherited[f["id"]] = parent.id
                    break
                parent_type = record_type(schema, parent.type_id, archived=True)
                source = next((pf for pf in parent_type["fields"] if not pf["archived"] and (
                    pf.get("library_id") == f.get("library_id") if f.get("library_id")
                    else pf["id"] == f["id"])), None)
                if source and source["id"] in parent.values:
                    candidate = parent.values[source["id"]]
                    try:
                        validate_values(db, row.owner_id, t, {f["id"]: candidate}, allow_archived=True)
                    except DomainError:
                        continue
                    values[f["id"]] = candidate
                    inherited[f["id"]] = parent.id
                    break
    from .record_contents import blocking
    result["blockers"] = blocking(db, row, schema)
    result["opens_as"] = opens_as(t)
    if result["opens_as"] == "item" and t.get("opens_as", "auto") == "auto":
        result["opens_as"] = opens_as(t, db.scalar(select(StructureRecord.id).where(
            StructureRecord.owner_id == row.owner_id, StructureRecord.parent_id == row.id,
            StructureRecord.archived.is_(False)).limit(1)) is not None)
    result.update(
        values=values,
        inherited=inherited,
        type_name=t["name"],
        capabilities=t["capabilities"],
        home=[{"id": p.id, "title": p.title, "type_id": p.type_id} for p in reversed(chain)],
        links=[
            serial(link)
            for link in db.scalars(
                select(StructureLink).where(
                    StructureLink.owner_id == row.owner_id,
                    (StructureLink.source_id == row.id) | (StructureLink.target_id == row.id),
                )
            )
        ],
    )
    return result


COMPACT = ("source", "opens_as", "id", "type_id", "type_name", "title", "parent_id", "home", "status_id", "status_meaning", "task_id", "note_id", "archived", "revision")


def records(
    db, owner, *, type_id=None, capability=None, parent_id=None, query="", archived=False, limit=25, offset=0, detail="full"
):
    schema = ensure(db, owner)
    if core_drift(db, owner):
        sync_core_records(db, owner, schema)
    from .hierarchy import adopt
    adopt(db, owner, schema)
    q = select(StructureRecord).where(StructureRecord.owner_id == owner, StructureRecord.archived == archived)
    if type_id:
        q = q.where(StructureRecord.type_id == type_id)
    if capability:
        q = q.where(
            StructureRecord.type_id.in_(
                [t["id"] for t in schema.definition["types"] if capability in t["capabilities"]]
            )
        )
    if parent_id:
        q = q.where(StructureRecord.parent_id == parent_id)
    if query:
        from .text_normalize import sql_filter

        q = q.where(sql_filter(query, StructureRecord.title))
    rows = list(
        db.scalars(
            q.order_by(StructureRecord.updated_at.desc(), StructureRecord.id).offset(offset).limit(limit + 1)
        )
    )
    def present(row):
        item = data(db, row, schema)
        if detail == "full":
            return item
        from .sources import compact_source
        return {k: compact_source(v) if k == "source" else v for k,v in item.items() if k in COMPACT}

    return {
        "items": [
            present(r)
            for r in rows[:limit]
        ],
        "has_more": len(rows) > limit,
        "next_offset": offset + limit if len(rows) > limit else None,
        "schema_revision": schema.revision,
    }


def snapshot_records(db, owner):
    return fingerprint(
        [
            (
                r.id,
                r.revision,
                r.type_id,
                r.parent_id,
                r.values,
                r.status_id,
                db.get(Task, r.task_id).revision if r.task_id else None,
                db.get(Note, r.note_id).revision if r.note_id else None,
            )
            for r in db.scalars(
                select(StructureRecord).where(StructureRecord.owner_id == owner).order_by(StructureRecord.id)
            )
        ]
    )


def schema_permission(db, owner):
    from .access import actor, role

    if role(db, owner, actor(db, owner)) != "owner":
        raise DomainError("SCHEMA_OWNER_REQUIRED", "Only the workspace owner can change its structure.", 403)


def preview(db, owner, args, command_id):
    from .access import actor

    schema_permission(db, owner)
    schema = ensure(db, owner)
    assert_schema(schema, args.expected_revision)
    from .field_library import prepare
    definition = Definition.model_validate(prepare(
        args.definition.model_dump(mode="json"), schema.definition, args.definition.model_fields_set,
        {t.id for t in args.definition.types if "opens_as" not in t.model_fields_set},
    )).model_dump(mode="json")
    types = {t["id"]: t for t in definition["types"]}
    oldtypes = {t["id"]: t for t in schema.definition["types"]}
    affected, issues, mappings = [], [], args.status_mappings
    for row in db.scalars(select(StructureRecord).where(StructureRecord.owner_id == owner)):
        t = types.get(row.type_id)
        if not t:
            issues.append(
                {"record_id": row.id, "message": "Archive a type instead of removing existing records."}
            )
            continue
        if t != oldtypes.get(row.type_id):
            affected.append({"id": row.id, "title": row.title})
        status = mappings.get(t["id"], {}).get(row.status_id, row.status_id)
        if status and status not in {s["id"] for s in t["statuses"]}:
            issues.append({"record_id": row.id, "message": "Map an existing status before removing it."})
        if row.parent_id and db.get(StructureRecord, row.parent_id).type_id not in t["parent_types"]:
            issues.append(
                {"record_id": row.id, "message": "Move this record before disallowing its parent type."}
            )
        current_record = data(db, row, schema)
        if "work" in t["capabilities"] and len(current_record["body"]) > 20000:
            issues.append(
                {
                    "record_id": row.id,
                    "message": "Shorten task details below 20,000 characters before adding actionable behavior; original content is retained.",
                }
            )
        current_values = current_record["values"]
        missing = set(current_values) - {f["id"] for f in t["fields"]}
        if missing:
            issues.append(
                {
                    "record_id": row.id,
                    "message": "Archive fields with saved values instead of removing their definitions.",
                }
            )
        for field in t["fields"]:
            if field["id"] in current_values:
                try:
                    validate_values(
                        db, owner, t, {field["id"]: current_values[field["id"]]}, allow_archived=True
                    )
                except DomainError as exc:
                    issues.append({"record_id": row.id, "message": exc.message})
        if row.task_id and "work" not in t["capabilities"]:
            active = db.scalar(
                select(Schedule.id)
                .where(Schedule.task_id == row.task_id, Schedule.status == "active")
                .limit(1)
            )
            if active or db.get(Task, row.task_id).external:
                issues.append(
                    {
                        "record_id": row.id,
                        "message": "Resolve active alerts and external bindings before removing actionable work.",
                    }
                )
    for relation in definition["relationships"]:
        links = list(
            db.scalars(
                select(StructureLink).where(
                    StructureLink.owner_id == owner, StructureLink.relationship_id == relation["id"]
                )
            )
        )
        if relation["cardinality"] in {"one_to_one", "many_to_one"} and len(
            {l.source_id for l in links}
        ) < len(links):
            issues.append({"message": "Resolve multiple outgoing links before narrowing cardinality."})
        if relation["cardinality"] in {"one_to_one", "one_to_many"} and len(
            {l.target_id for l in links}
        ) < len(links):
            issues.append({"message": "Resolve multiple incoming links before narrowing cardinality."})
    for link in db.scalars(select(StructureLink).where(StructureLink.owner_id == owner)):
        r = next((r for r in definition["relationships"] if r["id"] == link.relationship_id), None)
        if (
            not r
            or db.get(StructureRecord, link.source_id).type_id not in r["source_types"]
            or db.get(StructureRecord, link.target_id).type_id not in r["target_types"]
        ):
            issues.append({"link_id": link.id, "message": "Preserve or archive relationships still in use."})
    from types import SimpleNamespace
    from .record_contents import validate_block
    proposed = SimpleNamespace(definition=definition)
    for relation in definition["relationships"]:
        if relation.get("behavior") == "blocks" and not relation["archived"]:
            for link in db.scalars(select(StructureLink).where(
                StructureLink.owner_id == owner, StructureLink.relationship_id == relation["id"])):
                try:
                    validate_block(db, owner, proposed,
                        owned(db, StructureRecord, link.source_id, owner),
                        owned(db, StructureRecord, link.target_id, owner))
                except DomainError as exc:
                    issues.append({"link_id":link.id,"message":exc.message})
    impact = {
        "affected_count": len(affected),
        "affected_records": affected[:100],
        "issues": issues[:100],
        "blocking_count": len(issues),
        "snapshot": snapshot_records(db, owner),
        "status_mappings": mappings,
        "before": schema.definition,
        "request_id": command_id.split(":")[0],
    }
    row = StructureProposal(
        owner_id=owner,
        account_id=actor(db, owner),
        schema_revision=schema.revision,
        definition=definition,
        impact=impact,
        expires_at=now() + timedelta(minutes=30),
    )
    db.add(row)
    db.flush()
    return proposal_data(row)


def proposal_data(row):
    return {
        "id": row.id,
        "schema_revision": row.schema_revision,
        "definition": row.definition,
        "impact": {k: v for k, v in row.impact.items() if k not in {"snapshot", "before", "request_id"}},
        "expires_at": row.expires_at.isoformat(),
        "applied_at": row.applied_at.isoformat() if row.applied_at else None,
        "requires_confirmation": True,
    }


def apply(db, owner, args, command_id):
    from .access import actor
    from .domain import enqueue_job

    schema_permission(db, owner)
    schema = ensure(db, owner)
    assert_schema(schema, args.expected_revision)
    proposal = owned(db, StructureProposal, args.proposal_id, owner, lock=True)
    if proposal.account_id != actor(db, owner) or proposal.expires_at <= now() or proposal.applied_at:
        raise DomainError("STALE_PROPOSAL", "Preview the current structure again before applying it.", 409)
    if proposal.schema_revision != schema.revision or proposal.impact["snapshot"] != snapshot_records(
        db, owner
    ):
        raise DomainError(
            "STALE_PROPOSAL", "Records changed since the preview. Review an updated preview.", 409
        )
    if proposal.impact["blocking_count"]:
        raise DomainError("SCHEMA_MIGRATION_REQUIRED", "Resolve the issues shown in the preview first.", 409)
    if proposal.impact["request_id"] == command_id.split(":")[0]:
        raise DomainError(
            "SCHEMA_CONFIRMATION_REQUIRED",
            "Present the preview and wait for the user's next response before applying it.",
            409,
        )
    before_entries = dict(definition_entries(schema.definition))
    after_entries = dict(definition_entries(proposal.definition))
    changed_definitions = {
        key
        for key in before_entries.keys() | after_entries.keys()
        if before_entries.get(key) != after_entries.get(key)
    }
    schema.definition = proposal.definition
    schema.revision += 1
    schema.updated_at = now()
    for row in db.scalars(select(StructureRecord).where(StructureRecord.owner_id == owner)):
        status = proposal.impact["status_mappings"].get(row.type_id, {}).get(row.status_id)
        if status:
            row.status_id = status
        row.schema_revision = schema.revision
        t = record_type(schema, row.type_id, archived=True)
        if status and row.task_id:
            from .domain import TaskUpdate, mutate

            task = db.get(Task, row.task_id)
            meaning = next(s["meaning"] for s in t["statuses"] if s["id"] == status)
            mutate(
                db,
                owner,
                "task.update",
                TaskUpdate(task_id=task.id, expected_revision=task.revision, status=meaning),
                command_id,
            )
        if row.task_id:
            core = db.get(Task, row.task_id)
            row.title = core.title
            row.body = core.notes
        if row.note_id:
            core = db.get(Note, row.note_id)
            row.title = core.title
            row.body = core.content
        if row.task_id and "work" not in t["capabilities"]:
            task = db.get(Task, row.task_id)
            task.archived = True
            task.revision += 1
            row.provenance = {**row.provenance, "retired_task_id": row.task_id}
            row.task_id = None
        if row.note_id and "content" not in t["capabilities"]:
            row.provenance = {**row.provenance, "retired_note_id": row.note_id}
            row.note_id = None
        if "work" in t["capabilities"] and row.status_id is None:
            row.status_id = next(s["id"] for s in t["statuses"] if s["meaning"] in {"open", "backlog"})
        sync_capabilities(db, owner, row, t, {}, command_id)
    for key, entry in definition_entries(schema.definition):
        state = db.get(FieldUnderstanding, (owner, key))
        digest = fingerprint(entry)
        if not state:
            state = FieldUnderstanding(owner_id=owner, definition_id=key, fingerprint=digest)
            db.add(state)
        elif state.fingerprint == digest:
            continue
        else:
            state.fingerprint = digest
            state.revision += 1
            state.updated_at = now()
        state.status = "assessing"
        state.questions = []
        state.deferred_until = None
        state.understanding = {}
        state.answers = []
        enqueue_job(db, owner, "assess_field", {"definition_id": key, "fingerprint": digest})
    from .structure_models import RoutingPattern

    for pattern in db.scalars(
        select(RoutingPattern).where(RoutingPattern.owner_id == owner, RoutingPattern.status == "active")
    ):
        from .routing import rule_dependencies, validate_assignment

        try:
            validate_assignment(db, owner, schema, pattern.condition["type_id"], pattern.assignment)
            affected = bool(
                rule_dependencies(db, schema, pattern.condition["type_id"], pattern.assignment)
                & changed_definitions
            )
        except DomainError:
            affected = True
        if affected:
            pattern.status = "needs_review"
        else:
            pattern.schema_revision = schema.revision
        pattern.revision += 1
    proposal.applied_at = now()
    db.flush()
    emit(db, owner, "structure.changed", owner, schema.revision)
    return {
        "revision": schema.revision,
        "proposal_id": proposal.id,
        "applied": True,
        "definition": schema.definition,
    }


def sync_capabilities(db, owner, row, t, supplied, command_id, *, status=True, parent=False):
    from .domain import TaskCreate, TaskUpdate, mutate

    if "work" in t["capabilities"]:
        bindings = {f["id"]: f["binding"] for f in t["fields"] if f.get("binding") in TASK_BINDINGS}
        values = {binding: supplied[key] for key, binding in bindings.items() if key in supplied}
        # Cleared bindings (including reverted additions) return to the core task defaults.
        values.update({k: v for k, v in BINDING_DEFAULTS.items() if k in values and values[k] is None})
        meaning = next((s["meaning"] for s in t["statuses"] if s["id"] == row.status_id), "open")
        if not row.task_id:
            creation = TaskCreate(
                title=row.title,
                notes=row.body,
                status=meaning if meaning in {"open", "backlog"} else "open",
                **values,
            )
            core = mutate(db, owner, "task.create", creation, command_id)
            row.task_id = core["id"]
            if meaning not in {"open", "backlog"} or row.archived:
                mutate(
                    db,
                    owner,
                    "task.update",
                    TaskUpdate(
                        task_id=row.task_id,
                        expected_revision=core["revision"],
                        status=meaning,
                        archived=row.archived,
                    ),
                    command_id,
                )
        else:
            task = owned(db, Task, row.task_id, owner, lock=True)
            changes = {**values}
            if task.title != row.title:
                changes["title"] = row.title
            if task.notes != row.body:
                changes["notes"] = row.body
            # A workflow without the task's meaning shows the nearest status; only an explicit choice moves the task.
            if status and task.status != meaning:
                changes["status"] = meaning
            if parent and not task.external:
                chain = parent_chain(db, owner, row.parent_id, self_id=row.id)
                project = next((p for home in chain if (p := db.get(Project, home.id)) and p.owner_id == owner), None)
                project_id = project.id if project else None
                if task.project_id != project_id:
                    changes["project_id"] = project_id
            if task.archived != row.archived:
                changes["archived"] = row.archived
            if changes:
                mutate(
                    db,
                    owner,
                    "task.update",
                    TaskUpdate(task_id=task.id, expected_revision=task.revision, **changes),
                    command_id,
                )
                db.info.get("task_project_moves", set()).discard(task.id)
    if "content" in t["capabilities"]:
        from .note_schema import NoteCreate, NoteUpdate

        if not row.note_id:
            result = mutate(
                db, owner, "note.create", NoteCreate(title=row.title, content=row.body), command_id
            )
            row.note_id = result["id"]
        else:
            note = owned(db, Note, row.note_id, owner, lock=True)
            if (note.title, note.content, note.archived) != (row.title, row.body, row.archived):
                mutate(
                    db,
                    owner,
                    "note.update",
                    NoteUpdate(
                        note_id=note.id,
                        expected_revision=note.revision,
                        title=row.title,
                        content=row.body,
                        archived=row.archived,
                    ),
                    command_id,
                )


def mutate(db, owner, tool, args, command_id):
    if tool == "structure.restore":
        from .structure_schema import Preview

        prior = owned(db, StructureProposal, args.proposal_id, owner)
        if not prior.applied_at:
            raise DomainError("INVALID_ARGUMENT", "Choose an applied structure change.")
        return preview(
            db,
            owner,
            Preview(expected_revision=args.expected_revision, definition=prior.impact["before"]),
            command_id,
        )
    if tool == "structure.preview":
        return preview(db, owner, args, command_id)
    if tool == "structure.apply":
        return apply(db, owner, args, command_id)
    schema = ensure(db, owner)
    assert_schema(schema, args.schema_revision)
    if tool == "record.link":
        source = owned(db, StructureRecord, args.source_id, owner, lock=True)
        target = owned(db, StructureRecord, args.target_id, owner)
        check_revision(source, args.expected_revision)
        relation = next(
            (
                r
                for r in schema.definition["relationships"]
                if r["id"] == args.relationship_id and not r["archived"]
            ),
            None,
        )
        if (
            not relation
            or source.type_id not in relation["source_types"]
            or target.type_id not in relation["target_types"]
        ):
            raise DomainError("INVALID_RELATIONSHIP", "Choose a relationship allowed between these types.")
        q = select(StructureLink).where(
            StructureLink.owner_id == owner, StructureLink.relationship_id == args.relationship_id
        )
        existing = db.scalar(
            q.where(StructureLink.source_id == source.id, StructureLink.target_id == target.id)
        )
        if not args.remove and relation.get("behavior") == "blocks":
            from .record_contents import validate_block
            validate_block(db, owner, schema, source, target)
        if args.remove:
            if existing:
                db.delete(existing)
        elif not existing:
            if source.archived or target.archived:
                raise DomainError("ARCHIVED_RECORD", "Restore records before linking them.")
            card = relation["cardinality"]
            if card in {"one_to_one", "many_to_one"} and db.scalar(
                q.where(StructureLink.source_id == source.id)
            ):
                raise DomainError(
                    "RELATIONSHIP_CARDINALITY", "This source already has its allowed relationship."
                )
            if card in {"one_to_one", "one_to_many"} and db.scalar(
                q.where(StructureLink.target_id == target.id)
            ):
                raise DomainError(
                    "RELATIONSHIP_CARDINALITY", "This target already has its allowed relationship."
                )
            db.add(
                StructureLink(
                    owner_id=owner,
                    relationship_id=args.relationship_id,
                    source_id=source.id,
                    target_id=target.id,
                )
            )
        source.revision += 1
        source.updated_at = now()
        db.flush()
        emit(db, owner, "record.changed", source.id, source.revision)
        return data(db, source, schema)
    creating = tool == "record.create"
    if creating:
        t = record_type(schema, args.type_id)
        status = (
            args.status_id
            or next((s["id"] for s in t["statuses"] if s["meaning"] == "open"), None)
            or next((s["id"] for s in t["statuses"] if s["meaning"] == "backlog"), None)
        )
        row = StructureRecord(
            owner_id=owner,
            type_id=args.type_id,
            title=args.title,
            body=args.body,
            status_id=status,
            schema_revision=schema.revision,
            values={},
            provenance={"origin": "explicit"},
        )
        db.add(row)
        db.flush()
    else:
        row = owned(db, StructureRecord, args.record_id, owner, lock=True)
        reconcile_core(db, row, schema)
        check_revision(row, args.expected_revision)
        t = record_type(schema, row.type_id, archived=True)
        # Core edits made elsewhere remain authoritative; do not copy stale display values back.
        current = data(db, row, schema)
        row.title = current["title"]
        row.body = current["body"]
        row.status_id = current["status_id"]
        supplied = nulls("record.update", {k: getattr(args, k) for k in ("title", "body", "status_id", "archived") if k in args.model_fields_set})
        for key, value in supplied.items():
            setattr(row, key, value)
        row.revision += 1
        row.updated_at = now()
    if "work" in t["capabilities"] and row.status_id is None:
        raise DomainError("INVALID_STATUS", "Actionable records need a workflow status.")
    if row.status_id is not None and row.status_id not in {s["id"] for s in t["statuses"]}:
        raise DomainError("INVALID_STATUS", "Choose a status from this type's workflow.")
    if "parent_id" in args.model_fields_set:
        validate_parent(db, owner, row, t, args.parent_id)
        row.parent_id = args.parent_id
    if tool == "record.update" and "local_notes" in args.model_fields_set:
        row.local_notes = args.local_notes
    if tool == "record.update" and "sort_order" in args.model_fields_set:
        row.sort_order = args.sort_order
    clean = validate_values(db, owner, t, args.values)
    if tool == "record.create":
        from .routing import suggest

        routing = suggest(db, owner, row.type_id, row.title, args.model_dump(exclude_unset=True))
        row.provenance = {**row.provenance, "routing": routing}
        if routing.get("mode") == "automatic":
            inferred = routing["assignment"]
            if "parent_id" in inferred:
                row.parent_id = inferred["parent_id"]
            clean = {**inferred.get("values", {}), **clean}
    if creating or "move_before_id" in args.model_fields_set or ("parent_id" in args.model_fields_set and "sort_order" not in args.model_fields_set):
        siblings = list(db.scalars(select(StructureRecord).where(
            StructureRecord.owner_id == owner, StructureRecord.parent_id == row.parent_id,
            StructureRecord.id != row.id).order_by(StructureRecord.sort_order, StructureRecord.id)))
        before = getattr(args, "move_before_id", None)
        if before and before not in {s.id for s in siblings}:
            raise DomainError("INVALID_PARENT", "Choose a record in the destination home.")
        index = next((i for i, s in enumerate(siblings) if s.id == before), len(siblings))
        left = siblings[index-1].sort_order if index else None
        right = siblings[index].sort_order if index < len(siblings) else None
        rank = (left / 2 + right / 2 if left is not None and right is not None else
                right - 1024 if right is not None else (left or 0) + 1024)
        if rank in (left, right):
            raise DomainError("ORDER_DENSE", "These positions are too close. Move this item to the end first.")
        row.sort_order = rank
    row.values = {**row.values, **clean}
    if tool == "record.update" and args.reset_fields:
        permitted = {f["id"] for f in t["fields"] if not f["binding"]}
        if set(args.reset_fields) - permitted:
            raise DomainError("INVALID_FIELD", "Only classification overrides can be reset.")
        row.values = {k: v for k, v in row.values.items() if k not in args.reset_fields}
    timeline = {
        f["binding"]: row.values.get(f["id"])
        for f in t["fields"]
        if f["binding"] in {"start_date", "target_date"}
    }
    if (
        timeline.get("start_date")
        and timeline.get("target_date")
        and timeline["target_date"] < timeline["start_date"]
    ):
        raise DomainError("INVALID_ARGUMENT", "Target date must not precede the start.")
    sync_capabilities(
        db, owner, row, t, clean, command_id,
        status=creating or "status_id" in args.model_fields_set,
        parent="parent_id" in args.model_fields_set,
    )
    if row.task_id:
        from .hierarchy import project_parent
        project_parent(db, row)
    remember_core(db, row)
    db.flush()
    from .routing import human_command, observe

    # Browser edits are confirmations; agent, agent-driven UI and imported assignments never self-reinforce.
    routing = row.provenance.get("routing", {})
    assigned = routing.get("assignment", {})
    # Rule-routed records count only when the owner departs from the rule's assignment.
    departs = not routing.get("rules") or (tool == "record.update" and (
        ("parent_id" in args.model_fields_set and args.parent_id != assigned.get("parent_id"))
        or any(assigned.get("values", {}).get(k) != v for k, v in args.values.items())
    ))
    human = (
        human_command(command_id)
        and ("parent_id" in args.model_fields_set or bool(args.values))
        and departs
    )
    observe(db, row, command_id, human=human)
    remember_core(db, row)
    emit(db, owner, "record.changed", row.id, row.revision)
    return data(db, row, schema)


def observe_core(db, owner, tool, result, command_id, arguments=None):
    """Keep typed services discoverable through the registry without a second authority."""
    if tool.split(".")[0] not in {"task", "note"} or not isinstance(result, dict):
        return
    if not result.get("id") and isinstance(result.get("tasks"), list):
        # note.tasks, task.batch and task.selection_update report several tasks.
        applied = result.get("applied_ids")
        for item in result["tasks"]:
            if isinstance(item, dict) and (applied is None or item.get("id") in applied):
                observe_core(db, owner, "task.update", item, command_id)
        return
    if not result.get("id"):
        return
    kind = tool.split(".")[0]
    model = Task if kind == "task" else Note
    core = db.get(model, result["id"])
    if not core or core.owner_id != owner:
        return
    schema = ensure(db, owner)
    col = StructureRecord.task_id if kind == "task" else StructureRecord.note_id
    row = db.scalar(select(StructureRecord).where(StructureRecord.owner_id == owner, col == core.id))
    if row:
        row.title = core.title
        row.body = core.notes if kind == "task" else core.content
        row.archived = core.archived
        if kind == "task":
            t = record_type(schema, row.type_id, archived=True)
            row.status_id = task_status(t, core.status, row.status_id)
            if core.id in db.info.get("task_project_moves", set()):
                db.info["task_project_moves"].discard(core.id)
                follow_project(db, owner, row, core, t)
        row.revision += 1
        row.updated_at = now()
    else:
        t = import_type(schema, kind)
        parent = getattr(core, "project_id", None) or core.area_id or core.space_id
        if parent:
            home = db.get(StructureRecord, parent)
            if not home or home.archived or home.type_id not in t["parent_types"]:
                parent = None
        row = StructureRecord(
            id=core.id,
            owner_id=owner,
            type_id=t["id"],
            title=core.title,
            body=core.notes if kind == "task" else core.content,
            parent_id=parent,
            sort_order=(db.scalar(select(func.max(StructureRecord.sort_order)).where(StructureRecord.owner_id == owner, StructureRecord.parent_id == parent)) or 0)+1024,
            task_id=core.id if kind == "task" else None,
            note_id=core.id if kind == "note" else None,
            status_id=task_status(t, core.status) if kind == "task" else None,
            archived=core.archived,
            schema_revision=schema.revision,
            values={},
            provenance={"origin": "agent" if ":" in command_id else "manual"},
        )
        db.add(row)
    db.flush()
    if kind == "task":
        from .hierarchy import from_task
        explicit_home = ("parent_task_id" in (arguments or {})
                         or core.id in db.info.get("task_parent_moves", set()))
        db.info.get("task_parent_moves", set()).discard(core.id)
        from_task(db, owner, row, core, schema, explicit=explicit_home)
    if kind == "note" and row.note_id in db.info.get("note_tag_changes", set()):
        from .routing import human_command, observe
        observe(db, row, command_id, human=human_command(command_id))
        db.info["note_tag_changes"].discard(row.note_id)
    if tool == "task.create" and arguments is not None:
        from .routing import suggest

        explicit = (
            {"parent_id": row.parent_id}
            if row.parent_id or any(arguments.get(k) for k in ("project_id", "space_id", "area_id"))
            else {}
        )
        choice = suggest(db, owner, row.type_id, row.title, explicit)
        row.provenance = {**row.provenance, "routing": choice}
        if choice.get("mode") == "automatic":
            assignment = choice["assignment"]
            if "parent_id" in assignment:
                row.parent_id = assignment["parent_id"]
            row.values = {**assignment.get("values", {}), **row.values}
        from .hierarchy import project_parent
        project_parent(db, row)
        result["record_id"] = row.id
        result["routing"] = choice
        db.flush()
    remember_core(db, row)
    if kind == "task":
        result.update(serial(core))
    emit(db, owner, "record.changed", row.id, row.revision)


def follow_project(db, owner, row, core, t):
    """Legacy project moves update the record home when the destination has a registry row."""
    target = core.project_id
    if not target:
        if not (row.parent_id and db.get(Project, row.parent_id)):
            return
        target = next((h for h in (core.area_id, core.space_id) if h and db.get(StructureRecord, h)), None)
    if target:
        home = db.get(StructureRecord, target)
        if not home or home.owner_id != owner or home.archived or home.type_id not in t["parent_types"]:
            return
        try:
            parent_chain(db, owner, target, self_id=row.id)
        except DomainError:
            return
    row.parent_id = target


def import_type(schema, kind):
    capability = "work" if kind == "task" else "content"
    candidates = [
        t for t in schema.definition["types"] if not t["archived"] and capability in t["capabilities"]
    ]
    selected = next((t for t in candidates if t["id"] == kind), None) or next(iter(candidates), None)
    if not selected:
        raise DomainError(
            "NO_COMPATIBLE_TYPE",
            "Enable an actionable or content collection in Structure before creating this record.",
        )
    return selected


def remember_core(db, row):
    revisions = {
        kind: db.get(model, identity).revision
        for kind, model, identity in (("task", Task, row.task_id), ("note", Note, row.note_id))
        if identity
    }
    if revisions:
        row.provenance = {**row.provenance, "core_revisions": revisions}


def reconcile_core(db, row, schema):
    """Worker/integration updates must invalidate an older open record card too."""
    if row.task_id and row.provenance.get("home_version") == 1:
        from .hierarchy import project_parent
        project_parent(db, row)
    previous = row.provenance.get("core_revisions", {})
    revisions = {
        kind: db.get(model, identity).revision
        for kind, model, identity in (("task", Task, row.task_id), ("note", Note, row.note_id))
        if identity
    }
    if not revisions:
        return
    current = data(db, row, schema)
    changed = (bool(previous) and previous != revisions) or any(
        getattr(row, key) != current[key] for key in ("title", "body", "archived", "status_id")
    )
    if changed:
        for key in ("title", "body", "archived", "status_id"):
            setattr(row, key, current[key])
        row.revision += 1
        row.updated_at = now()
    remember_core(db, row)
    return changed


def core_drift(db, owner):
    """Cheap unlocked check for core rows the registry has not adopted or reconciled yet."""
    from sqlalchemy import or_

    for kind, model, column in (("task", Task, StructureRecord.task_id), ("note", Note, StructureRecord.note_id)):
        revision = StructureRecord.provenance["core_revisions"][kind].as_integer()
        drift = (
            select(StructureRecord.id)
            .join(model, model.id == column)
            .where(StructureRecord.owner_id == owner, revision.is_distinct_from(model.revision))
            .limit(1)
        )
        if db.scalar(drift):
            return True
        known = select(StructureRecord.id).where(
            StructureRecord.owner_id == owner,
            or_(
                column == model.id,
                StructureRecord.id == model.id,
                StructureRecord.provenance["retired_" + kind + "_id"].astext == model.id,
            ),
        )
        if db.scalar(select(model.id).where(model.owner_id == owner, ~known.exists()).limit(1)):
            return True
    return False


def sync_core_records(db, owner, schema):
    """Adopt recurring/imported core rows without reviving intentionally retired capabilities."""
    advisory(db, "workspace:" + owner)
    rows = list(db.scalars(select(StructureRecord).where(StructureRecord.owner_id == owner)))
    for row in rows:
        if reconcile_core(db, row, schema):
            # Open cards hold the old revision; tell them before their next save conflicts.
            emit(db, owner, "record.changed", row.id, row.revision)
    for kind, model, column in (
        ("task", Task, StructureRecord.task_id),
        ("note", Note, StructureRecord.note_id),
    ):
        retired = {r.provenance.get("retired_" + kind + "_id") for r in rows}
        missing = db.scalars(
            select(model).where(
                model.owner_id == owner,
                ~select(StructureRecord.id)
                .where(StructureRecord.owner_id == owner, column == model.id)
                .exists(),
            )
        )
        for core in missing:
            if core.id in retired or db.get(StructureRecord, core.id):
                continue
            try:
                import_type(schema, kind)
            except DomainError:
                continue
            observe_core(db, owner, kind + ".update", serial(core), "registry-import")
    db.flush()
