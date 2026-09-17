"""Saved changes and explicit, revision-checked compensating commands."""

from contextlib import contextmanager
from uuid import uuid4

from fastapi.encoders import jsonable_encoder
from sqlalchemy import event, inspect, or_, select

from . import models
from .access import actor, authorize_execution, role
from .config import get_settings
from .domain import DomainError, advisory, emit, owned
from .models import ActionChange, Command, Job
from .work_crypto import seal, unseal

MODELS = {
    "record_link": models.StructureLink,
    "record": models.StructureRecord,
    "task": models.Task,
    "note": models.Note,
    "project": models.Project,
    "goal": models.Goal,
    "space": models.Space,
    "area": models.Area,
    "actor": models.Actor,
    "schedule": models.Schedule,
    "planning": models.PlanningEntry,
}
KINDS = {model: kind for kind, model in MODELS.items()}
SKIP = {
    "occurrence_id",
    "is_template",
    "id",
    "owner_id",
    "revision",
    "created_at",
    "updated_at",
    "completed_at",
    "embedding",
    "external", "schema_revision", "provenance", "legacy_kind",
}


def ignored(kind):
    return SKIP | ({"task_id", "note_id"} if kind == "record" else set())

def snapshot(obj, *, before=False):
    values = {}
    state = inspect(obj)
    for column in obj.__table__.columns:
        history = state.attrs[column.name].history
        value = history.deleted[0] if before and history.deleted else getattr(obj, column.name)
        values[column.name] = value
    return jsonable_encoder(values)


@contextmanager
def journal(db, owner, command_id, tool, arguments=None):
    # Existing local installations without the recovery key retain their commands.
    # Cloud/durable work requires a configured key before accepting a request.
    if not get_settings().integration_encryption_key or db.info.get("action_journal"):
        yield
        return
    tracked = {}
    db.info["action_journal"] = True

    def record(session, context, instances):
        for obj in [*session.new, *session.dirty, *session.deleted]:
            if isinstance(obj, models.StructureLink) and tool != "record.link": continue
            if isinstance(obj, models.StructureRecord) and tool == "record.link": continue
            if isinstance(obj,models.StructureRecord) and not tool.startswith(("record.","routing.apply")):continue
            if isinstance(obj,(models.Task,models.Note)) and tool.startswith(("record.","routing.apply")):continue
            if type(obj) not in KINDS or getattr(obj, "owner_id", None) != owner:
                continue
            if id(obj) not in tracked:
                tracked[id(obj)] = (obj, None if obj in session.new else snapshot(obj, before=True))

    event.listen(db, "before_flush", record)
    try:
        yield
        db.flush()
        for obj, before in tracked.values():
            after = None if inspect(obj).deleted else snapshot(obj)
            if before == after:
                continue
            if (
                after
                and arguments
                and any(
                    key in arguments for key in ("task_ids", "project_ids", "goal_ids", "related_note_ids")
                )
            ):
                after["_undo_blocked"] = (
                    "This action changed links between records. Review those links in the record before reversing it."
                )
            db.add(
                ActionChange(
                    id=str(uuid4()),
                    owner_id=owner,
                    account_id=actor(db, owner),
                    command_id=command_id,
                    tool=tool,
                    entity_kind=KINDS[type(obj)],
                    entity_id=obj.id,
                    before_ciphertext=seal(before) if before else None,
                    after_ciphertext=seal(after) if after else None,
                )
            )
    finally:
        event.remove(db, "before_flush", record)
        db.info.pop("action_journal", None)


def has_linked_records(db, model, identity):
    # Index/cache references are not user work. All real incoming record links
    # must be reviewed before undoing a creation, even if they did not bump revision.
    for table in models.Base.metadata.sorted_tables:
        if table.name in {"task_references", "note_embeddings", "routing_observations"} or (table.name=="structure_records" and model in {models.Task,models.Note}):
            continue
        for column in table.columns:
            if (
                any(fk.column.table is model.__table__ for fk in column.foreign_keys)
                and db.scalar(select(column).where(column == identity).limit(1)) is not None
            ):
                return True
    return False


def inverse(db, change, *, lock=False):
    from .domain import COMMANDS

    model = MODELS.get(change.entity_kind)
    before, after = unseal(change.before_ciphertext), unseal(change.after_ciphertext)
    if change.reverted_by:
        return None, "Already reverted."
    if change.entity_kind == "record_link":
        return inverse_link(db, change, before, after, lock=lock)
    if after.get("_undo_blocked"):
        return None, after["_undo_blocked"]
    if not model or not after:
        return None, "This change cannot be restored automatically."
    receipt = db.get(Command, (change.owner_id, change.command_id))
    data = receipt.result.get("data", {}) if receipt else {}
    remote_id = data.get("job_id") or (data.get("external") or {}).get("job_id")
    remote = db.get(Job, remote_id) if remote_id else None
    if remote and remote.status != "succeeded":
        return None, "Wait for synchronization to settle before reversing this change."
    row = owned(db, model, change.entity_id, change.owner_id, lock=lock)
    if change.entity_kind == "record":
        recorded = after.get("provenance", {}).get("core_revisions", {})
        for kind, core_model, identity in (("task", models.Task, row.task_id), ("note", models.Note, row.note_id)):
            if identity and recorded.get(kind) != db.get(core_model, identity).revision:
                return None, "This record changed through another view or integration. Review it before reverting."
    if not before:
        if not hasattr(row, "archived"):
            return None, "Open this record to cancel or remove it."
        if snapshot(row) != after:
            return None, "This record changed after it was created. Review it before archiving."
        if has_linked_records(db, model, row.id):
            return None, "Other records are linked to this creation. Review those links before archiving it."
        changes = {"archived": True}
    else:
        changes = {key: before.get(key) for key in after if key not in ignored(change.entity_kind) and before.get(key) != after[key]}
        if not changes:
            return None, "This receipt has no reversible field changes."
        current = snapshot(row)
        if any(current.get(key) != after.get(key) for key in changes):
            return None, "One of these fields changed later. Open the record to compare changes."
    kind = change.entity_kind
    tool = kind + ".update"
    spec = COMMANDS.get(tool)
    if not spec or any(key not in spec.model_fields for key in changes):
        return None, "Some fields need the record's own controls to reverse safely."
    if kind == "planning":
        return None, "Open this event to review its dates and any Google synchronization before changing it."
    arguments = {kind + "_id": row.id, "expected_revision": row.revision, **changes}
    if kind=="record":
        from .structure import ensure, record_type
        schema = ensure(db, change.owner_id)
        arguments["schema_revision"] = schema.revision
        if before and "values" in changes:
            old,new=before.get("values",{}),after.get("values",{})
            arguments["values"]={k:old[k] for k in old if old.get(k)!=new.get(k)}
            bindings = {f["id"] for f in record_type(schema, row.type_id, archived=True)["fields"] if f["binding"]}
            added = set(new) - set(old)
            arguments["values"].update({k:None for k in added & bindings})
            arguments["reset_fields"] = sorted(added - bindings)
    # Automatic inverse is only for sparse local edits. Relationships have their own validation.
    return (tool, arguments), "Archive this created record" if not before else "Restore the changed fields"


def public_change(db, row):
    before, after = unseal(row.before_ciphertext), unseal(row.after_ciphertext)
    if row.entity_kind == "record_link":
        return public_link(db, row, before, after)
    try:
        command, reason = inverse(db, row)
    except DomainError as exc:
        command, reason = None, exc.message
    fields = {
        key: {"before": before.get(key), "after": value}
        for key, value in after.items()
        if key not in ignored(row.entity_kind)
        and not key.startswith("_")
        and (before.get(key) != value if before else value not in (None, "", False, 0, []))
    }
    references = {
        "space_id": models.Space,
        "area_id": models.Area,
        "project_id": models.Project,
        "goal_id": models.Goal,
        "parent_goal_id": models.Goal,
        "parent_task_id": models.Task,
        "assignee_id": models.Actor,
        "task_id": models.Task,
    }
    for key, model in references.items():
        if key not in fields:
            continue
        values = fields.pop(key)
        for side, identity in values.items():
            target = db.get(model, identity) if identity else None
            values[side] = (
                (getattr(target, "title", None) or getattr(target, "name", None))
                if target and target.owner_id == row.owner_id
                else None
            )
        fields[key.removesuffix("_id")] = values
    label = row.entity_kind
    if row.entity_kind == "record":
        from .structure import ensure, record_type
        definition = record_type(ensure(db, row.owner_id), after["type_id"], archived=True)
        label = definition["name"]
        value_change = fields.pop("values", None)
        if value_change:
            old, new = value_change["before"] or {}, value_change["after"] or {}
            for field in definition["fields"]:
                key = field["id"]
                if old.get(key) == new.get(key): continue
                def display(value):
                    if value is None: return None
                    if field["kind"] == "relation":
                        def title(identity):
                            target = db.get(models.StructureRecord, identity)
                            return target.title if target and target.owner_id == row.owner_id else "Unavailable record"
                        return [title(v) for v in value] if isinstance(value, list) else title(value)
                    if field["kind"] in {"select", "multiselect"}:
                        names = {o["id"]:o["name"] for o in field["options"]}
                        return [names.get(v,v) for v in value] if isinstance(value,list) else names.get(value,value)
                    return value
                fields[field.get("binding") or field["name"]] = {"before":display(old.get(key)),"after":display(new.get(key))}
        home = fields.pop("parent_id", None)
        if home:
            fields["Main home"] = {side:(target.title if (target:=db.get(models.StructureRecord, identity)) and target.owner_id==row.owner_id else None) if identity else None for side,identity in home.items()}
        status = fields.pop("status_id", None)
        if status:
            labels = {s["id"]:s["name"] for s in definition["statuses"]}
            fields["status"] = {side:labels.get(value,value) for side,value in status.items()}
        fields.pop("type_id", None)
    operation = "created" if not before else "updated"
    if before and before.get("status") != after.get("status"):
        if after.get("status") == "completed":
            operation = "completed"
        elif before.get("status") == "completed":
            operation = "reopened"
    if before and before.get("archived") != after.get("archived"):
        operation = "archived" if after.get("archived") else "restored"
    title = after.get("title", after.get("name", row.entity_kind.title()))
    return {
        "id": row.id,
        "command_id": row.command_id,
        "kind": row.entity_kind,
        "entity_id": row.entity_id,
        "title": title,
        "operation": operation,
        "summary": f"{operation.capitalize()} {label}: {title}",
        "request_id": row.command_id.split(":")[0],
        "revert_command_id": row.reverted_by,
        "fields": fields,
        "can_revert": bool(command),
        "revert_reason": reason,
        "reverted": bool(row.reverted_by),
    }


def changes_for_work(db, work):
    rows = list(
        db.scalars(
            select(ActionChange)
            .where(
                ActionChange.owner_id == work.owner_id,
                ActionChange.account_id == work.account_id,
                or_(
                    ActionChange.command_id.startswith(work.id + ":"),
                    ActionChange.command_id.in_(work.result.get("receipt_ids", [])),
                ),
            )
            .order_by(ActionChange.created_at, ActionChange.id)
        )
    )
    result = [
        public_change(db, change)
        for change in rows
        if change.entity_kind not in {"actor", "space", "area"}
        or change.tool.startswith(change.entity_kind + ".")
    ]
    # Include integration receipts and other non-reversible effects truthfully.
    known = {r.command_id for r in rows}
    for receipt in db.scalars(
        select(Command).where(
            Command.owner_id == work.owner_id,
            Command.account_id == work.account_id,
            or_(Command.id.startswith(work.id + ":"), Command.id.in_(work.result.get("receipt_ids", []))),
        )
    ):
        data = receipt.result.get("data", {})
        remote_id = data.get("job_id") or (data.get("external") or {}).get("job_id")
        remote = db.get(Job, remote_id) if remote_id else None
        if receipt.id in known:
            for item in result:
                if item["command_id"] == receipt.id:
                    item["remote_status"] = remote.status if remote else None
                    if remote and remote.status != "succeeded":
                        item["can_revert"] = False
                        item["revert_reason"] = (
                            "Wait for synchronization to settle before reversing this change."
                        )
            continue
        remote_result = remote.result or {} if remote else {}
        result.append(
            {
                "id": receipt.id,
                "command_id": receipt.id,
                "kind": "google_event"
                if remote and remote.kind == "google_write" and not remote_result.get("deleted")
                else "action",
                "entity_id": data.get("id") or remote_result.get("event_id"),
                "title": data.get("title", data.get("name", remote_result.get("title", "Saved action"))),
                "operation": "saved",
                "fields": {},
                "can_revert": False,
                "revert_reason": "Review this action in its record; automatic reversal is not available.",
                "remote_status": remote.status if remote else None,
                "reverted": False,
            }
        )
    return result


def revert(db, owner, account, change_id, command_id):
    from .domain import execute

    authorize_execution(db, owner, write=True)
    role(db, owner, account, lock=True)
    advisory(db, f"workspace:{owner}")
    advisory(db, f"revert:{change_id}")
    change = db.get(ActionChange, change_id, populate_existing=True, with_for_update=True)
    if not change or change.owner_id != owner or change.account_id != account:
        raise DomainError("NOT_FOUND", "That action is not available.", 404)
    if change.reverted_by:
        receipt = db.get(Command, (owner, change.reverted_by))
        return receipt.result if receipt else {"status": "succeeded", "already_reverted": True}
    action, reason = inverse(db, change, lock=True)
    if not action:
        raise DomainError("REVISION_CONFLICT", reason, 409)
    result = execute(db, owner, command_id, action[0], action[1])
    change.reverted_by = command_id
    emit(db, owner, "work.changed", change.command_id.split(":")[0])
    return result


def inverse_link(db, change, before, after, *, lock=False):
    from .structure import ensure
    saved = after or before
    source = owned(db, models.StructureRecord, saved["source_id"], change.owner_id, lock=lock)
    owned(db, models.StructureRecord, saved["target_id"], change.owner_id)
    if source.archived:
        return None, "Restore the source record before changing its links."
    current = db.scalar(select(models.StructureLink).where(models.StructureLink.owner_id == change.owner_id, models.StructureLink.source_id == saved["source_id"], models.StructureLink.target_id == saved["target_id"], models.StructureLink.relationship_id == saved["relationship_id"]))
    if after and (not current or current.id != change.entity_id):
        return None, "This relationship changed later. Review the current links."
    if not after and current:
        return None, "This relationship has already been restored."
    schema = ensure(db, change.owner_id)
    definition = next((r for r in schema.definition["relationships"] if r["id"] == saved["relationship_id"] and not r["archived"]), None)
    if not definition:
        return None, "This relationship definition changed. Review it in Structure."
    return ("record.link", {"source_id":source.id,"target_id":saved["target_id"],"relationship_id":saved["relationship_id"],"expected_revision":source.revision,"schema_revision":schema.revision,"remove":bool(after)}), "Remove this link" if after else "Restore this link"


def public_link(db, change, before, after):
    saved = after or before
    try:
        command, reason = (None, "Already reverted.") if change.reverted_by else inverse_link(db, change, before, after)
    except DomainError as exc:
        command, reason = None, exc.message
    source = db.get(models.StructureRecord, saved["source_id"])
    target = db.get(models.StructureRecord, saved["target_id"])
    operation = "linked" if after else "unlinked"
    return {"id":change.id,"command_id":change.command_id,"kind":"record","entity_id":saved["source_id"],"title":source.title if source else "Record","operation":operation,"summary":f"{operation.capitalize()}: {source.title if source else 'Record'} → {target.title if target else 'Record'}","request_id":change.command_id.split(":")[0],"revert_command_id":change.reverted_by,"fields":{"relationship":{"before":saved["relationship_id"] if before else None,"after":saved["relationship_id"] if after else None}},"can_revert":bool(command),"revert_reason":reason,"reverted":bool(change.reverted_by)}
