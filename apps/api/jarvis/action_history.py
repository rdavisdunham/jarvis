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
    "external",
}


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
        if table.name in {"task_references", "note_embeddings"}:
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
    if not before:
        if not hasattr(row, "archived"):
            return None, "Open this record to cancel or remove it."
        if snapshot(row) != after:
            return None, "This record changed after it was created. Review it before archiving."
        if has_linked_records(db, model, row.id):
            return None, "Other records are linked to this creation. Review those links before archiving it."
        changes = {"archived": True}
    else:
        changes = {key: before.get(key) for key in after if key not in SKIP and before.get(key) != after[key]}
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
    # Automatic inverse is only for sparse local edits. Relationships have their own validation.
    return (tool, arguments), "Archive this created record" if not before else "Restore the changed fields"


def public_change(db, row):
    before, after = unseal(row.before_ciphertext), unseal(row.after_ciphertext)
    try:
        command, reason = inverse(db, row)
    except DomainError as exc:
        command, reason = None, exc.message
    fields = {
        key: {"before": before.get(key), "after": value}
        for key, value in after.items()
        if key not in SKIP
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
        "summary": f"{operation.capitalize()} {row.entity_kind}: {title}",
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
