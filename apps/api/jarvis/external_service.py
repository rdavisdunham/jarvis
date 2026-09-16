"""One planner service behind the REST API, MCP and scoped queued agents."""

import copy
import hashlib
import json
from datetime import date, timedelta
from typing import Literal
from uuid import UUID

import jsonschema
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, or_, select

from . import bot_access, models
from .agent_work import public as public_work
from .agent_work import stable_id
from .db import session_scope
from .domain import DomainError, advisory, emit, execute, owned
from .work_crypto import seal

KINDS = {
    "task": models.Task,
    "note": models.Note,
    "project": models.Project,
    "goal": models.Goal,
    "space": models.Space,
    "area": models.Area,
    "actor": models.Actor,
}
Kind = Literal["task", "note", "project", "goal", "space", "area", "actor"]


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Search(Input):
    query: str = Field(default="", max_length=300)
    status: str | None = Field(default=None, max_length=30)
    space_id: UUID | None = None
    area_id: UUID | None = None
    project_id: UUID | None = None
    goal_id: UUID | None = None
    task_id: UUID | None = None
    parent_task_id: UUID | None = None
    assignee_id: UUID | None = None
    work_type: str | None = Field(default=None, max_length=80)
    due_from: date | None = None
    due_through: date | None = None
    archived: bool = False
    limit: int = Field(default=50, ge=1, le=100)
    offset: int = Field(default=0, ge=0, le=1000000)


class Mutation(Input):
    request_id: UUID
    tool: str = Field(max_length=70)
    arguments: dict


class RequestInput(Input):
    request_id: UUID
    message: str = Field(min_length=1, max_length=12000)
    thread_id: UUID | None = None


class ReplyInput(Input):
    request_id: UUID
    message: str = Field(min_length=1, max_length=12000)
    expected_revision: int = Field(ge=1)


def scope_for(kind):
    return ("tasks" if kind == "task" else "notes" if kind == "note" else "organization") + ":read"


def scrub(data, scopes):
    """Related records don't widen the scopes of the parent record."""
    if isinstance(data, list):
        return [scrub(item, scopes) for item in data]
    if not isinstance(data, dict):
        return data
    result = {k: scrub(v, scopes) for k, v in data.items()}
    if "notes:read" not in scopes and isinstance(result.get("notes"), list):
        result.pop("notes", None)
    if "tasks:read" not in scopes:
        for key in ("tasks", "task_count", "completed_task_count"):
            result.pop(key, None)
    if "organization:read" not in scopes:
        for key in ("goals", "projects", "spaces", "areas", "actors"):
            result.pop(key, None)
    return result


def record_data(db, row, scopes):
    from .notes import note_data
    from .productivity import data

    value = note_data(db, row) if isinstance(row, models.Note) else data(db, row)
    return scrub(value, scopes)


def get_record(db, bot, kind, identity):
    bot_access.authorize(db, bot.owner_id, scope_for(kind))
    return record_data(db, owned(db, KINDS[kind], str(identity), bot.owner_id), bot.scopes)


def search(db, bot, kind, args):
    bot_access.authorize(db, bot.owner_id, scope_for(kind))
    model = KINDS[kind]
    query = select(model).where(model.owner_id == bot.owner_id, model.archived == args.archived)
    if kind == "task":
        query = query.where(model.is_template.is_(False))
    if args.query:
        term = args.query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        columns = [
            getattr(model, key)
            for key in ("title", "name", "content", "notes", "description")
            if hasattr(model, key)
        ]
        query = query.where(or_(*(c.ilike("%" + term + "%", escape="\\") for c in columns)))
    values = args.model_dump(mode="json", exclude_none=True)
    for key in ("status", "space_id", "area_id", "project_id", "parent_task_id", "assignee_id", "work_type"):
        if key in values:
            if not hasattr(model, key):
                raise DomainError("INVALID_ARGUMENT", f"{key} is not a filter for {kind}.")
            if kind == "note" and key == "project_id":
                query = query.where(
                    or_(
                        model.project_id == values[key],
                        model.id.in_(
                            select(models.NoteProjectLink.note_id).where(
                                models.NoteProjectLink.project_id == values[key]
                            )
                        ),
                    )
                )
            else:
                query = query.where(getattr(model, key) == values[key])
    if args.task_id:
        if kind != "note":
            raise DomainError("INVALID_ARGUMENT", "task_id filters linked notes only.")
        query = query.where(
            model.id.in_(
                select(models.NoteTaskLink.note_id).where(
                    models.NoteTaskLink.task_id == str(args.task_id), models.NoteTaskLink.linked.is_(True)
                )
            )
        )
    if args.goal_id:
        projects = select(models.GoalProjectLink.project_id).where(
            models.GoalProjectLink.goal_id == str(args.goal_id)
        )
        if kind == "task":
            query = query.where(model.project_id.in_(projects))
        elif kind == "project":
            query = query.where(model.id.in_(projects))
        elif kind == "note":
            from .notes import scope_notes

            query = scope_notes(query, goal_id=str(args.goal_id))
        else:
            raise DomainError("INVALID_ARGUMENT", "goal_id filters tasks, projects or linked notes.")
    if args.due_from or args.due_through:
        if kind != "task":
            raise DomainError("INVALID_ARGUMENT", "Due-date filters apply to tasks only.")
        if args.due_from and args.due_through and args.due_from > args.due_through:
            raise DomainError("INVALID_ARGUMENT", "The start date must precede the end date.")
        if args.due_from:
            query = query.where(model.due_date >= args.due_from)
        if args.due_through:
            query = query.where(model.due_date <= args.due_through)
    total = db.scalar(select(func.count()).select_from(query.subquery()))
    rows = list(db.scalars(query.order_by(model.created_at, model.id).offset(args.offset).limit(args.limit)))
    return {
        "items": [record_data(db, row, bot.scopes) for row in rows],
        "total": total,
        "next_offset": args.offset + len(rows) if args.offset + len(rows) < total else None,
    }


def conversation(db, bot, thread_id=None):
    identity = stable_id(f"bot-conversation:{bot.id}:{thread_id or 'default'}")
    advisory(db, "bot-conversation:" + identity)
    row = db.get(models.Conversation, identity)
    if row is None:
        row = models.Conversation(
            id=identity, owner_id=bot.owner_id, device_id=bot.id, private=False, learning=False
        )
        db.add(row)
        db.flush()
    return row


def request_identity(bot, request_id):
    return stable_id(f"bot-request:{bot.id}:{request_id}")


def direct(db, bot, request_id, tool, arguments):
    """Receipt, journal, feed event and Activity item commit atomically with the effect."""
    identity = request_identity(bot, request_id)
    fingerprint = hashlib.sha256(json.dumps([tool, arguments], sort_keys=True).encode()).hexdigest()
    advisory(db, "work:" + identity)
    existing = db.get(models.AgentWork, identity)
    if existing:
        if tool != "action.revert":
            bot_access.check_command(db, bot.owner_id, tool, arguments)
        else:
            change = owned(db, models.ActionChange, arguments["action_id"], bot.owner_id)
            bot_access.check_command(db, bot.owner_id, change.entity_kind + ".update", {})
        if existing.input_hash != fingerprint:
            raise DomainError(
                "REVISION_CONFLICT", "This request ID was already used for different instructions.", 409
            )
        return direct_result(db, bot, existing)
    if tool == "action.revert":
        change = owned(db, models.ActionChange, arguments["action_id"], bot.owner_id)
        source = db.get(models.AgentWork, change.command_id.split(":")[0])
        if not source or source.credential_id != bot.id:
            raise DomainError("NOT_FOUND", "Only this bot's own actions can be reverted with its key.", 404)
        bot_access.check_command(db, bot.owner_id, change.entity_kind + ".update", {})
    else:
        bot_access.check_command(db, bot.owner_id, tool, arguments)
    conv = conversation(db, bot)
    job = models.Job(
        id=identity,
        owner_id=bot.owner_id,
        kind="external_command",
        status="succeeded",
        payload={},
        result={},
        finished_at=models.now(),
    )
    db.add(job)
    db.flush()
    row = models.AgentWork(
        id=identity,
        owner_id=bot.owner_id,
        account_id=bot.account_id,
        credential_id=bot.id,
        device_id=bot.id,
        conversation_id=conv.id,
        input_hash=fingerprint,
        input_ciphertext=seal({"message": tool}),
        expires_at=models.now() + timedelta(hours=24),
        result={},
    )
    db.add(row)
    db.flush()
    if tool == "action.revert":
        from .action_history import revert

        result = revert(db, bot.owner_id, bot.account_id, arguments["action_id"], identity + ":0")
    else:
        result = execute(db, bot.owner_id, identity + ":0", tool, arguments)
    row.result = {"message": "Saved.", "receipt_ids": [result["command_id"]]}
    job.result = {"work_id": identity, "status": "succeeded"}
    emit(db, bot.owner_id, "work.changed", identity, 1)
    db.flush()
    return direct_result(db, bot, row)


def direct_result(db, bot, row):
    receipt_ids = row.result.get("receipt_ids", [])
    receipt = db.get(models.Command, (bot.owner_id, receipt_ids[0])) if receipt_ids else None
    if not receipt:
        raise DomainError("REVISION_CONFLICT", "That request ID belongs to queued work.", 409)
    return scrub({**receipt.result, "request_id": row.id, "activity": public_work(db, row)}, bot.scopes)


def submit(db, bot, body):
    from .agent_work import enqueue

    identity = request_identity(bot, body.request_id)
    advisory(db, "work:" + identity)
    bot_access.authorize(db, bot.owner_id, "work:run", write=True)
    conv = conversation(db, bot, body.thread_id)
    row = enqueue(
        db,
        bot.owner_id,
        bot.account_id,
        bot.id,
        conv.id,
        request_identity(bot, body.request_id),
        body.message,
        credential_id=bot.id,
    )
    return public_work(db, row)


def get_work(db, bot, identity):
    row = db.get(models.AgentWork, str(identity))
    if not row or row.owner_id != bot.owner_id or row.credential_id != bot.id:
        raise DomainError("NOT_FOUND", "That request is not available to this bot.", 404)
    return row


def reply(db, bot, identity, body):
    from .agent_work import revise
    from .domain import check_revision

    advisory(db, "work:" + str(identity))
    bot_access.authorize(db, bot.owner_id, "work:run", write=True)
    row = get_work(db, bot, identity)
    if db.get(models.Job, row.id).kind != "agent_action":
        raise DomainError(
            "INVALID_ARGUMENT", "Edit the saved record instead of replying to a direct command."
        )
    receipt_key = "bot-reply:" + str(body.request_id)
    fingerprint = hashlib.sha256(
        json.dumps([str(identity), body.message, body.expected_revision]).encode()
    ).hexdigest()
    receipts = dict(row.result.get("reply_receipts", {}))
    if receipt_key in receipts:
        if receipts[receipt_key] != fingerprint:
            raise DomainError("REVISION_CONFLICT", "That reply ID belongs to different instructions.", 409)
        return public_work(db, row)
    check_revision(row, body.expected_revision)
    result = revise(db, row, body.message, continue_work=True)
    receipts[receipt_key] = fingerprint
    row.result = {**row.result, "reply_receipts": receipts}
    return result


def changes(db, bot, after=0, limit=100):
    allowed = [kind for kind in KINDS if scope_for(kind) in bot.scopes]
    events = models.Event
    high = db.scalar(select(func.max(events.id)).where(events.owner_id == bot.owner_id)) or 0
    if after > high:
        raise DomainError(
            "CURSOR_INVALID",
            "The change cursor is ahead of this workspace. Start a fresh synchronization.",
            409,
        )
    names = [kind + ".changed" for kind in allowed]
    if "organization:read" in bot.scopes:
        names.append("organization.changed")
    rows = list(
        db.scalars(
            select(events)
            .where(
                events.owner_id == bot.owner_id, events.id > after, events.id <= high, events.kind.in_(names)
            )
            .order_by(events.id)
            .limit(limit + 1)
        )
    )
    more = len(rows) > limit
    items = []
    for event in rows[:limit]:
        kind = event.kind.split(".")[0]
        if kind == "organization":
            # Legacy hierarchy-move events carry the affected record's ID.
            kind = next((k for k in allowed if db.get(KINDS[k], event.entity_id)), "organization")
        model = KINDS.get(kind)
        record = db.get(model, event.entity_id) if model else None
        items.append(
            {
                "cursor": event.id,
                "kind": kind,
                "entity_id": event.entity_id,
                "revision": event.revision,
                "changed_at": event.created_at.isoformat(),
                "deleted": record is None if model else False,
                "record": record_data(db, record, bot.scopes)
                if record and record.owner_id == bot.owner_id
                else None,
            }
        )
    return {
        "items": items,
        "current_cursor": high,
        "next_cursor": rows[limit - 1].id if more else high,
        "has_more": more,
    }


def backend_registry(row):
    from .tool_catalog import GROUPS
    from .tools import registry

    with session_scope() as db:
        bot = bot_access.authorize(db, row.owner_id, "work:run", credential_id=row.credential_id)
        definitions = [item for item in registry() if bot_access.tool_allowed(item["name"], bot.scopes)]
    names = {item["name"] for item in definitions}
    groups = {k: v for k, v in GROUPS.items() if any(name in names for name in v[1])}
    loader = copy.deepcopy(next(item for item in registry() if item["name"] == "tools_load"))
    loader["description"] = "Load permitted planner tools. Groups: " + ", ".join(groups)
    loader["parameters"]["properties"]["groups"]["items"]["enum"] = list(groups)
    return [loader, *definitions]


def backend_read(name, arguments, conversation_id):
    from .tools import READ_TOOLS

    try:
        jsonschema.validate(
            arguments, READ_TOOLS[name]["parameters"], format_checker=jsonschema.FormatChecker()
        )
    except jsonschema.ValidationError:
        raise DomainError("INVALID_ARGUMENT", "Check the read-tool arguments.") from None
    with session_scope() as db:
        bot = bot_access.check_tool(db, None, name)
        if name == "time_resolve":
            from .time_tools import resolve_time

            return resolve_time(arguments["local"], arguments["timezone"])
        if name in {"task_get", "note_read"}:
            kind = "task" if name == "task_get" else "note"
            return get_record(db, bot, kind, arguments[kind + "_id"])
        if name == "task_list":
            from .task_tools import list_tasks

            return list_tasks(db, bot.owner_id, arguments)
        if name == "task_resolve":
            from .task_context import resolve

            return resolve(
                db, bot.owner_id, None, conversation_id, arguments["scope"], arguments.get("query", "")
            )
        if name == "organization_list":
            from .productivity import snapshot

            return scrub(snapshot(db, bot.owner_id), bot.scopes)
        if name == "project_list":
            return {"projects": search(db, bot, "project", Search(limit=100))["items"]}
        if name == "note_search":
            from .notes import list_notes

            return scrub(
                list_notes(
                    db,
                    bot.owner_id,
                    arguments.get("query", ""),
                    arguments.get("project_id"),
                    arguments.get("task_id"),
                    offset=arguments.get("offset", 0),
                    archived=arguments.get("archived", False),
                    space_id=arguments.get("space_id"),
                    area_id=arguments.get("area_id"),
                    goal_id=arguments.get("goal_id"),
                ),
                bot.scopes,
            )
        raise DomainError("INVALID_ARGUMENT", "Unknown external read tool.")
