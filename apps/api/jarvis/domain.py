"""Transport-neutral commands. Every effect and receipt share a database transaction."""

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dateutil import rrule, tz
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select, text

from .config import get_settings
from .models import (
    Command,
    Event,
    Job,
    Memory,
    Notification,
    Occurrence,
    Outbox,
    OwnerSettings,
    Project,
    Schedule,
    Source,
    Task,
    now,
)


class DomainError(Exception):
    def __init__(self, code, message, status=400, data=None):
        self.code, self.message, self.status, self.data = code, message, status, data
        super().__init__(message)


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class TaskCreate(Args):
    deadline_alert: Literal["default", "on", "off"] = "default"
    alert_urgent: bool = False
    status: Literal["open", "backlog"] = "open"
    title: str = Field(min_length=1, max_length=500)
    notes: str = Field(default="", max_length=20000)
    project: str | None = Field(default=None, max_length=200)
    project_id: str | None = None
    parent_task_id: str | None = None
    assignee: str = Field(default="owner", min_length=1, max_length=100)
    work_type: str = Field(default="", max_length=80)
    tags: list[Annotated[str, Field(max_length=40)]] = Field(default_factory=list, max_length=20)
    space_id: str | None = None
    area_id: str | None = None
    assignee_id: str | None = None
    planned_date: date | None = None
    estimate_minutes: int | None = Field(default=None, ge=1, le=100000)
    due_date: date | None = None
    due_time: str | None = Field(
        default=None, pattern=r"^([01]\d|2[0-3]):[0-5]\d(?:[+-](?:[01]\d|2[0-3]):[0-5]\d)?$"
    )
    due_timezone: str | None = Field(default=None, max_length=100)
    priority: int = Field(default=0, ge=0, le=3)


class TaskChanges(Args):
    deadline_alert: Literal["default", "on", "off"] = "default"
    alert_urgent: bool = False
    title: str | None = Field(default=None, min_length=1, max_length=500)
    notes: str | None = Field(default=None, max_length=20000)
    project: str | None = Field(default=None, max_length=200)
    project_id: str | None = None
    parent_task_id: str | None = None
    assignee: str = Field(default="owner", min_length=1, max_length=100)
    work_type: str = Field(default="", max_length=80)
    tags: list[Annotated[str, Field(max_length=40)]] = Field(default_factory=list, max_length=20)
    space_id: str | None = None
    area_id: str | None = None
    assignee_id: str | None = None
    planned_date: date | None = None
    estimate_minutes: int | None = Field(default=None, ge=1, le=100000)
    due_date: date | None = None
    due_time: str | None = Field(
        default=None, pattern=r"^([01]\d|2[0-3]):[0-5]\d(?:[+-](?:[01]\d|2[0-3]):[0-5]\d)?$"
    )
    due_timezone: str | None = Field(default=None, max_length=100)
    priority: int | None = Field(default=None, ge=0, le=3)
    status: Literal["backlog", "open", "in_progress", "waiting", "deferred", "completed", "cancelled"] | None = None
    archived: bool | None = None


class TaskUpdate(TaskChanges):
    task_id: str
    expected_revision: int = Field(ge=1)


class TaskBatch(Args):
    items: list[TaskUpdate] = Field(min_length=1, max_length=1000)


class TaskSelectionUpdate(Args):
    selection_id: str
    changes: TaskChanges


class TaskState(Args):
    task_id: str
    expected_revision: int = Field(ge=1)


class ScheduleCreate(Args):
    title: str = Field(min_length=1, max_length=500)
    when: str = Field(max_length=100)
    timezone: str = "America/Chicago"
    recurrence: str | None = Field(default=None, max_length=250)
    task_id: str | None = None
    kind: str = "reminder"
    original_words: str = Field(default="", max_length=5000)
    project_id: str | None = None


from .productivity_schema import ProjectCreate, ProjectUpdate


class ScheduleUpdate(Args):
    schedule_id: str
    expected_revision: int = Field(ge=1)
    title: str | None = Field(default=None, min_length=1, max_length=500)
    when: str | None = Field(default=None, max_length=100)
    timezone: str | None = Field(default=None, max_length=100)
    recurrence: str | None = Field(default=None, max_length=250)
    task_id: str | None = None
    project_id: str | None = None


class ScheduleChange(Args):
    schedule_id: str
    expected_revision: int = Field(ge=1)
    when: str | None = None


class Capture(Args):
    content: str = Field(min_length=1, max_length=20000)


class Correct(Args):
    memory_id: str
    content: str = Field(min_length=1, max_length=20000)


class Forget(Args):
    memory_id: str
    delete_source: bool = False


class ResolveMemory(Args):
    review_id: str
    expected_revision: int = Field(ge=1)
    action: Literal["merge", "distinct", "defer"]
    content: str | None = Field(default=None, min_length=1, max_length=20000)


class NotificationAction(Args):
    until: str | None = Field(default=None, max_length=64)
    notification_id: str
    minutes: int = Field(default=10, ge=1, le=10080)


class WorkWindow(Args):
    days: list[int] = Field(default_factory=lambda: [0,1,2,3,4], max_length=7)
    start: str = Field(default="08:00", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    end: str = Field(default="17:00", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")


class SettingsUpdate(Args):
    routing_mode: Literal["off", "suggest", "automatic"] | None = None
    routing_learning: bool | None = None
    routing_review_enabled: bool | None = None
    routing_review_day: int | None = Field(default=None, ge=0, le=6)
    routing_review_hour: int | None = Field(default=None, ge=0, le=23)
    work_windows: list[WorkWindow] | None = Field(default=None, max_length=14)
    deadline_alerts: bool | None = None
    quiet_enabled: bool | None = None
    quiet_start: str | None = Field(default=None, pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    quiet_end: str | None = Field(default=None, pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    morning_summary: bool | None = None
    morning_hour: int | None = Field(default=None, ge=0, le=23)

    agent_profile: Literal["openai", "luna", "gemini", "groq"] | None = None
    agent_provider: Literal["openai", "gemini", "groq"] | None = None
    preferred_name: str | None = Field(default=None, min_length=1, max_length=80)
    history_enabled: bool | None = None
    memory_learning: bool | None = None
    deep_sleep_enabled: bool | None = None
    history_days: int | None = Field(default=None, ge=0, le=36500)
    timezone: str | None = None
    default_reminder_hour: int | None = Field(default=None, ge=0, le=23)
    monthly_budget_usd: float | None = Field(default=None, ge=0, le=10000)
    detailed_notifications: bool | None = None


COMMANDS = {
    "project.create": ProjectCreate,
    "project.update": ProjectUpdate,
    "schedule.update": ScheduleUpdate,
    "task.batch": TaskBatch,
    "task.selection_update": TaskSelectionUpdate,
    "task.create": TaskCreate,
    "task.update": TaskUpdate,
    "task.complete": TaskState,
    "task.reopen": TaskState,
    "schedule.create": ScheduleCreate,
    "schedule.cancel": ScheduleChange,
    "schedule.complete": ScheduleChange,
    "schedule.reschedule": ScheduleChange,
    "memory.capture": Capture,
    "memory.correct": Correct,
    "memory.forget": Forget,
    "memory.resolve": ResolveMemory,
    "notification.read": NotificationAction,
    "notification.complete": NotificationAction,
    "notification.snooze": NotificationAction,
    "notification.dismiss": NotificationAction,
    "settings.update": SettingsUpdate,
}


def serial(record):
    return jsonable_encoder(
        {
            c.name: getattr(record, c.name)
            for c in record.__table__.columns
            if c.name not in {"owner_id", "token_hash", "embedding", "fingerprint"}
        }
    )


def owned(db, model, record_id, owner, lock=False):
    q = select(model).where(model.id == record_id, model.owner_id == owner)
    if lock:
        q = q.with_for_update()
    obj = db.scalar(q)
    if obj is None:
        raise DomainError("NOT_FOUND", "That record is no longer available.", 404)
    return obj


def emit(db, owner, kind, entity, revision=None):
    # Assign event IDs only while holding the owner's commit-order lock.
    advisory(db, f"events:{owner}")
    db.add(Event(owner_id=owner, kind=kind, entity_id=entity, revision=revision))
    if kind in {"record.changed", "structure.changed", "task.changed", "note.changed", "project.changed", "goal.changed", "space.changed", "area.changed", "actor.changed"}:
        from .search_index import queue_index
        queue_index(db, owner)


def advisory(db, key):
    db.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"), {"key": key})


def preferences(db, owner):
    from .agent_models import default_provider, selected

    settings = get_settings()
    row = db.get(OwnerSettings, owner)
    values = {
        "agent_provider": default_provider(),
        "preferred_name": settings.owner_name,
        "history_enabled": True,
        "memory_learning": True,
        "deep_sleep_enabled": True,
        "routing_mode": "automatic", "routing_learning": True, "routing_review_enabled": True,
        "routing_review_day": 0, "routing_review_hour": 3,
        "work_windows": [{"days": [0,1,2,3,4], "start": "08:00", "end": "17:00"}],
        "deadline_alerts": True, "quiet_enabled": True, "quiet_start": "22:00", "quiet_end": "08:00",
        "morning_summary": False, "morning_hour": 8,
        "history_days": settings.history_days,
        "timezone": settings.timezone,
        "default_reminder_hour": settings.default_reminder_hour,
        "monthly_budget_usd": settings.monthly_budget_usd,
        "detailed_notifications": False,
        **(row.values if row else {}),
    }
    agent = selected(values)
    from .access import shared_preferences
    return shared_preferences(db, owner, {**values, "agent_profile": agent.profile_id, "agent_provider": agent.provider})


def zone(name):
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        raise DomainError("INVALID_ARGUMENT", "Choose a valid IANA time zone, such as America/Chicago.")


def parse_when(value, zone_name, default_hour=10):
    location = zone(zone_name)
    try:
        parsed = datetime.fromisoformat(value)
        if len(value) == 10:
            parsed = parsed.replace(hour=default_hour)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=location)
            if not tz.datetime_exists(parsed):
                raise DomainError(
                    "INVALID_ARGUMENT", "That local time does not exist because clocks move forward."
                )
            if tz.datetime_ambiguous(parsed):
                raise DomainError("AMBIGUOUS_REFERENCE", "That time occurs twice. Specify its UTC offset.")
        return parsed.astimezone(UTC)
    except ValueError:
        raise DomainError("INVALID_ARGUMENT", "Supply a date or ISO date/time for the reminder.")


def validate_recurrence(value):
    if not value:
        return None
    value = value.upper().removeprefix("RRULE:")
    try:
        parts = dict(p.split("=", 1) for p in value.split(";"))
        if parts.get("FREQ") not in {"DAILY", "WEEKLY", "MONTHLY"}:
            raise ValueError()
        if set(parts) - {"FREQ", "INTERVAL", "BYDAY", "BYMONTHDAY"}:
            raise ValueError()
        if not 1 <= int(parts.get("INTERVAL", "1")) <= 365:
            raise ValueError()
        if "BYDAY" in parts and (
            parts["FREQ"] != "WEEKLY"
            or any(d not in {"MO", "TU", "WE", "TH", "FR", "SA", "SU"} for d in parts["BYDAY"].split(","))
        ):
            raise ValueError()
        if "BYMONTHDAY" in parts and (
            parts["FREQ"] != "MONTHLY" or int(parts["BYMONTHDAY"]) not in [*range(1, 32), -1]
        ):
            raise ValueError()
        rrule.rrulestr(value)
    except (ValueError, TypeError):
        raise DomainError("INVALID_ARGUMENT", "Use daily, weekly/selected weekdays, or monthly recurrence.")
    return value


def next_occurrence(schedule, after):
    if not schedule.recurrence:
        return None
    local = zone(schedule.timezone)
    rule = rrule.rrulestr(schedule.recurrence, dtstart=schedule.anchor_at.astimezone(local))
    candidate = rule.after(after.astimezone(local), inc=False)
    for _ in range(10):
        if candidate is None:
            return None
        resolved = candidate.replace(fold=0).astimezone(UTC)
        if tz.datetime_exists(candidate) and resolved > after:
            return resolved
        candidate = rule.after(candidate, inc=False)
    raise DomainError("INVALID_ARGUMENT", "Recurrence could not resolve a valid local time.")


def first_occurrence(schedule):
    if not schedule.recurrence:
        return schedule.anchor_at
    local_anchor = schedule.anchor_at.astimezone(zone(schedule.timezone))
    first = next(iter(rrule.rrulestr(schedule.recurrence, dtstart=local_anchor)))
    # An explicit UTC offset chooses the requested fold for the first occurrence.
    if first.replace(tzinfo=None) == local_anchor.replace(tzinfo=None):
        return schedule.anchor_at
    return next_occurrence(schedule, schedule.anchor_at - timedelta(microseconds=1))


def check_revision(record, expected):
    if record.revision != expected:
        raise DomainError(
            "REVISION_CONFLICT",
            "This changed on another device. Review the latest version.",
            409,
            serial(record),
        )


def capture_source(db, owner, content, native_id, role="user", conversation=None, explicit=False):
    if role == "assistant":
        from .memory_review import record_question

        record_question(db, owner, content)
    if (
        conversation
        and not explicit
        and (conversation.private or not preferences(db, owner)["history_enabled"])
    ):
        return None
    existing = db.scalar(select(Source).where(Source.owner_id == owner, Source.native_id == native_id))
    if existing:
        return existing
    source = Source(
        owner_id=owner,
        content=content,
        native_id=native_id,
        role=role,
        conversation_id=conversation.id if conversation else None,
        kind="explicit" if explicit else "transcript",
        explicit=explicit,
    )
    db.add(source)
    db.flush()
    return source


def delete_source(db, source):
    """Keep tombstones and idempotency identifiers, remove stored source content."""
    advisory(db, f"memory:{source.owner_id}")
    source.deleted_at, source.content = now(), ""
    ids = set()
    for memory in db.scalars(select(Memory).where(Memory.source_id == source.id)):
        ids.add(memory.id)
        memory.suppressed, memory.content = True, ""
        memory.embedding, memory.evidence, memory.tags = None, "", []
    for receipt in db.scalars(select(Command).where(Command.owner_id == source.owner_id)):
        data = receipt.result.get("data", {})
        if data.get("id") in ids or data.get("source_id") == source.id:
            receipt.result = {**receipt.result, "data": {"id": data.get("id"), "deleted": True}}
    if source.native_id.startswith("chat:"):
        turn_id = source.native_id.split(":")[1]
        job = db.get(Job, turn_id)
        if job and job.result:
            job.result = {**job.result, "message": "This response was deleted.", "actions": []}


def enqueue_job(db, owner, kind, payload):
    job = Job(owner_id=owner, kind=kind, payload=payload)
    db.add(job)
    db.flush()
    db.add(Outbox(job_id=job.id))
    return job


def execute(db, owner, command_id, tool, arguments):
    from .access import command_access
    arguments=dict(arguments)
    command_access(db,owner,tool,arguments)
    if not command_id or len(command_id) > 100:
        raise DomainError("INVALID_ARGUMENT", "A stable command ID is required.")
    if tool not in COMMANDS:
        raise DomainError("INVALID_ARGUMENT", "That action is not available.")
    request_hash = hashlib.sha256(
        json.dumps({"tool": tool, "arguments": arguments}, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    advisory(db, f"command:{owner}:{command_id}")
    previous = db.get(Command, (owner, command_id))
    if previous:
        if previous.request_hash != request_hash:
            raise DomainError(
                "REVISION_CONFLICT", "This command ID already belongs to a different request.", 409
            )
        return previous.result
    try:
        args = COMMANDS[tool].model_validate(arguments)
    except ValidationError as exc:
        raise DomainError(
            "INVALID_ARGUMENT", "Please check the action details.", data=exc.errors(include_url=False, include_context=False)
        )
    if tool.startswith("memory."):
        advisory(db, f"memory:{owner}")
    if tool.startswith(
        ("task.", "project.", "schedule.", "notification.", "note.", "space.", "area.", "goal.", "actor.", "record.", "structure.", "routing.")
    ):
        # Serialize owner graph changes so two concurrent parent edits cannot create a cycle.
        advisory(db, f"workspace:{owner}")
    from .action_history import journal
    with journal(db, owner, command_id, tool, arguments):
        try:
            data = mutate(db, owner, tool, args, command_id)
        except ValidationError as exc:
            raise DomainError("INVALID_ARGUMENT","Check the fields for this record's behavior.",data=exc.errors(include_url=False,include_context=False)) from exc
        if tool.startswith(("task.", "note.")):
            from .structure import observe_core
            observe_core(db, owner, tool, data, command_id, arguments=arguments)
    from .work_coordination import record_saved_resources
    record_saved_resources(db, owner, data)
    result = {
        "command_id": command_id,
        "status": "succeeded",
        "data": data,
        "committed_at": now().isoformat(),
    }
    from .access import actor
    db.add(Command(owner_id=owner, id=command_id, request_hash=request_hash, result=result,
                   account_id=actor(db, owner)))
    db.flush()
    return result


def task_timing(db, owner, changes, task=None):
    values = {key: getattr(task, key, None) for key in ("due_date", "due_time", "due_timezone")}
    values.update(changes)
    if "due_date" in changes and changes["due_date"] is None:
        if changes.get("due_time"):
            raise DomainError("INVALID_ARGUMENT", "A due time needs a due date.")
        changes.update(due_time=None, due_timezone=None)
        return changes
    clock = values.get("due_time")
    if not clock:
        changes.update(due_time=None, due_timezone=None)
        return changes
    day = values.get("due_date")
    if not day:
        raise DomainError("INVALID_ARGUMENT", "Choose a due date before adding a due time.")
    location = values.get("due_timezone") or preferences(db, owner)["timezone"]
    instant = parse_when(day.isoformat() + "T" + clock, location)
    local = instant.astimezone(zone(location))
    if local.date() != day or local.strftime("%H:%M") != clock[:5]:
        raise DomainError(
            "INVALID_ARGUMENT", "The supplied UTC offset does not match that time zone and date."
        )
    changes.update(due_time=clock, due_timezone=location)
    return changes


def mutate(db, owner, tool, args, command_id):
    if tool.startswith("routing."):
        from .routing import mutate as routing_mutate
        return routing_mutate(db,owner,tool,args,command_id)
    from .organization import project_changes

    if tool.startswith(("structure.", "record.")):
        from .structure import mutate as mutate_structure
        return mutate_structure(db, owner, tool, args, command_id)
    if tool.startswith("linear."):
        from .linear_commands import mutate as linear_mutate

        return linear_mutate(db, owner, tool, args)
    if tool.startswith("planning."):
        from .planning import mutate as planning_mutate

        return planning_mutate(db, owner, tool, args)
    if tool in {"calendar.create", "calendar.update", "calendar.delete"}:
        from .google_writes import queue_write

        return queue_write(db, owner, tool, args)
    if tool == "calendar.select":
        from .google_calendar import select_calendar

        return select_calendar(db, owner, args)
    if tool.startswith("note."):
        from .notes import mutate_note

        return mutate_note(db, owner, tool, args)
    if tool == "task.selection_update":
        from .task_tools import apply_selection

        return apply_selection(db, owner, args, command_id)
    if tool == "task.batch":
        ids = [item.task_id for item in args.items]
        if len(set(ids)) != len(ids):
            raise DomainError("INVALID_ARGUMENT", "Choose each task only once.")
        for item in args.items:
            check_revision(owned(db, Task, item.task_id, owner, lock=True), item.expected_revision)
        tasks, applied_ids, unchanged_ids = [], [], []
        for item in args.items:
            current = owned(db, Task, item.task_id, owner)
            changes = item.model_dump(exclude_unset=True, exclude={"task_id", "expected_revision"})
            if all(getattr(current, key) == value for key, value in changes.items()):
                tasks.append(serial(current))
                unchanged_ids.append(current.id)
            else:
                tasks.append(mutate(db, owner, "task.update", item, command_id))
                applied_ids.append(current.id)
        return {
            "tasks": tasks,
            "requested_count": len(ids),
            "applied_count": len(applied_ids),
            "unchanged_count": len(unchanged_ids),
            "task_ids": ids,
            "applied_ids": applied_ids,
            "unchanged_ids": unchanged_ids,
        }
    if tool.startswith(("project.", "space.", "area.", "goal.", "actor.")):
        from .productivity import mutate as productivity_mutate

        return productivity_mutate(db, owner, tool, args)
    if tool == "task.create":
        task = Task(
            owner_id=owner,
            **project_changes(db, owner, task_timing(db, owner, args.model_dump(exclude_unset=True))),
        )
        db.add(task)
        db.flush()
        emit(db, owner, "task.changed", task.id, task.revision)
        return serial(task)
    if tool.startswith("task."):
        task = owned(db, Task, args.task_id, owner, lock=True)
        check_revision(task, args.expected_revision)
        changes = args.model_dump(exclude_unset=True, exclude={"task_id", "expected_revision"})
        if tool == "task.complete":
            changes = {"status": "completed"}
        elif tool == "task.reopen":
            changes = {"status": "open"}
        if "status" in changes and changes["status"] not in {
            "backlog",
            "open",
            "in_progress",
            "waiting",
            "deferred",
            "completed",
            "cancelled",
        }:
            raise DomainError("INVALID_ARGUMENT", "Unknown task status.")
        for key in ("title", "notes", "priority", "archived"):
            if key in changes and changes[key] is None:
                raise DomainError("INVALID_ARGUMENT", f"{key} cannot be empty.")
        if any(key in changes for key in ("due_date", "due_time", "due_timezone")):
            changes = task_timing(db, owner, changes, task)
        changes = project_changes(db, owner, changes, task)
        from .linear_sync import before_task_update

        before_task_update(db, owner, task, changes)
        for key, value in changes.items():
            setattr(task, key, value)
        if "status" in changes:
            task.completed_at = (task.completed_at or now()) if task.status == "completed" else None
        if task.status in {"completed", "cancelled"} or task.archived:
            from .task_alerts import finish_task

            finish_task(db, task, task.status, sync_external=False)
        else:
            task.revision += 1
            task.updated_at = now()
            emit(db, owner, "task.changed", task.id, task.revision)
        return serial(task)
    if tool == "schedule.create":
        if args.kind not in {"reminder", "recurring_task"}:
            raise DomainError("INVALID_ARGUMENT", "Choose a reminder or recurring task.")
        if args.project_id:
            owned(db, Project, args.project_id, owner)
        if args.task_id:
            owned(db, Task, args.task_id, owner, lock=True)
        if args.kind == "recurring_task" and (args.task_id or not args.recurrence):
            raise DomainError(
                "INVALID_ARGUMENT", "A recurring task needs a recurrence and creates its own tasks."
            )
        from .task_alerts import new_task

        linked = owned(db, Task, args.task_id, owner, lock=True) if args.task_id else None
        if linked and (linked.status in {"completed", "cancelled"} or linked.archived):
            raise DomainError("INVALID_ARGUMENT", "Reopen the task before adding another alert.")
        if not linked:
            linked = new_task(db, owner, args.title, args.project_id, template=bool(args.recurrence))
        p = preferences(db, owner)
        instant = parse_when(args.when, args.timezone, p["default_reminder_hour"])
        row = Schedule(
            owner_id=owner,
            title=args.title,
            timezone=args.timezone,
            anchor_at=instant,
            next_run_at=instant,
            task_id=linked.id,
            project_id=args.project_id,
            kind="recurring_task" if linked.is_template else "reminder",
            recurrence=validate_recurrence(args.recurrence),
            original_words=args.original_words,
        )
        if row.recurrence:
            row.next_run_at = first_occurrence(row)
        db.add(row)
        db.flush()
        emit(db, owner, "schedule.changed", row.id, row.revision)
        return serial(row)
    if tool.startswith("schedule."):
        row = owned(db, Schedule, args.schedule_id, owner, lock=True)
        check_revision(row, args.expected_revision)
        if tool == "schedule.update":
            changes = args.model_dump(exclude_unset=True, exclude={"schedule_id", "expected_revision"})
            for key in ("title", "when", "timezone"):
                if key in changes and changes[key] is None:
                    raise DomainError("INVALID_ARGUMENT", f"{key} cannot be null.")
            if row.status != "active" and any(k in changes for k in ("when", "timezone", "recurrence")):
                raise DomainError(
                    "INVALID_ARGUMENT", "Reschedule a finished reminder explicitly before editing its timing."
                )
            if changes.get("task_id"):
                owned(db, Task, changes["task_id"], owner)
            if changes.get("project_id"):
                owned(db, Project, changes["project_id"], owner)
            if "task_id" in changes and changes["task_id"] is None:
                raise DomainError(
                    "INVALID_ARGUMENT", "Alerts belong to a task. Choose a task instead of unlinking."
                )
            if row.kind == "recurring_task" and (
                changes.get("task_id", row.task_id) != row.task_id
                or not changes.get("recurrence", row.recurrence)
            ):
                raise DomainError(
                    "INVALID_ARGUMENT", "A recurring task needs a recurrence and creates its own tasks."
                )
            if "timezone" in changes:
                zone(changes["timezone"])
                if "when" not in changes:
                    raise DomainError("INVALID_ARGUMENT", "Supply the new local time when changing timezone.")
            if "recurrence" in changes:
                changes["recurrence"] = validate_recurrence(changes["recurrence"])
            timing = any(k in changes for k in ("when", "timezone", "recurrence"))
            when = changes.pop("when", None)
            for key, value in changes.items():
                setattr(row, key, value)
            if when:
                row.anchor_at = parse_when(
                    when, row.timezone, preferences(db, owner)["default_reminder_hour"]
                )
            if row.kind == "recurring_task" and row.task_id and "title" in changes:
                template = owned(db, Task, row.task_id, owner, lock=True)
                template.title = row.title
                template.revision += 1
                emit(db, owner, "task.changed", template.id, template.revision)
            if timing:
                row.next_run_at = first_occurrence(row)
                if not when and row.next_run_at < now() and row.recurrence:
                    row.next_run_at = next_occurrence(row, now())
        elif tool == "schedule.complete":
            if row.recurrence:
                raise DomainError(
                    "INVALID_ARGUMENT",
                    "Complete a delivered occurrence with notification.complete; the routine will keep running.",
                )
            row.status, row.next_run_at, row.completed_at = "completed", None, now()
            if row.task_id:
                from .task_alerts import finish_task

                finish_task(db, owned(db, Task, row.task_id, owner, lock=True))
            for notice in db.scalars(
                select(Notification)
                .join(Occurrence, Notification.occurrence_id == Occurrence.id)
                .where(Occurrence.schedule_id == row.id)
            ):
                notice.completed_at = notice.completed_at or row.completed_at
                notice.read_at = notice.read_at or row.completed_at
                emit(db, owner, "notification.changed", notice.id)
        elif tool == "schedule.cancel":
            row.status, row.next_run_at = "cancelled", None
            if row.task_id:
                template = owned(db, Task, row.task_id, owner, lock=True)
                if template.is_template:
                    from .task_alerts import finish_task

                    finish_task(db, template, "cancelled")
        else:
            if not args.when:
                raise DomainError("INVALID_ARGUMENT", "A new time is required.")
            instant = parse_when(args.when, row.timezone, preferences(db, owner)["default_reminder_hour"])
            row.anchor_at, row.status, row.completed_at = instant, "active", None
            if row.task_id:
                task = owned(db, Task, row.task_id, owner, lock=True)
                from .linear_sync import before_task_update

                before_task_update(db, owner, task, {"status": "open"})
                task.status, task.completed_at, task.archived = "open", None, False
                task.revision += 1
                emit(db, owner, "task.changed", task.id, task.revision)
            row.next_run_at = first_occurrence(row)
        previous_revision = row.revision
        row.revision += 1
        if tool == "schedule.update" and not timing:
            # Metadata edits must not invalidate a due occurrence already queued for delivery.
            for occurrence in db.scalars(
                select(Occurrence).where(
                    Occurrence.schedule_id == row.id,
                    Occurrence.revision == previous_revision,
                    Occurrence.status == "pending",
                )
            ):
                occurrence.revision = row.revision
        emit(db, owner, "schedule.changed", row.id, row.revision)
        return serial(row)
    if tool == "memory.resolve":
        from .memory_review import resolve

        return resolve(db, owner, args, command_id)
    if tool == "memory.capture":
        source = capture_source(db, owner, args.content, f"capture:{command_id}", explicit=True)
        memory = Memory(
            owner_id=owner,
            source_id=source.id,
            content=args.content,
            fingerprint=hashlib.sha256(" ".join(args.content.casefold().split()).encode()).hexdigest(),
        )
        db.add(memory)
        db.flush()
        enqueue_job(db, owner, "embed_memory", {"memory_id": memory.id})
        emit(db, owner, "memory.changed", memory.id)
        return serial(memory)
    if tool in {"memory.correct", "memory.forget"}:
        old = owned(db, Memory, args.memory_id, owner, lock=True)
        old.suppressed, old.embedding = True, None
        if tool == "memory.correct":
            source = capture_source(db, owner, args.content, f"correction:{command_id}", explicit=True)
            row = Memory(
                owner_id=owner,
                source_id=source.id,
                content=args.content,
                fingerprint=hashlib.sha256(" ".join(args.content.casefold().split()).encode()).hexdigest(),
                supersedes_id=old.id,
                fact_key=old.fact_key,
                revision=old.revision + 1,
            )
            db.add(row)
            db.flush()
            enqueue_job(db, owner, "embed_memory", {"memory_id": row.id})
            emit(db, owner, "memory.changed", row.id)
            return serial(row)
        old.content, old.evidence, old.tags = "", "", []
        if args.delete_source:
            source = owned(db, Source, old.source_id, owner, lock=True)
            delete_source(db, source)
        emit(db, owner, "memory.changed", old.id)
        return {"id": old.id, "forgotten": True}
    if tool.startswith("notification."):
        item = owned(db, Notification, args.notification_id, owner, lock=True)
        if tool == "notification.complete":
            if item.category not in {"reminder","deadline"} or not item.task_id:
                raise DomainError("INVALID_ARGUMENT","Open this notification to respond; it does not complete a task.")
            if item.task_id:
                from .task_alerts import finish_task

                finish_task(db, owned(db, Task, item.task_id, owner, lock=True))
            item.completed_at = item.completed_at or now()
            if item.occurrence_id:
                occurrence = db.get(Occurrence, item.occurrence_id)
                schedule = owned(db, Schedule, occurrence.schedule_id, owner, lock=True)
                occurrence.status = "completed"
                if not schedule.recurrence and schedule.status == "finished":
                    schedule.status, schedule.completed_at = "completed", item.completed_at
                    schedule.revision += 1
                    emit(db, owner, "schedule.changed", schedule.id, schedule.revision)
        elif tool == "notification.snooze":
            if item.completed_at:
                raise DomainError("INVALID_ARGUMENT", "This reminder is already completed.")
            if item.task_id:
                task = owned(db, Task, item.task_id, owner, lock=True)
                if task.status in {"completed", "cancelled"} or task.archived:
                    raise DomainError("INVALID_ARGUMENT", "Reopen this task before snoozing its alert.")
            from .notices import snooze
            snooze(db,item,args)
        elif tool == "notification.dismiss":
            item.dismissed_at = now()
        if tool != "notification.snooze": item.read_at = now()
        emit(db, owner, "notification.changed", item.id)
        return serial(item)
    if tool == "settings.update":
        advisory(db, f"budget:{owner}")
        values = args.model_dump(exclude_none=True)
        if "agent_profile" in values or "agent_provider" in values:
            from .agent_models import selected

            agent = selected(values, require_key=True)
            if (
                "agent_profile" in values
                and "agent_provider" in values
                and agent.provider != values["agent_provider"]
            ):
                raise DomainError("INVALID_ARGUMENT", "The selected model and provider do not match.")
            # Older clients can still select a provider; new clients select a model profile.
            values.update(agent_profile=agent.profile_id, agent_provider=agent.provider)
        if "timezone" in values:
            zone(values["timezone"])
        if "work_windows" in values and any(any(day<0 or day>6 for day in w["days"]) for w in values["work_windows"]):raise DomainError("INVALID_ARGUMENT","Work days must be Monday through Sunday.")
        row = db.get(OwnerSettings, owner)
        if row:
            row.values = {**row.values, **values}
        else:
            db.add(OwnerSettings(owner_id=owner, values=values))
        db.flush()
        emit(db, owner, "settings.changed", owner)
        return preferences(db, owner)
    raise DomainError("INVALID_ARGUMENT", "Unknown action.")


def scan_schedules(db, clock=None):
    clock = clock or db.scalar(select(__import__("sqlalchemy").func.now()))
    rows = db.scalars(
        select(Schedule)
        .where(Schedule.status == "active", Schedule.next_run_at <= clock)
        .order_by(Schedule.next_run_at)
        .limit(100)
        .with_for_update(skip_locked=True)
    ).all()
    for schedule in rows:
        due = schedule.next_run_at
        occurrence = Occurrence(schedule_id=schedule.id, revision=schedule.revision, scheduled_at=due)
        db.add(occurrence)
        db.flush()
        enqueue_job(db, schedule.owner_id, "reminder", {"occurrence_id": occurrence.id})
        schedule.next_run_at = next_occurrence(schedule, clock)
        if schedule.next_run_at is None:
            schedule.status = "finished"
    return len(rows)


def deliver_occurrence(db, job):
    advisory(db, f"workspace:{job.owner_id}")
    occurrence = db.get(Occurrence, job.payload["occurrence_id"])
    schedule = db.scalar(select(Schedule).where(Schedule.id == occurrence.schedule_id).with_for_update())
    if occurrence.status != "pending":
        return {"status": occurrence.status}
    if schedule.revision != occurrence.revision or schedule.status in {"cancelled", "completed"}:
        occurrence.status = "superseded"
        return {"status": "superseded"}
    task = owned(db, Task, schedule.task_id, job.owner_id, lock=True) if schedule.task_id else None
    if task and (task.status in {"completed", "cancelled"} or task.archived):
        occurrence.status = "suppressed"
        return {"status": "suppressed"}
    if schedule.kind == "recurring_task":
        template = task
        task = Task(
            owner_id=job.owner_id,
            title=template.title if template else schedule.title,
            notes=template.notes if template else "",
            space_id=template.space_id if template else None,
            area_id=template.area_id if template else None,
            assignee_id=template.assignee_id if template else None,
            estimate_minutes=template.estimate_minutes if template else None,
            assignee=template.assignee if template else "owner",
            priority=template.priority if template else 0,
            tags=template.tags if template else [],
            work_type=template.work_type if template else "",
            parent_task_id=template.id if template else None,
            occurrence_id=occurrence.id,
            project_id=template.project_id if template else schedule.project_id,
            project=template.project
            if template
            else (db.get(Project, schedule.project_id).name if schedule.project_id else None),
            due_date=occurrence.scheduled_at.astimezone(zone(schedule.timezone)).date(),
        )
        db.add(task)
        db.flush()
        emit(db, job.owner_id, "task.changed", task.id, task.revision)
    notification = Notification(
        owner_id=job.owner_id,
        occurrence_id=occurrence.id,
        title=task.title if task and schedule.kind == "recurring_task" else schedule.title,
        task_id=task.id if task else None,
        scheduled_at=occurrence.scheduled_at,
        body="Delivered after its scheduled time."
        if now() - occurrence.scheduled_at > timedelta(seconds=60)
        else "",
    )
    db.add(notification)
    db.flush()
    occurrence.status = "delivered"
    emit(db, job.owner_id, "notification.changed", notification.id)
    return {"notification_id": notification.id}


from .note_schema import NOTE_COMMANDS

COMMANDS.update(NOTE_COMMANDS)

from .google_schema import CalendarSelection

COMMANDS["calendar.select"] = CalendarSelection
from .google_schema import CalendarCreate, CalendarDelete, CalendarUpdate

COMMANDS.update(
    {"calendar.create": CalendarCreate, "calendar.update": CalendarUpdate, "calendar.delete": CalendarDelete}
)

from .planning_schema import PLANNING_COMMANDS

COMMANDS.update(PLANNING_COMMANDS)

from .linear_schema import LINEAR_COMMANDS

COMMANDS.update(LINEAR_COMMANDS)

from .productivity_schema import PRODUCTIVITY_COMMANDS

COMMANDS.update(PRODUCTIVITY_COMMANDS)

from .structure_schema import COMMANDS as STRUCTURE_COMMANDS
COMMANDS.update(STRUCTURE_COMMANDS)

from .routing_schema import COMMANDS as ROUTING_COMMANDS
COMMANDS.update(ROUTING_COMMANDS)
