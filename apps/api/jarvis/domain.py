"""Transport-neutral commands. Every effect and receipt share a database transaction."""

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from typing import Literal
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
    title: str = Field(min_length=1, max_length=500)
    notes: str = Field(default="", max_length=20000)
    project: str | None = Field(default=None, max_length=200)
    due_date: date | None = None
    priority: int = Field(default=0, ge=0, le=3)


class TaskUpdate(Args):
    task_id: str
    expected_revision: int = Field(ge=1)
    title: str | None = Field(default=None, min_length=1, max_length=500)
    notes: str | None = Field(default=None, max_length=20000)
    project: str | None = Field(default=None, max_length=200)
    due_date: date | None = None
    priority: int | None = Field(default=None, ge=0, le=3)
    status: str | None = None
    archived: bool | None = None


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
    notification_id: str
    minutes: int = Field(default=10, ge=1, le=10080)


class SettingsUpdate(Args):
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


def advisory(db, key):
    db.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"), {"key": key})


def preferences(db, owner):
    settings = get_settings()
    row = db.get(OwnerSettings, owner)
    return {
        "preferred_name": settings.owner_name,
        "history_enabled": True,
        "memory_learning": True,
        "deep_sleep_enabled": True,
        "history_days": settings.history_days,
        "timezone": settings.timezone,
        "default_reminder_hour": settings.default_reminder_hour,
        "monthly_budget_usd": settings.monthly_budget_usd,
        "detailed_notifications": False,
        **(row.values if row else {}),
    }


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
            "INVALID_ARGUMENT", "Please check the action details.", data=exc.errors(include_url=False)
        )
    if tool.startswith("memory."):
        advisory(db, f"memory:{owner}")
    data = mutate(db, owner, tool, args, command_id)
    result = {
        "command_id": command_id,
        "status": "succeeded",
        "data": data,
        "committed_at": now().isoformat(),
    }
    db.add(Command(owner_id=owner, id=command_id, request_hash=request_hash, result=result))
    db.flush()
    return result


def mutate(db, owner, tool, args, command_id):
    if tool == "task.create":
        task = Task(owner_id=owner, **args.model_dump())
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
        for key, value in changes.items():
            setattr(task, key, value)
        if "status" in changes:
            task.completed_at = (task.completed_at or now()) if task.status == "completed" else None
        task.revision += 1
        task.updated_at = now()
        emit(db, owner, "task.changed", task.id, task.revision)
        return serial(task)
    if tool == "schedule.create":
        if args.kind not in {"reminder", "recurring_task"}:
            raise DomainError("INVALID_ARGUMENT", "Choose a reminder or recurring task.")
        if args.task_id:
            owned(db, Task, args.task_id, owner, lock=True)
        if args.kind == "recurring_task" and (args.task_id or not args.recurrence):
            raise DomainError(
                "INVALID_ARGUMENT", "A recurring task needs a recurrence and creates its own tasks."
            )
        p = preferences(db, owner)
        instant = parse_when(args.when, args.timezone, p["default_reminder_hour"])
        row = Schedule(
            owner_id=owner,
            title=args.title,
            timezone=args.timezone,
            anchor_at=instant,
            next_run_at=instant,
            task_id=args.task_id,
            kind=args.kind,
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
        if tool == "schedule.complete":
            if row.recurrence:
                raise DomainError(
                    "INVALID_ARGUMENT",
                    "Complete a delivered occurrence with notification.complete; the routine will keep running.",
                )
            row.status, row.next_run_at, row.completed_at = "completed", None, now()
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
        else:
            if not args.when:
                raise DomainError("INVALID_ARGUMENT", "A new time is required.")
            instant = parse_when(args.when, row.timezone, preferences(db, owner)["default_reminder_hour"])
            row.anchor_at, row.status, row.completed_at = instant, "active", None
            row.next_run_at = first_occurrence(row)
        row.revision += 1
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
            instant = now() + timedelta(minutes=args.minutes)
            row = Schedule(
                owner_id=owner,
                title=item.title,
                timezone=preferences(db, owner)["timezone"],
                anchor_at=instant,
                next_run_at=instant,
                task_id=item.task_id,
            )
            db.add(row)
            item.dismissed_at = now()
        elif tool == "notification.dismiss":
            item.dismissed_at = now()
        item.read_at = now()
        emit(db, owner, "notification.changed", item.id)
        return serial(item)
    if tool == "settings.update":
        advisory(db, f"budget:{owner}")
        values = args.model_dump(exclude_none=True)
        if "timezone" in values:
            zone(values["timezone"])
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
        task = Task(
            owner_id=job.owner_id,
            title=schedule.title,
            occurrence_id=occurrence.id,
            due_date=occurrence.scheduled_at.astimezone(zone(schedule.timezone)).date(),
        )
        db.add(task)
        db.flush()
        emit(db, job.owner_id, "task.changed", task.id, task.revision)
    notification = Notification(
        owner_id=job.owner_id,
        occurrence_id=occurrence.id,
        title=schedule.title,
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
