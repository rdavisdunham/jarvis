"""First-party appointments and task work blocks, optionally mirrored to Google."""

from datetime import datetime, time, timedelta

from sqlalchemy import select

from .db import session_scope
from .domain import DomainError, advisory, check_revision, emit, enqueue_job, owned, serial, zone
from .google_auth import unseal
from .google_schema import CalendarCreate, CalendarRead, EventFields
from .google_writes import TERMINAL, account_source, event_body, queue_write, read_event
from .models import GoogleCalendarEvent, Job, PlanningEntry, Task, now


def data(db, row):
    result = serial(row)
    job = db.get(Job, row.google_job_id) if row.google_job_id else None
    result.pop("google_snapshot", None)
    if row.google_calendar_id:
        from .models import GoogleIdentity

        account = db.get(GoogleIdentity, row.owner_id)
        if not account or not account.calendar_enabled:
            result["google_state"] = "disconnected"
    if job:
        result["write_status"] = job.status
        result["write_message"] = (job.result or {}).get("message", "")
        if job.status in TERMINAL and job.status != "succeeded" and result["google_state"] == "pending":
            result["google_state"] = job.status
    return result


def mutable(db, row):
    job = db.get(Job, row.google_job_id) if row.google_job_id else None
    if job and job.status not in TERMINAL:
        raise DomainError("CALENDAR_PENDING", "Wait for the current Google change to finish.", 409)


def fields(args):
    result = {k: getattr(args, k) for k in EventFields.model_fields}
    event_body(EventFields(**result))
    return result


def bind_task(db, owner, task_id, kind):
    if kind == "block" and not task_id:
        raise DomainError("INVALID_ARGUMENT", "Choose the task this work block belongs to.")
    if task_id:
        task = owned(db, Task, task_id, owner)
        if task.archived or task.is_template:
            raise DomainError("INVALID_ARGUMENT", "Choose an active task occurrence for a work block.")


def queue_publication(db, row, operation, calendar_id=None):
    mutable(db, row)
    if operation == "create":
        receipt = queue_write(
            db, row.owner_id, "calendar.create", CalendarCreate(**row.fields, calendar_id=calendar_id)
        )
        job = db.get(Job, receipt["job_id"])
        row.google_calendar_id = calendar_id
        row.google_event_id = job.payload["provider_event"]
    else:
        account, source = account_source(db, row.owner_id, row.google_calendar_id, writing=True)
        if not row.google_snapshot or not row.google_snapshot.get("etag"):
            raise DomainError(
                "CALENDAR_CONFLICT", "Review the linked Google event before publishing changes.", 409
            )
        payload = {
            "operation": operation,
            "calendar_id": source.id,
            "provider_calendar": source.provider_id,
            "generation": account.generation,
            "provider_event": row.google_event_id,
            "etag": row.google_snapshot["etag"],
            "base_event": row.google_snapshot,
            "scope": "event",
        }
        if operation == "update":
            payload["body"] = event_body(EventFields(**row.fields))
        job = enqueue_job(db, row.owner_id, "google_write", payload)
    job.payload = {**job.payload, "planning_entry_id": row.id, "planning_revision": row.revision}
    row.google_job_id = job.id
    row.google_state = "pending"
    emit(db, row.owner_id, "google.write", job.id)


def mutate(db, owner, tool, args):
    advisory(db, f"workspace:{owner}")
    if tool == "planning.commit":
        from .planner import commit

        return commit(db, owner, args.plan_token)
    if tool == "planning.create":
        bind_task(db, owner, args.task_id, args.kind)
        row = PlanningEntry(owner_id=owner, kind=args.kind, task_id=args.task_id, fields=fields(args))
        db.add(row)
        db.flush()
        if args.google_calendar_id:
            queue_publication(db, row, "create", args.google_calendar_id)
    else:
        row = owned(db, PlanningEntry, args.entry_id, owner, lock=True)
        check_revision(row, args.expected_revision)
        mutable(db, row)
        if tool == "planning.update":
            if row.status != "active":
                raise DomainError("INVALID_ARGUMENT", "This calendar entry was cancelled.")
            bind_task(db, owner, args.task_id, row.kind)
            row.fields = fields(args)
            row.task_id = args.task_id
            row.revision += 1
            if row.google_calendar_id:
                if row.google_state in {"conflict", "missing", "disconnected"}:
                    raise DomainError(
                        "CALENDAR_CONFLICT",
                        "Resolve the Google difference or unlink the copy before editing.",
                        409,
                    )
                queue_publication(db, row, "update")
        elif tool == "planning.publish":
            if row.status != "active" or row.google_calendar_id:
                raise DomainError("INVALID_ARGUMENT", "Only an active, unpublished entry can be published.")
            row.revision += 1
            queue_publication(db, row, "create", args.calendar_id)
        elif tool == "planning.unlink":
            # Keep both records. Never recreate or delete the task because a remote copy disappears.
            row.google_calendar_id = row.google_event_id = row.google_snapshot = row.google_job_id = None
            row.google_state = "local"
            row.revision += 1
        elif tool == "planning.resolve":
            token = unseal(args.edit_token)
            if (
                token.get("purpose") != "calendar_edit"
                or token.get("owner") != owner
                or token.get("calendar_id") != row.google_calendar_id
                or token.get("provider_event") != row.google_event_id
                or datetime.fromisoformat(token["expires_at"]) <= now()
            ):
                raise DomainError("CALENDAR_STALE", "Refresh the Google comparison.", 409)
            account_source(db, owner, row.google_calendar_id, generation=token["generation"])
            row.google_snapshot = token["base_event"]
            row.revision += 1
            if args.choice == "google":
                row.fields = token["event_fields"]
                row.google_state = "synced"
                row.google_job_id = None
            else:
                queue_publication(db, row, "update")
        elif tool == "planning.delete":
            row.status = "cancelled"
            row.revision += 1
            if row.google_calendar_id:
                queue_publication(db, row, "delete")
    row.updated_at = now()
    emit(db, owner, "planning.changed", row.id, row.revision)
    return data(db, row)


def publication_completed(db, job, remote):
    entry_id = job.payload.get("planning_entry_id")
    if not entry_id:
        return
    row = db.get(PlanningEntry, entry_id)
    if row and row.google_job_id == job.id:
        row.google_snapshot = remote
        row.google_state = "synced" if job.payload["operation"] != "delete" else "deleted"
        emit(db, row.owner_id, "planning.changed", row.id, row.revision)


def reconcile(db, owner, calendar_id, events, full=False):
    incoming = {e["id"]: e for e in events}
    for row in db.scalars(
        select(PlanningEntry).where(
            PlanningEntry.owner_id == owner,
            PlanningEntry.google_calendar_id == calendar_id,
            PlanningEntry.status == "active",
        )
    ):
        job = db.get(Job, row.google_job_id) if row.google_job_id else None
        if job and job.status not in TERMINAL:
            continue
        remote = incoming.get(row.google_event_id)
        if not remote and not full:
            continue
        previous = row.google_state
        if not remote or remote.get("status") == "cancelled":
            row.google_state = "missing"
        elif row.google_snapshot and remote.get("etag") != row.google_snapshot.get("etag"):
            # Differences are reviewed; a Google edit never silently moves an Eridani deadline.
            row.google_state = "conflict"
        if previous != row.google_state:
            emit(db, owner, "planning.changed", row.id, row.revision)


def project(db, owner, start, end, timezone):
    from .google_projection import instant

    local = zone(timezone)
    begin, until = datetime.combine(start, time.min, local), datetime.combine(end, time.min, local)
    result = []
    for row in db.scalars(
        select(PlanningEntry).where(PlanningEntry.owner_id == owner, PlanningEntry.status == "active")
    ):
        payload = event_body(EventFields(**row.fields))
        point, all_day = instant(payload["start"], row.fields["timezone"])
        finish, _ = instant(payload["end"], row.fields["timezone"])
        if all_day:
            point = datetime.combine(point.date(), time.min, local)
            finish = datetime.combine(finish.date(), time.min, local)
        if finish <= begin or point >= until:
            continue
        task = db.get(Task, row.task_id) if row.task_id else None
        day = max(point.astimezone(local).date(), start)
        while day < end and day <= (finish.astimezone(local) - timedelta(microseconds=1)).date():
            result.append(
                {
                    "id": f"planning:{row.id}:{day}",
                    "entity_id": row.id,
                    "kind": row.kind,
                    "title": row.fields["title"],
                    "description": row.fields["description"],
                    "location": row.fields["location"],
                    "date": day.isoformat(),
                    "at": None if all_day else point.isoformat(),
                    "end_at": finish.isoformat(),
                    "busy_start": point.isoformat(),
                    "busy": row.fields["busy"],
                    "all_day": all_day,
                    "status": "active",
                    "project_id": task.project_id if task else None,
                    "task_id": row.task_id,
                    "revision": row.revision,
                    "projected": False,
                    "notification_id": None,
                    "calendar_title": "Eridani",
                    "google_state": row.google_state,
                }
            )
            day += timedelta(days=1)
    return result


def comparison(owner, entry_id):
    with session_scope() as db:
        row = owned(db, PlanningEntry, entry_id, owner)
        local = data(db, row)
        remote = db.scalar(
            select(GoogleCalendarEvent).where(
                GoogleCalendarEvent.calendar_id == row.google_calendar_id,
                GoogleCalendarEvent.provider_id == row.google_event_id,
            )
        )
        if not remote:
            raise DomainError(
                "NOT_FOUND",
                "The Google copy is unavailable. You can unlink it and keep your local entry.",
                404,
            )
        event_id = remote.id
    return {"local": local, "google": read_event(owner, CalendarRead(event_id=event_id))}


def busy_intervals(owner, begin, until):
    from .domain import preferences

    with session_scope() as db:
        timezone = preferences(db, owner)["timezone"]
        local = zone(timezone)
        rows = project(
            db,
            owner,
            begin.astimezone(local).date(),
            until.astimezone(local).date() + timedelta(days=1),
            timezone,
        )
    intervals = set()
    for entry in rows:
        if entry["busy"]:
            a, b = (
                max(begin, datetime.fromisoformat(entry["busy_start"])),
                min(until, datetime.fromisoformat(entry["end_at"])),
            )
            if a < b:
                intervals.add((a, b))
    return list(intervals)


def local_availability(owner, begin, until, minutes):
    merged = []
    for a, b in sorted(busy_intervals(owner, begin, until)):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    free, cursor = [], begin
    for a, b in merged + [(until, until)]:
        if a - cursor >= timedelta(minutes=minutes):
            free.append({"start": cursor.isoformat(), "end": a.isoformat()})
        cursor = max(cursor, b)
    return {
        "status": "fresh",
        "checked_at": now().isoformat(),
        "source": "eridani_only",
        "calendar_count": 0,
        "start": begin.isoformat(),
        "end": until.isoformat(),
        "free": free,
        "busy": [{"start": a.isoformat(), "end": b.isoformat()} for a, b in merged],
        "note": "Only Eridani appointments and work blocks are included. Google is not connected.",
    }
