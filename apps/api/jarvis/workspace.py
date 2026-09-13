"""Read-only calendar projection over tasks and durable reminder schedules."""

from datetime import UTC, date, datetime, time, timedelta

from sqlalchemy import select

from .domain import DomainError, next_occurrence, parse_when, zone
from .models import Notification, Occurrence, Schedule, Task


def calendar(db, owner, start: date, end: date, timezone: str):
    if not 0 < (end - start).days <= 62:
        raise DomainError("INVALID_ARGUMENT", "Choose a calendar range of 1 to 62 days.")
    local = zone(timezone)
    begin = datetime.combine(start, time.min, local).astimezone(UTC)
    until = datetime.combine(end, time.min, local).astimezone(UTC)
    tasks = list(db.scalars(select(Task).where(Task.owner_id == owner, Task.archived.is_(False))))
    task_map = {t.id: t for t in tasks}
    events = []
    for task in tasks:
        if not task.due_date or task.is_template:
            continue
        if task.occurrence_id and not task.due_time:
            occurrence = db.get(Occurrence, task.occurrence_id)
            routine = db.get(Schedule, occurrence.schedule_id) if occurrence else None
            if routine and task.due_date == occurrence.scheduled_at.astimezone(zone(routine.timezone)).date():
                continue
        instant = (
            parse_when(f"{task.due_date.isoformat()}T{task.due_time}", task.due_timezone or timezone)
            if task.due_time
            else None
        )
        day = instant.astimezone(local).date() if instant else task.due_date
        if not start <= day < end:
            continue
        events.append(
            {
                "id": "task:" + task.id,
                "entity_id": task.id,
                "kind": "task",
                "title": task.title,
                "date": day.isoformat(),
                "at": instant.isoformat() if instant else None,
                "status": task.status,
                "project_id": task.project_id,
                "task_id": task.id,
                "revision": task.revision,
                "projected": False,
                "notification_id": None,
            }
        )
    schedules = list(db.scalars(select(Schedule).where(Schedule.owner_id == owner)))
    truncated = False
    for schedule in schedules:
        linked = task_map.get(schedule.task_id)
        if schedule.task_id and linked is None:
            continue
        project_id = linked.project_id if linked else schedule.project_id
        seen = set()

        def add(
            instant,
            status,
            projected,
            notification_id=None,
            occurrence_task_id=None,
            *,
            seen=seen,
            schedule=schedule,
            project_id=project_id,
        ):
            key = instant.isoformat()
            if key in seen or not begin <= instant < until:
                return
            seen.add(key)
            events.append(
                {
                    "id": f"schedule:{schedule.id}:{key}",
                    "entity_id": schedule.id,
                    "kind": "routine" if schedule.kind == "recurring_task" else "reminder",
                    "title": schedule.title,
                    "date": instant.astimezone(local).date().isoformat(),
                    "at": key,
                    "status": status,
                    "project_id": project_id,
                    "task_id": occurrence_task_id or schedule.task_id,
                    "revision": schedule.revision,
                    "projected": projected,
                    "notification_id": notification_id,
                }
            )

        rows = db.execute(
            select(Occurrence, Notification)
            .outerjoin(Notification, Notification.occurrence_id == Occurrence.id)
            .where(
                Occurrence.schedule_id == schedule.id,
                Occurrence.scheduled_at >= begin,
                Occurrence.scheduled_at < until,
                Occurrence.status.notin_(["suppressed", "superseded"]),
            )
        ).all()
        for occurrence, notice in rows:
            if occurrence.revision != schedule.revision and occurrence.status == "pending":
                continue
            add(
                occurrence.scheduled_at,
                "completed" if notice and notice.completed_at else occurrence.status,
                False,
                notice.id if notice else None,
                notice.task_id if notice else None,
            )
        if (
            schedule.status == "active"
            and schedule.next_run_at
            and not (linked and linked.status in {"completed", "cancelled"})
        ):
            candidate = schedule.next_run_at
            if candidate < begin and schedule.recurrence:
                candidate = next_occurrence(schedule, begin - timedelta(microseconds=1))
            while candidate and candidate < until:
                add(candidate, "active", True)
                if len(events) >= 2000:
                    truncated = True
                    break
                candidate = next_occurrence(schedule, candidate)
        elif schedule.status != "active" and not rows and not schedule.recurrence and not linked:
            add(schedule.anchor_at, schedule.status, False)
        if truncated:
            break
    from .google_calendar import connection_status
    from .google_projection import project

    warnings = []
    google_events, incomplete = project(db, owner, start, end, timezone, warnings)
    from .models import PlanningEntry
    from .planning import project as local_project

    mirrored = {
        (r.google_calendar_id, r.google_event_id)
        for r in db.scalars(
            select(PlanningEntry).where(
                PlanningEntry.owner_id == owner, PlanningEntry.google_calendar_id.is_not(None)
            )
        )
    }
    google_events = [e for e in google_events if (e["calendar_id"], e["provider_id"]) not in mirrored]
    local_events = local_project(db, owner, start, end, timezone)
    # A deadline is an instant, not an assumed-duration booking.
    for event in events:
        if event["kind"] == "task" and event["at"]:
            at = datetime.fromisoformat(event["at"])
            event["conflicts"] = list(
                dict.fromkeys(
                    g["title"]
                    for g in google_events + local_events
                    if g["busy"]
                    and datetime.fromisoformat(g["busy_start"]) <= at < datetime.fromisoformat(g["end_at"])
                )
            )
    events.extend(google_events + local_events)
    truncated = truncated or incomplete or len(events) > 2000
    events.sort(key=lambda e: (e["date"], e["at"] or "", e["title"], e["id"]))
    return {
        "items": events[:2000],
        "start": start.isoformat(),
        "end": end.isoformat(),
        "timezone": timezone,
        "truncated": truncated,
        "warnings": warnings,
        "google": connection_status(db, owner),
    }
