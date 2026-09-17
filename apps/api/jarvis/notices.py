"""Deadline alerts, quiet hours and category-aware delivery over durable notifications."""

from datetime import UTC, timedelta
from zoneinfo import ZoneInfo
from sqlalchemy import select
from .domain import DomainError, advisory, parse_when, preferences
from .models import (
    AgentWork,
    Delivery,
    Job,
    Notification,
    Occurrence,
    OwnerSettings,
    Schedule,
    SharedWorkspace,
    Task,
    now,
)


def deadline(task, prefs):
    if (
        task.archived
        or task.is_template
        or task.status in {"completed", "cancelled"}
        or not task.due_date
        or not task.due_time
    ):
        return None
    if task.deadline_alert == "off" or (task.deadline_alert == "default" and not prefs["deadline_alerts"]):
        return None
    try:
        return parse_when(
            task.due_date.isoformat() + "T" + task.due_time, task.due_timezone or prefs["timezone"]
        )
    except DomainError:
        return None


def sync_deadlines(db, owner, instant=None):
    instant = instant or now()
    prefs = preferences(db, owner)
    if db.get(SharedWorkspace, owner):
        return
    for task in db.scalars(select(Task).where(Task.owner_id == owner)):
        key = "deadline:" + task.id
        notice = db.scalar(select(Notification).where(Notification.dedup_key == key))
        due = deadline(task, prefs)
        if not due:
            if notice and not notice.target.get("inactive"):
                notice.dismissed_at = instant
                notice.generation += 1
                notice.target = {**notice.target, "inactive": True}
            continue
        if not notice:
            # Never backfill overdue pushes. Future deadlines are recorded exactly once.
            if due <= instant:
                continue
            notice = Notification(
                owner_id=owner,
                dedup_key=key,
                category="deadline",
                title=task.title,
                body="Task deadline",
                task_id=task.id,
                scheduled_at=due,
                eligible_at=due,
                target={"view": "tasks", "task_id": task.id},
                importance="urgent" if task.alert_urgent else "normal",
            )
            db.add(notice)
        elif notice.scheduled_at != due or notice.target.get("inactive"):
            notice.scheduled_at = due
            notice.eligible_at = due
            notice.generation += 1
            notice.read_at = None
            notice.completed_at = None
            notice.dismissed_at = instant if due <= instant else None
            notice.target = {k: v for k, v in notice.target.items() if k != "inactive"}
        notice.title = task.title
        notice.importance = "urgent" if task.alert_urgent else "normal"
        # An explicit alert at this instant is the one notification for that task.
        explicit = db.scalar(
            select(Notification.id).where(
                Notification.task_id == task.id,
                Notification.category == "reminder",
                Notification.scheduled_at == due,
            )
        )
        if explicit:
            notice.dismissed_at = notice.dismissed_at or instant


def snooze(db, item, args):
    instant = now()
    prefs = preferences(db, item.owner_id)
    until = (
        parse_when(args.until, prefs["timezone"]) if args.until else instant + timedelta(minutes=args.minutes)
    )
    if until <= instant or until > instant + timedelta(days=30):
        raise DomainError("INVALID_ARGUMENT", "Choose a snooze time within the next 30 days.")
    item.eligible_at = until
    item.generation += 1
    item.dismissed_at = None
    item.read_at = None
    # The same notification is delivered again. Deadlines and recurrence never change.


def quiet_until(prefs, instant, urgent=False):
    if urgent or not prefs["quiet_enabled"] or prefs["quiet_start"] == prefs["quiet_end"]:
        return instant
    local = instant.astimezone(ZoneInfo(prefs["timezone"]))
    clock = local.strftime("%H:%M")
    start, end = prefs["quiet_start"], prefs["quiet_end"]
    quiet = (clock >= start or clock < end) if start > end else start <= clock < end
    if not quiet:
        return instant
    hour, minute = map(int, end.split(":"))
    end_at = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if end_at <= local:
        end_at += timedelta(days=1)
    from dateutil import tz

    return tz.resolve_imaginary(end_at).astimezone(UTC)


def eligible(db, n, instant=None):
    instant = instant or now()
    if n.read_at or n.dismissed_at or n.completed_at:
        return False
    if n.scheduled_at > instant or (n.eligible_at and n.eligible_at > instant):
        return False
    if (n.eligible_at or n.created_at) < instant - timedelta(days=1):
        return False
    if db.get(SharedWorkspace, n.owner_id):
        return False
    prefs = preferences(db, n.owner_id)
    if quiet_until(prefs, instant, n.importance == "urgent") > instant:
        return False
    if n.task_id:
        task = db.get(Task, n.task_id)
        if not task or task.archived or task.status in {"completed", "cancelled"}:
            return False
        if n.category == "deadline":
            if deadline(task, prefs) != n.scheduled_at:
                return False
            explicit = db.scalar(
                select(Schedule.id)
                .where(
                    Schedule.task_id == task.id,
                    Schedule.status.in_(["active", "finished"]),
                    ((Schedule.anchor_at == n.scheduled_at) | (Schedule.next_run_at == n.scheduled_at)),
                )
                .limit(1)
            )
            occurrence = db.scalar(
                select(Occurrence.id)
                .join(Schedule)
                .where(
                    Schedule.task_id == task.id,
                    Occurrence.scheduled_at == n.scheduled_at,
                    Occurrence.status.not_in(["cancelled", "expired"]),
                )
                .limit(1)
            )
            if explicit or occurrence:
                return False
    if n.category == "morning":
        task_ids = []
        for identity in n.target.get("task_ids", []):
            task = db.get(Task, identity)
            if task and not task.archived and task.status not in {"completed", "cancelled"}:
                task_ids.append(identity)
        if not task_ids and not n.target.get("work_ids"):
            return False
        n.body = f"{len(task_ids)} planned, due or overdue tasks" + (
            " · overnight updates" if n.target.get("work_ids") else ""
        )
    work_ids = n.target.get("work_ids", []) if n.target else []
    if work_ids:
        unseen = []
        for identity in work_ids:
            work = db.get(AgentWork, identity)
            job = db.get(Job, identity)
            if (
                work
                and job
                and not work.seen_at
                and not work.result.get("archived_at")
                and (n.category != "question" or job.status == "needs_input")
            ):
                unseen.append(identity)
        if not unseen and not (n.category == "morning" and task_ids):
            return False
    return True


def work_finished(db, row, status, message):
    if (
        db.get(SharedWorkspace, row.owner_id)
        or row.credential_id
        or row.result.get("quiet")
        or row.result.get("archived_at")
    ):
        return
    instant = now()
    category = (
        "question"
        if status == "needs_input"
        else "failure"
        if status in {"failed", "partial"}
        else "work_result"
        if status == "succeeded"
        else None
    )
    if not category:
        return
    # Rooted follow-ups supersede their earlier question notifications.
    root = row.result.get("continuation_root") or row.id
    for n in db.scalars(
        select(Notification).where(
            Notification.owner_id == row.owner_id,
            Notification.category == "question",
            Notification.dismissed_at.is_(None),
        )
    ):
        if root in n.target.get("work_ids", []) or row.id in n.target.get("work_ids", []):
            n.dismissed_at = instant
    key = "work:" + row.id + ":" + str(row.revision) + ":" + status
    if db.scalar(select(Notification.id).where(Notification.dedup_key == key)):
        return
    # Only coalesce results still waiting in their initial 30-second collection window.
    group = (
        db.scalar(
            select(Notification).where(
                Notification.owner_id == row.owner_id,
                Notification.category == "work_result",
                Notification.eligible_at > instant,
                Notification.created_at > instant - timedelta(seconds=30),
                Notification.read_at.is_(None),
            )
        )
        if category == "work_result"
        else None
    )
    if group:
        ids = list(dict.fromkeys([*group.target.get("work_ids", []), row.id]))
        group.target = {"view": "activity", "work_ids": ids}
        group.title = f"{len(ids)} requests finished"
        group.body = "Your saved changes are ready to review."
        return
    db.add(
        Notification(
            owner_id=row.owner_id,
            dedup_key=key,
            category=category,
            title="Eri needs your answer"
            if category == "question"
            else "Eri could not finish"
            if category == "failure"
            else "Eri finished your request",
            body=message[:2000],
            scheduled_at=instant,
            eligible_at=instant + timedelta(seconds=30 if category == "work_result" else 0),
            target={"view": "activity", "work_ids": [row.id]},
        )
    )


def morning(db, owner, instant=None):
    instant = instant or now()
    prefs = preferences(db, owner)
    if not prefs["morning_summary"] or db.get(SharedWorkspace, owner):
        return
    local = instant.astimezone(ZoneInfo(prefs["timezone"]))
    if local.hour != prefs["morning_hour"]:
        return
    key = "morning:" + owner + ":" + local.date().isoformat()
    if db.scalar(select(Notification.id).where(Notification.dedup_key == key)):
        return
    tasks = list(
        db.scalars(
            select(Task).where(
                Task.owner_id == owner,
                Task.archived.is_(False),
                Task.is_template.is_(False),
                Task.status.not_in(["completed", "cancelled"]),
            )
        )
    )
    tasks = [
        t for t in tasks if (t.due_date and t.due_date <= local.date()) or t.planned_date == local.date()
    ]
    held = list(
        db.scalars(
            select(Notification).where(
                Notification.owner_id == owner,
                Notification.scheduled_at <= instant,
                Notification.created_at > instant - timedelta(days=1),
                Notification.read_at.is_(None),
                Notification.dismissed_at.is_(None),
                Notification.completed_at.is_(None),
                Notification.category.in_(["deadline", "reminder", "work_result"]),
            )
        )
    )
    held = [
        n
        for n in held
        if not db.scalar(
            select(Delivery.id).where(
                Delivery.notification_id == n.id,
                Delivery.status == "submitted",
                Delivery.generation == n.generation,
            )
        )
        and (not n.task_id or n.task_id in {t.id for t in tasks})
        and (not n.eligible_at or n.eligible_at <= instant)
    ]
    if not tasks and not held:
        return
    for n in held:
        n.read_at = instant
    db.add(
        Notification(
            owner_id=owner,
            dedup_key=key,
            category="morning",
            title="Your day",
            body=f"{len(tasks)} planned, due or overdue tasks"
            + (f" · {len(held)} overnight updates" if held else ""),
            scheduled_at=instant,
            eligible_at=instant,
            target={
                "view": "tasks",
                "tab": "today",
                "task_ids": [t.id for t in tasks],
                "work_ids": list({identity for n in held for identity in n.target.get("work_ids", [])}),
            },
        )
    )


def scan(db):
    advisory(db, "notices:scan")
    owners = set(db.scalars(select(Task.owner_id).distinct())) | set(
        db.scalars(select(OwnerSettings.owner_id))
    )
    for owner in owners:
        sync_deadlines(db, owner)
        morning(db, owner)
