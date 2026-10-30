"""Deadline alerts, quiet hours, priorities, bundles and digests over durable notifications.

A notice's delivery, read, snoozed and completed states are separate facts on the row; `state` derives
the one state shown, and `apply` is the only place they change (the state machine). `plan` decides, for
the notices due now, which push alone, which ride together in one bundle, and which wait for the
owner's digest.
"""

import hashlib
from datetime import UTC, time, timedelta
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

# Deadline and explicit reminder for the same task within this window are one alert.
SAME_ALERT = timedelta(seconds=60)
# Priority levels, lowest first. Only "urgent" bypasses quiet hours; `notify_min_priority` sets which
# levels interrupt (push on their own or in a bundle); the rest wait for the digest.
LEVELS = ("low", "normal", "high", "urgent")
# Push vehicles for other notices. They are never items of their own on the Notifications page.
VEHICLES = {"bundle", "digest"}
# This many interrupting notices due in one pass become one bundled push.
BUNDLE_AT = 2
DIGEST_HOURS = {"hourly": 1, "every_3_hours": 3}
STATES = ("scheduled", "pending", "delivered", "read", "snoozed", "dismissed", "completed")
# action -> the states it may start from. Reading, dismissing or completing a snoozed notice ends the
# snooze; snoozing a dismissed notice brings it back (as before); a completed notice stays completed.
ALLOWED = {
    "deliver": {"pending"},
    "read": set(STATES),
    "dismiss": set(STATES),
    "complete": set(STATES),
    "snooze": set(STATES) - {"completed"},
}


def rank(level):
    return LEVELS.index(level) if level in LEVELS else 1


def task_importance(task):
    """A task's alerts: urgent only when its alert is marked urgent; priority High is high, Low is low."""
    if not task:
        return "normal"
    return "urgent" if task.alert_urgent else "high" if task.priority >= 3 else "low" if task.priority == 1 else "normal"


def state(n, instant=None):
    instant = instant or now()
    if n.completed_at:
        return "completed"
    if n.dismissed_at:
        return "dismissed"
    if n.snoozed_until and n.snoozed_until > instant:
        return "snoozed"
    if n.read_at:
        return "read"
    if n.delivered_at:
        return "delivered"
    if n.scheduled_at > instant or (n.eligible_at and n.eligible_at > instant):
        return "scheduled"
    return "pending"


def apply(n, action, instant=None, until=None):
    """Move one notice through the state machine; returns its new state."""
    instant = instant or now()
    if state(n, instant) not in ALLOWED[action]:
        raise DomainError(
            "INVALID_ARGUMENT",
            "This reminder is already completed." if action == "snooze" else "Only a due notice can be delivered.",
        )
    if action == "deliver":
        n.delivered_at = instant
    elif action == "snooze":
        # The same notification is delivered again. Deadlines and recurrence never change.
        n.snoozed_until = n.eligible_at = until
        n.generation += 1
        n.dismissed_at = n.read_at = n.delivered_at = n.bundle_id = None
    else:
        n.snoozed_until = None
        n.read_at = n.read_at or instant
        if action == "dismiss":
            n.dismissed_at = instant
        elif action == "complete":
            n.completed_at = n.completed_at or instant
    return state(n, instant)


def carried(db, n):
    return list(db.scalars(select(Notification).where(Notification.bundle_id == n.id)))


def spent(db, n, instant=None):
    """A bundle or digest with nothing left to deliver: everything it carried was seen, snoozed or done."""
    instant = instant or now()
    return n.category in VEHICLES and not any(
        state(m, instant) in {"pending", "delivered"} for m in carried(db, n)
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
                importance=task_importance(task),
            )
            db.add(notice)
        elif notice.scheduled_at != due or notice.target.get("inactive"):
            notice.scheduled_at = due
            notice.eligible_at = due
            notice.generation += 1
            notice.read_at = notice.delivered_at = notice.snoozed_until = notice.bundle_id = None
            notice.completed_at = None
            notice.dismissed_at = instant if due <= instant else None
            notice.target = {k: v for k, v in notice.target.items() if k != "inactive"}
        notice.title = task.title
        notice.importance = task_importance(task)
        # An explicit alert at (about) this instant is the one notification for that task.
        explicit = db.scalar(
            select(Notification.id)
            .where(
                Notification.task_id == task.id,
                Notification.category == "reminder",
                Notification.scheduled_at.between(due - SAME_ALERT, due + SAME_ALERT),
            )
            .limit(1)
        )
        if explicit:
            notice.dismissed_at = notice.dismissed_at or instant


def work_end(prefs, instant):
    """'after work' / 'end of day': today's work hours end, else 17:00."""
    local = instant.astimezone(ZoneInfo(prefs["timezone"]))
    for window in prefs.get("work_windows") or []:
        if local.weekday() in window.get("days", []):
            hour, minute = map(int, window["end"].split(":"))
            return time(hour, minute)
    return time(17)


def snooze(db, item, args):
    from .snooze_words import snooze_time

    instant = now()
    prefs = preferences(db, item.owner_id)
    until = (
        snooze_time(args.until, prefs["timezone"], instant, work_end(prefs, instant))
        if args.until
        else instant + timedelta(minutes=args.minutes)
    )
    if until <= instant or until > instant + timedelta(days=30):
        raise DomainError("INVALID_ARGUMENT", "Choose a snooze time within the next 30 days.")
    if item.category in VEHICLES:
        # Snoozing a bundle snoozes what it carried; the bundle itself is spent.
        for member in carried(db, item):
            if state(member, instant) in {"pending", "delivered"}:
                apply(member, "snooze", instant, until)
        apply(item, "read", instant)
        return
    apply(item, "snooze", instant, until)


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
    if n.category == "work_result" or n.read_at or n.dismissed_at or n.completed_at:
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
    if n.category in VEHICLES:
        # A bundle or digest goes out while something it carries is still waiting to be seen.
        members = [
            m for m in carried(db, n) if state(m, instant) in {"pending", "delivered"} and eligible(db, m, instant)
        ]
        if not members:
            return False
        n.title, n.body = bundle_text(n.category, members)
        return True
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
                    (Schedule.anchor_at.between(n.scheduled_at - SAME_ALERT, n.scheduled_at + SAME_ALERT))
                    | (Schedule.next_run_at.between(n.scheduled_at - SAME_ALERT, n.scheduled_at + SAME_ALERT)),
                )
                .limit(1)
            )
            occurrence = db.scalar(
                select(Occurrence.id)
                .join(Schedule)
                .where(
                    Schedule.task_id == task.id,
                    Occurrence.scheduled_at.between(n.scheduled_at - SAME_ALERT, n.scheduled_at + SAME_ALERT),
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
        held = n.target.get("held_count", 0)
        if not task_ids and not n.target.get("work_ids") and not held:
            return False
        n.body = f"{len(task_ids)} planned, due or overdue tasks" + (
            f" · {held} overnight updates" if held else " · overnight updates" if n.target.get("work_ids") else ""
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
        if not unseen and not (n.category == "morning" and (task_ids or n.target.get("held_count"))):
            return False
    return True


def work_finished(db, row, status, message):
    if (
        db.get(SharedWorkspace, row.owner_id)
        or row.credential_id
        or row.result.get("archived_at")
    ):
        return
    instant = now()
    category = (
        "question"
        if status == "needs_input"
        else "failure"
        if status in {"failed", "partial"}
        else None
    )
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
    # Success belongs in chat and Activity. Still retire the answered question.
    if not category or row.result.get("quiet"):
        return
    key = "work:" + row.id + ":" + str(row.revision) + ":" + status
    if db.scalar(select(Notification.id).where(Notification.dedup_key == key)):
        return
    db.add(
        Notification(
            owner_id=row.owner_id,
            dedup_key=key,
            category=category,
            title="Eri needs your answer"
            if category == "question"
            else "Eri could not finish",
            body=message[:2000],
            scheduled_at=instant,
            eligible_at=instant,
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
                Notification.delivered_at.is_(None),
                Notification.category.in_(["deadline", "reminder"]),
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
                # Held reminders are marked read above; the summary itself must stay deliverable.
                "held_count": len(held),
            },
        )
    )


def bundle_text(kind, members):
    noun = "reminder" if {m.category for m in members} <= {"reminder", "deadline"} else "update"
    title = f"{len(members)} {noun}{'' if len(members) == 1 else 's'}"
    return ("Digest: " + title if kind == "digest" else title), " · ".join(m.title for m in members)[:1000]


def digest_slot(prefs, instant):
    """The digest period `instant` falls in, or None when no digest may go out now."""
    cadence = prefs["digest_cadence"]
    local = instant.astimezone(ZoneInfo(prefs["timezone"]))
    if cadence == "daily":
        return local.date().isoformat() if local.hour >= prefs["morning_hour"] else None
    if cadence not in DIGEST_HOURS:
        return None
    step = DIGEST_HOURS[cadence]
    return f"{local.date().isoformat()}T{local.hour // step * step:02d}"


def bundle(db, owner, kind, members, instant, key=None):
    """One push that carries several notices. Each keeps its own row, state and destination."""
    key = key or "bundle:" + hashlib.sha256(
        ",".join(sorted(f"{m.id}:{m.generation}" for m in members)).encode()
    ).hexdigest()
    if db.scalar(select(Notification.id).where(Notification.dedup_key == key)):
        return None
    title, body = bundle_text(kind, members)
    row = Notification(
        owner_id=owner,
        dedup_key=key,
        category=kind,
        title=title,
        body=body,
        importance=max((m.importance for m in members), key=rank) if kind == "bundle" else "normal",
        scheduled_at=instant,
        eligible_at=instant,
        target={"view": "notifications", "notice_ids": [m.id for m in members]},
    )
    db.add(row)
    db.flush()
    for m in members:
        m.bundle_id = row.id
    return row


def collapse(notices):
    """One push per task: a second due notice for the same task rides with the first (no duplicate nudge)."""
    kept, by_task = [], {}
    for n in sorted(notices, key=lambda n: (rank(n.importance), n.scheduled_at), reverse=True):
        if n.task_id and n.task_id in by_task:
            n.bundle_id = by_task[n.task_id].id
            continue
        if n.task_id:
            by_task[n.task_id] = n
        kept.append(n)
    return kept


def plan(db, due, accepted, instant=None):
    """Of the eligible notices `due` now, return the notices to push: alone, a bundle, or a digest.

    `accepted(n)` says every active device of the owner may receive n (a new device gets no backlog,
    so such a notice is never folded into a push that would reach it)."""
    instant = instant or now()
    pushes, owners = [], {}
    for n in due:
        owners.setdefault(n.owner_id, []).append(n)
    for owner, notices in owners.items():
        prefs = preferences(db, owner)
        fresh, held = [], []
        for n in notices:
            if n.bundle_id:
                continue  # Its bundle (or the same task's other notice) carries this generation.
            started = db.scalar(
                select(Delivery.id).where(Delivery.notification_id == n.id, Delivery.generation == n.generation)
            )
            if started or n.category in VEHICLES | {"morning"} or n.importance == "urgent" or not accepted(n):
                pushes.append(n)
            elif rank(n.importance) >= rank(prefs["notify_min_priority"]):
                fresh.append(n)
            else:
                held.append(n)
        fresh = collapse(fresh)
        if len(fresh) >= BUNDLE_AT:
            pushes.append(bundle(db, owner, "bundle", fresh, instant))
        else:
            pushes.extend(fresh)
        slot = digest_slot(prefs, instant) if held else None
        if slot and (row := bundle(db, owner, "digest", collapse(held), instant, f"digest:{owner}:{slot}")):
            pushes.append(row)
    return [p for p in pushes if p]


def delivered(db, n, generation, instant=None):
    """A push for `n` at `generation` was accepted: it and whatever it carried are now delivered."""
    instant = instant or now()
    if n.generation != generation:
        return
    rows = [n]
    while rows:
        row = rows.pop()
        if state(row, instant) == "pending":
            apply(row, "deliver", instant)
        rows.extend(carried(db, row))  # A bundle's members, and the same-task notices each carries.


def scan(db):
    advisory(db, "notices:scan")
    owners = set(db.scalars(select(Task.owner_id).distinct())) | set(
        db.scalars(select(OwnerSettings.owner_id))
    )
    for owner in owners:
        sync_deadlines(db, owner)
        morning(db, owner)
