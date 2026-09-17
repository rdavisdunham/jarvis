from datetime import UTC, datetime
from sqlalchemy import select, func
from jarvis import notices
from jarvis.db import session_scope
from jarvis.domain import preferences
from jarvis.models import Task, Notification, Schedule, now
from test_structure import run, OWNER


def task(**kw):
    return run("task.create", {"title": "Deadline", **kw})


def test_only_future_timed_deadlines_get_one_notice():
    row = task(due_date="2027-01-20", due_time="09:00", due_timezone="America/Chicago")
    task(due_date="2027-01-20")
    task(due_date="2020-01-01", due_time="09:00")
    with session_scope() as db:
        notices.sync_deadlines(db, OWNER)
        db.flush()
        notices.sync_deadlines(db, OWNER)
        all = list(db.scalars(select(Notification)))
        assert len(all) == 1
        assert all[0].task_id == row["id"] and all[0].scheduled_at == datetime(2027, 1, 20, 15, tzinfo=UTC)
        assert not notices.eligible(db, all[0])


def test_deadline_edits_update_and_completion_cancels():
    row = task(due_date="2027-01-20", due_time="09:00")
    with session_scope() as db:
        notices.sync_deadlines(db, OWNER)
    row = run(
        "task.update", {"task_id": row["id"], "expected_revision": row["revision"], "due_time": "10:00"}
    )
    with session_scope() as db:
        notices.sync_deadlines(db, OWNER)
        n = db.scalar(select(Notification))
        assert n.generation == 2
    run("task.complete", {"task_id": row["id"], "expected_revision": row["revision"]})
    with session_scope() as db:
        notices.sync_deadlines(db, OWNER)
        assert db.scalar(select(Notification)).completed_at is not None


def test_snooze_reuses_notice_and_does_not_touch_task_or_schedule():
    row = task()
    with session_scope() as db:
        n = Notification(owner_id=OWNER, title="Test", task_id=row["id"], scheduled_at=now())
        db.add(n)
        db.flush()
        identity = n.id
    result = run("notification.snooze", {"notification_id": identity, "minutes": 60})
    with session_scope() as db:
        assert db.scalar(select(func.count(Schedule.id))) == 0
        assert db.scalar(select(func.count(Notification.id))) == 1
        assert db.get(Task, row["id"]).revision == 1
        assert db.get(Notification, identity).generation == 2
        assert not notices.eligible(db, db.get(Notification, identity))
    assert result["read_at"] is None


def test_quiet_hours_urgent_and_daytime_windows():
    with session_scope() as db:
        p = preferences(db, OWNER)
    night = datetime(2026, 9, 18, 4, tzinfo=UTC)
    assert notices.quiet_until(p, night) == datetime(2026, 9, 18, 13, tzinfo=UTC)
    assert notices.quiet_until(p, night, True) == night
    p.update(quiet_start="09:00", quiet_end="17:00")
    daytime = datetime(2026, 9, 18, 15, tzinfo=UTC)
    assert notices.quiet_until(p, daytime) == datetime(2026, 9, 18, 22, tzinfo=UTC)


def test_deadline_override_priority_not_urgency():
    task(due_date="2027-01-20", due_time="09:00", deadline_alert="off")
    task(due_date="2027-01-20", due_time="10:00", priority=3)
    with session_scope() as db:
        notices.sync_deadlines(db, OWNER)
        rows = list(db.scalars(select(Notification)))
        assert len(rows) == 1
        assert rows[0].importance == "normal"


def test_non_task_notice_cannot_complete_work():
    import pytest
    from jarvis.domain import DomainError

    with session_scope() as db:
        n = Notification(owner_id=OWNER, title="Review", category="morning", scheduled_at=now())
        db.add(n)
        db.flush()
        identity = n.id
    with pytest.raises(DomainError):
        run("notification.complete", {"notification_id": identity})


def test_deadline_does_not_duplicate_same_explicit_alert():
    row=task(due_date="2027-01-20",due_time="09:00",due_timezone="America/Chicago")
    run("schedule.create",{"title":"Deadline", "task_id":row["id"],"when":"2027-01-20T09:00","timezone":"America/Chicago"})
    with session_scope() as db:
        notices.sync_deadlines(db,OWNER);n=db.scalar(select(Notification))
        assert not notices.eligible(db,n,datetime(2027,1,20,15,tzinfo=UTC))


def test_future_deadline_reopens_after_completion():
    row=task(due_date="2027-01-20",due_time="09:00")
    with session_scope() as db:notices.sync_deadlines(db,OWNER)
    row=run("task.complete",{"task_id":row["id"],"expected_revision":row["revision"]})
    with session_scope() as db:notices.sync_deadlines(db,OWNER)
    run("task.reopen",{"task_id":row["id"],"expected_revision":row["revision"]})
    with session_scope() as db:
        notices.sync_deadlines(db,OWNER);n=db.scalar(select(Notification));assert n.dismissed_at is None and n.completed_at is None


def test_morning_summary_skips_newly_completed_tasks():
    row=task(due_date="2026-09-17")
    run("settings.update",{"morning_summary":True})
    instant=datetime(2026,9,17,13,tzinfo=UTC)
    with session_scope() as db:notices.morning(db,OWNER,instant)
    run("task.complete",{"task_id":row["id"],"expected_revision":row["revision"]})
    with session_scope() as db:assert not notices.eligible(db,db.scalar(select(Notification)),instant)
