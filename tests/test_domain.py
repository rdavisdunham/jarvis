from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from uuid import uuid4

import pytest
from jarvis import budget
from jarvis.db import session_scope
from jarvis.domain import (
    DomainError,
    capture_source,
    deliver_occurrence,
    execute,
    next_occurrence,
    parse_when,
    scan_schedules,
)
from jarvis.memory_service import search
from jarvis.models import (
    BudgetReservation,
    Conversation,
    Job,
    Notification,
    Occurrence,
    Outbox,
    Schedule,
    Source,
    Task,
    now,
)
from sqlalchemy import func, select


def run(tool, arguments, command_id=None, owner="davin"):
    with session_scope() as db:
        return execute(db, owner, command_id or str(uuid4()), tool, arguments)["data"]


def test_lost_reply_retry_and_distinct_intents():
    key = str(uuid4())
    first = run("task.create", {"title": "Buy milk"}, key)
    assert run("task.create", {"title": "Buy milk"}, key)["id"] == first["id"]
    assert run("task.create", {"title": "Buy milk"})["id"] != first["id"]
    with pytest.raises(DomainError, match="different request"):
        run("task.create", {"title": "Email Josh"}, key)


def test_concurrent_retries_commit_once():
    key = str(uuid4())
    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(lambda _: run("task.create", {"title": "One task"}, key)["id"], range(8)))
    assert len(set(ids)) == 1
    with session_scope() as db:
        assert db.scalar(select(func.count(Task.id))) == 1


def test_revision_conflict_and_owner_boundary():
    task = run("task.create", {"title": "Original"})
    changed = run("task.update", {"task_id": task["id"], "expected_revision": 1, "title": "Changed"})
    with pytest.raises(DomainError) as error:
        run("task.complete", {"task_id": task["id"], "expected_revision": 1})
    assert error.value.code == "REVISION_CONFLICT"
    with pytest.raises(DomainError) as error:
        run("task.complete", {"task_id": task["id"], "expected_revision": 2}, owner="stranger")
    assert error.value.code == "NOT_FOUND"
    assert changed["title"] == "Changed"


def test_completion_reopen_due_date_never_creates_reminder():
    task = run("task.create", {"title": "Finish report", "due_date": "2026-09-18"})
    done = run("task.complete", {"task_id": task["id"], "expected_revision": 1})
    assert done["status"] == "completed" and done["completed_at"]
    reopened = run("task.reopen", {"task_id": task["id"], "expected_revision": 2})
    assert reopened["status"] == "open" and reopened["completed_at"] is None
    with session_scope() as db:
        assert db.scalar(select(func.count(Schedule.id))) == 0


def test_invalid_fields_roll_back_without_receipt():
    with pytest.raises(DomainError):
        run("task.create", {"title": " ", "priority": 5})
    task = run("task.create", {"title": "Keep"})
    with pytest.raises(DomainError):
        run("task.update", {"task_id": task["id"], "expected_revision": 1, "title": None})
    with session_scope() as db:
        assert db.get(Task, task["id"]).title == "Keep"


def test_dst_one_off_and_recurrence():
    with pytest.raises(DomainError, match="does not exist"):
        parse_when("2026-03-08T02:30", "America/Chicago")
    with pytest.raises(DomainError, match="occurs twice"):
        parse_when("2026-11-01T01:30", "America/Chicago")
    assert parse_when("2026-09-18", "America/Chicago").hour == 15
    schedule = Schedule(
        timezone="America/Chicago",
        recurrence="FREQ=DAILY",
        anchor_at=parse_when("2026-03-07T02:30", "America/Chicago"),
    )
    assert next_occurrence(schedule, schedule.anchor_at).date().isoformat() == "2026-03-09"
    schedule.anchor_at = parse_when("2026-10-31T01:30", "America/Chicago")
    assert next_occurrence(schedule, schedule.anchor_at).isoformat() == "2026-11-01T06:30:00+00:00"


def test_monthly_31st_skips_short_month():
    row = Schedule(
        timezone="America/Chicago",
        recurrence="FREQ=MONTHLY;BYMONTHDAY=31",
        anchor_at=parse_when("2026-01-31T10:00", "America/Chicago"),
    )
    assert next_occurrence(row, row.anchor_at).date().isoformat() == "2026-03-31"


def test_outbox_atomic_rollback():
    run("schedule.create", {"title": "Due now", "when": (now() - timedelta(minutes=1)).isoformat()})
    with pytest.raises(RuntimeError), session_scope() as db:
        assert scan_schedules(db) == 1
        raise RuntimeError("Injected crash before acceptance")
    with session_scope() as db:
        assert db.scalar(select(func.count(Occurrence.id))) == 0
        assert db.scalar(select(func.count(Outbox.job_id))) == 0
        assert scan_schedules(db) == 1


def test_replay_delivers_once_and_cancellation_race():
    schedule = run(
        "schedule.create", {"title": "Email Josh", "when": (now() - timedelta(minutes=1)).isoformat()}
    )
    with session_scope() as db:
        assert scan_schedules(db) == 1
    run("schedule.cancel", {"schedule_id": schedule["id"], "expected_revision": 1})
    with session_scope() as db:
        job = db.scalar(select(Job))
        assert deliver_occurrence(db, job)["status"] == "superseded"
        assert db.scalar(select(func.count(Notification.id))) == 0


def test_late_repeat_coalesces_and_creates_independent_tasks():
    series = run(
        "schedule.create",
        {
            "title": "Take vitamins",
            "when": (now() - timedelta(days=4)).isoformat(),
            "recurrence": "FREQ=DAILY",
            "kind": "recurring_task",
        },
    )
    with session_scope() as db:
        assert scan_schedules(db) == 1
        db.flush()
        job = db.scalar(select(Job))
        deliver_occurrence(db, job)
        assert deliver_occurrence(db, job)["status"] == "delivered"
        task = db.scalar(select(Task))
        task_id = task.id
        assert db.scalar(select(func.count(Notification.id))) == 1
    run("task.complete", {"task_id": task_id, "expected_revision": 1})
    with session_scope() as db:
        row = db.get(Schedule, series["id"])
        assert row.status == "active"
        assert scan_schedules(db, clock=row.next_run_at) == 1
        db.flush()
        job = db.scalars(select(Job).order_by(Job.created_at.desc())).first()
        deliver_occurrence(db, job)
        assert db.scalar(select(func.count(Task.id))) == 2


def test_linked_reminder_suppressed_after_task_completion():
    task = run("task.create", {"title": "Done"})
    run(
        "schedule.create",
        {"title": "Reminder", "task_id": task["id"], "when": (now() - timedelta(minutes=1)).isoformat()},
    )
    run("task.complete", {"task_id": task["id"], "expected_revision": 1})
    with session_scope() as db:
        scan_schedules(db)
        db.flush()
        assert deliver_occurrence(db, db.scalar(select(Job)))["status"] == "suppressed"


def test_private_conversation_keeps_only_explicit_capture():
    with session_scope() as db:
        conv = Conversation(owner_id="davin", device_id=str(uuid4()), private=True)
        db.add(conv)
        db.flush()
        assert capture_source(db, "davin", "private words", "private-1", conversation=conv) is None
        assert db.scalar(select(func.count(Source.id))) == 0
    saved = run("memory.capture", {"content": "Explicitly remember this"})
    with session_scope() as db:
        assert db.get(Source, saved["source_id"]).explicit


def test_memory_correction_and_forgetting_filters_source():
    saved = run("memory.capture", {"content": "My favorite tea is green tea"})
    corrected = run("memory.correct", {"memory_id": saved["id"], "content": "My favorite tea is black tea"})
    with session_scope() as db:
        results = search(db, "davin", "tea")
        assert len(results) == 1 and results[0]["id"] == corrected["id"]
    run("memory.forget", {"memory_id": corrected["id"], "delete_source": True})
    with session_scope() as db:
        assert search(db, "davin", "tea") == []
        assert db.get(Source, corrected["source_id"]).content == ""


def test_concurrent_budget_reservation_and_usage_dedup():
    run("settings.update", {"monthly_budget_usd": 1})

    def attempt(i):
        try:
            with session_scope() as db:
                budget.reserve(db, "davin", f"reservation-{i}", 0.75, "test")
            return True
        except DomainError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(pool.map(attempt, [0, 1])) == 1
    with session_scope() as db:
        row = db.scalar(select(BudgetReservation))
        budget.record_usage(db, "davin", row.id, "provider-one", "test", {}, 0.25)
        db.flush()
        budget.record_usage(db, "davin", row.id, "provider-one", "test", {}, 0.25)
        budget.close(db, "davin", row.id)
    with session_scope() as db:
        assert budget.summary(db, "davin")["spent_usd"] == 0.25


def test_task_due_time_updates_export_and_no_implicit_reminder(client):
    task = run("task.create", {"title": "Finish report", "due_date": "2026-09-18", "due_time": "14:30"})
    assert task["due_timezone"] == "America/Chicago"
    moved = run("task.update", {"task_id": task["id"], "expected_revision": 1, "due_date": "2026-09-19"})
    assert moved["due_time"] == "14:30"
    assert "14:30" in client.get("/api/v1/export?format=csv").text
    cleared = run("task.update", {"task_id": task["id"], "expected_revision": 2, "due_date": None})
    assert cleared["due_time"] is None and cleared["due_timezone"] is None
    with session_scope() as db:
        assert db.scalar(select(func.count(Schedule.id))) == 0


@pytest.mark.parametrize(
    "values",
    [
        {"due_time": "12:30"},
        {"due_date": "2026-09-18", "due_time": "25:00"},
        {"due_date": "2026-03-08", "due_time": "02:30", "due_timezone": "America/Chicago"},
        {"due_date": "2026-11-01", "due_time": "01:30", "due_timezone": "America/Chicago"},
        {"due_date": "2026-09-18", "due_time": "12:30", "due_timezone": "Invalid/Zone"},
    ],
)
def test_invalid_or_ambiguous_task_due_times_do_not_save(values):
    with pytest.raises(DomainError):
        run("task.create", {"title": "Should not save", **values})
    with session_scope() as db:
        assert db.scalar(select(func.count(Task.id))) == 0


def test_repeated_dst_hour_can_use_an_explicit_offset():
    task = run(
        "task.create",
        {
            "title": "Fall back",
            "due_date": "2026-11-01",
            "due_time": "01:30-06:00",
            "due_timezone": "America/Chicago",
        },
    )
    assert task["due_time"] == "01:30-06:00"
