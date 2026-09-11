from datetime import UTC, datetime
from uuid import uuid4

from jarvis.db import session_scope
from jarvis.domain import execute, next_occurrence
from jarvis.models import Schedule


def test_fall_back_never_returns_an_instant_before_scan():
    schedule = Schedule(
        timezone="America/Chicago",
        recurrence="FREQ=DAILY",
        anchor_at=datetime(2026, 10, 31, 6, 30, tzinfo=UTC),
    )
    result = next_occurrence(schedule, datetime(2026, 11, 1, 7, 15, tzinfo=UTC))
    assert result == datetime(2026, 11, 2, 7, 30, tzinfo=UTC)


def test_weekdays_start_skips_a_weekend_anchor():
    with session_scope() as db:
        result = execute(
            db,
            "davin",
            str(uuid4()),
            "schedule.create",
            {
                "title": "Weekday routine",
                "when": "2026-09-12T10:00:00",
                "timezone": "America/Chicago",
                "recurrence": "FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR",
            },
        )
        assert result["data"]["next_run_at"] == "2026-09-14T15:00:00+00:00"


def test_edit_completed_task_preserves_completion_time():
    with session_scope() as db:
        task = execute(db, "davin", str(uuid4()), "task.create", {"title": "Finished"})["data"]
        completed = execute(
            db, "davin", str(uuid4()), "task.complete", {"task_id": task["id"], "expected_revision": 1}
        )["data"]
        updated = execute(
            db,
            "davin",
            str(uuid4()),
            "task.update",
            {"task_id": task["id"], "expected_revision": 2, "notes": "Updated later"},
        )["data"]
        assert completed["completed_at"] == updated["completed_at"]


def test_explicit_fall_back_offset_is_kept_on_first_occurrence():
    with session_scope() as db:
        result = execute(
            db,
            "davin",
            str(uuid4()),
            "schedule.create",
            {
                "title": "Explicit fall back",
                "when": "2026-11-01T01:30:00-06:00",
                "timezone": "America/Chicago",
                "recurrence": "FREQ=DAILY",
            },
        )
        assert result["data"]["next_run_at"] == "2026-11-01T07:30:00+00:00"
