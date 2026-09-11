from datetime import timedelta
from uuid import uuid4

import pytest
from jarvis.db import session_scope
from jarvis.domain import DomainError, deliver_occurrence, execute, scan_schedules
from jarvis.models import Job, Notification, Schedule, now
from jarvis.tools import call_tool
from sqlalchemy import select


def command(tool, args, key=None):
    with session_scope() as db:
        return execute(db, "davin", key or str(uuid4()), tool, args)["data"]


def reminder(recurrence=None):
    return command(
        "schedule.create",
        {
            "title": "Synthetic reminder",
            "when": (now() - timedelta(minutes=1)).replace(microsecond=0).isoformat(),
            "timezone": "America/Chicago",
            "recurrence": recurrence,
        },
    )


def deliver():
    with session_scope() as db:
        scan_schedules(db)
        for job in db.scalars(select(Job).where(Job.kind == "reminder")):
            deliver_occurrence(db, job)
        db.flush()
        return db.scalar(select(Notification)).id


def test_delivered_reminder_can_be_completed_and_keeps_history(client):
    s = reminder()
    n = deliver()
    key = str(uuid4())
    args = {"notification_id": n}
    first = command("notification.complete", args, key)
    assert command("notification.complete", args, key)["completed_at"] == first["completed_at"]
    rows = client.get("/api/v1/schedules").json()["items"]
    assert rows[0]["id"] == s["id"] and rows[0]["status"] == "completed"
    assert rows[0]["completed_at"]
    assert client.get("/api/v1/notifications").json()["items"][0]["completed_at"]


def test_completing_before_delivery_supersedes_queued_job():
    s = reminder()
    with session_scope() as db:
        scan_schedules(db)
    command("schedule.complete", {"schedule_id": s["id"], "expected_revision": 1})
    with session_scope() as db:
        job = db.scalar(select(Job).where(Job.kind == "reminder"))
        assert deliver_occurrence(db, job)["status"] == "superseded"
        assert db.scalar(select(Notification)) is None


def test_recurring_occurrence_completion_does_not_cancel_future():
    s = reminder("FREQ=DAILY")
    n = deliver()
    command("notification.complete", {"notification_id": n})
    with session_scope() as db:
        row = db.get(Schedule, s["id"])
        assert row.status == "active" and row.next_run_at > now()
    with pytest.raises(DomainError, match="occurrence"):
        command("schedule.complete", {"schedule_id": s["id"], "expected_revision": 1})


async def test_navigation_checks_owner_and_disallows_arbitrary_destinations():
    s = reminder()
    result = await call_tool("davin", "turn", 0, "ui_show", {"view": "reminders", "entity_id": s["id"]})
    assert result["ui_action"]["entity_id"] == s["id"]
    with pytest.raises(DomainError):
        await call_tool("stranger", "turn", 0, "ui_show", {"view": "reminders", "entity_id": s["id"]})
    with pytest.raises(DomainError):
        await call_tool("davin", "turn", 0, "ui_show", {"view": "https://example.com"})
