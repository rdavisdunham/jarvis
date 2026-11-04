"""The context housekeeping tick: flag gate, pipeline schedules, one lint job per weekly slot, notices per owner
with failures isolated, and the worker's housekeeping hook."""

from datetime import UTC, datetime

import pytest
from jarvis import worker
from jarvis.config import get_settings
from jarvis.context import lint, notices, pipeline, scheduler
from jarvis.context import settings as context_settings
from jarvis.context.models import ContextProposal, ContextSource
from jarvis.db import session_scope
from jarvis.models import Job, Notification
from sqlalchemy import func, select

AT = datetime(2026, 11, 2, 15, 0, tzinfo=UTC)


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(context_settings, "enabled", lambda: True)
    calls = []
    monkeypatch.setattr(pipeline, "schedule_due", lambda db, at: calls.append(at) or {"continued": 0})
    return calls


def source(db, owner, title="Pilot scope"):
    row = ContextSource(owner_id=owner, connector="api", external_id=title, title=title, status="ready",
                        current_revision=1, created_at=AT, updated_at=AT)
    db.add(row)
    db.flush()
    db.add(ContextProposal(owner_id=owner, source_id=row.id, kind="create", fingerprint=title, title=title))
    db.flush()
    return row


def lint_jobs(db, owner="davin"):
    return db.scalar(select(func.count()).select_from(Job).where(Job.owner_id == owner, Job.kind == "context_lint"))


def test_tick_does_nothing_while_context_is_off(monkeypatch):
    monkeypatch.setattr(context_settings, "enabled", lambda: False)
    monkeypatch.setattr(pipeline, "schedule_due", lambda db, at: pytest.fail("schedule_due ran"))
    with session_scope() as db:
        source(db, "davin")
        assert scheduler.tick(db, AT)["ready"] == 0
        assert not db.scalar(select(Notification.id))


def test_tick_runs_schedules_and_announces_ready_meetings(enabled, monkeypatch):
    monkeypatch.setattr(lint, "due", lambda db, owner, at: False)
    with session_scope() as db:
        source(db, "davin")
        out = scheduler.tick(db, AT)
        assert enabled == [AT] and out["ready"] == 1 and out["failed"] == []
        assert scheduler.tick(db, AT)["ready"] == 0


def test_lint_is_queued_once_per_weekly_slot(enabled, monkeypatch):
    due = {"value": True}
    monkeypatch.setattr(lint, "due", lambda db, owner, at: due["value"])
    monkeypatch.setattr(lint, "slot", lambda db, owner, at: "2026-11-01T08:00:00+00:00")
    with session_scope() as db:
        source(db, "davin")
        assert scheduler.tick(db, AT)["lint"] == 1
        assert scheduler.tick(db, AT)["lint"] == 0  # pending job
        db.execute(Job.__table__.update().values(status="failed"))
        assert scheduler.tick(db, AT)["lint"] == 0  # this slot already had its job
        assert lint_jobs(db) == 1
        monkeypatch.setattr(lint, "slot", lambda db, owner, at: "2026-11-08T08:00:00+00:00")
        assert scheduler.tick(db, AT)["lint"] == 1
        due["value"] = False
        db.execute(Job.__table__.update().values(status="succeeded"))
        monkeypatch.setattr(lint, "slot", lambda db, owner, at: "2026-11-15T08:00:00+00:00")
        assert scheduler.tick(db, AT)["lint"] == 0
        [payload] = db.scalars(select(Job.payload).where(Job.kind == "context_lint", Job.status == "succeeded")
                               .order_by(Job.created_at.desc()).limit(1))
        assert payload == {"slot": "2026-11-08T08:00:00+00:00"}


def test_one_owner_failing_does_not_block_another(enabled, monkeypatch):
    monkeypatch.setattr(lint, "due", lambda db, owner, at: False)
    real = notices.sync

    def sync(db, owner, at):
        if owner == "alpha":
            source(db, owner, "written then rolled back")
            raise RuntimeError("boom")
        return real(db, owner, at)

    monkeypatch.setattr(notices, "sync", sync)
    with session_scope() as db:
        source(db, "alpha")
        source(db, "davin")
        out = scheduler.tick(db, AT)
        assert out["failed"] == ["alpha"] and out["ready"] == 1
        assert db.scalar(select(func.count()).select_from(ContextSource).where(ContextSource.owner_id == "alpha")) == 1


def test_housekeeping_calls_the_tick_only_when_enabled(monkeypatch):
    calls = []
    monkeypatch.setattr(scheduler, "tick", lambda db, at: calls.append(at))
    worker.housekeeping()
    assert calls == []
    monkeypatch.setenv("JARVIS_CONTEXT_ENABLED", "true")
    get_settings.cache_clear()
    try:
        worker.housekeeping()
    finally:
        monkeypatch.delenv("JARVIS_CONTEXT_ENABLED")
        get_settings.cache_clear()
    assert len(calls) == 1


def test_real_lint_due_queues_the_dream_slot(enabled):
    with session_scope() as db:
        source(db, "davin")
        assert lint.due(db, "davin", AT)
        assert scheduler.tick(db, AT)["lint"] == 1
        assert db.scalar(select(Job.payload).where(Job.kind == "context_lint")) == {"slot": lint.slot(db, "davin", AT)}
