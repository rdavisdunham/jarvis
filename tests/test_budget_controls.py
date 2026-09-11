from uuid import uuid4

import pytest
from jarvis import budget, memory_learning, worker
from jarvis.db import session_scope
from jarvis.domain import DomainError, execute
from jarvis.models import BudgetReservation, Job, Usage
from sqlalchemy import select


def spend(amount):
    with session_scope() as db:
        execute(db, "davin", str(uuid4()), "settings.update", {"monthly_budget_usd": 10})
        key = str(uuid4())
        budget.reserve(db, "davin", key, amount, "test-model")
        budget.record_usage(db, "davin", key, key, "test-model", {}, amount)
        budget.close(db, "davin", key)


def test_budget_separates_uncertain_holds_warns_and_preserves_local_commands():
    spend(8)
    with session_scope() as db:
        budget.reserve(db, "davin", "uncertain", 0.5, "test-model")
        budget.close(db, "davin", "uncertain", uncertain=True)
        data = budget.summary(db, "davin")
        assert data["budget_mode"] == "warning"
        assert data["spent_usd"] == 8 and data["uncertain_usd"] == 0.5
        assert data["active_reserved_usd"] == 0 and data["projected_month_usd"] >= 8
        assert db.scalar(select(Usage)).tokens["pricing_version"] == budget.PRICING_VERSION
        budget.reserve(db, "davin", "active", 1.5, "test-model")
        assert budget.summary(db, "davin")["budget_mode"] == "paused"
        assert (
            execute(db, "davin", "local", "task.create", {"title": "Local still works"})["status"]
            == "succeeded"
        )


def test_optional_calls_defer_at_95_percent_without_blocking_an_explicit_request():
    spend(9.5)
    with session_scope() as db:
        with pytest.raises(DomainError) as caught:
            budget.reserve(db, "davin", "optional", 0.1, "test-model", optional=True)
        assert caught.value.code == "BUDGET_DEFERRED"
        assert db.get(BudgetReservation, "optional") is None
        budget.reserve(db, "davin", "explicit", 0.1, "test-model")


def test_deferred_memory_job_resumes_once_after_budget_room_returns(monkeypatch):
    spend(9.5)
    with session_scope() as db:
        # An embed job exercises the real optional provider reservation without any network request.
        execute(db, "davin", "capture", "memory.capture", {"content": "My cat is Hayes."})
        job = db.scalar(select(Job).where(Job.kind == "embed_memory"))
        jid = job.id
    from jarvis.config import get_settings

    monkeypatch.setattr(get_settings(), "openai_api_key", "synthetic")
    memory_learning.process(jid)
    with session_scope() as db:
        assert db.get(Job, jid).status == "deferred_budget"
        execute(db, "davin", "more-room", "settings.update", {"monthly_budget_usd": 20})
    worker.housekeeping()
    worker.housekeeping()
    with session_scope() as db:
        prior = db.get(Job, jid)
        assert prior.status == "cancelled"
        resumed = db.get(Job, prior.result["resumed_as"])
        assert resumed.status == "queued"
        assert len(list(db.scalars(select(Job).where(Job.kind == "embed_memory")))) == 2
