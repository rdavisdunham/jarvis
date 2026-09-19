import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from jarvis import budget
from jarvis.config import get_settings
from jarvis.cost_features import current_feature, feature
from jarvis.db import session_scope
from jarvis.models import BudgetReservation, Usage
from sqlalchemy import select


def test_tracking_only_records_without_blocking_on_old_holds(monkeypatch):
    monkeypatch.setattr(get_settings(), "budget_enforcement_enabled", False)
    with session_scope() as db:
        budget.reserve(db, "davin", "old", 10000, "test")
        budget.close(db, "davin", "old", uncertain=True)
        budget.reserve(db, "davin", "new", 1, "test", optional=True)
        budget.ensure_room(db, "davin", "new", 2)
        budget.record_usage(db, "davin", "new", "request", "test", {}, .01, feature="assistant")
        budget.record_usage(db, "davin", "new", "request", "test", {}, .01, feature="assistant")
        db.flush()
        data = budget.summary(db, "davin")
        assert data["budget_mode"] == "tracking_only"
        assert data["spent_usd"] == .01
        assert data["uncertain_usd"] == 10000
        assert len(list(db.scalars(select(Usage)))) == 1
        assert data["report"]["features"][0]["id"] == "assistant"


def test_windows_owner_isolation_and_unknown_usage(monkeypatch):
    instant = datetime(2026, 9, 19, 12, tzinfo=UTC)
    monkeypatch.setattr(budget, "now", lambda: instant)
    monkeypatch.setattr(get_settings(), "cost_tracking_since", instant - timedelta(days=2))
    with session_scope() as db:
        for key, days, owner, tag, cost in [
            ("week-boundary", 7, "davin", "voice", 1),
            ("month-only", 8, "davin", "note_organization", 2),
            ("month-boundary", 30, "davin", None, 3),
            ("old", 31, "davin", "voice", 100),
            ("foreign", 1, "someone-else", "voice", 100),
            ("future", -1, "davin", "voice", 100),
        ]:
            db.add(BudgetReservation(id=key, owner_id=owner, model="test", amount=Decimal(cost)))
            db.flush()
            db.add(Usage(request_id=key, reservation_id=key, owner_id=owner, model="test",
                         amount=Decimal(cost), tokens={"feature": tag} if tag else {},
                         created_at=instant-timedelta(days=days)))
        db.flush()
        report = budget.usage_report(db, "davin")
        assert report["last_7_days_usd"] == 1
        assert report["last_30_days_usd"] == 6
        assert report["partial_7_days"] and report["partial_30_days"]
        assert {f["id"] for f in report["features"]} == {"voice", "note_organization", "unattributed"}
        assert budget.summary(db, "davin")["projected_month_usd"] is None


@pytest.mark.asyncio
async def test_parallel_feature_scopes_propagate_to_threads_and_reset():
    @feature("note_organization")
    async def notes():
        await asyncio.sleep(0)
        assert await asyncio.to_thread(current_feature.get) == "note_organization"
        return current_feature.get()

    @feature("memory_retrieval")
    async def memory():
        await asyncio.sleep(0)
        assert await asyncio.to_thread(current_feature.get) == "memory_retrieval"
        return current_feature.get()

    assert await asyncio.gather(notes(), memory()) == ["note_organization", "memory_retrieval"]
    assert current_feature.get() == "unattributed"

    @feature("note_search")
    def fails():
        raise ValueError()
    with pytest.raises(ValueError):
        fails()
    assert current_feature.get() == "unattributed"


def test_incomplete_extraction_still_records_reported_usage(monkeypatch):
    import httpx
    from jarvis import memory_learning
    from jarvis.domain import DomainError
    original = httpx.Client
    monkeypatch.setattr(get_settings(), "openai_api_key", "synthetic")
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={
        "status": "incomplete", "usage": {"input_tokens": 1000, "output_tokens": 100},
    }))
    monkeypatch.setattr(memory_learning.httpx, "Client", lambda **kw: original(transport=transport))
    @feature("note_organization")
    def call():
        return memory_learning.provider_request("davin", "responses", {}, "gpt-5.6-luna", .08)
    with pytest.raises(DomainError, match="Extraction did not finish"):
        call()
    with session_scope() as db:
        usage = db.scalar(select(Usage))
        assert usage.amount == Decimal("0.000320")
        assert usage.tokens["feature"] == "note_organization"
        assert db.get(BudgetReservation, usage.reservation_id).state == "closed"


def test_missing_embedding_usage_is_uncertain_not_a_zero_charge(monkeypatch):
    import httpx
    from jarvis import memory_learning
    original = httpx.Client
    monkeypatch.setattr(get_settings(), "openai_api_key", "synthetic")
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"data": []}))
    monkeypatch.setattr(memory_learning.httpx, "Client", lambda **kw: original(transport=transport))
    with pytest.raises(ValueError, match="Embedding response omitted"):
        memory_learning.provider_request("davin", "embeddings", {}, "text-embedding-3-small", .002)
    with session_scope() as db:
        assert db.scalar(select(Usage)) is None
        assert db.scalar(select(BudgetReservation)).state == "uncertain"
