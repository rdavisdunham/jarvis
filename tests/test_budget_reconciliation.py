from datetime import timedelta
from decimal import Decimal
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from jarvis import budget, conversation
from jarvis.config import get_settings
from jarvis.db import session_scope
from jarvis.domain import DomainError, preferences
from jarvis.models import BudgetReservation, Conversation, Usage, now
from jarvis.voice import Controller
from sqlalchemy import select


def test_stale_reservations_are_uncertain_not_active_and_never_forgiven():
    with session_scope() as db:
        row = budget.reserve(db, "davin", "stale", 5, "gpt-realtime-2.1")
        row.last_activity_at = now() - timedelta(minutes=4)
        db.flush()
        assert budget.summary(db, "davin")["uncertain_usd"] == 5
        assert budget.summary(db, "davin")["active_reserved_usd"] == 0
        assert budget.expire_abandoned(db, "davin") == 1
        assert row.state == "uncertain"
        assert budget.summary(db, "davin")["reserved_usd"] == 5


def test_provider_evidence_reconciliation_is_audited_idempotent_and_owner_scoped():
    with session_scope() as db:
        budget.reserve(db, "davin", "held", 5, "gpt-realtime-2.1")
        budget.record_usage(db, "davin", "held", "response", "gpt-realtime-2.1", {}, 0.02)
        budget.close(db, "davin", "held", uncertain=True)
    with session_scope() as db, pytest.raises(DomainError):
        budget.reconcile(db, "other", "held", 0.01, "provider statement fixture")
    with session_scope() as db:
        budget.reconcile(db, "davin", "held", 0.01, "provider statement fixture")
        db.flush()
        budget.reconcile(db, "davin", "held", 0.01, "provider statement fixture")
        db.flush()
        assert budget.summary(db, "davin")["reserved_usd"] == 0
        assert budget.summary(db, "davin")["spent_usd"] == pytest.approx(0.01)
        assert db.get(Usage, "response").amount == Decimal("0.020000")
        assert db.get(BudgetReservation, "held").settlement["reason"] == "provider_evidence"
    with session_scope() as db, pytest.raises(DomainError):
        budget.reconcile(db, "davin", "held", 0, "different evidence")


@pytest.mark.parametrize(
    "status,expected",
    [(400, "closed"), (401, "closed"), (429, "closed"), (408, "uncertain"), (500, "uncertain")],
)
async def test_text_rejections_release_allowance_but_timeouts_keep_it(monkeypatch, status, expected):
    monkeypatch.setattr(get_settings(), "openai_api_key", "test")
    monkeypatch.setattr(conversation, "prompt_context", AsyncMock(return_value=""))

    class Client:
        def __init__(self, **_):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def post(self, url, **_):
            return httpx.Response(status, request=httpx.Request("POST", url))

    monkeypatch.setattr(conversation.httpx, "AsyncClient", Client)
    with session_scope() as db:
        conv = Conversation(owner_id="davin", device_id="test", private=True, learning=False)
        db.add(conv)
        db.flush()
        cid = conv.id
    rid = str(uuid4())
    result = await conversation.chat("davin", "test", rid, cid, "Hello")
    assert result["status"] == "failed"
    with session_scope() as db:
        assert db.get(BudgetReservation, rid).state == expected


async def test_realtime_transcription_duration_is_counted_once(monkeypatch):
    with session_scope() as db:
        conv = Conversation(owner_id="davin", device_id="test", private=True, learning=False)
        db.add(conv)
        db.flush()
        c = Controller("davin", "test", conv.id, None, preferences(db, "davin"))
        budget.reserve(db, "davin", c.id, 5, c.model)
    c.pending_transcription.add("item")
    event = {
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": "item",
        "transcript": "Test",
        "usage": {"type": "duration", "seconds": 60},
    }
    await c.event(event)
    await c.event(event)
    await c.close()
    with session_scope() as db:
        assert db.get(BudgetReservation, c.id).state == "closed"
        usages = list(db.scalars(select(Usage)))
        assert len(usages) == 1 and usages[0].model == "gpt-live-transcribe"
        assert float(usages[0].amount) == pytest.approx(0.017)


async def test_response_without_usage_is_not_silently_marked_paid():
    with session_scope() as db:
        c = Controller("davin", "test", "unused", None, preferences(db, "davin"))
        budget.reserve(db, "davin", c.id, 5, c.model)
    await c.event({"type": "response.done", "response": {"id": "unknown", "status": "cancelled"}})
    await c.close()
    with session_scope() as db:
        assert db.get(BudgetReservation, c.id).state == "uncertain"
