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


def historic_unconfirmed(rid, amount, recorded, days=45):
    with session_scope() as db:
        budget.reserve(db, "davin", rid, amount, "gpt-realtime-2.1")
        budget.record_usage(db, "davin", rid, rid + ":response", "gpt-realtime-2.1", {}, recorded)
        budget.close(db, "davin", rid, uncertain=True)
        db.flush()
        then = now() - timedelta(days=days)
        db.get(BudgetReservation, rid).created_at = then
        db.get(Usage, rid + ":response").created_at = then


def test_three_usage_buckets_stay_separate():
    historic_unconfirmed("old", 5, 0.25)
    with session_scope() as db:
        budget.reserve(db, "davin", "active", 2, "gpt-5.6-luna")
        budget.record_usage(db, "davin", "active", "active:1", "gpt-5.6-luna", {}, 0.5)
        budget.reserve(db, "davin", "lost", 1, "gpt-5.6-luna")
        budget.close(db, "davin", "lost", uncertain=True)
        db.flush()
        buckets = budget.summary(db, "davin")["buckets"]
        assert buckets["recorded"] == {"usd": pytest.approx(0.5), "records": 1}
        assert buckets["reserved"] == {"usd": pytest.approx(1.5), "sessions": 1}
        unconfirmed = buckets["unconfirmed"]
        assert unconfirmed["usd"] == pytest.approx(5.75) and unconfirmed["sessions"] == 2
        assert unconfirmed["historic_usd"] == pytest.approx(4.75) and unconfirmed["historic_sessions"] == 1
        holds = {h["id"]: h for h in budget.holds(db, "davin")}
        assert (holds["old"]["bucket"], holds["old"]["historic"]) == ("unconfirmed", True)
        assert (holds["lost"]["bucket"], holds["lost"]["historic"]) == ("unconfirmed", False)
        assert holds["active"]["bucket"] == "reserved"


def test_reconcile_settles_only_sessions_with_provider_evidence():
    historic_unconfirmed("old-a", 5, 0.25)
    historic_unconfirmed("old-b", 3, 0.10, days=70)
    historic_unconfirmed("old-c", 2, 0, days=40)
    with session_scope() as db:
        budget.reserve(db, "davin", "active", 2, "gpt-5.6-luna")
    statement = [
        {"reservation": "old-a", "final_usd": "0.40", "evidence": "OpenAI usage export 2026-09, sess_a"},
        {"reservation": "old-c", "final_usd": "0", "evidence": ""},  # a zero without evidence is refused
        {"reservation": "active", "final_usd": "0", "evidence": "not unconfirmed"},
    ]
    lookup = budget.evidence_file_lookup(statement)
    with session_scope() as db:
        dry = budget.reconcile_unconfirmed(db, "davin", lookup, apply=False)
        assert [h["id"] for h in dry["settled"]] == ["old-a"]
        assert db.get(BudgetReservation, "old-a").state == "uncertain"
    with session_scope() as db:
        report = budget.reconcile_unconfirmed(db, "davin", lookup)
        assert [h["id"] for h in report["settled"]] == ["old-a"]
        assert {h["id"]: h["why"] for h in report["unconfirmed"]} == {"old-b": "no_provider_evidence"}
        assert {h["id"]: h["why"] for h in report["rejected"]} == {"old-c": "incomplete_evidence"}
        assert db.get(BudgetReservation, "old-a").settlement["reason"] == "provider_evidence"
        assert db.get(BudgetReservation, "old-b").state == "uncertain"
        assert db.get(BudgetReservation, "old-c").state == "uncertain"
        assert db.get(BudgetReservation, "active").state == "reserved"
        # The adjustment lands in the session's own month, not in this month's recorded charges.
        adjustment = db.get(Usage, "reconcile:old-a")
        assert adjustment.amount == Decimal("0.150000") and adjustment.created_at < now() - timedelta(days=30)
        buckets = budget.summary(db, "davin")["buckets"]
        assert buckets["recorded"]["usd"] == 0
        assert buckets["reserved"]["usd"] == pytest.approx(2)
        assert buckets["unconfirmed"]["sessions"] == 2
        assert buckets["unconfirmed"]["usd"] == pytest.approx(2.9 + 2)
    with session_scope() as db:
        # Re-running with the same statement is a no-op; a failing provider lookup is not evidence.
        assert budget.reconcile_unconfirmed(db, "davin", lookup)["settled"] == []
        failing = budget.reconcile_unconfirmed(db, "davin", lambda hold: 1 / 0)
        assert {h["why"] for h in failing["unconfirmed"]} == {"lookup_failed:ZeroDivisionError"}


def test_evidence_statement_names_each_reservation_once():
    with pytest.raises(ValueError):
        budget.evidence_file_lookup([{"final_usd": "1", "evidence": "x"}])
    with pytest.raises(ValueError):
        budget.evidence_file_lookup([{"reservation": "a", "final_usd": "1", "evidence": "x"}] * 2)
