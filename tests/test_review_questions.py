from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
from unittest.mock import AsyncMock
import time
import pytest
from sqlalchemy import select
from jarvis import review_questions as q, structure
from jarvis.db import session_scope
from jarvis.domain import DomainError, ReviewDefer, capture_source, execute, preferences
from jarvis.models import Conversation, Job, Memory, MemoryReview, ReviewDelivery, now
from jarvis.structure_models import FieldUnderstanding, RoutingReview

OWNER = "davin"


def setup_question(kind="field"):
    with session_scope() as db:
        schema = structure.ensure(db, OWNER)
        if kind == "field":
            row = db.get(FieldUnderstanding, (OWNER, "type:client"))
            row.status = "needs_input"
            row.questions = ["Does Client mean a company or a person?"]
            row.revision += 1
            key = "field:" + row.definition_id
            mirror = RoutingReview(
                owner_id=OWNER,
                period="field:test",
                status="pending",
                summary={"definition_id": row.definition_id},
                questions=[],
            )
            db.add(mirror)
        else:
            memories = []
            for name in ("Hayes", "Haze"):
                source = capture_source(db, OWNER, "My cat is " + name, str(uuid4()), explicit=True)
                m = Memory(owner_id=OWNER, source_id=source.id, content=source.content, fingerprint=name)
                db.add(m)
                db.flush()
                memories.append(m)
            row = MemoryReview(
                owner_id=OWNER,
                pair_key=str(uuid4()),
                memory_ids=[m.id for m in memories],
                memory_revisions=[m.revision for m in memories],
            )
            db.add(row)
            db.flush()
            key = "memory:" + row.id
        conv = Conversation(owner_id=OWNER, device_id="device", private=False, learning=True)
        db.add(conv)
        db.flush()
        return key, row.revision, conv.id


@pytest.mark.parametrize("kind", ["memory", "field"])
def test_unified_inbox_without_duplicate_mirrors(kind):
    key, rev, conv = setup_question(kind)
    with session_scope() as db:
        page = q.listing(db, OWNER)
        assert len(page["items"]) == 1
        assert page["items"][0]["key"] == key
        assert page["items"][0]["revision"] == rev
        assert page["items"][0]["answer_tool"] in {"memory.resolve", "routing.understand"}
        assert all(r["status"] == "not_run" for r in page["learning"]["runs"])
        assert q.listing(db, "other")["items"] == []


@pytest.mark.parametrize("kind", ["memory", "field"])
@pytest.mark.parametrize("until", ["today", "tomorrow", "week"])
def test_defer_is_source_state_and_revision_guarded(kind, until):
    key, rev, conv = setup_question(kind)
    with session_scope() as db:
        result = execute(
            db,
            OWNER,
            str(uuid4()),
            "review.defer",
            {"question_key": key, "expected_revision": rev, "until": until},
        )
        assert result["data"]["status"] == "deferred"
        assert q.listing(db, OWNER)["items"][0]["status"] == "deferred"
        assert q.reserve(db, OWNER, "device", conv, "text") is None
    with session_scope() as db, pytest.raises(DomainError):
        q.defer(db, OWNER, ReviewDefer(question_key=key, expected_revision=rev))


@pytest.mark.parametrize("event", ["forwarded", "interrupted", "presented"])
def test_delivery_never_resolves_a_question(event):
    key, rev, conv = setup_question()
    channel = "text" if event == "presented" else "voice"
    with session_scope() as db:
        offer = q.reserve(db, OWNER, "device", conv, channel)
        result = q.acknowledge(db, OWNER, "device", offer["id"], event)
        assert result["state"] == event
        assert q.listing(db, OWNER)["items"][0]["status"] == "pending"


def test_cross_device_reservation_is_atomic():
    key, rev, conv = setup_question()
    with session_scope() as db:
        other = Conversation(owner_id=OWNER, device_id="other", private=False, learning=True)
        db.add(other)
        db.flush()
        other_id = other.id

    def reserve(args):
        with session_scope() as db:
            return q.reserve(db, OWNER, *args, "text")

    with ThreadPoolExecutor(max_workers=2) as pool:
        offers = list(pool.map(reserve, [("device", conv), ("other", other_id)]))
    assert sum(bool(o) for o in offers) == 1


def test_expiration_releases_cross_device_lease_without_answer():
    key, rev, conv = setup_question()
    with session_scope() as db:
        offer = q.reserve(db, OWNER, "device", conv, "voice")
        row = db.get(ReviewDelivery, offer["id"])
        row.expires_at = now() - timedelta(minutes=1)
        row.created_at = now() - timedelta(minutes=11)
        other = Conversation(owner_id=OWNER, device_id="other", private=False, learning=True)
        db.add(other)
        db.flush()
        assert q.reserve(db, OWNER, "other", other.id, "text")
        assert row.state == "expired"
        assert q.listing(db, OWNER)["items"][0]["status"] == "pending"


def test_voice_cannot_claim_completed_playback():
    key, rev, conv = setup_question()
    with session_scope() as db:
        offer = q.reserve(db, OWNER, "device", conv, "voice")
        with pytest.raises(DomainError):
            q.acknowledge(db, OWNER, "device", offer["id"], "presented")


def test_changed_source_expires_old_invitation():
    key, rev, conv = setup_question()
    with session_scope() as db:
        offer = q.reserve(db, OWNER, "device", conv, "text")
        field = db.get(FieldUnderstanding, (OWNER, "type:client"))
        field.revision += 1
        field.status = "ready"
        assert q.acknowledge(db, OWNER, "device", offer["id"], "presented")["state"] == "expired"


def test_deleted_memory_or_changed_definition_is_stale():
    key, rev, conv = setup_question("memory")
    with session_scope() as db:
        row = db.get(MemoryReview, key.split(":")[1])
        db.get(Memory, row.memory_ids[0]).suppressed = True
        assert q.listing(db, OWNER, status="stale")["total"] == 1
        assert q.reserve(db, OWNER, "device", conv, "text") is None


def test_changed_field_definition_is_not_offered():
    key, rev, conv = setup_question()
    with session_scope() as db:
        db.get(FieldUnderstanding, (OWNER, "type:client")).fingerprint = "outdated"
        assert q.listing(db, OWNER, status="stale")["total"] == 1
        assert q.reserve(db, OWNER, "device", conv, "text") is None


def test_wrong_device_or_owner_cannot_acknowledge():
    key, rev, conv = setup_question()
    with session_scope() as db:
        offer = q.reserve(db, OWNER, "device", conv, "text")
        for owner, device in [(OWNER, "other"), ("other", "device")]:
            with pytest.raises(DomainError):
                q.acknowledge(db, owner, device, offer["id"], "presented")


def test_text_presentation_throttles_across_conversations_but_keeps_inbox():
    key, rev, conv = setup_question()
    with session_scope() as db:
        offer = q.reserve(db, OWNER, "device", conv, "text")
        q.acknowledge(db, OWNER, "device", offer["id"], "presented")
        other = Conversation(owner_id=OWNER, device_id="other", private=False, learning=True)
        db.add(other)
        db.flush()
        assert q.reserve(db, OWNER, "other", other.id, "text") is None
        assert q.listing(db, OWNER)["total"] == 1


def test_required_task_clarification_prevents_offer(monkeypatch):
    key, rev, conv = setup_question()
    monkeypatch.setattr(q, "work_pending", lambda *a: True)
    with session_scope() as db:
        assert q.reserve(db, OWNER, "device", conv, "text") is None


def test_actual_job_result_is_reported_with_provenance():
    setup_question()
    with session_scope() as db:
        job = Job(
            owner_id=OWNER,
            kind="review_memory",
            payload={},
            status="succeeded",
            finished_at=now(),
            result={"scanned": 17, "merged": 2, "queued_questions": 1},
        )
        db.add(job)
        db.flush()
        run = next(r for r in q.listing(db, OWNER)["learning"]["runs"] if r["kind"] == "review_memory")
        assert run["id"] == job.id and run["result"]["merged"] == 2


@pytest.mark.parametrize("status", ["pending", "deferred", "resolved", "stale"])
def test_inbox_status_filter(status):
    setup_question()
    with session_scope() as db:
        page = q.listing(db, OWNER, status=status)
        assert page["total"] == (1 if status == "pending" else 0)


def test_transcript_does_not_acknowledge_invitation():
    key, rev, conv = setup_question("memory")
    with session_scope() as db:
        offer = q.reserve(db, OWNER, "device", conv, "voice")
        capture_source(db, OWNER, "Is your cat Hayes or Haze?", str(uuid4()), role="assistant")
        assert db.get(ReviewDelivery, offer["id"]).state == "reserved"
        assert db.get(MemoryReview, key.split(":")[1]).last_offered_at is None


async def test_live_only_offers_separately_and_interruption_keeps_question():
    key, rev, conv = setup_question()
    from jarvis.live_voice import LiveController

    with session_scope() as db:
        c = LiveController(OWNER, "device", conv, None, preferences(db, OWNER))
    c.send = AsyncMock()
    c.groups = [{"received_at": time.monotonic() - 10}]
    await c.offer_review()
    event = c.send.await_args.args[0]
    assert event["type"] == "session.commentary.append"
    assert "Optional review invitation" in event["content"]
    assert len(event["content"].encode()) < 1000
    with session_scope() as db:
        delivery = db.scalar(select(ReviewDelivery))
        assert delivery.state == "forwarded"
    c.release_review()
    with session_scope() as db:
        assert db.scalar(select(ReviewDelivery)).state == "interrupted"
        assert q.listing(db, OWNER)["items"][0]["status"] == "pending"


async def test_live_does_not_offer_over_required_question():
    key, rev, conv = setup_question()
    from jarvis.live_voice import LiveController

    with session_scope() as db:
        c = LiveController(OWNER, "device", conv, None, preferences(db, OWNER))
    c.send = AsyncMock()
    c.pending_question = True
    c.groups = [{"received_at": time.monotonic() - 20}]
    await c.offer_review()
    c.send.assert_not_called()


def test_http_inbox_is_authenticated_and_validates_arguments(client):
    setup_question()
    assert client.get("/api/v1/questions").status_code == 200
    assert client.get("/api/v1/questions?limit=999").status_code == 422
    assert client.get("/api/v1/questions?category=unknown").status_code == 422
    client.cookies.clear()
    assert client.get("/api/v1/questions").status_code == 401


def test_http_reservation_rejects_foreign_conversation(client):
    setup_question()
    with session_scope() as db:
        other = Conversation(owner_id="foreign", device_id="foreign", private=False, learning=True)
        db.add(other)
        db.flush()
        identity = other.id
    response = client.post("/api/v1/questions/reserve", json={"conversation_id": identity})
    assert response.status_code in {403, 404}


def test_read_does_not_offer_or_resolve(client):
    setup_question()
    for _ in range(2):
        assert client.get("/api/v1/questions").json()["counts"]["pending"] == 1
    with session_scope() as db:
        assert db.scalar(select(ReviewDelivery)) is None


def test_routing_run_keeps_job_error_alongside_review_summary():
    setup_question()
    with session_scope() as db:
        review = RoutingReview(
            owner_id=OWNER,
            period="failed:test",
            status="failed",
            summary={"message": "Review needs retry"},
            questions=[],
        )
        db.add(review)
        db.flush()
        job = Job(
            owner_id=OWNER,
            kind="review_routing",
            payload={"review_id": review.id},
            status="failed",
            result={"error": "UNDERSTANDING_UNAVAILABLE"},
        )
        db.add(job)
        db.flush()
        run = next(r for r in q.learning(db, OWNER)["runs"] if r["kind"] == "review_routing")
        assert run["id"] == job.id and run["status"] == "failed"
        assert run["result"]["review_id"] == review.id
        assert run["result"]["job_result"]["error"] == "UNDERSTANDING_UNAVAILABLE"
        assert run["result"]["summary"]["message"] == "Review needs retry"
