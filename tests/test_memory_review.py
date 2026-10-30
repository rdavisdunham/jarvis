from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from jarvis import memory_review as review
from jarvis.db import session_scope
from jarvis.domain import DomainError, capture_source, execute, preferences
from jarvis.memory_learning import fingerprint
from jarvis.models import Job, Memory, MemoryReview, Source, now
from jarvis.tools import call_tool
from sqlalchemy import select


def remember(content, owner="davin", attribution="automatic"):
    with session_scope() as db:
        source = capture_source(db, owner, content, str(uuid4()), explicit=True)
        memory = Memory(
            owner_id=owner,
            source_id=source.id,
            content=content,
            evidence=content,
            fingerprint=fingerprint(content),
            embedding=[1.0, 0.0],
            attribution=attribution,
        )
        db.add(memory)
        db.flush()
        return memory.id, source.id


def run():
    with session_scope() as db:
        job = review.queue_review(db, "davin", manual=True)
        jid = job.id
    review.process(jid)
    return jid


def ambiguous():
    a = remember("The user has a cat named Hayes.")
    b = remember("The user has a cat named Haze.")
    run()
    with session_scope() as db:
        r = db.scalar(select(MemoryReview))
        return a, b, r.id, r.revision


def test_exact_duplicates_merge_but_preserve_sources_and_retry_is_noop():
    a, sa = remember("The user prefers tea.")
    b, sb = remember("The user prefers tea.", attribution="owner_statement")
    other, _ = remember("The user prefers tea.", owner="someone-else")
    jid = run()
    review.process(jid)
    with session_scope() as db:
        assert db.get(Memory, a).merged_into_id == b
        assert db.get(Memory, a).suppressed and db.get(Memory, a).embedding is None
        assert not db.get(Memory, b).suppressed
        assert not db.get(Memory, other).suppressed
        assert db.get(Source, sa).content and db.get(Source, sb).content
        assert db.get(Job, jid).result == {"scanned": 2, "merged": 1, "queued_questions": 0}


def test_spelling_candidates_are_queued_not_guessed_and_not_duplicated():
    a, b, rid, _ = ambiguous()
    run()
    with session_scope() as db:
        assert len(list(db.scalars(select(MemoryReview)))) == 1
        assert not db.get(Memory, a[0]).suppressed and not db.get(Memory, b[0]).suppressed
        data = review.review_data(db, db.get(MemoryReview, rid))
        assert "haze" in data["question"] and "hayes" in data["question"]
        assert review.pending_reviews(db, "someone-else") == []


def test_owner_resolution_has_receipt_provenance_and_new_embedding_job():
    a, b, rid, revision = ambiguous()
    args = {
        "review_id": rid,
        "expected_revision": revision,
        "action": "merge",
        "content": "The user's cat is named Hayes.",
    }
    with session_scope() as db:
        result = execute(db, "davin", "confirmed", "memory.resolve", args)
        assert execute(db, "davin", "confirmed", "memory.resolve", args) == result
        memory = db.get(Memory, result["data"]["memory_id"])
        assert memory.content == args["content"] and memory.attribution == "owner_statement"
        assert db.get(Source, memory.source_id).explicit
        assert db.get(Memory, a[0]).merged_into_id == memory.id
        assert db.get(Memory, b[0]).merged_into_id == memory.id
        assert db.scalar(select(Job).where(Job.kind == "embed_memory")).payload["memory_id"] == memory.id
        assert review.pending_reviews(db, "davin") == []


def test_distinct_answer_is_not_asked_again_next_run():
    a, b, rid, revision = ambiguous()
    with session_scope() as db:
        execute(
            db,
            "davin",
            "different",
            "memory.resolve",
            {"review_id": rid, "expected_revision": revision, "action": "distinct"},
        )
    run()
    with session_scope() as db:
        assert review.pending_reviews(db, "davin") == []
        assert not db.get(Memory, a[0]).suppressed and not db.get(Memory, b[0]).suppressed


def test_defer_and_context_cooldown_leave_question_pending():
    _, _, rid, revision = ambiguous()
    with session_scope() as db:
        context = review.context_for_agent(db, "davin")
        assert rid in context and "not a fact" in context
        assert rid in review.context_for_agent(db, "davin")  # Preparation is not delivery.
        review.record_question(db, "davin", "Is your cat named Hayes or Haze?")
        assert review.context_for_agent(db, "davin") == ""
        execute(
            db,
            "davin",
            "later",
            "memory.resolve",
            {"review_id": rid, "expected_revision": revision, "action": "defer"},
        )
        assert review.pending_reviews(db, "davin") == []
        row = db.get(MemoryReview, rid)
        assert row.status == "pending"
        row.deferred_until = now() - timedelta(seconds=1)
        row.last_offered_at = now() - timedelta(days=2)
        assert rid in review.context_for_agent(db, "davin")


def test_deleted_or_corrected_candidate_cannot_be_resolved_or_injected():
    a, _, rid, revision = ambiguous()
    with session_scope() as db:
        execute(db, "davin", "forget", "memory.forget", {"memory_id": a[0], "delete_source": True})
    with pytest.raises(DomainError, match="changed"), session_scope() as db:
        execute(
            db,
            "davin",
            "stale",
            "memory.resolve",
            {
                "review_id": rid,
                "expected_revision": revision,
                "action": "merge",
                "content": "The user has a cat named Hayes.",
            },
        )
    with session_scope() as db:
        assert review.context_for_agent(db, "davin") == ""
        assert db.get(MemoryReview, rid).status == "stale"


def test_weekly_slot_catches_up_once_and_respects_timezone_and_disable():
    remember("The user likes tea.")
    instant = datetime(2026, 11, 1, 8, 59, tzinfo=UTC)  # 02:59 after US fall-back
    with session_scope() as db:
        prefs = preferences(db, "davin")
        start, next_at = review.period(prefs, instant)
        assert next_at == datetime(2026, 11, 1, 9, tzinfo=UTC)
        job = review.queue_review(db, "davin", instant=instant)
        assert review.queue_review(db, "davin", instant=instant).id == job.id
        job.status, job.finished_at = "succeeded", instant
        assert review.queue_review(db, "davin", instant=instant) is None
        new = review.queue_review(db, "davin", instant=next_at)
        assert new.id != job.id and start < next_at
        execute(db, "davin", "disable", "settings.update", {"deep_sleep_enabled": False})
        assert review.queue_review(db, "davin", manual=True) is None
        jid = new.id
    review.process(jid)
    with session_scope() as db:
        assert db.get(Job, jid).result == {"skipped": True}


async def test_review_api_and_agent_tools_enforce_owner_and_revision(client):
    _, _, rid, revision = ambiguous()
    data = client.get("/api/v1/memory").json()
    assert len(data["reviews"]) == 1 and "legacy" not in data
    assert data["maintenance"]["last_run_at"]
    assert (await call_tool("someone-else", "read", 0, "memory_review_list", {}))["reviews"] == []
    args = {"review_id": rid, "expected_revision": revision, "action": "distinct"}
    with pytest.raises(DomainError, match="no longer available"):
        await call_tool("someone-else", "deny", 0, "memory_resolve", args)
    result = await call_tool("davin", "allowed", 0, "memory_resolve", args)
    assert result["data"]["status"] == "distinct"
    with pytest.raises(DomainError, match="changed"):
        await call_tool("davin", "stale", 0, "memory_resolve", args)
    assert client.post("/api/v1/memory/review").status_code == 200


def test_similarity_never_merges_different_numbers_or_unrelated_facts():
    remember("The user has 2 cats.")
    remember("The user has 3 cats.")
    remember("The user prefers tea.")
    remember("The user prefers coffee.")
    run()
    with session_scope() as db:
        assert len(review.active_memories(db, "davin")) == 4
        # EVAL-005: different numbers are never merged, but the owner is asked once.
        reviews = list(db.scalars(select(MemoryReview)))
        assert [r.kind for r in reviews] == ["numeric"]
        assert {'"2"', '"3"'} <= set(review.review_data(db, reviews[0])["question"].replace(",", "").split())


async def test_startup_context_does_not_consume_the_clarification_offer():
    from jarvis.memory_service import prompt_context

    _, _, rid, _ = ambiguous()
    assert rid not in await prompt_context("davin")
    assert rid not in await prompt_context("davin", "cat")


def test_weekly_enqueue_is_unique_across_concurrent_workers():
    from concurrent.futures import ThreadPoolExecutor

    remember("The user likes tea.")

    def enqueue(_):
        with session_scope() as db:
            return review.queue_review(db, "davin").id

    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(enqueue, range(4)))
    assert len(set(ids)) == 1


def test_review_failure_is_retryable_without_partial_merges(monkeypatch):
    a, _ = remember("The user likes tea.")
    b, _ = remember("The user likes tea.")
    with session_scope() as db:
        jid = review.queue_review(db, "davin").id

    def fail(_):
        raise RuntimeError("test failure")
        yield

    monkeypatch.setattr(review, "possible_pairs", fail)
    with pytest.raises(RuntimeError, match="test failure"):
        review.process(jid)
    with session_scope() as db:
        assert db.get(Job, jid).status == "retrying"
        assert not db.get(Memory, a).suppressed and not db.get(Memory, b).suppressed


def test_review_failure_is_visible_and_manual_retry_queues_new_run():
    remember("The user likes tea.")
    with session_scope() as db:
        job = review.queue_review(db, "davin", manual=True)
        job.status, job.result = "failed", {"error": "SyntheticError"}
    with session_scope() as db:
        result = review.status(db, "davin")
        assert result["status"] == "failed" and result["last_error"] == "SyntheticError"
        assert not result["running"]
        retry = review.queue_review(db, "davin", manual=True)
        assert retry.status == "queued"
        assert review.status(db, "davin")["running"]


def test_unasked_review_remains_available_and_private_output_records_question():
    _, _, rid, _ = ambiguous()
    with session_scope() as db:
        assert rid in review.context_for_agent(db, "davin")
        review.record_question(db, "davin", "Your task is saved.")
        assert rid in review.context_for_agent(db, "davin")
        review.record_question(db, "davin", "Is it Hayes or Haze?")
        assert review.context_for_agent(db, "davin") == ""


def embedded(content, vector, fact_key="", attribution="automatic"):
    from jarvis.memory_learning import EMBEDDING_MODEL

    mid, sid = remember(content, attribution=attribution)
    with session_scope() as db:
        row = db.get(Memory, mid)
        row.embedding, row.embedding_model, row.fact_key = vector, EMBEDDING_MODEL, fact_key
    return mid, sid


def set_profile(**values):
    with session_scope() as db:
        execute(db, "davin", str(uuid4()), "settings.update", values)


def test_semantic_paraphrases_are_queued_with_provenance_never_merged():
    a, sa = embedded("The user prefers tea.", [1.0, 0.10, 0.0], "owner.preferred_drink")
    b, _ = embedded("The user likes to drink tea in the morning.", [1.0, 0.14, 0.0], "owner.drink.morning")
    c, _ = embedded("The user has a cat named Hayes.", [0.0, 0.0, 1.0])
    # Vectors from another (or no) embedding model are never compared.
    remember("The user enjoys tea.")
    jid = run()
    with session_scope() as db:
        assert all(not db.get(Memory, m).suppressed for m in (a, b, c))
        reviews = list(db.scalars(select(MemoryReview)))
        assert [r.kind for r in reviews] == ["paraphrase"]
        assert sorted(reviews[0].memory_ids) == sorted([a, b])
        assert reviews[0].details["similarity"] >= review.SEMANTIC
        data = review.review_data(db, reviews[0])
        assert "same thing" in data["question"]
        prov = data["provenance"]
        assert prov["method"] == "paraphrase" and prov["similarity"] == reviews[0].details["similarity"]
        assert {s["memory_id"]: s["source_id"] for s in prov["sources"]}[a] == sa
        assert {s["fact_key"] for s in prov["sources"]} == {"owner.preferred_drink", "owner.drink.morning"}
        assert db.get(Job, jid).result["queued_questions"] == 1
        from jarvis.review_questions import items

        item = next(q for q in items(db, "davin") if q["kind"] == "memory")
        assert item["review_kind"] == "paraphrase" and item["evidence"]["method"] == "paraphrase"
    run()
    with session_scope() as db:
        assert len(list(db.scalars(select(MemoryReview)))) == 1


def test_profile_disagreement_is_a_question_and_never_edits_either_side():
    set_profile(preferred_name="Davis", timezone="America/Chicago")
    dave, _ = embedded("The user's name is Dave.", [], "owner.name")
    embedded("The user's full name is Davis Dunham.", [], "owner.full_name")
    embedded("The user is on Central time.", [], "owner.time_zone")
    berlin, _ = embedded("The user's timezone is Europe/Berlin.", [], "owner.timezone")
    run()
    with session_scope() as db:
        reviews = {r.memory_ids[0]: r for r in db.scalars(select(MemoryReview).where(MemoryReview.kind == "profile"))}
        assert set(reviews) == {dave, berlin}
        name = reviews[dave]
        assert name.details["profile"] == {"field": "preferred_name", "value": "Davis", "set_by": "owner"}
        assert reviews[berlin].details["profile"]["stated"] == "Europe/Berlin"
        data = review.review_data(db, name)
        assert '"Davis"' in data["question"] and [c["id"] for c in data["candidates"]] == [dave]
        assert data["provenance"]["profile"]["field"] == "preferred_name"
        context = review.context_for_agent(db, "davin")
        assert '"profile"' in context and "never changes the profile" in context
        review.record_question(db, "davin", "Is your name Dave or Davis?")
        assert db.get(MemoryReview, name.id).last_offered_at
        rid, revision, zone_id = name.id, name.revision, reviews[berlin].id
    with session_scope() as db:
        result = execute(db, "davin", "profile-answer", "memory.resolve", {
            "review_id": rid, "expected_revision": revision, "action": "merge",
            "content": "The user's nickname is Dave; they go by Davis.",
        })["data"]
        assert db.get(Memory, dave).merged_into_id == result["memory_id"]
        assert preferences(db, "davin")["preferred_name"] == "Davis"
    # A profile change makes the remaining profile question stale rather than answerable.
    set_profile(timezone="Europe/Berlin")
    with session_scope() as db:
        assert review.pending_reviews(db, "davin") == []
        assert db.get(MemoryReview, zone_id).status == "stale"


def test_agent_filed_memories_are_not_owner_statements():
    with session_scope() as db:
        agent = execute(db, "davin", "turn-1:0", "memory.capture", {"content": "The user prefers oolong."})
        owner = execute(db, "davin", "plain", "memory.capture", {"content": "The user prefers oolong."})
        assert db.get(Memory, agent["data"]["id"]).attribution == "agent_capture"
        assert db.get(Memory, owner["data"]["id"]).attribution == "owner_statement"
    run()
    with session_scope() as db:
        # The exact duplicate keeps the owner's own statement, whichever was saved first.
        assert db.get(Memory, agent["data"]["id"]).merged_into_id == owner["data"]["id"]
