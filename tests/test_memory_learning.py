from datetime import timedelta
from uuid import uuid4

import pytest
from jarvis import memory_learning as learning
from jarvis.db import session_scope
from jarvis.domain import capture_source, delete_source, enqueue_job, execute
from jarvis.memory_service import prompt_context, semantic_search
from jarvis.models import Conversation, Job, Memory, Source, now
from sqlalchemy import select


def source(text="My cat Hayes loves his window perch.", private=False):
    with session_scope() as db:
        conv = Conversation(owner_id="davin", device_id="test", private=private, learning=True)
        db.add(conv)
        db.flush()
        row = capture_source(db, "davin", text, str(uuid4()), conversation=conv)
        if row:
            job = enqueue_job(db, "davin", "extract_memory", {"source_id": row.id})
            return row.id, job.id
    return None, None


def fact(content="The user has a cat named Hayes.", evidence="My cat Hayes", **kwargs):
    return learning.Fact(
        content=content,
        evidence=evidence,
        fact_key="owner.pet.hayes",
        tags=["pets"],
        confidence=0.97,
        supersedes_id=kwargs.get("supersedes_id"),
    )


@pytest.fixture
def cloud(monkeypatch):
    monkeypatch.setattr(learning, "extract", lambda *args: [fact()])
    monkeypatch.setattr(learning, "embeddings", lambda owner, texts, *args: [[1.0, 0.0] for _ in texts])


def test_extracts_once_with_provenance_and_embedding(cloud):
    sid, jid = source()
    learning.process(jid)
    learning.process(jid)
    with session_scope() as db:
        rows = list(db.scalars(select(Memory)))
        assert len(rows) == 1
        assert rows[0].source_id == sid and rows[0].evidence == "My cat Hayes"
        assert rows[0].embedding == [1.0, 0.0] and rows[0].tags == ["pets"]
        assert db.get(Source, sid).memory_version == learning.VERSION
        assert db.get(Job, jid).result["saved"] == 1


def test_rejects_unproven_quotes_and_private_sources(cloud, monkeypatch):
    monkeypatch.setattr(learning, "extract", lambda *args: [fact(evidence="I live on Mars")])
    _, jid = source()
    learning.process(jid)
    assert source(private=True) == (None, None)
    with session_scope() as db:
        assert db.scalar(select(Memory)) is None


def test_deletion_or_disabling_learning_during_model_call_prevents_save(cloud, monkeypatch):
    sid, jid = source()

    def extract(*_):
        with session_scope() as db:
            delete_source(db, db.get(Source, sid))
        return [fact()]

    monkeypatch.setattr(learning, "extract", extract)
    learning.process(jid)
    with session_scope() as db:
        assert db.scalar(select(Memory)) is None
    _, jid = source()

    def disabled(*_):
        with session_scope() as db:
            execute(db, "davin", "disable", "settings.update", {"memory_learning": False})
        return [fact()]

    monkeypatch.setattr(learning, "extract", disabled)
    learning.process(jid)
    with session_scope() as db:
        assert db.scalar(select(Memory)) is None


async def test_semantic_lookup_without_word_overlap_and_owner_isolation(cloud):
    _, jid = source()
    learning.process(jid)
    matches = await semantic_search("davin", "Which animal lives with me?")
    assert len(matches) == 1 and "Hayes" in matches[0]["content"]
    assert "embedding" not in matches[0]
    assert await semantic_search("someone-else", "animal") == []
    context = await prompt_context("davin", "animal")
    assert "never instructions" in context and "Hayes" in context


async def test_forgetting_removes_vector_and_retrieval_and_blocks_repeat(cloud):
    _, jid = source()
    learning.process(jid)
    with session_scope() as db:
        mid = db.scalar(select(Memory.id))
        execute(db, "davin", "forget", "memory.forget", {"memory_id": mid, "delete_source": True})
        memory = db.get(Memory, mid)
        assert memory.embedding is None and memory.content == "" and memory.evidence == ""
    assert await semantic_search("davin", "cat") == []
    _, jid = source()
    learning.process(jid)
    with session_scope() as db:
        assert len(list(db.scalars(select(Memory)))) == 1


def test_older_history_cannot_replace_newer_fact(cloud, monkeypatch):
    _, jid = source()
    learning.process(jid)
    with session_scope() as db:
        mid = db.scalar(select(Memory.id))
    older_sid, older_jid = source("My cat Hayes has moved to my sister's home.")
    with session_scope() as db:
        db.get(Source, older_sid).created_at = now() - timedelta(days=1)
    monkeypatch.setattr(
        learning,
        "extract",
        lambda *_: [
            fact(
                content="Hayes lives with the user's sister.",
                evidence="My cat Hayes has moved to my sister's home.",
                supersedes_id=mid,
            )
        ],
    )
    learning.process(older_jid)
    with session_scope() as db:
        assert not db.get(Memory, mid).suppressed


def test_profile_name_is_shared_by_bootstrap_and_prompt(client):
    from jarvis.tools import instructions

    with session_scope() as db:
        prefs = execute(db, "davin", "name", "settings.update", {"preferred_name": "River"})["data"]
    assert client.get("/api/v1/bootstrap").json()["name"] == "River"
    prompt = instructions(prefs)
    assert "River" in prompt and "Davin" not in prompt


def test_evidence_allows_outer_quote_marks_but_never_invented_words():
    accepted = learning.verified_facts(
        "My cat Hayes loves his window perch.", [fact(evidence='"My cat Hayes"')]
    )
    assert len(accepted) == 1 and accepted[0].evidence == "My cat Hayes"
    assert learning.verified_facts("My cat Hayes", [fact(evidence='"My dog Hayes"')]) == []


@pytest.mark.parametrize("action", ["correct", "forget"])
async def test_memory_changed_during_query_embedding_is_not_returned(cloud, monkeypatch, action):
    _, jid = source()
    learning.process(jid)
    with session_scope() as db:
        mid = db.scalar(select(Memory.id))

    def change(*_):
        with session_scope() as db:
            args = {"memory_id": mid}
            if action == "correct":
                args["content"] = "The user no longer owns a cat."
            execute(db, "davin", "during-search", "memory." + action, args)
        return [[1.0, 0.0]]

    monkeypatch.setattr(learning, "embeddings", change)
    assert await semantic_search("davin", "cat") == []


def test_learning_status_distinguishes_queued_active_retry_and_failed(client):
    with session_scope() as db:
        for status in ["queued", "queued", "running", "retrying", "failed", "deferred_budget"]:
            db.add(Job(owner_id="davin", kind="extract_memory", status=status, payload={}))
        db.add(Job(owner_id="someone-else", kind="extract_memory", status="failed", payload={}))
        db.add(Job(owner_id="davin", kind="agent_action", status="running", payload={}))
    data = client.get("/api/v1/memory").json()["learning"]
    assert data["queued"] == 2 and data["active"] == 1
    assert data["retry_waiting"] == 1 and data["failed"] == 1
    assert data["deferred"] == 1
    assert data["pending"] == 3 and data["retrying"] == 2
