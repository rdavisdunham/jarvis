"""Audit regressions: spoken numbers, dream conflicts, embedding outages, list coverage."""

from datetime import timedelta
from uuid import uuid4

import pytest
from jarvis import memory_learning as learning
from jarvis import memory_review as review
from jarvis.config import get_settings
from jarvis.db import session_scope
from jarvis.domain import capture_source, enqueue_job, execute
from jarvis.memory_learning import fingerprint
from jarvis.memory_service import prompt_context
from jarvis.memory_service import search as memory_search
from jarvis.models import Conversation, Job, Memory, MemoryReview, Note, Source, now
from jarvis.text_normalize import canon
from sqlalchemy import select

OWNER = "davin"


@pytest.mark.parametrize(
    "a, b",
    [
        ("test test 123", "Test, test, one, two, three"),
        ("route 66", "Route sixty-six"),
        ("route 66", "route six six"),
        ("23 apples", "twenty three apples"),
        ("1000 people", "1,000 people"),
        ("123", "one hundred and twenty three"),
    ],
)
def test_canon_equates_spoken_and_written_numbers(a, b):
    assert canon(a) == canon(b) and canon(canon(a)) == canon(a)


def test_canon_keeps_decimals_and_separate_multi_digit_numbers():
    assert canon("3.5 cats") != canon("35 cats")
    assert canon("nineteen eighty four") == "19 84"


def remember(content, fact_key="", created=None):
    with session_scope() as db:
        source = capture_source(db, OWNER, content, str(uuid4()), explicit=True)
        row = Memory(
            owner_id=OWNER,
            source_id=source.id,
            content=content,
            evidence=content,
            fact_key=fact_key,
            fingerprint=fingerprint(content),
            embedding=[1.0, 0.0],
            embedding_model=learning.EMBEDDING_MODEL,
        )
        if created:
            row.created_at = created
        db.add(row)
        db.flush()
        return row.id


def dream():
    with session_scope() as db:
        jid = review.queue_review(db, OWNER, manual=True).id
    review.process(jid)


def test_numeric_contradiction_queues_question_and_spoken_offer_counts():
    remember("The user has 2 cats.")
    remember("The user has three cats.")
    dream()
    with session_scope() as db:
        rows = list(db.scalars(select(MemoryReview)))
        assert [r.kind for r in rows] == ["numeric"]
        assert {'"2"', '"3"'} <= set(review.review_data(db, rows[0])["question"].replace(",", "").split())
        review.record_question(db, OWNER, "Quick check: do you have two or three cats?")
        assert db.get(MemoryReview, rows[0].id).last_offered_at


def test_fact_key_conflict_asks_about_two_newest_and_resolution_keeps_key():
    remember("The user lives in Boston.", "owner.home.city", now() - timedelta(days=3))
    old = remember("The user lives in Austin.", "owner.home.city", now() - timedelta(days=2))
    new = remember("The user lives in Denver.", "owner.home.city", now() - timedelta(days=1))
    remember("The user likes tea.", "owner.drink")
    dream()
    with session_scope() as db:
        rows = list(db.scalars(select(MemoryReview)))
        assert len(rows) == 1 and rows[0].kind == "conflict"
        assert set(rows[0].memory_ids) == {old, new}
        rid, revision = rows[0].id, rows[0].revision
    with session_scope() as db:
        result = execute(
            db,
            OWNER,
            str(uuid4()),
            "memory.resolve",
            {"review_id": rid, "expected_revision": revision, "action": "merge", "content": "The user lives in Denver."},
        )["data"]
        assert db.get(Memory, result["memory_id"]).fact_key == "owner.home.city"


def source(text):
    with session_scope() as db:
        conv = Conversation(owner_id=OWNER, device_id="test", private=False, learning=True)
        db.add(conv)
        db.flush()
        row = capture_source(db, OWNER, text, str(uuid4()), conversation=conv)
        return row.id, enqueue_job(db, OWNER, "extract_memory", {"source_id": row.id}).id


def fact(content, evidence, key="owner.home.city", supersedes=None):
    return learning.Fact(
        content=content, evidence=evidence, fact_key=key, tags=["home"], confidence=0.97, supersedes_id=supersedes
    )


def test_embedding_outage_commits_fact_and_embed_job_suppresses_late_duplicate(monkeypatch):
    existing = remember("The user lives in Austin, Texas.", "owner.home.city")
    sid, jid = source("I live in Austin.")
    extractions = []
    monkeypatch.setattr(learning, "extract", lambda *a: extractions.append(1) or [fact("The user lives in Austin.", "I live in Austin")])

    def outage(*a):
        raise RuntimeError("embedding provider down")

    monkeypatch.setattr(learning, "embeddings", outage)
    learning.process(jid)
    with session_scope() as db:
        row = db.scalar(select(Memory).where(Memory.source_id == sid))
        assert row and row.embedding is None and not row.suppressed
        assert db.get(Job, jid).status == "succeeded" and db.get(Source, sid).memory_version == learning.VERSION
        embed = db.scalar(select(Job).where(Job.kind == "embed_memory"))
        assert embed.payload == {"memory_id": row.id, "dedupe": True}
        mid, eid = row.id, embed.id
    monkeypatch.setattr(learning, "embeddings", lambda owner, texts, *a: [[1.0, 0.0] for _ in texts])
    learning.process(eid)
    learning.process(jid)
    with session_scope() as db:
        row = db.get(Memory, mid)
        assert row.suppressed and row.merged_into_id == existing
        assert not db.get(Memory, existing).suppressed
    assert extractions == [1]


def test_failed_extraction_is_requeued_with_backoff_then_stops():
    sid, jid = source("My favorite color is green.")
    with session_scope() as db:
        job = db.get(Job, jid)
        job.status, job.payload = "failed", {**job.payload, "version": learning.VERSION}
        job.finished_at = now()
    with session_scope() as db:
        assert learning.queue_backfill(db) == 0
        db.get(Job, jid).finished_at = now() - timedelta(hours=7)
    with session_scope() as db:
        assert learning.queue_backfill(db) == 1
        assert learning.queue_backfill(db) == 0
    with session_scope() as db:
        for job in db.scalars(select(Job).where(Job.kind == "extract_memory")):
            job.status, job.finished_at = "failed", now() - timedelta(days=30)
            job.payload = {**job.payload, "version": learning.VERSION}
        db.add(Job(owner_id=OWNER, kind="extract_memory", status="failed",
                   payload={"source_id": sid, "version": learning.VERSION}, finished_at=now() - timedelta(days=30)))
    with session_scope() as db:
        assert learning.queue_backfill(db) == 0


def test_reasserting_superseded_value_is_saved_but_forgotten_fact_stays_forgotten(monkeypatch):
    monkeypatch.setattr(learning, "embeddings", lambda owner, texts, *a: [[1.0, 0.0] for _ in texts])
    austin = "The user lives in Austin."
    _, first = source("I live in Austin.")
    monkeypatch.setattr(learning, "extract", lambda *a: [fact(austin, "I live in Austin")])
    learning.process(first)
    with session_scope() as db:
        aid = db.scalar(select(Memory.id))
    _, second = source("I moved to Denver.")
    monkeypatch.setattr(learning, "extract", lambda *a: [fact("The user lives in Denver.", "I moved to Denver", supersedes=aid)])
    learning.process(second)
    with session_scope() as db:
        did = db.scalar(select(Memory.id).where(Memory.content.like("%Denver%")))
    _, third = source("I moved back to Austin.")
    monkeypatch.setattr(learning, "extract", lambda *a: [fact(austin, "I moved back to Austin", supersedes=did)])
    learning.process(third)
    with session_scope() as db:
        active = [m.content for m in db.scalars(select(Memory).where(Memory.suppressed.is_(False)))]
        assert active == [austin]
        current = db.scalar(select(Memory.id).where(Memory.suppressed.is_(False)))
        execute(db, OWNER, str(uuid4()), "memory.forget", {"memory_id": current})
    _, fourth = source("I live in Austin.")
    monkeypatch.setattr(learning, "extract", lambda *a: [fact(austin, "I live in Austin")])
    learning.process(fourth)
    with session_scope() as db:
        assert db.scalar(select(Memory.id).where(Memory.suppressed.is_(False))) is None


def test_memory_lexical_search_matches_numbers_both_ways():
    a = remember("test, test, one, two, three")
    b = remember("Drive route 66 west")
    with session_scope() as db:
        assert [m["id"] for m in memory_search(db, OWNER, "test test 123")] == [a]
        assert [m["id"] for m in memory_search(db, OWNER, "route sixty-six")] == [b]
    c = remember("call 911 now")
    with session_scope() as db:
        assert [m["id"] for m in memory_search(db, OWNER, "nine one one")] == [c]


async def test_prompt_context_has_real_newline_and_reuses_query_vector(monkeypatch):
    remember("The user has a cat named Hayes.")
    calls = []
    monkeypatch.setattr(learning, "embeddings", lambda owner, texts, *a: calls.append(texts) or [[1.0, 0.0]])
    first = await prompt_context(OWNER, "cat Hayes")
    second = await prompt_context(OWNER, "cat Hayes")
    assert "\\n" not in first and "here.\n[" in first and second == first
    assert len(calls) == 1


def test_note_keyword_search_and_list_filter_match_spoken_numbers(client, monkeypatch):
    from jarvis import notes

    def create(title, content):
        r = client.post("/api/v1/commands", json={"command_id": str(uuid4()), "tool": "note.create",
                                                  "arguments": {"title": title, "content": content}})
        assert r.status_code == 200, r.text
        return r.json()["data"]["id"]

    spoken = create("Mic check", "test, test, one, two, three")
    written = create("Trip", "Drive route 66 to the coast")
    create("Other", "Nothing relevant 50%")
    with session_scope() as db:
        assert [n["id"] for n in notes.list_notes(db, OWNER, query="test test 123")["items"]] == [spoken]
        assert [n["id"] for n in notes.list_notes(db, OWNER, query="route sixty six")["items"]] == [written]
        assert [n["id"] for n in notes.list_notes(db, OWNER, query="50%")["items"]] != []
        assert notes.list_notes(db, OWNER, query="_")["items"] == []
    monkeypatch.setattr(notes, "embeddings", lambda *a, **k: (_ for _ in ()).throw(ValueError("offline")))
    assert [n["id"] for n in notes.search_notes(OWNER, "test test 1 2 3")["items"]] == [spoken]
    assert [n["id"] for n in notes.search_notes(OWNER, "Route Sixty-Six")["items"]] == [written]


def test_note_semantic_search_batches_embedding_reads(client, monkeypatch):
    from jarvis import notes
    from sqlalchemy import event
    from jarvis.db import engine

    ids = []
    for title in ("Tent", "Stove", "Lantern"):
        r = client.post("/api/v1/commands", json={"command_id": str(uuid4()), "tool": "note.create",
                                                  "arguments": {"title": title, "content": "Camping gear"}})
        ids.append(r.json()["data"]["id"])
    monkeypatch.setattr(notes, "embeddings", lambda owner, texts, *a, **k: [[1.0, 0.0] for _ in texts])
    with session_scope() as db:
        jobs = list(db.scalars(select(Job.id).where(Job.kind == "embed_note")))
    for job in jobs:
        notes.index_note(job)
    statements = []
    listener = lambda conn, cursor, statement, *a: statements.append(statement)
    event.listen(engine(), "before_cursor_execute", listener)
    try:
        found = notes.search_notes(OWNER, "outdoor equipment")
    finally:
        event.remove(engine(), "before_cursor_execute", listener)
    assert {n["id"] for n in found["items"]} == set(ids)
    assert sum("FROM note_embeddings" in s for s in statements) == 1


def test_search_service_exact_label_and_fts_accept_spoken_numbers(monkeypatch):
    from jarvis import search_index as indexing
    from jarvis import search_service as service
    from test_structure import create

    monkeypatch.setattr(get_settings(), "semantic_search_enabled", True)
    vectors = lambda owner, texts, *a: [[0.0, 1.0] for _ in texts]
    monkeypatch.setattr(indexing, "embeddings", vectors)
    monkeypatch.setattr(service, "embeddings", vectors)
    route = create(title="Route 66")
    mic = create("task", "Mic check", body="test, test, one, two, three")
    with session_scope() as db:
        job = indexing.queue_index(db, OWNER, force=True)
    indexing.index_workspace(job)
    found = service.search(OWNER, OWNER, {"query": "route sixty-six"})
    assert found["resolved"] == ["record:" + route["id"]]
    assert any(c["exact"] and c["key"] == "record:" + route["id"] for c in found["resolutions"])
    found = service.search(OWNER, OWNER, {"query": "test test 123"})
    assert mic["id"] in {m["record"]["id"] for m in found["possible"]["items"]}


def test_index_generation_change_keeps_vectors_for_unchanged_text(monkeypatch):
    from jarvis import search_index as indexing
    from jarvis.search_models import SearchDocument
    from test_structure import create, run

    monkeypatch.setattr(get_settings(), "semantic_search_enabled", True)
    stable = create("task", "Stable task")
    moving = create("task", "Moving task")
    calls = []

    def changed(owner, texts, *a):
        if not calls:
            run("record.update", dict(record_id=moving["id"], expected_revision=moving["revision"],
                                      schema_revision=1, title="Moved task"))
        calls.append(len(texts))
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(indexing, "embeddings", changed)
    with session_scope() as db:
        job = indexing.queue_index(db, OWNER, force=True)
    indexing.index_workspace(job)
    with session_scope() as db:
        stored = {d.target_key: d for d in db.scalars(select(SearchDocument))}
        # Previously the whole paid batch was discarded; unchanged documents now keep vectors.
        assert stored and all(d.vectors for d in stored.values())
        assert "record:" + moving["id"] not in stored
    with session_scope() as db:
        job = db.get(indexing.SearchIndexState, OWNER).job_id
    indexing.index_workspace(job)
    with session_scope() as db:
        final = {d.target_key for d in db.scalars(select(SearchDocument))}
        assert "record:" + stable["id"] in final
        # Only documents without kept vectors are paid for again.
        assert sum(calls[1:]) == len(final) - len(stored)


def note_setup(client):
    r = client.post("/api/v1/commands", json={"command_id": str(uuid4()), "tool": "notelist.setup", "arguments": {}})
    movies = r.json()["data"]["items"][0]
    r = client.post("/api/v1/commands", json={"command_id": str(uuid4()), "tool": "note.create", "arguments": {
        "title": "Recommendations", "content": "Sam recommended Arrival and After Yang. Save these movies to watch."}})
    n = r.json()["data"]
    with session_scope() as db:
        jid = db.scalar(select(Job.id).where(Job.kind == "organize_note", Job.payload["note_id"].as_string() == n["id"]))
    return movies, n, jid


def entry(movies, title, evidence, **kw):
    return {"title": title, "evidence": evidence, "list_ids": [movies["id"]], "confidence": kw.get("confidence", 0.99),
            "save_intent": kw.get("save_intent", True), "existing_note_id": None}


def test_list_coverage_pass_recovers_omitted_item(client, monkeypatch):
    from jarvis.note_list_schema import OrganizationResult
    from jarvis.note_lists import process

    movies, n, jid = note_setup(client)
    calls = []

    def infer(owner, schema, prompt, payload):
        calls.append(payload.get("possibly_omitted"))
        if len(calls) == 1:
            return OrganizationResult(classifications=[], entries=[
                entry(movies, "Arrival", "Sam recommended Arrival"),
                entry(movies, "Dune", "Sam recommended Arrival", confidence=0.4)])
        return OrganizationResult(classifications=[], entries=[entry(movies, "After Yang", "After Yang")])

    monkeypatch.setattr("jarvis.routing.infer", infer)
    process(jid)
    assert calls == [None, ["After Yang"]]
    with session_scope() as db:
        titles = set(db.scalars(select(Note.title).where(Note.content == "")))
        assert titles == {"Arrival", "After Yang"}
        result = db.get(Job, jid).result
        assert result["possible_omissions"] == []
        assert result["rejections"] == [{"title": "Dune", "reason": "low_confidence"}]


def test_list_coverage_records_omission_without_creating_from_rejected_pass(client, monkeypatch):
    from jarvis.note_list_schema import OrganizationResult
    from jarvis.note_lists import process

    movies, n, jid = note_setup(client)
    first = OrganizationResult(classifications=[], entries=[entry(movies, "Arrival", "Sam recommended Arrival")])
    second = OrganizationResult(classifications=[], entries=[entry(movies, "After Yang", "After Yang", save_intent=False)])
    answers = iter([first, second])
    monkeypatch.setattr("jarvis.routing.infer", lambda *a: next(answers))
    process(jid)
    with session_scope() as db:
        assert set(db.scalars(select(Note.title).where(Note.content == ""))) == {"Arrival"}
        result = db.get(Job, jid).result
        assert result["possible_omissions"] == ["After Yang"]
        assert {"title": "After Yang", "reason": "no_save_intent"} in result["rejections"]
