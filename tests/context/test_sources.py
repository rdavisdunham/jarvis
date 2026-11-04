"""Source lifecycle: idempotent receipt, revisions, metadata refresh, limits, chunking, deletion and tombstones."""

from datetime import UTC, datetime

import pytest
from jarvis.config import get_settings
from jarvis.context import models as m
from jarvis.context import sources
from jarvis.context.schema import Segment, SourceEnvelope
from jarvis.db import session_scope
from jarvis.domain import DomainError
from jarvis.models import Job
from sqlalchemy import func, select

OWNER = "davin"


def receive(envelope, owner=OWNER, **values):
    with session_scope() as db:
        return sources.receive(db, owner, envelope, **values)


def jobs(kind="context_ingest"):
    with session_scope() as db:
        return list(db.scalars(select(Job).where(Job.kind == kind).order_by(Job.created_at)))


def segments_of(revision_id):
    with session_scope() as db:
        return [(s.seq, s.kind, s.speaker_label, s.speaker_attribution, s.text, s.char_start, s.char_end)
                for s in db.scalars(select(m.ContextSegment).where(m.ContextSegment.revision_id == revision_id)
                                    .order_by(m.ContextSegment.seq))]


def changed(envelope, **values):
    return envelope.model_copy(update=values)


def test_receipt_persists_revision_segments_chunks_and_one_ingest_job(envelope):
    result = receive(envelope)
    assert (result.created, result.revised, result.unchanged, result.revision, result.status) == (True, True, False, 1, "queued")
    rows = segments_of(result.revision_id)
    assert [r[1] for r in rows] == ["transcript"] * 5 + ["private_note"]
    assert rows[-1][3] == "owner" and rows[-1][4] == "Check the clinic count against the contract."
    assert rows[1][2:4] == ("Priya", "other")
    text = "\n".join(r[4] for r in rows)
    assert all(text[start:end] == body for *_, body, start, end in rows)
    with session_scope() as db:
        revision = db.get(m.ContextSourceRevision, result.revision_id)
        assert revision.content_hash == __import__("hashlib").sha256(text.encode()).hexdigest()
        assert revision.change_kind == "content" and revision.calendar_event_id is None
        assert revision.participants[0]["email"] == "davin@example.com"
        chunks = list(db.scalars(select(m.ContextChunk).where(m.ContextChunk.revision_id == result.revision_id)
                                 .order_by(m.ContextChunk.seq)))
        assert [(c.segment_start, c.segment_end) for c in chunks] == [(0, 4), (5, 5)]
        assert chunks[0].text.startswith("Davin: Thanks for joining.")
    [job] = jobs()
    assert job.payload == {"source_id": result.source_id, "revision_id": result.revision_id}
    assert result.job_id == job.id


def test_repeated_delivery_is_one_source_without_new_work(envelope):
    first = receive(envelope)
    again = receive(envelope)
    assert (again.source_id, again.revision_id, again.change, again.job_id) == (first.source_id, first.revision_id, "none", None)
    with session_scope() as db:
        assert db.scalar(select(func.count()).select_from(m.ContextSource)) == 1
        assert db.scalar(select(func.count()).select_from(m.ContextSourceRevision)) == 1
    assert len(jobs()) == 1


def test_summary_or_metadata_change_refreshes_display_without_reextraction(envelope):
    first = receive(envelope)
    result = receive(changed(envelope, summary="Upstream summary", title="Pilot scope (final)", external_revision="rev-2"))
    assert (result.change, result.revision_id, result.job_id) == ("metadata", first.revision_id, None)
    with session_scope() as db:
        revision = db.get(m.ContextSourceRevision, first.revision_id)
        assert (revision.summary, revision.external_revision, revision.change_kind) == ("Upstream summary", "rev-2", "content")
        assert "metadata_updated_at" in revision.payload
        assert db.get(m.ContextSource, first.source_id).title == "Pilot scope (final)"
    assert len(jobs()) == 1


def test_changed_transcript_creates_a_new_immutable_revision(envelope):
    first = receive(envelope)
    lines = list(envelope.segments) + [Segment(seq=9, speaker_label="Marco", text="One more thing.")]
    result = receive(changed(envelope, segments=lines, external_revision="rev-2"))
    assert (result.change, result.revision, result.revised) == ("content", 2, True)
    assert len(segments_of(first.revision_id)) == 6 and len(segments_of(result.revision_id)) == 7
    assert segments_of(result.revision_id)[5][1:5] == ("transcript", "Marco", None, "One more thing.")
    assert [j.payload["revision_id"] for j in jobs()] == [first.revision_id, result.revision_id]


def test_private_note_change_is_a_content_change(envelope):
    receive(envelope)
    assert receive(changed(envelope, private_notes="Different note.\n\nSecond paragraph.")).change == "content"


def test_text_envelopes_are_split_into_segments():
    envelope = SourceEnvelope(connector="upload", external_id="f1", occurred_at=datetime(2026, 11, 3, tzinfo=UTC),
                              text="Davin: One.\n\nPriya: Two.", external_revision="r1")
    rows = segments_of(receive(envelope).revision_id)
    assert [(r[2], r[4]) for r in rows] == [("Davin", "One."), ("Priya", "Two.")]


def test_secrets_are_redacted_before_storage(envelope):
    lines = [Segment(seq=0, speaker_label="Davin", text="The key is sk-proj-abcdefghijklmnopqrstuvwxyz123")]
    rows = segments_of(receive(changed(envelope, segments=lines, private_notes=None)).revision_id)
    assert rows[0][4] == "The key is [redacted]"


def test_limits_are_enforced_with_explicit_errors(envelope, monkeypatch):
    monkeypatch.setenv("JARVIS_CONTEXT_MAX_SOURCE_CHARS", "1000")
    get_settings.cache_clear()
    try:
        with pytest.raises(DomainError) as error:
            receive(changed(envelope, segments=[Segment(seq=0, text="x" * 1001)]))
        assert (error.value.code, error.value.status) == ("SOURCE_TOO_LARGE", 413)
        assert error.value.data["max_source_chars"] == 1000
    finally:
        monkeypatch.delenv("JARVIS_CONTEXT_MAX_SOURCE_CHARS")
        get_settings.cache_clear()
    monkeypatch.setattr(sources, "MAX_SEGMENTS", 3)
    with pytest.raises(DomainError, match="more than 3 segments"):
        receive(envelope)
    assert jobs() == []


def test_connection_must_belong_to_the_owner(envelope):
    with session_scope() as db:
        connection = m.ContextConnection(owner_id="someone-else", connector="granola", external_account="x@y.z")
        db.add(connection)
        db.flush()
        other = connection.id
    with pytest.raises(DomainError) as error:
        receive(envelope, connection_id=other)
    assert error.value.status == 404


def test_chunker_packs_by_budget_and_splits_only_oversized_segments():
    segments = [Segment(seq=0, text="a" * 40), Segment(seq=1, text="b" * 40), Segment(seq=2, text="c" * 200),
                Segment(seq=3, text="d" * 10), Segment(seq=4, kind="private_note", attribution="owner", text="e")]
    chunks = sources._chunk_segments(segments, budget=25)
    assert [(c.segment_start, c.segment_end) for c in chunks] == [(0, 1), (2, 2), (2, 2), (3, 3), (4, 4)]
    assert chunks[1].text + chunks[2].text == "c" * 200 and chunks[0].token_count == 21


def test_api_ids_are_namespaced_by_label():
    assert sources.api_external_id("Zoom", "123") == "zoom:123"
    for label, external_id in (("bad label", "1"), ("", "1"), ("ok", ""), ("ok", "a\nb")):
        with pytest.raises(DomainError):
            sources.api_external_id(label, external_id)


def test_reads_hide_text_in_lists_and_exclude_deleted(envelope):
    result = receive(envelope)
    with session_scope() as db:
        listing = sources.list_sources(db, OWNER)
        [item] = listing["items"]
        assert "segments" not in item and item["segment_count"] == 6 and item["status"] == "queued"
        detail = sources.get_source(db, OWNER, result.source_id)
        assert detail["segments"][0]["text"].startswith("Thanks") and detail["revisions"][0]["revision"] == 1
        db.add(m.ContextRun(owner_id=OWNER, stage="extract", status="failed", source_id=result.source_id,
                            revision_id=result.revision_id, error="model timed out", coverage={"0": "failed"}))
        db.flush()
        item = sources.list_sources(db, OWNER)["items"][0]
        assert item["last_error"]["message"] == "model timed out" and item["last_error"]["retryable"] is False
        assert item["processing"]["coverage"] == {"0": "failed"}
        with pytest.raises(DomainError):
            sources.get_source(db, "someone-else", result.source_id)


def seed_evidence_and_brief(db, result):
    segment = db.scalar(select(m.ContextSegment).where(m.ContextSegment.revision_id == result.revision_id,
                                                       m.ContextSegment.seq == 2))
    assertion = m.ContextAssertion(owner_id=OWNER, kind="decision", statement="Launch on Postgres.",
                                   subject_type="project", subject_id="p1", predicate="decided")
    db.add(assertion)
    db.flush()
    db.add_all([
        m.ContextEvidence(owner_id=OWNER, assertion_id=assertion.id, assertion_version=1, source_id=result.source_id,
                          revision_id=result.revision_id, segment_id=segment.id, quote="we launch the pilot"),
        m.ContextBrief(owner_id=OWNER, entity_type="project", entity_id="p1", fingerprint="f" * 64,
                       dependencies={"sources": [result.source_id]}),
        m.ContextBrief(owner_id=OWNER, entity_type="project", entity_id="p2", fingerprint="e" * 64),
    ])
    db.execute(m.ContextChunk.__table__.update().values(embedding=[0.1] * 512, embedding_model="test"))


def test_deletion_excludes_now_and_purge_blanks_content_but_keeps_a_tombstone(envelope, monkeypatch):
    result = receive(envelope)
    with session_scope() as db:
        seed_evidence_and_brief(db, result)
        deleted = sources.delete_source(db, OWNER, result.source_id)
        assert deleted["status"] == "deleted" and deleted["job_id"]
        assert sources.list_sources(db, OWNER)["items"] == []
        with pytest.raises(DomainError):
            sources.get_source(db, OWNER, result.source_id)
        assert sources.delete_source(db, OWNER, result.source_id)["job_id"] is None  # idempotent
    [job] = jobs("context_delete")
    assert job.payload == {"source_id": result.source_id, "through_revision": 1}
    propagated = []
    monkeypatch.setattr(sources, "propagate_source_deletion", lambda db, owner, sid: propagated.append((owner, sid)))
    with session_scope() as db:
        summary = sources.purge_source(db, OWNER, result.source_id, through_revision=1)
    assert summary["segments"] == 6 and summary["evidence"] == 1 and summary["stale_briefs"] == 1
    assert propagated == [(OWNER, result.source_id)]
    with session_scope() as db:
        assert set(db.scalars(select(m.ContextSegment.text))) == {""}
        assert set(db.scalars(select(m.ContextSegment.speaker_label))) == {None}
        assert db.scalar(select(func.count()).select_from(m.ContextSegment).where(
            m.ContextSegment.tsv.op("@@")(func.to_tsquery("english", "pilot")))) == 0
        assert set(db.execute(select(m.ContextChunk.text, m.ContextChunk.embedding)).all()) == {("", None)}
        assert db.scalar(select(m.ContextEvidence.quote)) == ""
        assert {b.entity_id: b.stale for b in db.scalars(select(m.ContextBrief))} == {"p1": True, "p2": False}
        source = db.get(m.ContextSource, result.source_id)
        revision = db.get(m.ContextSourceRevision, result.revision_id)
        assert (source.connector, source.external_id, source.title) == ("api", "meeting-001", "")
        assert revision.content_hash and revision.payload == {"purged": True} and revision.participants == []
    again = receive(envelope)
    assert (again.change, again.status, again.job_id) == ("deleted", "deleted", None)
    with session_scope() as db:
        assert db.scalar(select(func.count()).select_from(m.ContextSourceRevision)) == 1
        assert sources.list_sources(db, OWNER)["items"] == []


def test_new_content_after_deletion_restores_the_source_and_purge_spares_it(envelope):
    first = receive(envelope)
    with session_scope() as db:
        sources.delete_source(db, OWNER, first.source_id)
    restored = receive(changed(envelope, segments=[Segment(seq=0, speaker_label="Davin", text="A new take.")]))
    assert (restored.change, restored.revision, restored.status) == ("content", 2, "queued")
    with session_scope() as db:
        sources.purge_source(db, OWNER, first.source_id, through_revision=1)
        assert db.get(m.ContextSource, first.source_id).deleted_at is None
    assert segments_of(restored.revision_id)[0][4] == "A new take."
    assert {r[4] for r in segments_of(first.revision_id)} == {""}
