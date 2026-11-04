"""Applying engine results: the guard, admission with exact evidence, same-source dedupe across revisions and
lifecycles, replay, chunk replacement, speaker and mention links that never overwrite human decisions."""

from jarvis.context import apply, decisions, pipeline, sources
from jarvis.context import models as m
from jarvis.context.doubles import rule_based
from jarvis.context.schema import (
    CandidateAssertion,
    Chunk,
    EntityRef,
    EvidenceRef,
    ExtractionResult,
    ExtractRequest,
    LinkProposal,
)
from jarvis.db import session_scope
from sqlalchemy import func, select

OWNER = "davin"


def received(envelope):
    with session_scope() as db:
        return sources.receive(db, OWNER, envelope)


def make_run(db, source_id, revision_id, stage="extract", **values):
    run = m.ContextRun(owner_id=OWNER, stage=stage, status="running", source_id=source_id, revision_id=revision_id,
                       decision_watermark=decisions.watermark(db, OWNER), **values)
    db.add(run)
    db.flush()
    return run


def result_for(db, revision_id, envelope) -> ExtractionResult:
    segments = apply.segments(db, revision_id)
    return rule_based(ExtractRequest(owner_id=OWNER, source_revision_id=revision_id, occurred_at=envelope.occurred_at,
                                     timezone=envelope.timezone, segments=segments))


def admit(envelope, receipt):
    with session_scope() as db:
        run = make_run(db, receipt.source_id, receipt.revision_id)
        source, revision = db.get(m.ContextSource, receipt.source_id), db.get(m.ContextSourceRevision,
                                                                              receipt.revision_id)
        assert apply.guard(db, OWNER, run, source_id=source.id, revision_id=revision.id, watermark=True) is None
        return apply.apply_extraction(db, OWNER, run, source, revision, result_for(db, receipt.revision_id, envelope))


def count(model):
    with session_scope() as db:
        return db.scalar(select(func.count()).select_from(model))


def test_guard_reports_each_reason(envelope):
    receipt = received(envelope)
    with session_scope() as db:
        run = make_run(db, receipt.source_id, receipt.revision_id)
        check = {"source_id": receipt.source_id, "revision_id": receipt.revision_id, "watermark": True}
        assert apply.guard(db, OWNER, run, **check) is None
        decisions.record(db, OWNER, "dismiss", "question", "q-1", command_id=None, principal=OWNER)
        assert apply.guard(db, OWNER, run, **check) == "decisions"
        assert apply.guard(db, OWNER, run, **{**check, "watermark": False}) is None
        pipeline.pause(db, OWNER)
        assert apply.guard(db, OWNER, run, **check) == "paused"
        pipeline.resume(db, OWNER)
        assert apply.guard(db, OWNER, run, **check) == "epoch"  # the pause moved the epoch
    newer = received(envelope.model_copy(update={"private_notes": "A different note.", "external_revision": "rev-2"}))
    with session_scope() as db:
        run = make_run(db, receipt.source_id, receipt.revision_id, epoch=1)
        assert apply.guard(db, OWNER, run, source_id=receipt.source_id, revision_id=receipt.revision_id) == "revision"
        assert apply.guard(db, OWNER, run, source_id=receipt.source_id, revision_id=newer.revision_id) is None
        sources.delete_source(db, OWNER, receipt.source_id)
        assert apply.guard(db, OWNER, run, source_id=receipt.source_id) == "deleted"


def test_admission_has_exact_evidence_and_valid_from_defaults_to_the_meeting(envelope):
    receipt = received(envelope)
    applied = admit(envelope, receipt)
    assert len(applied.admitted) == 3 and applied.attached == [] and applied.rejected == []
    with session_scope() as db:
        for row in db.scalars(select(m.ContextAssertion)):
            assert row.valid_from == envelope.occurred_at and row.valid_precision == "instant"
            assert (row.subject_type, row.subject_id) == ("source", receipt.source_id)
            assert row.current_version == 1 and len(list(db.scalars(select(m.ContextAssertionVersion).where(
                m.ContextAssertionVersion.assertion_id == row.id)))) == 1
            evidence = list(db.scalars(select(m.ContextEvidence).where(m.ContextEvidence.assertion_id == row.id)))
            assert evidence and all(e.revision_id == receipt.revision_id for e in evidence)
            for e in evidence:
                assert db.get(m.ContextSegment, e.segment_id).text[e.char_start:e.char_end] == e.quote


def test_replaying_the_same_result_adds_nothing(envelope):
    receipt = received(envelope)
    first = admit(envelope, receipt)
    before = (count(m.ContextAssertion), count(m.ContextEvidence), count(m.ContextAssertionVersion))
    again = admit(envelope, receipt)
    assert again.admitted == [] and again.attached == [] and again.assertion_ids == first.assertion_ids
    assert (count(m.ContextAssertion), count(m.ContextEvidence), count(m.ContextAssertionVersion)) == before


def test_inexact_evidence_is_refused(envelope):
    receipt = received(envelope)
    bad = CandidateAssertion(kind="fact", statement="Invented.", evidence=[
        EvidenceRef(segment_seq=1, quote="The pilot covers 41", char_start=0, char_end=19)])
    other = CandidateAssertion(kind="fact", statement="Elsewhere.", evidence=[
        EvidenceRef(segment_seq=1, quote="The pilot", char_start=0, char_end=9, source_revision_id="other")])
    with session_scope() as db:
        run = make_run(db, receipt.source_id, receipt.revision_id)
        out = apply.apply_extraction(db, OWNER, run, db.get(m.ContextSource, receipt.source_id),
                                     db.get(m.ContextSourceRevision, receipt.revision_id),
                                     ExtractionResult(assertions=[bad, other]))
    assert out.assertion_ids == [None, None] and [r["reason"] for r in out.rejected] == ["inexact_evidence"] * 2
    assert count(m.ContextAssertion) == 0


def test_a_revised_meeting_never_brings_back_retracted_or_superseded_statements(envelope):
    first = received(envelope)
    applied = admit(envelope, first)
    with session_scope() as db:
        rows = {r.statement: r for r in db.scalars(select(m.ContextAssertion))}
        decision_row = rows["we launch the pilot on Postgres with pgvector."]
        decisions.wrong(db, OWNER, decision_row.id, 1, command_id=None, principal=OWNER)
        budget = rows["I'll send the revised budget by 2026-11-20."]
        budget.lifecycle = "superseded"
    revised = envelope.model_copy(update={"private_notes": "Decision: we add a second wave.",
                                          "external_revision": "rev-2"})
    second = received(revised)
    out = admit(revised, second)
    assert set(applied.admitted) <= set(out.assertion_ids)
    assert len(out.admitted) == 1 and len(out.attached) == 3
    with session_scope() as db:
        lifecycles = {r.statement: r.lifecycle for r in db.scalars(select(m.ContextAssertion))}
        assert lifecycles["we launch the pilot on Postgres with pgvector."] == "retracted"
        assert lifecycles["I'll send the revised budget by 2026-11-20."] == "superseded"
        assert lifecycles["we add a second wave."] == "current"
        revisions = set(db.scalars(select(m.ContextEvidence.revision_id).where(
            m.ContextEvidence.assertion_id == decision_row.id)))
        assert revisions == {first.revision_id, second.revision_id}


def test_replace_chunks_is_idempotent_and_keeps_embeddings(envelope):
    receipt = received(envelope)
    chunks = [Chunk(seq=0, segment_start=0, segment_end=5, token_count=10, text="all of it")]
    with session_scope() as db:
        assert apply.replace_chunks(db, OWNER, receipt.revision_id, chunks)
        row = db.scalar(select(m.ContextChunk).where(m.ContextChunk.revision_id == receipt.revision_id))
        row.embedding, row.embedding_model = [0.5] * 512, "test"
    with session_scope() as db:
        assert not apply.replace_chunks(db, OWNER, receipt.revision_id, chunks)
        row = db.scalar(select(m.ContextChunk).where(m.ContextChunk.revision_id == receipt.revision_id))
        assert (row.text, row.embedding_model) == ("all of it", "test")


def test_speakers_and_links_never_overwrite_human_decisions(envelope):
    receipt = received(envelope)
    applied = admit(envelope, receipt)
    target = applied.admitted[0]
    proposal = LinkProposal(mention="Priya", candidate=EntityRef(entity_type="actor", entity_id="act-1"),
                            status="inferred")
    with session_scope() as db:
        rows = apply.segment_rows(db, receipt.revision_id)
        rows[1].speaker_link_status = "confirmed"
        rows[1].speaker_actor_id = "act-human"
        changed = apply.apply_speakers(db, OWNER, receipt.revision_id, [proposal], "run-1")
        assert changed == 1  # segment 3 (also Priya); segment 1 was decided by a person
        assert [(r.seq, r.speaker_actor_id) for r in apply.segment_rows(db, receipt.revision_id)
                if r.speaker_label == "Priya"] == [(1, "act-human"), (3, "act-1")]
        assert apply.apply_links(db, OWNER, [apply.RoleLink(target, "responsible", proposal)], None) == \
            {"created": 1, "updated": 0, "kept": 0}
        assert apply.apply_links(db, OWNER, [apply.RoleLink(target, "responsible", proposal)], None)["kept"] == 1
        link = db.scalar(select(m.ContextEntityLink))
        link.status = "rejected"
        moved = proposal.model_copy(update={"candidate": EntityRef(entity_type="actor", entity_id="act-2")})
        assert apply.apply_links(db, OWNER, [apply.RoleLink(target, "responsible", moved)], None)["kept"] == 1
        assert (link.target_id, link.status) == ("act-1", "rejected")


def test_work_keys_come_from_update_and_link_actions():
    from jarvis.context.schema import ProposedAction
    ev = [EvidenceRef(segment_seq=0, quote="x", char_start=0, char_end=1)]
    result = ExtractionResult(
        assertions=[CandidateAssertion(kind="commitment", statement="Do x.", evidence=ev,
                                       responsible=[EntityRef(entity_type="actor", mention="Priya")])],
        actions=[ProposedAction(kind="update", title="Do x.", evidence=ev, assertion_indexes=[0],
                                target=EntityRef(entity_type="task", entity_id="task-9"))])
    assert apply.work_keys(result) == {0: "task:task-9"}
    assert apply.work_keys(result.model_copy(update={"actions": []})) == {}
