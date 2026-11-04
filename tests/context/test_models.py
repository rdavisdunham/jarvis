"""Context tables (jarvis/context/models.py) and their registration points: worker queue, scopes, cost labels."""

from datetime import timedelta

import pytest
from jarvis import worker
from jarvis.context import models as m
from jarvis.context import settings as context_settings
from jarvis.db import session_scope
from jarvis.domain import enqueue_job
from jarvis.models import Base, now
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError, StatementError

OWNER = "davin"


def source_graph(db):
    """One meeting through every context table; returns the rows by name."""
    connection = m.ContextConnection(owner_id=OWNER, connector="granola", external_account="davin@example.test",
                                     credential_ciphertext="gAAAA-ciphertext")
    db.add(connection)
    db.flush()
    source = m.ContextSource(owner_id=OWNER, connection_id=connection.id, connector="granola", external_id="note-1",
                             title="Pricing review", current_revision=1)
    db.add(source)
    db.flush()
    revision = m.ContextSourceRevision(owner_id=OWNER, source_id=source.id, revision=1, content_hash="a" * 64,
                                       occurred_at=now(), participants=[{"email": "sam@example.test"}])
    db.add(revision)
    db.flush()
    segments = [
        m.ContextSegment(owner_id=OWNER, revision_id=revision.id, seq=0, speaker_label="Sam",
                         text="We decided to raise the enterprise price next quarter.", char_start=0, char_end=55),
        m.ContextSegment(owner_id=OWNER, revision_id=revision.id, seq=1, kind="private_note",
                         text="Sam owns the pricing memo.", char_start=56, char_end=82),
    ]
    db.add_all(segments)
    chunk = m.ContextChunk(owner_id=OWNER, revision_id=revision.id, seq=0, segment_start=0, segment_end=1,
                           token_count=20, text=" ".join(s.text for s in segments))
    run = m.ContextRun(owner_id=OWNER, stage="extract", source_id=source.id, revision_id=revision.id,
                       model="gpt-5.6-luna", prompt_version="extract-v1", cost_usd=0.0125, epoch=0)
    db.add_all([chunk, run, m.ContextCursor(owner_id=OWNER, connection_id=connection.id, name="notes",
                                            value={"updated_after": "2026-11-04"})])
    db.flush()
    decision = m.ContextDecision(owner_id=OWNER, target_type="assertion", target_id="pending", operation="confirm",
                                 origin="human", principal="davin")
    assertion = m.ContextAssertion(owner_id=OWNER, kind="decision", statement="Raise the enterprise price next quarter.",
                                   subject_type="project", subject_id="project-1", predicate="selected_option",
                                   valid_precision="quarter", run_id=run.id)
    db.add_all([decision, assertion])
    db.flush()
    db.add_all([
        m.ContextAssertionVersion(owner_id=OWNER, assertion_id=assertion.id, version=1, statement=assertion.statement,
                                  lifecycle="current", decision_id=decision.id, run_id=run.id),
        m.ContextEvidence(owner_id=OWNER, assertion_id=assertion.id, assertion_version=1, source_id=source.id,
                          revision_id=revision.id, segment_id=segments[0].id, quote=segments[0].text, char_start=0,
                          char_end=55),
        m.ContextEntityLink(owner_id=OWNER, assertion_id=assertion.id, subject_type="assertion",
                            subject_id=assertion.id, target_type="actor", target_id="actor-sam", mention="Sam",
                            role="responsible", status="inferred", score=0.91, margin=0.4, run_id=run.id),
        m.ContextActorIdentity(owner_id=OWNER, actor_id="actor-sam", namespace="email",
                               value="Sam@Example.test", normalized="sam@example.test"),
        m.ContextActorMerge(owner_id=OWNER, winner_actor_id="actor-sam", loser_actor_id="actor-sam-2",
                            snapshot={"tasks": ["t1"]}, decision_id=decision.id),
    ])
    proposal = m.ContextProposal(owner_id=OWNER, source_id=source.id, revision_id=revision.id, kind="create",
                                 fingerprint="f" * 64, title="Draft the pricing memo", responsible="actor-sam",
                                 assertion_ids=[assertion.id])
    db.add(proposal)
    db.flush()
    db.add_all([
        m.ContextPublication(owner_id=OWNER, proposal_id=proposal.id, proposal_revision=1, destination="eridani",
                             principal="davin", operation_id="op-" + proposal.id),
        m.ContextBrief(owner_id=OWNER, entity_type="project", entity_id="project-1", fingerprint="b" * 64,
                       sections=[{"kind": "current_state", "claims": [assertion.id]}], run_id=run.id),
        m.ContextOwnerState(owner_id=OWNER),
    ])
    db.flush()
    return {"connection": connection, "source": source, "revision": revision, "segments": segments, "chunk": chunk,
            "run": run, "assertion": assertion, "proposal": proposal}


def test_every_context_table_is_registered_and_takes_a_row():
    tables = {t for t in Base.metadata.tables if t.startswith("context_")}
    assert len(tables) == 18
    with session_scope() as db:
        rows = source_graph(db)
        ids = {k: v.id for k, v in rows.items() if hasattr(v, "id")}
    with session_scope() as db:
        for table in tables:
            assert db.scalar(text(f"SELECT count(*) FROM {table}")) >= 1, table
        source = db.get(m.ContextSource, ids["source"])
        assert (source.status, source.availability, source.kind) == ("received", "available", "meeting")
        run = db.get(m.ContextRun, ids["run"])
        assert (run.status, run.engine, run.coverage, run.stage_results, float(run.cost_usd)) == (
            "queued", "native", {}, {}, 0.0125)
        assertion = db.get(m.ContextAssertion, ids["assertion"])
        assert (assertion.lifecycle, assertion.modality, assertion.explicit, assertion.responsible) == (
            "current", "asserted", True, [])
        state = db.get(m.ContextOwnerState, OWNER)
        assert (state.epoch, state.paused, state.settings) == (0, False, {})


def test_server_defaults_cover_rows_written_without_the_orm():
    """An old writer during a rolling deploy inserts with SQL only; defaults must fill every non-null column."""
    with session_scope() as db:
        db.execute(text("INSERT INTO context_owner_state (owner_id, created_at, updated_at) VALUES ('x', now(), now())"))
        db.execute(text("INSERT INTO context_sources (id, owner_id, connector, external_id, created_at, updated_at) "
                        "VALUES ('s', 'x', 'api', 'e', now(), now())"))
        row = db.execute(text("SELECT status, current_revision, title FROM context_sources WHERE id='s'")).one()
    assert tuple(row) == ("received", 0, "")


@pytest.mark.parametrize("make", [
    lambda r: m.ContextSource(owner_id=OWNER, connector="granola", external_id="note-1"),
    lambda r: m.ContextSourceRevision(owner_id=OWNER, source_id=r["source"].id, revision=1, content_hash="b" * 64),
    lambda r: m.ContextSegment(owner_id=OWNER, revision_id=r["revision"].id, seq=0, text="x", char_start=0, char_end=1),
    lambda r: m.ContextChunk(owner_id=OWNER, revision_id=r["revision"].id, seq=0, segment_start=0, segment_end=0,
                             text="x"),
    lambda r: m.ContextCursor(owner_id=OWNER, connection_id=r["connection"].id, name="notes"),
    lambda r: m.ContextProposal(owner_id=OWNER, source_id=r["source"].id, kind="create", fingerprint="f" * 64,
                                title="again"),
    lambda r: m.ContextBrief(owner_id=OWNER, entity_type="project", entity_id="project-1", fingerprint="c" * 64),
    lambda r: m.ContextActorIdentity(owner_id=OWNER, actor_id="actor-other", namespace="email", value="sam@example.test",
                                     normalized="sam@example.test"),
], ids=["source", "revision", "segment", "chunk", "cursor", "proposal", "brief-version", "strong-identity"])
def test_uniqueness_holds(make):
    with session_scope() as db:
        rows = source_graph(db)
        with pytest.raises(IntegrityError), db.begin_nested():
            db.add(make(rows))
            db.flush()


def test_ambiguous_aliases_may_name_several_actors_but_not_twice_and_null_scope_is_one_value():
    def identity(actor, namespace="alias", value="alex", scope=None):
        return m.ContextActorIdentity(owner_id=OWNER, actor_id=actor, namespace=namespace, value=value,
                                      normalized=value, source_scope=scope)

    with session_scope() as db:
        db.add_all([identity("a1"), identity("a2"), identity("a1", "calendar_name"), identity("a2", "calendar_name"),
                    identity("a1", "granola_speaker", "speaker a", "source-1"),
                    identity("a2", "granola_speaker", "speaker a", "source-2")])
        db.flush()
        for duplicate in (identity("a1"), identity("a3", "granola_speaker", "speaker a", "source-1")):
            with pytest.raises(IntegrityError), db.begin_nested():
                db.add(duplicate)
                db.flush()


def test_sourceless_proposals_dedupe_by_fingerprint():
    with session_scope() as db:
        db.add(m.ContextProposal(owner_id=OWNER, kind="suggestion", fingerprint="l" * 64, title="Lint"))
        db.flush()
        with pytest.raises(IntegrityError), db.begin_nested():
            db.add(m.ContextProposal(owner_id=OWNER, kind="suggestion", fingerprint="l" * 64, title="Lint again"))
            db.flush()


def test_tsvector_is_generated_and_searchable():
    with session_scope() as db:
        rows = source_graph(db)
        query = func.plainto_tsquery("english", "memo")
        hits = db.scalars(select(m.ContextSegment.seq).where(m.ContextSegment.tsv.op("@@")(query))).all()
        assert hits == [1]
        assert db.scalar(select(func.count()).select_from(m.ContextChunk).where(m.ContextChunk.tsv.op("@@")(query))) == 1
        decided = func.plainto_tsquery("english", "enterprise prices")
        assert db.scalar(select(m.ContextAssertion.id).where(m.ContextAssertion.tsv.op("@@")(decided))) == rows["assertion"].id
        segment = rows["segments"][0]
        segment.text = "Nothing about cost here."
        db.flush()
        db.refresh(segment)
        assert "enterpris" not in segment.tsv and "cost" in segment.tsv


def test_512_dimension_vectors_round_trip_and_rank_by_cosine_distance():
    def unit(i):
        return [1.0 if j == i else 0.0 for j in range(512)]

    with session_scope() as db:
        rows = source_graph(db)
        rows["chunk"].embedding, rows["chunk"].embedding_model = unit(0), "text-embedding-3-small@512"
        rows["assertion"].embedding = [0.6 if j == 0 else 0.8 if j == 1 else 0.0 for j in range(512)]
        other = m.ContextChunk(owner_id=OWNER, revision_id=rows["revision"].id, seq=1, segment_start=1, segment_end=1,
                               text="other", embedding=unit(1))
        db.add(other)
        db.flush()
        db.expire_all()
        stored = db.get(m.ContextChunk, rows["chunk"].id).embedding
        assert len(stored) == 512 and list(stored[:2]) == [1.0, 0.0]
        query = unit(0)
        distance = m.ContextChunk.embedding.cosine_distance(query)
        ranked = db.execute(select(m.ContextChunk.seq, distance).where(m.ContextChunk.owner_id == OWNER)
                            .order_by(distance)).all()
        assert [r.seq for r in ranked] == [0, 1] and ranked[0][1] == pytest.approx(0.0) and ranked[1][1] == pytest.approx(1.0)
        raw = db.scalar(text("SELECT embedding <=> CAST(:q AS vector) FROM context_assertions WHERE id = :id"),
                        {"q": str(query), "id": rows["assertion"].id})
        assert raw == pytest.approx(0.4)
        with pytest.raises(StatementError, match="512 dimensions"), db.begin_nested():
            other.embedding = [1.0] * 3
            db.flush()


def test_settings_record_the_pgvector_decision_and_defaults():
    current = context_settings.current()
    assert (current.embedding_storage, current.embedding_dimensions, m.DIMENSIONS) == ("pgvector", 512, 512)
    assert (current.enabled, current.queue_concurrency, current.source_budget_usd, current.model_timeout_s,
            current.poll_minutes, current.granola_enabled) == (False, 2, 1.0, 90, 5, False)
    assert current.max_source_chars >= 100_000 and "granola_api_key" not in repr(current)
    assert context_settings.enabled() is False


def test_context_jobs_route_to_their_queue_and_pipeline(monkeypatch):
    with session_scope() as db:
        context_job = enqueue_job(db, OWNER, "context_extract", {}).id
        other_job = enqueue_job(db, OWNER, "reminder", {}).id
    sent = {}

    class Client:
        def enqueue(self, options, job_id):
            sent[job_id] = options["queue_name"]

    worker.dispatch_outbox(Client())
    assert sent == {context_job: "jarvis-context", other_job: "jarvis"}
    with pytest.raises(NotImplementedError, match="B6"):
        worker.perform_job(context_job)
    from jarvis.context import pipeline

    monkeypatch.setattr(pipeline, "perform", lambda job_id: ("pipeline", job_id))
    assert worker.perform_job(context_job) == ("pipeline", context_job)


def test_cost_labels_exist_for_every_context_stage():
    from jarvis.cost_features import LABELS, feature

    for name in ("context_extraction", "context_linking", "context_reconciliation", "context_embeddings",
                 "context_synthesis"):
        assert LABELS[name]
        feature(name)


def test_context_scopes_are_personal_only_and_sources_write_implies_no_read(client):
    from test_accounts import client_for, shared

    def mint(who, scopes):
        return who.post("/api/v1/bot-keys", json={"name": "ctx", "scopes": scopes, "expires_in_days": 30})

    assert {"context:read", "context:write", "sources:write"} <= set(client.get("/api/v1/bot-keys").json()["scopes"])
    assert mint(client, ["sources:write"]).json()["credential"]["scopes"] == ["sources:write"]
    assert mint(client, ["context:write"]).json()["credential"]["scopes"] == ["context:read", "context:write"]
    guest = client_for("guest")
    shared(client, guest)
    for scopes in (["context:read"], ["context:write"], ["sources:write"], ["tasks:read", "context:read"]):
        refused = mint(client, scopes)
        assert refused.status_code == 403 and refused.json()["error"]["code"] == "PERSONAL_WORKSPACE", scopes
    assert mint(client, ["tasks:read"]).status_code == 201


def test_runs_record_epoch_and_watermark_for_staleness():
    with session_scope() as db:
        rows = source_graph(db)
        run = rows["run"]
        run.decision_watermark, run.status, run.finished_at = now() - timedelta(minutes=1), "stale", now()
        db.get(m.ContextOwnerState, OWNER).epoch = 1
        db.flush()
        stale = db.scalars(select(m.ContextRun.id).join(m.ContextOwnerState, m.ContextOwnerState.owner_id == m.ContextRun.owner_id)
                           .where(m.ContextRun.epoch < m.ContextOwnerState.epoch)).all()
        assert stale == [run.id]
