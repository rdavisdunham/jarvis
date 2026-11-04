"""The context job pipeline: stage chaining through `domain.enqueue_job`, replay, crash and retry without repeated
model calls, staleness (a change during an in-flight model call, pause, moved decisions), budget continuation,
embedding failure, Questions, scheduling. Model and embedding calls are mocked; jobs run through `perform` as the
worker would."""

import json
from datetime import timedelta
from pathlib import Path

import httpx
import pytest
from jarvis import bot_access, memory_learning
from jarvis.config import get_settings
from jarvis.context import apply, decisions, pipeline, sources
from jarvis.context import models as m
from jarvis.context.native import embed as embedding
from jarvis.context.schema import Segment, SourceEnvelope
from jarvis.db import session_scope
from jarvis.domain import DomainError, enqueue_job, execute
from jarvis.models import Actor, Job, now
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert

from .fixtures.extraction import long_meeting

OWNER = "davin"
ACTION_PASS = ["context_actions", "context_actions_verify"]
RECORDED = json.loads((Path(__file__).parent / "fixtures/extraction/pilot_meeting.json").read_text())
STAGES = ["context_ingest", "context_extract", "context_link", "context_reconcile", "context_embed"]
BRIEF_FORMAT = "context_brief"


# ---------------------------------------------------------------- helpers shared with test_end_to_end


def drain(owner=OWNER, limit=500) -> list[str]:
    """Run queued context jobs oldest first, as the worker would, until none are left. Returns the kinds run."""
    kinds = []
    for _ in range(limit):
        with session_scope() as db:
            job = db.scalar(select(Job).where(Job.owner_id == owner, Job.kind.like("context\\_%"),
                                              Job.status.in_(("queued", "retrying")))
                            .order_by(Job.created_at, Job.id).limit(1))
            if job is None:
                return kinds
            job_id, kind = job.id, job.kind
        pipeline.perform(job_id)
        kinds.append(kind)
    raise AssertionError("context jobs did not drain")


def use_doubles(owner=OWNER):
    with session_scope() as db:
        db.execute(insert(m.ContextOwnerState).values(owner_id=owner, settings={"engine": "double"})
                   .on_conflict_do_update(index_elements=["owner_id"], set_={"settings": {"engine": "double"}}))


def receive(envelope, owner=OWNER):
    with session_scope() as db:
        return sources.receive(db, owner, envelope)


def jobs(kind=None, owner=OWNER):
    with session_scope() as db:
        query = select(Job).where(Job.owner_id == owner, Job.kind.like("context\\_%"))
        if kind:
            query = query.where(Job.kind == kind)
        return [(j.kind, j.status, j.payload, j.result) for j in db.scalars(query.order_by(Job.created_at, Job.id))]


def runs(stage=None):
    with session_scope() as db:
        query = select(m.ContextRun).order_by(m.ContextRun.created_at, m.ContextRun.id)
        if stage:
            query = query.where(m.ContextRun.stage == stage)
        return [(r.stage, r.status, r.error, dict(r.stage_results or {}), dict(r.coverage or {}))
                for r in db.scalars(query)]


def assertions():
    with session_scope() as db:
        return {r.statement: r.lifecycle for r in db.scalars(select(m.ContextAssertion))}


def state() -> tuple:
    """Everything the pipeline writes, for no-change comparisons."""
    with session_scope() as db:
        return tuple(db.scalar(select(func.count()).select_from(t)) for t in (
            m.ContextAssertion, m.ContextAssertionVersion, m.ContextEvidence, m.ContextEntityLink, m.ContextChunk,
            m.ContextActorIdentity, m.ContextRun, Job, Actor))


def vector(owner, texts, timeout=20):
    return [[1.0] + [0.0] * 511 for _ in texts]


@pytest.fixture
def vectors(monkeypatch):
    monkeypatch.setattr(memory_learning, "embeddings", vector)


@pytest.fixture
def provider(monkeypatch):
    """install(handler, briefs=None): answer the provider's HTTP calls with handler(body) -> (status, json).
    Returns the list of extraction and reconciliation format names sent. Brief synthesis calls go to `briefs`
    when given, else they are refused (the brief keeps its deterministic sections); either way they are recorded
    in `install.briefs`, not in the returned list."""
    original = httpx.Client
    monkeypatch.setattr(get_settings(), "openai_api_key", "synthetic")
    sent, synthesis = [], []

    def install(handler, briefs=None):
        def respond(request):
            body = json.loads(request.content)
            name = body["text"]["format"]["name"]
            if name == BRIEF_FORMAT:
                synthesis.append(name)
                if briefs is None:
                    return httpx.Response(400, json={"error": {"message": "not scripted"}})
                status, data = briefs(body)
                return httpx.Response(status, json=data)
            sent.append(name)
            status, data = handler(body)
            return httpx.Response(status, json=data)
        def client(**kw):  # httpx is shared: clients built with their own transport (Granola's) keep it
            return original(**kw) if kw.get("transport") else original(transport=httpx.MockTransport(respond))
        monkeypatch.setattr(memory_learning.httpx, "Client", client)
        return sent
    install.briefs = synthesis
    return install


def recorded(body):
    text = json.dumps(RECORDED[body["text"]["format"]["name"]])
    return 200, {"id": "resp_rec", "status": "completed", "service_tier": "default",
                 "usage": {"input_tokens": 2000, "output_tokens": 600},
                 "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}


# Marco's checklist was assigned to him, not promised: an action without a commitment assertion.
NATIVE = {"The pilot covers 40 clinics in the first wave.", "The pilot launches on Postgres with pgvector.",
          "Priya will send the revised budget by 2026-11-20."}


# ---------------------------------------------------------------- the happy path


def test_receipt_flows_through_every_stage(envelope, vectors):
    use_doubles()
    receipt = receive(envelope)
    after = [*STAGES, "context_proposals", "context_brief"]  # Priya's brief: her commitment links to her actor
    assert drain() == after
    assert [(k, s) for k, s, *_ in jobs()] == [(k, "succeeded") for k in after]
    assert [(stage, status) for stage, status, *_ in runs()] == [
        ("ingest", "succeeded"), ("extract", "succeeded"), ("link", "succeeded"), ("reconcile", "succeeded"),
        ("embed", "succeeded"), ("proposals", "succeeded"), ("brief", "succeeded")]
    with session_scope() as db:
        source = db.get(m.ContextSource, receipt.source_id)
        assert source.status == "ready"
        extract_run = db.scalar(select(m.ContextRun).where(m.ContextRun.stage == "extract"))
        assert extract_run.coverage == {"0": "no_content", "1": "no_content", "2": "extracted", "3": "extracted",
                                        "4": "extracted", "5": "no_content"}
        assert extract_run.engine == "double" and extract_run.decision_watermark == 0
        # Every assertion carries exact evidence and an embedding; chunks are the extraction chunker's.
        rows = list(db.scalars(select(m.ContextAssertion)))
        assert len(rows) == 3 and all(r.embedding is not None and r.embedding_model for r in rows)
        for row in rows:
            ev = list(db.scalars(select(m.ContextEvidence).where(m.ContextEvidence.assertion_id == row.id)))
            assert ev and all(db.get(m.ContextSegment, e.segment_id).text[e.char_start:e.char_end] == e.quote
                              for e in ev)
        chunks = list(db.scalars(select(m.ContextChunk).where(m.ContextChunk.revision_id == receipt.revision_id)))
        first_line = "[0] Davin: Thanks for joining. We need to settle the pilot scope today."
        assert [c.text.splitlines()[0] for c in chunks] == [first_line]
        assert all(c.embedding is not None for c in chunks)
        # Speakers: the owner's lines go to the owner's actor; Priya's e-mail seeded an exact identity.
        speakers = {s.speaker_label: (s.speaker_actor_id, s.speaker_link_status)
                    for s in apply.segment_rows(db, receipt.revision_id) if s.speaker_label}
        owner_actor = db.scalar(select(Actor.id).where(Actor.owner_id == OWNER, Actor.name == "owner"))
        priya = db.scalar(select(Actor.id).where(Actor.owner_id == OWNER, Actor.name == "Priya"))
        assert speakers == {"Davin": (owner_actor, "exact"), "Priya": (priya, "exact"), "Marco": (None, "unresolved")}
        links = {(link.mention, link.role, link.status, link.target_id) for link in db.scalars(select(m.ContextEntityLink))
                 if link.assertion_id}
        assert links == {("Priya", "responsible", "exact", priya), ("Marco", "responsible_mention", "unresolved", None)}
        assert embedding.status(db, OWNER)["unembedded_assertions"] == 0
        # The reconcile stage queued Priya's brief: deterministic unless the owner turns synthesis on.
        brief = db.scalar(select(m.ContextBrief))
        assert (brief.entity_type, brief.entity_id, brief.watermarks["engine"]) == ("actor", priya, "deterministic")


def test_replaying_every_job_and_redelivering_change_nothing(envelope, vectors):
    use_doubles()
    receive(envelope)
    drain()
    before = state()
    with session_scope() as db:
        ids = list(db.scalars(select(Job.id).where(Job.kind.like("context\\_%"))))
        for job in db.scalars(select(Job).where(Job.id.in_(ids))):
            job.status = "retrying"  # as if the outbox delivered every job again after a lost acknowledgement
    for job_id in ids:
        pipeline.perform(job_id)
    assert receive(envelope).change == "none"
    assert drain() == []
    assert state() == before


def test_seeding_runs_as_the_owner_even_under_a_bot_credential(envelope, vectors):
    use_doubles()
    receive(envelope)
    token = bot_access.principal.set("cred-not-allowed-to-create-actors")
    try:
        drain()
    finally:
        bot_access.principal.reset(token)
    with session_scope() as db:
        assert set(db.scalars(select(Actor.name).where(Actor.owner_id == OWNER))) >= {"owner", "Davin", "Priya"}


def test_unknown_kind_fails(vectors):
    with session_scope() as db:
        odd = enqueue_job(db, OWNER, "context_nonsense", {}).id
    pipeline.perform(odd)
    with session_scope() as db:
        assert db.get(Job, odd).status == "failed"


# ---------------------------------------------------------------- native extraction: resumption and staleness


def test_crash_after_model_calls_resumes_without_repeating_them(envelope, provider, vectors, monkeypatch):
    sent = provider(recorded)
    receive(envelope)
    original = apply.apply_extraction
    crashes = []

    def crash_once(*args, **kwargs):
        if not crashes:
            crashes.append(1)
            raise RuntimeError("worker died")
        return original(*args, **kwargs)
    monkeypatch.setattr(apply, "apply_extraction", crash_once)
    pipeline.perform(_next("context_ingest"))
    extract_job = _next("context_extract")
    with pytest.raises(RuntimeError):
        pipeline.perform(extract_job)
    assert jobs("context_extract")[0][1] == "retrying" and assertions() == {}
    assert sent == [*ACTION_PASS, "context_detect", "context_enrich", "context_verify"]
    pipeline.perform(extract_job)  # DBOS retries the step
    drain()
    assert sent == [*ACTION_PASS, "context_detect", "context_enrich", "context_verify"]
    assert set(assertions()) == NATIVE


def test_crash_between_chunks_resumes_from_saved_chunks(provider, vectors):
    model = long_meeting.ScriptedModel()
    calls = []

    def flaky(body):
        calls.append(body["text"]["format"]["name"])
        if calls.count("context_detect") == 3 and calls[-1] == "context_detect" and "crashed" not in calls:
            calls.append("crashed")
            raise RuntimeError("worker died mid-extraction")
        return model(body)
    provider(flaky)
    segments = long_meeting.segments()
    receive(SourceEnvelope(connector="api", external_id="long-1", title="Long", occurred_at=long_meeting.MEETING_AT,
                           timezone=long_meeting.TIMEZONE, segments=segments, external_revision="1"))
    pipeline.perform(_next("context_ingest"))
    job = _next("context_extract")
    with pytest.raises(RuntimeError):
        pipeline.perform(job)
    saved = runs("extract")[0][3]["chunks"]
    assert sorted(saved) == ["0", "1", "actions"]  # the action pass runs first and is saved too
    detected_before = model.calls.count("context_detect")
    pipeline.perform(job)
    chunks = len(runs("extract")[0][3]["result"]["diagnostics"]["chunks"])
    assert chunks > 3
    # Only the interrupted chunk and the ones after it were asked again.
    assert model.calls.count("context_detect") - detected_before == chunks - 2
    drain()
    extracted = runs("extract")[0][3]
    assert extracted["result"]["diagnostics"]["partial"] is False
    distinct = {(a["kind"], a["statement"]) for a in extracted["result"]["assertions"]}
    with session_scope() as db:
        assert db.scalar(select(func.count()).select_from(m.ContextAssertion)) == len(distinct)
        assert db.scalar(select(func.count()).select_from(m.ContextEvidence)) == sum(
            len(a["evidence"]) for a in extracted["result"]["assertions"])


def test_a_new_revision_during_the_model_call_makes_the_result_stale(envelope, provider, vectors):
    revised = envelope.model_copy(update={"segments": [*envelope.segments, Segment(
        seq=5, speaker_label="Marco", attribution="other", text="One more thing about training.")],
        "external_revision": "rev-2"})

    def racing(body):
        if body["text"]["format"]["name"] == "context_detect" and len(jobs("context_ingest")) == 1:
            receive(revised)
        return recorded(body)
    sent = provider(racing)
    first = receive(envelope)
    drain()
    extract_runs = runs("extract")
    assert [(status, error) for _, status, error, *_ in extract_runs] == [("stale", "stale: revision"),
                                                                        ("succeeded", None)]
    assert len(sent) == 10 and set(assertions()) == NATIVE  # 2 action-pass and 3 chunk calls per revision
    with session_scope() as db:
        cited = set(db.scalars(select(m.ContextEvidence.revision_id)))
        assert first.revision_id not in cited and len(cited) == 1
        assert db.get(m.ContextSource, first.source_id).status == "ready"


def test_pause_rejects_outstanding_results_and_resume_continues_without_new_calls(envelope, provider, vectors):
    def pausing(body):
        if body["text"]["format"]["name"] == "context_detect":
            with session_scope() as db:
                pipeline.pause(db, OWNER)
        return recorded(body)
    sent = provider(pausing)
    receive(envelope)
    drain()
    assert runs("extract")[0][1:3] == ("stale", "stale: paused") and assertions() == {}
    assert len(sent) == 5 and jobs("context_link") == []
    with session_scope() as db:
        assert pipeline.schedule_due(db)["continued"] == 0  # still paused (and the flag is off)
        assert pipeline.resume(db, OWNER)["continued"] == 1
    drain()
    assert len(sent) == 5, "the completed action pass and chunk were reused"
    assert set(assertions()) == NATIVE
    with session_scope() as db:
        assert pipeline.resume(db, OWNER)["continued"] == 0  # each interruption continues once


def test_a_decision_during_the_model_call_requeues_and_reuses_the_chunks(envelope, provider, vectors):
    def deciding(body):
        if body["text"]["format"]["name"] == "context_verify" and len(jobs("context_extract")) == 1:
            with session_scope() as db:
                decisions.record(db, OWNER, "dismiss", "question", "q-1", command_id=None, principal=OWNER)
        return recorded(body)
    sent = provider(deciding)
    receive(envelope)
    drain()
    assert [(status, error) for _, status, error, *_ in runs("extract")] == [("stale", "stale: decisions"),
                                                                           ("succeeded", None)]
    assert jobs("context_extract")[1][2]["requeues"] == 1
    assert len(sent) == 5 and set(assertions()) == NATIVE


def test_source_budget_cap_waits_for_an_explicit_continuation(envelope, provider, vectors, monkeypatch):
    sent = provider(recorded)
    monkeypatch.setattr(get_settings(), "context_source_budget_usd", 0.000001)
    receipt = receive(envelope)
    drain()
    with session_scope() as db:
        assert db.get(m.ContextSource, receipt.source_id).status == "awaiting_input"
    assert sent == [] and runs("extract")[0][1] == "partial"
    assert set(runs("extract")[0][4].values()) == {"skipped"}
    with session_scope() as db:
        assert "continue_after" not in runs("extract")[0][3]
        assert pipeline.continue_source(db, OWNER, receipt.source_id, 2.0)
    drain()
    assert len(sent) == 5 and set(assertions()) == NATIVE
    with session_scope() as db:
        assert db.get(m.ContextSource, receipt.source_id).status == "ready"


# ---------------------------------------------------------------- embeddings


def test_permanent_embedding_failure_is_visible_while_lexical_search_works(envelope, monkeypatch):
    use_doubles()

    def down(owner, texts, timeout=20):
        raise DomainError("INTEGRATION_UNAVAILABLE", "no key", 503)
    monkeypatch.setattr(memory_learning, "embeddings", down)
    monkeypatch.setattr(get_settings(), "context_enabled", True)
    receipt = receive(envelope)
    drain()
    with session_scope() as db:
        assert db.get(m.ContextSource, receipt.source_id).status == "ready"
        status = embedding.status(db, OWNER)
        assert status["unembedded_assertions"] == 3 and status["unembedded_chunks"] == 1
        assert status["last_run"]["status"] == "failed"
        assert status["last_run"]["error"] == "embedding failed: INTEGRATION_UNAVAILABLE"
        assert sources.get_source(db, OWNER, receipt.source_id)["last_error"]["stage"] == "embed"
        hits = db.scalars(select(m.ContextAssertion.statement).where(
            m.ContextAssertion.tsv.op("@@")(func.plainto_tsquery("english", "pgvector"))))
        assert list(hits) == ["we launch the pilot on Postgres with pgvector."]
    # Retries back off (5 minutes, 6 hours, 24 hours), then the failure is permanent until an operator retries.
    later = now()
    for step in (timedelta(minutes=6), timedelta(hours=7), timedelta(hours=25)):
        later += step
        with session_scope() as db:
            assert pipeline.schedule_due(db, later)["continued"] == 1
        drain()
    with session_scope() as db:
        assert pipeline.schedule_due(db, later + timedelta(days=30))["continued"] == 0
        last = db.scalar(select(m.ContextRun).where(m.ContextRun.stage == "embed")
                         .order_by(m.ContextRun.created_at.desc()).limit(1))
        assert last.stage_results.get("permanent") is True
    monkeypatch.setattr(memory_learning, "embeddings", vector)
    with session_scope() as db:
        assert pipeline.retry_embeddings(db, OWNER)
        assert pipeline.retry_embeddings(db, OWNER) is None  # one at a time
    drain()
    with session_scope() as db:
        assert embedding.status(db, OWNER)["unembedded_assertions"] == 0
        assert embedding.status(db, OWNER)["unembedded_chunks"] == 0


# ---------------------------------------------------------------- linking and Questions


def test_an_ambiguous_responsible_person_becomes_one_question(envelope, vectors):
    use_doubles()
    with session_scope() as db:
        for n in range(2):
            execute(db, OWNER, f"marco-{n}", "actor.create", {"name": "Marco", "allow_duplicate": True})
    receive(envelope)
    drain()
    with session_scope() as db:
        link = db.scalar(select(m.ContextEntityLink).where(m.ContextEntityLink.mention == "Marco"))
        assert (link.role, link.status, link.target_id) == ("responsible", "unresolved", None)
        questions = decisions.question_items(db, OWNER)
        assert [q["evidence"]["mention"] for q in questions] == ["Marco"]


def test_deletion_job_purges_and_retracts(envelope, vectors):
    use_doubles()
    receipt = receive(envelope)
    drain()
    with session_scope() as db:
        sources.delete_source(db, OWNER, receipt.source_id)
    assert drain() == ["context_delete"]
    assert set(assertions().values()) == {"retracted"}
    with session_scope() as db:
        assert set(db.scalars(select(m.ContextEvidence.quote))) == {""}


# ---------------------------------------------------------------- scheduling


def _next(kind):
    with session_scope() as db:
        return db.scalar(select(Job.id).where(Job.kind == kind, Job.status.in_(("queued", "retrying")))
                         .order_by(Job.created_at).limit(1))


def test_schedule_due_polls_granola_and_backfills_behind_live_work(monkeypatch):
    settings = get_settings()
    with session_scope() as db:
        connection = m.ContextConnection(owner_id=OWNER, connector="granola", external_account="me@example.com")
        db.add(connection)
        db.flush()
        connection_id = connection.id
        assert pipeline.schedule_due(db) == {"granola_sync": 0, "granola_backfill": 0, "continued": 0}
    monkeypatch.setattr(settings, "context_enabled", True)
    with session_scope() as db:
        assert pipeline.schedule_due(db)["granola_sync"] == 0  # Granola itself is off
    monkeypatch.setattr(settings, "context_granola_enabled", True)
    with session_scope() as db:
        assert pipeline.schedule_due(db) == {"granola_sync": 1, "granola_backfill": 1, "continued": 0}
        assert pipeline.schedule_due(db) == {"granola_sync": 0, "granola_backfill": 0, "continued": 0}
        for job in db.scalars(select(Job)):
            job.status = "succeeded"
        assert pipeline.schedule_due(db)["granola_sync"] == 0  # polled less than poll_minutes ago
        soon = now() + timedelta(minutes=settings.context_poll_minutes)
        db.add(m.ContextCursor(owner_id=OWNER, connection_id=connection_id, name="backfill",
                               value={"completed_at": now().isoformat(), "days": 90}))
        assert pipeline.schedule_due(db, soon) == {"granola_sync": 1, "granola_backfill": 0, "continued": 0}
    with session_scope() as db:
        for job in db.scalars(select(Job)):
            job.status = "succeeded"
        db.execute(m.ContextCursor.__table__.delete())
        later = now() + timedelta(hours=1)
        pipeline.pause(db, OWNER)
        assert pipeline.schedule_due(db, later) == {"granola_sync": 0, "granola_backfill": 0, "continued": 0}
        pipeline.resume(db, OWNER)
        enqueue_job(db, OWNER, "context_extract", {"source_id": "s", "revision_id": "r"})
        # New meetings go first: no backfill while ingestion or extraction is waiting.
        assert pipeline.schedule_due(db, later) == {"granola_sync": 1, "granola_backfill": 0, "continued": 0}
