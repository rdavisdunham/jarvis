"""The Stage B gate scenario end to end: fixture histories (package A5) and a synthetic Granola transcript flow
`sources.receive` -> context_ingest -> context_extract -> context_link -> context_reconcile -> context_embed
through `domain.enqueue_job` and `pipeline.perform`, with model and embedding calls mocked.

Checked: every assertion has exact evidence, the coverage ledger is complete, the same history delivered in
different orders converges to the same state (keyed chains included), redelivery and job replay change nothing,
and the Granola poller, scheduled by `schedule_due`, feeds the native extractor."""

import copy
import json
import random
from datetime import timedelta
from pathlib import Path

import httpx
import pytest
from cryptography.fernet import Fernet
from jarvis import memory_learning
from jarvis.config import get_settings
from jarvis.context import models as m
from jarvis.context import pipeline, sources, temporal
from jarvis.context.connectors import granola
from jarvis.context.schema import EntityRef
from jarvis.db import session_scope
from jarvis.models import Actor, Job
from sqlalchemy import select

from evals.context.runner import FIXTURES, HISTORIES, load_history

from . import test_pipeline
from .fixtures.extraction import long_meeting
from .test_pipeline import drain, receive, state, use_doubles

provider = test_pipeline.provider  # the mocked extraction provider fixture


@pytest.fixture
def vectors(monkeypatch):
    monkeypatch.setattr(memory_learning, "embeddings", test_pipeline.vector)


GRANOLA = Path(__file__).parent / "fixtures" / "granola"


# ---------------------------------------------------------------- fixture histories


def deliver(history, owner: str, order: list) -> None:
    """Each meeting arrives and is processed in turn; deletions follow, as the A5 runner orders them."""
    use_doubles(owner)
    for meeting in order:
        receive(meeting.envelope, owner)
        drain(owner)
    for meeting in sorted((x for x in history.meetings if x.deleted_at), key=lambda x: (x.deleted_at, x.key)):
        with session_scope() as db:
            source = db.scalar(select(m.ContextSource).where(m.ContextSource.owner_id == owner,
                                                             m.ContextSource.external_id == meeting.key))
            sources.delete_source(db, owner, source.id)
        drain(owner)


def signature(owner: str) -> list:
    """The owner's assertions without IDs: kind, statement, modality, lifecycle, intervals and evidence spans."""
    def iso(value):
        return value.isoformat() if value else None
    with session_scope() as db:
        external = {s.id: s.external_id for s in db.scalars(select(m.ContextSource).where(
            m.ContextSource.owner_id == owner))}
        seqs = {s.id: s.seq for s in db.scalars(select(m.ContextSegment).where(m.ContextSegment.owner_id == owner))}
        statements = {}
        rows = []
        assertions = list(db.scalars(select(m.ContextAssertion).where(m.ContextAssertion.owner_id == owner)))
        for a in assertions:
            statements[a.id] = temporal.normalized(a.statement)
        for a in assertions:
            spans = sorted([external[e.source_id], seqs.get(e.segment_id), e.char_start, e.char_end, e.role]
                           for e in db.scalars(select(m.ContextEvidence).where(m.ContextEvidence.assertion_id == a.id)))
            rows.append([a.kind, statements[a.id], a.modality, a.lifecycle, iso(a.valid_from), iso(a.valid_to),
                         iso(a.due_at), statements.get(a.supersedes_id), spans])
    return sorted(rows, key=json.dumps)


def orders(history):
    chronological = sorted(history.meetings, key=lambda x: (x.occurred_at, x.key))
    shuffled = list(chronological)
    random.Random(7).shuffle(shuffled)
    return {"chronological": chronological, "reverse": chronological[::-1], "shuffled": shuffled}


def check_ledger(owner: str) -> None:
    with session_scope() as db:
        for run in db.scalars(select(m.ContextRun).where(m.ContextRun.owner_id == owner,
                                                         m.ContextRun.stage == "extract")):
            texts = apply_texts(db, run.revision_id)
            assert run.status == "succeeded" and set(run.coverage) == {str(seq) for seq in texts}
        for a in db.scalars(select(m.ContextAssertion).where(m.ContextAssertion.owner_id == owner)):
            evidence = list(db.scalars(select(m.ContextEvidence).where(m.ContextEvidence.assertion_id == a.id)))
            assert evidence, a.statement
            for e in evidence:
                if e.segment_id:  # deleted sources keep a content-free tombstone
                    assert db.get(m.ContextSegment, e.segment_id).text[e.char_start:e.char_end] == e.quote
        live = db.scalars(select(m.ContextSource).where(m.ContextSource.owner_id == owner,
                                                        m.ContextSource.status != "deleted"))
        assert {s.status for s in live} == {"ready"}


def apply_texts(db, revision_id):
    return {s.seq: s.text for s in db.scalars(select(m.ContextSegment).where(m.ContextSegment.revision_id == revision_id))}


@pytest.mark.parametrize("name", HISTORIES)
def test_fixture_history_converges_across_orders_and_repeats_are_no_ops(name, vectors):
    history = load_history(FIXTURES / name)
    states = {}
    for label, order in orders(history).items():
        owner = f"{history.owner_id}-{label}"
        deliver(history, owner, order)
        check_ledger(owner)
        states[label] = signature(owner)
    assert states["chronological"] and states["reverse"] == states["chronological"] == states["shuffled"]
    # Redelivery dedupes by content; replaying every job of a finished history changes nothing.
    owner = f"{history.owner_id}-chronological"
    before = state()
    for meeting in history.meetings:
        assert receive(meeting.envelope, owner).unchanged
    with session_scope() as db:
        ids = list(db.scalars(select(Job.id).where(Job.owner_id == owner)))
        for job in db.scalars(select(Job).where(Job.id.in_(ids))):
            job.status = "retrying"
    for job_id in ids:
        pipeline.perform(job_id)
    assert drain(owner) == [] and state() == before and signature(owner) == states["chronological"]


def test_keyed_chains_converge_whatever_the_arrival_order(vectors, monkeypatch):
    """Every dated commitment is tied to one task, so their due dates form one exclusive chain across meetings:
    the latest-effective value is current, the others superseded, whatever order the meetings arrive in."""
    original = pipeline.run_extractor

    def linked(engine, request, *args, **kwargs):
        result = original(engine, request, *args, **kwargs)
        task = EntityRef(entity_type="task", entity_id="task-portal")
        return result.model_copy(update={"assertions": [
            a.model_copy(update={"subject": task}) if a.key == "due_date" else a for a in result.assertions]})
    monkeypatch.setattr(pipeline, "run_extractor", linked)
    history = load_history(FIXTURES / "history_alpha")
    states = {}
    for label, order in orders(history).items():
        owner = f"keyed-{label}"
        deliver(history, owner, order)
        states[label] = signature(owner)
    assert states["reverse"] == states["chronological"] == states["shuffled"]
    keyed = [row for row in states["chronological"] if row[0] == "commitment" and row[6]]
    lifecycles = [row[3] for row in keyed]
    assert lifecycles.count("current") == 1 and "superseded" in lifecycles


# ---------------------------------------------------------------- a synthetic Granola transcript, native engine

ACCOUNT = "alex.morgan@fernhill.example"
NOTE_ID = "not_Z9y8X7w6V5u4T3"
LINES = [
    ({"source": "microphone", "attribution": "me"}, "Thanks both, let's settle the onboarding plan today."),
    ({"source": "speaker", "attribution": "them", "name": "Priya Natarajan"},
     "We decided to start onboarding with the analytics team."),
    ({"source": "speaker", "attribution": "them", "name": "Priya Natarajan"},
     "Sounds good. I'll send the data export by Friday."),
    ({"source": "microphone", "attribution": "me"}, "I'll book the training rooms by Friday."),
]


class Granola:
    """One synthetic note owned by the connected account, served as Granola's REST API would."""

    def __init__(self):
        note = json.loads((GRANOLA / "note_not_A1b2C3d4E5f6G7.json").read_text())
        note.update(id=NOTE_ID, title="Northwind onboarding plan")
        self.note = note
        self.transcript = [{"speaker": speaker, "text": text, "start_time": f"2026-11-03T15:0{i}:00Z",
                            "end_time": f"2026-11-03T15:0{i}:30Z"} for i, (speaker, text) in enumerate(LINES)]
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request.url.path)
        parts = request.url.path.strip("/").split("/")
        if parts == ["v1", "notes"]:
            summary = {k: self.note[k] for k in ("id", "object", "title", "owner", "created_at", "updated_at",
                                                 "deleted_at")}
            return httpx.Response(200, json={"notes": [summary], "hasMore": False, "cursor": None})
        if parts == ["v1", "notes", NOTE_ID]:
            return httpx.Response(200, json=copy.deepcopy(self.note))
        if parts == ["v1", "notes", NOTE_ID, "transcript"]:
            return httpx.Response(200, json={"transcript": self.transcript, "hasMore": False, "cursor": None})
        return httpx.Response(404)


def scripted(model):
    def answer(body):
        if body["text"]["format"]["name"] == "context_reconciliation":
            text = json.dumps({"pairs": []})
            return 200, {"id": "resp_r", "status": "completed", "service_tier": "default",
                         "usage": {"input_tokens": 100, "output_tokens": 10},
                         "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}
        return model(body)
    return answer


def test_granola_poll_feeds_the_native_pipeline(provider, vectors, monkeypatch):
    settings = get_settings()
    for key, value in (("context_enabled", True), ("context_granola_enabled", True),
                       ("integration_encryption_key", Fernet.generate_key().decode())):
        monkeypatch.setattr(settings, key, value)
    monkeypatch.setattr(granola.TokenBucket, "acquire", lambda self: None)
    fake = Granola()
    client_for = granola._client_for
    monkeypatch.setattr(granola, "_client_for", lambda connection, report, when, transport, sleep: client_for(
        connection, report, when, httpx.MockTransport(fake), lambda _: None))
    model = long_meeting.ScriptedModel()
    sent = provider(scripted(model))
    with session_scope() as db:
        db.add(m.ContextConnection(owner_id="davin", connector="granola", external_account=ACCOUNT,
                                   credential_ciphertext=granola.seal_credential("gk_synthetic")))
    with session_scope() as db:
        assert pipeline.schedule_due(db) == {"granola_sync": 1, "granola_backfill": 1, "continued": 0}
    kinds = drain()
    assert kinds[:2] == ["context_granola_sync", "context_granola_backfill"]
    assert kinds[2:7] == ["context_ingest", "context_extract", "context_link", "context_reconcile", "context_embed"]
    assert set(kinds[7:]) == {"context_brief"} and provider.briefs  # native synthesis was asked (and refused)
    assert sent.count("context_detect") == 1 and "context_reconciliation" not in sent
    with session_scope() as db:
        source = db.scalar(select(m.ContextSource))
        assert (source.connector, source.external_id, source.status) == ("granola", NOTE_ID, "ready")
        rows = {a.statement: a for a in db.scalars(select(m.ContextAssertion))}
        assert set(rows) == {"The team decided to start onboarding with the analytics team.",
                             "Priya Natarajan will send the data export by Friday.",
                             "Alex Morgan will book the training rooms by Friday."}
        assert all(a.lifecycle == "current" and a.embedding is not None for a in rows.values())
        assert {a.due_at.date().isoformat() for a in rows.values() if a.kind == "commitment"} == {"2026-11-06"}
        owner_actor = db.scalar(select(Actor.id).where(Actor.owner_id == "davin", Actor.name == "owner"))
        priya = db.scalar(select(Actor.id).where(Actor.owner_id == "davin", Actor.name == "Priya Natarajan"))
        links = {(x.mention, x.status, x.target_id) for x in db.scalars(select(m.ContextEntityLink))
                 if x.role == "responsible"}
        # The note-taker's lines are exact; Priya's label names exactly one attendee, so it is inferred.
        assert links == {("Alex Morgan", "exact", owner_actor), ("Priya Natarajan", "inferred", priya)}
        extract_run = db.scalar(select(m.ContextRun).where(m.ContextRun.stage == "extract"))
        assert set(extract_run.coverage.values()) <= {"extracted", "no_content"} and len(extract_run.coverage) == 5
        assert float(extract_run.cost_usd) > 0 and extract_run.prompt_version
    # The next poll finds the note unchanged: nothing is received or extracted again.
    with session_scope() as db:
        later = db.scalar(select(Job.created_at).order_by(Job.created_at.desc()).limit(1))
        assert pipeline.schedule_due(db, later + timedelta(minutes=settings.context_poll_minutes))["granola_sync"] == 1
    assert drain() == ["context_granola_sync"]
    assert sent.count("context_detect") == 1
