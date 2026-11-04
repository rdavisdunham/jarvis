"""Pipeline wiring with the merged packages: the bridge script against B1's real PUT route, `context.changed`
events for views.changes (assertions, links, source revisions, deletion), brief jobs (C3's prepare, synthesize,
commit), E3's pause and resume followed by `schedule_due`, and the Settings panel's Granola job payloads.
Model and embedding calls are mocked or replaced by the deterministic doubles."""

import json
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import httpx
import pytest
from jarvis.config import get_settings
from jarvis.context import decisions, pipeline, sources, views
from jarvis.context import models as m
from jarvis.context.doubles import DoubleSynthesizer
from jarvis.context.routes import admin
from jarvis.db import session_scope
from jarvis.domain import enqueue_job
from jarvis.models import Actor, Event, Job
from sqlalchemy import select

from scripts import bridge_submit

from .conftest import LINES
from .test_pipeline import OWNER, STAGES, drain, jobs, receive, runs, use_doubles, vector

CHANGED = "context.changed"


@pytest.fixture
def vectors(monkeypatch):
    from jarvis import memory_learning

    monkeypatch.setattr(memory_learning, "embeddings", vector)


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(get_settings(), "context_enabled", True)


def events() -> list[tuple[str, int | None]]:
    with session_scope() as db:
        return [(e.entity_id, e.revision) for e in db.scalars(
            select(Event).where(Event.owner_id == OWNER, Event.kind == CHANGED).order_by(Event.id))]


def heads() -> dict[str, int]:
    with session_scope() as db:
        return {r.id: r.current_version for r in db.scalars(select(m.ContextAssertion))}


# ---------------------------------------------------------------- the bridge through the real route


def vtt() -> str:
    cues = [f"00:00:{i * 10 + 1:02d}.000 --> 00:00:{i * 10 + 9:02d}.000\n<v {who}>{text}</v>"
            for i, (who, _, text) in enumerate(LINES)]
    return "WEBVTT\n\n" + "\n\n".join(cues) + "\n"


class Through(httpx.BaseTransport):
    """Hands the bridge's requests to the FastAPI app (B1's real route, auth and scope checks) via TestClient."""

    def __init__(self):
        from fastapi.testclient import TestClient
        from jarvis.api import app

        self.app = TestClient(app)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
        answer = self.app.request(request.method, str(request.url), content=request.read(), headers=headers)
        kept = {k: v for k, v in answer.headers.items() if k.lower() not in ("content-length", "content-encoding")}
        return httpx.Response(answer.status_code, headers=kept, content=answer.content)


def step() -> str:
    """Run the oldest queued context job only."""
    with session_scope() as db:
        job = db.scalar(select(Job).where(Job.owner_id == OWNER, Job.kind.like("context\\_%"),
                                          Job.status.in_(("queued", "retrying"))).order_by(Job.created_at, Job.id))
        job_id, kind = job.id, job.kind
    pipeline.perform(job_id)
    return kind


def test_bridge_submission_through_the_real_route_feeds_the_pipeline(client, tmp_path, vectors, monkeypatch):

    monkeypatch.setattr(get_settings(), "context_enabled", True)
    token = client.post("/api/v1/bot-keys", json={"name": "bridge", "scopes": ["sources:write"],
                                                  "expires_in_days": 30}).json()["token"]
    (tmp_path / "2026-11-03 15-00 pilot.vtt").write_text(vtt())
    (tmp_path / "kickoff.json").write_text(json.dumps({  # a version 1 envelope with attendees' e-mails
        "version": 1, "external_id": "kickoff", "title": "Kickoff", "occurred_at": "2026-11-02T10:00:00+00:00",
        "participants": [{"name": "Priya", "email": "priya@example.com", "speaker_label": "Priya"}],
        "segments": [{"seq": 0, "speaker_label": "Priya", "text": "I'll send the contract by 2026-11-12."}]}))
    use_doubles()
    transport = Through()
    submit = lambda: bridge_submit.run(tmp_path, "http://testserver", token, "recorder",
                                       ZoneInfo("Europe/Lisbon"), transport=transport)
    assert submit()["created"] == 2
    with session_scope() as db:
        rows = list(db.scalars(select(m.ContextSource).order_by(m.ContextSource.created_at)))
        assert {(r.owner_id, r.status) for r in rows} == {(OWNER, "queued")} and len(rows) == 2
        assert db.scalar(select(m.ContextActorIdentity.id)) is None  # nothing seeded under the bot credential
        assert db.scalar(select(Actor.id).where(Actor.name == "Priya")) is None
    kinds = drain()
    assert kinds.count("context_ingest") == 2 and kinds.count("context_reconcile") == 2
    assert set(kinds) <= {*STAGES, "context_brief"}
    assert all(status == "succeeded" for _, status, *_ in jobs())
    with session_scope() as db:
        assert {r.status for r in db.scalars(select(m.ContextSource))} == {"ready"}
        statements = set(db.scalars(select(m.ContextAssertion.statement)))
        assert {"we launch the pilot on Postgres with pgvector.", "I'll send the revised budget by 2026-11-20.",
                "I'll send the contract by 2026-11-12."} <= statements
        # The ingest job, running as the owner, seeded the attendee's identity and actor.
        priya = db.scalar(select(Actor.id).where(Actor.owner_id == OWNER, Actor.name == "Priya"))
        assert priya and db.scalar(select(m.ContextActorIdentity.actor_id).where(
            m.ContextActorIdentity.owner_id == OWNER, m.ContextActorIdentity.normalized == "priya@example.com")) == priya
    assert submit()["unchanged"] == 2 and drain() == []


# ---------------------------------------------------------------- change events


def test_every_written_version_emits_context_changed(envelope, vectors):
    use_doubles()
    receipt = receive(envelope)
    assert events() == [(receipt.source_id, 1)]
    drain()
    emitted = dict(events())
    for assertion_id, version in heads().items():
        assert emitted[assertion_id] == version
    with session_scope() as db:
        for link in db.scalars(select(m.ContextEntityLink).where(m.ContextEntityLink.assertion_id.is_not(None))):
            assert emitted[link.id] == link.revision
        brief = db.scalar(select(m.ContextBrief))
        assert emitted[brief.id] == brief.version  # C3's commit emits its own
        seen = {item["id"]: item for item in views.changes(db, OWNER)["items"]}
        assert seen[receipt.source_id]["type"] == "source" and not seen[receipt.source_id]["deleted"]
        assert {seen[i]["type"] for i in heads()} == {"assertion"}

    before = len(events())
    replay_ids = [job_id for job_id, in _job_ids()]
    for job_id in replay_ids:
        pipeline.perform(job_id)
    assert len(events()) == before, "replays emit nothing"

    revised = receive(envelope.model_copy(update={"private_notes": "New note.", "external_revision": "rev-2"}))
    assert events()[-1] == (receipt.source_id, 2) and revised.revision == 2
    drain()

    with session_scope() as db:
        sources.delete_source(db, OWNER, receipt.source_id)
    assert events()[-1] == (receipt.source_id, 2)
    old = heads()
    drain()  # the purge retracts what only this source supported
    retracted = {i: v for i, v in heads().items() if v != old[i]}
    assert retracted and set(retracted.items()) <= set(events())
    with session_scope() as db:
        assert all(r.lifecycle == "retracted" for r in db.scalars(select(m.ContextAssertion)))
        assert db.scalar(select(m.ContextBrief)).stale  # briefs.mark_stale(source_id=...) in the purge
        seen = {item["id"]: item for item in views.changes(db, OWNER)["items"]}
        assert seen[receipt.source_id]["deleted"] and all(seen[i]["deleted"] for i in retracted)


def _job_ids():
    with session_scope() as db:
        return list(db.execute(select(Job.id).where(Job.owner_id == OWNER, Job.kind.like("context\\_%"))))


# ---------------------------------------------------------------- briefs


def test_brief_jobs_skip_unchanged_inputs_refuse_bad_types_and_requeue_when_inputs_move(envelope, vectors,
                                                                                         monkeypatch):
    use_doubles()
    receive(envelope)
    drain()
    with session_scope() as db:
        brief = db.scalar(select(m.ContextBrief))
        target = (brief.entity_type, brief.entity_id)
        enqueue_job(db, OWNER, "context_brief", {"entity_type": target[0], "entity_id": target[1]})
        enqueue_job(db, OWNER, "context_brief", {"entity_type": "galaxy", "entity_id": "x"})
    assert drain() == ["context_brief", "context_brief"]
    unchanged, bad = runs("brief")[-2:]
    assert unchanged[1] == "succeeded" and unchanged[3]["unchanged"] is True
    assert bad[1] == "failed" and bad[2] == "INVALID_ARGUMENT" and bad[3]["permanent"] is True
    assert [s for _, s, *_ in jobs("context_brief")] == ["succeeded"] * 3  # nothing left to retry

    moved = []

    class Moving(DoubleSynthesizer):  # the owner marks one input Wrong while the model is working (once)
        def synthesize(self, request):
            if moved:
                return super().synthesize(request)
            moved.append(True)
            with session_scope() as db:
                row = db.scalar(select(m.ContextAssertion).join(
                    m.ContextEntityLink, m.ContextEntityLink.assertion_id == m.ContextAssertion.id).where(
                    m.ContextEntityLink.target_id == target[1], m.ContextAssertion.lifecycle == "current"))
                decisions.wrong(db, OWNER, row.id, row.current_version, command_id=None, principal=OWNER)
            return super().synthesize(request)

    monkeypatch.setattr(pipeline, "synthesizer", lambda engine: Moving())
    with session_scope() as db:
        for row in db.scalars(select(m.ContextBrief)):  # so prepare_brief has something to synthesize
            db.delete(row)
        enqueue_job(db, OWNER, "context_brief", {"entity_type": target[0], "entity_id": target[1]})
    assert drain() == ["context_brief", "context_brief"]  # the second is the requeue
    first, again = runs("brief")[-2:]
    assert first[3]["resynthesize"] is True and again[3]["resynthesize"] is False
    assert jobs("context_brief")[-1][2]["requeues"] == 1
    with session_scope() as db:
        engines = [(b.watermarks.get("engine"), bool(b.watermarks.get("resynthesize")))
                   for b in db.scalars(select(m.ContextBrief).order_by(m.ContextBrief.version))]
        assert engines == [("deterministic", True), ("synthesized", False)]


# ---------------------------------------------------------------- E3's pause and resume


def test_admin_pause_rejects_results_and_schedule_due_requeues_what_is_still_current(envelope, vectors, enabled):
    use_doubles()
    receipt = receive(envelope)
    assert step() == "context_ingest"
    with session_scope() as db:
        admin.pause(db, OWNER)
    assert drain() == ["context_extract"]
    assert runs("extract")[-1][1:3] == ("stale", "stale: paused")
    receive(envelope.model_copy(update={"private_notes": "Changed.", "external_revision": "rev-2"}))
    assert drain() == ["context_ingest"]  # revision 2's ingest is refused too
    with session_scope() as db:
        admin.resume(db, OWNER)  # E3's resume queues nothing itself
    assert drain() == []
    with session_scope() as db:
        assert pipeline.schedule_due(db)["continued"] == 1  # revision 2's ingest; revision 1 is outdated
    kinds = drain()
    assert kinds[:5] == STAGES
    with session_scope() as db:
        source = db.get(m.ContextSource, receipt.source_id)
        assert (source.status, source.current_revision) == ("ready", 2)
        assert pipeline.schedule_due(db)["continued"] == 0


# ---------------------------------------------------------------- Settings' Granola payloads


@dataclass
class Outcome:
    connection: str
    days: int | None = None


def test_granola_jobs_take_the_settings_payloads(monkeypatch):
    from jarvis.context import connectors

    calls = []

    class Fake:
        def poll(self, db, owner, connection):
            calls.append(("poll", connection.external_account))
            return Outcome(connection.external_account)

        def backfill(self, db, owner, connection, days):
            calls.append(("backfill", connection.external_account, days))
            return Outcome(connection.external_account, days)

    monkeypatch.setattr(connectors, "resolve", lambda name: Fake())
    with session_scope() as db:
        ids = {}
        for account, on in (("a@example.com", True), ("b@example.com", True), ("off@example.com", False)):
            row = m.ContextConnection(owner_id=OWNER, connector="granola", external_account=account, enabled=on)
            db.add(row)
            db.flush()
            ids[account] = row.id
        db.add(m.ContextConnection(owner_id="someone-else", connector="granola", external_account="x@example.com"))
        enqueue_job(db, OWNER, "context_granola_sync", {})
        enqueue_job(db, OWNER, "context_granola_sync", {"connection_id": ids["off@example.com"]})
        enqueue_job(db, OWNER, "context_granola_backfill", {"connection_id": ids["a@example.com"], "days": 400})
        enqueue_job(db, OWNER, "context_granola_backfill", {"connection_id": ids["b@example.com"]})
        enqueue_job(db, OWNER, "context_granola_sync", {"connection_id": "missing"})
    drain()
    assert calls == [("poll", "a@example.com"), ("poll", "b@example.com"), ("poll", "off@example.com"),
                     ("backfill", "a@example.com", 365), ("backfill", "b@example.com", 90)]
    results = [r for *_, r in jobs()]
    assert results[0] == {"connections": {ids["a@example.com"]: {"connection": "a@example.com", "days": None},
                                          ids["b@example.com"]: {"connection": "b@example.com", "days": None}}}
    assert results[3] == {"connection": "b@example.com", "days": 90}
    assert results[4] == {"skipped": "connection not found"}
