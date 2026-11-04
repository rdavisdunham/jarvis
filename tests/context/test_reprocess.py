"""POST /api/v1/context/reviews/{source_id}/reprocess: the owner's explicit re-run of a meeting. A fresh extract
run of the current revision (no duplicate skip, no chunk results reused from earlier runs), keyed by the command
ID, then the normal downstream stages. Approved and published proposals never change, dismissals hold, and it is
refused while the meeting is in flight, to agents and to bot keys. Extraction runs through the double."""

import pytest
from fastapi.testclient import TestClient
from jarvis.api import app
from jarvis.config import get_settings
from jarvis.context import models as m
from jarvis.context import pipeline
from jarvis.db import session_scope
from jarvis.models import Job
from sqlalchemy import select

from .test_pipeline import drain, use_doubles, vectors  # noqa: F401
from .test_proposals import OWNER, by_title, meeting, process, rows

BASE = "/api/v1/context/reviews"


@pytest.fixture(autouse=True)
def enabled(monkeypatch, vectors):  # noqa: F811
    monkeypatch.setenv("JARVIS_CONTEXT_ENABLED", "true")
    get_settings.cache_clear()
    use_doubles()
    yield
    get_settings.cache_clear()


def code(response):
    return response.status_code, response.json()["error"]["code"]


def reprocess(client, source_id, command_id):
    return client.post(f"{BASE}/{source_id}/reprocess", json={"command_id": command_id})


def extract_runs(source_id):
    with session_scope() as db:
        return [(r.status, r.stage_results or {}) for r in db.scalars(
            select(m.ContextRun).where(m.ContextRun.source_id == source_id, m.ContextRun.stage == "extract")
            .order_by(m.ContextRun.created_at, m.ContextRun.id))]


def test_reprocess_runs_a_fresh_extraction_and_keeps_decided_proposals(client, monkeypatch):
    got = process(meeting())
    review = client.get(f"{BASE}/{got.source_id}").json()
    assert review["busy"] is False
    mine, priya = by_title(review["proposals"], "circulate"), by_title(review["proposals"], "budget")
    approved = client.post(f"{BASE}/{got.source_id}/approve", json={
        "command_id": "approve-1", "destination": "eridani",
        "items": [{"proposal_id": mine["id"], "revision": mine["revision"]}]})
    assert approved.status_code == 200, approved.text
    dismissed = client.post(f"{BASE}/{got.source_id}/dismiss", json={
        "command_id": "dismiss-1", "items": [{"proposal_id": priya["id"], "revision": priya["revision"]}]})
    assert dismissed.status_code == 200, dismissed.text
    drain()
    before = {p["id"]: p for p in rows(got.source_id)}
    assert before[mine["id"]]["disposition"] == "published" and before[priya["id"]]["disposition"] == "dismissed"
    with session_scope() as db:  # a saved chunk result an ordinary retry would reuse
        run = db.scalar(select(m.ContextRun).where(m.ContextRun.source_id == got.source_id,
                                                   m.ContextRun.stage == "extract"))
        run.stage_results = {**run.stage_results, "chunks": {"0": {"key": "old"}}}
    seen = []
    original = pipeline.run_extractor
    monkeypatch.setattr(pipeline, "run_extractor",
                        lambda engine, request, cached, *a, **k: seen.append(cached) or original(
                            engine, request, cached, *a, **k))

    first = reprocess(client, got.source_id, "reprocess-1")
    assert first.status_code == 200, first.text
    assert first.json()["reprocess"]["replayed"] is False and first.json()["review"]["busy"] is True
    again = reprocess(client, got.source_id, "reprocess-1")  # the UI resends after a lost connection
    assert again.json()["reprocess"] == {"job_id": first.json()["reprocess"]["job_id"], "replayed": True}
    assert code(reprocess(client, got.source_id, "reprocess-2")) == (409, "CONTEXT_BUSY")
    with session_scope() as db:
        job = db.get(Job, first.json()["reprocess"]["job_id"])
        assert job.kind == "context_extract" and job.payload["reprocess"] == "reprocess-1"
        assert job.payload["revision_id"] == got.revision_id

    kinds = drain()
    assert kinds[:4] == ["context_extract", "context_link", "context_reconcile", "context_embed"]
    assert "context_proposals" in kinds
    assert seen == [{}]  # nothing reused from the earlier run's chunks
    runs = extract_runs(got.source_id)
    assert len(runs) == 2 and runs[1][0] == "succeeded" and "skipped" not in runs[1][1]
    assert runs[1][1]["payload"]["reprocess"] == "reprocess-1"
    after = {p["id"]: p for p in rows(got.source_id)}
    assert set(after) == set(before)
    for proposal_id in (mine["id"], priya["id"]):
        assert {k: after[proposal_id][k] for k in ("disposition", "revision", "title", "task_id")} == \
            {k: before[proposal_id][k] for k in ("disposition", "revision", "title", "task_id")}
    final = client.get(f"{BASE}/{got.source_id}").json()
    assert final["busy"] is False and final["source"]["status"] == "ready"
    assert reprocess(client, got.source_id, "reprocess-2").status_code == 200  # idle again: allowed


def test_an_ordinary_duplicate_extract_still_skips():
    got = process(meeting())
    with session_scope() as db:
        pipeline.enqueue_job(db, OWNER, "context_extract", {"source_id": got.source_id,
                                                            "revision_id": got.revision_id})
    assert drain() == ["context_extract"]
    assert extract_runs(got.source_id)[-1][1].get("skipped") == "duplicate"


def test_reprocess_is_the_owners_alone(client):
    got = process(meeting())
    assert code(reprocess(client, got.source_id, "ui-agent:reprocess:1")) == (403, "HUMAN_REQUIRED")
    minted = client.post("/api/v1/bot-keys", json={"name": "ctx", "scopes": ["context:read", "context:write"],
                                                   "expires_in_days": 30})
    headers = {"Authorization": "Bearer " + minted.json()["token"]}
    bot = TestClient(app)
    external = bot.post(f"/api/v1/external/context/reviews/{got.source_id}/reprocess", headers=headers,
                        json={"command_id": "c1"})
    assert external.status_code in (404, 405)
    assert bot.post(f"{BASE}/{got.source_id}/reprocess", headers=headers,
                    json={"command_id": "c1"}).status_code in (401, 403)
    assert code(reprocess(client, "00000000-0000-0000-0000-000000000000", "c2")) == (404, "NOT_FOUND")
    with session_scope() as db:
        assert not db.scalar(select(Job.id).where(Job.payload["reprocess"].astext.is_not(None)).limit(1))
