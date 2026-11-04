"""Meeting context for the owner's browser session (/api/v1/context/*), the `context.resolve` command behind
Questions, and real (C3) briefs read through the service. Requests go through the real app."""

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from jarvis import tools
from jarvis.api import app
from jarvis.config import get_settings
from jarvis.context import service
from jarvis.context.models import ContextAssertion, ContextDecision, ContextSource
from jarvis.context.native import retrieve as native_retrieve
from jarvis.db import session_scope
from jarvis.domain import COMMANDS, execute
from jarvis.models import Job
from sqlalchemy import func, select

from .test_service import OWNER, bot, mint, priya, seed, source_id, suggestion, vec

BASE = "/api/v1/context"


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setenv("JARVIS_CONTEXT_ENABLED", "true")
    get_settings.cache_clear()
    monkeypatch.setattr(native_retrieve, "query_vector", lambda owner, text: vec(0))
    yield
    get_settings.cache_clear()


def code(response):
    return response.status_code, response.json()["error"]["code"]


def test_session_reads_mirror_the_bot_routes(client):
    ids = seed()
    entity = {"entity_type": "actor", "entity_id": ids.priya}
    found = client.post(f"{BASE}/search", json={"query": "what did we decide", "entity": entity})
    assert found.status_code == 200, found.text
    results = {r["id"]: r for r in found.json()["results"]}
    assert ids.new in results and ids.old not in results
    assert client.get(f"{BASE}/evidence/{ids.old}", params={"include_history": True}).json()["versions"]
    brief = client.post(f"{BASE}/brief", json={"entity": entity}).json()
    assert {i for s in brief["sections"] for c in s["claims"] for i in c["assertion_ids"]} >= {ids.new}
    assert client.post(f"{BASE}/open-loops", json={}).status_code == 200
    assert client.post(f"{BASE}/meeting-prep", json={"entity": entity}).json()["event"] is None
    assert client.get(f"{BASE}/changes", params={"after": 0}).json()["has_more"] in (True, False)
    listed = client.get(f"{BASE}/sources").json()
    assert listed["total"] == 5
    detail = client.get(f"{BASE}/sources/{source_id(ids.new)}").json()
    assert detail["segments"][0]["text"].startswith("We decided to launch the pilot on Postgres")
    page = client.get(f"{BASE}/sources/{source_id(ids.old)}/segments").json()
    assert page["segments"][0]["supports"][0]["status"] == "historical"
    anonymous = TestClient(app)
    assert anonymous.post(f"{BASE}/search", json={"query": "x"}).status_code == 401
    del client.headers["X-CSRF-Token"]
    assert client.post(f"{BASE}/search", json={"query": "x"}).status_code == 403  # CSRF required


def test_session_delete_flag_and_shared_workspace(client, monkeypatch):
    from test_accounts import client_for, shared

    ids = seed()
    gone = source_id(ids.why)
    deleted = client.delete(f"{BASE}/sources/{gone}")
    assert deleted.status_code == 200 and deleted.json()["status"] == "deleted"
    assert code(client.get(f"{BASE}/sources/{gone}")) == (404, "NOT_FOUND")
    monkeypatch.setenv("JARVIS_CONTEXT_ENABLED", "false")
    get_settings.cache_clear()
    assert code(client.post(f"{BASE}/search", json={"query": "x"})) == (403, "CONTEXT_DISABLED")
    monkeypatch.setenv("JARVIS_CONTEXT_ENABLED", "true")
    get_settings.cache_clear()
    shared(client, client_for("guest"))
    assert code(client.post(f"{BASE}/search", json={"query": "x"})) == (403, "PERSONAL_WORKSPACE")
    assert code(client.get(f"{BASE}/sources")) == (403, "PERSONAL_WORKSPACE")
    assert code(client.delete(f"{BASE}/sources/{source_id(ids.new)}")) == (403, "PERSONAL_WORKSPACE")


def test_a_real_brief_rebuild_queues_synthesis_once(client):
    ids = seed()
    reader, _ = mint(client, "context:read")

    def jobs():
        with session_scope() as db:
            return db.scalar(select(func.count()).select_from(Job).where(Job.kind == "context_brief"))

    with bot(reader):
        first = service.brief(service.BriefInput(entity=priya(ids)))
        again = service.brief(service.BriefInput(entity=priya(ids), include_history=True))
    assert first["basis"] == "rebuilt" and again["basis"] == "stored" and jobs() == 1
    cited = {i for s in first["sections"] for c in s["claims"] for i in c["assertion_ids"]}
    assert ids.new in cited and ids.why in cited and ids.old not in cited
    assert any("superseded" in g for g in first["gaps"])  # the "Changed" claim waits for include_history
    history_cited = {i for s in again["sections"] for c in s["claims"] for i in c["assertion_ids"]}
    assert ids.old in history_cited and [h["id"] for h in again["history"]] == [ids.old]


def test_context_resolve_command_is_human_from_the_browser_and_agent_from_dani(client):
    ids = seed()
    assert "context.resolve" in COMMANDS and "context.resolve" in tools.VOICE_MUTATIONS
    writer, headers = mint(client, "context:write")
    with bot(writer):
        wrong = service.feedback(suggestion(ids))
        confirm = service.feedback(suggestion(ids, target_id=ids.why, suggestion="confirm"))
    listed = client.get("/api/v1/questions").json()["items"]
    item = next(q for q in listed if q["key"] == wrong["question_key"])
    assert (item["kind"], item["agent_tool"]) == ("context", "context_resolve")
    bot_call = TestClient(app).post("/api/v1/external/commands", headers=headers, json={
        "request_id": str(uuid4()), "tool": "context.resolve",
        "arguments": {"question_key": wrong["question_key"], "expected_revision": 1, "action": "accept"}})
    assert code(bot_call) == (403, "INSUFFICIENT_SCOPE")  # agents suggest; they never answer the owner's Questions
    answered = client.post("/api/v1/commands", json={"command_id": str(uuid4()), "tool": "context.resolve",
                                                     "arguments": {**item["answer_args"], "action": "accept"}})
    assert answered.status_code == 200, answered.text
    deferred = client.post("/api/v1/commands", json={"command_id": str(uuid4()), "tool": "review.defer", "arguments": {
        "question_key": confirm["question_key"], "expected_revision": 1, "until": "week"}})
    assert deferred.status_code == 200, deferred.text
    with session_scope() as db:
        applied = db.scalar(select(ContextDecision).where(ContextDecision.target_id == ids.new,
                                                          ContextDecision.operation == "wrong"))
        assert applied.origin == "human" and db.get(ContextAssertion, ids.new).lifecycle == "retracted"
        execute(db, OWNER, "turn-1:0", "context.resolve",
                {"question_key": confirm["question_key"], "expected_revision": 2, "action": "accept"})
        dani = db.scalar(select(ContextDecision).where(ContextDecision.target_id == ids.why,
                                                       ContextDecision.operation == "confirm",
                                                       ContextDecision.target_type == "assertion"))
        assert dani.origin == "agent"
        assert service.feedback_items(db, OWNER) == []


def test_context_resolve_refuses_when_off_or_shared(client, monkeypatch):
    from test_accounts import client_for, shared

    args = {"question_key": "context-feedback:x", "expected_revision": 1, "action": "dismiss"}
    monkeypatch.setenv("JARVIS_CONTEXT_ENABLED", "false")
    get_settings.cache_clear()
    off = client.post("/api/v1/commands", json={"command_id": str(uuid4()), "tool": "context.resolve",
                                                "arguments": args})
    assert code(off) == (403, "CONTEXT_DISABLED")
    monkeypatch.setenv("JARVIS_CONTEXT_ENABLED", "true")
    get_settings.cache_clear()
    shared(client, client_for("guest"))
    refused = client.post("/api/v1/commands", json={"command_id": str(uuid4()), "tool": "context.resolve",
                                                    "arguments": args})
    assert code(refused) == (403, "PERSONAL_WORKSPACE")
    with session_scope() as db:
        assert db.scalar(select(func.count()).select_from(ContextSource)) == 0
