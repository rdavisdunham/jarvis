"""/api/v1/external/context/*: the REST twins of the context MCP tools. Capabilities per scope, the flag, the
personal-workspace and viewer refusals, and agent feedback recorded as an agent suggestion."""

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from jarvis.api import app
from jarvis.config import get_settings
from jarvis.context.models import ContextAssertion, ContextDecision
from jarvis.context.native import retrieve as native_retrieve
from jarvis.db import session_scope
from jarvis.models import BotCredential
from sqlalchemy import select

from .test_service import mint, seed, source_id, stub_views, task_for, vec

BASE = "/api/v1/external/context"
READ_TOOLS = {"context_search", "context_brief", "context_evidence", "context_source", "context_open_loops",
              "context_meeting_prep", "context_changes"}


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setenv("JARVIS_CONTEXT_ENABLED", "true")
    get_settings.cache_clear()
    monkeypatch.setattr(native_retrieve, "query_vector", lambda owner, text: vec(0))
    yield
    get_settings.cache_clear()


def http():
    return TestClient(app)


def code(response):
    return response.status_code, response.json()["error"]["code"]


def tools(headers):
    response = http().get("/api/v1/external/capabilities", headers=headers)
    assert response.status_code == 200, response.text
    return {t["name"] for t in response.json()["tools"] if t["name"].startswith("context_")}


def test_capabilities_list_context_tools_only_for_context_keys(client, monkeypatch):
    _, reader = mint(client, "context:read")
    _, writer = mint(client, "context:write")
    _, tasks = mint(client, "tasks:read")
    _, submitter = mint(client, "sources:write")
    assert tools(reader) == READ_TOOLS
    assert tools(writer) == READ_TOOLS | {"context_feedback"}
    assert tools(tasks) == set() and tools(submitter) == set()
    listed = http().get("/api/v1/external/capabilities", headers=writer).json()["tools"]
    feedback = next(t for t in listed if t["name"] == "context_feedback")
    assert feedback["annotations"]["readOnlyHint"] is False and "request_id" in feedback["inputSchema"]["required"]
    search = next(t for t in listed if t["name"] == "context_search")
    assert search["annotations"]["readOnlyHint"] is True and "EntityRef" in search["inputSchema"]["$defs"]
    monkeypatch.setenv("JARVIS_CONTEXT_ENABLED", "false")
    get_settings.cache_clear()
    assert tools(writer) == set()
    assert code(http().post(f"{BASE}/search", headers=reader, json={"query": "x"})) == (403, "CONTEXT_DISABLED")


def test_rest_reads_return_cited_context_and_drill_down(client):
    ids = seed()
    _, reader = mint(client, "context:read")
    found = http().post(f"{BASE}/search", headers=reader,
                        json={"query": "what did we decide", "entity": {"entity_type": "actor",
                                                                         "entity_id": ids.priya}})
    assert found.status_code == 200, found.text
    results = {r["id"]: r for r in found.json()["results"]}
    assert ids.new in results and ids.old not in results
    evidence = http().get(f"{BASE}/evidence/{ids.new}", headers=reader).json()
    assert evidence["evidence"][0]["quote"] == results[ids.new]["citation"]["quote"]
    old = http().get(f"{BASE}/evidence/{ids.old}", params={"include_history": "true"}, headers=reader).json()
    assert old["status"] == "historical" and old["versions"]
    page = http().get(f"{BASE}/sources/{source_id(ids.new)}", params={"segment_start": 0, "segment_end": 0},
                      headers=reader).json()
    assert page["segments"][0]["supports"][0]["assertion_id"] == ids.new
    assert http().post(f"{BASE}/search", headers=reader, json={}).status_code == 422
    assert http().post(f"{BASE}/search", headers=reader, json={"query": "x", "extra": 1}).status_code == 422
    assert code(http().get(f"{BASE}/evidence/{uuid4()}", headers=reader)) == (404, "NOT_FOUND")
    assert code(http().post(f"{BASE}/search", json={"query": "x"})) == (401, "NOT_AUTHORIZED")


def test_rest_views_and_changes(client, monkeypatch):
    ids = seed()
    stub_views(monkeypatch, ids, task_for())
    _, reader = mint(client, "context:read")
    loops = http().post(f"{BASE}/open-loops", headers=reader, json={"waiting_for": True}).json()
    assert [i["assertion_id"] for i in loops["items"]] == [ids.budget]
    prep = http().post(f"{BASE}/meeting-prep", headers=reader, json={"calendar_event_id": "e1"}).json()
    assert prep["event"]["title"] == "Pilot sync"
    brief = http().post(f"{BASE}/brief", headers=reader,
                        json={"entity": {"entity_type": "actor", "entity_id": ids.priya}}).json()
    assert brief["sections"][0]["claims"][0]["assertion_ids"] == [ids.new, ids.why]
    feed = http().get(f"{BASE}/changes", params={"after": 4, "limit": 10}, headers=reader).json()
    assert feed["next_cursor"] == 6
    assert http().get(f"{BASE}/changes", params={"limit": 500}, headers=reader).status_code == 422


def test_feedback_over_rest_is_an_agent_suggestion_and_needs_context_write(client):
    ids = seed()
    _, reader = mint(client, "context:read")
    writer_id, writer = mint(client, "context:write")
    body = {"request_id": str(uuid4()), "target_type": "assertion", "target_id": ids.new, "expected_revision": 1,
            "suggestion": "wrong", "explanation": "The later notes contradict it."}
    assert code(http().post(f"{BASE}/feedback", headers=reader, json=body)) == (403, "INSUFFICIENT_SCOPE")
    saved = http().post(f"{BASE}/feedback", headers=writer, json=body)
    assert saved.status_code == 200, saved.text
    receipt = saved.json()
    assert (receipt["origin"], receipt["applied"], receipt["status"]) == ("agent", False, "pending_review")
    assert http().post(f"{BASE}/feedback", headers=writer, json=body).json() == receipt
    assert http().post(f"{BASE}/feedback", headers=writer,
                       json={**body, "suggestion": "approve"}).status_code == 422  # never approves
    assert http().post(f"{BASE}/feedback", headers=writer,
                       json={**body, "target_type": "proposal"}).status_code == 422  # never touches proposals
    with session_scope() as db:
        row = db.get(ContextDecision, receipt["feedback_id"])
        assert (row.origin, row.principal) == ("agent", "bot:" + writer_id)
        assert db.get(ContextAssertion, ids.new).lifecycle == "current"


def test_shared_workspaces_and_viewers_are_refused(client):
    from test_accounts import client_for, shared

    seed()
    guest = client_for("guest")
    shared(client, guest, role="viewer")
    minted = guest.post("/api/v1/bot-keys", json={"name": "ctx", "scopes": ["tasks:read"], "expires_in_days": 30})
    assert minted.status_code == 201, minted.text
    headers = {"Authorization": "Bearer " + minted.json()["token"]}
    refused = client.post("/api/v1/bot-keys", json={"name": "ctx", "scopes": ["context:read"],
                                                    "expires_in_days": 30})
    assert refused.json()["error"]["code"] == "PERSONAL_WORKSPACE"
    with session_scope() as db:  # a key minted before the personal-only rule
        row = db.scalar(select(BotCredential).where(BotCredential.account_id == "guest"))
        row.scopes = ["context:read", "context:write", "tasks:read"]
    assert code(http().post(f"{BASE}/search", headers=headers, json={"query": "pilot"})) == (
        403, "PERSONAL_WORKSPACE")
    assert code(http().get(f"{BASE}/changes", headers=headers)) == (403, "PERSONAL_WORKSPACE")
    feedback = {"request_id": str(uuid4()), "target_type": "assertion", "target_id": str(uuid4()),
                "expected_revision": 1, "suggestion": "wrong", "explanation": "x"}
    assert code(http().post(f"{BASE}/feedback", headers=headers, json=feedback)) == (403, "READ_ONLY")
