"""Context tools over the real MCP transport (stateless Streamable HTTP, JSON responses).

Stage gate scenario: an agent holding a `context:read` key asks "what did we decide with Priya and why" and gets
the current decision and its rationale with citations; the superseded decision appears only with
include_history."""

import json
from uuid import uuid4

import pytest
from jarvis.config import get_settings
from jarvis.context import mcp as context_mcp
from jarvis.context.native import retrieve as native_retrieve
from jarvis.external_mcp import definitions

from .test_service import mint, seed, stub_views, task_for, vec

PROTOCOL = "2025-11-25"


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setenv("JARVIS_CONTEXT_ENABLED", "true")
    get_settings.cache_clear()
    monkeypatch.setattr(native_retrieve, "query_vector", lambda owner, text: vec(0))
    yield
    get_settings.cache_clear()


def rpc(client, headers, method, params=None):
    response = client.post("/api/v1/external/mcp/", headers={
        **headers, "Accept": "application/json, text/event-stream", "MCP-Protocol-Version": PROTOCOL},
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}})
    assert response.status_code == 200, response.text
    return response.json()["result"]


def call(client, headers, name, arguments):
    result = rpc(client, headers, "tools/call", {"name": name, "arguments": arguments})
    payload = json.loads(result["content"][0]["text"])
    return result.get("isError", False), payload


def test_tools_list_by_scope(client):
    _, reader = mint(client, "context:read")
    _, writer = mint(client, "context:write")
    rpc(client, reader, "initialize", {"protocolVersion": PROTOCOL, "capabilities": {},
                                       "clientInfo": {"name": "test", "version": "1"}})
    listed = {t["name"]: t for t in rpc(client, reader, "tools/list")["tools"]}
    assert {n for n in listed if n.startswith("context_")} == set(context_mcp.TOOLS) - {"context_feedback"}
    assert listed["context_search"]["annotations"]["readOnlyHint"] is True
    names = {t["name"] for t in rpc(client, writer, "tools/list")["tools"]}
    assert set(context_mcp.TOOLS) <= names


def test_what_did_we_decide_with_x_and_why_over_mcp(client):
    ids = seed()
    _, reader = mint(client, "context:read")
    entity = {"entity_type": "actor", "entity_id": ids.priya}
    error, answer = call(client, reader, "context_search", {
        "query": "what did we decide with Priya and why", "entity": entity, "kinds": ["decision", "rationale"]})
    assert not error, answer
    results = {r["id"]: r for r in answer["results"]}
    assert set(results) == {ids.new, ids.why}
    assert all(r["status"] == "current" and r["citation"]["quote"] for r in results.values())
    assert "MySQL" not in json.dumps(answer)
    error, history = call(client, reader, "context_search", {
        "query": "what did we decide with Priya and why", "entity": entity, "kinds": ["decision"],
        "include_history": True})
    assert not error
    past = {r["id"]: r for r in history["results"]}
    assert past[ids.old]["status"] == "historical" and past[ids.old]["superseded_by"] == ids.new
    error, spans = call(client, reader, "context_evidence", {"assertion_id": ids.new})
    assert not error and spans["evidence"][0]["quote"].startswith("We decided to launch the pilot on Postgres")
    error, source = call(client, reader, "context_source", {"source_id": spans["evidence"][0]["source_id"]})
    assert not error and source["segments"][0]["supports"][0]["status"] == "current"


def test_brief_over_mcp_cites_current_claims_and_adds_history_on_request(client, monkeypatch):
    ids = seed()
    stub_views(monkeypatch, ids, task_for())
    _, reader = mint(client, "context:read")
    entity = {"entity_type": "actor", "entity_id": ids.priya}
    error, brief = call(client, reader, "context_brief", {"entity": entity})
    assert not error, brief
    claims = [c for s in brief["sections"] for c in s["claims"]]
    assert {i for c in claims for i in c["assertion_ids"]} == {ids.new, ids.why}
    assert "history" not in brief
    error, with_history = call(client, reader, "context_brief", {"entity": entity, "include_history": True})
    assert not error and [h["id"] for h in with_history["history"]] == [ids.old]


def test_mcp_refusals_and_feedback(client, monkeypatch):
    ids = seed()
    _, reader = mint(client, "context:read")
    _, writer = mint(client, "context:write")
    feedback = {"request_id": str(uuid4()), "target_type": "assertion", "target_id": ids.new,
                "expected_revision": 1, "suggestion": "wrong", "explanation": "Contradicted on 11-04."}
    error, refused = call(client, reader, "context_feedback", feedback)
    assert error and refused["error"]["code"] == "INSUFFICIENT_SCOPE"
    error, invalid = call(client, reader, "context_search", {"limit": 5})
    assert error and invalid["error"]["code"] == "INVALID_ARGUMENT"
    error, receipt = call(client, writer, "context_feedback", feedback)
    assert not error and (receipt["origin"], receipt["applied"]) == ("agent", False)
    _, tasks_only = mint(client, "tasks:read")
    error, denied = call(client, tasks_only, "context_search", {"query": "pilot"})
    assert error and denied["error"]["code"] == "INSUFFICIENT_SCOPE"
    monkeypatch.setenv("JARVIS_CONTEXT_ENABLED", "false")
    get_settings.cache_clear()
    error, off = call(client, reader, "context_search", {"query": "pilot"})
    assert error and off["error"]["code"] == "CONTEXT_DISABLED"


def test_definitions_follow_scopes():
    reads = {d["name"] for d in definitions({"context:read"})}
    assert reads >= set(context_mcp.TOOLS) - {"context_feedback"} and "context_feedback" not in reads
    assert not {d["name"] for d in definitions({"tasks:read"})} & set(context_mcp.TOOLS)
