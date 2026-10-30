from datetime import timedelta
import json

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from jarvis import agent_sessions as sessions
from jarvis.api import app
from jarvis.config import get_settings
from jarvis.db import session_scope
from jarvis.external_mcp import definitions, dispatch
from jarvis.bot_access import bind
from jarvis.domain import DomainError
from jarvis.models import Job, LinearIssue, now
from jarvis.session_models import AgentSession
from sqlalchemy import func, select
from test_organization_foundation import OWNER, create


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    monkeypatch.setenv("JARVIS_INTEGRATION_ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("JARVIS_OPENAI_API_KEY", "synthetic")
    monkeypatch.setenv("JARVIS_COST_TRACKING_ENABLED", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def key(client, scopes):
    response = client.post(
        "/api/v1/bot-keys", json={"name": "Collector", "scopes": scopes, "expires_in_days": 30}
    )
    assert response.status_code == 201, response.text
    value = response.json()
    return value["credential"], {"Authorization": "Bearer " + value["token"]}


def issue(identifier, title="Fix login"):
    record = create("task", title)
    with session_scope() as db:
        db.add(
            LinearIssue(
                owner_id=OWNER,
                workspace_id="acme",
                remote_id="remote-" + identifier,
                task_id=record["task_id"],
                snapshot={"identifier": identifier, "url": f"https://linear.app/acme/issue/{identifier}/x"},
            )
        )
    return record


def report(headers, session_id="s1", tool="claude_code", **values):
    body = {
        "environment": "wsl",
        "cwd": "/home/davin/acme-app",
        "repo": "acme-app",
        "branch": "main",
        "title": "Investigate flaky login test",
        "first_prompt": "The login test fails on CI",
        "last_active_at": now().isoformat(),
        "resume": {"cmd": f"claude --resume {session_id}"},
        **values,
    }
    return TestClient(app).put(f"/api/v1/external/sessions/{tool}/{session_id}", headers=headers, json=body)


def test_paths_compare_across_wsl_and_windows_spellings():
    assert sessions.normalize_path("//wsl.localhost/Ubuntu/home/davin/x/") == "/home/davin/x"
    assert sessions.normalize_path("\\\\wsl$\\Ubuntu\\home\\davin") == "/home/davin"
    assert sessions.normalize_path("/mnt/c/Users/davin/Documents") == "C:/Users/davin/Documents"
    assert sessions.normalize_path("c:\\Users\\davin") == "C:/Users/davin"
    assert sessions.under("C:/Users/Davin/Documents/acme", "C:/users/davin/documents")
    assert not sessions.under("/home/davin/acme-app2", "/home/davin/acme-app")
    assert sessions.issue_keys("davin/eng-12-fix and ENG-12, gpt-5") == ["ENG-12", "GPT-5"]


def test_reports_are_idempotent_and_filed_by_branch_issue(client):
    target = issue("ENG-12")
    _, headers = key(client, ["sessions:write", "records:read"])
    first = report(headers, branch="davin/eng-12-flaky-login")
    assert first.status_code == 200, first.text
    again = report(headers, branch="davin/eng-12-flaky-login", turn_count=4)
    assert again.json()["id"] == first.json()["id"]
    filed = again.json()["filed"]
    assert filed == {"source": "branch", "record_id": target["id"], "task_id": target["task_id"], "record_title": "Fix login"}
    assert again.json()["issue_refs"][0]["identifier"] == "ENG-12"
    with session_scope() as db:
        assert db.scalar(select(func.count()).select_from(AgentSession)) == 1


def test_mentions_file_only_when_unambiguous_and_unknown_keys_never_file(client):
    one, two = issue("ENG-1", "One"), issue("ENG-2", "Two")
    _, headers = key(client, ["sessions:write"])
    single = report(headers, "a", first_prompt="Pick up ENG-1 using gpt-5 and utf-8").json()
    assert single["filed"]["source"] == "mention" and single["filed"]["record_id"] == one["id"]
    assert single["filed"]["record_title"] is None  # Without records:read the key never sees record titles.
    both = report(headers, "b", first_prompt="Compare ENG-1 with ENG-2").json()
    assert both["filed"]["record_id"] is None
    assert [r["identifier"] for r in both["issue_refs"]] == ["ENG-1", "ENG-2"]
    branch_wins = report(headers, "c", branch="eng-2-fix", first_prompt="Like ENG-1").json()
    assert branch_wins["filed"]["record_id"] == two["id"]


def test_folder_rules_file_sessions_and_manual_filing_survives(client):
    acme = create("client", "Acme")
    project = create("project", "Acme portal", parent_id=acme["id"])
    _, headers = key(client, ["sessions:write"])
    report(headers, "rule", cwd="//wsl.localhost/Ubuntu/home/davin/acme-app/web")
    report(headers, "manual", cwd="/home/davin/acme-app")
    row = next(i for i in client.get("/api/v1/sessions").json()["items"] if i["session_id"] == "manual")
    assert client.post(f"/api/v1/sessions/{row['id']}/link", json={"record_id": project["id"]}).status_code == 200
    rules = client.post("/api/v1/sessions/rules", json={"path_prefix": "/home/davin/acme-app/", "record_id": acme["id"]})
    assert rules.status_code == 200, rules.text
    assert rules.json()["items"][0]["path_prefix"] == "/home/davin/acme-app"
    listed = {i["session_id"]: i["filed"] for i in client.get("/api/v1/sessions").json()["items"]}
    assert listed["rule"]["source"] == "path_rule" and listed["rule"]["record_title"] == "Acme"
    assert listed["manual"] == {"source": "manual", "record_id": project["id"], "task_id": project.get("task_id"), "record_title": "Acme portal"}
    # A client's sessions include those filed under its projects.
    scoped = client.get("/api/v1/sessions", params={"record_id": acme["id"]}).json()
    assert {i["session_id"] for i in scoped["items"]} == {"rule", "manual"}
    rule_id = rules.json()["items"][0]["id"]
    assert client.delete(f"/api/v1/sessions/rules/{rule_id}").status_code == 200
    unfiled = client.get("/api/v1/sessions", params={"unfiled": True}).json()
    assert [i["session_id"] for i in unfiled["items"]] == ["rule"]


def test_bot_keys_manage_folder_rules(client):
    acme = create("client", "Acme")
    credential, headers = key(client, ["sessions:write", "sessions:read"])
    _, reader = key(client, ["sessions:read"])
    report(headers, "rule", cwd="/home/davin/acme-app/web")
    external = TestClient(app)
    body = {"path_prefix": "/home/davin/acme-app", "record_id": acme["id"]}
    assert external.post("/api/v1/external/sessions/rules", headers=reader, json=body).status_code == 403
    saved = external.post("/api/v1/external/sessions/rules", headers=headers, json=body)
    assert saved.status_code == 200, saved.text
    assert external.get("/api/v1/external/sessions/rules", headers=reader).json() == saved.json()
    filed = external.get("/api/v1/external/sessions", headers=headers).json()["items"][0]["filed"]
    assert filed["source"] == "path_rule" and filed["record_id"] == acme["id"]
    rule_id = saved.json()["items"][0]["id"]
    assert external.delete(f"/api/v1/external/sessions/rules/{rule_id}", headers=headers).json() == {"items": []}
    names = [d["name"] for d in definitions(credential["scopes"])]
    assert {"session_rule", "session_rules"} <= set(names)
    with bind(credential["id"]):
        assert dispatch("session_rule", {"path_prefix": "/home/davin/acme-app/", "record_id": acme["id"]})["items"][0]["path_prefix"] == "/home/davin/acme-app"
        assert len(dispatch("session_rules", {})["items"]) == 1



@pytest.mark.parametrize("can_read_records", [False, True])
def test_folder_rule_responses_respect_record_title_scope(client, can_read_records):
    first, second = create("client", "Private client title"), create("project", "Private project title")
    scopes = ["sessions:write"] + (["records:read"] if can_read_records else [])
    credential, headers = key(client, scopes)
    _, reader = key(client, ["sessions:read"] + (["records:read"] if can_read_records else []))
    external = TestClient(app)
    first_body = {"path_prefix": "/work/first", "record_id": first["id"]}
    second_body = {"path_prefix": "/work/second", "record_id": second["id"]}
    # Browser callers retain their normal authorized record labels.
    browser_rules = client.post("/api/v1/sessions/rules", json=first_body).json()
    assert browser_rules["items"][0]["record_title"] == first["title"]

    def check(response, count):
        items = response["items"]
        assert len(items) == count
        expected = {first["id"]: first["title"], second["id"]: second["title"]}
        for item in items:
            assert item["record_title"] == (expected[item["record_id"]] if can_read_records else None)

    check(external.get("/api/v1/external/sessions/rules", headers=reader).json(), 1)
    saved = external.post("/api/v1/external/sessions/rules", headers=headers, json=second_body)
    assert saved.status_code == 200, saved.text
    check(saved.json(), 2)
    with bind(credential["id"]):
        check(dispatch("session_rules", {}), 2)
        check(dispatch("session_rule", second_body), 2)
    second_id = next(r["id"] for r in saved.json()["items"] if r["record_id"] == second["id"])
    deleted = external.delete(f"/api/v1/external/sessions/rules/{second_id}", headers=headers)
    assert deleted.status_code == 200, deleted.text
    check(deleted.json(), 1)


def test_search_ranks_issue_keys_folders_and_text(client):
    issue("ENG-7", "Billing export")
    _, headers = key(client, ["sessions:write", "sessions:read"])
    report(headers, "billing", title="Billing CSV export", branch="eng-7-export", repo="ledger")
    report(headers, "other", title="Refactor auth middleware", repo="gateway", last_active_at=(now() - timedelta(days=2)).isoformat())
    external = TestClient(app)
    found = external.get("/api/v1/external/sessions", headers=headers, params={"query": "ENG-7"}).json()
    assert [i["session_id"] for i in found["items"]] == ["billing"]
    assert found["semantic"] is False
    assert [i["session_id"] for i in external.get("/api/v1/external/sessions", headers=headers, params={"query": "middleware"}).json()["items"]] == ["other"]
    assert [i["session_id"] for i in external.get("/api/v1/external/sessions", headers=headers, params={"query": "gateway"}).json()["items"]] == ["other"]
    recent = external.get("/api/v1/external/sessions", headers=headers).json()
    assert [i["session_id"] for i in recent["items"]] == ["billing", "other"]
    assert recent["items"][0]["resume"] == {"cmd": "claude --resume billing"}
    assert "excerpt" not in recent["items"][0]


def test_search_results_are_json_safe_as_eri_tool_results(client):
    """Eri's runners (work_runner, conversation, voice) json.dumps a tool result as is. On 2026-10-29 a chat
    turn that called session_search failed with "The task model could not finish this request" because the
    items carried datetime objects; the wire already shows ISO strings, so the search must return those."""
    _, headers = key(client, ["sessions:write", "sessions:read"])
    report(headers, "billing", title="Billing CSV export", started_at=(now() - timedelta(hours=1)).isoformat())
    with session_scope() as db:
        row = db.scalar(select(AgentSession).where(AgentSession.session_id == "billing"))
        row.state, row.state_since = "working", now()
    for listing in (sessions.Listing(), sessions.Listing(query="billing"), sessions.Listing(live=True)):
        found = sessions.search(OWNER, listing)
        assert [i["session_id"] for i in found["items"]] == ["billing"]
        json.dumps(found)  # TypeError before the fix: Object of type datetime is not JSON serializable
        item = found["items"][0]
        assert isinstance(item["started_at"], str) and isinstance(item["last_active_at"], str)
        assert isinstance(item["live_state"]["since"], str)
    # The HTTP route carried these as ISO strings before; the tool result now matches it exactly.
    over_http = TestClient(app).get("/api/v1/external/sessions", headers=headers).json()["items"][0]
    assert over_http["started_at"] == item["started_at"] and over_http["live_state"] == item["live_state"]


def test_scopes_and_input_limits(client):
    _, reader = key(client, ["sessions:read"])
    _, writer = key(client, ["tasks:write"])
    assert report(reader).status_code == 403
    assert report(writer).status_code == 403
    assert TestClient(app).get("/api/v1/external/sessions", headers=writer).status_code == 403
    _, headers = key(client, ["sessions:write"])
    assert report(headers, "bad id!").status_code in {400, 422}
    assert report(headers, tool="gemini").status_code == 422
    assert report(headers, resume={"url": "javascript:alert(1)"}).status_code == 422
    assert report(headers, pr_urls=["https://github.com/acme/app/pull/3", "https://github.com/acme/app/pull/3"]).json()["pr_urls"] == [
        "https://github.com/acme/app/pull/3"
    ]


def test_mcp_tools_link_before_the_collector_reports(client):
    target = issue("ENG-9")
    credential, _ = key(client, ["sessions:write"])
    names = [d["name"] for d in definitions(credential["scopes"])]
    assert {"session_search", "session_link"} <= set(names)
    with bind(credential["id"]):
        with pytest.raises(DomainError) as missing:
            dispatch("session_link", {"tool": "codex_cli", "session_id": "t1", "identifier": "ENG-404"})
        assert missing.value.code == "NOT_FOUND"
        with pytest.raises(DomainError):
            dispatch("session_link", {"tool": "codex_cli", "session_id": "t1", "identifier": "ENG-9", "clear": True})
        linked = dispatch("session_link", {"tool": "codex_cli", "session_id": "t1", "identifier": "eng-9"})
        assert linked["filed"]["source"] == "explicit" and linked["filed"]["task_id"] == target["task_id"]
        found = dispatch("session_search", {"unfiled": False})
        assert [i["session_id"] for i in found["items"]] == ["t1"]
    _, headers = key(client, ["sessions:write"])
    later = report(headers, "t1", tool="codex_cli", branch="main", cwd="/home/davin/elsewhere").json()
    assert later["filed"]["source"] == "explicit" and later["title"] == "Investigate flaky login test"


def test_settled_sessions_get_one_summary_and_semantic_search(client, monkeypatch):
    calls = []
    monkeypatch.setattr(sessions, "summarize_text", lambda owner, source: calls.append(source) or "Fixed the flaky login test.")
    monkeypatch.setattr(sessions, "embed", lambda owner, text: [1.0] + [0.0] * 511)
    monkeypatch.setattr(sessions, "query_vector", lambda owner, query: [1.0] + [0.0] * 511)
    _, headers = key(client, ["sessions:write", "sessions:read"])
    report(headers, "busy")
    settled = (now() - timedelta(minutes=30)).isoformat()
    report(headers, "quiet", last_active_at=settled)
    report(headers, "quiet", last_active_at=settled)
    with session_scope() as db:
        jobs = list(db.scalars(select(Job).where(Job.kind == "summarize_session")))
    assert [j.payload["session"] for j in jobs] == [db_id("quiet")]
    sessions.summarize(jobs[0].id)
    assert calls[0]["title"] == "Investigate flaky login test"
    report(headers, "quiet", last_active_at=settled)
    with session_scope() as db:
        assert db.scalar(select(func.count()).select_from(Job).where(Job.kind == "summarize_session")) == 1
        assert db.get(Job, jobs[0].id).status == "succeeded"
    found = TestClient(app).get("/api/v1/external/sessions", headers=headers, params={"query": "authentication bug"}).json()
    assert found["semantic"] is True
    assert [(i["session_id"], i["summary"]) for i in found["items"]] == [("quiet", "Fixed the flaky login test.")]


def db_id(session_id):
    with session_scope() as db:
        return db.scalar(select(AgentSession.id).where(AgentSession.session_id == session_id))


def test_archived_sessions_leave_listings(client):
    _, headers = key(client, ["sessions:write"])
    row = report(headers, "noise").json()
    assert client.post(f"/api/v1/sessions/{row['id']}/archive", json={"archived": True}).status_code == 200
    assert client.get("/api/v1/sessions").json()["items"] == []


def test_live_terminals_list_separately_and_never_trigger_a_new_summary(client):
    _, headers = key(client, ["sessions:write", "sessions:read"])
    live = {"state": "live", "workspace": "Acme", "terminal": "surf-1", "host": "DavisPC"}
    running = report(headers, "running", resume={"hint": "Live in remotea → Acme", "live": live})
    assert running.status_code == 200, running.text
    assert running.json()["resume"] == {"hint": "Live in remotea → Acme", "live": live}
    report(headers, "stopped")
    listed = client.get("/api/v1/sessions", params={"live": "true"}).json()["items"]
    assert [s["session_id"] for s in listed] == ["running"]
    assert TestClient(app).get("/api/v1/external/sessions", params={"live": "true"}, headers=headers).json()["items"][0]["resume"]["live"] == live
    with session_scope() as db:
        before = db.scalar(select(AgentSession.fingerprint).where(AgentSession.session_id == "running"))
    closed = report(headers, "running", resume={"cmd": "remotea open --group Acme -- claude --resume running"})
    assert "live" not in closed.json()["resume"]
    with session_scope() as db:
        assert db.scalar(select(AgentSession.fingerprint).where(AgentSession.session_id == "running")) == before
    assert client.get("/api/v1/sessions", params={"live": "true"}).json()["items"] == []
    for bad in ({**live, "state": "exited"}, {**live, "workspace": "x" * 201}, {**live, "pid": 1}):
        assert report(headers, "bad", resume={"live": bad}).status_code == 422
