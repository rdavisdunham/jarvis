"""One-click setup (task bbdb6440) and the guided readiness page (task 0a1274a7, Astra review E-Q1).

Enable mints the link key and the MCP key and hands both over in one setup code for `remotea eridani setup`; the
setup page, the probe route, control_status and Eri's start proposal all say what stands in a run's way the same
way, each problem with its fix."""

import base64
import json
import logging
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from jarvis import agent_control
from jarvis.api import app
from jarvis.auth import digest
from jarvis.db import session_scope
from jarvis.domain import DomainError
from jarvis.models import BotCredential, now
from jarvis.remotea_routes import MCP_KEY_NAME, MCP_KEY_SCOPES
from jarvis.session_models import AgentHost
from sqlalchemy import select
from test_agent_link import bearer, connect, enable, hello, refused
from test_agent_runs import (
    CAPS,
    host,  # noqa: F401 - fixture
    quiet,  # noqa: F401 - fixture
    start,
    task,
)
from test_agent_security import V2_HARNESSES, V2_TARGET

PREFIX = "eridani-setup:1:"


def decode(code):
    """What remotea's parseSetupCode does: base64url (no padding) JSON after the prefix."""
    assert code.startswith(PREFIX)
    raw = code[len(PREFIX):]
    assert "=" not in raw and all(c.isalnum() or c in "-_" for c in raw)
    return json.loads(base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)))


def keys(name):
    with session_scope() as db:
        return [(row.id, row.revoked_at, sorted(row.scopes)) for row in db.scalars(
            select(BotCredential).where(BotCredential.name == name).order_by(BotCredential.created_at))]


def capabilities(token):
    response = TestClient(app).get("/api/v1/external/capabilities", headers={"Authorization": "Bearer " + token})
    return response


# ---------------------------------------------------------------- setup code (bbdb6440)


def test_enable_mints_the_link_key_and_an_mcp_key_in_one_setup_code(client, caplog):
    caplog.set_level(logging.INFO)
    body = enable(client)
    code = decode(body["setup_code"])
    assert set(code) == {"url", "link_key", "mcp_key"}
    assert code["url"] == "http://testserver" or code["url"].startswith("http")
    assert code["link_key"] == body["token"] and code["mcp_key"].startswith("eri_bot_")
    assert code["mcp_key"] != code["link_key"]

    # Exactly the MCP server's scopes; the link key keeps its own and never gets tasks:write.
    [(mcp_id, revoked, scopes)] = keys(MCP_KEY_NAME)
    assert revoked is None
    assert scopes == ["agents:session", "notes:read", "notes:write", "sessions:read", "sessions:write",
                      "tasks:read", "tasks:write"]
    assert set(MCP_KEY_SCOPES) <= set(scopes) and "agents:control" not in scopes
    [(_, _, link_scopes)] = keys("remotea")
    assert "tasks:write" not in link_scopes and "agents:control" in link_scopes

    # Only hashes are stored, and neither key nor the code reaches the log.
    with session_scope() as db:
        row = db.get(BotCredential, mcp_id)
        assert row.token_hash == digest(code["mcp_key"]) and row.prefix == code["mcp_key"][:16]
    for secret in (code["mcp_key"], code["link_key"], body["setup_code"]):
        assert secret not in caplog.text
    assert mcp_id in caplog.text  # The audit line names the credential.

    # The MCP key offers the run tools and session_link, and cannot open the link.
    tools = {t["name"] for t in capabilities(code["mcp_key"]).json()["tools"]}
    assert {"task_plan", "task_step", "task_interview", "submit_plan", "session_link"} <= tools
    assert refused(bearer(code["mcp_key"])).status_code == 403

    # Status shows the MCP key's prefix, never a token; enabling again mints nothing.
    status = client.get("/api/v1/remotea/status").json()
    assert status["mcp_key"]["prefix"] == code["mcp_key"][:16] and "token" not in status["mcp_key"]
    assert status["setup_command"] == "remotea eridani setup"
    again = enable(client)
    assert again["token"] is None and again["setup_code"] is None
    assert len(keys(MCP_KEY_NAME)) == 1


def test_a_new_setup_code_revokes_the_previous_keys(client):
    first = decode(enable(client)["setup_code"])
    response = client.post("/api/v1/remotea/setup-code", json={"machine": "desk"})
    assert response.status_code == 200, response.text
    second = decode(response.json()["setup_code"])
    assert second["machine"] == "desk" and second["link_key"] == response.json()["token"]
    assert second["mcp_key"] != first["mcp_key"] and second["link_key"] != first["link_key"]
    mcp = keys(MCP_KEY_NAME)
    assert len(mcp) == 2 and mcp[0][1] is not None and mcp[1][1] is None
    assert capabilities(first["mcp_key"]).status_code == 401
    assert capabilities(second["mcp_key"]).status_code == 200
    assert refused(bearer(first["link_key"])).status_code == 401
    # No body works too; no machine then.
    assert "machine" not in decode(client.post("/api/v1/remotea/setup-code").json()["setup_code"])
    assert client.post("/api/v1/remotea/setup-code", json={"machine": "x" * 101}).status_code == 422


def test_new_link_key_keeps_the_mcp_key_and_disable_revokes_both(client):
    first = decode(enable(client)["setup_code"])
    rotated = enable(client, rotate=True)
    code = decode(rotated["setup_code"])
    assert "mcp_key" not in code and code["link_key"] == rotated["token"] != first["link_key"]
    assert capabilities(first["mcp_key"]).status_code == 200

    client.post("/api/v1/remotea/disable")
    assert capabilities(first["mcp_key"]).status_code == 401
    assert all(revoked is not None for _, revoked, _ in keys(MCP_KEY_NAME) + keys("remotea"))
    refused_code = client.post("/api/v1/remotea/setup-code", json={})
    assert refused_code.status_code == 403 and refused_code.json()["error"]["code"] == "REMOTEA_DISABLED"
    assert decode(enable(client)["setup_code"])["mcp_key"].startswith("eri_bot_")


def test_setup_code_is_owner_and_personal_workspace_only(client):
    from test_accounts import client_for, shared

    enable(client)
    guest = client_for("guest")
    shared(client, guest)
    for who in (client, guest):
        for method, path in (("post", "/setup-code"), ("get", "/setup"), ("post", "/targets/x/probe")):
            response = getattr(who, method)("/api/v1/remotea" + path)
            assert response.status_code == 403, (path, response.text)


# ---------------------------------------------------------------- readiness page (E-Q1)


def v2_hello(token, targets=(V2_TARGET,), **values):
    with connect(bearer(token)) as ws:
        hello(ws, caps=CAPS, harnesses=V2_HARNESSES, caps_version=2, targets=list(targets), **values)
    with session_scope() as db:
        db.scalar(select(AgentHost)).lease_until = now() + timedelta(hours=1)


def test_setup_page_walks_host_targets_and_fixes(client, host):  # noqa: F811
    old = {"id": "legacy", "repo": "/home/davin/legacy", "worktree_root": "/home/davin/lwt", "harnesses": ["claude"],
           "verify": ["tests"]}
    v2_hello(host.token, targets=(V2_TARGET, old), build_sha="a" * 40)
    body = client.get("/api/v1/remotea/setup").json()
    assert body["ready"] is True  # The legacy target can start a run; jarvis cannot.
    [h] = body["hosts"]
    assert h["connected"] is True and h["build"] == "aaaaaaa" and h["caps_version"] == 2 and h["problems"] == []
    assert h["hello_at"]

    targets = {t["host_target"]: t for t in body["targets"]}
    jarvis = targets["jarvis"]
    assert jarvis["execution_target_id"] == host.target_id and jarvis["key"] == host.target_id
    assert jarvis["harnesses"]["claude"] == {"label": "Claude Code", "ready": True, "enabled": True,
                                             "version": "2.1.312", "error": None, "launch": "probed", "reason": None}
    assert jarvis["harnesses"]["codex"]["ready"] is False and jarvis["harnesses"]["codex"]["error"] == "codex: command not found"
    codes = {p["code"]: p for p in jarvis["problems"]}
    assert set(codes) == {"HARNESS_NOT_INSTALLED", "VERIFY_NOT_READY", "MCP_NOT_CONFIGURED"}
    assert codes["HARNESS_NOT_INSTALLED"]["message"].startswith("Codex is enabled but not installed")
    assert codes["HARNESS_NOT_INSTALLED"]["blocks"] is False
    assert "scripts/verify.sh" in codes["VERIFY_NOT_READY"]["fix"] and "env.PATH" in codes["VERIFY_NOT_READY"]["fix"]
    assert codes["MCP_NOT_CONFIGURED"]["blocks"] is True and "remotea eridani setup" in codes["MCP_NOT_CONFIGURED"]["fix"]
    assert jarvis["ready"] is False and jarvis["line"].startswith("jarvis on desk: Agents on jarvis get no Eridani tools")

    legacy = targets["legacy"]  # Advertised only, by an older remotea: facts unknown, not problems.
    assert legacy["execution_target_id"] is None and legacy["key"] == f"host:{host.id}:legacy"
    assert legacy["harnesses"]["claude"]["ready"] is True and legacy["harnesses"]["claude"]["launch"] == "unknown (host too old)"
    assert legacy["verify"] is None and legacy["mcp"] is None and legacy["problems"] == []
    assert "Verify readiness: unknown (host too old)." in legacy["notes"] and legacy["ready"] is True

    assert body["lines"] == [jarvis["line"]]
    with session_scope() as db:
        status = agent_control.status(db, "davin", {})
    assert status["setup"] == [jarvis["line"]] and "Settings → Connections → remotea" in status["setup_hint"]


def test_probe_reads_the_last_hello_and_says_so(client, host):  # noqa: F811
    v2_hello(host.token)
    saved = client.post(f"/api/v1/remotea/targets/{host.target_id}/probe")
    assert saved.status_code == 200, saved.text
    body = saved.json()
    assert body["source"] == "hello" and body["hello_at"] and "no readiness probe" in body["note"]
    assert body["host"]["connected"] is True and body["target"]["host_target"] == "jarvis"
    assert {p["code"] for p in body["target"]["problems"]} >= {"MCP_NOT_CONFIGURED"}
    advertised = client.post(f"/api/v1/remotea/targets/host:{host.id}:jarvis/probe").json()
    assert advertised["target"]["problems"] == body["target"]["problems"]
    for key in (f"host:{host.id}:nope", "00000000-0000-4000-8000-000000000000", "not-a-target"):
        missing = client.post(f"/api/v1/remotea/targets/{key}/probe")
        assert missing.status_code == 404 and missing.json()["error"]["code"] == "NOT_FOUND", key

    # Fixed on the host: MCP configured, verify on PATH. The target is ready; isolation unknown is only a note.
    fixed = {**V2_TARGET, "mcp": {"configured": True},
             "verify_status": [{"id": "tests", "ready": True, "isolated": False, "isolation": "bwrap",
                                "isolation_reason": "bwrap (bubblewrap) is not installed on the service PATH"},
                               {"id": "lint", "ready": True, "isolated": False, "isolation": "none",
                                "isolation_reason": 'the entry sets isolation "none"'}]}
    v2_hello(host.token, targets=(fixed,))
    target = client.post(f"/api/v1/remotea/targets/{host.target_id}/probe").json()["target"]
    assert target["ready"] is True and target["line"] is None
    assert target["verify"][0]["isolation_reason"].startswith("bwrap (bubblewrap)")
    assert target["notes"] == ["Verify tests runs without a sandbox: bwrap (bubblewrap) is not installed on the "
                               "service PATH."]
    page = client.get("/api/v1/remotea/setup").json()
    assert page["ready"] is True and page["lines"] == []
    with session_scope() as db:
        assert "setup" not in agent_control.status(db, "davin", {})


def test_offline_degraded_and_missing_targets_each_get_one_line(client, host):  # noqa: F811
    v2_hello(host.token, targets=(), degraded_reason="journal_unavailable (write failed: ENOSPC)")
    body = client.get("/api/v1/remotea/setup").json()
    [h] = body["hosts"]
    assert [p["code"] for p in h["problems"]] == ["HOST_DEGRADED"] and "free disk space" in h["problems"][0]["fix"]
    [saved] = body["targets"]  # The saved target its host stopped advertising.
    assert saved["advertised"] is False and [p["code"] for p in saved["problems"]] == ["TARGET_NOT_ADVERTISED"]
    assert 'id "jarvis"' in saved["problems"][0]["fix"]
    assert body["lines"] == [
        "desk: Host journal degraded: journal_unavailable (write failed: ENOSPC). Fix: " + h["problems"][0]["fix"],
        saved["line"]]

    with session_scope() as db:
        db.get(AgentHost, host.id).lease_until = now() - timedelta(seconds=1)
    offline = client.get("/api/v1/remotea/setup").json()
    assert offline["ready"] is False
    assert offline["hosts"][0]["problems"][0]["code"] == "HOST_OFFLINE"
    assert "systemctl --user start remotea" in offline["hosts"][0]["problems"][0]["fix"]
    assert offline["lines"][0].startswith("desk: The remotea server desk is not connected")


def test_no_host_yet_says_to_run_setup(client):
    enable(client)
    body = client.get("/api/v1/remotea/setup").json()
    assert body == {"ready": False, "hosts": [], "targets": [], "lines": [
        "No remotea server has connected yet. Fix: run `remotea eridani setup` in WSL and paste a setup code from "
        "Settings → Connections → remotea (New setup code) when asked."]}


def test_eri_hears_the_setup_problem_before_confirming_a_start(client, host):  # noqa: F811
    """A start check_host allows (Claude is installed) still stalls when the agent gets no Eridani tools; the
    proposal tells Eri to say so, with the fix, before the owner confirms."""
    from test_agent_voice import say

    v2_hello(host.token)
    created = task("Needs tools", execution_target_id=host.target_id)
    told = say("control.start", task_id=created["id"])
    # The start also reports MCP_UNCONFIGURED (it blocks unless the run is freestyle); the setup line adds the fix.
    assert told["status"] == "pending_confirmation"
    assert [p["code"] for p in told["problems"]] == ["MCP_UNCONFIGURED"] and told["problems"][0]["freestyle_ok"] is True
    assert "Also tell the owner before they confirm: jarvis on desk: Agents on jarvis get no Eridani tools" in told["next"]
    assert "remotea eridani setup" in told["next"]

    v2_hello(host.token, targets=({**V2_TARGET, "mcp": {"configured": True}},))
    other = task("Has tools", execution_target_id=host.target_id)
    assert "Also tell the owner" not in say("control.start", task_id=other["id"]).get("next", "")


# ---------------------------------------------------------------- target templates (remotea S-Q3)

TEMPLATE = {"ready": False, "problems": [
    {"check": "migrations", "level": "warn", "reason": "migrations.dir is not a folder in the repository checkout"},
    {"check": "verify.tests", "level": "warn", "reason": "its program is not on the service PATH (uv)"},
    {"check": "repo", "level": "fail", "reason": "/home/davin/jarvis is not a git checkout"},
    {"check": "prepare.web", "level": "fail", "reason": "npm is not on the service PATH"}]}
NOT_READY = {**V2_TARGET, "harnesses": [], "mcp": {"configured": True}, "template": TEMPLATE, "prepare": ["api", "web"]}
WHY = ("The jarvis target is not ready: repo: /home/davin/jarvis is not a git checkout; prepare.web: npm is not on "
       "the service PATH.")


def test_hello_keeps_template_and_prepare_and_drops_malformed_ones():
    from jarvis.agent_link import Hello

    frame = Hello.model_validate({"t": "hello", "server_id": "s1", "targets": [
        {"id": "a", "template": {"ready": False, "problems": [
            {"check": "base", "level": "fail", "reason": "  main is\nnot a branch  "},
            {"check": "repo", "level": "fatal", "reason": "unknown level"}, {"level": "fail", "reason": "no check"},
            "junk", {"check": "launch", "level": "warn", "reason": "x" * 400}]}, "prepare": ["api", 7, "web"]},
        {"id": "b", "template": {"ready": "yes"}, "prepare": "api"},
        {"id": "c", "template": {"ready": True}},
        {"id": "d", "repo": "/r"}]})
    a, b, c, d = frame.targets
    assert a["template"] == {"ready": False, "problems": [
        {"check": "base", "level": "fail", "reason": "main is not a branch"},
        {"check": "launch", "level": "warn", "reason": "x" * 300}]}  # Fails first; one clipped line each.
    assert a["prepare"] == ["api", "web"]
    assert b == {"id": "b"}  # Malformed: dropped, never refused.
    assert c["template"] == {"ready": True, "problems": []} and "prepare" not in c
    assert d == {"id": "d", "repo": "/r"}  # An older host: nothing added.


def test_a_template_that_is_not_ready_explains_itself(client, host):  # noqa: F811
    v2_hello(host.token, targets=(NOT_READY,))
    target = next(t for t in client.get("/api/v1/remotea/setup").json()["targets"] if t["host_target"] == "jarvis")
    assert target["template"]["ready"] is False and target["prepare"] == ["api", "web"]
    assert [p["check"] for p in target["template"]["problems"]][:2] == ["repo", "prepare.web"]  # Fails first.
    codes = [p["code"] for p in target["problems"]]
    # Not "no coding agent can start" or "Claude is enabled but not installed": the template's own reasons.
    assert "TARGET_NOT_READY" in codes and "NO_HARNESS" not in codes and "HARNESS_NOT_INSTALLED" not in codes
    problem = next(p for p in target["problems"] if p["code"] == "TARGET_NOT_READY")
    assert problem["message"] == WHY and problem["blocks"] is True and "remotea doctor" in problem["fix"]
    assert target["harnesses"]["claude"]["reason"] == WHY
    assert "Template migrations: migrations.dir is not a folder in the repository checkout." in target["notes"]
    assert not any("verify.tests" in n for n in target["notes"])  # VERIFY_NOT_READY covers it.
    assert target["ready"] is False and target["line"].startswith("jarvis on desk: The jarvis target is not ready")

    created = task("Template", execution_target_id=host.target_id)
    body = client.get(f"/api/v1/remotea/runs/preflight?task_id={created['id']}").json()
    assert body["problems"] == [{"code": "TARGET_NOT_READY", "message": WHY, "blocking": True}]
    assert body["harnesses"] == {"claude": False, "codex": False}
    assert body["harness_reasons"] == {"claude": WHY, "codex": WHY}
    assert body["template"]["ready"] is False
    with pytest.raises(DomainError) as caught:
        start(host, created["id"])
    assert caught.value.code == "TARGET_NOT_READY" and caught.value.message == WHY
    assert caught.value.data["template"]["problems"][0]["check"] == "repo"

    listed = client.get("/api/v1/remotea/targets").json()
    assert listed["advertised"][0]["template"]["ready"] is False and listed["advertised"][0]["prepare"] == ["api", "web"]
    saved = next(t for t in listed["items"] if t["id"] == host.target_id)
    assert saved["template"]["ready"] is False

    ready = {**NOT_READY, "harnesses": ["claude"], "template": {"ready": True, "problems": [TEMPLATE["problems"][0]]}}
    v2_hello(host.token, targets=(ready,))
    fixed = next(t for t in client.get("/api/v1/remotea/setup").json()["targets"] if t["host_target"] == "jarvis")
    assert "TARGET_NOT_READY" not in [p["code"] for p in fixed["problems"]] and fixed["ready"] is True
    again = client.get(f"/api/v1/remotea/runs/preflight?task_id={created['id']}").json()
    assert again["harnesses"]["claude"] is True and not [p for p in again["problems"] if p["code"] == "TARGET_NOT_READY"]
