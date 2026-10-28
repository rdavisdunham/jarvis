import json
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from jarvis import agent_link
from jarvis.api import app
from jarvis.db import session_scope
from jarvis.models import BotCredential, Event, now
from jarvis.session_models import AgentHost, AgentParity, AgentSession, AgentTurn
from sqlalchemy import func, select
from starlette.testclient import WebSocketDenialResponse
from starlette.websockets import WebSocketDisconnect
from test_agent_sessions import key, report

PATH = "/api/v1/external/agents/link"


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setenv("JARVIS_COST_TRACKING_ENABLED", "false")
    from jarvis.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def enable(client, **body):
    response = client.post("/api/v1/remotea/enable", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def bearer(token):
    return {"Authorization": "Bearer " + token, "User-Agent": "remotea-server/test"}


def connect(headers, **kwargs):
    return TestClient(app).websocket_connect(PATH, headers=headers, subprotocols=["eridani.v1"], **kwargs)


frames = iter(range(1, 10**9))


def frame(t, **values):
    return {"v": 1, "id": f"m{next(frames)}", "t": t, "at": now().isoformat(), **values}


def hello(ws, server_id="desk-1", **values):
    ws.send_json(frame("hello", server_id=server_id, boot_id="boot-1", hostname="desk", machine="WSL",
                       proto=1, caps=["session.report", "turn", "status"], **values))
    welcome = ws.receive_json()
    assert welcome["t"] == "welcome", welcome
    return welcome


def send(ws, t, **values):
    message = frame(t, **values)
    ws.send_json(message)
    reply = ws.receive_json()
    assert reply["t"] == "ack" and reply["acked_id"] == message["id"], reply
    return reply


def refused(headers, **kwargs):
    with pytest.raises(WebSocketDenialResponse) as caught, connect(headers, **kwargs):
        pass
    return caught.value


SESSION = {
    "environment": "wsl", "cwd": "/home/davin/acme-app", "repo": "acme-app", "branch": "main",
    "title": "Investigate flaky login test", "first_prompt": "The login test fails on CI",
    "resume": {"cmd": "claude --resume s1"},
}


def test_handshake_refuses_bad_keys_missing_scope_origin_and_subprotocol(client):
    token = enable(client)["token"]
    assert refused(bearer("eri_bot_not-a-real-key")).status_code == 401
    _, plain = key(client, ["sessions:write"])
    denied = refused(plain)
    assert denied.status_code == 403 and denied.json()["error"]["code"] == "INSUFFICIENT_SCOPE"
    assert refused({**bearer(token), "Origin": "http://testserver"}).status_code == 403
    with pytest.raises(WebSocketDenialResponse) as caught, TestClient(app).websocket_connect(PATH, headers=bearer(token)):
        pass
    assert caught.value.status_code == 400


def test_hello_welcome_takes_a_lease_and_reports_the_host(client):
    token = enable(client)["token"]
    with connect(bearer(token)) as ws:
        assert ws.accepted_subprotocol == "eridani.v1"
        welcome = hello(ws)
        assert welcome["conn_epoch"] == 1 and welcome["lease_seconds"] == 90
        assert {"turn", "status", "session.report"} <= set(welcome["caps"])
        hosts = client.get("/api/v1/remotea/status").json()["hosts"]
        assert [(h["id"], h["name"], h["connected"]) for h in hosts] == [(welcome["host_id"], "desk", True)]
    hosts = client.get("/api/v1/remotea/status").json()["hosts"]
    assert hosts[0]["connected"] is False and hosts[0]["last_seen_at"]


def test_hello_must_come_first(client):
    token = enable(client)["token"]
    with connect(bearer(token)) as ws:
        ws.send_json(frame("turn", tool="claude", provider_session_id="s1", provider_turn_id="t1"))
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_json()
    assert closed.value.code == 4400


def test_second_connection_supersedes_the_first(client):
    token = enable(client)["token"]
    with connect(bearer(token)) as first:
        hello(first)
        with connect(bearer(token)) as second:
            welcome = hello(second)
            assert welcome["conn_epoch"] == 2
            bye = first.receive_json()
            assert bye["t"] == "bye" and bye["reason"] == "superseded"
            with pytest.raises(WebSocketDisconnect) as closed:
                first.receive_json()
            assert closed.value.code == 4001
            send(second, "session.report", tool="claude_code", session_id="s1", report=SESSION)
        with session_scope() as db:
            assert db.scalar(select(AgentHost.conn_epoch)) == 2


def test_stale_epoch_is_fenced_in_the_database(client):
    token = enable(client)["token"]
    with connect(bearer(token)) as ws:
        hello(ws)
        with session_scope() as db:
            db.scalar(select(AgentHost)).conn_epoch = 9  # Another replica admitted a newer connection.
        ws.send_json(frame("session.report", tool="claude_code", session_id="s1", report=SESSION))
        bye = ws.receive_json()
        assert bye["t"] == "bye" and bye["reason"] == "superseded"
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_json()
    assert closed.value.code == 4001
    with session_scope() as db:
        assert db.scalar(select(func.count()).select_from(AgentSession)) == 0
        assert db.scalar(select(AgentHost.lease_until)) > now()  # The newer connection keeps its lease.


def test_session_report_over_the_link_equals_the_http_put(client):
    token = enable(client)["token"]
    stamp = now().isoformat()
    http = report(bearer(token), session_id="s1", last_active_at=stamp)
    assert http.status_code == 200, http.text
    with connect(bearer(token)) as ws:
        welcome = hello(ws)
        body = {**SESSION, "last_active_at": stamp, "resume": {"cmd": "claude --resume s1"}}
        http = report(bearer(token), session_id="s1", **body).json()
        ack = send(ws, "session.report", tool="claude_code", session_id="s1", report=body, surf="surf-7", incarnation=1)
    assert ack["ok"] and ack["outcome"]["session"] == http
    with session_scope() as db:
        row = db.scalar(select(AgentSession))
        assert (row.host_id, row.surf, row.incarnation) == (welcome["host_id"], "surf-7", 1)
        assert db.scalar(select(func.count()).select_from(AgentSession)) == 1


def test_duplicate_turn_with_a_new_transport_id_is_stored_once(client):
    token = enable(client)["token"]
    turn = {"tool": "claude", "surf": "surf-1", "incarnation": 1, "generation": 0, "provider_session_id": "s1",
            "provider_turn_id": "s1:3", "root": True, "seq": 3, "text": "Done: fixed the flaky test.", "truncated": False}
    with connect(bearer(token)) as ws:
        hello(ws)
        first = send(ws, "turn", **turn)
        again = send(ws, "turn", **turn)  # Same logical turn, new envelope id.
        sub = send(ws, "turn", **{**turn, "provider_turn_id": "s1:4", "root": False})
        long = send(ws, "turn", **{**turn, "provider_turn_id": "s1:5", "seq": 5, "text": "é" * 40_000})
        bound = send(ws, "turn", **{**turn, "provider_turn_id": "s1:6", "run_id": "00000000-0000-0000-0000-000000000000"})
    assert first["ok"] and "turn_id" in first["outcome"]
    assert again["ok"] and again["outcome"] == {"duplicate": True}
    assert sub["outcome"] == {"ignored": "subagent"}
    assert long["outcome"]["truncated"] is True
    assert not bound["ok"] and bound["error"]["code"] == "RUN_NOT_BOUND"
    with session_scope() as db:
        turns = list(db.scalars(select(AgentTurn).order_by(AgentTurn.seq)))
        assert [t.provider_turn_id for t in turns] == ["s1:3", "s1:5"]
        assert len(turns[1].text.encode()) <= agent_link.TURN_TEXT_BYTES and turns[1].truncated
        assert db.scalar(select(AgentSession.last_turn_seq)) == 5
        kinds = [e.kind for e in db.scalars(select(Event).order_by(Event.id))]
    assert kinds.count("agent.changed") >= 3  # host connect, two stored turns, disconnect


def test_status_updates_state_in_order_and_emits_session_changed(client):
    token = enable(client)["token"]
    start = now()
    with connect(bearer(token)) as ws:
        hello(ws)
        send(ws, "session.report", tool="claude_code", session_id="s1", report=SESSION, surf="surf-1", incarnation=1)
        with session_scope() as db:
            cursor = db.scalar(select(func.max(Event.id)))
        ack = send(ws, "status", surf="surf-1", incarnation=1, state="working", since=start.isoformat())
        assert ack["outcome"]["applied"] is True
        stale = send(ws, "status", surf="surf-1", incarnation=1, state="idle", since=(start - timedelta(minutes=1)).isoformat())
        assert stale["outcome"]["applied"] is False
        unknown = send(ws, "status", surf="nope", state="idle", since=start.isoformat())
        assert unknown["error"]["code"] == "UNKNOWN_SURFACE"
        with session_scope() as db:
            events = [(e.kind, e.entity_id) for e in db.scalars(select(Event).where(Event.id > cursor))]
            row = db.scalar(select(AgentSession))
            assert row.state == "working" and row.state_since == start
            assert ("session.changed", row.id) in events
        listed = client.get("/api/v1/sessions?live=true").json()["items"]
        assert listed[0]["live_state"]["state"] == "working"
        page = client.get("/api/v1/remotea/sessions").json()["items"]
        assert page[0]["host"]["connected"] is True and page[0]["live_state"]["state"] == "working"
        send(ws, "exit", surf="surf-1", incarnation=1, since=(start + timedelta(seconds=5)).isoformat(), code=0)
        with session_scope() as db:
            assert db.scalar(select(AgentSession.state)) == "exited"


def test_snapshot_reconciles_and_answers_with_runs(client):
    token = enable(client)["token"]
    start = now()
    with connect(bearer(token)) as ws:
        hello(ws)
        for sid, surf in (("s1", "a"), ("s2", "b")):
            send(ws, "session.report", tool="claude_code", session_id=sid, report=SESSION, surf=surf, incarnation=1)
            send(ws, "status", surf=surf, incarnation=1, state="idle", since=start.isoformat())
        message = frame("sessions.snapshot", complete=True, at=(start + timedelta(seconds=1)).isoformat(),
                        surfaces=[{"surf": "a", "incarnation": 1, "state": "working", "since": (start + timedelta(seconds=1)).isoformat()}])
        ws.send_json(message)
        runs = ws.receive_json()
        ack = ws.receive_json()
    assert runs["t"] == "runs.snapshot" and runs["runs"] == []
    assert ack["outcome"] == {"applied": 1, "ended": 1}
    with session_scope() as db:
        states = dict(db.execute(select(AgentSession.session_id, AgentSession.state)).all())
    assert states == {"s1": "working", "s2": "exited"}


def test_chunked_frames_reassemble(client):
    token = enable(client)["token"]
    inner = json.dumps(frame("session.report", tool="claude_code", session_id="s9", report=SESSION))
    with connect(bearer(token)) as ws:
        hello(ws)
        half = len(inner) // 2
        ws.send_json(frame("chunk", part={"of": "big-1", "i": 1, "n": 2}, data=inner[half:]))
        ws.send_json(frame("chunk", part={"of": "big-1", "i": 0, "n": 2}, data=inner[:half]))
        ack = ws.receive_json()
    assert ack["ok"] and ack["acked_id"] == json.loads(inner)["id"]


def test_parity_records_differences_without_changing_the_session(client):
    token = enable(client)["token"]
    headers = bearer(token)
    report(headers, session_id="s1", **SESSION)
    external = TestClient(app)
    with session_scope() as db:
        stored = db.scalar(select(AgentSession))
        body = {**SESSION, "last_active_at": stored.last_active_at.isoformat()}
    same = external.post("/api/v1/external/agents/parity", headers=headers,
                         json={"tool": "claude_code", "session_id": "s1", "report": body})
    assert same.status_code == 200, same.text
    assert same.json() == {"matched": True, "missing": False, "fields": []}
    differ = external.post("/api/v1/external/agents/parity", headers=headers,
                           json={"tool": "claude_code", "session_id": "s1", "report": {**body, "branch": "eng-12-fix", "turn_count": 4}})
    assert differ.json() == {"matched": False, "missing": False, "fields": ["branch", "turn_count"]}
    with connect(headers) as ws:
        hello(ws)
        ack = send(ws, "session.report", tool="claude_code", session_id="s404", report=SESSION, parity=True)
    assert ack["outcome"] == {"matched": False, "missing": True, "fields": []}
    with session_scope() as db:
        assert db.scalar(select(AgentSession.branch)) == "main"
        assert db.scalar(select(func.count()).select_from(AgentParity)) == 2
    summary = client.get("/api/v1/remotea/parity").json()
    assert (summary["total"], summary["matched"], summary["mismatched"], summary["missing"]) == (2, 0, 1, 1)
    assert summary["fields"] == {"branch": 1, "turn_count": 1} and not summary["clean"]
    assert summary["recent"][0]["diff"] or summary["recent"][1]["diff"]


def test_remotea_is_gated_until_enabled(client):
    assert client.get("/api/v1/remotea/status").json()["enabled"] is False
    for path in ("/api/v1/remotea/sessions", "/api/v1/remotea/parity"):
        response = client.get(path)
        assert response.status_code == 403 and response.json()["error"]["code"] == "REMOTEA_DISABLED"
    assert client.get("/api/v1/bootstrap").json()["preferences"]["remotea_enabled"] is False
    token = enable(client)["token"]
    assert client.get("/api/v1/remotea/sessions").status_code == 200
    assert client.get("/api/v1/bootstrap").json()["preferences"]["remotea_enabled"] is True
    assert client.post("/api/v1/remotea/disable").json()["enabled"] is False
    assert client.get("/api/v1/remotea/parity").status_code == 403
    denied = refused(bearer(token))
    assert denied.status_code == 403 and denied.json()["error"]["code"] == "REMOTEA_DISABLED"
    parity = TestClient(app).post("/api/v1/external/agents/parity", headers=bearer(token),
                                  json={"tool": "claude_code", "session_id": "s1", "report": SESSION})
    assert parity.status_code == 403


def test_disabling_closes_a_connected_link(client):
    token = enable(client)["token"]
    with connect(bearer(token)) as ws:
        hello(ws)
        client.post("/api/v1/remotea/disable")
        ack = ws.send_json(frame("session.report", tool="claude_code", session_id="s1", report=SESSION))
        bye = ws.receive_json()
        assert ack is None and bye["t"] == "bye" and bye["reason"] == "disabled"
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_json()
    assert closed.value.code == 4003


def test_enabling_mints_the_link_key_once(client):
    first = enable(client)
    assert first["enabled"] and first["token"].startswith("eri_bot_")
    assert first["connect_command"] == "remotea eridani connect --url ws://testserver/api/v1/external/agents/link"
    assert set(first["key"]["scopes"]) == {"agents:control", "agents:session", "sessions:read", "sessions:write",
                                           "tasks:read", "records:read"}
    status = client.get("/api/v1/remotea/status").json()
    assert "token" not in status and status["key"]["id"] == first["key"]["id"]
    again = enable(client)
    assert again["token"] is None and again["key"]["id"] == first["key"]["id"]
    rotated = enable(client, rotate=True)
    assert rotated["token"] and rotated["token"] != first["token"] and rotated["key"]["id"] != first["key"]["id"]
    with session_scope() as db:
        assert db.get(BotCredential, first["key"]["id"]).revoked_at is not None
    assert refused(bearer(first["token"])).status_code == 401


def test_viewers_cannot_hold_control_keys(client):
    from jarvis.models import WorkspaceMember
    from test_accounts import client_for, shared

    guest = client_for("guest")
    workspace = shared(client, guest)
    token = enable(guest)["token"]
    with session_scope() as db:
        db.get(WorkspaceMember, (workspace["id"], "guest")).role = "viewer"
    response = guest.post("/api/v1/bot-keys", json={"name": "x", "scopes": ["agents:control"], "expires_in_days": 30})
    assert response.status_code == 403
    assert guest.post("/api/v1/bot-keys", json={"name": "x", "scopes": ["sessions:read"], "expires_in_days": 30}).status_code == 201
    assert refused(bearer(token)).status_code == 403
