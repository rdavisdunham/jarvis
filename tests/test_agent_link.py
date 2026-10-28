import json
import time
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
    ws.send_json(frame("hello", **{"server_id": server_id, "boot_id": "boot-1", "hostname": "desk", "machine": "WSL",
                                    "proto": 1, "caps": ["session.report", "turn", "status"], **values}))
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
        # End the connection from the server side (a binary frame is refused with 1003) and watch the lease
        # go while the handler is still alive. Leaving the block first would race the test client's
        # teardown, which can cancel the handler before its release runs.
        ws.send_bytes(b"\x00")
        with pytest.raises(WebSocketDisconnect):
            ws.receive_text()
        for _ in range(100):
            hosts = client.get("/api/v1/remotea/status").json()["hosts"]
            if hosts[0]["connected"] is False:
                break
            time.sleep(0.02)
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
    denied = refused(bearer(token))  # Disabling revoked the link key (stage F, L1).
    assert denied.status_code == 401 and denied.json()["error"]["code"] == "ACCESS_REVOKED"
    parity = TestClient(app).post("/api/v1/external/agents/parity", headers=bearer(token),
                                  json={"tool": "claude_code", "session_id": "s1", "report": SESSION})
    assert parity.status_code == 401


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
    assert closed.value.code == 4403


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
    """Agent keys are personal-only (stage F, D1): no member of a shared workspace mints or uses one,
    whatever their role; viewers get read-only keys only."""
    from jarvis.models import WorkspaceMember
    from test_accounts import client_for, shared

    guest = client_for("guest")
    workspace = shared(client, guest)
    assert guest.post("/api/v1/remotea/enable", json={}).json()["error"]["code"] == "PERSONAL_WORKSPACE"
    response = guest.post("/api/v1/bot-keys", json={"name": "x", "scopes": ["agents:control"], "expires_in_days": 30})
    assert response.status_code == 403
    with session_scope() as db:
        db.get(WorkspaceMember, (workspace["id"], "guest")).role = "viewer"
    response = guest.post("/api/v1/bot-keys", json={"name": "x", "scopes": ["agents:control"], "expires_in_days": 30})
    assert response.status_code == 403
    assert guest.post("/api/v1/bot-keys", json={"name": "x", "scopes": ["sessions:read"], "expires_in_days": 30}).status_code == 201


def ms(value):
    return int(value.timestamp() * 1000)


def test_host_wire_format_from_remotea(client):
    """Frames exactly as remotea's stage B link sends them (epoch-ms times, provider ids, in-frame parts)."""
    token = enable(client)["token"]
    start = now()
    sid = "0f8fad5b-d9cb-469f-a165-70867728950e"
    with connect(bearer(token)) as ws:
        ws.send_json({"v": 1, "id": "h1", "conn_epoch": None, "at": ms(start), "t": "hello", "server_id": "59e4bb0b-8fe5",
                      "boot_id": "b1", "hostname": "DavisPC", "machine": "e2e-box", "proto": 1, "version": "0.1.0",
                      "caps": ["session.report", "session.report.parity", "turn", "status", "exit", "sessions.snapshot",
                               "chunking", "journal"],
                      "harnesses": {"claude": {"version": None, "observe": True, "capture": True}}})
        welcome = ws.receive_json()
        assert welcome["t"] == "welcome" and welcome["acked_id"] == "h1"
        assert isinstance(welcome["at"], int) and isinstance(welcome["server_time"], int)
        epoch = welcome["conn_epoch"]

        def put(message):  # The host stamps every effectful frame with the shared digest.
            ws.send_json({**message, "digest": agent_link.digest(message)} if "digest" in message else message)
        envelope = {"v": 1, "conn_epoch": epoch, "at": ms(start)}
        reported = {**envelope, "id": "r1", "t": "session.report", "entity": "claude_code:" + sid, "revision": "e7e9",
                    "tool": "claude_code", "session_id": sid, "report": SESSION, "op_id": "op-r1", "digest": "x"}
        put(reported)
        ack = ws.receive_json()
        assert ack["acked_id"] == "r1" and ack["op_id"] == "op-r1" and ack["ok"]
        put({**envelope, "id": "s1", "t": "status", "entity": "status:surf-1", "surf": "surf-1", "incarnation": 1,
                      "revision": ms(start), "state": "idle", "since": ms(start), "provider": "claude",
                      "provider_session_id": sid, "op_id": "op-s1", "digest": "x"})
        assert ws.receive_json()["outcome"]["applied"] is True
        put({**envelope, "id": "t1", "t": "turn", "surf": "surf-1", "incarnation": 1, "run_id": None,
                      "generation": 0, "provider": "claude", "tool": "claude_code", "provider_session_id": sid,
                      "provider_turn_id": sid + ":surf-1:2", "root": True, "seq": 2, "text": "", "truncated": False,
                      "missing": True, "source": "hook", "ended_at": ms(start), "op_id": "op-t1", "digest": "x"})
        assert ws.receive_json()["outcome"]["missing"] is True
        later = start + timedelta(seconds=2)
        for i, surfaces in ((2, []), (1, [{"surf": "surf-1", "incarnation": 1, "state": "live", "agent_state": "working",
                                           "since": ms(later), "provider": "claude", "provider_session_id": sid,
                                           "turn_seq": 2, "run_id": None, "generation": 0}])):
            put({**envelope, "id": f"p{i}", "t": "sessions.snapshot", "part": {"of": "snap-1", "i": i, "n": 2},
                          "surfaces": surfaces, "journal": {"pending": [], "uncertain": []}, "op_id": "op-snap", "digest": "x"})
        assert ws.receive_json()["outcome"] == {"pending": True}
        assert ws.receive_json()["t"] == "runs.snapshot"
        assert ws.receive_json()["outcome"] == {"applied": 1, "ended": 0}
        put({**envelope, "id": "x1", "t": "exit", "surf": "surf-1", "incarnation": 1, "code": 0,
                      "exited_at": ms(later + timedelta(seconds=1)), "op_id": "op-x1", "digest": "x"})
        assert ws.receive_json()["outcome"]["applied"] is True
        put({**envelope, "id": "c1", "t": "command.outcome", "command_id": "op-unknown", "state": "rejected",
                      "detail": "unsupported", "op_id": "op-c1", "digest": "x"})
        assert ws.receive_json()["outcome"] == {"recorded": False}
        put({**envelope, "id": "bye1", "t": "bye", "reason": "shutdown"})
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_json()
    assert closed.value.code == 1000
    with session_scope() as db:
        row = db.scalar(select(AgentSession))
        assert row.state == "exited" and ms(row.state_since) == ms(later + timedelta(seconds=1))
        assert db.scalar(select(func.count()).select_from(AgentTurn)) == 1


def test_connect_probe_does_not_take_the_lease(client):
    token = enable(client)["token"]
    with connect(bearer(token)) as live:
        epoch = hello(live)["conn_epoch"]
        with connect(bearer(token)) as checking:
            welcome = hello(checking, boot_id="probe-1234")
            assert welcome["probe"] is True and welcome["conn_epoch"] == epoch
            checking.send_json(frame("bye", reason="probe"))
        assert send(live, "session.report", tool="claude_code", session_id="s1", report=SESSION)["ok"]


def test_revoked_key_closes_with_4401(client):
    first = enable(client)
    with connect(bearer(first["token"])) as ws:
        hello(ws)
        client.post(f"/api/v1/bot-keys/{first['key']['id']}/revoke")
        ws.send_json(frame("session.report", tool="claude_code", session_id="s1", report=SESSION))
        assert ws.receive_json()["t"] == "bye"
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_json()
    assert closed.value.code == 4401


def test_digest_matches_the_host_definition():
    message = {"v": 1, "id": "e", "conn_epoch": 2, "at": 1, "t": "status", "state": "idle", "surf": "s", "op_id": "o", "digest": "d"}
    expected = __import__("hashlib").sha256('{"state":"idle","surf":"s","t":"status"}'.encode()).hexdigest()
    assert agent_link.digest(message) == expected


def test_inbound_digest_mismatch_is_refused_and_logged(client, caplog):
    token = enable(client)["token"]
    with connect(bearer(token)) as ws:
        hello(ws)
        message = frame("session.report", tool="claude_code", session_id="s1", report=SESSION, op_id="op-bad")
        ws.send_json({**message, "digest": "0" * 64})
        ack = ws.receive_json()
        assert not ack["ok"] and ack["error"]["code"] == "DIGEST_MISMATCH" and ack["op_id"] == "op-bad"
        assert "retry" not in ack
        good = {**message, "id": "again", "op_id": "op-good"}
        ws.send_json({**good, "digest": agent_link.digest(good)})
        assert ws.receive_json()["ok"]
    assert "op_id=op-bad" in caplog.text
    with session_scope() as db:
        assert db.scalar(select(func.count()).select_from(AgentSession)) == 1


# ---------------------------------------------------------------- stage C frames

C_CAPS = ["session.report", "turn", "status", "exit", "sessions.snapshot", "command.outcome", "question", "context",
          "spawn", "input", "answer", "stop", "runs.snapshot", "worktrees.report"]
C_HARNESSES = {"claude": {"version": "2.1.312", "observe": True, "capture": True, "question": True, "context": True,
                          "resume": True, "spawn": True}}


def signed(t, **values):
    message = frame(t, op_id=f"op-{next(frames)}", **values)
    return {**message, "digest": agent_link.digest(message)}


def put(ws, t, **values):
    message = signed(t, **values)
    ws.send_json(message)
    reply = ws.receive_json()
    assert reply["t"] == "ack" and reply["acked_id"] == message["id"], reply
    return reply


def target(client, host_id):
    response = client.post("/api/v1/remotea/targets", json={"host_id": host_id, "host_target": "jarvis"})
    assert response.status_code == 200, response.text
    return response.json()


def test_manual_control_round_trip_over_the_link(client):
    token = enable(client)["token"]
    task = client.post("/api/v1/commands", json={"command_id": "c-task", "tool": "task.create",
                                                 "arguments": {"title": "Wire the run drawer"}})
    assert task.status_code == 200, task.text
    task_id = task.json()["data"]["id"]
    with connect(bearer(token)) as ws:
        welcome = hello(ws, caps=C_CAPS, harnesses=C_HARNESSES,
                        targets=[{"id": "jarvis", "repo": "/home/davin/jarvis", "worktree_root": "/home/davin/wt"}])
        host_id = welcome["host_id"]
        snap = signed("sessions.snapshot", surfaces=[], journal={"pending": [], "uncertain": []})
        ws.send_json(snap)
        assert ws.receive_json()["t"] == "runs.snapshot"
        assert ws.receive_json()["acked_id"] == snap["id"]
        missing = client.post("/api/v1/remotea/runs", json={"task_id": task_id})
        assert missing.status_code == 409 and missing.json()["error"]["code"] == "NEEDS_TARGET"
        assert missing.json()["error"]["data"]["advertised"][0]["repo"] == "/home/davin/jarvis"
        saved = target(client, host_id)
        assert saved["repo"] == "/home/davin/jarvis" and saved["worktree_root"] == "/home/davin/wt"
        started = client.post("/api/v1/remotea/runs", json={"task_id": task_id, "execution_target_id": saved["id"], "interview": False})
        assert started.status_code == 200, started.text
        run_id = started.json()["id"]
        spawn = ws.receive_json()  # Drained right after the commit, not at the next tick.
        assert spawn["t"] == "spawn" and spawn["run_id"] == run_id and spawn["digest"] == agent_link.digest(spawn)
        assert isinstance(spawn["expires_at"], int) and spawn["target"]["id"] == "jarvis"
        surf = agent_link.coordinator.surf_for(run_id, 1, 1)
        ws.send_json(frame("ack", acked_id=spawn["id"], op_id=spawn["op_id"], ok=True,
                           outcome={"state": "accepted", "surf": surf, "incarnation": 1}))
        put(ws, "command.outcome", command_id=spawn["op_id"], state="executed", kind="spawn",
            result={"surf": surf, "incarnation": 1, "worktree": "/home/davin/wt/task", "branch": spawn["target"]["branch"],
                    "base": "main"})
        request = frame("context.request", req_id="ctx-1", surf=surf, incarnation=1, provider="claude",
                        provider_session_id="sess-c", source="startup", cwd="/home/davin/wt/task", run_id=run_id,
                        generation=1)
        ws.send_json(request)
        context = ws.receive_json()  # Answered by req_id; never acked.
        assert context["t"] == "context.response" and context["req_id"] == "ctx-1"
        assert context["run_token"] in context["text"] and "/home/davin/wt/task" in context["text"]
        detail = client.get(f"/api/v1/remotea/runs/{run_id}").json()
        assert detail["state"] == "executing" and detail["worktree"] == "/home/davin/wt/task"
        turn = put(ws, "turn", tool="claude_code", provider="claude", surf=surf, incarnation=1, run_id=run_id,
                   generation=1, provider_session_id="sess-c", provider_turn_id="p-1", root=True, seq=1,
                   text="Built the drawer.", source="hook")
        assert turn["ok"] and "turn_id" in turn["outcome"]
        asked = put(ws, "question", question_id="q-1", surf=surf, incarnation=1, run_id=run_id, generation=1,
                    provider="claude", provider_session_id="sess-c", provider_request_id="toolu_9", kind="ask",
                    group_id="g", hold_until=int((now() + timedelta(minutes=9)).timestamp() * 1000),
                    items=[{"id": "i1", "text": "Ship it?", "choices": [{"id": "y", "label": "Yes"}, {"id": "n", "label": "No"}]}])
        qid = asked["outcome"]["question"]
        assert client.get(f"/api/v1/remotea/runs/{run_id}").json()["state"] == "waiting_answer"
        answered = client.post(f"/api/v1/remotea/questions/{qid}/answer", json={"answers": [{"item_id": "i1", "choice_ids": ["y"]}]})
        assert answered.status_code == 200 and answered.json()["status"] == "submitted"
        down = ws.receive_json()
        assert down["t"] == "answer" and down["provider_request_id"] == "toolu_9" and down["answers"][0]["choice_ids"] == ["y"]
        ws.send_json(frame("ack", acked_id=down["id"], op_id=down["op_id"], ok=True, outcome={"state": "accepted"}))
        put(ws, "question.resolved", question_id="q-1", generation=1, outcome="accepted", still_open=False,
            answer_echo={"Ship it?": "Yes"})
        assert client.get(f"/api/v1/remotea/runs/{run_id}").json()["state"] == "executing"
        put(ws, "worktrees.report", worktrees=[{"target": "jarvis", "path": "/home/davin/wt/old", "branch": "task/old",
                                                "candidate": False, "unpushed": 2, "reasons": ["2 unpushed commits"]}])
        assert client.get("/api/v1/remotea/unmerged").json()["items"][0]["unpushed"] == 2
        stopped = client.post(f"/api/v1/remotea/runs/{run_id}/stop", json={})
        assert stopped.json()["state"] == "stopping"
        stop = ws.receive_json()
        assert stop["t"] == "stop" and stop["generation"] == 2 and stop["surf"] == surf
        ws.send_json(frame("ack", acked_id=stop["id"], op_id=stop["op_id"], ok=True, outcome={"state": "accepted"}))
        assert client.get(f"/api/v1/remotea/runs/{run_id}").json()["state"] == "stopping"  # Not until confirmed.
        put(ws, "exit", surf=surf, incarnation=1, code=130, exited_at=int(now().timestamp() * 1000) + 5000,
            provider="claude", provider_session_id="sess-c")
    detail = client.get(f"/api/v1/remotea/runs/{run_id}").json()
    assert detail["state"] == "stopped"
    assert [c["kind"] for c in detail["commands"]] == ["spawn", "answer", "stop"]
    assert detail["commands"][0]["status"] == "executed" and detail["questions"][0]["status"] == "accepted"
    assert detail["turns"][0]["text"] == "Built the drawer."
    with session_scope() as db:
        assert db.get(agent_link.AgentHost, host_id).caps["targets"][0]["id"] == "jarvis"


def test_reconnect_reconciles_before_resending_unacked_commands(client):
    token = enable(client)["token"]
    task_id = client.post("/api/v1/commands", json={"command_id": "c-2", "tool": "task.create",
                                                    "arguments": {"title": "Reconnect"}}).json()["data"]["id"]
    with connect(bearer(token)) as ws:
        host_id = hello(ws, caps=C_CAPS, harnesses=C_HARNESSES)["host_id"]
        saved = target(client, host_id)
        run_id = client.post("/api/v1/remotea/runs", json={"task_id": task_id, "execution_target_id": saved["id"], "interview": False}).json()["id"]
        # Not reconciled yet on this connection: nothing is sent before the snapshot.
    with connect(bearer(token)) as ws:
        hello(ws, caps=C_CAPS, harnesses=C_HARNESSES)
        snap = signed("sessions.snapshot", surfaces=[], journal={"pending": [], "uncertain": []})
        ws.send_json(snap)
        runs = ws.receive_json()
        assert runs["t"] == "runs.snapshot" and runs["runs"] == [{"run_id": run_id, "generation": 1, "desired": "continue"}]
        assert ws.receive_json()["acked_id"] == snap["id"]
        first = ws.receive_json()
        assert first["t"] == "spawn" and first["run_id"] == run_id
        with session_scope() as db:  # Unacked for a minute: the same operation goes out again.
            row = db.scalar(select(agent_link.AgentCommand))
            row.updated_at = now() - timedelta(minutes=2)
        with session_scope() as db:
            frames = agent_link.drain(db, db.get(agent_link.AgentHost, host_id), 0)
        assert [(f["op_id"], f["digest"]) for f in frames] == [(first["op_id"], first["digest"])]


def test_turns_for_runs_of_other_hosts_are_refused(client):
    token = enable(client)["token"]
    with connect(bearer(token)) as ws:
        hello(ws, caps=C_CAPS)
        reply = put(ws, "turn", tool="claude", surf="s", provider_session_id="x", provider_turn_id="t", root=True,
                    run_id="00000000-0000-0000-0000-000000000000", generation=1)
    assert not reply["ok"] and reply["error"]["code"] == "RUN_NOT_BOUND"
