"""The remotea page's home (task d6b29e9c): New session starts a freestyle run with the owner's own prompt, runs can
be dismissed from the page, every live session says where it came from, and chats from other harnesses are listed
apart."""

import uuid
from datetime import timedelta

from jarvis.db import session_scope
from jarvis.models import Task, now
from jarvis.session_models import AgentRun, AgentSession
from test_agent_runs import (  # noqa: F401 - fixture
    OWNER,
    attached,
    commands,
    quiet,
    state,
    task,
)
from test_agent_runs import host as host


def new_session(client, host, prompt="Fix the flaky login test\nIt fails on CI about one run in five.", **extra):
    r = client.post("/api/v1/remotea/sessions/new", json={"prompt": prompt, "execution_target_id": host.target_id, **extra})
    assert r.status_code == 200, r.text
    return r.json()


def test_new_session_starts_a_freestyle_run_with_the_owners_prompt(host, client):
    body = new_session(client, host)
    assert body["freestyle"] is True and body["origin"] == "session"
    assert body["task"]["title"] == "Fix the flaky login test"
    with session_scope() as db:
        run = db.get(AgentRun, body["id"])
        t = db.get(Task, run.task_id)
        assert t.notes.startswith("Fix the flaky login test\nIt fails") and t.execution_target_id == host.target_id
    spawn = commands(body["id"], "spawn")[0].payload
    assert spawn["prompt"].startswith("Fix the flaky login test\nIt fails on CI")
    assert "New session" in spawn["prompt"] and "submit_plan" in spawn["prompt"]
    assert "Work on this Eridani task" not in spawn["prompt"]  # Not the planned-run prompt.


def test_new_session_titles_a_long_first_line_and_repeats_once(host, client):
    action = str(uuid.uuid4())
    first = new_session(client, host, prompt="x" * 300, action_id=action)
    assert len(first["task"]["title"]) == 120 and first["task"]["title"].endswith("…")
    again = new_session(client, host, prompt="x" * 300, action_id=action)
    assert again["id"] == first["id"]


def test_new_session_needs_an_owned_target(host, client):
    r = client.post("/api/v1/remotea/sessions/new", json={"prompt": "hi", "execution_target_id": str(uuid.uuid4())})
    assert r.status_code == 404


def test_dismiss_stops_an_active_run_and_hides_it_from_the_page(host, client):
    body = new_session(client, host)
    attached(host, body["id"])
    r = client.post(f"/api/v1/remotea/runs/{body['id']}/dismiss", json={})
    assert r.status_code == 200, r.text
    assert r.json()["dismissed_at"]
    assert state(body["id"])[0] in ("stopping", "stopped")
    assert any(c.kind == "stop" for c in commands(body["id"]))
    page = client.get("/api/v1/remotea/runs?dismissed=false").json()["items"]
    assert body["id"] not in [i["id"] for i in page]
    everything = client.get("/api/v1/remotea/runs").json()["items"]
    assert body["id"] in [i["id"] for i in everything]  # The task's history keeps it.
    # A repeat is harmless.
    assert client.post(f"/api/v1/remotea/runs/{body['id']}/dismiss", json={}).status_code == 200


def add_session(host, session_id, **values):
    with session_scope() as db:
        row = AgentSession(owner_id=OWNER, tool="claude", session_id=session_id, title=session_id,
                           **{"last_active_at": now(), "started_at": now(), **values})
        db.add(row)
        db.flush()
        return row.id


def test_sessions_say_where_they_came_from_and_other_chats_are_apart(host, client):
    body = new_session(client, host)
    attached(host, body["id"], session_id="run-sess")
    add_session(host, "opened-in-remotea", host_id=host.id, surf="surf-9", state="idle", state_since=now())
    add_session(host, "laptop-chat", machine="laptop", repo="rdavisdunham/eridani", summary="Tidied the CSS")
    add_session(host, "old-chat", last_active_at=now() - timedelta(days=30))
    live = {i["session_id"]: i for i in client.get("/api/v1/remotea/sessions").json()["items"]}
    assert live["opened-in-remotea"]["origin"] == "remotea" and live["opened-in-remotea"]["run"] is None
    if "run-sess" in live:  # Listed once the host reports its state.
        assert live["run-sess"]["origin"] == "run" and live["run-sess"]["run"]["id"] == body["id"]
    assert "laptop-chat" not in live
    other = client.get("/api/v1/remotea/other-chats").json()["items"]
    assert [i["session_id"] for i in other] == ["laptop-chat"]  # Not the remotea one, not one older than 14 days.
    assert other[0]["origin"] == "other" and other[0]["machine"] == "laptop" and other[0]["summary"] == "Tidied the CSS"


def test_a_run_bound_session_reports_its_run(host, client):
    body = new_session(client, host)
    attached(host, body["id"], session_id="run-sess")
    with session_scope() as db:
        run = db.get(AgentRun, body["id"])
        session = db.get(AgentSession, run.session_pk)
        session.state, session.state_since = "working", now()
    live = {i["session_id"]: i for i in client.get("/api/v1/remotea/sessions").json()["items"]}
    assert live["run-sess"]["origin"] == "run"
    assert live["run-sess"]["run"] == {"id": body["id"], "state": live["run-sess"]["run"]["state"], "origin": "session",
                                       "task_id": body["task"]["id"], "dismissed": False}
