"""Reopen terminal (`run.reopen`, run_reopen.py): resume a run's exited worker in place on its host."""

import json
import uuid
from datetime import timedelta

from jarvis import agent_control as control
from jarvis import agent_link, run_reopen
from jarvis import run_coordinator as coordinator
from jarvis.db import session_scope
from jarvis.models import now
from jarvis.session_models import AgentCommand, AgentHost, AgentRun, AgentSession
from protocol_harness import FIXTURES, drift, shape
from sqlalchemy import select
from test_agent_runs import HUMAN, OWNER, attached, commands, quiet, start, state, task  # noqa: F401 - quiet is a fixture
from test_agent_runs import host as host  # noqa: F401 - pytest fixture

URL = "/api/v1/remotea/runs/{}/reopen"


def capable(host, *, connected=True):
    """The host's hello lists run.reopen; its lease is live (or lapsed)."""
    with session_scope() as db:
        row = db.get(AgentHost, host.id)
        row.caps = {**row.caps, "caps": [*(c for c in row.caps.get("caps", []) if c != "run.reopen"), "run.reopen"]}
        row.lease_until = now() + (timedelta(hours=1) if connected else -timedelta(minutes=1))


def exited(host):
    """A run whose agent exited: needs_you(exited) on a session at incarnation 1."""
    run_id = start(host, task()["id"])["id"]
    attached(host, run_id)
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        session = db.get(AgentSession, row.session_pk)
        session.state = "exited"
        coordinator.on_exit(db, session, "exit code 0")
    assert state(run_id) == ("needs_you", "exited")
    return run_id


def revision(run_id):
    with session_scope() as db:
        return db.get(AgentRun, run_id).state_revision


def reopen(client, run_id, **body):
    return client.post(URL.format(run_id), json={"action_id": str(uuid.uuid4()), "expected_revision": revision(run_id), **body})


def refusal(response):
    assert response.status_code == 409, response.text
    error = response.json()["error"]
    assert error["code"] == "REOPEN_UNAVAILABLE", error
    return error["data"]["reason"]


def view(run_id):
    with session_scope() as db:
        return control.public_run(db, db.get(AgentRun, run_id), detail=True)


def outcome(op_id, state, detail="", result=None):
    with session_scope() as db:
        row = db.scalar(select(AgentCommand).where(AgentCommand.op_id == op_id))
        coordinator.on_outcome(db, row, state, detail, result)


# ---------------------------------------------------------------- when it is offered


def test_reopen_is_offered_only_for_an_exited_agent_on_a_connected_host_that_can_resume_it(host, client):
    working = start(host, task("Working")["id"])["id"]
    attached(host, working, "sess-w")
    capable(host)
    assert refusal(reopen(client, working)) == "not_exited"
    assert view(working)["reopen"] == {"available": False, "pending": False, "why": None}

    run_id = exited(host)
    assert view(run_id)["reopen"] == {"available": True, "pending": False, "why": None}
    with session_scope() as db:  # An older host: no run.reopen in its hello.
        row = db.get(AgentHost, host.id)
        row.caps = {**row.caps, "caps": [c for c in row.caps["caps"] if c != "run.reopen"]}
    assert refusal(reopen(client, run_id)) == "host_too_old"
    assert view(run_id)["reopen"]["why"] == run_reopen.WHY["host_too_old"]
    capable(host, connected=False)
    assert refusal(reopen(client, run_id)) == "host_offline"
    capable(host)
    with session_scope() as db:  # The terminal took control of the session.
        row = db.get(AgentRun, run_id)
        row.sync_state = {**row.sync_state, "controller": "terminal"}
    assert refusal(reopen(client, run_id)) == "terminal_control"
    with session_scope() as db:  # No surface or session to resume.
        row = db.get(AgentRun, run_id)
        row.sync_state = {k: v for k, v in row.sync_state.items() if k != "controller"}
        row.surf = ""
    assert refusal(reopen(client, run_id)) == "no_surface"
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        session_pk = row.session_pk
        row.surf, row.session_pk = coordinator.surf_for(run_id, 1, 1), None
    assert refusal(reopen(client, run_id)) == "no_surface"
    with session_scope() as db:
        db.get(AgentRun, run_id).session_pk = session_pk
    with session_scope() as db:  # Another needs_you reason (here: the host went offline mid-run) is not an exit.
        row = db.get(AgentRun, run_id)
        row.state_reason = "host_offline"
    assert refusal(reopen(client, run_id)) == "not_exited"
    with session_scope() as db:
        db.get(AgentRun, run_id).state_reason = "exited"
    assert not commands(run_id, "run.reopen")
    assert reopen(client, run_id).status_code == 200


def test_a_stopped_or_finished_run_is_never_reopened(host, client):
    """remotea clears its stop flag on exit, so Eridani itself refuses a run the owner stopped or cancelled."""
    capable(host)
    run_id = exited(host)
    with session_scope() as db:  # Stopped while its agent was gone: stopping, then settled.
        control.stop_run(db, OWNER, run_id, control.StopInput(), actor=HUMAN)
    assert state(run_id)[0] in ("stopping", "stopped")
    assert refusal(reopen(client, run_id)) == "stopped"
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        coordinator.finish(db, row, detail="test")
    assert state(run_id)[0] == "stopped"
    assert refusal(reopen(client, run_id)) == "stopped"
    assert view(run_id)["reopen"] == {"available": False, "pending": False, "why": None}
    # Even parked in needs_you(exited), a run with a recorded stop request is refused.
    other = exited(host)
    with session_scope() as db:
        row = db.get(AgentRun, other)
        row.stop_requested_at = now()
        assert run_reopen.unavailable(db, row) == "stopped"
    assert refusal(reopen(client, other)) == "stopped"
    assert not commands(run_id, "run.reopen") and not commands(other, "run.reopen")


def test_a_reviewer_surface_or_an_unfinished_handoff_is_not_the_runs_worker(host, monkeypatch):
    run_id = exited(host)
    capable(host)
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        row.sync_state = {**row.sync_state, "handoff": {"attempt": 1, "done": False}}
        assert run_reopen.unavailable(db, row) == "handoff"
        row.sync_state = {**row.sync_state, "handoff": {"attempt": 1, "done": True}}
        assert run_reopen.unavailable(db, row) is None
        roles = row.roles
        row.roles = {"planner": {"harness": "claude"}, "reviewer": {"harness": "codex"}}
        monkeypatch.setattr(coordinator, "review_for_surface", lambda db, run, surf: object() if surf == row.surf else None)
        assert run_reopen.unavailable(db, row) == "reviewer"
        row.roles = roles


# ---------------------------------------------------------------- the command


def test_reopen_queues_the_command_with_the_exited_incarnation_and_a_five_minute_expiry(host, client):
    run_id = exited(host)
    capable(host)
    with session_scope() as db:  # The session is ahead of the run's own record of the incarnation.
        row = db.get(AgentRun, run_id)
        session = db.get(AgentSession, row.session_pk)
        session.surf, session.incarnation = row.surf, 3
        surf = row.surf
    action_id = str(uuid.uuid4())
    body = {"action_id": action_id, "expected_revision": revision(run_id)}
    first = client.post(URL.format(run_id), json=body)
    assert first.status_code == 200, first.text
    assert first.json()["reopen"] == {"available": False, "pending": True, "why": run_reopen.WHY["pending"]}
    [sent] = commands(run_id, "run.reopen")
    frame = sent.payload
    assert {k: frame[k] for k in ("t", "run_id", "generation", "surf", "incarnation")} == {
        "t": "run.reopen", "run_id": run_id, "generation": 1, "surf": surf, "incarnation": 3}
    assert frame["command_id"] == sent.op_id and sent.status == "queued"
    assert coordinator.TTL["run.reopen"] == timedelta(minutes=5)
    assert timedelta(minutes=4) < sent.expires_at - now() <= timedelta(minutes=5)
    assert "run.reopen" not in coordinator.QUIET_COMMANDS and agent_link.COMMANDS["run.reopen"] == "run.reopen"
    events = [h.get("event") for h in view(run_id)["history"]]
    assert "reopen_requested" in events and "command" in events  # On the timeline, not quiet.
    # The response was lost: the same action id returns the first result; a new click waits for the first.
    assert client.post(URL.format(run_id), json=body).json() == first.json()
    assert refusal(reopen(client, run_id)) == "pending"
    assert len(commands(run_id, "run.reopen")) == 1
    stale = client.post(URL.format(run_id), json={"action_id": str(uuid.uuid4()), "expected_revision": 1})
    assert stale.status_code == 409 and stale.json()["error"]["code"] == "REVISION_CONFLICT"


def test_the_frame_eridani_sends_matches_the_protocol_fixture(host, client):
    run_id = exited(host)
    capable(host)
    assert reopen(client, run_id).status_code == 200
    with session_scope() as db:
        row = db.get(AgentHost, host.id)
        [frame] = [f for f in agent_link.drain(db, row, row.conn_epoch) if f["t"] == "run.reopen"]
        envelope = agent_link.wire({"v": agent_link.PROTO, "id": "m1", "conn_epoch": row.conn_epoch, "at": now(), **frame})
    recorded = json.loads((FIXTURES / "2" / "eridani.run.reopen.json").read_text())["shape"]
    assert drift(recorded, shape(json.loads(json.dumps(envelope)))) == []


# ---------------------------------------------------------------- outcomes


def test_an_executed_reopen_brings_the_run_back_to_its_phase_and_continues_a_managed_run(host, client, monkeypatch):
    nudged = []
    monkeypatch.setattr(coordinator, "continue_safely", lambda db, run, reason: nudged.append((run.id, run.state, reason)))
    run_id = exited(host)
    capable(host)
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        row.sync_state = {**row.sync_state, "execution_mode": "managed"}
        surf = row.surf
    assert reopen(client, run_id).status_code == 200
    [sent] = commands(run_id, "run.reopen")
    outcome(sent.op_id, "executed", result={"surf": surf, "incarnation": 2})
    assert state(run_id) == ("executing", "reopened")
    with session_scope() as db:
        assert db.get(AgentRun, run_id).incarnation == 2
    assert nudged == [(run_id, "executing", "reopened")]
    shown = view(run_id)
    assert shown["reason_text"] == coordinator.REASONS["reopened"]
    assert shown["reopen"] == {"available": False, "pending": False, "why": None}
    assert commands(run_id, "run.reopen")[0].status == "executed"
    # A replayed outcome changes nothing.
    outcome(sent.op_id, "executed", result={"surf": surf, "incarnation": 2})
    assert state(run_id) == ("executing", "reopened") and len(nudged) == 1


def test_an_executed_reopen_nudges_a_managed_run_with_a_continue_prompt(host, client):
    run_id = exited(host)
    capable(host)
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        row.sync_state = {**row.sync_state, "execution_mode": "managed"}
        for command in db.scalars(select(AgentCommand).where(AgentCommand.run_id == run_id,
                                                                         AgentCommand.kind == "input")):
            command.status = "executed"  # The execution prompt reached the agent before it exited.
        surf = row.surf
    assert reopen(client, run_id).status_code == 200
    outcome(commands(run_id, "run.reopen")[0].op_id, "executed", result={"surf": surf, "incarnation": 2})
    nudge = commands(run_id, "input")[-1]
    assert nudge.status == "queued" and nudge.payload["incarnation"] == 2
    assert nudge.payload["text"].startswith("Continue the current task")


def test_a_refused_or_failed_reopen_leaves_the_run_waiting_with_the_reason(host, client):
    run_id = exited(host)
    capable(host)
    assert reopen(client, run_id).status_code == 200
    outcome(commands(run_id, "run.reopen")[0].op_id, "rejected", "no conversation to resume")
    assert state(run_id) == ("needs_you", "exited")
    shown = view(run_id)
    failure = [h for h in shown["history"] if h.get("event") == "reopen_failed"]
    assert failure[-1]["state"] == "rejected" and failure[-1]["detail"] == "no conversation to resume"
    assert shown["reason_detail"] == "Reopen failed: no conversation to resume"
    assert shown["attention_detail"]["detail"] == "Reopen failed: no conversation to resume"
    assert shown["reopen"]["available"] is True  # The owner can try again.

    assert reopen(client, run_id).status_code == 200
    assert view(run_id)["reason_detail"] == "exit code 0"  # A new attempt clears the old failure.
    outcome(commands(run_id, "run.reopen")[-1].op_id, "failed", "x" * 500)
    assert state(run_id) == ("needs_you", "exited")
    last = view(run_id)["history"][-1]
    assert (last["event"], last["state"], last["detail"]) == ("reopen_failed", "failed", "x" * 300)

    assert reopen(client, run_id).status_code == 200
    outcome(commands(run_id, "run.reopen")[-1].op_id, "uncertain", "the host restarted")
    assert state(run_id) == ("needs_you", "exited")  # Not delivery_uncertain: the run still waits on its exited agent.


def test_an_unanswered_reopen_expires_and_frees_the_button(host, client):
    run_id = exited(host)
    capable(host)
    assert reopen(client, run_id).status_code == 200
    with session_scope() as db:
        coordinator.sweep(db, now() + timedelta(minutes=6))
    assert commands(run_id, "run.reopen")[0].status == "expired"
    shown = view(run_id)
    assert shown["reason_detail"] == "Reopen failed: the host did not take the reopen in time"
    assert shown["reopen"]["available"] is True


def test_a_reopen_whose_agent_exited_again_keeps_the_run_waiting(host, client):
    run_id = exited(host)
    capable(host)
    assert reopen(client, run_id).status_code == 200
    with session_scope() as db:  # The new incarnation's exit arrived before the reopen's outcome.
        row = db.get(AgentRun, run_id)
        session = db.get(AgentSession, row.session_pk)
        session.surf, session.incarnation, session.state = row.surf, 2, "exited"
        surf = row.surf
    outcome(commands(run_id, "run.reopen")[0].op_id, "executed", result={"surf": surf, "incarnation": 2})
    assert state(run_id) == ("needs_you", "exited")
    assert view(run_id)["reason_detail"] == "Reopen failed: the reopened agent exited again"


def test_a_queued_reopen_goes_stale_once_the_run_moves_on(host, client):
    run_id = exited(host)
    capable(host)
    assert reopen(client, run_id).status_code == 200
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        command = db.scalar(select(AgentCommand).where(AgentCommand.run_id == run_id,
                                                                   AgentCommand.kind == "run.reopen"))
        assert coordinator.stale(db, command, row) is None
        row.generation += 1
        assert coordinator.stale(db, command, row) == "an older generation"
        row.generation -= 1
        coordinator.transition(db, row, "executing", reason="steer")
        assert coordinator.stale(db, command, row) == "the run no longer waits on an exited agent"
