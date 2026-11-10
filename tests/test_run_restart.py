"""Restart run (run_restart.py): a finished run continues as a new run linked to it, in plan or pilot mode."""

import uuid
from datetime import timedelta

from jarvis import run_coordinator as coordinator
from jarvis.db import session_scope
from jarvis.models import Task, now
from jarvis.session_models import AgentCommand, AgentHost, AgentRun, AgentSession, AgentTurn
from sqlalchemy import func, select
from test_agent_runs import (  # noqa: F401 - fixtures
    OWNER,
    attached,
    commands,
    quiet,
    start,
    task,
)
from test_agent_runs import host as host
from test_linear import run as domain

NOTE = "Also add a dark mode toggle to the settings page."


def finished(host, state="completed", *, complete_task=None, session_id="sess-old", **start_values):
    """A run on a fresh task that ran (attached to a Claude session) and ended in `state`."""
    created = task()
    run_id = start(host, created["id"], **start_values)["id"]
    if session_id:
        attached(host, run_id, session_id=session_id)
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        row.state, row.state_reason, row.finished_at, row.run_token_hash = state, "test", now(), None
        row.summary = "Added the settings page."
    if complete_task if complete_task is not None else state == "completed":
        with session_scope() as db:
            revision = db.get(Task, created["id"]).revision
        domain("task.complete", task_id=created["id"], expected_revision=revision)
    return run_id, created["id"]


def restart(client, run_id, mode="plan", note=NOTE, **extra):
    return client.post(f"/api/v1/remotea/runs/{run_id}/restart", json={"mode": mode, "note": note, **extra})


def ok(response):
    assert response.status_code == 200, response.text
    return response.json()


def refused(response, code="RESTART_UNAVAILABLE"):
    assert response.status_code == 409, response.text
    error = response.json()["error"]
    assert error["code"] == code, error
    return error


def spawn_of(run_id):
    return commands(run_id, "spawn")[0].payload


def caps(host, *extra):
    with session_scope() as db:
        row = db.get(AgentHost, host.id)
        row.caps = {**row.caps, "caps": [*row.caps["caps"], *extra]}


def detail(client, run_id):
    return ok(client.get(f"/api/v1/remotea/runs/{run_id}"))


# ---------------------------------------------------------------- the two modes


def test_plan_restart_reopens_the_task_and_plans_the_added_work(host, client):
    old, task_id = finished(host)
    with session_scope() as db:
        assert db.get(Task, task_id).status == "completed"
    before = detail(client, old)
    assert before["restart"] == {"available": True, "why": None} and before["restarted_as"] is None
    new = ok(restart(client, old, expected_revision=before["state_revision"]))
    assert new["id"] != old and new["freestyle"] is False and new["state"] == "spawning"
    assert new["restarted_from"]["id"] == old and new["restarted_from"]["state"] == "completed"
    assert new["restarted_from"]["finished_at"] and new["restarted_from"]["mode"] == "plan"
    assert new["restarted_from"]["resumed"] is None  # The host has not said yet.
    with session_scope() as db:
        row, previous = db.get(AgentRun, new["id"]), db.get(AgentRun, old)
        assert db.get(Task, task_id).status == "open"  # Reopened through task.reopen.
        assert previous.state == "completed"  # The finished run stays finished.
        assert (row.branch, row.execution_target_id, row.tool, row.model) == (
            previous.branch, previous.execution_target_id, previous.tool, previous.model)
        assert (row.sync_state or {}).get("start_phase") == "planning" and not row.plan_pending
        assert row.restarted_from_run_id == old
    spawn = spawn_of(new["id"])
    assert "continues earlier work" in spawn["prompt"] and NOTE in spawn["prompt"]
    assert "submit_plan" in spawn["prompt"] and "ERIDANI_REQUEST" in spawn["prompt"]
    # No spawn.resume: a fresh conversation, so the prompt summarizes the earlier run.
    assert "resume_session_id" not in spawn and "ERIDANI_EARLIER" in spawn["prompt"]
    assert "Added the settings page." in spawn["prompt"]
    notes = [h for h in new["history"] if h.get("event") == "restarted"]
    assert notes == [notes[0]] and notes[0]["mode"] == "plan" and notes[0]["text"] == NOTE
    assert notes[0]["actor"]["kind"] == "human" and notes[0]["restarted_from"] == old
    after = detail(client, old)
    assert after["restarted_as"] == {"id": new["id"], "state": "spawning"}
    assert after["restart"]["available"] is False and "already working" in after["restart"]["why"]
    assert any(h.get("event") == "restarted" and h.get("restarted_as") == new["id"] for h in after["history"])


def test_pilot_restart_is_a_freestyle_run_whose_first_message_is_the_note(host, client):
    old, task_id = finished(host, "stopped")
    new = ok(restart(client, old, mode="pilot", note="  Let's look at the flaky test together.  "))
    assert new["freestyle"] is True and new["roles"] is None and new["restarted_from"]["mode"] == "pilot"
    assert spawn_of(new["id"])["prompt"] == "Let's look at the flaky test together."
    with session_scope() as db:
        assert db.get(Task, task_id).status != "completed"


def test_pilot_restart_without_a_note_opens_or_waits(host, client):
    old, _ = finished(host, "failed")
    new = ok(restart(client, old, mode="pilot", note=""))
    assert spawn_of(new["id"])["prompt"] == control_opening()
    # With resume, nothing is sent: the resumed agent waits at its prompt.
    caps(host, "spawn.resume")
    with session_scope() as db:
        db.get(AgentRun, new["id"]).state = "stopped"
    again = ok(restart(client, old, mode="pilot", note=""))
    spawn = spawn_of(again["id"])
    assert spawn["prompt"] == "" and spawn["resume_session_id"] == "sess-old"


def control_opening():
    from jarvis.agent_prompts import RESTART_PILOT_OPENING

    return RESTART_PILOT_OPENING


def test_plan_restart_of_a_hyperplan_run_keeps_its_roles(host, client):
    caps(host, "spawn.roles")
    old, _ = finished(host, reviewer={"harness": "codex"}, rounds=3, approve_final=True)
    new = ok(restart(client, old))
    assert new["roles"]["reviewer"]["harness"] == "codex" and new["roles"]["rounds"] == 3
    assert new["roles"]["planner"]["harness"] == "claude"
    pilot_old, _ = finished(host, reviewer={"harness": "codex"})
    assert ok(restart(client, pilot_old, mode="pilot"))["roles"] is None  # Pilot has no roles.


# ---------------------------------------------------------------- resume


def test_resume_needs_the_cap_the_same_harness_and_a_known_session(host, client):
    old, _ = finished(host)
    assert "resume_session_id" not in spawn_of(ok(restart(client, old))["id"])  # No cap.
    caps(host, "spawn.resume")
    with session_scope() as db:
        for row in db.scalars(select(AgentRun).where(AgentRun.restarted_from_run_id == old)):
            row.state = "stopped"
    resumed = ok(restart(client, old))
    spawn = spawn_of(resumed["id"])
    assert spawn["resume_session_id"] == "sess-old" and "ERIDANI_EARLIER" not in spawn["prompt"]
    assert "resume that run's conversation" in spawn["prompt"]
    # Another harness: the Claude conversation cannot be resumed by Codex.
    codex, _ = finished(host, harness="codex", model="gpt-5.6-sol")
    with session_scope() as db:
        session = db.get(AgentSession, db.get(AgentRun, codex).session_pk)
        session.tool = "claude_code"
    assert "resume_session_id" not in spawn_of(ok(restart(client, codex))["id"])
    # No session known: fresh.
    unknown, _ = finished(host, session_id=None)
    assert "resume_session_id" not in spawn_of(ok(restart(client, unknown))["id"])


def test_the_spawn_outcome_records_whether_the_host_resumed(host, client):
    caps(host, "spawn.resume")
    old, _ = finished(host)
    new = ok(restart(client, old))["id"]
    with session_scope() as db:
        spawn = db.scalar(select(AgentCommand).where(AgentCommand.run_id == new, AgentCommand.kind == "spawn"))
        coordinator.on_outcome(db, spawn, "executed", "", {"surf": coordinator.surf_for(new, 1, 1), "incarnation": 1,
                                                          "resumed": True})
    shown = detail(client, new)
    assert shown["restarted_from"]["resumed"] is True
    assert next(h for h in shown["history"] if h.get("event") == "restarted")["resumed"] is True


def test_a_resumed_conversation_keeps_each_runs_turns_apart(host, client):
    caps(host, "spawn.resume")
    old, _ = finished(host)
    new = ok(restart(client, old))["id"]
    with session_scope() as db:
        session_pk = db.get(AgentRun, old).session_pk
        db.add(AgentTurn(owner_id=OWNER, session_pk=session_pk, run_id=new, seq=99, role="assistant",
                         text="New work", provider_session_id="sess-old", provider_turn_id="t-99"))
    assert "New work" not in [t["text"] for t in detail(client, old)["turns"]]


# ---------------------------------------------------------------- refusals


def test_restart_refuses_a_run_that_has_not_finished(host, client):
    created = task()
    active = start(host, created["id"])["id"]
    error = refused(restart(client, active))
    assert error["data"]["reason"] == "not_finished"
    assert detail(client, active)["restart"] == {"available": False, "why": None}


def test_restart_refuses_while_another_run_works_on_the_task(host, client):
    old, task_id = finished(host, "stopped")
    other = start(host, task_id)["id"]
    error = refused(restart(client, old), "RUN_ACTIVE")
    assert error["data"]["run_id"] == other


def test_restart_refuses_a_cancelled_or_archived_task(host, client):
    old, task_id = finished(host, "stopped")
    with session_scope() as db:
        db.get(Task, task_id).status = "cancelled"
    assert refused(restart(client, old))["data"]["reason"] == "task_closed"
    with session_scope() as db:
        row = db.get(Task, task_id)
        row.status, row.archived = "open", True
    assert refused(restart(client, old))["data"]["reason"] == "task_closed"
    assert "archived" in detail(client, old)["restart"]["why"]


def test_restart_refuses_an_offline_host(host, client):
    old, _ = finished(host, "failed")
    with session_scope() as db:
        db.get(AgentHost, host.id).lease_until = now() - timedelta(minutes=5)
    assert refused(restart(client, old))["data"]["reason"] == "host_offline"


def test_plan_restart_needs_a_note(host, client):
    old, _ = finished(host)
    assert refused(restart(client, old, note="   "))["data"]["reason"] == "note_required"
    with session_scope() as db:
        assert db.get(Task, db.get(AgentRun, old).task_id).status == "completed"  # Nothing changed.
    assert client.post(f"/api/v1/remotea/runs/{old}/restart", json={"mode": "later"}).status_code == 422
    assert client.post(f"/api/v1/remotea/runs/{old}/restart", json={"mode": "plan", "note": "x" * 4001}).status_code == 422


def test_restart_checks_the_revision(host, client):
    old, _ = finished(host)
    stale = detail(client, old)["state_revision"] - 1 or 99
    assert refused(restart(client, old, expected_revision=stale), "REVISION_CONFLICT")


# ---------------------------------------------------------------- idempotency


def test_a_repeated_action_id_starts_one_run(host, client):
    old, _ = finished(host)
    action = str(uuid.uuid4())
    first = ok(restart(client, old, action_id=action))
    again = ok(restart(client, old, action_id=action))
    assert again["id"] == first["id"]
    with session_scope() as db:
        assert db.scalar(select(func.count()).select_from(AgentRun).where(AgentRun.restarted_from_run_id == old)) == 1
    refused(restart(client, old, action_id=action, note="Something else"), "ACTION_MISMATCH")


def test_restarted_as_names_the_latest_restart(host, client):
    old, _ = finished(host, "stopped")
    first = ok(restart(client, old, mode="pilot"))["id"]
    with session_scope() as db:
        db.get(AgentRun, first).state = "stopped"
    second = ok(restart(client, old, mode="pilot"))["id"]
    assert detail(client, old)["restarted_as"]["id"] == second
    assert detail(client, second)["restarted_from"]["id"] == old
    listed = {r["id"]: r for r in client.get("/api/v1/remotea/runs").json()["items"]}
    assert listed[old]["restarted_as"]["id"] == second and listed[first]["restarted_from"]["id"] == old


def test_the_resume_spawn_matches_the_recorded_protocol_fixture(host, client):
    """The spawn frame with `resume_session_id` (and the outcome's `result.resumed`) is in tests/fixtures/protocol/2/."""
    import json

    from protocol_harness import FIXTURES, drift, shape

    caps(host, "spawn.resume")
    old, _ = finished(host)
    new = ok(restart(client, old))["id"]
    with session_scope() as db:
        row = db.scalar(select(AgentCommand).where(AgentCommand.run_id == new, AgentCommand.kind == "spawn"))
        frame = {**row.payload, "op_id": row.op_id, "digest": row.digest, "v": 1, "id": "m1", "conn_epoch": 1, "at": 1}
    assert frame["resume_session_id"] == "sess-old"
    recorded = json.loads((FIXTURES / "2" / "eridani.spawn.json").read_text())["shape"]
    assert recorded["keys"]["resume_session_id"] == {"required": False, "shape": {"types": ["string"]}}
    assert drift(recorded, shape(frame)) == []
    outcome = json.loads((FIXTURES / "2" / "host.command.outcome.json").read_text())["shape"]
    assert outcome["keys"]["result"]["shape"]["keys"]["resumed"]["shape"]["types"] == ["boolean"]
