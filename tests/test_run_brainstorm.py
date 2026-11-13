"""Brainstorm: a run that opens as a conversation and plans when the owner says so (run_brainstorm.py)."""

from uuid import uuid4

import pytest
from jarvis import agent_control as control
from jarvis import agent_plan, agent_prompts, run_brainstorm
from jarvis import run_coordinator as coordinator
from jarvis.db import session_scope
from jarvis.domain import DomainError
from jarvis.models import Task
from jarvis.session_models import AgentRun
from test_agent_plan import PLAN, call, start, turn
from test_agent_runs import HUMAN, OWNER, attached, commands, quiet, task  # noqa: F401 - pytest fixtures
from test_agent_runs import host as host  # noqa: F401 - pytest fixture


def talking(host, **values):
    made = task("Make the greeting friendlier", **values)
    run_id = start(host, made["id"], brainstorm=True)["id"]
    attached(host, run_id)
    return made["id"], run_id


def where(run_id):
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        return row.state, row.state_reason


def say(run_id, text):
    with session_scope() as db:
        control.send_message(db, OWNER, run_id, control.MessageInput(text=text), actor=HUMAN)


def submit(run_id, task_id):
    with session_scope() as db:
        revision = db.get(Task, task_id).plan_revision
    return call(agent_plan.submit_plan, agent_plan.SubmitPlanInput, run_id,
                **{**PLAN, "expected_plan_revision": revision, "submission_id": str(uuid4())})


def test_the_run_opens_as_a_conversation_and_plans_when_the_owner_says_so(host):
    task_id, run_id = talking(host)
    spawn = commands(run_id, "spawn")[0]
    assert "ONE question per turn" in spawn.payload["prompt"] and "task_interview tool once" not in spawn.payload["prompt"]
    assert where(run_id)[0] == "planning"
    with session_scope() as db:
        shown = control.public_run(db, db.get(AgentRun, run_id), detail=True)
    assert shown["brainstorm"]["open"] is True and shown["brainstorm"]["turns"] == 0

    # The agent's opening turn ends without a plan: that is its question, not a failure.
    turn(run_id, 1)
    assert where(run_id) == ("needs_you", "brainstorm")
    say(run_id, "Warmer, but still one line. Option B.")
    assert where(run_id)[0] == "planning"
    assert commands(run_id, "input")[-1].payload["text"] == "Warmer, but still one line. Option B."
    turn(run_id, 2)
    assert where(run_id) == ("needs_you", "brainstorm")

    # "Plan it": the conversation closes and the agent is asked for the plan.
    with session_scope() as db:
        run_brainstorm.plan_it(db, db.get(AgentRun, run_id, with_for_update=True), actor=HUMAN)
    assert where(run_id)[0] == "planning"
    assert "ready for the plan" in commands(run_id, "input")[-1].payload["text"]
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        value = run_brainstorm.public(row)
        assert value["open"] is False and value["turns"] == 2 and value["ended_by"] == "owner"
        with pytest.raises(DomainError):
            run_brainstorm.plan_it(db, row, actor=HUMAN)  # Only once.
    submit(run_id, task_id)
    assert where(run_id) == ("needs_you", "plan_approval")  # The usual approval, whatever the route.


def test_a_turn_without_a_plan_after_the_conversation_is_a_missing_plan_again(host):
    _, run_id = talking(host)
    turn(run_id, 1)
    with session_scope() as db:
        run_brainstorm.plan_it(db, db.get(AgentRun, run_id, with_for_update=True), actor=HUMAN)
    turn(run_id, 2)
    assert where(run_id) == ("needs_you", "no_plan")


def test_a_plan_asked_for_in_the_conversation_closes_it(host):
    task_id, run_id = talking(host, run_route="direction")
    turn(run_id, 1)
    say(run_id, "That's it, plan it.")
    submit(run_id, task_id)
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        assert run_brainstorm.public(row)["ended_by"] == "plan_submitted"
        # The conversation was the direction: no brief is asked for, and the plan takes the ordinary approval.
        assert "`brief`" not in agent_prompts.brainstorm(db.get(Task, task_id), [])
        assert (row.sync_state or {}).get("direction") is None
    assert where(run_id) == ("needs_you", "plan_approval")


def test_brainstorm_is_refused_with_a_ready_made_plan_and_ignored_for_freestyle(host):
    made = task("Rename a helper")
    with pytest.raises(DomainError):
        start(host, made["id"], brainstorm=True, interview=False)
    run_id = start(host, made["id"], brainstorm=True, freestyle=True)["id"]
    with session_scope() as db:
        assert run_brainstorm.public(db.get(AgentRun, run_id)) is None


def test_the_route_asks_for_the_plan(host, client):
    _, run_id = talking(host)
    turn(run_id, 1)
    with session_scope() as db:
        revision = db.get(AgentRun, run_id).state_revision
    stale = client.post(f"/api/v1/remotea/runs/{run_id}/brainstorm/plan", json={"expected_revision": revision + 5})
    assert stale.status_code == 409
    done = client.post(f"/api/v1/remotea/runs/{run_id}/brainstorm/plan", json={"expected_revision": revision})
    assert done.status_code == 200 and done.json()["brainstorm"]["open"] is False and done.json()["state"] == "planning"
