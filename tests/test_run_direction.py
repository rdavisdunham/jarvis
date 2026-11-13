"""Direction check: the owner answers the agent's brief once, then the revised plan starts (run_direction.py)."""

from jarvis import agent_judge, agent_plan, agent_prompts, run_direction
from jarvis import run_coordinator as coordinator
from jarvis.db import session_scope
from jarvis.models import Task
from jarvis.session_models import AgentRun
from test_agent_plan import PLAN, call
from test_agent_runs import HUMAN, OWNER, attached, quiet, task  # noqa: F401 - pytest fixtures
from test_agent_runs import host as host  # noqa: F401 - pytest fixture
from test_agent_plan import start

BRIEF = {"reading": "Rename the helper everywhere it is used.", "approach": "One mechanical rename, then the tests.",
         "leaves_alone": "The public API names.", "questions": ["Rename the CLI flag too?", " "]}


def planning(host, route="direction"):
    made = task("Rename a helper", run_route=route) if route else task("Rename a helper")
    result = start(host, made["id"])
    attached(host, result["id"])
    return made["id"], result["id"]


def submit(run_id, task_id, **extra):
    with session_scope() as db:
        revision = db.get(Task, task_id).plan_revision
    return call(agent_plan.submit_plan, agent_plan.SubmitPlanInput, run_id,
                **{**PLAN, "expected_plan_revision": revision, **extra})


def state(run_id):
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        return row.state, row.state_reason, coordinator.frozen(db, row).approval_status


def public(run_id):
    with session_scope() as db:
        return run_direction.public(db, db.get(AgentRun, run_id))


def test_the_owner_answers_the_brief_once_and_the_revised_plan_starts(host):
    task_id, run_id = planning(host)
    with session_scope() as db:
        assert "`brief`" in agent_prompts.interview(db.get(Task, task_id), [])
    submit(run_id, task_id, brief=BRIEF)
    assert state(run_id) == ("needs_you", "plan_approval", "pending")
    shown = public(run_id)
    assert shown["open"] and shown["brief"]["questions"] == ["Rename the CLI flag too?"]
    assert shown["brief"]["reading"] == BRIEF["reading"] and shown["answer"] is None
    with session_scope() as db:
        coordinator.revise_plan(db, db.get(AgentRun, run_id, with_for_update=True), "Yes, and keep the old flag as an alias.",
                                actor=HUMAN)
    shown = public(run_id)
    assert not shown["open"] and shown["answer"] == "Yes, and keep the old flag as an alias."
    submit(run_id, task_id)  # The revised plan: no second stop.
    assert state(run_id)[0] == "executing" and state(run_id)[2] == "approved"
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        assert [h["actor"] for h in row.history if h.get("event") == "plan_approved"] == [{"kind": "direction_check"}]
        assert run_direction.evidence(row) == {"brief": {**BRIEF, "questions": ["Rename the CLI flag too?"]},
                                               "owner_answer": "Yes, and keep the old flag as an alias.", "accepted": False}
        assert "`direction`" in agent_judge.SYSTEM


def test_accepting_the_brief_starts_the_plan(host):
    task_id, run_id = planning(host)
    submit(run_id, task_id, brief=BRIEF)
    with session_scope() as db:
        coordinator.approve_plan(db, db.get(AgentRun, run_id, with_for_update=True), actor=HUMAN)
    assert state(run_id)[0] == "executing"
    shown = public(run_id)
    assert shown["accepted_at"] and not shown["open"]


def test_a_migration_after_the_answer_still_stops_and_other_routes_are_unchanged(host):
    task_id, run_id = planning(host)
    submit(run_id, task_id, brief=BRIEF)
    with session_scope() as db:
        coordinator.revise_plan(db, db.get(AgentRun, run_id, with_for_update=True), "Go ahead.", actor=HUMAN)
    submit(run_id, task_id, migration=True)
    assert state(run_id) == ("needs_you", "plan_approval", "pending")
    other_task, other = planning(host, route=None)
    with session_scope() as db:
        assert "`brief`" not in agent_prompts.interview(db.get(Task, other_task), [])
    submit(other, other_task, brief=BRIEF)  # A brief nobody asked for is kept but changes nothing.
    with session_scope() as db:
        coordinator.revise_plan(db, db.get(AgentRun, other, with_for_update=True), "Change it.", actor=HUMAN)
    submit(other, other_task)
    assert state(other) == ("needs_you", "plan_approval", "pending")
    assert public(other)["open"] is False and public(other)["answer"] is None


def test_a_direction_run_plans_from_the_task_instead_of_interviewing_first(host):
    """The brief carries the agent's questions: an interview before it would ask the owner twice."""
    from jarvis import agent_control as control

    direction, other = task("Make the greeting friendlier", run_route="direction"), task("Rename a helper")
    with session_scope() as db:
        assert control.preflight(db, OWNER, direction["id"])["interview"] is False
        assert control.preflight(db, OWNER, other["id"])["interview"] is True
    states = {}
    for name, made in (("direction", direction), ("other", other)):
        run_id = start(host, made["id"])["id"]
        attached(host, run_id)
        with session_scope() as db:
            states[name] = db.get(AgentRun, run_id).state
    assert states["other"] == "interviewing" and states["direction"] != "interviewing", states


def test_an_answer_sent_before_the_planning_turn_ends_is_not_a_missing_plan(host):
    """A managed session's plan stops for the owner the moment it is submitted, while the agent is still writing
    its closing words. An owner who answers in those seconds must not see "ended planning without a plan"."""
    from uuid import uuid4

    from test_agent_plan import turn

    task_id, run_id = planning(host)
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        row.sync_state = {**row.sync_state, "execution_mode": "managed"}
    submit(run_id, task_id, brief=BRIEF, submission_id=str(uuid4()))
    assert state(run_id)[:2] == ("needs_you", "plan_approval")
    with session_scope() as db:
        coordinator.revise_plan(db, db.get(AgentRun, run_id, with_for_update=True), "Keep the old flag.", actor=HUMAN)
    turn(run_id, 1)  # The turn that submitted the first plan ends only now.
    assert state(run_id)[:2] == ("planning", "plan_revise")
    submit(run_id, task_id, submission_id=str(uuid4()))
    assert state(run_id)[0] != "needs_you"
