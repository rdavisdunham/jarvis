"""Stage D: the interview phase, submit_plan, plan revisions, child seeding and awaiting_integration."""

from types import SimpleNamespace

import pytest
from jarvis import agent_control as control
from jarvis import agent_plan
from jarvis import run_coordinator as coordinator
from jarvis.agent_actor import Actor
from jarvis.db import session_scope
from jarvis.domain import DomainError
from jarvis.models import Note, Notification, Task
from jarvis.session_models import AgentCommand, AgentHost, AgentInterview, AgentRun, AgentRunPlan
from sqlalchemy import select
from test_agent_runs import OWNER, attached, quiet, state, task  # noqa: F401 - fixture
from test_agent_runs import host as host  # noqa: F401 - pytest fixture
from test_linear import run

HUMAN = Actor("human")
BOT = SimpleNamespace(owner_id=OWNER, id=None, scopes=["agents:session"])


@pytest.fixture
def verifying_host(host):  # noqa: F811
    """The host advertises verify and a `tests` verify id on its target."""
    with session_scope() as db:
        row = db.get(AgentHost, host.id)
        row.caps = {**row.caps, "caps": [*row.caps["caps"], "verify"],
                    "targets": [{"id": "jarvis", "repo": "/home/davin/jarvis", "worktree_root": "/home/davin/wt",
                                 "verify": ["tests"]}]}
    return host


def start(host, task_id, **values):
    with session_scope() as db:
        return control.start_run(db, OWNER, host.account_id,
                                 control.StartInput(task_id=task_id, execution_target_id=host.target_id, **values),
                                 actor=HUMAN)


def token(run_id):
    with session_scope() as db:
        return coordinator.run_token(db.get(AgentRun, run_id))


def call(fn, model, run_id, **body):
    with session_scope() as db:
        return fn(db, BOT, model.model_validate({"run_token": token(run_id), **body}))


def turn(run_id, seq):
    with session_scope() as db:
        row = db.get(AgentRun, run_id, with_for_update=True)
        coordinator.record_turn(db, row, SimpleNamespace(generation=row.generation, seq=seq, interim=False, synthetic=False))
        return row.state, row.state_reason


def inputs(run_id):
    with session_scope() as db:
        return [c.payload["text"] for c in db.scalars(select(AgentCommand).where(
            AgentCommand.run_id == run_id, AgentCommand.kind == "input").order_by(AgentCommand.created_at))]


PLAN = {"prd": "Make the widget fast.", "steps": [{"title": "Profile"}, {"title": "Fix the hot loop"}],
        "success_criteria": [{"text": "p95 under 50 ms"}], "expected_plan_revision": 0}


def interviewing(host, title="Speed up widgets"):  # noqa: F811
    created = task(title)
    result = start(host, created["id"])
    attached(host, result["id"])
    return created, result["id"]


def test_a_task_without_a_prd_starts_with_the_interview(host):  # noqa: F811
    created, run_id = interviewing(host)
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        spawn = db.scalar(select(AgentCommand).where(AgentCommand.run_id == run_id, AgentCommand.kind == "spawn"))
        assert row.plan_revision == 0 and coordinator.frozen(db, row) is None
        assert "task_interview" in spawn.payload["prompt"] and "verify" not in spawn.payload
    assert state(run_id) == ("interviewing", "started")


def test_interview_answers_planning_and_execution_after_settle(host):  # noqa: F811
    created, run_id = interviewing(host)
    asked = call(agent_plan.task_interview, agent_plan.InterviewInput, run_id, questions=[
        {"id": "scope", "question": "Which widgets?", "options": [{"label": "All"}, {"label": "Only charts"}]},
        {"id": "budget", "question": "Latency target?"},
        {"id": "where", "question": "Platforms?", "options": [{"label": "Web"}, {"label": "iOS"}], "multi": True}])
    with pytest.raises(DomainError) as twice:
        call(agent_plan.task_interview, agent_plan.InterviewInput, run_id, questions=[{"id": "x", "question": "?"}])
    assert twice.value.code == "INTERVIEW_OPEN"
    with session_scope() as db:
        assert db.scalar(select(Notification).where(Notification.category == "agent_question"))
        row = db.get(AgentInterview, asked["interview_id"])
        assert row.questions[1]["free_text"] is True  # No options → free text.
    assert turn(run_id, 1) == ("interviewing", "started")  # The interview turn ends; the operator has the move.
    with session_scope() as db, pytest.raises(DomainError) as bad:
        agent_plan.answer_interview(db, OWNER, asked["interview_id"], agent_plan.InterviewAnswers(
            answers={"scope": "Some", "budget": "50ms", "where": ["Web"]}), actor=HUMAN)
    assert bad.value.code == "INVALID_ARGUMENT"
    with session_scope() as db:
        answered = agent_plan.answer_interview(db, OWNER, asked["interview_id"], agent_plan.InterviewAnswers(
            answers={"scope": "Only charts", "budget": "50 ms", "where": ["Web", "iOS"]}), actor=HUMAN)
    assert answered["status"] == "answered" and answered["answers"]["where"] == ["Web", "iOS"]
    assert state(run_id)[0] == "planning"
    assert "Only charts" in inputs(run_id)[-1] and "submit_plan" in inputs(run_id)[-1]
    with session_scope() as db, pytest.raises(DomainError) as closed:
        agent_plan.answer_interview(db, OWNER, asked["interview_id"], agent_plan.InterviewAnswers(
            answers={"scope": "All", "budget": "1", "where": ["Web"]}), actor=HUMAN)
    assert closed.value.code == "INTERVIEW_CLOSED"

    plan = call(agent_plan.submit_plan, agent_plan.SubmitPlanInput, run_id, **PLAN)
    assert plan["plan_revision"] == 1 and plan["task_plan_revision"] >= 1
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        frozen = coordinator.frozen(db, row)
        prd = db.get(Note, plan["prd_note_id"])
        refreshed = db.get(Task, created["id"])
        assert row.state == "planning" and row.plan_pending  # Not executing until the planning turn settles.
        assert prd.title == "PRD: Speed up widgets" and "prd" in prd.tags and refreshed.prd_note_id == prd.id
        assert frozen.revision == 1 and frozen.criteria[0]["text"] == "p95 under 50 ms" and len(frozen.steps) == 2
        assert frozen.task_plan_revision == refreshed.plan_revision
    assert turn(run_id, 2) == ("executing", "plan_frozen")
    assert "Your plan is frozen" in inputs(run_id)[-1]
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        assert row.turn_count == 0 and not row.plan_pending  # Interview and planning turns never count.
    with pytest.raises(DomainError) as locked:
        call(agent_plan.submit_plan, agent_plan.SubmitPlanInput, run_id, **{**PLAN, "expected_plan_revision": 1})
    assert locked.value.code == "PLAN_LOCKED"


def test_planning_turn_with_a_pending_question_waits(host):  # noqa: F811
    created, run_id = interviewing(host)
    call(agent_plan.submit_plan, agent_plan.SubmitPlanInput, run_id, **PLAN)  # Straight from the interview phase.
    with session_scope() as db:
        row = db.get(AgentRun, run_id, with_for_update=True)
        coordinator.transition(db, row, "waiting_answer", reason="question")
        db.add(coordinator.AgentQuestion(owner_id=OWNER, session_pk=row.session_pk, host_id=host.id, run_id=row.id,
                                         generation=row.generation, question_id="q1", provider_request_id="t1",
                                         group_id="t1", kind="ask", items=[{"id": "a", "text": "?"}], status="open"))
    assert turn(run_id, 1)[0] == "planning"  # Back to its phase, but a question is still open.
    with session_scope() as db:
        question = db.scalar(select(coordinator.AgentQuestion).where(coordinator.AgentQuestion.run_id == run_id))
        question.status = "accepted"
    assert turn(run_id, 2) == ("executing", "plan_frozen")


def test_a_planning_turn_without_a_plan_needs_you(host):  # noqa: F811
    created, run_id = interviewing(host)
    assert turn(run_id, 1) == ("needs_you", "no_plan")


def test_submit_plan_is_run_scoped_and_checks_revision_and_verify(verifying_host):  # noqa: F811
    created, run_id = interviewing(verifying_host)
    with session_scope() as db, pytest.raises(DomainError) as missing:
        agent_plan.submit_plan(db, BOT, agent_plan.SubmitPlanInput.model_validate(PLAN))
    assert missing.value.code == "RUN_TOKEN_REQUIRED"
    with pytest.raises(DomainError) as stale:
        call(agent_plan.submit_plan, agent_plan.SubmitPlanInput, run_id, **{**PLAN, "expected_plan_revision": 7})
    assert stale.value.code == "REVISION_CONFLICT"
    with pytest.raises(DomainError) as unknown:
        call(agent_plan.submit_plan, agent_plan.SubmitPlanInput, run_id,
             **{**PLAN, "verify": {"target_id": "jarvis", "verify_id": "deploy"}})
    assert unknown.value.code == "VERIFY_UNKNOWN"
    with pytest.raises(DomainError):
        call(agent_plan.submit_plan, agent_plan.SubmitPlanInput, run_id,
             **{**PLAN, "success_criteria": [{"text": "tests pass", "kind": "verify", "verify_id": "lint"}],
                "verify": {"target_id": "jarvis", "verify_id": "tests"}})
    plan = call(agent_plan.submit_plan, agent_plan.SubmitPlanInput, run_id,
                **{**PLAN, "success_criteria": [{"text": "tests pass", "kind": "verify", "verify_id": "tests"}],
                   "verify": {"target_id": "jarvis", "verify_id": "tests"}})
    assert plan["verify"] == {"target_id": "jarvis", "verify_id": "tests", "by": "agent"}
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        assert coordinator.desired(db, row).get("verify") is None  # Pending: the host learns it once executing.
    turn(run_id, 1)
    with session_scope() as db:
        assert coordinator.desired(db, db.get(AgentRun, run_id))["verify"] == {"verify_id": "tests", "plan_revision": 1}


def test_a_task_with_a_prd_skips_the_interview_and_spawns_with_its_verify(verifying_host):  # noqa: F811
    with session_scope() as db:
        note = Note(owner_id=OWNER, title="PRD", content="Do it.")
        db.add(note)
        db.flush()
        note_id = note.id
    created = task("Planned", prd_note_id=note_id, success_criteria=[{"text": "done", "kind": "verify", "verify_id": "tests"}],
                   verify={"target_id": "jarvis", "verify_id": "tests"})
    result = start(verifying_host, created["id"])
    attached(verifying_host, result["id"])
    assert state(result["id"]) == ("executing", "started")
    with session_scope() as db:
        spawn = db.scalar(select(AgentCommand).where(AgentCommand.run_id == result["id"], AgentCommand.kind == "spawn"))
        assert spawn.payload["verify"] == {"verify_id": "tests", "plan_revision": 1}


def test_children_seed_linked_steps_and_are_frozen_as_required(host):  # noqa: F811
    parent = task("Parent")
    first = run("task.create", title="Child one", parent_task_id=parent["id"])
    second = run("task.create", title="Child two", parent_task_id=parent["id"])
    run("task.create", title="Unrelated")
    result = start(host, parent["id"])
    attached(host, result["id"])
    with session_scope() as db:
        spawn = db.scalar(select(AgentCommand).where(AgentCommand.run_id == result["id"], AgentCommand.kind == "spawn"))
        assert "Child one" in spawn.payload["prompt"] and "Child two" in spawn.payload["prompt"]
    with pytest.raises(DomainError) as foreign:
        call(agent_plan.submit_plan, agent_plan.SubmitPlanInput, result["id"],
             **{**PLAN, "steps": [{"title": "x", "task_id": parent["id"]}]})
    assert foreign.value.code == "INVALID_ARGUMENT"
    plan = call(agent_plan.submit_plan, agent_plan.SubmitPlanInput, result["id"],
                **{**PLAN, "steps": [{"title": "Do child two first", "task_id": second["id"]}, {"title": "Plain step"}]})
    linked = [s.get("task_id") for s in plan["steps"]]
    assert linked == [second["id"], None, first["id"]]  # The agent's order, then the children it left out.
    with session_scope() as db:
        frozen = db.scalar(select(AgentRunPlan).where(AgentRunPlan.run_id == result["id"]))
        assert set(frozen.required_children) == {first["id"], second["id"]}
        note = db.get(Note, plan["prd_note_id"])
        assert "## Sub-tasks" in note.content and "Child one" in note.content


def test_replan_advances_the_generation_and_freezes_the_next_revision(host):  # noqa: F811
    created, run_id = interviewing(host)
    call(agent_plan.submit_plan, agent_plan.SubmitPlanInput, run_id, **PLAN)
    turn(run_id, 1)
    with session_scope() as db:
        before = db.get(AgentRun, run_id).generation
        control.replan(db, OWNER, run_id, control.ReplanInput(reason="Scope changed"), actor=HUMAN)
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        assert row.generation == before + 1 and row.state == "planning"
        commands = list(db.scalars(select(AgentCommand).where(AgentCommand.run_id == run_id, AgentCommand.kind == "input")
                                   .order_by(AgentCommand.created_at)))
        assert commands[-1].payload["mode"] == "steer" and "Scope changed" in commands[-1].payload["text"]
        task_revision = db.get(Task, created["id"]).plan_revision
    again = call(agent_plan.submit_plan, agent_plan.SubmitPlanInput, run_id,
                 **{**PLAN, "prd": "Make it faster still.", "expected_plan_revision": task_revision})
    assert again["plan_revision"] == 2
    with session_scope() as db:
        assert db.get(Note, again["prd_note_id"]).content.startswith("Make it faster still.")
        assert coordinator.desired(db, db.get(AgentRun, run_id)).get("verify", "absent") is None  # Cleared on re-plan.


def test_awaiting_integration_until_the_child_branch_is_merged(host):  # noqa: F811
    parent = task("Parent")
    child = run("task.create", title="Child", parent_task_id=parent["id"])
    parent_run = start(host, parent["id"])
    attached(host, parent_run["id"])
    call(agent_plan.submit_plan, agent_plan.SubmitPlanInput, parent_run["id"], **PLAN)
    turn(parent_run["id"], 1)
    child_run = start(host, child["id"], interview=False)
    with session_scope() as db:
        assert db.get(AgentRun, child_run["id"]).base_branch == db.get(AgentRun, parent_run["id"]).branch
    with session_scope() as db:
        row = db.get(AgentRun, parent_run["id"], with_for_update=True)
        row.turn_count, row.turn_seq = 1, 1
        coordinator.transition(db, row, "judging")
        plan = coordinator.frozen(db, row)
        pending = coordinator.integration_pending(db, row, plan)
        assert pending and pending[0]["reason"] == "child run not completed"
        attempt = coordinator.AgentAttempt(owner_id=OWNER, run_id=row.id, generation=row.generation, kind="judge",
                                           turn_seq=1, plan_revision=row.plan_revision, status="done")
        db.add(attempt)
        db.flush()
        coordinator.on_verdict(db, attempt, {"verdict": "met", "gap": "", "criteria": [], "steps": []})
        assert row.state == "awaiting_integration"
        child_row = db.get(AgentRun, child_run["id"])
        child_row.state, parent_branch, child_branch = "completed", row.branch, child_row.branch
    with session_scope() as db:
        hosts = db.get(AgentHost, host.id)
        hosts.caps = {**hosts.caps, "worktrees": {"worktrees": [
            {"branch": child_branch, "base": parent_branch, "merged": False}]}}
        coordinator.check_integrations(db, OWNER, host.id)
        assert db.get(AgentRun, parent_run["id"]).state == "awaiting_integration"
        hosts.caps = {**hosts.caps, "worktrees": {"worktrees": [
            {"branch": child_branch, "base": parent_branch, "merged": True}]}}
        coordinator.check_integrations(db, OWNER, host.id)
        assert db.get(AgentRun, parent_run["id"]).state == "executing"  # Pull, re-verify and judge the combined code.
    assert "merged into" in inputs(parent_run["id"])[-1]



def test_a_restored_surface_returns_to_the_planning_phase_not_execution(host):  # noqa: F811
    created, run_id = interviewing(host)
    with session_scope() as db:
        row = db.get(AgentRun, run_id, with_for_update=True)
        coordinator.transition(db, row, "needs_you", reason="host_offline")
        coordinator.reconcile(db, OWNER, host.id, [SimpleNamespace(
            surf=row.surf, run_id=run_id, state="idle", incarnation=1, generation=1, session_id=None, tool="claude")])
    assert state(run_id) == ("interviewing", "surface_restored")
