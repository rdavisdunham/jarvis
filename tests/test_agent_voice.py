"""Stage E: Eri voice control of agent runs (control.* through domain.mutate, control_status) — the same
command service as the remotea page, present only while remotea is enabled."""

import asyncio
from uuid import uuid4

import pytest
from jarvis import agent_control as control
from jarvis import bot_access
from jarvis import run_coordinator as coordinator
from jarvis.db import session_scope
from jarvis.domain import COMMANDS, DomainError, execute
from jarvis.models import OwnerSettings
from jarvis.session_models import AgentCommand, AgentRun, ExecutionTarget
from jarvis.tool_catalog import DESCRIPTIONS, GROUPS
from jarvis.tool_policy import CONTROL_TOOLS, remotea_tools
from jarvis.tools import READ_TOOLS, VOICE_MUTATIONS, call_tool, registry
from sqlalchemy import select
from test_agent_runs import OWNER, asked, attached, state, task
from test_agent_runs import host as host  # noqa: F401 - pytest fixture
from test_agent_runs import quiet as quiet  # noqa: F401 - pytest fixture


def say(tool, **arguments):
    with session_scope() as db:
        return execute(db, OWNER, str(uuid4()), tool, arguments)["data"]


def confirm(pending):
    """Stage F: control_start / control_message are two-phase; the owner taps Confirm on the remotea page."""
    assert pending["status"] == "pending_confirmation", pending
    with session_scope() as db:
        return control.decide(db, OWNER, pending["pending_id"], confirm=True, via="app",
                              actor=control.Actor("human"))["run"]


def names(definitions):
    return {d["name"] for d in definitions}


def set_enabled(on):
    with session_scope() as db:
        row = db.get(OwnerSettings, OWNER)
        row.values = {**(row.values or {}), "remotea_enabled": on}


@pytest.fixture
def targeted(host):  # noqa: F811
    """Tasks resolve to the host's one execution target without asking."""
    with session_scope() as db:
        db.get(ExecutionTarget, host.target_id).record_id = None
    return host


def default_target(host, task_id):  # noqa: F811
    with session_scope() as db:
        from jarvis.models import Task

        db.get(Task, task_id).execution_target_id = host.target_id


# ---------------------------------------------------------------- registry and placement


def test_control_tools_are_registered_described_and_grouped():
    tools = {"control.start", "control.message", "control.confirm", "control.stop", "control.answer", "control.approve",
             "control.plan_review"}
    assert tools <= set(COMMANDS) and tools <= VOICE_MUTATIONS
    assert set(control.COMMANDS) == tools
    assert "control_status" in READ_TOOLS
    full = names(registry())
    assert set(CONTROL_TOOLS) <= full
    assert all(name in DESCRIPTIONS for name in CONTROL_TOOLS if name != "control_status")
    assert set(CONTROL_TOOLS) <= set(GROUPS["agents"][1]) and "navigation_find" in GROUPS["agents"][1]


def test_the_registry_has_control_tools_only_while_remotea_is_enabled(client):
    off = names(remotea_tools(registry(), OWNER))
    assert not off & set(CONTROL_TOOLS) and "task_list" in off
    client.post("/api/v1/remotea/enable", json={})
    on = names(remotea_tools(registry(), OWNER))
    assert set(CONTROL_TOOLS) <= on


def test_external_bots_never_get_control_tools():
    every = set(bot_access.SCOPES)
    assert not any(bot_access.tool_allowed(name, every) for name in CONTROL_TOOLS)


def test_control_is_refused_while_remotea_is_off(host):  # noqa: F811
    set_enabled(False)
    with pytest.raises(DomainError) as off:
        say("control.start", task="anything")
    assert off.value.code == "REMOTEA_DISABLED"
    result = asyncio.run(call_tool(OWNER, "turn-1", 0, "control_status", {}))
    assert result["enabled"] is False


# ---------------------------------------------------------------- dispatch reaches the coordinator


def test_voice_start_by_name_reaches_the_same_start_run(host):  # noqa: F811
    created = task("Ship the voice controls")
    default_target(host, created["id"])
    started = confirm(say("control.start", task="voice controls"))
    assert started["task"]["id"] == created["id"] and started["state"] == "spawning"
    with session_scope() as db:
        run = db.get(AgentRun, started["id"])
        [spawn] = db.scalars(select(AgentCommand).where(AgentCommand.run_id == run.id, AgentCommand.kind == "spawn"))
        assert run.history[0]["actor"]["kind"] == "human" and spawn.status == "queued"
    with pytest.raises(DomainError) as twice:
        confirm(say("control.start", task_id=created["id"]))
    assert twice.value.code == "RUN_ACTIVE"


def test_ambiguous_or_unknown_names_ask_instead_of_guessing(host):  # noqa: F811
    task("Fix login on web")
    task("Fix login on iOS")
    with pytest.raises(DomainError) as ambiguous:
        say("control.start", task="fix login")
    assert ambiguous.value.code == "AMBIGUOUS_TASK" and len(ambiguous.value.data["candidates"]) == 2
    with pytest.raises(DomainError) as unknown:
        say("control.start", task="nonexistent widget")
    assert unknown.value.code == "NOT_FOUND"


def test_message_stop_and_answer_by_voice(host):  # noqa: F811
    created = task("Voice driven run")
    default_target(host, created["id"])
    run_id = confirm(say("control.start", task_id=created["id"]))["id"]
    attached(host, run_id)
    confirm(say("control.message", text="Also update the docs."))  # The only active run.
    with session_scope() as db:
        inputs = [c.payload for c in db.scalars(select(AgentCommand).where(AgentCommand.run_id == run_id,
                                                                            AgentCommand.kind == "input"))]
        assert inputs[-1]["text"] == "Also update the docs." and inputs[-1]["mode"] == "queue"
    question_pk, _ = asked(host, run_id)
    status = asyncio.run(call_tool(OWNER, "turn-2", 0, "control_status", {"task": "voice driven"}))
    [spoken] = status["runs"]
    assert spoken["run_id"] == run_id and spoken["questions"][0]["question_id"] == question_pk
    item = spoken["questions"][0]["items"][0]
    say("control.answer", question_id=question_pk,
        answers=[{"item_id": item["item_id"], "choice_ids": [item["choices"][0]["id"]]}])
    with session_scope() as db:
        answer = db.scalar(select(AgentCommand).where(AgentCommand.run_id == run_id, AgentCommand.kind == "answer"))
        assert answer.payload["question_id"] == "q-toolu_1" and answer.payload["answers"][0]["choice_ids"] == ["pg"]
    say("control.stop", task="voice driven")
    assert state(run_id) == ("stopping", "stopped_by_user")


def test_voice_answers_a_planning_interview(host):  # noqa: F811
    from jarvis.session_models import AgentInterview

    created = task("Interviewed by voice")
    default_target(host, created["id"])
    run_id = confirm(say("control.start", task_id=created["id"]))["id"]
    attached(host, run_id)
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        assert run.state == "interviewing"
        db.add(AgentInterview(owner_id=OWNER, run_id=run_id, generation=run.generation, status="open",
                              questions=[{"id": "q1", "question": "Which DB?", "options": [{"label": "Postgres"}],
                                          "multi": False, "free_text": False}], answers={}))
    status = asyncio.run(call_tool(OWNER, "turn-3", 0, "control_status", {}))
    interview = status["runs"][0]["interview"]
    assert interview["questions"][0]["question"] == "Which DB?"
    say("control.answer", interview_id=interview["id"], responses={"q1": "Postgres"})
    assert state(run_id) == ("planning", "interview_answered")


def test_status_lists_active_runs_with_host_state(host):  # noqa: F811
    created = task("Status check")
    default_target(host, created["id"])
    run_id = confirm(say("control.start", task_id=created["id"]))["id"]
    result = asyncio.run(call_tool(OWNER, "turn-4", 0, "control_status", {}))
    assert result["enabled"] and result["hosts"][0]["connected"]
    assert [r["run_id"] for r in result["runs"]] == [run_id] and result["runs"][0]["state"] == "spawning"
    with session_scope() as db:
        coordinator.request_stop(db, db.get(AgentRun, run_id), actor=control.Actor("human"))
    after = asyncio.run(call_tool(OWNER, "turn-5", 0, "control_status", {}))
    assert after["runs"] == [] and after["recently_finished"][0]["run_id"] == run_id


# ---------------------------------------------------------------- approve, plan review and the open run (eri-chat-runs)


def judged_met(run_id, summary="Every criterion is met: the endpoint returns 200 and the tests pass."):
    """A run the judge passed, as the coordinator leaves it: awaiting_approval with a done judge attempt."""
    from jarvis.session_models import AgentAttempt

    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        run.state, run.state_reason, run.state_revision = "awaiting_approval", "judge_met", run.state_revision + 1
        db.add(AgentAttempt(owner_id=OWNER, run_id=run_id, generation=run.generation, kind="judge", status="done",
                            result={"verdict": "met", "summary": summary, "gap": "", "criteria": []}))


def test_approve_by_voice_only_for_a_run_awaiting_approval(host):  # noqa: F811
    from jarvis.models import Task

    created = task("Approve me by voice")
    default_target(host, created["id"])
    run_id = confirm(say("control.start", task_id=created["id"]))["id"]
    attached(host, run_id)
    with pytest.raises(DomainError) as early:
        say("control.approve", task_id=created["id"])
    assert early.value.code == "NOT_AWAITING_APPROVAL" and early.value.data["state"] == state(run_id)[0]
    assert state(run_id)[0] in early.value.message and "remotea page" in early.value.message
    with session_scope() as db:
        assert db.get(Task, created["id"]).status != "completed"  # Nothing was approved.
    judged_met(run_id)
    status = asyncio.run(call_tool(OWNER, "turn-6", 0, "control_status", {}))
    [spoken] = status["runs"]
    assert spoken["state"] == "awaiting_approval" and spoken["awaiting_approval"]["verdict"] == "met"
    assert spoken["awaiting_approval"]["summary"].startswith("Every criterion is met")
    assert "control_approve" in spoken["awaiting_approval"]["next"]
    done = say("control.approve")  # The only active run.
    assert done["status"] == "approved" and done["run_id"] == run_id and done["task"]["status"] == "completed"
    with session_scope() as db:
        assert db.get(Task, created["id"]).status == "completed"
        run = db.get(AgentRun, run_id)
        assert run.state == "stopping" and run.sync_state["completion"]["approved"] is True
    with pytest.raises(DomainError) as again:
        say("control.approve", run_id=run_id)
    assert again.value.code == "NOT_AWAITING_APPROVAL" and "stopping" in again.value.message


def test_plan_review_by_voice_decides_each_proposal_and_freezes_the_plan(host):  # noqa: F811
    from jarvis import agent_plan
    from jarvis.models import Task
    from test_agent_plan import call, turn
    from test_agent_plan import start as start_run

    created = task("Reviewed by voice", success_criteria=[{"id": "c1", "text": "compares only resume.cmd"},
                                                          {"id": "c2", "text": "docs updated"}])
    run_id = start_run(host, created["id"])["id"]
    attached(host, run_id)
    with pytest.raises(DomainError) as early:
        say("control.plan_review", run_id=run_id, decisions=[{"id": "c1", "accept": True}])
    assert early.value.code == "NO_PLAN_REVIEW" and state(run_id)[0] in early.value.message
    with session_scope() as db:
        plan_revision = db.get(Task, created["id"]).plan_revision or 0
    call(agent_plan.submit_plan, agent_plan.SubmitPlanInput, run_id, prd="Parity.", steps=[{"title": "Do it"}],
         expected_plan_revision=plan_revision, success_criteria=[
             {"id": "c1", "text": "compares resume.cmd and resume.url", "because": "You said both"},
             {"id": "c2", "text": "docs updated", "drop": True, "because": "No docs here"}])
    assert turn(run_id, 1) == ("needs_you", "plan_review")
    status = asyncio.run(call_tool(OWNER, "turn-7", 0, "control_status", {"task": "reviewed by voice"}))
    [spoken] = status["runs"]
    review = spoken["plan_review"]
    assert [c["id"] for c in review["proposals"]["criteria"]] == ["c1", "c2"] and review["plan_revision"] == 1
    assert "control_plan_review" in review["next"]
    out = say("control.plan_review", task="reviewed by voice", decisions=[
        {"id": "c1", "accept": True, "text": "compares resume.cmd and resume.url only"}, {"id": "c2", "accept": False}])
    assert out["status"] == "plan_reviewed" and out["state"] == "executing" and out["plan_revision"] == 2
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        assert row.plan_revision == 2 and not row.plan_pending
        texts = [c["text"] for c in db.get(Task, created["id"]).success_criteria]
        assert texts == ["compares resume.cmd and resume.url only", "docs updated"]
        assert [h for h in row.history if h.get("event") == "plan_reviewed"][0]["accepted"] == ["c1"]
    with pytest.raises(DomainError) as again:
        say("control.plan_review", run_id=run_id, decisions=[])
    assert again.value.code == "NO_PLAN_REVIEW" and "executing" in again.value.message


def test_a_target_defaults_to_the_run_open_on_the_owners_screen(host, monkeypatch):  # noqa: F811
    """Two runs are active, so a bare control command is ambiguous; with the remotea page's run open on the device
    that sent the request (UIContext.open_run_id), the command means that run. A run that is not the owner's does
    not count."""
    from jarvis import device_bridge
    from jarvis.access import execution
    from jarvis.agent_control_schema import Target

    first, second = task("Open run one"), task("Open run two")
    default_target(host, first["id"])
    default_target(host, second["id"])
    one = confirm(say("control.start", task_id=first["id"]))["id"]
    two = confirm(say("control.start", task_id=second["id"]))["id"]
    with session_scope() as db, pytest.raises(DomainError) as ambiguous:
        control.run_for(db, OWNER, Target())
    assert ambiguous.value.code == "AMBIGUOUS_RUN"
    screen = {"open_run_id": two}
    monkeypatch.setattr(device_bridge, "context", lambda owner, device: dict(screen))
    token = execution.set({"id": "work-1", "owner_id": OWNER, "account_id": host.account_id, "device_id": "laptop",
                           "revision": 1, "credential_id": None})
    try:
        with session_scope() as db:
            assert control.run_for(db, OWNER, Target()) == two
            assert control.run_for(db, OWNER, Target(run_id=one)) == one  # An explicit target still wins.
            assert control.run_for(db, OWNER, Target(task_id=first["id"])) == one
        screen["open_run_id"] = "00000000-0000-0000-0000-000000000000"
        with session_scope() as db, pytest.raises(DomainError) as unknown:
            control.run_for(db, OWNER, Target())
        assert unknown.value.code == "AMBIGUOUS_RUN"
        screen.clear()
        with session_scope() as db, pytest.raises(DomainError):
            control.run_for(db, OWNER, Target())
    finally:
        execution.reset(token)
