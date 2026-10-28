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
    assert {"control.start", "control.message", "control.stop", "control.answer"} <= set(COMMANDS)
    assert {"control.start", "control.message", "control.stop", "control.answer"} <= VOICE_MUTATIONS
    assert set(control.COMMANDS) == {"control.start", "control.message", "control.stop", "control.answer"}
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
    started = say("control.start", task="voice controls")
    assert started["task"]["id"] == created["id"] and started["state"] == "spawning"
    with session_scope() as db:
        run = db.get(AgentRun, started["id"])
        [spawn] = db.scalars(select(AgentCommand).where(AgentCommand.run_id == run.id, AgentCommand.kind == "spawn"))
        assert run.history[0]["actor"]["kind"] == "human" and spawn.status == "queued"
    with pytest.raises(DomainError) as twice:
        say("control.start", task_id=created["id"])
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
    run_id = say("control.start", task_id=created["id"])["id"]
    attached(host, run_id)
    say("control.message", text="Also update the docs.")  # The only active run.
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
    run_id = say("control.start", task_id=created["id"])["id"]
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
    run_id = say("control.start", task_id=created["id"])["id"]
    result = asyncio.run(call_tool(OWNER, "turn-4", 0, "control_status", {}))
    assert result["enabled"] and result["hosts"][0]["connected"]
    assert [r["run_id"] for r in result["runs"]] == [run_id] and result["runs"][0]["state"] == "spawning"
    with session_scope() as db:
        coordinator.request_stop(db, db.get(AgentRun, run_id), actor=control.Actor("human"))
    after = asyncio.run(call_tool(OWNER, "turn-5", 0, "control_status", {}))
    assert after["runs"] == [] and after["recently_finished"][0]["run_id"] == run_id
