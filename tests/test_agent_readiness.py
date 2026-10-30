"""Readiness and freshness at decision points (E-S6, task f8d32b61): the verify deadline follows the target's
timeout, workspace completeness gates auto-complete, the default verify prefers a ready id, a target without
Eridani tools does not wait for submit_plan, and each freshness fact is reported apart."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
from jarvis import agent_control as control
from jarvis import agent_link, agent_plan
from jarvis import run_coordinator as coordinator
from jarvis.db import session_scope
from jarvis.domain import DomainError
from jarvis.models import OwnerSettings, Task, now
from jarvis.session_models import AgentAttempt, AgentCommand, AgentHost, AgentRun
from sqlalchemy import select
from test_agent_attempts import attempts, planned, result
from test_agent_judge import judged, stub, verdict
from test_agent_plan import HUMAN, interviewing, spawned, start, turn, verifying_host  # noqa: F401 - fixture
from test_agent_runs import OWNER, attached, quiet, state, task  # noqa: F401 - fixture
from test_agent_runs import host as host  # noqa: F401 - pytest fixture


def target_facts(host, **values):  # noqa: F811
    """Replace the jarvis target's hello entry (as a caps_version 2 host would send it)."""
    with session_scope() as db:
        row = db.get(AgentHost, host.id)
        row.caps = {**row.caps, "targets": [{**row.caps["targets"][0], **values}]}


def auto_complete_on():
    with session_scope() as db:
        db.get(OwnerSettings, OWNER).values = {**db.get(OwnerSettings, OWNER).values, "remotea_auto_complete": True}


# ---------------------------------------------------------------- 1. the verify deadline follows the target's timeout


def test_verify_timeout_reads_the_hosts_advert():
    item = {"id": "jarvis", "verify": ["tests", {"id": "web", "timeout": 900}],
            "verify_status": [{"id": "tests", "ready": True, "timeout": 1800}, {"id": "lint", "ready": True, "timeout": True},
                              {"id": "huge", "ready": True, "timeout": 10 ** 9}]}
    assert control.verify_timeout(item, "tests") == 1800  # On the status row.
    assert control.verify_timeout(item, "web") == 900  # On a verify entry sent as an object.
    assert control.verify_timeout(item, "lint") is None and control.verify_timeout(item, "huge") is None  # Malformed.
    assert control.verify_timeout({"verify": ["tests"]}, "tests") is None  # Plain ids: the host did not say.
    assert control.verify_timeout(None, "tests") is None
    assert control.target_verify_status(item)[0] == {"id": "tests", "ready": True, "timeout": 1800}


def test_the_deadline_follows_the_verify_entrys_timeout(verifying_host):  # noqa: F811
    target_facts(verifying_host, verify=[{"id": "tests", "timeout": 1800}])
    run_id = planned(verifying_host)
    turn(run_id, 3)
    with session_scope() as db:
        attempt = db.scalar(select(AgentAttempt).where(AgentAttempt.run_id == run_id, AgentAttempt.kind == "verify"))
        assert attempt.result["timeout_s"] == 1800
        allowed = attempt.deadline_at - attempt.created_at
        assert timedelta(seconds=1800 + 119) < allowed < timedelta(seconds=1800 + 121)
    with session_scope() as db:
        coordinator.sweep(db, now() + timedelta(minutes=25))  # Past the old fixed 11 minutes: still running.
    assert attempts(run_id)[0].status == "running" and state(run_id)[0] == "verifying"
    with session_scope() as db:
        coordinator.sweep(db, now() + timedelta(minutes=33))
    assert attempts(run_id)[0].status == "timed_out" and state(run_id)[0] == "judging"


def test_a_short_or_missing_timeout_keeps_the_eleven_minute_floor(verifying_host):  # noqa: F811
    target_facts(verifying_host, verify=[{"id": "tests", "timeout": 60}])
    run_id = planned(verifying_host)
    turn(run_id, 3)
    with session_scope() as db:
        attempt = db.scalar(select(AgentAttempt).where(AgentAttempt.run_id == run_id, AgentAttempt.kind == "verify"))
        assert attempt.deadline_at - attempt.created_at >= coordinator.VERIFY_DEADLINE - timedelta(seconds=1)
        assert attempt.result["timeout_s"] == 60
    target_facts(verifying_host, verify=["tests"])  # Today's remotea: plain ids, no timeout.
    second = planned(verifying_host, title="Second")
    turn(second, 3)
    with session_scope() as db:
        attempt = db.scalar(select(AgentAttempt).where(AgentAttempt.run_id == second, AgentAttempt.kind == "verify"))
        allowed = attempt.deadline_at - attempt.created_at
        assert coordinator.VERIFY_DEADLINE - timedelta(seconds=1) < allowed < coordinator.VERIFY_DEADLINE + timedelta(seconds=1)
        assert "timeout_s" not in attempt.result


# ---------------------------------------------------------------- 2. workspace completeness and isolation


INCOMPLETE = {"isolated": False, "isolation": "none", "isolation_reason": "bwrap (bubblewrap) is not installed on the service PATH",
              "workspace_complete": False, "workspace_reasons": ["untracked files capped at 2000 (2400 found)"]}


def test_verify_result_keeps_isolation_and_completeness_and_drops_malformed_values():
    frame = agent_link.VerifyResult.model_validate({
        "attempt_id": "a", "run_id": "r", "generation": 1, "turn_seq": 1, "plan_revision": 1, **INCOMPLETE})
    assert coordinator.workspace_facts(frame) == INCOMPLETE
    odd = agent_link.VerifyResult.model_validate({
        "attempt_id": "a", "run_id": "r", "generation": 1, "turn_seq": 1, "plan_revision": 1,
        "isolated": "yes", "isolation": "docker", "workspace_complete": 0, "workspace_reasons": "lots"})
    assert (odd.isolated, odd.isolation, odd.workspace_complete, odd.workspace_reasons) == (None, None, None, [])
    assert coordinator.workspace_facts(odd) == {}  # An older host: nothing claimed either way.
    complete = agent_link.VerifyResult.model_validate({
        "attempt_id": "a", "run_id": "r", "generation": 1, "turn_seq": 1, "plan_revision": 1,
        "isolated": True, "isolation": "bwrap", "workspace_complete": True, "workspace_reasons": []})
    assert coordinator.workspace_facts(complete) == {"isolated": True, "isolation": "bwrap", "workspace_complete": True}


def test_an_incomplete_fingerprint_is_not_fresh_enough_to_auto_complete(verifying_host, client):  # noqa: F811
    auto_complete_on()
    run_id = planned(verifying_host)
    turn(run_id, 3)
    result(verifying_host, run_id, attempts(run_id)[0].id, **INCOMPLETE)
    [attempt] = attempts(run_id)
    assert attempt.status == "passed"  # The exit code and the evidence's freshness are separate facts.
    assert {k: attempt.result[k] for k in INCOMPLETE} == INCOMPLETE
    assert judged(run_id, stub(verdict())) == ("needs_you", "verify_incomplete")
    detail = client.get(f"/api/v1/remotea/runs/{run_id}").json()
    assert detail["reason_detail"] == "Verification evidence was incomplete: untracked files capped at 2000 (2400 found)"
    shown = next(a for a in detail["attempts"] if a["kind"] == "verify")
    assert shown["isolated"] is False and shown["workspace_complete"] is False
    assert shown["isolation_reason"] == INCOMPLETE["isolation_reason"] and shown["exit_code"] == 0
    assert detail["verification"]["evidence"] == "incomplete"
    with session_scope() as db:
        assert db.get(Task, db.get(AgentRun, run_id).task_id).status != "completed"
        spoken = control.status(db, OWNER, {"run_id": run_id})["runs"][0]
    assert "incomplete (untracked files capped at 2000 (2400 found))" in spoken["freshness"]
    assert "not sandboxed (bwrap (bubblewrap) is not installed on the service PATH)" in spoken["freshness"]


def test_a_complete_fingerprint_still_auto_completes(verifying_host):  # noqa: F811
    auto_complete_on()
    run_id = planned(verifying_host)
    turn(run_id, 3)
    result(verifying_host, run_id, attempts(run_id)[0].id, isolated=True, isolation="bwrap", workspace_complete=True)
    assert judged(run_id, stub(verdict())) == ("stopping", "judge_met")


def test_without_auto_complete_an_incomplete_fingerprint_waits_for_approval_as_before(verifying_host):  # noqa: F811
    run_id = planned(verifying_host)
    turn(run_id, 3)
    result(verifying_host, run_id, attempts(run_id)[0].id, **INCOMPLETE)
    assert judged(run_id, stub(verdict())) == ("awaiting_approval", "judge_met")


# ---------------------------------------------------------------- 3. the default verify prefers a ready id


def test_pick_verify_prefers_a_ready_id():
    assert agent_plan.pick_verify({"lint", "tests"}, {"tests": False, "lint": True}) == "lint"
    assert agent_plan.pick_verify({"lint", "tests"}, {"tests": False, "lint": False}) == "tests"  # None ready: as before.
    assert agent_plan.pick_verify({"lint", "tests"}, {"tests": False}) == "lint"  # Unreported counts as ready.
    assert agent_plan.pick_verify({"lint", "tests"}) == "tests" and agent_plan.pick_verify(set(), {"x": True}) is None


def test_the_default_verify_is_the_ready_one_and_preflight_says_when_none_is(verifying_host, client):  # noqa: F811
    target_facts(verifying_host, verify=["lint", "tests"],
                 verify_status=[{"id": "tests", "ready": False, "reason": "its program is not on the service PATH (uv)"},
                                {"id": "lint", "ready": True}])
    created = task("Needs tests", success_criteria=[{"id": "c1", "text": "pytest passes"}], notes="Do it.")
    pre = client.get(f"/api/v1/remotea/runs/preflight?task_id={created['id']}&execution_target_id={verifying_host.target_id}").json()
    assert pre["verify_ready"] is True and pre["problems"] == []
    run = start(verifying_host, created["id"], interview=False)
    with session_scope() as db:
        plan = coordinator.frozen(db, db.get(AgentRun, run["id"]))
        assert plan.verify["verify_id"] == "lint" and plan.verify["by"] == "default"

    # The task names the unready id: a warning, not a refusal (the run starts and records the failure).
    named = task("Names tests", verify={"target_id": verifying_host.target_id, "verify_id": "tests"})
    pre = client.get(f"/api/v1/remotea/runs/preflight?task_id={named['id']}&execution_target_id={verifying_host.target_id}").json()
    assert pre["verify_ready"] is False
    assert pre["problems"] == [{"code": "VERIFY_NOT_READY", "blocking": False,
                                "message": "This target's verify command is not ready (tests: its program is not on the service PATH (uv))."}]

    # None ready: the default stays today's choice, and the preflight lists why.
    target_facts(verifying_host, verify_status=[{"id": "tests", "ready": False, "reason": "uv missing"},
                                                {"id": "lint", "ready": False, "reason": "no cwd web"}])
    other = task("Nothing ready", success_criteria=[{"id": "c1", "text": "the test suite passes"}], notes="Do it.")
    pre = client.get(f"/api/v1/remotea/runs/preflight?task_id={other['id']}&execution_target_id={verifying_host.target_id}").json()
    assert pre["verify_ready"] is False and pre["problems"][0]["code"] == "VERIFY_NOT_READY"
    assert "tests: uv missing" in pre["problems"][0]["message"] and "lint: no cwd web" in pre["problems"][0]["message"]
    run = start(verifying_host, other["id"], interview=False)
    with session_scope() as db:
        assert coordinator.frozen(db, db.get(AgentRun, run["id"])).verify["verify_id"] == "tests"


# ---------------------------------------------------------------- 4. no Eridani tools: no waiting for submit_plan


NO_TOOLS = {"configured": False, "reason": "no MCP key file at /home/davin/.config/remotea/eridani-mcp-token"}


def test_a_spawn_without_tools_needs_you_at_once(host):  # noqa: F811
    _, run_id = interviewing(host)
    assert state(run_id)[0] == "interviewing"
    spawned(run_id, NO_TOOLS)
    assert state(run_id) == ("needs_you", "no_plan")  # Not after a whole planning turn.
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        assert "no Eridani tools" in row.history[-1]["detail"] and "no MCP key file" in row.history[-1]["detail"]
    # Freestyle picks the conversation up from there.
    with session_scope() as db:
        control.set_freestyle(db, OWNER, run_id, control.FreestyleInput(enabled=True), actor=HUMAN)
    assert state(run_id) == ("executing", "freestyle")


def test_a_spawn_with_tools_or_a_frozen_plan_keeps_going(host):  # noqa: F811
    _, run_id = interviewing(host)
    spawned(run_id, {"configured": True})
    assert state(run_id)[0] == "interviewing"
    frozen = start(host, task("Planned", notes="Do it.")["id"], interview=False)
    attached(host, frozen["id"])
    spawned(frozen["id"], NO_TOOLS)  # Execution needs no submit_plan.
    assert state(frozen["id"])[0] == "executing"


def test_a_target_without_tools_blocks_start_unless_freestyle(host, client):  # noqa: F811
    target_facts(host, mcp=NO_TOOLS)
    created = task("No tools")
    pre = client.get(f"/api/v1/remotea/runs/preflight?task_id={created['id']}&execution_target_id={host.target_id}").json()
    assert [p["code"] for p in pre["problems"]] == ["MCP_UNCONFIGURED"]
    assert pre["problems"][0]["blocking"] and pre["problems"][0]["freestyle_ok"]
    assert pre["problems"][0]["message"] == f"Agents on this target have no Eridani tools: {NO_TOOLS['reason']}."
    with pytest.raises(DomainError) as refused:
        start(host, created["id"])
    assert refused.value.code == "MCP_UNCONFIGURED"
    r = client.post("/api/v1/remotea/runs", json={"task_id": created["id"], "execution_target_id": host.target_id,
                                                  "freestyle": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["freestyle"] is True and body["plan_revision"] == 1  # No interview: nothing could be submitted.
    attached(host, body["id"])
    spawned(body["id"], NO_TOOLS)
    assert state(body["id"])[0] == "executing"
    assert turn(body["id"], 1)[0] == "executing"  # Recorded, not judged.


def test_eri_hears_a_non_blocking_problem_as_a_warning(monkeypatch):
    ready = {"needs_target": False, "need": None, "target": {"id": "t", "name": "jarvis", "repo": ""},
             "problems": [{"code": "VERIFY_NOT_READY", "blocking": False,
                           "message": "This target's verify command is not ready (tests: x)."}]}
    monkeypatch.setattr(control, "readiness", lambda *_: ready)
    pending = SimpleNamespace(tool="control.start", arguments={"task_id": "x"})
    told = control.proposal(SimpleNamespace(get=lambda *_: pending), OWNER, {"pending_id": "p", "next": ""})["next"]
    assert "can still start" in told and "Confirming fails" not in told
    ready["problems"][0]["blocking"] = True
    told = control.proposal(SimpleNamespace(get=lambda *_: pending), OWNER, {"pending_id": "p", "next": ""})["next"]
    assert "Confirming fails" in told


# ---------------------------------------------------------------- 5. freshness facts, kept apart


def test_public_run_reports_each_freshness_fact(verifying_host, client):  # noqa: F811
    run_id = planned(verifying_host)
    turn(run_id, 3)
    detail = client.get(f"/api/v1/remotea/runs/{run_id}").json()
    assert detail["host"]["last_seen_at"] and detail["verification"]["evidence"] == "pending"
    assert [c["kind"] for c in detail["pending_controls"]] == ["verify"]  # Sent, not yet taken by the host.
    result(verifying_host, run_id, attempts(run_id)[0].id, exit_code=1, workspace_complete=True, isolated=True)
    detail = client.get(f"/api/v1/remotea/runs/{run_id}").json()
    assert detail["verification"]["status"] == "failed" and detail["verification"]["evidence"] == "fresh"
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        row.turn_seq = 4  # A newer turn: the same verification no longer speaks for it.
        assert control.verification_fact(db, row)["evidence"] == "stale"
        uncertain = db.scalar(select(AgentCommand).where(AgentCommand.run_id == run_id, AgentCommand.kind == "verify"))
        uncertain.status, uncertain.created_at = "uncertain", now()  # Sent in the current state.
        db.flush()
        assert control.pending_controls(db, row)[0]["status"] == "uncertain"


def test_freshness_text_names_each_fact_for_eri():
    clock = now()
    item = {"state": "awaiting_approval", "state_since": (clock - timedelta(minutes=4)).isoformat(),
            "last_turn_at": (clock - timedelta(minutes=2)).isoformat(),
            "host": {"connected": True, "last_seen_at": (clock - timedelta(seconds=5)).isoformat()},
            "verification": {"verify_id": "tests", "status": "passed", "evidence": "fresh", "workspace_complete": True,
                             "at": (clock - timedelta(minutes=3)).isoformat(), "isolated": True,
                             "isolation_reason": None, "workspace_reasons": []},
            "pending_controls": [{"kind": "input", "mode": "steer", "status": "uncertain",
                                  "at": (clock - timedelta(seconds=40)).isoformat()}]}
    assert control.freshness_text(item, clock) == (
        "verify tests passed 3 min ago for the current turn on a complete workspace; sandboxed; host seen 5 s ago; "
        "last turn 2 min ago; in awaiting approval for 4 min; steer sent 40 s ago, delivery uncertain")
    silent = {"state": "executing", "host": {"connected": False, "last_seen_at": (clock - timedelta(minutes=4)).isoformat()}}
    assert control.freshness_text(silent, clock) == "host silent for 4 min"
