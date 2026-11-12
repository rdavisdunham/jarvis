"""Phase 1 end to end: a real Eridani, a real remotea server and a scripted agent that plans by itself.

Two flows no person starts or steers beyond what the route asks for (docs/END_TO_END_UPGRADE_PLAN.md, phase 1):

- Auto: the readiness pass routes a labelled task, auto-start spawns the run, the agent's plan goes ahead without
  an approval, the work is verified and judged.
- Direction check: the agent sends a brief with its plan, the owner answers once, the revised plan starts.

This process plays the worker (the readiness job, the auto-start sweep and the judge, each with a scripted model),
as test_protocol_e2e.py does for the judge. Not run in CI: it needs a built remotea checkout (REMOTEA_CHECKOUT).
"""

from datetime import timedelta

import pytest
from jarvis import agent_judge as judge
from jarvis import auto_start, readiness
from jarvis import run_coordinator as coordinator
from jarvis.db import session_scope
from jarvis.domain import execute
from jarvis.models import Job, Task, now
from jarvis.session_models import AgentAttempt, AgentRun
from protocol_harness import LINK_PATH, LinkProxy, RemoteaHost, missing, remotea_checkout, tail, until
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from test_protocol_e2e import OWNER, STEP, run_state
from test_protocol_e2e import stack as stack  # noqa: F401

REASON = missing()
pytestmark = [pytest.mark.e2e_protocol, pytest.mark.skipif(REASON is not None, reason=REASON or "")]


def boot(stack, **script):
    """Eridani, the link and a remotea host whose agent plans by itself; returns (eri, proxy, host, host_id)."""
    eri = stack.eridani.start()
    enabled = eri.post("/api/v1/remotea/enable")
    stack.proxy = proxy = LinkProxy(f"ws://127.0.0.1:{eri.port}{LINK_PATH}").start()
    stack.host = host = RemoteaHost(remotea_checkout(), stack.tmp, proxy.url, enabled["token"])
    host.configure_mcp(enabled["setup_code"], eri.base)
    host.script(**script)
    host.start()
    hosts = until(lambda: [h for h in eri.get("/api/v1/remotea/status")["hosts"] if h["connected"]], 45,
                  "the host to show as connected")
    return eri, proxy, host, hosts[0]["id"]


def new_task(title, **changes):
    with session_scope() as db:
        made = execute(db, OWNER, f"phase1-create-{title}", "task.create", {"title": title})["data"]
        if changes:
            made = execute(db, OWNER, f"phase1-update-{title}", "task.update",
                           {"task_id": made["id"], "expected_revision": made["revision"], **changes})["data"]
    return made["id"]


def save_target(eri, host, host_id, task_id):
    return eri.post("/api/v1/remotea/targets", {"host_id": host_id, "host_target": host.TARGET, "repo": str(host.repo),
                                               "worktree_root": str(host.worktrees), "task_id": task_id})


def task_row(task_id):
    with session_scope() as db:
        row = db.get(Task, task_id)
        db.expunge(row)
        return row


def history(run_id, event):
    with session_scope() as db:
        return [h for h in db.get(AgentRun, run_id).history if h.get("event") == event]


def judge_met(run_id):
    """The queued judge job, run here as the worker would, with a model that finds every criterion of the plan met.
    Returns the evidence the judge was given."""
    def queued():
        with session_scope() as db:
            ids = set(db.scalars(select(AgentAttempt.id).where(AgentAttempt.run_id == run_id, AgentAttempt.kind == "judge",
                                                               AgentAttempt.status == "queued")))
            jobs = db.scalars(select(Job).where(Job.kind == "agent_judge", Job.status == "queued")).all()
            return next((j.id for j in jobs if (j.payload or {}).get("attempt") in ids), None)
    job_id = until(queued, STEP, "a queued judge job")
    seen = []

    def model(evidence):
        seen.append(evidence)
        return judge.Verdict(verdict="met", gap="", summary="scripted judge", steps=[], criteria=[
            judge.CriterionVerdict(id=c["id"], status="met", evidence="the verify attempt and the changes")
            for c in evidence["plan"]["criteria"]])
    judge.process(job_id, model=model)
    # A met verdict is held while the host scans the worktree for the pull request (the completion check); the
    # worker then finishes the same attempt. Play that part until the run leaves judging.
    def settled():
        if run_state(run_id).state != "judging":
            return True
        if not any(h.get("event") == "outcome" and h.get("kind") == "worktrees.scan" for h in history(run_id, "outcome")):
            return False  # The host's scan is still on its way; the sweep comes after it, as it does in production.
        try:
            with session_scope() as db:  # The supervisor's sweep, a minute later: the wait for a pull request is over.
                coordinator.sweep(db, now() + judge.EVIDENCE_WAIT + timedelta(seconds=5))
            again = queued()
            if again:
                judge.process(again, model=model)
        except OperationalError:
            pass  # A lock met the API process; the next round tries again, as the worker's retry would.
        return False
    until(settled, STEP, "the held verdict to be committed", interval=0.3)
    return seen[0]


def explain(stack, eri):
    with session_scope() as db:
        for row in db.scalars(select(AgentRun)):
            print("run", row.state, row.state_reason, "turn", row.turn_seq,
                  [(h.get("event") or h.get("to"), h.get("reason") or h.get("kind")) for h in row.history[-8:]])
    print(f"--- stub\n{stack.host.stub_entries()[-12:]}\n--- remotea-server log\n{tail(stack.host.log)}\n"
          f"--- eridani log\n{tail(eri.log)}")


def test_a_labelled_task_runs_from_creation_to_a_judged_result_with_no_owner_action(stack, monkeypatch):
    monkeypatch.setattr(readiness, "enabled", lambda: True)
    eri, proxy, host, host_id = boot(stack, criterion="The greeting says hello")
    try:
        # The owner's one-time setup: a repository, and auto-start allowed in it.
        setup = new_task("Set up the repository")
        target = save_target(eri, host, host_id, setup)
        target_id = target.get("id") or task_row(setup).execution_target_id
        assert target_id
        with session_scope() as db:
            auto_start.save(db, OWNER, auto_start.Settings(enabled=True, targets=[target_id]))

        # A task arrives with the label. Nothing below is an owner action.
        task_id = new_task("Add a greeting", tags=["ai-assignable"], execution_target_id=target_id)
        with session_scope() as db:
            job = db.scalar(select(Job.id).where(Job.kind == readiness.KIND, Job.payload["task_id"].astext == task_id))
        readiness.process(job, model=lambda body: readiness.Answer(
            agent_runnable=True, migration=False, production_change=False, client_visible=False, spends_money=False,
            criteria=["A greeting is added"], reason="A small code change."))
        ready = task_row(task_id)
        assert ready.run_route == "auto" and ready.readiness["state"] == "ready", ready.readiness
        assert ready.verify == {"target_id": target_id, "verify_id": host.VERIFY_ID, "by": "default"}

        with session_scope() as db:
            auto_start.sweep(db)
        with session_scope() as db:
            run_id = db.scalar(select(AgentRun.id).where(AgentRun.task_id == task_id))
        assert run_id and task_row(task_id).readiness["auto_start"]["run_id"] == run_id

        # The agent plans by itself; the plan needs no approval; the work starts.
        proxy.wait("eridani", "spawn", lambda f: f["run_id"] == run_id, timeout=STEP)
        until(lambda: host.stub_entries("plan"), STEP, "the agent to submit its plan")
        assert host.stub_entries("plan")[0]["plan"] == 1, host.stub_entries("plan")
        until(lambda: run_state(run_id).state in ("executing", "verifying", "judging"), STEP, "the plan to start by itself")
        assert [h["actor"] for h in history(run_id, "plan_approved")] == [{"kind": "auto_start"}]
        with session_scope() as db:
            assert [c["text"] for c in db.get(Task, task_id).success_criteria] == ["The greeting says hello"]

        # Verified on the host, then judged with the run's changes in evidence.
        until(lambda: run_state(run_id).state == "judging", STEP, "the work to be verified and reach the judge")
        evidence = judge_met(run_id)
        assert evidence["verify"]["status"] == "passed", evidence["verify"]
        # This scratch repository has no pull request, and the task's policy is "a pull request is open": the met
        # verdict stops for the owner instead of completing. Nothing was approved or merged without them.
        after = run_state(run_id)
        assert (after.state, after.reason) == ("needs_you", "pr_unknown"), after
        assert [f["path"] for f in evidence["changes"]["files"]] == ["stub-change-2.txt"], evidence["changes"]
        assert task_row(task_id).status != "completed"
        public = eri.get(f"/api/v1/remotea/runs/{run_id}")
        assert public["auto_started"] is True
        assert len(host.stub_entries("start")) == 1
    except BaseException:
        explain(stack, eri)
        raise


def test_the_owner_answers_the_brief_once_and_the_revised_plan_starts(stack):
    eri, proxy, host, host_id = boot(stack)
    try:
        task_id = new_task("Rename the helper", run_route="direction")
        save_target(eri, host, host_id, task_id)
        run_id = eri.post("/api/v1/remotea/runs", {"task_id": task_id, "harness": "claude", "model": "sonnet"})["id"]

        # The agent's first plan comes with a brief, and the run stops for it.
        until(lambda: run_state(run_id).state == "needs_you", STEP, "the run to stop for the direction check")
        current = run_state(run_id)
        assert current.reason == "plan_approval", current
        shown = eri.get(f"/api/v1/remotea/runs/{run_id}")["direction"]
        assert shown["open"] is True and shown["brief"]["questions"] == ["Anything to avoid?"], shown
        assert host.stub_entries("plan")[0]["brief"] is True

        # One answer. The agent revises, and this time nothing stops it.
        eri.post(f"/api/v1/remotea/runs/{run_id}/plan-review",
                 {"action": "revise", "note": "Avoid the public API.", "expected_revision": current.revision})
        until(lambda: len(host.stub_entries("plan")) == 2, STEP, "the agent to submit its revised plan")
        assert "Avoid the public API." in host.stub_entries("prompt")[-1]["prompt"] or \
            any("Avoid the public API." in e["prompt"] for e in host.stub_entries("prompt"))
        until(lambda: run_state(run_id).state in ("executing", "verifying", "judging", "needs_you"), STEP,
              "the revised plan to start")
        after = run_state(run_id)
        assert after.state != "needs_you" or after.reason != "plan_approval", after
        assert [h["actor"] for h in history(run_id, "plan_approved")] == [{"kind": "direction_check"}]
        shown = eri.get(f"/api/v1/remotea/runs/{run_id}")["direction"]
        assert shown["open"] is False and shown["answer"] == "Avoid the public API."
    except BaseException:
        explain(stack, eri)
        raise


