"""A real agent, end to end: Claude Code itself plans and does a small task in a scratch repository.

The other protocol suites script the agent. This one launches the installed `claude` through a real remotea
server against a real Eridani, so it finds what a script cannot: a prompt the agent misreads, a tool it cannot
call, a launch flag a new Claude Code release dropped, a plan the judge cannot read.

    labelled task -> readiness (scripted) -> auto-start -> the agent plans over MCP -> plan goes ahead by itself
    -> the agent edits and commits -> the host verifies -> the judge reads the diff -> met -> approved, completed

It spends real tokens (one short Sonnet-sized session, and one judge call when JARVIS_OPENAI_API_KEY is set), so
it only runs when asked: ERIDANI_REAL_AGENT=1. How to run it: docs/AGENT_CONTROL.md, "Testing the protocol end
to end".

The agent runs in a scratch HOME. It is given this machine's current Claude access token and nothing else: no
refresh token (so the real login cannot be rotated away), no real ~/.claude, no transcripts for the collector.
"""

import json
import os
import shutil
import time

import pytest
from jarvis import agent_judge as judge
from jarvis import auto_start, readiness, run_ci
from jarvis.config import get_settings
from jarvis.db import session_scope
from jarvis.models import Job, Task
from jarvis.session_models import AgentAttempt, AgentRun
from protocol_harness import LINK_PATH, LinkProxy, RealAgentHost, access_token, missing, remotea_checkout, tail, until
from sqlalchemy import select
from test_phase1_protocol_e2e import history, new_task, save_target, task_row
from test_protocol_e2e import OWNER, ci_errors, run_state
from test_protocol_e2e import stack as stack  # noqa: F401

PLAN = 420   # Seconds a real agent may take to read the task and send its plan.
WORK = 600   # Seconds it may take to make the change, commit, and be verified.


def reason():
    if os.environ.get("ERIDANI_REAL_AGENT") != "1":
        return "spends real tokens: set ERIDANI_REAL_AGENT=1 to run it"
    if not shutil.which("claude"):
        return "claude is not on PATH"
    return missing() or access_token()[1]


REASON = reason()
pytestmark = [pytest.mark.real_agent, pytest.mark.skipif(REASON is not None, reason=REASON or "")]


def judge_jobs(run_id):
    with session_scope() as db:
        ids = set(db.scalars(select(AgentAttempt.id).where(AgentAttempt.run_id == run_id, AgentAttempt.kind == "judge",
                                                           AgentAttempt.status == "queued")))
        jobs = db.scalars(select(Job).where(Job.kind == "agent_judge", Job.status == "queued")).all()
        return next((j.id for j in jobs if (j.payload or {}).get("attempt") in ids), None)


def verdicts(run_id):
    with session_scope() as db:
        rows = db.scalars(select(AgentAttempt).where(AgentAttempt.run_id == run_id, AgentAttempt.kind == "judge")
                          .order_by(AgentAttempt.created_at)).all()
        return [(r.status, dict(r.result or {})) for r in rows]


def explain(stack, eri, host):
    with session_scope() as db:
        for row in db.scalars(select(AgentRun)):
            print("run", row.state, row.state_reason, "turns", row.turn_count,
                  [(h.get("event") or h.get("to"), h.get("reason") or h.get("kind")) for h in row.history[-12:]])
            print("trouble", (row.sync_state or {}).get("trouble"))
    print("commits", host.commits())
    print(f"--- remotea-server log\n{tail(host.log, 80)}\n--- eridani log\n{tail(eri.log, 40)}")


def boot(stack):
    eri = stack.eridani.start()
    enabled = eri.post("/api/v1/remotea/enable")
    stack.proxy = proxy = LinkProxy(f"ws://127.0.0.1:{eri.port}{LINK_PATH}").start()
    stack.host = host = RealAgentHost(remotea_checkout(), stack.tmp, proxy.url, enabled["token"])
    host.configure_mcp(enabled["setup_code"], eri.base)
    host.start()
    hosts = until(lambda: [h for h in eri.get("/api/v1/remotea/status")["hosts"] if h["connected"]], 45,
                  "the host to show as connected")
    return eri, proxy, host, hosts[0]["id"]


def test_a_real_agent_takes_a_labelled_task_from_creation_to_a_met_verdict(stack, monkeypatch):
    monkeypatch.setattr(readiness, "enabled", lambda: True)
    real_judge = bool(get_settings().openai_api_key)
    eri, proxy, host, host_id = boot(stack)
    started = time.monotonic()
    try:
        setup = new_task("Set up the scratch repository")
        target = save_target(eri, host, host_id, setup)
        target_id = target.get("id") or task_row(setup).execution_target_id
        with session_scope() as db:
            auto_start.save(db, OWNER, auto_start.Settings(enabled=True, targets=[target_id]))

        # The task as a person would file it: a title, a sentence, the label. Nothing below is an owner action.
        task_id = new_task("Add a greet function", tags=["ai-assignable"], execution_target_id=target_id,
                           completion_policy="code_ready",
                           notes="In greeting.py, add greet(name) that returns 'Hello, <name>!' (so greet('Ada') is "
                                 "'Hello, Ada!'). Commit the change.")
        with session_scope() as db:
            job = db.scalar(select(Job.id).where(Job.kind == readiness.KIND, Job.payload["task_id"].astext == task_id))
        readiness.process(job, model=lambda body: readiness.Answer(
            agent_runnable=True, migration=False, production_change=False, client_visible=False, spends_money=False,
            criteria=["greet('Ada') returns 'Hello, Ada!'"], reason="A small code change."))
        assert task_row(task_id).run_route == "auto", task_row(task_id).readiness
        with session_scope() as db:
            auto_start.sweep(db)
        with session_scope() as db:
            run_id = db.scalar(select(AgentRun.id).where(AgentRun.task_id == task_id))
        assert run_id, task_row(task_id).readiness

        # The agent reads the task and sends a plan through Eridani's tools; nobody approves it.
        proxy.wait("eridani", "spawn", lambda f: f["run_id"] == run_id, timeout=30)
        until(lambda: history(run_id, "plan_approved") or run_state(run_id).state == "needs_you", PLAN,
              "the agent's plan to arrive and go ahead", interval=1)
        assert [h["actor"] for h in history(run_id, "plan_approved")] == [{"kind": "auto_start"}], run_state(run_id)
        with session_scope() as db:
            task = db.get(Task, task_id)
            criteria, steps = [c["text"] for c in task.success_criteria], [s["title"] for s in task.steps]
        assert criteria and steps, (criteria, steps)
        planned = time.monotonic() - started

        # It makes the change; the host verifies it; the judge reads the diff.
        seen = []
        for _ in range(3):  # A real judge may ask for one more round; three is a loop.
            until(lambda: judge_jobs(run_id) or run_state(run_id).state in ("needs_you", "completed", "stopped", "failed"),
                  WORK, "the work to be verified and reach the judge", interval=1)
            job_id = judge_jobs(run_id)
            if not job_id:
                break

            def scripted(evidence):
                seen.append(evidence)
                return judge.Verdict(verdict="met", gap="", summary="scripted judge", steps=[], criteria=[
                    judge.CriterionVerdict(id=c["id"], status="met", evidence="the verify attempt and the changes")
                    for c in evidence["plan"]["criteria"]])
            judge.process(job_id, model=None if real_judge else scripted)
            until(lambda: run_state(run_id).state != "judging", 60, "the verdict to be applied", interval=0.5)
            if run_state(run_id).state not in ("executing", "verifying", "judging"):
                break

        # Met. The scratch repository has no GitHub remote, so the wait for a pull request's checks cannot be
        # answered: after three failed reads the run stops for the owner (run_ci.unreadable), who approves.
        results = verdicts(run_id)
        assert results and results[-1][1].get("verdict") == "met", results
        for attempt in range(run_ci.ERROR_POLLS):
            until(lambda n=attempt + 1: ci_errors(run_id) >= n or run_state(run_id).state != "awaiting_ci", 60,
                  "the failed CI read to be recorded", interval=0.5)
            if run_state(run_id).state != "awaiting_ci":
                break
            with session_scope() as db:
                run_ci.request(db, db.get(AgentRun, run_id, with_for_update=True))
        until(lambda: run_state(run_id).state != "awaiting_ci", 60, "the unreadable CI to stop the wait", interval=0.5)
        waiting = run_state(run_id)
        assert (waiting.state, waiting.reason) == ("needs_you", "ci_unavailable"), waiting
        eri.post(f"/api/v1/remotea/runs/{run_id}/approve", {"expected_revision": waiting.revision})
        until(lambda: run_state(run_id).state == "completed", 60, "the run to complete", interval=0.5)

        after = run_state(run_id)
        files = sorted({f for _, changed in host.commits() for f in changed})
        print(f"\nreal agent: planned in {planned:.0f}s, finished in {time.monotonic() - started:.0f}s; state "
              f"{after.state}/{after.reason}; turns {after.turn_count}; criteria {criteria}; steps {steps}; "
              f"commits {host.commits()}; judge {'real' if real_judge else 'scripted'}: "
              f"{[(s, r.get('verdict'), r.get('summary')) for s, r in results]}")
        assert "greeting.py" in files, host.commits()
        assert task_row(task_id).status == "completed"
        if seen:
            assert seen[-1]["verify"]["status"] == "passed", seen[-1]["verify"]
            assert "greeting.py" in [f["path"] for f in seen[-1]["changes"]["files"]], seen[-1]["changes"]
    except BaseException:
        explain(stack, eri, host)
        raise


def test_a_real_agent_writes_a_direction_brief_and_takes_one_answer(stack):
    eri, proxy, host, host_id = boot(stack)
    try:
        # Deliberately open: which greeting, for whom, in what words is the owner's call.
        task_id = new_task("Make the greeting friendlier", run_route="direction", completion_policy="code_ready",
                           notes="greeting.py should have a greet(name) function that feels warmer than a plain hello.")
        save_target(eri, host, host_id, task_id)
        run_id = eri.post("/api/v1/remotea/runs", {"task_id": task_id, "harness": "claude", "model": "sonnet"})["id"]

        until(lambda: run_state(run_id).state == "needs_you", PLAN, "the run to stop for the direction check", interval=1)
        current = run_state(run_id)
        assert current.reason == "plan_approval", current
        shown = eri.get(f"/api/v1/remotea/runs/{run_id}")["direction"]
        brief = shown["brief"]
        print(f"\nreal agent brief: {json.dumps(brief, indent=1)}")
        assert shown["open"] is True and brief["reading"].strip() and brief["approach"].strip(), shown
        assert len(brief.get("questions") or []) <= 3, brief

        # One answer; the revised plan carries it and starts without a second stop.
        answer = "Return exactly 'Hey <name>, good to see you!' and change nothing else."
        eri.post(f"/api/v1/remotea/runs/{run_id}/plan-review",
                 {"action": "revise", "note": answer, "expected_revision": current.revision})
        until(lambda: history(run_id, "plan_approved") or (run_state(run_id).state == "needs_you"
                                                           and run_state(run_id).revision != current.revision + 1
                                                           and not history(run_id, "plan_approved")
                                                           and run_state(run_id).reason != "plan_approval"),
              PLAN, "the revised plan to start", interval=1)
        assert [h["actor"] for h in history(run_id, "plan_approved")] == [{"kind": "direction_check"}], run_state(run_id)
        with session_scope() as db:
            criteria = [c["text"] for c in db.get(Task, task_id).success_criteria]
        print(f"revised criteria: {criteria}")
        assert any("good to see you" in c.lower() for c in criteria), criteria
        eri.post(f"/api/v1/remotea/runs/{run_id}/stop", {"mode": "kill"})  # The brief was the point; spend no more.
    except BaseException:
        explain(stack, eri, host)
        raise


def test_a_real_agent_talks_a_vague_task_through_and_plans_when_told(stack):
    """Brainstorm (run_brainstorm.py): the agent opens a conversation, asks one thing at a time, and plans on "plan it"."""
    eri, proxy, host, host_id = boot(stack)
    try:
        task_id = new_task("Make the greeting friendlier", completion_policy="code_ready",
                           notes="greeting.py should greet people in a way that feels warmer.")
        save_target(eri, host, host_id, task_id)
        run_id = eri.post("/api/v1/remotea/runs", {"task_id": task_id, "harness": "claude", "model": "sonnet",
                                                   "brainstorm": True})["id"]

        def said():
            turns = [t for t in eri.get(f"/api/v1/remotea/runs/{run_id}")["turns"] if t.get("role") != "user" and t["text"].strip()]
            return max(turns, key=lambda t: t["seq"])["text"] if turns else ""

        def waiting(turns):
            now_ = run_state(run_id)
            shown = eri.get(f"/api/v1/remotea/runs/{run_id}")["brainstorm"]
            return (now_.state, now_.reason) == ("needs_you", "brainstorm") and shown["turns"] >= turns

        until(lambda: waiting(1), PLAN, "the agent's opening message", interval=1)
        opening = said()
        print(f"\nreal agent opens:\n{opening}\n")
        assert opening.count("?") >= 1, opening
        with session_scope() as db:
            assert not db.get(Task, task_id).success_criteria, "nothing is planned while the conversation is open"

        eri.post(f"/api/v1/remotea/runs/{run_id}/message",
                 {"text": "I want it to say exactly 'Hey <name>, good to see you!'. Nothing else changes.", "mode": "queue"})
        until(lambda: waiting(2), PLAN, "the agent's second message", interval=1)
        second = said()
        print(f"real agent replies:\n{second}\n")
        assert second and second != opening

        current = run_state(run_id)
        eri.post(f"/api/v1/remotea/runs/{run_id}/brainstorm/plan", {"expected_revision": current.revision})
        until(lambda: (run_state(run_id).state, run_state(run_id).reason) == ("needs_you", "plan_approval"), PLAN,
              "the plan to arrive for approval", interval=1)
        with session_scope() as db:
            task = db.get(Task, task_id)
            criteria = [c["text"] for c in task.success_criteria]
        print(f"plan criteria after the conversation: {criteria}")
        assert any("good to see you" in c.lower() for c in criteria), criteria
        assert not host.commits(), "no code changes before the plan is approved"
        eri.post(f"/api/v1/remotea/runs/{run_id}/stop", {"mode": "kill"})
    except BaseException:
        explain(stack, eri, host)
        raise
