"""GitHub CI as a run's final verification (run_ci.py): the host's ci.status answer, awaiting_ci, a green head
completing the run, a red head sending the agent the failed jobs, the 90-minute wait and the read model."""

from datetime import timedelta

import pytest
from jarvis import agent_control as control
from jarvis import run_attention, run_ci
from jarvis import run_coordinator as coordinator
from jarvis.db import session_scope
from jarvis.models import Notification, now
from jarvis.session_models import AgentCommand, AgentHost, AgentRun
from sqlalchemy import select
from test_agent_attempts import attempts, planned, result
from test_agent_judge import judged, stub, verdict
from test_agent_plan import HUMAN, inputs, turn, verifying_host  # noqa: F401 - fixture
from test_agent_runs import OWNER, quiet, state  # noqa: F401 - fixture
from test_agent_runs import host as host

HEAD = "1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b"
PR = {"number": 42, "url": "https://github.com/davis/jarvis/pull/42", "head_sha": HEAD}
RUN = "https://github.com/davis/jarvis/actions/runs/777"


def checks(*buckets):
    names = ("Backend and migrations", "Frontend and browser", "Production container")
    return [{"name": names[i], "bucket": b, "state": b.upper(), "link": f"{RUN}/job/{i}"} for i, b in enumerate(buckets)]


def answer(**values):
    return {"pr": PR, "checks": checks("pass", "pass", "pass"), "worktree_head": HEAD, **values}


# ---------------------------------------------------------------- parsing (pure)


def test_parse_folds_checks_into_one_state():
    assert run_ci.parse(answer())["state"] == "passed"
    assert run_ci.parse(answer(checks=checks("pass", "pending", "fail")))["state"] == "pending"  # Wait for terminal.
    assert run_ci.parse(answer(checks=checks("pass", "fail", "skipping")))["state"] == "failed"
    assert run_ci.parse(answer(checks=checks("pass", "cancel")))["state"] == "cancelled"
    assert run_ci.parse(answer(checks=[]))["state"] == "absent"
    assert run_ci.parse(answer(pr=None))["state"] == "no_pr"
    assert run_ci.parse(answer(worktree_head="ffffffffffffffffffffffffffffffffffffffff"))["state"] == "unpushed"
    assert run_ci.parse(None)["state"] == "no_pr" and run_ci.parse("junk")["checks"] == []


def test_parse_reads_gh_states_without_a_bucket():
    rows = [{"name": "a", "state": "SUCCESS"}, {"name": "b", "state": "SKIPPED"}, {"name": "c", "state": "TIMED_OUT"},
            {"name": "d", "state": "IN_PROGRESS"}, {"name": "e", "state": "CANCELLED"}]
    assert [c["bucket"] for c in run_ci.parse(answer(checks=rows))["checks"]] == \
        ["pass", "skipping", "fail", "pending", "cancel"]


def test_parse_keeps_the_failed_jobs_their_run_and_at_most_40_log_lines():
    log = [f"\x1b[31mline {i}\x1b[0m" for i in range(100)]
    facts = run_ci.parse(answer(checks=checks("pass", "fail"), failed_logs=[{"name": "Frontend and browser", "lines": log}]))
    assert facts["failed"] == [{"name": "Frontend and browser", "link": f"{RUN}/job/1"}]
    assert facts["run_url"] == RUN and facts["head_sha"] == HEAD and facts["pr_number"] == 42
    assert len(facts["log"]) == run_ci.LOG_LINES and facts["log"][-1] == "line 99" and "\x1b" not in "".join(facts["log"])
    assert run_ci.parse(answer(checks=[{"name": "x", "bucket": "fail", "link": "javascript:alert(1)"}]))["failed"] == \
        [{"name": "x", "link": None}]


def test_only_a_quick_entry_under_code_ready_or_pr_open_needs_ci(monkeypatch):
    from types import SimpleNamespace as Plan

    monkeypatch.setattr(run_ci, "quick_ids", lambda: {"quick"})
    assert run_ci.required(Plan(verify={"verify_id": "quick"}, completion_policy="code_ready"))
    assert run_ci.required(Plan(verify={"verify_id": "quick"}, completion_policy="pr_open"))
    assert not run_ci.required(Plan(verify={"verify_id": "quick"}, completion_policy="pr_merged"))
    assert not run_ci.required(Plan(verify={"verify_id": "full"}, completion_policy="code_ready"))
    assert not run_ci.required(Plan(verify=None, completion_policy="code_ready")) and not run_ci.required(None)


# ---------------------------------------------------------------- the loop


@pytest.fixture
def ci_host(verifying_host, monkeypatch):  # noqa: F811
    """The jarvis target's `tests` entry is the quick check, and the host answers ci.status."""
    monkeypatch.setattr(run_ci, "quick_ids", lambda: {"tests"})
    with session_scope() as db:
        row = db.get(AgentHost, verifying_host.id)
        row.caps = {**row.caps, "caps": [*row.caps["caps"], "ci.status"]}
    return verifying_host


def met(host, run_id, seq=3):
    """A turn, its quick check passing, and the judge finding the criteria met."""
    turn(run_id, seq)
    result(host, run_id, attempts(run_id)[-1].id, turn_seq=seq)
    return judged(run_id, stub(verdict()))


def ci_commands(run_id):
    with session_scope() as db:
        return list(db.scalars(select(AgentCommand).where(AgentCommand.run_id == run_id, AgentCommand.kind == "ci.status")
                               .order_by(AgentCommand.created_at)))


def report(run_id, state_="executed", detail="", **values):
    """The host's command.outcome for the newest ci.status command (a fresh poll when that one is answered)."""
    with session_scope() as db:
        row = db.scalar(select(AgentCommand).where(AgentCommand.run_id == run_id, AgentCommand.kind == "ci.status")
                        .order_by(AgentCommand.created_at.desc()).limit(1))
        if row.status not in ("queued", "sent"):
            row = run_ci.request(db, db.get(AgentRun, run_id, with_for_update=True))
        coordinator.on_outcome(db, row, state_, detail, answer(**values) if state_ == "executed" else None)
    return state(run_id)


def ci_notices(run_id):
    with session_scope() as db:
        return list(db.scalars(select(Notification).where(Notification.category == "agent_ci_failed",
                                                          Notification.target["run_id"].astext == run_id)))


def test_met_on_the_quick_check_waits_for_ci_then_green_asks_for_approval(ci_host):
    run_id = planned(ci_host)
    assert met(ci_host, run_id) == ("awaiting_ci", "ci_pending")
    [command] = ci_commands(run_id)
    assert command.payload["turn_seq"] == 3 and command.payload["log_lines"] == 40
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        assert not any(e.get("kind") == "ci.status" for e in run.history if e.get("event") == "command")
        assert coordinator.awaiting_owner(db, run)  # Not working time: excluded from the wall clock.
    assert report(run_id, checks=checks("pass", "pending")) == ("awaiting_ci", "ci_pending")
    assert report(run_id) == ("awaiting_approval", "judge_met")
    [record] = attempts(run_id, "ci")  # One row per head, updated on every poll.
    assert record.status == "passed" and record.verify_id == "github_ci" and record.result["polls"] == 2
    assert record.result["source"] == "github_ci" and record.result["head_sha"] == HEAD
    assert [c["name"] for c in record.result["checks"]] == ["Backend and migrations", "Frontend and browser",
                                                            "Production container"]
    assert ci_notices(run_id) == []  # No notice on green: the run completes (or asks for approval).


def test_green_ci_completes_the_task_under_auto_complete(ci_host):
    with session_scope() as db:
        from jarvis.models import OwnerSettings

        row = db.get(OwnerSettings, OWNER)
        row.values = {**row.values, "remotea_auto_complete": True}
    run_id = planned(ci_host)
    assert met(ci_host, run_id)[0] == "awaiting_ci"
    assert report(run_id)[0] == "stopping"


def test_red_ci_sends_the_agent_the_failed_jobs_once_per_head(ci_host):
    run_id = planned(ci_host)
    met(ci_host, run_id)
    log = [f"E   assert {i} == 0" for i in range(60)]
    failing = {"checks": checks("pass", "fail"), "failed_logs": [{"name": "Frontend and browser", "lines": log}]}
    assert report(run_id, **failing) == ("executing", "ci_failed")
    gap = inputs(run_id)[-1]
    assert "Frontend and browser" in gap and HEAD[:7] in gap and "E   assert 59 == 0" in gap and "assert 19 == 0" not in gap
    assert RUN in gap and "never merge" in gap
    [notice] = ci_notices(run_id)
    assert notice.dedup_key == f"agent:agent_ci_failed:{run_id}:{HEAD}" and "Frontend and browser" in notice.body
    # The agent's next turn did not push a fix: the same head fails again and Davin decides.
    assert met(ci_host, run_id, seq=4) == ("awaiting_ci", "ci_pending")
    assert report(run_id, **failing) == ("needs_you", "ci_failed_again")
    assert len(ci_notices(run_id)) == 1  # Dedup per head SHA.
    with session_scope() as db:
        shown = run_attention.attention(db, db.get(AgentRun, run_id))
    assert shown["headline"].startswith("GitHub CI failed again") and shown["last_output"][-1] == "E   assert 59 == 0"


def test_no_pull_request_asks_the_agent_to_push_once_per_turn(ci_host):
    run_id = planned(ci_host)
    met(ci_host, run_id)
    assert report(run_id, pr=None) == ("executing", "ci_push")
    assert "no pull request" in inputs(run_id)[-1] and "never merge" in inputs(run_id)[-1]
    assert met(ci_host, run_id, seq=4)[0] == "awaiting_ci"
    assert report(run_id, pr=None)[0] == "executing"
    assert met(ci_host, run_id, seq=5)[0] == "awaiting_ci"
    assert report(run_id, pr=None) == ("awaiting_ci", "ci_pending")  # PUSH_PROMPTS spent: wait for the deadline.


def test_polls_every_two_minutes_and_needs_you_after_the_wait(ci_host):
    run_id = planned(ci_host)
    met(ci_host, run_id)
    report(run_id, checks=checks("pass", "pending"))
    with session_scope() as db:
        coordinator.sweep(db, now() + timedelta(seconds=30))
    assert len(ci_commands(run_id)) == 1
    with session_scope() as db:
        coordinator.sweep(db, now() + timedelta(minutes=3))
    assert len(ci_commands(run_id)) == 2
    with session_scope() as db:
        coordinator.sweep(db, now() + timedelta(minutes=91))
    assert state(run_id) == ("needs_you", "ci_timeout")
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        shown = control.public_run(db, run)["attention_detail"]
        pushes = [n.body for n in db.scalars(select(Notification).where(
            Notification.category == "agent_stuck", Notification.target["run_id"].astext == run_id))]
    assert shown["headline"].startswith(f"CI on {HEAD[:7]} was still running after 1 h 31 min (Frontend and browser)")
    assert any(shown["headline"] in body for body in pushes)


def test_the_wait_is_configurable(ci_host, monkeypatch):
    run_id = planned(ci_host)
    met(ci_host, run_id)
    monkeypatch.setattr(run_ci, "wait_seconds", lambda: 600)
    with session_scope() as db:
        coordinator.sweep(db, now() + timedelta(minutes=11))
    assert state(run_id) == ("needs_you", "ci_timeout")
    assert "reported no CI status" in control_detail(run_id)


def control_detail(run_id):
    with session_scope() as db:
        return control.public_run(db, db.get(AgentRun, run_id))["attention_detail"]["headline"]


def test_a_host_that_cannot_read_ci_says_why_on_the_timeout(ci_host):
    run_id = planned(ci_host)
    met(ci_host, run_id)
    assert report(run_id, "failed", "gh: not logged in") == ("awaiting_ci", "ci_pending")
    with session_scope() as db:
        assert control.public_run(db, db.get(AgentRun, run_id))["ci"]["error"] == "gh: not logged in"
        coordinator.sweep(db, now() + timedelta(minutes=95))
    assert "could not read CI" in control_detail(run_id) and "gh: not logged in" in control_detail(run_id)


def test_a_host_without_ci_status_parks_the_run(verifying_host, monkeypatch):  # noqa: F811
    monkeypatch.setattr(run_ci, "quick_ids", lambda: {"tests"})
    run_id = planned(verifying_host)
    assert met(verifying_host, run_id) == ("needs_you", "ci_unavailable")
    assert ci_commands(run_id) == []


def test_a_full_entry_pass_still_counts_on_its_own(ci_host, monkeypatch):
    monkeypatch.setattr(run_ci, "quick_ids", lambda: {"quick"})
    run_id = planned(ci_host)
    assert met(ci_host, run_id) == ("awaiting_approval", "judge_met")
    assert ci_commands(run_id) == []


def test_a_message_resumes_the_run_and_a_stale_answer_never_moves_it(ci_host):
    run_id = planned(ci_host)
    met(ci_host, run_id)
    with session_scope() as db:
        control.send_message(db, OWNER, run_id, control.MessageInput(text="Also fix the docs"), actor=HUMAN)
    assert state(run_id)[0] == "executing"
    assert report(run_id)[0] == "executing"  # Recorded, but the run moved on.
    assert attempts(run_id, "ci")[-1].status == "passed"


def test_the_read_model_shows_the_wait_and_the_checks(ci_host):
    run_id = planned(ci_host)
    met(ci_host, run_id)
    with session_scope() as db:
        ci = control.public_run(db, db.get(AgentRun, run_id))["ci"]
    assert ci["state"] == "requested" and ci["since"] and ci["wait_until"] and ci["checks"] == []
    report(run_id, checks=checks("pass", "pending"))
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        ci = control.public_run(db, run)["ci"]
        detail = control.public_run(db, run, detail=True)
    assert ci["state"] == "pending" and ci["current"] and ci["head_sha"] == HEAD and ci["run_url"] == RUN
    assert ci["pr_url"] == PR["url"] and [c["bucket"] for c in ci["checks"]] == ["pass", "pending"]
    row = next(a for a in detail["attempts"] if a["kind"] == "ci")
    assert row["source"] == "github_ci" and row["head_sha"] == HEAD and row["status"] == "pending"
