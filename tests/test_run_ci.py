"""GitHub CI as a run's final verification (run_ci.py): the host's ci.status answer, awaiting_ci, a green head
completing the run, a red head sending the agent the failed jobs, the 90-minute wait and the read model."""

from datetime import timedelta

import pytest
from jarvis import agent_control as control
from jarvis import run_attention, run_ci
from jarvis import run_coordinator as coordinator
from jarvis.db import session_scope
from jarvis.models import Notification, Task, now
from jarvis.session_models import AgentCommand, AgentHost, AgentRun, AgentSession
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
    assert run_ci.parse(answer(checks=checks("pass", "pending")))["state"] == "pending"
    fast = run_ci.parse(answer(checks=checks("pass", "pending", "fail")))
    assert fast["state"] == "failed" and fast["running"]  # Fail fast: one failed check is enough.
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


def test_every_plan_under_code_ready_or_pr_open_consults_ci(monkeypatch):
    from types import SimpleNamespace as Plan

    monkeypatch.setattr(run_ci, "quick_ids", lambda: {"quick"})
    assert run_ci.required(Plan(verify={"verify_id": "quick"}, completion_policy="code_ready"))
    assert run_ci.required(Plan(verify={"verify_id": "quick"}, completion_policy="pr_open"))
    assert not run_ci.required(Plan(verify={"verify_id": "quick"}, completion_policy="pr_merged"))
    # CI is preferred wherever it exists (2026-11-07): a full entry waits for it too (or CI stands in for it).
    assert run_ci.required(Plan(verify={"verify_id": "full"}, completion_policy="code_ready"))
    assert run_ci.quick(Plan(verify={"verify_id": "quick"})) and run_ci.quick(Plan(verify=None))
    assert not run_ci.quick(Plan(verify={"verify_id": "full"}))
    # The ruleset: a plan with no verify entry relies on CI (and needs you when the pull request has none).
    assert run_ci.required(Plan(verify=None, completion_policy="code_ready"))
    assert run_ci.required(Plan(verify={}, completion_policy="pr_open"))
    assert not run_ci.required(Plan(verify=None, completion_policy="pr_merged")) and not run_ci.required(None)


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


def resumed(run_id, text="Next part"):
    with session_scope() as db:
        control.send_message(db, OWNER, run_id, control.MessageInput(text=text), actor=HUMAN)
    return state(run_id)


def test_a_full_entry_waits_for_ci_once_then_ci_stands_in_for_it(ci_host, monkeypatch):
    monkeypatch.setattr(run_ci, "quick_ids", lambda: {"quick"})  # `tests` is now a full local entry.
    run_id = planned(ci_host)
    # Nothing known about the repository's CI yet: the full check runs, then the run waits for CI once.
    assert met(ci_host, run_id) == ("awaiting_ci", "ci_pending")
    assert report(run_id) == ("awaiting_approval", "judge_met")
    with session_scope() as db:
        assert run_ci.target_ci(db, db.get(AgentRun, run_id)) is True
    # The repository has CI: the next turn skips the full check and waits for CI on the pull request instead.
    assert resumed(run_id)[0] == "executing"
    turn(run_id, 4)
    assert state(run_id) == ("awaiting_ci", "ci_verify")
    assert len(attempts(run_id)) == 1  # No local check ran for turn 4.
    assert report(run_id) == ("judging", "ci_passed")  # Green CI is this turn's verification: the judge reads it.
    assert judged(run_id, stub(verdict())) == ("awaiting_approval", "judge_met")
    assert "verify_by_ci" in attempts(run_id, "judge")[-1].result["rules"]


def test_red_ci_standing_in_sends_the_gap_and_no_ci_after_all_runs_the_local_check(ci_host, monkeypatch):
    monkeypatch.setattr(run_ci, "quick_ids", lambda: {"quick"})
    run_id = planned(ci_host)
    met(ci_host, run_id)
    report(run_id)
    resumed(run_id)
    turn(run_id, 4)
    assert report(run_id, checks=checks("pass", "fail")) == ("executing", "ci_failed")
    turn(run_id, 5)
    assert state(run_id) == ("awaiting_ci", "ci_verify")
    assert report(run_id, checks=[]) == ("awaiting_ci", "ci_verify")
    assert report(run_id, checks=[]) == ("verifying", "ci_absent_local")
    assert [a.turn_seq for a in attempts(run_id)] == [3, 5]  # The full check runs for turn 5 after all.


def test_a_full_entry_on_a_repository_without_ci_finishes_on_its_own_pass(ci_host, monkeypatch):
    monkeypatch.setattr(run_ci, "quick_ids", lambda: {"quick"})
    run_id = planned(ci_host)
    met(ci_host, run_id)
    report(run_id, checks=[])
    assert report(run_id, checks=[]) == ("awaiting_approval", "judge_met")  # No CI on the pull request.
    with session_scope() as db:
        assert run_ci.target_ci(db, db.get(AgentRun, run_id)) is False
    polls = len(ci_commands(run_id))
    resumed(run_id)
    assert met(ci_host, run_id, seq=4) == ("awaiting_approval", "judge_met")  # Known: no CI, no wait.
    assert len(ci_commands(run_id)) == polls


def test_a_full_entry_pass_stands_when_the_host_cannot_read_ci(verifying_host, monkeypatch):  # noqa: F811
    monkeypatch.setattr(run_ci, "quick_ids", lambda: {"quick"})
    run_id = planned(verifying_host)
    assert met(verifying_host, run_id) == ("awaiting_approval", "judge_met")
    assert ci_commands(run_id) == []


def test_ci_status_and_merge_commands_reach_a_host_that_takes_them(ci_host):
    from jarvis import agent_link

    run_id = planned(ci_host)
    met(ci_host, run_id)
    with session_scope() as db:
        row = db.get(AgentHost, ci_host.id)
        sent = [f for f in agent_link.drain(db, row, row.conn_epoch) if f["t"] == "ci.status"]
        assert len(sent) == 1 and sent[0]["target"] == "jarvis" and sent[0]["log_lines"] == 40
        # A settled run's CI is still read (the drawer's merge panel): not cancelled as stale.
        run = db.get(AgentRun, run_id)
        command = run_ci.request(db, run)
        run.state = "completed"
        assert coordinator.stale(db, command, run) is None


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


def test_fail_fast_sends_the_gap_while_other_checks_still_run(ci_host):
    run_id = planned(ci_host)
    met(ci_host, run_id)
    failing = {"checks": checks("fail", "pending"), "failed_logs": [{"name": "Backend and migrations", "lines": ["E boom"]}]}
    assert report(run_id, **failing) == ("executing", "ci_failed")
    assert "E boom" in inputs(run_id)[-1]


def test_fail_fast_waits_a_little_for_the_failed_jobs_log(ci_host):
    run_id = planned(ci_host)
    met(ci_host, run_id)
    no_log = {"checks": checks("fail", "pending")}
    assert report(run_id, **no_log) == ("awaiting_ci", "ci_pending")
    assert report(run_id, **no_log) == ("awaiting_ci", "ci_pending")
    assert report(run_id, **no_log) == ("executing", "ci_failed")  # LOG_WAITS spent: the gap goes without a log.
    assert "Backend and migrations" in inputs(run_id)[-1]


def parked_infra(host, run_id):  # noqa: F811
    """A turn whose quick check could not run on the host (a missing program): needs_you(verify_infra)."""
    turn(run_id, 3)
    result(host, run_id, attempts(run_id)[-1].id, exit_code=127, tail="bash: uv: command not found")
    return judged(run_id, stub(verdict()))


def poll(clock_minutes):
    with session_scope() as db:
        coordinator.sweep(db, now() + timedelta(minutes=clock_minutes))


def test_green_ci_unparks_a_run_whose_local_check_could_not_run(ci_host):
    run_id = planned(ci_host)
    assert parked_infra(ci_host, run_id) == ("needs_you", "verify_infra")
    poll(1)
    assert len(ci_commands(run_id)) == 1
    assert report(run_id, checks=checks("pass", "pending")) == ("needs_you", "verify_infra")  # Only green moves it.
    assert report(run_id) == ("judging", "ci_passed")
    assert judged(run_id, stub(verdict())) == ("awaiting_approval", "judge_met")
    judge = attempts(run_id, "judge")[-1]
    assert "verify_by_ci" in judge.result["rules"] and "verify_infra" not in judge.result["rules"]


def test_green_ci_unparks_a_timed_out_or_unavailable_wait(ci_host):
    run_id = planned(ci_host)
    met(ci_host, run_id)
    poll(91)
    assert state(run_id) == ("needs_you", "ci_timeout")
    poll(94)
    assert report(run_id) == ("awaiting_approval", "judge_met")


def test_a_host_upgraded_to_ci_status_unparks_ci_unavailable(verifying_host, monkeypatch):  # noqa: F811
    monkeypatch.setattr(run_ci, "quick_ids", lambda: {"tests"})
    run_id = planned(verifying_host)
    assert met(verifying_host, run_id) == ("needs_you", "ci_unavailable")
    poll(3)
    assert ci_commands(run_id) == []  # No cap, no question.
    with session_scope() as db:
        row = db.get(AgentHost, verifying_host.id)
        row.caps = {**row.caps, "caps": [*row.caps["caps"], "ci.status"]}
    poll(3)
    assert report(run_id) == ("awaiting_approval", "judge_met")


def test_parked_runs_are_polled_less_often_and_not_after_a_day(ci_host):
    run_id = planned(ci_host)
    parked_infra(ci_host, run_id)
    poll(1)
    report(run_id, checks=checks("pass", "pending"))  # Answers the question: nothing in flight.

    def due(asked_minutes, at_minutes):
        with session_scope() as db:
            run = db.get(AgentRun, run_id)
            run_ci.save(run, requested_at=(now() + timedelta(minutes=asked_minutes)).isoformat())
            return run_ci.parked_poll_due(db, run, now() + timedelta(minutes=at_minutes))

    assert due(0, 3) and not due(2, 3)  # Every 2 min during the CI wait…
    assert not due(90, 95) and due(90, 101)  # …then every 10 min…
    assert not due(0, 25 * 60)  # …and not after a day.


# ---------------------------------------------------------------- the ruleset's other rows (Davis, 2026-11-05)


def met_without_verify(run_id, seq=3):
    """A turn on a plan with no verify entry: no local check runs, the judge finds the criteria met."""
    turn(run_id, seq)
    return judged(run_id, stub(verdict()))


def events(run_id, name):
    with session_scope() as db:
        return [h for h in db.get(AgentRun, run_id).history if h.get("event") == name]


def test_a_plan_without_a_verify_entry_relies_on_ci(ci_host):
    run_id = planned(ci_host, verify=False)
    assert met_without_verify(run_id) == ("awaiting_ci", "ci_pending")
    assert attempts(run_id) == []  # Nothing ran locally.
    assert report(run_id, checks=checks("pass", "pending")) == ("awaiting_ci", "ci_pending")
    assert report(run_id) == ("awaiting_approval", "judge_met")


def test_a_quick_pass_finishes_the_run_when_the_pull_request_has_no_ci(ci_host):
    run_id = planned(ci_host)
    assert met(ci_host, run_id) == ("awaiting_ci", "ci_pending")
    assert report(run_id, checks=[]) == ("awaiting_ci", "ci_pending")  # Once could be a slow workflow start.
    assert report(run_id, checks=[]) == ("awaiting_approval", "judge_met")
    [absent] = events(run_id, "ci_absent")
    assert "no CI checks" in absent["detail"]
    # The decision is this turn's: a later turn (a new tuple) consults CI again.
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        assert run_ci.ci_state(run)["no_ci"] == run_ci.tuple_key(run) == f"{run.generation}:{run.turn_seq}:{run.plan_revision}"


def test_no_verify_and_no_ci_needs_your_verification(ci_host):
    run_id = planned(ci_host, verify=False)
    assert met_without_verify(run_id) == ("awaiting_ci", "ci_pending")
    assert report(run_id, checks=[]) == ("awaiting_ci", "ci_pending")
    assert report(run_id, checks=[]) == ("needs_you", "verify_manual")
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        detail = run_attention.attention(db, run)
        assert detail["headline"].startswith("Needs your verification")
        assert "approve it as verified" in detail["action"]
        revision = run.state_revision
    with session_scope() as db:
        control.approve(db, OWNER, run_id, control.ApproveInput(expected_revision=revision, note="Clicked through it."), actor=HUMAN)
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        assert run.state in ("stopping", "completed")  # Settles as completed once the host confirms the stop.
        assert db.get(Task, run.task_id).status == "completed"
        [entry] = [h for h in run.history if h.get("event") == "approve_override"]
        assert entry["reason"] == "verify_manual" and entry["manual_verification"] is True
        assert run.sync_state["completion"]["policy_override"]["manual_verification"] is True


def test_no_pull_request_after_two_push_prompts_needs_your_verification_without_a_local_check(ci_host):
    run_id = planned(ci_host, verify=False)
    for seq in (3, 4):
        assert met_without_verify(run_id, seq) == ("awaiting_ci", "ci_pending")
        assert report(run_id, pr=None) == ("executing", "ci_push")
    assert met_without_verify(run_id, 5) == ("awaiting_ci", "ci_pending")
    assert report(run_id, pr=None) == ("needs_you", "verify_manual")


def test_manual_verification_keeps_pr_required_run_open_until_verified_head_is_delivered(ci_host):
    run_id = planned(ci_host, verify=False)
    for seq in (3, 4):
        assert met_without_verify(run_id, seq) == ("awaiting_ci", "ci_pending")
        assert report(run_id, pr=None) == ("executing", "ci_push")
    assert met_without_verify(run_id, 5) == ("awaiting_ci", "ci_pending")
    assert report(run_id, pr=None) == ("needs_you", "verify_manual")
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        run.completion_policy = coordinator.frozen(db, run).completion_policy = "pr_open"
        control.approve(db, OWNER, run_id, control.ApproveInput(expected_revision=run.state_revision), actor=HUMAN)
        assert (run.state, run.state_reason) == ("executing", "manual_pr_delivery")
        assert db.get(Task, run.task_id).status != "completed"
        assert coordinator.manual_verification(run)["head"] == HEAD
        assert any(h.get("kind") == "manual_pr_delivery" and h.get("event") == "sent" for h in run.history)
    # The worker's delivery turn still goes through the judge and the PR's usual CI check.
    turn(run_id, 6)
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        session = db.get(AgentSession, run.session_pk)
        session.pr_urls = [PR["url"]]
        coordinator.session_reported(db, session)
    assert judged(run_id, stub(verdict())) == ("awaiting_ci", "ci_pending")
    assert report(run_id, checks=[]) == ("awaiting_ci", "ci_pending")
    assert report(run_id, checks=[])[0] in ("stopping", "completed")
    with session_scope() as db:
        assert db.get(Task, db.get(AgentRun, run_id).task_id).status == "completed"


def test_green_ci_later_unparks_a_run_that_needed_your_verification(ci_host):
    run_id = planned(ci_host, verify=False)
    met_without_verify(run_id)
    report(run_id, checks=[])
    assert report(run_id, checks=[]) == ("needs_you", "verify_manual")
    poll(5)
    assert report(run_id) == ("awaiting_approval", "judge_met")
