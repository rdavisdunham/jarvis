"""A verification that runs long or times out: the soft-threshold notice, the classifier, the structured stop reason
on the run (attention_detail) and Approve as a recorded policy override."""

from datetime import timedelta

import pytest
from jarvis import agent_control as control
from jarvis import run_attention
from jarvis import run_coordinator as coordinator
from jarvis.db import session_scope
from jarvis.models import Notification, Task, now
from jarvis.session_models import AgentAttempt, AgentHost, AgentRun
from sqlalchemy import select
from test_agent_attempts import attempts, planned, result
from test_agent_judge import judged, stub, verdict
from test_agent_plan import HUMAN, turn, verifying_host  # noqa: F401 - fixture
from test_agent_runs import OWNER, quiet, state  # noqa: F401 - fixture
from test_agent_runs import host as host  # noqa: F401 - pytest fixture

HEADLINE_30 = "The host stopped verification after 30 min (its timeout: 30 min) at 94% of the suite, no failures."
PROGRESS = "\n".join([*(f"{'.' * 70} [{p:3d}%]" for p in range(80, 95, 2)), "...................... [ 94%]"])


def advertise(host, **entry):
    """The jarvis target's verify_status row as remotea sends it (launch-caps.ts: `timeout_s`)."""
    with session_scope() as db:
        row = db.get(AgentHost, host.id)
        row.caps = {**row.caps, "targets": [{**row.caps["targets"][0],
                                             "verify_status": [{"id": "tests", "ready": True, **entry}]}]}


def slow_notices(run_id):
    with session_scope() as db:
        return list(db.scalars(select(Notification).where(Notification.category == "agent_verify_slow",
                                                          Notification.target["run_id"].astext == run_id)))


# ---------------------------------------------------------------- the classifier


@pytest.mark.parametrize("tail", [
    'psycopg.errors.FeatureNotSupported: extension "vector" is not available\nDETAIL: Could not open extension control file',
    'sqlalchemy.exc.NotSupportedError: (psycopg.errors.FeatureNotSupported) extension "vector" is not available',
    'ERROR:  extension "pg_trgm" is not available',
])
def test_a_missing_postgres_extension_is_the_operators_database_problem(tail):
    category, action = coordinator.classify_verify("failed", {"exit_code": 1, "tail": tail + "\n3 failed, 10 errors"})
    assert category == "infra_database" and "test database" in action


def test_real_test_failures_still_go_to_the_agent():
    assert coordinator.classify_verify("failed", {"exit_code": 1, "tail": "FAILED tests/test_x.py::test_y\n1 failed"})[0] \
        == "tests_failed"


# ---------------------------------------------------------------- the ceiling and the soft threshold


def test_the_hosts_timeout_s_sets_the_ceiling(verifying_host):  # noqa: F811
    advertise(verifying_host, timeout_s=7200)
    run_id = planned(verifying_host)
    turn(run_id, 3)
    with session_scope() as db:
        attempt = db.scalar(select(AgentAttempt).where(AgentAttempt.run_id == run_id, AgentAttempt.kind == "verify"))
        assert attempt.result["timeout_s"] == 7200
        assert timedelta(seconds=7200 + 119) < attempt.deadline_at - attempt.created_at < timedelta(seconds=7200 + 121)


def test_one_slow_notice_per_attempt_and_the_run_keeps_verifying(verifying_host):  # noqa: F811
    advertise(verifying_host, timeout_s=7200)
    run_id = planned(verifying_host)
    turn(run_id, 3)
    with session_scope() as db:
        coordinator.sweep(db, now() + timedelta(minutes=29))
    assert slow_notices(run_id) == []
    for minutes in (31, 45, 90):
        with session_scope() as db:
            coordinator.sweep(db, now() + timedelta(minutes=minutes))
    [notice] = slow_notices(run_id)
    assert "running for 30 min; it will stop at 2 h" in notice.body and notice.target["view"] == "remotea"
    assert notice.dedup_key == f"agent:agent_verify_slow:{attempts(run_id)[0].id}"
    assert state(run_id)[0] == "verifying" and attempts(run_id)[0].status == "running"
    assert attempts(run_id)[0].result["slow_notice_at"]


def test_the_soft_threshold_is_configurable_per_entry_and_skipped_under_the_ceiling(verifying_host):  # noqa: F811
    advertise(verifying_host, timeout_s=3600, notify_after_s=600)
    run_id = planned(verifying_host)
    turn(run_id, 3)
    with session_scope() as db:
        coordinator.sweep(db, now() + timedelta(minutes=11))
    [notice] = slow_notices(run_id)
    assert "running for 10 min; it will stop at 1 h" in notice.body
    advertise(verifying_host, timeout_s=900)  # Stops before the default 30-minute threshold: nothing to say.
    second = planned(verifying_host, title="Short")
    turn(second, 3)
    with session_scope() as db:
        coordinator.sweep(db, now() + timedelta(minutes=14))
    assert slow_notices(second) == []


def test_a_second_attempt_gets_its_own_notice(verifying_host):  # noqa: F811
    advertise(verifying_host, timeout_s=7200)
    run_id = planned(verifying_host)
    turn(run_id, 3)
    with session_scope() as db:
        coordinator.sweep(db, now() + timedelta(minutes=31))
        run = db.get(AgentRun, run_id)
        attempt = db.get(AgentAttempt, attempts(run_id)[0].id)
        attempt.status = "superseded"
        coordinator.start_verify(db, run, coordinator.frozen(db, run), on_demand=True)
    with session_scope() as db:
        coordinator.sweep(db, now() + timedelta(minutes=31))
    assert len(slow_notices(run_id)) == 2


# ---------------------------------------------------------------- the structured stop reason


def timed_out_run(host):
    advertise(host, timeout_s=1800)
    run_id = planned(host)
    turn(run_id, 3)
    result(host, run_id, attempts(run_id)[0].id, exit_code=None, timed_out=True, duration_ms=1_800_000,
           tail=PROGRESS)
    assert judged(run_id, stub(verdict("met"))) == ("needs_you", "verify_infra")
    return run_id


def test_a_timed_out_verification_leads_with_a_headline_and_action(verifying_host):  # noqa: F811
    run_id = timed_out_run(verifying_host)
    with session_scope() as db:
        public = control.public_run(db, db.get(AgentRun, run_id))
    shown = public["attention_detail"]
    assert shown["reason"] == "verify_infra"
    assert shown["headline"] == HEADLINE_30
    assert shown["action"] == "Raise the verify timeout or verify again."
    assert shown["last_output"][-1] == "...................... [ 94%]" and len(shown["last_output"]) == 5
    assert public["attention"] == "error" and public["reason_text"] == coordinator.REASONS["verify_infra"]  # Kept.
    with session_scope() as db:
        bodies = list(db.scalars(select(Notification.body).where(Notification.category == "agent_stuck",
                                                                 Notification.target["run_id"].astext == run_id)))
    assert any(HEADLINE_30 in b and "Raise the verify timeout" in b for b in bodies), bodies


def test_the_headline_counts_failures_and_falls_back_to_the_last_line():
    progress, failures = run_attention.progress_of("....F...E. [ 40%]\n..F [ 50%]")
    assert (progress, failures) == (50, 3)
    assert run_attention.progress_of("collecting ...\n2 failed, 1 error in 3s") == (None, 2)
    attempt = AgentAttempt(created_at=now() - timedelta(minutes=12), updated_at=now())
    assert run_attention.timeout_headline(attempt, {"tail": "building web\nvite v5 compiling"}) == \
        "The host stopped verification after 12 min at “vite v5 compiling”, no failures."
    long = run_attention.tail_lines("x" * 500)
    assert len(long[0]) == run_attention.LINE_CHARS and long[0].endswith("…")


@pytest.mark.parametrize("category,tail,words", [
    ("infra_database", 'psycopg.errors.FeatureNotSupported: extension "vector" is not available', "test database"),
    ("infra_missing_binary", "bash: line 1: uv: command not found", "program is missing"),
])
def test_other_infra_categories_have_their_own_headline(verifying_host, category, tail, words):  # noqa: F811
    run_id = planned(verifying_host)
    turn(run_id, 3)
    result(verifying_host, run_id, attempts(run_id)[0].id, exit_code=1 if "vector" in tail else 127, tail=tail)
    assert judged(run_id, stub(verdict("met"))) == ("needs_you", "verify_infra")
    with session_scope() as db:
        shown = control.public_run(db, db.get(AgentRun, run_id))["attention_detail"]
    assert attempts(run_id)[0].result["category"] == category
    assert words in shown["headline"] and shown["action"].endswith("Or verify again.")
    assert shown["last_output"] == [tail]


@pytest.mark.parametrize("reason", sorted(run_attention.ATTENTION))
def test_every_stop_reason_has_a_plain_headline_and_action(reason):
    run = AgentRun(id="r", state="needs_you", state_reason=reason)
    shown = run_attention.attention(None, run, "the detail") if reason not in run_attention.VERIFY_REASONS else None
    if shown is not None:
        assert shown == {"reason": reason, "headline": run_attention.ATTENTION[reason][0],
                         "action": run_attention.ATTENTION[reason][1], "detail": "the detail", "last_output": []}
    headline, action = run_attention.ATTENTION[reason]
    assert headline.endswith(".") and action.endswith(".") and "_" not in headline + action


def test_working_runs_have_no_attention_detail():
    assert run_attention.attention(None, AgentRun(state="executing", state_reason="turn")) is None
    capped = run_attention.attention(None, AgentRun(state="capped", state_reason="turn_cap"))
    assert capped["headline"] == run_attention.ATTENTION["turn_cap"][0]


# ---------------------------------------------------------------- Approve from needs_you


def test_approve_from_needs_you_completes_and_records_the_override(verifying_host):  # noqa: F811
    run_id = timed_out_run(verifying_host)
    with session_scope() as db:
        revision = db.get(AgentRun, run_id).state_revision
        control.approve(db, OWNER, run_id, control.ApproveInput(
            expected_revision=revision, note="Suite was green at 94%; CI passed."), actor=HUMAN)
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        assert db.get(Task, run.task_id).status == "completed"
        [entry] = [h for h in run.history if h.get("event") == "approve_override"]
        assert entry["reason"] == "verify_infra" and entry["note"] == "Suite was green at 94%; CI passed."
        assert entry["verification"]["status"] == "timed_out" and entry["actor"]["kind"] == "human"
        assert run.sync_state["completion"]["policy_override"]["reason"] == "verify_infra"
        assert run.sync_state["completion"]["approved"] is True


def test_approve_from_awaiting_approval_is_no_override(verifying_host):  # noqa: F811
    run_id = planned(verifying_host)
    turn(run_id, 3)
    result(verifying_host, run_id, attempts(run_id)[0].id, exit_code=0, tail="ok")
    assert judged(run_id, stub(verdict("met"))) == ("awaiting_approval", "judge_met")
    with session_scope() as db:
        control.approve(db, OWNER, run_id, control.ApproveInput(
            expected_revision=db.get(AgentRun, run_id).state_revision), actor=HUMAN)
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        assert not [h for h in run.history if h.get("event") == "approve_override"]
        assert "policy_override" not in run.sync_state["completion"]


def test_approve_route_takes_the_note(verifying_host, client):  # noqa: F811
    run_id = timed_out_run(verifying_host)
    with session_scope() as db:
        revision = db.get(AgentRun, run_id).state_revision
    response = client.post(f"/api/v1/remotea/runs/{run_id}/approve",
                           json={"expected_revision": revision, "note": "Approved anyway"})
    assert response.status_code == 200, response.text
    with session_scope() as db:
        assert [h["note"] for h in db.get(AgentRun, run_id).history if h.get("event") == "approve_override"] == \
            ["Approved anyway"]


# ---------------------------------------------------------------- Eridani's deadline vs the host's result


def swept_run(host):
    """A host that advertises no timeout: Eridani gives up at 11 minutes while the host is still running."""
    run_id = planned(host)
    turn(run_id, 3)
    with session_scope() as db:
        db.get(AgentHost, host.id).lease_until = now() + timedelta(days=1)
        coordinator.sweep(db, now() + timedelta(minutes=12))
    assert attempts(run_id)[0].status == "timed_out" and attempts(run_id)[0].result["swept"] is True
    assert judged(run_id, stub(verdict("met"))) == ("needs_you", "verify_infra")
    return run_id


def test_the_headline_says_eridani_stopped_waiting_and_why(verifying_host):  # noqa: F811
    run_id = swept_run(verifying_host)
    with session_scope() as db:
        shown = control.public_run(db, db.get(AgentRun, run_id))["attention_detail"]
    assert shown["headline"] == ("Eridani stopped waiting for verification after 11 min (its 11-minute default: the host "
                                 "did not advertise the verify entry's timeout); the host had not reported a result.")
    assert shown["action"].startswith("Verify again.")


def test_a_late_pass_is_recorded_and_judged_again(verifying_host):  # noqa: F811
    run_id = swept_run(verifying_host)
    attempt_id = attempts(run_id)[0].id
    out = result(verifying_host, run_id, attempt_id, exit_code=0, tail="1450 passed in 1790s", duration_ms=1_790_000)
    assert out == {"attempt": attempt_id, "status": "passed", "late": True}
    [attempt] = attempts(run_id)
    assert attempt.status == "passed" and attempt.result["swept"]["deadline_at"] and "category" not in attempt.result
    assert state(run_id) == ("judging", "verify_late_result")
    assert judged(run_id, stub(verdict("met"))) == ("awaiting_approval", "judge_met")


def test_a_late_host_timeout_replaces_the_swept_outcome_without_moving_the_run(verifying_host):  # noqa: F811
    run_id = swept_run(verifying_host)
    result(verifying_host, run_id, attempts(run_id)[0].id, exit_code=None, timed_out=True, duration_ms=1_800_000,
           tail=PROGRESS)
    assert state(run_id) == ("needs_you", "verify_infra")
    with session_scope() as db:
        shown = control.public_run(db, db.get(AgentRun, run_id))["attention_detail"]
    assert shown["headline"] == "The host stopped verification after 30 min at 94% of the suite, no failures."
    # A second copy of the result is late in the ordinary way: kept on the attempt, nothing else changes.
    assert result(verifying_host, run_id, attempts(run_id)[0].id, exit_code=0)["discarded"] == "late"


def test_parallel_verifications_on_one_host_are_named_not_queued(verifying_host):  # noqa: F811
    advertise(verifying_host, timeout_s=7200)
    first, second = planned(verifying_host, title="First"), planned(verifying_host, title="Second")
    turn(first, 3)
    turn(second, 3)
    assert attempts(first)[0].status == attempts(second)[0].status == "running"  # No lane: both run at once.
    with session_scope() as db:
        assert control.public_run(db, db.get(AgentRun, first))["verification"]["alongside"] == [second]
        coordinator.sweep(db, now() + timedelta(minutes=31))
    assert "1 other verification" in slow_notices(first)[0].body


def test_run_ad1a1c02_eridani_gave_up_at_11_min_then_the_hosts_30_min_kill_is_kept(verifying_host):  # noqa: F811
    """The incident: no advertised timeout reached Eridani, its sweep settled the attempt at ~11 min, and the host's
    own kill at 1800 s arrived ~19 min later. Both stages read plainly, and the host's result is kept."""
    run_id = swept_run(verifying_host)
    with session_scope() as db:
        before = control.public_run(db, db.get(AgentRun, run_id))["attention_detail"]["headline"]
    assert before.startswith("Eridani stopped waiting for verification after 11 min")
    result(verifying_host, run_id, attempts(run_id)[0].id, exit_code=143, timed_out=True, duration_ms=1_800_341,
           tail=PROGRESS)
    [attempt] = attempts(run_id)
    assert attempt.status == "timed_out" and attempt.result["exit_code"] == 143 and attempt.result["swept"]["at"]
    with session_scope() as db:
        after = control.public_run(db, db.get(AgentRun, run_id))["attention_detail"]["headline"]
    assert after == "The host stopped verification after 30 min at 94% of the suite, no failures."
