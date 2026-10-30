"""Stage D: verification attempts bound to (run, generation, turn_seq, plan_revision); deadline, late and host results."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
from jarvis import agent_control as control
from jarvis import agent_link
from jarvis import run_coordinator as coordinator
from jarvis.db import session_scope
from jarvis.domain import DomainError
from jarvis.models import Job, Note, now
from jarvis.session_models import AgentAttempt, AgentCommand, AgentHost, AgentRun
from sqlalchemy import select
from test_agent_plan import HUMAN, start, turn, verifying_host  # noqa: F401 - fixture
from test_agent_runs import OWNER, attached, quiet, state, task  # noqa: F401 - fixture
from test_agent_runs import host as host  # noqa: F401 - pytest fixture


def planned(host, *, verify=True, title="Verified task"):  # noqa: F811
    with session_scope() as db:
        note = Note(owner_id=OWNER, title="PRD", content="Do it.")
        db.add(note)
        db.flush()
        note_id = note.id
    values = {"prd_note_id": note_id, "success_criteria": [{"id": "c1", "text": "tests pass", "kind": "verify",
                                                            "verify_id": "tests"}] if verify else [{"id": "c1", "text": "done"}]}
    if verify:
        values["verify"] = {"target_id": "jarvis", "verify_id": "tests"}
    result = start(host, task(title, **values)["id"])
    attached(host, result["id"])
    return result["id"]


def attempts(run_id, kind="verify"):
    with session_scope() as db:
        return [SimpleNamespace(**{c: getattr(a, c) for c in ("id", "status", "generation", "turn_seq", "plan_revision",
                                                              "verify_id", "result", "initiator")})
                for a in db.scalars(select(AgentAttempt).where(AgentAttempt.run_id == run_id, AgentAttempt.kind == kind)
                                    .order_by(AgentAttempt.created_at))]


def result(host, run_id, attempt_id, **values):  # noqa: F811
    frame = agent_link.VerifyResult.model_validate({
        "attempt_id": attempt_id, "run_id": run_id, "generation": 1, "turn_seq": 3, "plan_revision": 1,
        "verify_id": "tests", "workspace_fingerprint": "abc:def", "exit_code": 0, "tail": "ok", **values})
    with session_scope() as db:
        return coordinator.verify_result(db, host.id, frame)


def test_a_turn_binds_a_verify_attempt_and_sends_the_command(verifying_host):  # noqa: F811
    run_id = planned(verifying_host)
    assert turn(run_id, 3) == ("verifying", "turn")
    [attempt] = attempts(run_id)
    assert (attempt.generation, attempt.turn_seq, attempt.plan_revision, attempt.verify_id) == (1, 3, 1, "tests")
    with session_scope() as db:
        command = db.scalar(select(AgentCommand).where(AgentCommand.run_id == run_id, AgentCommand.kind == "verify"))
        assert {k: command.payload[k] for k in ("attempt_id", "run_id", "generation", "turn_seq", "plan_revision",
                                                 "verify_id")} == {"attempt_id": attempt.id, "run_id": run_id,
                                                                    "generation": 1, "turn_seq": 3, "plan_revision": 1,
                                                                    "verify_id": "tests"}
        assert command.digest == agent_link.digest(command.payload)
    assert result(verifying_host, run_id, "nope")["discarded"] == "unknown attempt"
    assert result(verifying_host, run_id, attempt.id, turn_seq=2)["discarded"] == "tuple mismatch"
    assert state(run_id)[0] == "verifying"
    assert result(verifying_host, run_id, attempt.id)["status"] == "passed"
    assert state(run_id)[0] == "judging"
    with session_scope() as db:
        judge = db.scalar(select(AgentAttempt).where(AgentAttempt.run_id == run_id, AgentAttempt.kind == "judge"))
        assert judge.result["verify_attempt_id"] == attempt.id and judge.turn_seq == 3
        assert db.scalar(select(Job).where(Job.kind == "agent_judge")).payload == {"attempt": judge.id}


def test_deadline_times_out_and_a_late_result_is_discarded(verifying_host):  # noqa: F811
    run_id = planned(verifying_host)
    turn(run_id, 3)
    [attempt] = attempts(run_id)
    with session_scope() as db:
        coordinator.sweep(db, now() + timedelta(minutes=10))
    assert attempts(run_id)[0].status == "running"  # 10 minutes is the host's; the grace minute is ours.
    with session_scope() as db:
        db.get(AgentHost, verifying_host.id).lease_until = now() + timedelta(days=1)
        coordinator.sweep(db, now() + timedelta(minutes=12))
    assert attempts(run_id)[0].status == "timed_out" and state(run_id)[0] == "judging"
    assert result(verifying_host, run_id, attempt.id)["discarded"] == "late"
    assert attempts(run_id)[0].status == "timed_out" and attempts(run_id)[0].result["late"]


def test_failing_stale_and_refused_results_are_non_passes(verifying_host):  # noqa: F811
    run_id = planned(verifying_host)
    turn(run_id, 3)
    assert result(verifying_host, run_id, attempts(run_id)[0].id, exit_code=1)["status"] == "failed"
    other = planned(verifying_host, title="Second")
    turn(other, 3)
    assert result(verifying_host, other, attempts(other)[0].id, stale_workspace=True)["status"] == "stale"
    third = planned(verifying_host, title="Third")
    turn(third, 3)
    with session_scope() as db:
        row = db.scalar(select(AgentCommand).where(AgentCommand.run_id == third, AgentCommand.kind == "verify"))
        coordinator.on_outcome(db, row, "rejected", "unknown verify id")
    assert attempts(third)[0].status == "refused" and state(third)[0] == "judging"
    fourth = planned(verifying_host, title="Fourth")
    turn(fourth, 3)
    with session_scope() as db:
        row = db.scalar(select(AgentCommand).where(AgentCommand.run_id == fourth, AgentCommand.kind == "verify"))
        coordinator.on_outcome(db, row, "uncertain", "host restarted")
    assert attempts(fourth)[0].status == "uncertain" and state(fourth)[0] == "judging"  # Not delivery_uncertain.


def test_a_host_without_verify_refuses_at_once(host):  # noqa: F811
    run_id = planned(host)
    turn(run_id, 1)
    assert attempts(run_id)[0].status == "refused" and state(run_id)[0] == "judging"
    with session_scope() as db:
        assert db.scalar(select(AgentCommand).where(AgentCommand.run_id == run_id, AgentCommand.kind == "verify")) is None


def test_host_attempts_are_adopted_by_tuple_or_kept_aside(verifying_host):  # noqa: F811
    run_id = planned(verifying_host)
    turn(run_id, 3)
    assert result(verifying_host, run_id, "local:other", turn_seq=2)["discarded"] == "no matching attempt"
    assert state(run_id)[0] == "verifying"
    kept = [a for a in attempts(run_id) if a.initiator == "host"]
    assert kept and kept[0].result["unmatched"]
    adopted = result(verifying_host, run_id, "local:abc")
    assert adopted["status"] == "passed" and state(run_id)[0] == "judging"
    mine = [a for a in attempts(run_id) if a.initiator == "eridani"][0]
    assert mine.result["adopted"] == "local:abc"


def test_an_older_generation_result_never_moves_the_run(verifying_host):  # noqa: F811
    run_id = planned(verifying_host)
    turn(run_id, 3)
    [attempt] = attempts(run_id)
    with session_scope() as db:
        control.send_message(db, OWNER, run_id, control.MessageInput(text="Change course", mode="steer"), actor=HUMAN)
    assert result(verifying_host, run_id, attempt.id, generation=2)["discarded"] == "tuple mismatch"
    assert result(verifying_host, run_id, attempt.id)["discarded"] == "late"  # Superseded by the message.
    assert state(run_id) == ("executing", "steer")
    with session_scope() as db:
        row = db.get(AgentRun, run_id, with_for_update=True)
        coordinator.record_turn(db, row, SimpleNamespace(generation=2, seq=4, interim=False, synthetic=False))
    assert state(run_id)[0] == "verifying" and attempts(run_id)[-1].generation == 2


def test_on_demand_verify_shares_the_machinery_and_never_moves_the_run(verifying_host):  # noqa: F811
    run_id = planned(verifying_host)
    with session_scope() as db:
        detail = control.verify_now(db, OWNER, run_id, actor=HUMAN)
    assert detail["attempts"][0]["on_demand"] and state(run_id)[0] == "executing"
    with session_scope() as db, pytest.raises(DomainError) as busy:
        control.verify_now(db, OWNER, run_id, actor=HUMAN)
    assert busy.value.code == "VERIFY_RUNNING"
    [attempt] = attempts(run_id)
    assert result(verifying_host, run_id, attempt.id, turn_seq=0)["status"] == "passed"
    assert state(run_id)[0] == "executing"
    with session_scope() as db:
        assert db.scalar(select(Job).where(Job.kind == "agent_judge")) is None
    plain = planned(verifying_host, verify=False, title="No verify")
    with session_scope() as db, pytest.raises(DomainError) as none:
        control.verify_now(db, OWNER, plain, actor=HUMAN)
    assert none.value.code == "NO_VERIFY"


def test_verify_commands_go_stale_on_reconnect_when_the_attempt_settled(verifying_host):  # noqa: F811
    run_id = planned(verifying_host)
    turn(run_id, 3)
    [attempt] = attempts(run_id)
    with session_scope() as db:
        db.get(AgentAttempt, attempt.id).status = "timed_out"
        row = db.scalar(select(AgentCommand).where(AgentCommand.run_id == run_id, AgentCommand.kind == "verify"))
        assert coordinator.stale(db, row, db.get(AgentRun, run_id)) == "the verification attempt is settled"


# ---------------------------------------------------------------- why a verification did not pass


@pytest.mark.parametrize("status, facts, readiness, category, words", [
    ("failed", {"exit_code": 1, "tail": "FAILED tests/test_x.py::test_y - assert 1 == 2\n1 failed, 9 passed"}, None,
     "tests_failed", "Fix the failing tests"),
    ("failed", {"exit_code": 127, "tail": "bash: line 1: uv: command not found"}, None, "infra_missing_binary", "`uv`"),
    ("error", {"exit_code": None, "error": "spawn pnpm ENOENT"}, None, "infra_missing_binary", "`pnpm`"),
    ("error", {"exit_code": None, "error": "spawn uv ENOENT"},
     "its program is not on the service PATH (use an absolute path or set PATH in the entry's env)",
     "infra_missing_binary", "service PATH"),
    ("failed", {"exit_code": 2, "tail": "E   ModuleNotFoundError: No module named 'pytest_asyncio'\n1 error"}, None,
     "infra_missing_dependency", "`pytest_asyncio`"),
    ("failed", {"exit_code": 1, "tail": "Error: Cannot find module 'vitest/config'"}, None,
     "infra_missing_dependency", "`config`"),
    ("failed", {"exit_code": 1, "tail": "psycopg.OperationalError: connection failed: connection to server at "
                "\"127.0.0.1\", port 5432 failed: Connection refused\n3 failed, 10 errors"}, None,
     "infra_database", "disposable test database"),
    ("failed", {"exit_code": 1, "tail": "sqlalchemy.exc.OperationalError: (psycopg.OperationalError) could not connect "
                "to server"}, None, "infra_database", "test database"),
    ("timed_out", {"exit_code": None, "tail": "....", "timed_out": True}, None, "infra_timeout", "timeout"),
    ("timed_out", {"detail": "No result within the deadline."}, None, "infra_timeout", "online"),
    ("refused", {"detail": "unknown verify id"}, None, "infra_refused", "unknown verify id"),
    ("refused", {"detail": "The remotea server does not run verifications."}, None, "infra_refused", "Update remotea"),
    ("uncertain", {"detail": "host restarted"}, None, "infra_uncertain", "could not tell"),
    ("error", {"exit_code": None, "error": "no such folder apps/web"}, None, "infra_refused", "eridani.json"),
    ("error", {"exit_code": None, "error": "something odd"}, None, "infra_uncertain", "something odd"),
])
def test_verify_failures_are_classified(status, facts, readiness, category, words):
    found, action = coordinator.classify_verify(status, facts, readiness)
    assert found == category and found in coordinator.VERIFY_CATEGORIES
    assert words in action and "\n" not in action


def test_passes_and_stale_or_canceled_checks_get_no_category():
    for status in ("passed", "stale", "canceled"):
        assert coordinator.classify_verify(status, {"exit_code": 1, "tail": "uv: command not found"}) == (None, "")
    # An HTTP "404: Not Found" in a failing test's output is not a missing program.
    assert coordinator.classify_verify("failed", {"exit_code": 1, "tail": "GET /x 404: Not Found"})[0] == "tests_failed"
    # The readiness reason only explains a check that did not run, never real test failures.
    assert coordinator.classify_verify("failed", {"exit_code": 1, "tail": "1 failed"},
                                       "its program is not on the service PATH")[0] == "tests_failed"


def test_a_settled_attempt_records_its_category_and_public_attempt_shows_it(verifying_host):  # noqa: F811
    with session_scope() as db:
        row = db.get(AgentHost, verifying_host.id)
        row.caps = {**row.caps, "targets": [{**row.caps["targets"][0], "verify_status": [
            {"id": "tests", "ready": False, "reason": "its program is not on the service PATH"}]}]}
    run_id = planned(verifying_host)
    turn(run_id, 3)
    assert result(verifying_host, run_id, attempts(run_id)[0].id, exit_code=None, error="spawn uv ENOENT")["status"] == "error"
    [attempt] = attempts(run_id)
    assert attempt.result["category"] == "infra_missing_binary" and "service PATH" in attempt.result["action"]
    with session_scope() as db:
        shown = control.public_attempt(db.get(AgentAttempt, attempt.id))
    assert shown["category"] == "infra_missing_binary" and shown["action"] == attempt.result["action"]
    other = planned(verifying_host, title="Failing")
    turn(other, 3)
    result(verifying_host, other, attempts(other)[0].id, exit_code=1, tail="2 failed")
    assert attempts(other)[0].result["category"] == "tests_failed"
