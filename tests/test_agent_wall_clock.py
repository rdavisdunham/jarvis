"""The wall clock counts the agent's working time: owner waits (an open interview, waiting_answer, needs_you) are
excluded, resuming from wall_clock grants time, and the run token lives as long as the waits it sat through."""

import importlib.util
from datetime import timedelta
from pathlib import Path

import pytest
from jarvis import agent_control as control
from jarvis import agent_plan
from jarvis import run_coordinator as coordinator
from jarvis.db import session_scope
from jarvis.domain import DomainError
from jarvis.models import now
from jarvis.session_models import AgentRun
from sqlalchemy import text
from test_agent_plan import HUMAN, PLAN, call, inputs, interviewing, token
from test_agent_runs import OWNER, quiet, state  # noqa: F401 - fixture
from test_agent_runs import host as host  # noqa: F401 - pytest fixture

QUESTIONS = [{"id": "scope", "question": "Which widgets?", "options": [{"label": "All"}, {"label": "Only charts"}]}]
ANSWERS = agent_plan.InterviewAnswers(answers={"scope": "All"})


def asked(host):  # noqa: F811
    _, run_id = interviewing(host)
    interview = call(agent_plan.task_interview, agent_plan.InterviewInput, run_id, questions=QUESTIONS)
    return run_id, interview["interview_id"]


def backdate(run_id, started=None, waiting=None, **values):
    """Move the run's start (and its wait in progress) into the past: hours ago."""
    with session_scope() as db:
        row = db.get(AgentRun, run_id, with_for_update=True)
        if started is not None:
            row.started_at = now() - timedelta(hours=started)
        if waiting is not None:
            row.owner_wait_since = now() - timedelta(hours=waiting)
        for key, value in values.items():
            setattr(row, key, value)


def sweep(hours=0):
    with session_scope() as db:
        coordinator.sweep(db, now() + timedelta(hours=hours))


def to(run_id, target, reason=""):
    with session_scope() as db:
        coordinator.transition(db, db.get(AgentRun, run_id, with_for_update=True), target, actor=HUMAN, reason=reason)


def answer(interview_id):
    with session_scope() as db:
        return agent_plan.answer_interview(db, OWNER, interview_id, ANSWERS, actor=HUMAN)


def fields(run_id, *names):
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        return tuple(getattr(row, n) for n in names)


def test_an_open_interview_is_not_working_time(host):  # noqa: F811
    run_id, interview_id = asked(host)
    assert fields(run_id, "owner_wait_since")[0] is not None
    backdate(run_id, started=10, waiting=9)  # Asked an hour in, answered nine hours later.
    sweep()
    assert state(run_id) == ("interviewing", "started")  # Waiting on the owner never trips the wall clock.
    answer(interview_id)
    assert state(run_id) == ("planning", "interview_answered")
    since, waited = fields(run_id, "owner_wait_since", "owner_wait_total_s")
    assert since is None and 9 * 3600 <= waited < 9 * 3600 + 60
    with session_scope() as db:
        deadline = coordinator.wall_clock_deadline(db.get(AgentRun, run_id))
    assert timedelta(hours=4, minutes=59) < deadline - now() <= timedelta(hours=5)  # 6 h of work, one used.
    sweep()
    assert state(run_id)[0] == "planning"
    sweep(hours=5.1)  # Real working time past the limit still trips it.
    assert state(run_id) == ("needs_you", "wall_clock")


def test_needs_you_and_waiting_answer_are_not_working_time(host):  # noqa: F811
    _, run_id = interviewing(host)
    backdate(run_id, started=12)
    to(run_id, "needs_you", "no_plan")
    backdate(run_id, waiting=10)
    to(run_id, "interviewing", "queue")
    assert fields(run_id, "owner_wait_since")[0] is None
    sweep(hours=3.9)
    assert state(run_id)[0] == "interviewing"  # 2 h worked of 12 h elapsed: 4 h left.
    to(run_id, "waiting_answer", "question")
    backdate(run_id, started=22, waiting=10)  # 6 h past the deadline if the question counted.
    sweep()
    assert state(run_id)[0] == "waiting_answer"  # A harness question in progress never trips it.
    to(run_id, "interviewing", "answered")
    assert 20 * 3600 <= fields(run_id, "owner_wait_total_s")[0] < 20 * 3600 + 60
    sweep(hours=3.9)
    assert state(run_id)[0] == "interviewing"
    sweep(hours=4.1)
    assert state(run_id) == ("needs_you", "wall_clock")


def test_working_time_over_the_limit_trips_the_wall_clock(host):  # noqa: F811
    _, run_id = interviewing(host)
    backdate(run_id, started=7)
    sweep()
    assert state(run_id) == ("needs_you", "wall_clock")
    assert fields(run_id, "owner_wait_since")[0] is not None  # Parked: now the owner's move.


def test_resuming_from_wall_clock_grants_time_and_does_not_bounce(host):  # noqa: F811
    """The regression on run b97694f3: a message moved the run out of needs_you(wall_clock) and the next sweep put
    it straight back."""
    _, run_id = interviewing(host)
    backdate(run_id, started=6.01)
    sweep()
    assert state(run_id) == ("needs_you", "wall_clock")
    expires = fields(run_id, "run_token_expires_at")[0]
    with session_scope() as db:
        control.send_message(db, OWNER, run_id, control.MessageInput(text="Carry on."), actor=HUMAN)
    assert state(run_id) == ("interviewing", "queue")
    limit, history, slid = fields(run_id, "wall_clock_limit", "history", "run_token_expires_at")
    grant = [h for h in history if h.get("event") == "wall_clock_extended"]
    assert len(grant) == 1 and 7200 <= grant[0]["seconds"] < 7300 and limit == 6 * 3600 + grant[0]["seconds"]
    assert slid - expires >= timedelta(seconds=grant[0]["seconds"])
    sweep()
    assert state(run_id) == ("interviewing", "queue")
    sweep(hours=1.9)
    assert state(run_id)[0] == "interviewing"
    sweep(hours=2.1)
    assert state(run_id) == ("needs_you", "wall_clock")


def test_the_interview_from_b97694f3_survives_a_message_and_takes_the_answers(host):  # noqa: F811
    """Parked on wall_clock with the interview open (the old sweep counted the wait): a message keeps the run
    interviewing, the form stays answerable, and the answers resume planning with the wait excluded."""
    run_id, interview_id = asked(host)
    backdate(run_id, started=45, waiting=44)
    to(run_id, "needs_you", "wall_clock")
    with session_scope() as db:
        control.send_message(db, OWNER, run_id, control.MessageInput(text="Here I am."), actor=HUMAN)
    sweep()
    assert state(run_id)[0] == "interviewing" and fields(run_id, "owner_wait_since")[0] is not None
    answer(interview_id)
    assert state(run_id) == ("planning", "interview_answered")
    sweep(hours=0.5)
    assert state(run_id)[0] == "planning"
    assert call(agent_plan.submit_plan, agent_plan.SubmitPlanInput, run_id, **PLAN)["plan_revision"] == 1


def test_answering_the_interview_from_needs_you_wall_clock_resumes_planning(host):  # noqa: F811
    run_id, interview_id = asked(host)
    to(run_id, "needs_you", "wall_clock")
    answered = answer(interview_id)
    assert answered["status"] == "answered" and state(run_id) == ("planning", "interview_answered")
    assert "All" in inputs(run_id)[-1] and "submit_plan" in inputs(run_id)[-1]
    history = fields(run_id, "history")[0]
    assert any(h.get("event") == "wall_clock_extended" for h in history)


def test_answering_from_other_needs_you_reasons_is_refused(host):  # noqa: F811
    run_id, interview_id = asked(host)
    to(run_id, "needs_you", "exited")
    with pytest.raises(DomainError) as refused:
        answer(interview_id)
    assert refused.value.code == "STALE_GENERATION"


def test_the_run_token_lives_through_the_wait_and_slides_when_it_ends(host):  # noqa: F811
    run_id, interview_id = asked(host)
    run_token = token(run_id)
    expired = now() - timedelta(hours=1)
    backdate(run_id, started=40, waiting=39, run_token_expires_at=expired)
    with session_scope() as db:
        assert coordinator.run_for_token(db, OWNER, run_token) is not None  # The wait in progress counts already.
    answer(interview_id)
    slid = fields(run_id, "run_token_expires_at")[0]
    assert timedelta(hours=38, minutes=59) < slid - expired <= timedelta(hours=39, minutes=1)  # From the stored expiry.
    with session_scope() as db:
        assert coordinator.run_for_token(db, OWNER, run_token).id == run_id
    backdate(run_id, run_token_expires_at=now() - timedelta(seconds=1))
    with session_scope() as db:
        assert coordinator.run_for_token(db, OWNER, run_token) is None  # Not waiting: the stored expiry rules.


def migration():
    path = Path(__file__).resolve().parents[1] / "migrations/versions/0043_run_owner_wait.py"
    spec = importlib.util.spec_from_file_location("run_owner_wait", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_migration_backfills_waits_in_progress(host):  # noqa: F811
    open_run, interview_id = asked(host)
    _, parked = interviewing(host, title="Parked")
    to(parked, "needs_you", "exited")
    _, working = interviewing(host, title="Working")
    with session_scope() as db:
        for run_id in (open_run, parked, working):
            db.get(AgentRun, run_id).owner_wait_since = None  # As before the migration.
    with session_scope() as db:
        for statement in migration().BACKFILL:
            db.execute(text(statement))
    with session_scope() as db:
        from jarvis.session_models import AgentInterview

        assert db.get(AgentRun, open_run).owner_wait_since == db.get(AgentInterview, interview_id).created_at
        assert db.get(AgentRun, parked).owner_wait_since == db.get(AgentRun, parked).state_since
        assert db.get(AgentRun, working).owner_wait_since is None
