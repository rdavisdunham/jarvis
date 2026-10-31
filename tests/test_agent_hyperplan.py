"""Hyperplan (docs/AGENT_CONTROL.md, "Hyperplan and roles"): a planner, fresh read-only reviewers per round and an
optional separate worker, all in one run (run_coordinator.py, agent_plan.py, migration 0041)."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
from jarvis import agent_control as control
from jarvis import agent_link, agent_plan
from jarvis import run_coordinator as coordinator
from jarvis.agent_actor import Actor
from jarvis.db import session_scope
from jarvis.domain import DomainError
from jarvis.models import Task, now
from jarvis.session_models import AgentCommand, AgentHost, AgentPlanReview, AgentQuestion, AgentRun
from sqlalchemy import select
from test_agent_plan import BOT, PLAN, call, inputs, start, token, turn
from test_agent_runs import OWNER, attached, commands, quiet, state, task  # noqa: F401 - fixture
from test_agent_runs import host as host  # noqa: F401 - pytest fixture
from test_agent_voice import say

HUMAN = Actor("human")
REVIEWER = {"harness": "codex", "model": "gpt-6-astra"}
WORKER = {"harness": "claude", "model": "sonnet"}


@pytest.fixture
def roles_host(host):  # noqa: F811
    """The host advertises spawn.roles (remotea #36)."""
    with session_scope() as db:
        row = db.get(AgentHost, host.id)
        row.caps = {**row.caps, "caps": [*row.caps["caps"], "spawn.roles"]}
    return host


def hyper(host, *, rounds=2, reviewer=REVIEWER, worker=None, approve_final=True, title="Hyperplan the widget"):  # noqa: F811
    """A hyperplan run in planning (the task skips the interview), its planner attached on attempt 1."""
    created = task(title)
    body = {"harness": "claude", "model": "fable", "rounds": rounds, "approve_final": approve_final, "interview": False,
            **({"reviewer": reviewer} if reviewer else {}), **({"worker": worker} if worker else {})}
    result = start(host, created["id"], **body)
    attached(host, result["id"])
    return created, result["id"]


def plan_revision(task_id):
    with session_scope() as db:
        return db.get(Task, task_id).plan_revision or 0


def submit(run_id, task_id, **extra):
    return call(agent_plan.submit_plan, agent_plan.SubmitPlanInput, run_id,
                **{**PLAN, "expected_plan_revision": plan_revision(task_id), **extra})


def reviews(run_id):
    with session_scope() as db:
        return [SimpleNamespace(**{c: getattr(r, c) for c in ("id", "round", "status", "verdict", "via", "attempt", "surf",
                                                               "harness", "model", "findings", "resolutions",
                                                               "plan_revision", "incarnation", "token_hash")})
                for r in db.scalars(select(AgentPlanReview).where(AgentPlanReview.run_id == run_id)
                                    .order_by(AgentPlanReview.round))]


def review_token(run_id):
    """The live reviewer's token, as its SessionStart context re-derives it."""
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        review = db.scalar(select(AgentPlanReview).where(AgentPlanReview.run_id == run_id,
                                                         AgentPlanReview.status.in_(("spawning", "reviewing"))))
        return coordinator.review_token(row, review)


def reviewer_up(run_id, incarnation=3):
    """The host started the newest reviewer spawn (command.outcome executed with its surface)."""
    with session_scope() as db:
        spawn = db.scalars(select(AgentCommand).where(AgentCommand.run_id == run_id, AgentCommand.kind == "spawn")
                           .order_by(AgentCommand.created_at.desc())).first()
        assert spawn.payload["role"] == "reviewer"
        surf = coordinator.surf_for(run_id, spawn.generation, spawn.payload["attempt"])
        coordinator.on_outcome(db, spawn, "executed", "", {"surf": surf, "incarnation": incarnation})
        return surf


def review(run_id, verdict="revise", findings=(), questions=()):
    with session_scope() as db:
        return agent_plan.submit_plan_review(db, BOT, agent_plan.SubmitReviewInput.model_validate({
            "run_token": review_token(run_id), "verdict": verdict, "findings": list(findings),
            "questions_for_user": list(questions), "summary": f"Review says {verdict}."}))


def reviewer_turn(host, run_id, surf, text="", seq=1, session="rev-1"):  # noqa: F811
    """The reviewer's turn frame through the link (agent_link.turn), as remotea sends it."""
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        frame = agent_link.Turn(tool="codex", surf=surf, incarnation=3, run_id=run_id, generation=row.generation,
                                provider_session_id=f"{session}-{surf[-6:]}", provider_turn_id=f"t-{surf}-{seq}", seq=seq,
                                text=text)
        return agent_link.turn(db, BOT, host.id, frame)


BLOCKING = [{"id": "F1", "severity": "blocking", "area": "db", "issue": "No migration for the new column.",
             "suggestion": "Add 0042."},
            {"id": "F2", "severity": "minor", "area": "docs", "issue": "Docs lag.", "suggestion": ""}]
RESOLVED = [{"finding_id": "F1", "decision": "accepted", "note": "Added the migration step."},
            {"finding_id": "F2", "decision": "rejected", "note": "Docs are out of scope."}]


def one_round(host, created, run_id, findings=BLOCKING, resolve=RESOLVED, verdict="revise"):  # noqa: F811
    """planning → plan_reviewing → planning: the reviewer spawns, reviews, ends its turn; the planner resubmits."""
    surf = reviewer_up(run_id)
    assert state(run_id) == ("plan_reviewing", "review")
    review(run_id, verdict, findings)
    reviewer_turn(host, run_id, surf, text="done")
    assert state(run_id) == ("planning", "review_done")
    submit(run_id, created["id"], review_resolutions=list(resolve))
    return surf


# ---------------------------------------------------------------- the loop


@pytest.mark.parametrize("rounds", [1, 2, 3])
def test_the_loop_runs_n_fresh_reviews_then_waits_for_approval(roles_host, rounds):
    created, run_id = hyper(roles_host, rounds=rounds)
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        planner_spawn = db.scalar(select(AgentCommand).where(AgentCommand.run_id == run_id, AgentCommand.kind == "spawn"))
        assert row.roles == {"planner": {"harness": "claude", "model": "fable"}, "reviewer": REVIEWER, "worker": None,
                             "rounds": rounds, "approve_final": True}
        assert row.tool == "claude" and row.model == "fable" and row.plan_round == 0
        # Same worker as the planner: the planner is writable (no mode) and keeps its session into execution.
        assert planner_spawn.payload["role"] == "planner" and "mode" not in planner_spawn.payload
        assert "A reviewer reads your plan" in planner_spawn.payload["prompt"] and "verify" not in planner_spawn.payload
        assert "A different model executes" not in planner_spawn.payload["prompt"]
    assert state(run_id) == ("planning", "started")  # A planning start: never a frozen-plan execute.
    submit(run_id, created["id"])
    surfaces = set()
    for n in range(1, rounds + 1):
        assert turn(run_id, n) == ("plan_reviewing", "review")
        spawn = commands(run_id, "spawn")[-1]
        assert spawn.payload["role"] == "reviewer" and spawn.payload["mode"] == "review" and spawn.payload["share"] is True
        assert spawn.payload["harness"] == "codex" and spawn.payload["model"] == "gpt-6-astra"
        assert spawn.payload["attempt"] == n + 1  # A fresh session (a new attempt) every round.
        assert f"review {n} of {rounds}" in spawn.payload["prompt"]
        if n > 1:
            assert "Review 1, F1 (blocking, db)" in spawn.payload["prompt"] and "-> accepted" in spawn.payload["prompt"]
        surfaces.add(one_round(roles_host, created, run_id))
        with session_scope() as db:
            assert db.get(AgentRun, run_id).plan_round == n
        assert "submit_plan" in inputs(run_id)[-1] and f"Review {n} of {rounds}" in inputs(run_id)[-1]
        stop = commands(run_id, "stop")[-1]
        assert stop.payload["surf"] in surfaces and stop.payload["mode"] == "kill" and stop.payload["incarnation"] == 3
    assert len(surfaces) == rounds
    assert turn(run_id, rounds + 1) == ("needs_you", "plan_approval")
    done = reviews(run_id)
    assert [r.round for r in done] == list(range(1, rounds + 1)) and {r.status for r in done} == {"done"}
    assert done[0].resolutions[0] == RESOLVED[0] and all(r.token_hash is None for r in done)
    with session_scope() as db:
        public = control.public_run(db, db.get(AgentRun, run_id), detail=True)
    assert public["plan_round"] == rounds and public["roles"]["reviewer"] == REVIEWER
    assert public["actor"] == {"role": "planner", "harness": "claude", "model": "fable"}
    assert public["reviews"][0]["counts"] == {"blocking": 1, "major": 0, "minor": 1}
    assert public["reviews"][0]["verdict"] == "revise" and public["reviews"][0]["via"] == "tool"
    assert public["reason_text"].startswith("The plan is final")
    with session_scope() as db:
        result = agent_plan.review_plan(db, OWNER, run_id, agent_plan.PlanReviewInput(
            action="approve", expected_revision=public["state_revision"]), actor=HUMAN)
    assert result["state"] == "executing" and "Your plan is frozen" in inputs(run_id)[-1]
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        assert row.surf == coordinator.surf_for(run_id, 1, 1)  # The planner's session works: no handoff.


def test_a_clean_approve_skips_the_remaining_rounds(roles_host):
    created, run_id = hyper(roles_host, rounds=3)
    submit(run_id, created["id"])
    assert turn(run_id, 1) == ("plan_reviewing", "review")
    surf = reviewer_up(run_id)
    review(run_id, "approve", [{"id": "F1", "severity": "minor", "issue": "Name the flag."}])
    reviewer_turn(roles_host, run_id, surf)
    assert "last review" in inputs(run_id)[-1] and "remaining rounds are skipped" in inputs(run_id)[-1]
    # The planner may end its turn without resubmitting: the approved revision stands.
    assert turn(run_id, 2) == ("needs_you", "plan_approval")
    assert len(reviews(run_id)) == 1


def test_a_revise_review_needs_every_blocking_and_major_finding_answered(roles_host):
    created, run_id = hyper(roles_host, rounds=2)
    submit(run_id, created["id"])
    turn(run_id, 1)
    surf = reviewer_up(run_id)
    review(run_id, "revise", BLOCKING + [{"id": "F3", "severity": "major", "issue": "No rollback."}])
    reviewer_turn(roles_host, run_id, surf)
    with pytest.raises(DomainError) as unresolved:
        submit(run_id, created["id"], review_resolutions=[RESOLVED[0]])
    assert unresolved.value.code == "REVIEW_UNRESOLVED" and unresolved.value.data["unresolved"] == ["F3"]
    with pytest.raises(DomainError) as unknown:
        submit(run_id, created["id"], review_resolutions=[{"finding_id": "F9", "decision": "accepted"}])
    assert unknown.value.code == "INVALID_ARGUMENT"
    # Ending the turn without resubmitting after a revise review needs the operator.
    assert turn(run_id, 2) == ("needs_you", "no_plan")


def test_the_planner_asks_questions_between_rounds(roles_host):
    created, run_id = hyper(roles_host, rounds=2)
    submit(run_id, created["id"])
    turn(run_id, 1)
    surf = reviewer_up(run_id)
    review(run_id, "revise", BLOCKING, questions=["Which database?"])
    reviewer_turn(roles_host, run_id, surf)
    asked = call(agent_plan.task_interview, agent_plan.InterviewInput, run_id,
                 questions=[{"id": "db", "question": "Postgres or SQLite?"}])
    assert state(run_id) == ("interviewing", "interview")
    assert turn(run_id, 2) == ("interviewing", "interview")  # The operator has the move.
    with session_scope() as db:
        agent_plan.answer_interview(db, OWNER, asked["interview_id"],
                                    agent_plan.InterviewAnswers(answers={"db": "Postgres"}), actor=HUMAN)
    assert state(run_id)[0] == "planning"
    assert "continue incorporating review 1" in inputs(run_id)[-1] and "Postgres" in inputs(run_id)[-1]
    submit(run_id, created["id"], review_resolutions=RESOLVED)
    assert turn(run_id, 3) == ("plan_reviewing", "review")
    assert reviews(run_id)[-1].round == 2


def test_task_interview_while_planning_stays_refused_for_a_classic_run(host):  # noqa: F811
    created = task("Classic planning")
    run_id = start(host, created["id"])["id"]
    attached(host, run_id)
    submit(run_id, created["id"])
    with pytest.raises(DomainError) as locked:
        call(agent_plan.task_interview, agent_plan.InterviewInput, run_id, questions=[{"id": "q", "question": "?"}])
    assert locked.value.code == "PLAN_LOCKED"


def test_the_operator_sends_the_final_plan_back_with_a_note(roles_host):
    created, run_id = hyper(roles_host, rounds=1)
    submit(run_id, created["id"])
    turn(run_id, 1)
    one_round(roles_host, created, run_id)
    assert turn(run_id, 2) == ("needs_you", "plan_approval")
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        with pytest.raises(DomainError) as missing:
            agent_plan.review_plan(db, OWNER, run_id, agent_plan.PlanReviewInput(expected_revision=row.state_revision),
                                   actor=HUMAN)
        assert missing.value.code == "INVALID_ARGUMENT"
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        agent_plan.review_plan(db, OWNER, run_id, agent_plan.PlanReviewInput(
            action="revise", note="Split step 2.", expected_revision=row.state_revision), actor=HUMAN)
    assert state(run_id) == ("planning", "plan_revise") and "Split step 2." in inputs(run_id)[-1]
    submit(run_id, created["id"])
    assert turn(run_id, 3) == ("needs_you", "plan_approval")  # No rounds left: straight back to approval.
    assert len(reviews(run_id)) == 1


def test_without_approve_final_execution_starts_after_the_last_review(roles_host):
    created, run_id = hyper(roles_host, rounds=1, approve_final=False)
    submit(run_id, created["id"])
    turn(run_id, 1)
    one_round(roles_host, created, run_id)
    assert turn(run_id, 2) == ("executing", "plan_frozen")


# ---------------------------------------------------------------- the reviewer's scope


def test_the_reviewer_token_only_reviews_and_never_moves_the_run(roles_host):
    created, run_id = hyper(roles_host, rounds=2)
    submit(run_id, created["id"])
    turn(run_id, 1)
    surf = reviewer_up(run_id)
    reviewer = review_token(run_id)
    with session_scope() as db:  # It reads the plan.
        read = agent_plan.task_plan(db, BOT, agent_plan.PlanQuery(run_token=reviewer))
        assert read["actor"]["capability"] == "plan_reviewer" and read["task_id"] == created["id"]
    for fn, model, body in (
            (agent_plan.submit_plan, agent_plan.SubmitPlanInput, {**PLAN, "expected_plan_revision": 1}),
            (agent_plan.task_interview, agent_plan.InterviewInput, {"questions": [{"id": "q", "question": "?"}]}),
            (agent_plan.task_step, agent_plan.StepInput, {"task_id": created["id"], "step_id": "s1", "status": "done",
                                                          "expected_revision": 0})):
        with session_scope() as db, pytest.raises(DomainError) as refused:
            fn(db, BOT, model.model_validate({"run_token": reviewer, **body}))
        assert refused.value.code == "REVIEWER_SCOPE", fn.__name__
    with session_scope() as db, pytest.raises(DomainError) as planner:  # The planner's token cannot review.
        agent_plan.submit_plan_review(db, BOT, agent_plan.SubmitReviewInput.model_validate(
            {"run_token": token(run_id), "verdict": "approve"}))
    assert planner.value.code == "REVIEWER_ONLY"
    # Its SessionStart attaches nothing to the run, and its context carries the reviewer token, not the run's.
    with session_scope() as db:
        context = coordinator.context(db, OWNER, roles_host.id, SimpleNamespace(
            run_id=run_id, provider_session_id="rev-ctx", generation=1, provider="codex", surf=surf, incarnation=3, cwd=""))
        assert context["run_token"] == reviewer and "plan reviewer" in context["text"]
        row = db.get(AgentRun, run_id)
        assert row.session_pk is not None and row.surf == coordinator.surf_for(run_id, 1, 1)
        planner_pk = row.session_pk
    # A question from the reviewer's surface never pauses the run.
    with session_scope() as db:
        question, _ = coordinator.record_question(db, OWNER, roles_host.id, SimpleNamespace(
            provider="codex", provider_session_id="rev-ctx", run_id=run_id, provider_request_id="r1", generation=1,
            question_id="q1", group_id="g1", kind="choice", items=[], surf=surf, incarnation=3, provider_turn_id="",
            hold_until=now() + timedelta(minutes=5)))
        assert question.run_id is None
    assert state(run_id) == ("plan_reviewing", "review")
    with session_scope() as db, pytest.raises(DomainError) as verify:  # Only a reviewer is working: nothing to verify.
        control.verify_now(db, OWNER, run_id, actor=HUMAN)
    assert verify.value.code == "NOT_STARTED"
    review(run_id, "approve")
    with pytest.raises(DomainError) as twice:
        review(run_id, "revise")
    assert twice.value.code == "REVIEW_SUBMITTED"
    reviewer_turn(roles_host, run_id, surf)
    with session_scope() as db:
        assert db.get(AgentRun, run_id).session_pk == planner_pk  # Never bound as the run's session.
        assert db.scalar(select(AgentQuestion).where(AgentQuestion.run_id == run_id)) is None
    with session_scope() as db, pytest.raises(DomainError) as dead:  # The round is over: its token is dead.
        agent_plan.submit_plan_review(db, BOT, agent_plan.SubmitReviewInput.model_validate(
            {"run_token": reviewer, "verdict": "approve"}))
    assert dead.value.code == "RUN_TOKEN_REQUIRED"


def test_a_reviewer_that_ends_its_turn_without_the_tool_is_read_as_revise(roles_host):
    created, run_id = hyper(roles_host, rounds=2)
    submit(run_id, created["id"])
    turn(run_id, 1)
    surf = reviewer_up(run_id)
    reviewer_turn(roles_host, run_id, surf, text="Step 2 cannot work: the table is missing.")
    done = reviews(run_id)[0]
    assert done.verdict == "revise" and done.via == "turn" and done.status == "done"
    assert "the table is missing" in inputs(run_id)[-1]
    submit(run_id, created["id"])  # No findings to resolve.
    assert turn(run_id, 2) == ("plan_reviewing", "review")


# ---------------------------------------------------------------- the worker handoff


def test_a_different_worker_takes_over_after_approval(roles_host):
    created, run_id = hyper(roles_host, reviewer=None, worker=WORKER)
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        planner_spawn = db.scalar(select(AgentCommand).where(AgentCommand.run_id == run_id, AgentCommand.kind == "spawn"))
        assert row.roles["reviewer"] is None and row.roles["worker"] == WORKER
        assert planner_spawn.payload["role"] == "planner" and planner_spawn.payload["mode"] == "plan"  # Read-only.
        assert "A different model executes" in planner_spawn.payload["prompt"]
        assert "A reviewer reads" not in planner_spawn.payload["prompt"]
    planner_token = token(run_id)
    submit(run_id, created["id"])
    assert turn(run_id, 1) == ("needs_you", "plan_approval")  # No reviewer: straight to the approval stop.
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        result = agent_plan.review_plan(db, OWNER, run_id, agent_plan.PlanReviewInput(
            action="approve", expected_revision=row.state_revision), actor=HUMAN)
    assert result["state"] == "handing_off" and result["actor"] == {"role": "worker", "harness": "claude", "model": "sonnet"}
    planner_surf = coordinator.surf_for(run_id, 1, 1)
    stop = commands(run_id, "stop")[-1]
    assert stop.payload == {**stop.payload, "surf": planner_surf, "mode": "kill", "role": "planner"}
    spawn = commands(run_id, "spawn")[-1]
    assert spawn.payload["role"] == "worker" and "mode" not in spawn.payload and spawn.payload["attempt"] == 2
    assert spawn.payload["model"] == "sonnet" and "the plan is final" in spawn.payload["prompt"]
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        assert row.session_pk is None and row.tool == "claude" and row.model == "sonnet"
        assert coordinator.run_for_token(db, OWNER, planner_token) is None  # The planner's token is revoked.
        assert coordinator.desired(db, row)["attempt"] == 2 and "surf" not in coordinator.desired(db, row)
        kills = [e for e in coordinator.desired_entries(db, row) if e.get("surf") == planner_surf]
        assert kills == [{"run_id": run_id, "generation": row.generation, "desired": "stop", "stop_mode": "kill",
                          "surf": planner_surf}]
    # The retired planner's late report cannot take the run back; its SessionStart gets no token.
    with session_scope() as db:
        coordinator.attach(db, roles_host.id, run_id, 1, "claude", "sess-1", surf=planner_surf, incarnation=1)
        context = coordinator.context(db, OWNER, roles_host.id, SimpleNamespace(
            run_id=run_id, provider_session_id="sess-1", generation=1, provider="claude", surf=planner_surf,
            incarnation=1, cwd=""))
        assert "run_token" not in context
    assert state(run_id)[0] == "handing_off"
    with session_scope() as db:  # The planner's stop outcome moves nothing.
        coordinator.on_outcome(db, db.get(AgentCommand, stop.id), "executed", "", {"surf": planner_surf, "exited": True})
    assert state(run_id)[0] == "handing_off"
    worker_surf = coordinator.surf_for(run_id, 1, 2)
    with session_scope() as db:
        coordinator.attach(db, roles_host.id, run_id, 1, "claude", "worker-1", surf=worker_surf, incarnation=1)
    assert state(run_id) == ("executing", "handoff")
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        worker_pk = row.session_pk
        assert row.surf == worker_surf and (row.sync_state["handoff"]["done"] is True)
        assert coordinator.desired(db, row)["surf"] == worker_surf
        public = control.public_run(db, row)
        assert public["handoff"]["done"] is True and public["handoff"]["planner"] == {"harness": "claude", "model": "fable"}
        # Steer and the turn loop now target the worker.
        control.send_message(db, OWNER, run_id, control.MessageInput(text="Use the fast path.", mode="steer"), actor=HUMAN)
    message = commands(run_id, "input")[-1]
    assert message.payload["surf"] == worker_surf
    with session_scope() as db:
        coordinator.attach(db, roles_host.id, run_id, 1, "claude", "sess-1", surf=planner_surf, incarnation=1)
        assert db.get(AgentRun, run_id).session_pk == worker_pk
        coordinator.on_exit(db, agent_link.host_session(db, OWNER, roles_host.id, "claude", "sess-1"))
    assert state(run_id)[0] == "executing"  # The retired planner's exit is not the run's.


def test_a_worker_same_as_the_planner_is_a_classic_run(roles_host):
    created = task("Same worker")
    result = start(roles_host, created["id"], harness="claude", model="opus", worker={"harness": "claude"})
    with session_scope() as db:
        row = db.get(AgentRun, result["id"])
        spawn = db.scalar(select(AgentCommand).where(AgentCommand.run_id == row.id, AgentCommand.kind == "spawn"))
        assert row.roles is None and "role" not in spawn.payload and "mode" not in spawn.payload
        public = control.public_run(db, row)
        assert public["roles"] is None and public["reviews"] == [] and public["plan_round"] == 0
        assert public["actor"] == {"role": "agent", "harness": "claude", "model": "opus"} and public["handoff"] is None
        assert coordinator.desired_entries(db, row) == [coordinator.desired(db, row)]
        assert "surf" not in coordinator.desired(db, row) and "attempt" not in coordinator.desired(db, row)


def test_reviewer_and_split_worker_together(roles_host):
    created, run_id = hyper(roles_host, rounds=1, worker=WORKER)
    submit(run_id, created["id"])
    turn(run_id, 1)
    one_round(roles_host, created, run_id, findings=[{"id": "F1", "severity": "major", "issue": "Racy."}],
              resolve=[{"finding_id": "F1", "decision": "rejected", "note": "Single writer."}])
    assert turn(run_id, 2) == ("needs_you", "plan_approval")
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        agent_plan.review_plan(db, OWNER, run_id, agent_plan.PlanReviewInput(action="approve",
                                                                              expected_revision=row.state_revision),
                               actor=HUMAN)
    spawn = commands(run_id, "spawn")[-1]
    assert spawn.payload["role"] == "worker" and spawn.payload["attempt"] == 3
    assert "Racy." in spawn.payload["prompt"] and "Single writer." in spawn.payload["prompt"]  # The decisions digest.


# ---------------------------------------------------------------- failures, timeouts and host_busy


def test_a_reviewer_that_times_out_is_given_up_and_planning_goes_on(roles_host):
    created, run_id = hyper(roles_host, rounds=2)
    submit(run_id, created["id"])
    turn(run_id, 1)
    surf = reviewer_up(run_id)
    with session_scope() as db:
        coordinator.sweep(db, now() + coordinator.REVIEW_DEADLINE - timedelta(minutes=1))
    assert state(run_id) == ("plan_reviewing", "review")
    with session_scope() as db:
        coordinator.sweep(db, now() + coordinator.REVIEW_DEADLINE + timedelta(minutes=1))
    gone = reviews(run_id)[0]
    assert gone.status == "failed" and gone.via == "timeout"
    assert state(run_id) == ("needs_you", "plan_approval")  # "No review": the loop moves on, never re-spawns.
    assert commands(run_id, "stop")[-1].payload["surf"] == surf
    with session_scope() as db:
        history = db.get(AgentRun, run_id).history
        assert any(h.get("event") == "review_failed" for h in history)
    reviewer_turn(roles_host, run_id, surf, text="late")  # A late turn from the killed reviewer moves nothing.
    assert state(run_id) == ("needs_you", "plan_approval")


def test_a_busy_host_delays_the_reviewer_then_needs_you(roles_host):
    created, run_id = hyper(roles_host, rounds=1)
    submit(run_id, created["id"])
    turn(run_id, 1)
    busy = {"limit": 2, "running": 2}
    with session_scope() as db:
        spawn = db.scalars(select(AgentCommand).where(AgentCommand.run_id == run_id, AgentCommand.kind == "spawn")
                           .order_by(AgentCommand.created_at.desc())).first()
        coordinator.on_outcome(db, spawn, "rejected", "target_busy", busy)
    assert state(run_id) == ("plan_reviewing", "review") and reviews(run_id)[0].status == "spawning"
    sent = len(commands(run_id, "spawn"))
    with session_scope() as db:
        coordinator.sweep(db, now() + coordinator.REVIEW_BACKOFF + timedelta(seconds=1))
    assert len(commands(run_id, "spawn")) == sent + 1  # Sent again after the backoff, same attempt.
    resent = commands(run_id, "spawn")[-1]
    assert resent.payload["attempt"] == 2 and resent.payload["role"] == "reviewer"
    with session_scope() as db:
        review_row = db.scalar(select(AgentPlanReview).where(AgentPlanReview.run_id == run_id))
        review_row.blocked_since = now() - coordinator.REVIEW_BUSY_LIMIT - timedelta(seconds=1)
        coordinator.on_outcome(db, db.get(AgentCommand, resent.id), "rejected", "target_busy", busy)
    assert state(run_id) == ("needs_you", "host_busy")
    with session_scope() as db:
        coordinator.sweep(db, now() + coordinator.REVIEW_BACKOFF_MAX + timedelta(seconds=1))
    surf = reviewer_up(run_id)  # A slot freed up: the review round resumes.
    assert state(run_id) == ("plan_reviewing", "review")
    review(run_id, "approve")
    reviewer_turn(roles_host, run_id, surf)
    assert state(run_id) == ("planning", "review_done")


def test_a_busy_host_delays_the_worker_then_needs_you(roles_host):
    created, run_id = hyper(roles_host, reviewer=None, worker=WORKER, approve_final=False)
    submit(run_id, created["id"])
    assert turn(run_id, 1) == ("handing_off", "handoff")
    busy = {"limit": 2, "running": 2}
    worker_spawn = commands(run_id, "spawn")[-1]
    with session_scope() as db:
        coordinator.on_outcome(db, db.get(AgentCommand, worker_spawn.id), "rejected", "target_busy", busy)
    assert state(run_id) == ("handing_off", "handoff")
    with session_scope() as db:
        coordinator.sweep(db, now() + coordinator.REVIEW_BACKOFF + timedelta(seconds=1))
    resent = commands(run_id, "spawn")[-1]
    assert resent.id != worker_spawn.id and resent.payload["attempt"] == 2 and resent.payload["role"] == "worker"
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        row.sync_state = {**row.sync_state, "handoff": {
            **row.sync_state["handoff"],
            "blocked_since": (now() - coordinator.REVIEW_BUSY_LIMIT - timedelta(seconds=1)).isoformat()}}
        coordinator.on_outcome(db, db.get(AgentCommand, resent.id), "rejected", "target_busy", busy)
    assert state(run_id) == ("needs_you", "host_busy")
    with session_scope() as db:
        coordinator.sweep(db, now() + coordinator.REVIEW_BACKOFF_MAX + timedelta(seconds=1))
    assert state(run_id) == ("handing_off", "handoff")
    with session_scope() as db:
        coordinator.attach(db, roles_host.id, run_id, 1, "claude", "worker-1", surf=coordinator.surf_for(run_id, 1, 2),
                           incarnation=1)
    assert state(run_id) == ("executing", "handoff")


def test_a_worker_spawn_that_fails_fails_the_run(roles_host):
    created, run_id = hyper(roles_host, reviewer=None, worker=WORKER, approve_final=False)
    submit(run_id, created["id"])
    turn(run_id, 1)
    with session_scope() as db:
        spawn = db.get(AgentCommand, commands(run_id, "spawn")[-1].id)
        coordinator.on_outcome(db, spawn, "failed", "claude not found")
    assert state(run_id) == ("failed", "spawn_failed")


def test_stopping_during_a_review_kills_the_reviewer_too(roles_host):
    created, run_id = hyper(roles_host, rounds=2)
    submit(run_id, created["id"])
    turn(run_id, 1)
    surf = reviewer_up(run_id)
    with session_scope() as db:
        coordinator.request_stop(db, db.get(AgentRun, run_id), actor=HUMAN)
    assert state(run_id)[0] == "stopping" and reviews(run_id)[0].status == "skipped"
    stops = {c.payload["surf"]: c.payload for c in commands(run_id, "stop")}
    assert stops[surf]["mode"] == "kill" and coordinator.surf_for(run_id, 1, 1) in stops
    with session_scope() as db:  # The reviewer's stop outcome does not settle the run's stop.
        reviewer_stop = next(c for c in commands(run_id, "stop") if c.payload["surf"] == surf)
        coordinator.on_outcome(db, db.get(AgentCommand, reviewer_stop.id), "executed", "", {"exited": True})
    assert state(run_id)[0] == "stopping"


def test_a_queued_reviewer_spawn_is_canceled_not_mistaken_for_the_run(roles_host):
    created, run_id = hyper(roles_host, rounds=1)
    submit(run_id, created["id"])
    turn(run_id, 1)  # The reviewer's spawn is still queued.
    with session_scope() as db:
        coordinator.request_stop(db, db.get(AgentRun, run_id), actor=HUMAN)
    assert state(run_id)[0] == "stopping"  # The planner is still up: the stop waits for it.
    assert commands(run_id, "spawn")[-1].status == "canceled"


# ---------------------------------------------------------------- the link: snapshots and reconcile


def test_runs_snapshot_has_one_entry_per_surface(roles_host):
    created, run_id = hyper(roles_host, rounds=2)
    submit(run_id, created["id"])
    turn(run_id, 1)
    surf = reviewer_up(run_id)
    planner = coordinator.surf_for(run_id, 1, 1)
    with session_scope() as db:
        entries = coordinator.desired_entries(db, db.get(AgentRun, run_id))
    assert [(e.get("surf"), e["desired"]) for e in entries] == [(planner, "continue"), (surf, "continue")]
    assert "verify" not in entries[1]
    review(run_id, "revise")
    reviewer_turn(roles_host, run_id, surf)
    with session_scope() as db:
        entries = coordinator.desired_entries(db, db.get(AgentRun, run_id))
    assert [(e.get("surf"), e["desired"], e.get("stop_mode")) for e in entries] == [
        (planner, "continue", None), (surf, "stop", "kill")]  # A lost reviewer stop is carried out on reconnect.


def test_reconcile_fails_a_reviewer_the_host_lost(roles_host):
    created, run_id = hyper(roles_host, rounds=1)
    submit(run_id, created["id"])
    turn(run_id, 1)
    reviewer_up(run_id)
    planner = coordinator.surf_for(run_id, 1, 1)
    with session_scope() as db:
        runs = coordinator.reconcile(db, OWNER, roles_host.id, [SimpleNamespace(
            surf=planner, run_id=run_id, state="idle", incarnation=1, session_id="sess-1", tool="claude", generation=1)])
    assert reviews(run_id)[0].status == "failed"
    assert state(run_id) == ("needs_you", "plan_approval")
    assert {e.get("surf") for e in runs if e["run_id"] == run_id} >= {planner}


# ---------------------------------------------------------------- starting


def test_a_host_without_spawn_roles_refuses_hyperplan(host):  # noqa: F811
    created = task("No roles")
    for extra in ({"reviewer": REVIEWER}, {"worker": WORKER}):
        with pytest.raises(DomainError) as refused:
            start(host, created["id"], **extra)
        assert refused.value.code == "HOST_NO_ROLES"
    with session_scope() as db:
        out = control.preflight(db, OWNER, created["id"], "claude", host.target_id, reviewer_harness="codex")
    assert out["roles"] is False and any(p["code"] == "HOST_NO_ROLES" and p["role"] == "reviewer" for p in out["problems"])
    start(host, created["id"], worker={"harness": "claude", "model": "opus"})  # Same as the planner: a classic run.


def test_role_harnesses_are_checked_on_the_target(roles_host):
    created = task("Per-role harness")
    with session_scope() as db:
        row = db.get(AgentHost, roles_host.id)
        row.caps = {**row.caps, "harnesses": {**row.caps["harnesses"], "codex": {**row.caps["harnesses"]["codex"],
                                                                                 "spawn": False}}}
    with pytest.raises(DomainError) as refused:
        start(roles_host, created["id"], reviewer=REVIEWER)
    assert refused.value.code == "HARNESS_UNAVAILABLE" and refused.value.data["role"] == "reviewer"
    with session_scope() as db:
        out = control.preflight(db, OWNER, created["id"], "claude", roles_host.target_id, reviewer_harness="codex",
                                worker_harness="claude")
    assert out["roles"] is True and [p["role"] for p in out["problems"] if p["code"] == "HARNESS_UNAVAILABLE"] == ["reviewer"]
    with pytest.raises(DomainError) as freestyle:
        start(roles_host, created["id"], worker=WORKER, freestyle=True)
    assert freestyle.value.code == "INVALID_ARGUMENT"


def test_start_input_validates_roles_and_accepts_the_new_models():
    body = control.StartInput(task_id="t", harness="claude", model="fable", reviewer=REVIEWER, worker=WORKER, rounds=10)
    assert control.run_roles(body) == {"planner": {"harness": "claude", "model": "fable"}, "reviewer": REVIEWER,
                                       "worker": WORKER, "rounds": 10, "approve_final": True}
    assert control.run_roles(control.StartInput(task_id="t", reviewer={"harness": "codex"}))["reviewer"] == {
        "harness": "codex", "model": control.HARNESSES["codex"]}
    assert control.HARNESSES == {"claude": "opus", "codex": "gpt-5.6-sol"}  # Defaults unchanged.
    for bad in ({"rounds": 0}, {"rounds": 11}, {"reviewer": {"harness": "codex", "model": "-x"}}):
        with pytest.raises(ValueError):
            control.StartInput(task_id="t", **bad)
    with pytest.raises(DomainError):
        control.run_roles(control.StartInput(task_id="t", reviewer={"harness": "opencode"}))


def test_a_task_with_a_frozen_plan_still_plans_under_hyperplan(roles_host):
    created = task("Planned already", success_criteria=[{"id": "c1", "text": "It works"}],
                   verify={"target_id": "jarvis", "verify_id": "tests"})
    result = start(roles_host, created["id"], reviewer=REVIEWER)
    with session_scope() as db:
        row = db.get(AgentRun, result["id"])
        spawn = db.scalar(select(AgentCommand).where(AgentCommand.run_id == row.id, AgentCommand.kind == "spawn"))
        assert row.plan_revision == 0 and "verify" not in spawn.payload and "write the PRD" in spawn.payload["prompt"]
    attached(roles_host, result["id"])
    assert state(result["id"]) == ("planning", "started")


# ---------------------------------------------------------------- Eri and MCP


def test_eri_proposes_and_confirms_a_hyperplan_start(roles_host):
    created = task("Voice hyperplan")
    pending = say("control.start", task_id=created["id"], harness="claude", model="fable", rounds=2,
                  reviewer={"harness": "codex", "model": "gpt-6-astra"}, worker={"harness": "claude", "model": "sonnet"})
    assert "Codex gpt-6-astra reviewing (2 rounds)" in pending["summary"]
    assert "Claude Code sonnet working" in pending["summary"]
    with session_scope() as db:
        started = control.decide(db, OWNER, pending["pending_id"], confirm=True, via="app", actor=HUMAN,
                                 execution_target_id=roles_host.target_id)["run"]
    with session_scope() as db:
        row = db.get(AgentRun, started["id"])
        assert row.model == "fable" and row.roles["reviewer"] == REVIEWER and row.roles["worker"] == WORKER
        assert row.roles["rounds"] == 2 and row.roles["approve_final"] is True


def test_control_plan_review_approves_the_final_plan(roles_host):
    created, run_id = hyper(roles_host, reviewer=None, worker=WORKER)
    submit(run_id, created["id"])
    turn(run_id, 1)
    with session_scope() as db:
        spoken = control.spoken_run(db, db.get(AgentRun, run_id))
    assert spoken["plan_approval"]["plan_revision"] == 1 and spoken["actor"]["role"] == "planner"
    assert "control_plan_review" in spoken["plan_approval"]["next"]
    with pytest.raises(DomainError) as missing:
        say("control.plan_review", run_id=run_id)
    assert missing.value.code == "INVALID_ARGUMENT"
    out = say("control.plan_review", run_id=run_id, action="approve")
    assert out["status"] == "plan_approved" and out["state"] == "handing_off"
    with pytest.raises(DomainError) as again:
        say("control.plan_review", run_id=run_id, action="approve")
    assert again.value.code == "NO_PLAN_APPROVAL"


def test_submit_plan_review_is_an_mcp_tool_for_session_keys():
    from jarvis.external_mcp import definitions

    tools = {d["name"]: d for d in definitions({"agents:session"})}
    assert "submit_plan_review" in tools and "verdict" in str(tools["submit_plan_review"])
    assert "review_resolutions" in str(tools["submit_plan"])
    assert "submit_plan_review" not in {d["name"] for d in definitions({"agents:read"})}
