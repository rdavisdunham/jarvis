"""Stage C: the run state machine, start/attach/stop/answer, reconcile and sweeps (run_coordinator.py)."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
from jarvis import agent_control as control
from jarvis import agent_link
from jarvis import run_coordinator as coordinator
from jarvis.agent_actor import Actor
from jarvis.db import session_scope
from jarvis.domain import DomainError
from jarvis.models import Job, Notification, Task, now
from jarvis.session_models import (
    AgentCommand,
    AgentHost,
    AgentQuestion,
    AgentRun,
    AgentRunPlan,
    AgentSession,
    AgentSessionRule,
    ExecutionTarget,
)
from sqlalchemy import func, select
from test_agent_link import bearer, connect, enable, hello
from test_linear import imported, linear, run  # noqa: F401 - pytest fixture
from test_organization_foundation import create

OWNER = "davin"
HUMAN = Actor("human")
CAPS = ["session.report", "turn", "status", "exit", "sessions.snapshot", "command.outcome", "question", "context",
        "spawn", "input", "answer", "stop", "runs.snapshot"]
HARNESSES = {"claude": {"version": "2.1.312", "observe": True, "capture": True, "question": True, "context": True,
                        "resume": True, "spawn": True},
             "codex": {"version": "0.163.0", "observe": True, "capture": True, "question": False, "spawn": True}}


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setenv("JARVIS_COST_TRACKING_ENABLED", "false")
    from jarvis.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def host(client):
    token = enable(client)["token"]
    with connect(bearer(token)) as ws:
        hello(ws, caps=CAPS, harnesses=HARNESSES,
              targets=[{"id": "jarvis", "repo": "/home/davin/jarvis", "worktree_root": "/home/davin/wt"}])
    with session_scope() as db:
        row = db.scalar(select(AgentHost))
        row.lease_until = now() + timedelta(hours=1)  # Connected for direct service calls.
        target = ExecutionTarget(owner_id=OWNER, host_id=row.id, host_target="jarvis", name="jarvis",
                                 repo="/home/davin/jarvis", worktree_root="/home/davin/wt", launch={}, verify=[])
        db.add(target)
        db.flush()
        return SimpleNamespace(id=row.id, account_id=row.account_id, token=token, target_id=target.id)


def start(host, task_id, **values):
    with session_scope() as db:
        # Stage C's manual run on the plan the task has; stage D's interview path is in test_agent_plan.py.
        body = control.StartInput(task_id=task_id, **{"execution_target_id": host.target_id, "interview": False, **values})
        return control.start_run(db, OWNER, host.account_id, body, actor=HUMAN)


def task(title="Ship stage C", **values):
    created = run("task.create", title=title)
    if values:
        created = run("task.update", task_id=created["id"], expected_revision=created["revision"], **values)
    return created


def with_run(fn):
    def call(run_id, *args, **kwargs):
        with session_scope() as db:
            row = db.get(AgentRun, run_id, with_for_update=True)
            result = fn(db, row, *args, **kwargs)
            return result
    return call


def attached(host, run_id, session_id="sess-1"):
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        coordinator.attach(db, host.id, run_id, row.generation, "claude", session_id,
                           surf=coordinator.surf_for(run_id, 1, 1), incarnation=1)
    with session_scope() as db:
        return db.get(AgentRun, run_id)


def state(run_id):
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        return row.state, row.state_reason


def commands(run_id, kind=None):
    with session_scope() as db:
        query = select(AgentCommand).where(AgentCommand.run_id == run_id).order_by(AgentCommand.created_at)
        if kind:
            query = query.where(AgentCommand.kind == kind)
        return [SimpleNamespace(**{c: getattr(r, c) for c in ("id", "op_id", "kind", "status", "generation", "payload",
                                                               "digest", "outcome", "expires_at")})
                for r in db.scalars(query)]


# ---------------------------------------------------------------- the transition map


def test_every_listed_transition_is_legal_and_others_are_refused(host):
    created = task()
    run_id = start(host, created["id"])["id"]
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        for source, targets in coordinator.TRANSITIONS.items():
            for target in sorted(targets):
                row.state, row.stop_final = source, "completed" if target == "completed" else "stopped"
                db.flush()
                coordinator.transition(db, row, target, reason="test")
                assert row.state == target
        assert row.history[-1]["actor"] == {"kind": "run_judge", "verified": True}
        for source, target in (("executing", "completed"), ("stopping", "executing"), ("completed", "executing"),
                               ("spawning", "awaiting_approval"), ("stopped", "stopping"), ("failed", "executing"),
                               ("needs_you", "completed")):
            row.state = source
            db.flush()
            with pytest.raises(DomainError) as refused:
                coordinator.transition(db, row, target)
            assert refused.value.code == "ILLEGAL_TRANSITION"
        row.state, row.stop_final = "stopping", "stopped"
        db.flush()
        with pytest.raises(DomainError, match="stopping to completed"):
            coordinator.transition(db, row, "completed")  # Only an approval settles a stop as completed.
        assert set(coordinator.STATES) >= {"interviewing", "planning", "awaiting_integration", "needs_you"}


def test_compare_and_set_on_state_revision(host):
    run_id = start(host, task()["id"])["id"]
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        with pytest.raises(DomainError) as stale:
            coordinator.transition(db, row, "executing", expected_revision=row.state_revision + 5)
        assert stale.value.code == "REVISION_CONFLICT"
    with session_scope() as first:
        old = first.get(AgentRun, run_id)
        with session_scope() as second:
            coordinator.transition(second, second.get(AgentRun, run_id), "executing")
        with pytest.raises(DomainError) as raced:
            coordinator.transition(first, old, "needs_you")  # Read before the other writer committed.
        assert raced.value.code == "REVISION_CONFLICT"
    assert state(run_id) == ("executing", "")


# ---------------------------------------------------------------- start


def test_start_run_writes_the_run_frozen_plan_and_spawn_together(host):
    note = run("note.create", title="PRD", content="Build manual remote control.")
    created = task(steps=[{"id": "s1", "title": "Migration"}], success_criteria=[{"text": "CI green"}],
                   prd_note_id=note["id"], turn_cap=9)
    result = start(host, created["id"])
    assert result["state"] == "spawning" and result["generation"] == 1 and result["plan_revision"] == 1
    assert result["branch"] == "task/" + created["id"] and result["base_branch"] == "main" and result["turn_cap"] == 9
    [spawn] = commands(result["id"], "spawn")
    body = spawn.payload
    assert spawn.status == "queued" and spawn.digest == agent_link.digest(body)
    assert body["t"] == "spawn" and body["command_id"] == spawn.op_id and body["run_id"] == result["id"]
    assert body["target"] == {"id": "jarvis", "repo": "/home/davin/jarvis", "branch": "task/" + created["id"], "base": "main"}
    assert body["harness"] == "claude" and body["model"] == "opus" and body["attempt"] == 1
    assert body["run_token"].startswith("ert_") and "Build manual remote control." in body["prompt"]
    assert "s1: Migration" in body["prompt"] and isinstance(body["expires_at"], int)
    with session_scope() as db:
        plan = db.scalar(select(AgentRunPlan))
        assert (plan.revision, plan.prd_text, plan.title) == (1, "Build manual remote control.", "Ship stage C")
        assert [s["id"] for s in plan.steps] == ["s1"] and plan.criteria[0]["text"] == "CI green"
        assert plan.task_plan_revision == db.get(Task, created["id"]).plan_revision
    with pytest.raises(DomainError) as twice:
        start(host, created["id"])
    assert twice.value.code == "RUN_ACTIVE"


def test_start_run_is_one_transaction(host, monkeypatch):
    created = task()

    def broken(*args, **kwargs):
        raise RuntimeError("outbox write failed")
    monkeypatch.setattr(coordinator, "command", broken)
    with pytest.raises(RuntimeError):
        start(host, created["id"])
    with session_scope() as db:
        assert db.scalar(select(func.count()).select_from(AgentRun)) == 0
        assert db.scalar(select(func.count()).select_from(AgentRunPlan)) == 0


def test_start_refuses_offline_hosts_unknown_harnesses_and_opencode(host):
    created = task()
    with pytest.raises(DomainError) as opencode:
        start(host, created["id"], harness="opencode")
    assert opencode.value.code == "HARNESS_DISABLED"
    with session_scope() as db:
        db.get(AgentHost, host.id).lease_until = now() - timedelta(seconds=1)
    with pytest.raises(DomainError) as offline:
        start(host, created["id"])
    assert offline.value.code == "HOST_OFFLINE"


def test_target_resolution_order(host):
    client_record = create("project", "Acme")
    record = create("task", "Fix login", parent_id=client_record["id"])
    tid = record["task_id"]
    with session_scope() as db:
        other = ExecutionTarget(owner_id=OWNER, host_id=host.id, host_target="acme", repo="/home/davin/acme",
                                worktree_root="/home/davin/wt", launch={}, verify=[])
        db.add(other)
        db.flush()
        other_id = other.id

    def resolved(explicit=None):
        with session_scope() as db:
            return control.resolve_target(db, OWNER, db.get(Task, tid), explicit).host_target

    with pytest.raises(DomainError) as missing:
        resolved()
    assert missing.value.code == "NEEDS_TARGET" and missing.value.status == 409
    assert {c["host_target"] for c in missing.value.data["candidates"]} == {"jarvis", "acme"}
    assert missing.value.data["advertised"][0]["host_target"] == "jarvis"
    assert [(r["title"], r["type_id"]) for r in missing.value.data["records"]] == [("Fix login", "task"), ("Acme", "project")]
    with session_scope() as db:  # A folder rule maps the client record to one repository.
        db.add(AgentSessionRule(owner_id=OWNER, path_prefix="/home/davin/acme", record_id=client_record["id"]))
    assert resolved() == "acme"
    with session_scope() as db:  # A saved one-time selection for the task's own record wins over rules.
        control.save_target(db, OWNER, control.TargetInput(host_id=host.id, host_target="jarvis", record_id=record["id"]))
    assert resolved() == "jarvis"
    with session_scope() as db:  # The task's own target wins over the chain.
        control.save_target(db, OWNER, control.TargetInput(host_id=host.id, host_target="acme", task_id=tid,
                                                            repo="/home/davin/acme"))
    assert resolved() == "acme"
    assert resolved(host.target_id) == "jarvis"  # An explicit choice wins over everything.
    with session_scope() as db:
        assert db.get(Task, tid).execution_target_id == other_id  # The same host target is reused, not duplicated.
    sibling = create("task", "Fix signup", parent_id=client_record["id"])["task_id"]
    with session_scope() as db:  # "Only this task" with a target saved only for a record binds the task to it.
        scoped = control.save_target(db, OWNER, control.TargetInput(host_id=host.id, host_target="remotea",
                                                                     repo="/home/davin/remotea", record_id=client_record["id"]))
        count = db.scalar(select(func.count()).select_from(ExecutionTarget))
        control.save_target(db, OWNER, control.TargetInput(host_id=host.id, host_target="remotea", task_id=sibling,
                                                            repo="/home/davin/remotea"))
        assert db.get(Task, sibling).execution_target_id == scoped["id"]
        assert db.scalar(select(func.count()).select_from(ExecutionTarget)) == count


def test_child_task_worktree_starts_from_the_parent_run_branch(host):
    parent = task("Parent")
    parent_run = start(host, parent["id"])
    child = run("task.create", title="Child", parent_task_id=parent["id"])
    assert start(host, child["id"])["base_branch"] == parent_run["branch"] == "task/" + parent["id"]


def test_issue_tasks_use_the_linear_branch_name(host, linear):  # noqa: F811
    tid, _ = imported()
    assert start(host, tid)["branch"] == "davin/eri-1-shared-task"


# ---------------------------------------------------------------- attach and observed start


def test_attach_only_for_the_assigned_host_and_generation(host, client):
    result = start(host, task()["id"])
    with session_scope() as db:
        intruder = AgentHost(owner_id=OWNER, account_id=host.account_id, credential_id=db.get(AgentHost, host.id).credential_id,
                             server_id="other", name="other", machine="", boot_id="", proto=1, conn_epoch=1, caps={})
        db.add(intruder)
        db.flush()
        intruder_id = intruder.id
    with session_scope() as db, pytest.raises(DomainError) as wrong_host:
        coordinator.attach(db, intruder_id, result["id"], 1, "claude", "sess-x")
    assert wrong_host.value.code == "RUN_NOT_BOUND"
    with session_scope() as db, pytest.raises(DomainError) as wrong_generation:
        coordinator.attach(db, host.id, result["id"], 7, "claude", "sess-x")
    assert wrong_generation.value.code == "STALE_GENERATION"
    assert state(result["id"]) == ("spawning", "start")
    row = attached(host, result["id"])
    assert row.state == "executing" and row.started_at and row.session_pk
    with session_scope() as db:
        session = db.get(AgentSession, row.session_pk)
        assert (session.run_id, session.task_id, session.binding_source) == (row.id, row.task_id, "spawned")
        assert db.get(Task, row.task_id).status == "in_progress"
    again = attached(host, result["id"], session_id="sess-other")  # A stray report cannot rebind the run.
    assert again.session_pk == row.session_pk


@pytest.fixture
def started_state(monkeypatch):
    import test_linear

    plain = test_linear.Fake.directory

    def directory(self, teams):
        found = plain(self, teams)
        found["states"].append({"id": "doing", "name": "In Progress", "type": "started", "team": {"id": "team"}})
        return found
    monkeypatch.setattr(test_linear.Fake, "directory", directory)


def test_a_refused_linear_status_never_undoes_the_observed_start(host, linear):  # noqa: F811
    tid, _ = imported()  # This workspace has no started workflow state.
    row = attached(host, start(host, tid)["id"])
    assert row.state == "executing" and row.sync_state["linear"]["in_progress"]["code"].startswith("LINEAR")
    with session_scope() as db:
        assert db.get(Task, tid).status != "in_progress"


def test_observed_start_moves_the_linear_issue_to_in_progress(host, started_state, linear):  # noqa: F811
    tid, _ = imported()
    result = start(host, tid)
    with session_scope() as db:
        assert db.get(Task, tid).status != "in_progress"  # Spawn queued is not a start.
        before = db.scalar(select(func.count()).select_from(Job).where(Job.kind.like("linear%")))
    attached(host, result["id"])
    with session_scope() as db:
        assert db.get(Task, tid).status == "in_progress"
        assert db.scalar(select(func.count()).select_from(Job).where(Job.kind.like("linear%"))) == before + 1


# ---------------------------------------------------------------- stop, outcomes and sweeps


def test_stop_is_desired_state_until_termination_is_confirmed(host):
    result = start(host, task()["id"])
    attached(host, result["id"])
    with session_scope() as db:
        control.stop_run(db, OWNER, result["id"], control.StopInput(), actor=HUMAN)
    assert state(result["id"]) == ("stopping", "stopped_by_user")
    [stop] = commands(result["id"], "stop")
    assert stop.generation == 2 and stop.payload["mode"] == "interrupt" and stop.payload["surf"].startswith("surf-eri-")
    with session_scope() as db:  # Asking again (kill) re-issues, still stopping.
        control.stop_run(db, OWNER, result["id"], control.StopInput(mode="kill"), actor=HUMAN)
    assert state(result["id"])[0] == "stopping" and len(commands(result["id"], "stop")) == 2
    with session_scope() as db:
        coordinator.on_outcome(db, db.get(AgentCommand, stop.id), "rejected", "stale incarnation")
    assert state(result["id"]) == ("stopping", "stop_refused")
    second = commands(result["id"], "stop")[1]
    with session_scope() as db:
        coordinator.on_outcome(db, db.get(AgentCommand, second.id), "executed", "exited")
        row = db.get(AgentRun, result["id"])
        assert row.state == "stopped" and row.finished_at and row.run_token_hash is None


def test_stop_before_the_host_took_the_spawn_settles_at_once(host):
    result = start(host, task()["id"])
    with session_scope() as db:
        control.stop_run(db, OWNER, result["id"], control.StopInput(), actor=HUMAN)
    assert state(result["id"])[0] == "stopped"
    assert commands(result["id"], "spawn")[0].status == "canceled" and not commands(result["id"], "stop")


def test_exit_while_stopping_confirms_and_exit_otherwise_needs_you(host):
    first = start(host, task("One")["id"])
    attached(host, first["id"], "sess-a")
    with session_scope() as db:
        row = db.get(AgentRun, first["id"])
        coordinator.on_exit(db, db.get(AgentSession, row.session_pk), "exit code 0")
    assert state(first["id"]) == ("needs_you", "exited")
    second = start(host, task("Two")["id"])
    attached(host, second["id"], "sess-b")
    with session_scope() as db:
        control.stop_run(db, OWNER, second["id"], control.StopInput(), actor=HUMAN)
    with session_scope() as db:
        row = db.get(AgentRun, second["id"])
        coordinator.on_exit(db, db.get(AgentSession, row.session_pk))
    assert state(second["id"])[0] == "stopped"


def test_uncertain_delivery_moves_the_run_to_needs_you(host):
    result = start(host, task()["id"])
    attached(host, result["id"])
    with session_scope() as db:
        control.send_message(db, OWNER, result["id"], control.MessageInput(text="Also update the docs"), actor=HUMAN)
    [message] = commands(result["id"], "input")
    assert message.payload["mode"] == "queue" and message.generation == 1
    with session_scope() as db:
        coordinator.on_outcome(db, db.get(AgentCommand, message.id), "uncertain", "server restarted mid-paste")
        assert db.scalar(select(Notification).where(Notification.category == "agent_stuck")).target["run_id"] == result["id"]
    assert state(result["id"]) == ("needs_you", "delivery_uncertain")
    with session_scope() as db:  # A message from Davin recovers the run.
        control.send_message(db, OWNER, result["id"], control.MessageInput(text="Carry on", mode="steer"), actor=HUMAN)
    assert state(result["id"]) == ("executing", "steer")
    assert commands(result["id"], "input")[-1].generation == 2


def test_expiry_sweep_and_limits(host):
    stuck = start(host, task("Stuck")["id"])
    with session_scope() as db:
        coordinator.sweep(db, now() + timedelta(minutes=20))
    assert state(stuck["id"]) == ("needs_you", "spawn_expired")
    assert commands(stuck["id"], "spawn")[0].status == "expired"
    slow = start(host, task("Slow")["id"])
    attached(host, slow["id"])
    with session_scope() as db:
        db.get(AgentHost, host.id).lease_until = now() + timedelta(days=1)
        coordinator.sweep(db, now() + timedelta(hours=1))
    assert state(slow["id"]) == ("needs_you", "no_progress")


def test_turns_count_only_the_current_generation_and_hit_the_cap(host):
    result = start(host, task(turn_cap=2)["id"])
    attached(host, result["id"])
    turn = SimpleNamespace(generation=1, interim=False, synthetic=False)
    with session_scope() as db:
        row = db.get(AgentRun, result["id"])
        coordinator.record_turn(db, row, SimpleNamespace(generation=0, interim=False, synthetic=False))
        coordinator.record_turn(db, row, SimpleNamespace(generation=1, interim=True, synthetic=False))
        coordinator.record_turn(db, row, turn)
        assert row.turn_count == 1 and row.state == "executing"
        coordinator.record_turn(db, row, turn)
        assert row.turn_count == 2 and (row.state, row.state_reason) == ("needs_you", "turn_cap")


# ---------------------------------------------------------------- questions


def asked(host, run_id, request_id="toolu_1", generation=1, **values):
    frame = agent_link.Question.model_validate({
        "question_id": "q-" + request_id, "surf": coordinator.surf_for(run_id, 1, 1), "incarnation": 1, "run_id": run_id,
        "generation": generation, "provider": "claude", "provider_session_id": "sess-1", "provider_request_id": request_id,
        "group_id": "g1", "hold_until": int((now() + timedelta(minutes=9)).timestamp() * 1000),
        "items": [{"id": "i1", "text": "Which database?", "choices": [{"id": "pg", "label": "Postgres"},
                                                                       {"id": "lite", "label": "SQLite"}]}], **values})
    with session_scope() as db:
        row, duplicate = coordinator.record_question(db, OWNER, host.id, frame)
        return row.id, duplicate


def answer(question_id, **item):
    with session_scope() as db:
        body = control.AnswerInput(answers=[{"item_id": "i1", **item}])
        return control.answer_question(db, OWNER, question_id, body, actor=HUMAN)


def resolved(host, question_id, outcome, **values):
    with session_scope() as db:
        question = db.get(AgentQuestion, question_id)
        frame = agent_link.QuestionResolved.model_validate({"question_id": question.question_id, "generation": 1,
                                                            "outcome": outcome, **values})
        return coordinator.question_resolved(db, host.id, frame).status


def test_question_lifecycle_first_answer_wins(host):
    result = start(host, task()["id"])
    attached(host, result["id"])
    qid, duplicate = asked(host, result["id"])
    assert not duplicate and asked(host, result["id"]) == (qid, True)  # A resent question is stored once.
    assert state(result["id"]) == ("waiting_answer", "question")
    with pytest.raises(DomainError, match="offered"):
        answer(qid, choice_ids=["mysql"])
    submitted = answer(qid, choice_ids=["pg"])
    assert submitted["status"] == "submitted" and submitted["answers"] == [{"item_id": "i1", "choice_ids": ["pg"]}]
    with pytest.raises(DomainError) as second_phone:
        answer(qid, text="SQLite please")
    assert second_phone.value.code == "ANSWER_PENDING"
    [sent] = commands(result["id"], "answer")
    assert sent.payload["provider_request_id"] == "toolu_1" and sent.payload["question_id"] == "q-toolu_1"
    assert state(result["id"])[0] == "waiting_answer"  # Submission is not acceptance.
    assert resolved(host, qid, "accepted", answer_echo={"Which database?": "Postgres"}) == "accepted"
    assert state(result["id"]) == ("executing", "answered")
    assert resolved(host, qid, "timed_out", still_open=True) == "accepted"  # A late resolution changes nothing.
    with pytest.raises(DomainError) as closed:
        answer(qid, choice_ids=["lite"])
    assert closed.value.code == "QUESTION_CLOSED"
    with session_scope() as db:
        assert db.scalar(select(Notification).where(Notification.category == "agent_question")) is not None


def test_timed_out_question_falls_back_to_the_terminal(host):
    result = start(host, task()["id"])
    attached(host, result["id"])
    qid, _ = asked(host, result["id"])
    assert resolved(host, qid, "timed_out", still_open=True) == "timed_out"
    assert state(result["id"]) == ("needs_you", "question_timed_out")
    with pytest.raises(DomainError) as closed:
        answer(qid, choice_ids=["pg"])
    assert closed.value.code == "QUESTION_CLOSED"
    assert resolved(host, qid, "answered_locally", answer_echo={"Which database?": "SQLite"}) == "answered_locally"
    assert state(result["id"]) == ("executing", "answered")


def test_stale_answers_are_refused(host):
    result = start(host, task()["id"])
    attached(host, result["id"])
    qid, _ = asked(host, result["id"])
    with session_scope() as db:  # A steer moves the run to the next generation; the old question is stale.
        control.send_message(db, OWNER, result["id"], control.MessageInput(text="Use Postgres", mode="steer"), actor=HUMAN)
    with pytest.raises(DomainError) as stale:
        answer(qid, choice_ids=["pg"])
    assert stale.value.code == "STALE_GENERATION"


def test_failed_answer_reopens_the_question(host):
    result = start(host, task()["id"])
    attached(host, result["id"])
    qid, _ = asked(host, result["id"])
    answer(qid, choice_ids=["pg"])
    [sent] = commands(result["id"], "answer")
    with session_scope() as db:
        coordinator.on_outcome(db, db.get(AgentCommand, sent.id), "rejected", "stale: answered elsewhere")
        assert db.get(AgentQuestion, qid).status == "open"
    assert answer(qid, choice_ids=["lite"])["status"] == "submitted"


# ---------------------------------------------------------------- reconcile


def surface(rid, state="live", generation=1, **values):
    body = {"surf": coordinator.surf_for(rid, 1, 1), "incarnation": 1, "state": state, "agent_state": "working",
            "since": int(now().timestamp() * 1000), "provider": "claude", "provider_session_id": "sess-" + rid[:8],
            "run_id": rid, "generation": generation}
    return agent_link.SnapshotSurface.model_validate({**body, **values})


def test_reconcile_rejects_stale_commands_and_settles_runs(host):
    live = start(host, task("Live")["id"])
    attached(host, live["id"], "sess-live")
    with session_scope() as db:
        control.send_message(db, OWNER, live["id"], control.MessageInput(text="first"), actor=HUMAN)
        control.send_message(db, OWNER, live["id"], control.MessageInput(text="now this", mode="steer"), actor=HUMAN)
    lost = start(host, task("Lost")["id"])
    attached(host, lost["id"], "sess-lost")
    stopping = start(host, task("Stopping")["id"])
    attached(host, stopping["id"], "sess-stop")
    with session_scope() as db:
        control.stop_run(db, OWNER, stopping["id"], control.StopInput(), actor=HUMAN)
    unacked = start(host, task("Unacked")["id"])
    with session_scope() as db:  # The spawn went out but its ack was lost.
        db.scalar(select(AgentCommand).where(AgentCommand.run_id == unacked["id"])).status = "sent"
    with session_scope() as db:
        runs = coordinator.reconcile(db, OWNER, host.id, [
            surface(live["id"], generation=2), surface(unacked["id"], run_id=None, provider_session_id=None)])
    queued, steered = commands(live["id"], "input")
    assert queued.status == "rejected" and "older generation" in queued.outcome["detail"]
    assert steered.status == "queued"
    assert state(lost["id"]) == ("needs_you", "surface_lost")
    assert state(stopping["id"])[0] == "stopped"
    assert commands(unacked["id"], "spawn")[0].status == "acked"  # Recovered from the deterministic surface id.
    with session_scope() as db:
        assert db.get(AgentRun, unacked["id"]).surf == coordinator.surf_for(unacked["id"], 1, 1)
    desired = {r["run_id"]: r["desired"] for r in runs}
    assert desired == {live["id"]: "continue", lost["id"]: "continue", unacked["id"]: "continue"}
    with session_scope() as db:  # The terminal comes back: the run heals itself.
        coordinator.reconcile(db, OWNER, host.id, [surface(lost["id"])])
    assert state(lost["id"]) == ("executing", "surface_restored")


def test_uncertain_journal_entries_after_a_host_restart(host):
    result = start(host, task()["id"])
    attached(host, result["id"])
    with session_scope() as db:
        control.send_message(db, OWNER, result["id"], control.MessageInput(text="go"), actor=HUMAN)
    [message] = commands(result["id"], "input")
    with session_scope() as db:
        db.get(AgentCommand, message.id).status = "acked"
        coordinator.reconcile(db, OWNER, host.id, [surface(result["id"])], {"pending": [], "uncertain": [message.op_id]})
    assert state(result["id"]) == ("needs_you", "delivery_uncertain")
    assert commands(result["id"], "input")[0].status == "uncertain"


# ---------------------------------------------------------------- approve and context


def test_approve_completes_the_task_as_a_human_and_settles_completed(host):
    result = start(host, task()["id"])
    attached(host, result["id"])
    with session_scope() as db:
        approved = control.approve(db, OWNER, result["id"], control.ApproveInput(
            expected_revision=db.get(AgentRun, result["id"]).state_revision), actor=HUMAN)
    assert approved["state"] == "stopping" and approved["task"]["status"] == "completed"
    [stop] = commands(result["id"], "stop")
    with session_scope() as db:
        coordinator.on_outcome(db, db.get(AgentCommand, stop.id), "executed")
        row = db.get(AgentRun, result["id"])
        assert row.state == "completed" and row.sync_state["completion"]["approved"] is True


def test_context_request_returns_the_frozen_plan_and_a_working_run_token(host):
    result = start(host, task(steps=[{"id": "s1", "title": "Write tests"}])["id"])
    frame = agent_link.ContextRequest.model_validate({
        "req_id": "r1", "surf": coordinator.surf_for(result["id"], 1, 1), "incarnation": 1, "provider": "claude",
        "provider_session_id": "sess-ctx", "source": "startup", "cwd": "/home/davin/wt/x", "run_id": result["id"],
        "generation": 1})
    with session_scope() as db:
        reply = coordinator.context(db, OWNER, host.id, frame)
    assert "<<<ERIDANI_TASK_" in reply["text"] and "s1: Write tests" in reply["text"] and reply["run_token"] in reply["text"]
    assert reply["scope"] == "run"
    with session_scope() as db:
        assert coordinator.run_for_token(db, OWNER, reply["run_token"]).id == result["id"]
        assert coordinator.run_for_token(db, "someone-else", reply["run_token"]) is None
    assert state(result["id"])[0] == "executing"  # SessionStart named the provider session: observed start.
    unlinked = agent_link.ContextRequest.model_validate({"req_id": "r2", "provider": "claude", "provider_session_id": "s9",
                                                          "cwd": "/tmp"})
    with session_scope() as db:
        reply = coordinator.context(db, OWNER, host.id, unlinked)
    assert "session_link" in reply["text"] and "s9" in reply["text"]
    assert reply["scope"] == "candidates" and "run_token" not in reply and len(reply["text"]) <= 4000


def test_every_transition_reason_has_page_text():
    """The run card and drawer show REASON_TEXT for the current state; a reason without text showed the raw
    token ("plan_frozen") on the page. Every reason the coordinator, planner or control layer passes to
    transition() must have one, and an unknown reason renders as nothing rather than as the token."""
    import re
    from pathlib import Path

    from jarvis.agent_control import REASON_TEXT

    root = Path(__file__).resolve().parents[1] / "apps" / "api" / "jarvis"
    used = set()
    for name in ("run_coordinator.py", "agent_plan.py", "agent_control.py"):
        used |= set(re.findall(r'reason="([a-z_]+)"\s*[,)]', (root / name).read_text()))  # Not "spawn_" + state.
    used |= {"spawn_failed", "spawn_rejected", "spawn_uncertain", "spawn_expired"}  # run_coordinator's "spawn_" + outcome
    missing = sorted(r for r in used if r not in REASON_TEXT)
    assert not missing, f"transition reasons without page text: {missing}"
    assert REASON_TEXT.get("no_such_reason", "") == ""


# ---------------------------------------------------------------- idempotent run actions (agent_receipts.py)


def test_a_repeated_message_with_the_same_action_id_acts_once(host, client):
    import uuid

    from jarvis.session_models import AgentActionReceipt

    run_id = start(host, task()["id"])["id"]
    attached(host, run_id)
    action_id = str(uuid.uuid4())
    url = f"/api/v1/remotea/runs/{run_id}/message"
    body = {"text": "Also update the docs.", "mode": "steer", "action_id": action_id}
    first = client.post(url, json=body)
    assert first.status_code == 200, first.text
    with session_scope() as db:
        generation = db.get(AgentRun, run_id).generation
    again = client.post(url, json=body)  # The response was lost; the drawer retries with the same id.
    assert again.status_code == 200 and again.json() == first.json()
    assert len(commands(run_id, "input")) == 1  # One prompt queued, one steer generation.
    with session_scope() as db:
        assert db.get(AgentRun, run_id).generation == generation
        receipt = db.get(AgentActionReceipt, (OWNER, action_id))
        assert receipt.kind == "message" and receipt.run_id == run_id
    other = client.post(url, json={**body, "text": "Something else."})
    assert other.status_code == 409 and other.json()["error"]["code"] == "ACTION_MISMATCH"
    elsewhere = client.post(f"/api/v1/remotea/runs/{run_id}/stop", json={"action_id": action_id})
    assert elsewhere.status_code == 409 and elsewhere.json()["error"]["code"] == "ACTION_MISMATCH"
    assert len(commands(run_id, "input")) == 1 and not commands(run_id, "stop")
    fresh = client.post(url, json={**body, "action_id": str(uuid.uuid4())})  # A new id is a new message.
    assert fresh.status_code == 200 and len(commands(run_id, "input")) == 2
    plain = client.post(url, json={"text": "No id: acts every time."})
    assert plain.status_code == 200 and len(commands(run_id, "input")) == 3
    assert client.post(url, json={"text": "x", "action_id": "not-a-uuid"}).status_code == 422


def test_a_repeated_stop_returns_the_first_result_and_a_refused_action_leaves_no_receipt(host, client):
    import uuid

    from jarvis.session_models import AgentActionReceipt

    run_id = start(host, task()["id"])["id"]
    action_id = str(uuid.uuid4())
    # Not started yet: the message is refused, nothing is stored, and the same id may act once it can.
    early = client.post(f"/api/v1/remotea/runs/{run_id}/message", json={"text": "hi", "action_id": action_id})
    assert early.status_code == 409 and early.json()["error"]["code"] == "NOT_STARTED"
    with session_scope() as db:
        assert db.get(AgentActionReceipt, (OWNER, action_id)) is None
    attached(host, run_id)
    stop_id = str(uuid.uuid4())
    url = f"/api/v1/remotea/runs/{run_id}/stop"
    first = client.post(url, json={"mode": "interrupt", "action_id": stop_id})
    assert first.status_code == 200, first.text
    again = client.post(url, json={"mode": "interrupt", "action_id": stop_id})
    assert again.json() == first.json() and len(commands(run_id, "stop")) == 1


def test_retention_drops_old_action_receipts():
    from jarvis import agent_retention as retention
    from jarvis.session_models import AgentActionReceipt

    with session_scope() as db:
        db.add(AgentActionReceipt(owner_id=OWNER, action_id="old", run_id="r", kind="stop", digest="d", result={},
                                  created_at=now() - timedelta(days=8)))
        db.add(AgentActionReceipt(owner_id=OWNER, action_id="new", run_id="r", kind="stop", digest="d", result={}))
    with session_scope() as db:
        assert retention.sweep(db)["action_receipts"] == 1
        assert db.get(AgentActionReceipt, (OWNER, "new")) is not None
