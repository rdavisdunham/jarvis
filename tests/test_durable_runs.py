"""Durable decisions and outbox receipts across retries, disconnections and explicit pause."""
from datetime import timedelta
from uuid import uuid4

import pytest
from jarvis import agent_control as control
from jarvis import agent_decisions as decisions
from jarvis import agent_link, agent_plan, agent_retention
from jarvis import agent_receipts as actions
from jarvis import run_coordinator as coordinator
from jarvis.agent_actor import Actor
from jarvis.db import session_scope
from jarvis.domain import DomainError
from jarvis.models import now
from jarvis.session_models import AgentActionReceipt, AgentCommand, AgentHost, AgentQuestion, AgentRun
from sqlalchemy import func, select
from test_agent_plan import BOT, PLAN, call
from test_agent_runs import HUMAN, OWNER, attached, quiet, task  # noqa: F401
from test_agent_runs import host as host  # noqa: PLC0414


@pytest.fixture
def durable(host):
    with session_scope() as db:
        row = db.get(AgentHost, host.id)
        row.caps = {**row.caps, "caps": [*row.caps["caps"], "decisions.durable", "commands.durable", "run.pause", "activity"]}
    return host


def new_run(host, attach=True):
    created = task()
    with session_scope() as db:
        result = control.start_run(db, OWNER, host.account_id, control.StartInput(
            task_id=created["id"], execution_target_id=host.target_id), actor=HUMAN)
    if attach:
        attached(host, result["id"])
    return result["id"]


def plan(host):
    run_id = new_run(host)
    submission_id = uuid4()
    result = call(agent_plan.submit_plan, agent_plan.SubmitPlanInput, run_id, submission_id=submission_id, **PLAN)
    return run_id, submission_id, result


def respond(identity, action="approve", **values):
    body = decisions.RespondInput(action_id=uuid4(), expected_revision=1, action=action, **values)
    with session_scope() as db:
        return decisions.respond(db, OWNER, identity, body, actor=HUMAN), body


def test_submission_is_immediately_visible_and_retry_receipt_is_stable(durable):
    run_id, submission_id, first = plan(durable)
    # No provider turn-end happened. The plan and its approval already exist.
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        public = control.public_run(db, run, detail=True)
        assert public["plan"]["revision"] == 1
        assert public["state_reason"] == "plan_approval"
        assert public["decisions"][0]["id"] == f"plan:{run_id}:1"
        assert public["decisions"][0]["status"] == "pending"
        assert actions.lookup(db, OWNER, submission_id)["status"] == "saved"
    repeated = call(agent_plan.submit_plan, agent_plan.SubmitPlanInput, run_id, submission_id=submission_id, **PLAN)
    assert repeated == first
    with session_scope() as db:
        assert db.get(AgentRun, run_id).plan_revision == 1
    with pytest.raises(DomainError, match="different request"):
        call(agent_plan.submit_plan, agent_plan.SubmitPlanInput, run_id, submission_id=submission_id,
             **{**PLAN, "prd": "A different plan"})


def test_message_cannot_hide_or_approve_pending_plan_and_explicit_approval_is_idempotent(durable):
    run_id, _, _ = plan(durable)
    with session_scope() as db:
        before = db.get(AgentRun, run_id)
        generation = before.generation
        control.send_message(db, OWNER, run_id, control.MessageInput(text="A useful note", mode="steer"), actor=HUMAN)
        assert before.generation == generation
        assert before.state_reason == "plan_approval"
        assert decisions.for_run(db, before)[0]["status"] == "pending"
    result, body = respond(f"plan:{run_id}:1")
    assert result["decision"]["status"] == "approved"
    assert result["result"]["state"] == "executing"
    with session_scope() as db:
        repeated = decisions.respond(db, OWNER, f"plan:{run_id}:1", body, actor=HUMAN)
        assert repeated["result"] == result["result"]
        assert len([h for h in db.get(AgentRun, run_id).history if h.get("event") == "plan_approved"]) == 1


def test_old_plan_decision_cannot_execute_after_revision(durable):
    run_id, _, _ = plan(durable)
    respond(f"plan:{run_id}:1", "revise", text="Add measurements")
    with session_scope() as db:
        task_revision = db.get(control.Task, db.get(AgentRun, run_id).task_id).plan_revision
    call(agent_plan.submit_plan, agent_plan.SubmitPlanInput, run_id,
         **{**PLAN, "expected_plan_revision": task_revision})
    with pytest.raises(DomainError) as stale:
        respond(f"plan:{run_id}:1")
    assert stale.value.code == "DECISION_CLOSED"
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        assert run.plan_revision == 2 and run.state_reason == "plan_approval"


def test_paused_runs_save_offline_messages_without_resuming_and_keep_receipts(durable):
    run_id = new_run(durable)
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        body = decisions.PauseInput(action_id=uuid4(), expected_revision=run.state_revision)
        actions.once(db, OWNER, run_id, "pause", body,
                     lambda: decisions.set_paused(db, OWNER, run_id, body, True, actor=HUMAN))
        host = db.get(AgentHost, durable.id)
        host.lease_until = now() - timedelta(days=1)
        message = control.MessageInput(text="Persist this offline", action_id=uuid4())
        actions.once(db, OWNER, run_id, "message", message,
                     lambda: control.send_message(db, OWNER, run_id, message, actor=HUMAN))
        assert run.paused
        command = db.scalar(select(AgentCommand).where(AgentCommand.action_id == str(message.action_id)))
        assert command.expires_at is None
        assert actions.lookup(db, OWNER, message.action_id)["status"] == "queued"
        assert not [f for f in agent_link.drain(db, host, host.conn_epoch) if f["t"] == "input"]
        coordinator.sweep(db, now() + timedelta(days=90))
        assert run.paused and command.status == "queued"
        receipt = db.get(AgentActionReceipt, (OWNER, str(message.action_id)))
        receipt.created_at = now() - timedelta(days=30)
        agent_retention.sweep(db, now() + timedelta(days=1))
        assert db.get(AgentActionReceipt, (OWNER, str(message.action_id)))


def test_prestart_message_materializes_surface_once_before_transmission(durable):
    run_id = new_run(durable, attach=False)
    with session_scope() as db:
        control.send_message(db, OWNER, run_id, control.MessageInput(text="Queued before attach"), actor=HUMAN)
        command = db.scalar(select(AgentCommand).where(AgentCommand.run_id == run_id, AgentCommand.kind == "input"))
        assert command.status == "deferred"
        assert not [f for f in agent_link.drain(db, db.get(AgentHost, durable.id), 1) if f["t"] == "input"]
    attached(durable, run_id)
    with session_scope() as db:
        frames = agent_link.drain(db, db.get(AgentHost, durable.id), 1)
        frame = next(f for f in frames if f["t"] == "input")
        assert frame["surf"] and frame["incarnation"] == 1 and frame["expires_at"] == 0
        assert frame["digest"] == agent_link.digest({k: v for k, v in frame.items() if k not in ("op_id", "digest")})


def test_durable_question_survives_human_wait_and_saved_answer_is_not_consumption(durable):
    run_id = new_run(durable)
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        question, _ = coordinator.record_question(db, OWNER, durable.id, agent_link.Question(
            run_id=run_id, generation=run.generation, provider="claude", provider_session_id="sess-1",
            question_id="question-stable", provider_request_id="request-stable", surf=run.surf,
            incarnation=1, durable=True, hold_until=0, items=[{"id": "scope", "text": "What scope?"}]))
        assert question.hold_until is None
        question_id = question.id
        coordinator.sweep(db, now() + timedelta(days=10))
        assert question.status == "open"
    result, _ = respond(question_id, "answer", answers=[{"item_id": "scope", "text": "Approved scope"}])
    assert result["decision"]["status"] == "submitted"
    assert result["receipt"]["status"] == "queued"
    with session_scope() as db:
        question = db.get(AgentQuestion, question_id)
        command = db.scalar(select(AgentCommand).where(AgentCommand.op_id == question.answer_op_id))
        coordinator.on_outcome(db, command, "uncertain", "Cannot prove consumption")
        assert question.status == "submitted" and question.answers[0]["text"] == "Approved scope"
        assert actions.lookup(db, OWNER, result["action_id"])["status"] == "uncertain"
        assert not [f for f in agent_link.drain(db, db.get(AgentHost, durable.id), 1) if f["t"] == "answer"]


def test_unbounded_defaults_and_truthful_activity_fences(durable):
    run_id = new_run(durable)
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        assert run.turn_cap == 0 and run.wall_clock_limit == 0
        assert coordinator.wall_clock_deadline(run) is None
        activity = agent_link.Activity(run_id=run_id, generation=run.generation, surf=run.surf, incarnation=1,
                                       kind="testing", label="pytest", sequence=2, session_id="sess-1", at=now())
        assert agent_link.record_activity(db, BOT, durable.id, activity) == {"saved": True}
        assert agent_link.record_activity(db, BOT, durable.id, activity) == {"duplicate": True}
        stale = activity.model_copy(update={"generation": run.generation + 1, "kind": "editing"})
        assert agent_link.record_activity(db, BOT, durable.id, stale) == {"ignored": True}
        assert control.public_run(db, run)["activity"]["kind"] == "testing"


def test_browser_start_receipt_survives_retry_and_cross_device_preferences(durable, client):
    created = task()
    action_id = str(uuid4())
    body = {"task_id": created["id"], "execution_target_id": durable.target_id, "action_id": action_id}
    first = client.post("/api/v1/remotea/runs", json=body)
    assert first.status_code == 200, first.text
    second = client.post("/api/v1/remotea/runs", json=body)
    assert second.status_code == 200 and second.json()["id"] == first.json()["id"]
    receipt = client.get(f"/api/v1/remotea/actions/{action_id}").json()
    assert receipt["run_id"] == first.json()["id"] and receipt["status"] == "queued"
    preferences = {"planner": {"harness": "codex", "model": "actual-host-model", "effort": "high",
                                "permission_profile": "read-only"}, "reviewer": None, "worker": None}
    assert client.put("/api/v1/remotea/role-preferences", json=preferences).status_code == 200
    assert client.get("/api/v1/remotea/role-preferences").json() == preferences
    key = "composer:" + first.json()["id"]
    saved = client.put("/api/v1/remotea/drafts/" + key, json={"value": {"text": "cross-device"}, "expected_revision": 0})
    assert saved.json() == {"value": {"text": "cross-device"}, "revision": 1}
    assert client.get("/api/v1/remotea/drafts/" + key).json() == saved.json()
    assert client.put("/api/v1/remotea/drafts/" + key, json={"value": "stale", "expected_revision": 0}).status_code == 409


def test_full_permission_action_is_visible_and_identity_reuse_cannot_change_it(durable):
    run_id = new_run(durable)
    action = {"tool": "shell", "input": {"command": "echo inspectable\n" * 500}, "cwd": "/workspace", "digest": "a" * 64}
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        frame = agent_link.Question(run_id=run_id, generation=run.generation, provider_session_id="sess-1",
            question_id="permission", provider_request_id="permission-provider-id", kind="permission",
            durable=True, action=action, request_digest="a" * 64,
            items=[{"id": "permission", "text": "Allow action?", "choices": [
                {"id": "allow_once", "label": "Allow once"}, {"id": "deny", "label": "Deny"}]}])
        row, _ = coordinator.record_question(db, OWNER, durable.id, frame)
        public = decisions.present(db, row, "question", run)
        assert public["payload"]["action"]["input"]["command"] == action["input"]["command"]
        assert len(public["payload"]["action"]["input"]["command"]) > 4000
        assert public["payload"]["request_digest"] == "a" * 64
        with pytest.raises(DomainError) as mismatch:
            coordinator.record_question(db, OWNER, durable.id, frame.model_copy(update={"request_digest": "b" * 64}))
        assert mismatch.value.code == "REQUEST_MISMATCH"


def test_typed_approved_requires_human_and_one_unambiguous_decision(durable):
    run_id, _, _ = plan(durable)
    with session_scope() as db:
        control.send_message(db, OWNER, run_id, control.MessageInput(text="Approved"), actor=Actor("run_session"))
        assert coordinator.frozen(db, db.get(AgentRun, run_id)).approval_status == "pending"
    with session_scope() as db:
        result = control.send_message(db, OWNER, run_id, control.MessageInput(
            text="Approved", expected_revision=db.get(AgentRun, run_id).state_revision), actor=HUMAN)
        assert result["state"] == "executing"
        assert coordinator.frozen(db, db.get(AgentRun, run_id)).approval_status == "approved"


def test_managed_resume_continues_only_approved_active_work(durable):
    run_id, _, _ = plan(durable)
    respond(f"plan:{run_id}:1")
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        host = db.get(AgentHost, durable.id)
        host.caps = {**host.caps, "caps": [*host.caps["caps"], "managed.sessions"]}
        run.sync_state = {**run.sync_state, "execution_mode": "managed"}
        # Prior execution prompt has a settled receipt; no uncertain external effect can be replayed.
        for command in db.scalars(select(AgentCommand).where(AgentCommand.run_id == run_id, AgentCommand.kind == "input")):
            command.status = "executed"
        pause = decisions.PauseInput(action_id=uuid4(), expected_revision=run.state_revision)
        decisions.set_paused(db, OWNER, run_id, pause, True, actor=HUMAN)
        resume = decisions.PauseInput(action_id=uuid4(), expected_revision=run.state_revision)
        decisions.set_paused(db, OWNER, run_id, resume, False, actor=HUMAN)
        inputs = list(db.scalars(select(AgentCommand).where(AgentCommand.run_id == run_id, AgentCommand.kind == "input")
                                  .order_by(AgentCommand.created_at)))
        assert "Do not repeat an external side effect" in inputs[-1].payload["text"]
        count = len(inputs)
        coordinator.continue_safely(db, run, "host_recovered")
        assert db.scalar(select(func.count()).select_from(AgentCommand).where(
            AgentCommand.run_id == run_id, AgentCommand.kind == "input")) == count


def test_managed_planner_defaults_to_host_model_and_approval_spawns_separate_worker(durable):
    with session_scope() as db:
        host_row = db.get(AgentHost, durable.id)
        codex = {**host_row.caps["harnesses"]["codex"], "managed": True, "managed_questions": ["ask", "approval"]}
        host_row.caps = {**host_row.caps, "caps": [*host_row.caps["caps"], "managed.sessions", "spawn.roles", "role.options"],
                         "harnesses": {**host_row.caps["harnesses"], "codex": codex},
                         "targets": [{**host_row.caps["targets"][0], "launch": {"codex": {"ready": True, "managed": True}}}],
                         "model_catalog": [{"harness": "codex", "model": "actual-default", "default": True,
                                            "efforts": ["high"], "permission_profiles": ["read-only", "workspace-write"]}]}
    created = task()
    with session_scope() as db:
        result = control.start_run(db, OWNER, durable.account_id, control.StartInput(
            task_id=created["id"], execution_target_id=durable.target_id, harness="codex", effort="high",
            permission_profile="read-only"), actor=HUMAN)
        run_id = result["id"]
        row = db.get(AgentRun, run_id)
        assert row.model == "actual-default"
        assert row.roles["worker"]["permission_profile"] == "workspace-write"
        spawn = db.scalar(select(AgentCommand).where(AgentCommand.run_id == run_id, AgentCommand.kind == "spawn"))
        assert spawn.payload["execution"] == "managed" and spawn.payload["mode"] == "plan"
        assert spawn.payload["effort"] == "high" and spawn.payload["model"] == "actual-default"
        coordinator.attach(db, durable.id, run_id, row.generation, "codex", "codex-planner",
                           surf=coordinator.surf_for(run_id, 1, 1), incarnation=1)
        assert control.public_run(db, row)["structured_answers"]
        planner_token = coordinator.run_token(row)
    with pytest.raises(DomainError) as missing:
        call(agent_plan.submit_plan, agent_plan.SubmitPlanInput, run_id, **PLAN)
    assert missing.value.code == "SUBMISSION_ID_REQUIRED"
    call(agent_plan.submit_plan, agent_plan.SubmitPlanInput, run_id, submission_id=uuid4(), **PLAN)
    result, _ = respond(f"plan:{run_id}:1")
    assert result["result"]["state"] == "handing_off"
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        spawns = list(db.scalars(select(AgentCommand).where(AgentCommand.run_id == run_id, AgentCommand.kind == "spawn")
                                 .order_by(AgentCommand.created_at)))
        worker = spawns[-1]
        assert worker.payload["role"] == "worker" and worker.payload["execution"] == "managed"
        assert worker.payload["permission_profile"] == "workspace-write" and worker.payload["effort"] == "high"
        assert coordinator.run_for_token(db, OWNER, planner_token) is None
        coordinator.attach(db, durable.id, run_id, run.generation, "codex", "codex-worker",
                           surf=coordinator.handoff_surf(run), incarnation=1)
        assert run.state == "executing" and coordinator.frozen(db, run).approval_status == "approved"


def test_approval_saved_while_paused_waits_for_explicit_resume(durable):
    run_id, _, _ = plan(durable)
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        pause = decisions.PauseInput(action_id=uuid4(), expected_revision=run.state_revision)
        decisions.set_paused(db, OWNER, run_id, pause, True, actor=HUMAN)
    result, _ = respond(f"plan:{run_id}:1")
    assert result["decision"]["status"] == "approved" and result["result"]["paused"]
    assert result["result"]["state"] == "needs_you" and result["receipt"]["terminal"]
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        assert run.plan_pending and run.paused
        resume = decisions.PauseInput(action_id=uuid4(), expected_revision=run.state_revision)
        decisions.set_paused(db, OWNER, run_id, resume, False, actor=HUMAN)
        assert run.state == "executing" and not run.paused and not run.plan_pending


def test_pr_open_default_applies_to_coding_runs_and_explicit_policy_is_preserved(durable):
    created = task()
    with session_scope() as db:
        original = db.get(control.Task, created["id"])
        assert original.completion_policy == "code_ready" and not original.completion_policy_explicit
        result = control.start_run(db, OWNER, durable.account_id, control.StartInput(
            task_id=created["id"], execution_target_id=durable.target_id), actor=HUMAN)
        assert result["completion_policy"] == "pr_open"
        assert original.completion_policy == "code_ready"  # Generic task metadata keeps its default.
    configured = task("Explicit policy", completion_policy="code_ready")
    with session_scope() as db:
        assert db.get(control.Task, configured["id"]).completion_policy_explicit
        result = control.start_run(db, OWNER, durable.account_id, control.StartInput(
            task_id=configured["id"], execution_target_id=durable.target_id), actor=HUMAN)
        assert result["completion_policy"] == "code_ready"


def test_model_catalog_rejects_unsupported_remembered_model_without_effort(durable):
    with session_scope() as db:
        host_row = db.get(AgentHost, durable.id)
        host_row.caps = {**host_row.caps, "model_catalog": [{"harness": "claude", "model": "available"}]}
    created = task()
    with session_scope() as db, pytest.raises(DomainError) as unavailable:
        control.start_run(db, OWNER, durable.account_id, control.StartInput(
            task_id=created["id"], execution_target_id=durable.target_id, model="retired-model"), actor=HUMAN)
    assert unavailable.value.code == "MODEL_UNAVAILABLE"


def test_recreated_question_keeps_one_decision_and_rebinds_current_callback(durable):
    run_id = new_run(durable)
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        frame = agent_link.Question(run_id=run_id, generation=run.generation, provider_session_id="sess-1",
            question_id="recreated", provider_request_id="old-callback", durable=True, request_digest="a" * 64,
            recovery="pending", surf=run.surf, incarnation=1, items=[{"id": "answer", "text": "Which?"}])
        first, _ = coordinator.record_question(db, OWNER, durable.id, frame)
        first_id = first.id
        revived = frame.model_copy(update={"provider_request_id": "new-callback", "incarnation": 2})
        second, duplicate = coordinator.record_question(db, OWNER, durable.id, revived)
        assert duplicate and second.id == first_id and second.incarnation == 2
        assert db.scalar(select(func.count()).select_from(AgentQuestion).where(AgentQuestion.question_id == "recreated")) == 1
    result, _ = respond(first_id, "answer", answers=[{"item_id": "answer", "text": "Saved"}])
    with session_scope() as db:
        question = db.get(AgentQuestion, first_id)
        command = db.scalar(select(AgentCommand).where(AgentCommand.op_id == question.answer_op_id))
        assert command.payload["provider_request_id"] == "new-callback" and command.payload["incarnation"] == 2
        original_op, original_digest = command.op_id, command.digest
        # The adapter proves the previous callback never attempted consent; a changed callback gets a NEW op.
        revived = revived.model_copy(update={"provider_request_id": "third-callback", "incarnation": 3})
        coordinator.record_question(db, OWNER, durable.id, revived)
        replacement = db.scalar(select(AgentCommand).where(AgentCommand.op_id == question.answer_op_id))
        assert replacement.op_id != original_op and command.digest == original_digest
        assert replacement.payload["provider_request_id"] == "third-callback"
        assert replacement.action_id == result["action_id"] and command.status == "superseded"
        assert actions.lookup(db, OWNER, result["action_id"])["status"] == "queued"
        # attention_required provides no safe replay proof, so another callback never gets another delivery.
        old_op = question.answer_op_id
        attention = revived.model_copy(update={"provider_request_id": "uncertain-callback", "incarnation": 4,
                                                "recovery": "attention_required"})
        coordinator.record_question(db, OWNER, durable.id, attention)
        assert question.answer_op_id == old_op and question.status == "submitted"


def test_terminal_takeover_is_sticky_and_blocks_resume_cancel_and_queued_effects(durable):
    run_id = new_run(durable)
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        control.send_message(db, OWNER, run_id, control.MessageInput(text="queued before takeover"), actor=HUMAN)
        frame = agent_link.Status(surf=run.surf, incarnation=1, state="needs_you", since=now(),
                                  execution="managed", controller="terminal", reason="Terminal has control")
        coordinator.on_status(db, db.get(control.AgentSession, run.session_pk), frame, host_id=durable.id)
        assert run.paused and control.public_run(db, run)["controller"] == "terminal"
        assert coordinator.desired(db, run)["desired"] == "pause"
        for act in (lambda: control.send_message(db, OWNER, run_id, control.MessageInput(text="must not send"), actor=HUMAN),
                    lambda: decisions.set_paused(db, OWNER, run_id, decisions.PauseInput(
                        action_id=uuid4(), expected_revision=run.state_revision), False, actor=HUMAN),
                    lambda: control.stop_run(db, OWNER, run_id, control.StopInput(), actor=HUMAN)):
            with pytest.raises(DomainError) as manual:
                act()
            assert manual.value.code == "TERMINAL_CONTROL"
        assert not [f for f in agent_link.drain(db, db.get(AgentHost, durable.id), 1) if f["t"] in ("input", "stop")]
        ordinary = frame.model_copy(update={"controller": "eridani", "state": "working"})
        coordinator.on_status(db, db.get(control.AgentSession, run.session_pk), ordinary, host_id=durable.id)
        assert run.paused and run.sync_state["controller"] == "terminal"


def test_managed_codex_answer_does_not_require_legacy_claude_question_hook(durable):
    run_id = new_run(durable)
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        run.tool = "codex"
        host_row = db.get(AgentHost, durable.id)
        host_row.caps = {**host_row.caps, "harnesses": {"claude": {"question": False},
            "codex": {"question": False, "managed": True, "managed_questions": ["ask", "approval"]}}}
        frame = agent_link.Question(run_id=run_id, generation=run.generation, provider="codex",
            provider_session_id="managed-codex-thread", question_id="codex-q", provider_request_id="codex-request",
            durable=True, surf=run.surf, incarnation=1, items=[{"id": "a", "text": "Which scope?"}])
        question, _ = coordinator.record_question(db, OWNER, durable.id, frame)
        coordinator.answer(db, question, [{"item_id": "a", "text": "This scope"}], actor=HUMAN)
        assert question.status == "submitted"


def test_managed_catalog_without_authoritative_default_requires_explicit_model(durable):
    with session_scope() as db:
        host_row = db.get(AgentHost, durable.id)
        host_row.caps = {**host_row.caps, "caps": [*host_row.caps["caps"], "managed.sessions", "role.options"],
            "harnesses": {**host_row.caps["harnesses"], "claude": {**host_row.caps["harnesses"]["claude"], "managed": True}},
            "model_catalog": [{"harness": "claude", "model": "first-is-not-default", "efforts": []}]}
    created = task()
    with session_scope() as db, pytest.raises(DomainError) as unavailable:
        control.start_run(db, OWNER, durable.account_id, control.StartInput(
            task_id=created["id"], execution_target_id=durable.target_id), actor=HUMAN)
    assert unavailable.value.code == "MODEL_DEFAULT_UNAVAILABLE"


def test_queued_text_survives_phase_revision_but_steer_and_approval_remain_fenced(durable):
    run_id, submission_id, _ = plan(durable)
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        before = run.state_revision
        run.state_revision += 1
        saved = control.send_message(db, OWNER, run_id, control.MessageInput(
            text="An offline note", expected_revision=before), actor=HUMAN)
        assert saved["state_reason"] == "plan_approval"
        for text, mode in (("steer", "steer"), ("Approved", "queue")):
            with pytest.raises(DomainError) as stale:
                control.send_message(db, OWNER, run_id, control.MessageInput(
                    text=text, mode=mode, expected_revision=before), actor=HUMAN)
            assert stale.value.code == "REVISION_CONFLICT"
        receipt = agent_plan.task_plan(db, BOT, agent_plan.PlanQuery(submission_id=submission_id,
                                                                    run_token="ert_retired_token"))
        assert receipt["submission"]["run_id"] == run_id and receipt["submission"]["kind"] == "submit_plan"


def test_late_exact_receipt_clears_only_automatic_pause_and_denial_cannot_be_overridden(durable):
    run_id = new_run(durable)
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        row = db.get(control.AgentSession, run.session_pk)
        frame = agent_link.Status(surf=run.surf, incarnation=1, state="needs_you", since=now(), paused=True,
            pause_reason="receipt_uncertain", attention={"kind": "receipt_uncertain", "question_id": "q"})
        coordinator.on_status(db, row, frame, host_id=durable.id)
        assert run.paused and control.public_run(db, run)["pause_reason"] == "receipt_uncertain"
        pause = decisions.PauseInput(action_id=uuid4(), expected_revision=run.state_revision)
        decisions.set_paused(db, OWNER, run_id, pause, True, actor=HUMAN)
        coordinator.clear_receipt_pause(db, run, "q")
        assert run.paused and run.sync_state["operator_paused"] and not run.sync_state.get("automatic_pause")
        denied = frame.model_copy(update={"pause_reason": "permission_denied", "attention": {"kind": "permission_denied"}})
        coordinator.on_status(db, row, denied, host_id=durable.id)
        resume = decisions.PauseInput(action_id=uuid4(), expected_revision=run.state_revision)
        with pytest.raises(DomainError) as forbidden:
            decisions.set_paused(db, OWNER, run_id, resume, False, actor=HUMAN)
        assert forbidden.value.code == "PERMISSION_DENIED" and run.paused


def test_pause_resume_rebinding_uses_new_operation_after_verified_incarnation_change(durable):
    run_id = new_run(durable)
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        run.paused = False
        old = coordinator.command(db, run, "run.resume", {"surf": run.surf, "incarnation": 1})
        original_digest = old.digest
        run.incarnation = 2  # A verified host snapshot arrived after this intent was saved.
        assert coordinator.refresh_control_fence(db, old, run)
        newer = db.scalar(select(AgentCommand).where(AgentCommand.op_id == old.outcome["superseded_by"]))
        assert old.status == "superseded" and old.digest == original_digest
        assert newer.op_id != old.op_id and newer.payload["incarnation"] == 2
        assert newer.payload["plan_revision"] == run.plan_revision
