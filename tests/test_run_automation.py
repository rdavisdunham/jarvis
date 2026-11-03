"""Direct human input holds the automatic loop without pausing current provider work."""
from datetime import timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy import select

from jarvis import agent_automation as automation, agent_control as control, agent_decisions, agent_link, agent_retention
from jarvis import agent_receipts as actions, run_coordinator as coordinator
from jarvis.db import session_scope
from jarvis.domain import DomainError
from jarvis.models import now
from jarvis.session_models import AgentAttempt, AgentCommand, AgentHost, AgentRun, AgentSession, AgentTurn
from test_agent_runs import HUMAN, OWNER, attached, start, task
from test_agent_runs import host as host  # noqa: F401


@pytest.fixture
def direct_host(host):
    with session_scope() as db:
        row = db.get(AgentHost, host.id)
        row.caps = {**row.caps, "caps": [*row.caps["caps"], "commands.durable", "decisions.durable", "input.direct", "output.live", "run.pause"]}
    return host


def running(host, *, attach=True):
    identity = start(host, task()["id"])["id"]
    if attach:
        attached(host, identity)
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        run.sync_state = {**run.sync_state, "execution_mode": "managed"}
        if run.session_pk:
            session = db.get(AgentSession, run.session_pk)
            session.surf, session.incarnation, session.state = run.surf, run.incarnation, "idle"
        for command in db.scalars(select(AgentCommand).where(AgentCommand.run_id == identity)):
            command.status = "executed"
    return identity


def send(db, run_id, text="Check in", action_id=None):
    body = control.MessageInput(text=text, mode="direct", action_id=action_id or uuid4())
    result = actions.once(db, OWNER, run_id, "message", body,
        lambda: control.send_message(db, OWNER, run_id, body, actor=HUMAN))
    command = db.scalar(select(AgentCommand).where(AgentCommand.action_id == str(body.action_id), AgentCommand.kind == "input"))
    return body, result, command


def receipt(db, host, run, command, stage="handed_to_harness", **changes):
    return agent_link.record_input_receipt(db, SimpleNamespace(owner_id=OWNER), host.id,
        agent_link.InputReceipt(run_id=run.id, generation=run.generation, surf=run.surf,
            incarnation=run.incarnation, input_id=command.op_id, session_id="sess-1", turn_id="turn-1",
            stage=stage, at=now(), **changes))


def resume(db, identity):
    run = db.get(AgentRun, identity)
    body = automation.ResumeInput(action_id=uuid4(), expected_automation_revision=automation.state(run)["revision"])
    return automation.resume(db, OWNER, identity, body, actor=HUMAN)


def test_direct_approved_is_literal_and_saved_once_with_atomic_hold(direct_host):
    identity = running(direct_host)
    message_id = uuid4()
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        plan = coordinator.frozen(db, run)
        plan.approval_status, run.plan_pending, run.state, run.state_reason = "pending", True, "needs_you", "plan_approval"
        body, first, command = send(db, identity, "Approved", message_id)
        assert command.payload["text"] == "Approved" and command.payload["mode"] == "direct"
        assert plan.approval_status == "pending" and run.plan_pending
        assert first["automation"]["held"] and first["automation"]["revision"] == 1 and not run.paused
        again = actions.once(db, OWNER, identity, "message", body,
            lambda: control.send_message(db, OWNER, identity, body, actor=HUMAN))
        assert again == first and automation.state(run)["revision"] == 1
        assert actions.lookup(db, OWNER, message_id)["delivery_stage"] == "saved"
        frames = agent_link.drain(db, db.get(AgentHost, direct_host.id), 1)
        assert any(frame["op_id"] == command.op_id for frame in frames)  # Pending plan does not swallow chat.
        assert agent_decisions.for_run(db, run)[0]["status"] == "pending"


@pytest.mark.parametrize("stage", ["interviewing", "planning", "plan_reviewing", "handing_off", "waiting_answer", "needs_you", "awaiting_approval", "verifying", "judging"])
def test_every_active_phase_can_hold_without_hiding_decisions(direct_host, stage):
    identity = running(direct_host)
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        run.state = stage
        _, result, _ = send(db, identity)
        assert result["automation"]["held"] and not run.paused
        assert run.state == ("executing" if stage in ("verifying", "judging") else stage)


def test_rejected_input_and_old_hosts_do_not_change_hold(host):
    identity = running(host)
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        with pytest.raises(DomainError, match="cannot deliver direct"):
            send(db, identity)
        assert not automation.held(run)
        assert not list(db.scalars(select(AgentCommand).where(AgentCommand.run_id == identity, AgentCommand.kind == "automation.hold")))


def test_offline_and_prestart_hold_is_durable_and_pause_still_blocks_delivery(direct_host):
    identity = running(direct_host, attach=False)
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        db.get(AgentHost, direct_host.id).lease_until = now() - timedelta(days=1)
        body, _, command = send(db, identity, "Saved while offline")
        assert command.status == "deferred" and command.expires_at is None
        op_id, digest = command.op_id, command.digest
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        assert automation.held(run) and actions.lookup(db, OWNER, body.action_id)["delivery_stage"] == "saved"
        coordinator.attach(db, direct_host.id, identity, 1, "claude", "sess-1", surf=coordinator.surf_for(identity, 1, 1), incarnation=1)
        run.paused = True
        frames = agent_link.drain(db, db.get(AgentHost, direct_host.id), 2)
        assert not any(frame["op_id"] == op_id for frame in frames)
        assert automation.held(run) and run.paused
        run.paused = False
        frames = agent_link.drain(db, db.get(AgentHost, direct_host.id), 3)
        direct = next(frame for frame in frames if frame["op_id"] == op_id)
        assert direct["surf"] == run.surf and direct["automation_hold_revision"] == 1
        assert direct["digest"] != digest  # Never sent with an empty fence; materialized once.


def test_hold_suppresses_old_automatic_prompts_and_new_followup(direct_host):
    identity = running(direct_host)
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        queued = coordinator.say(db, run, "Old automatic gap", "gap")
        sent = coordinator.say(db, run, "In-transit automatic gap", "gap")
        sent.status = "sent"
        send(db, identity)
        assert queued.status == "superseded" and coordinator.stale(db, sent, run)
        seq = run.turn_seq
        coordinator.record_turn(db, run, SimpleNamespace(generation=run.generation, seq=seq + 1))
        assert run.turn_count == 1 and run.turn_seq == seq + 1
        assert run.sync_state["automation_pending"]["kind"] == "after_turn"
        assert not db.scalar(select(AgentAttempt.id).where(AgentAttempt.run_id == identity, AgentAttempt.kind.in_(("verify", "judge"))))
        new = coordinator.say(db, run, "Another automatic gap", "gap")
        assert new.status == "superseded"


def test_resume_waits_for_handoff_not_model_read_and_release_ack_is_authoritative(direct_host):
    identity = running(direct_host)
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        _, _, command = send(db, identity)
        with pytest.raises(DomainError) as caught:
            resume(db, identity)
        assert caught.value.code == "DELIVERY_PENDING" and automation.held(run)
        assert receipt(db, direct_host, run, command) == {"saved": True}
        result = resume(db, identity)
        assert result["automation"] == {**result["automation"], "held": True, "revision": 2, "releasing": True}
        release = db.scalar(select(AgentCommand).where(AgentCommand.op_id == automation.state(run)["release_op_id"]))
        assert command.outcome["result"]["provider_ack"] == "pending"
        coordinator.on_outcome(db, release, "executed")
        assert not automation.held(run) and automation.state(run)["revision"] == 2


def test_receipt_ack_is_monotonic_across_late_generic_outcomes_and_wrong_session(direct_host):
    identity = running(direct_host)
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        body, _, command = send(db, identity)
        assert receipt(db, direct_host, run, command) == {"saved": True}
        coordinator.on_outcome(db, command, "executed", result={"provider_ack": "pending", "receipt": "handed_to_harness"})
        assert receipt(db, direct_host, run, command, "provider_acknowledged") == {"saved": True}
        coordinator.on_outcome(db, command, "executed", result={"provider_ack": "pending"})
        assert command.outcome["result"]["receipt_stage"] == "provider_acknowledged"
        assert command.outcome["result"]["provider_ack"] == "confirmed"
        assert actions.lookup(db, OWNER, body.action_id)["delivery_stage"] == "provider_acknowledged"
        wrong = agent_link.InputReceipt(run_id=identity, generation=1, surf=run.surf, incarnation=1,
            input_id=command.op_id, session_id="other-session", stage="uncertain", at=now())
        assert agent_link.record_input_receipt(db, SimpleNamespace(owner_id=OWNER), direct_host.id, wrong) == {"ignored": True}


def test_two_devices_and_old_release_cannot_unhold_new_message(direct_host):
    identity = running(direct_host)
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        _, _, first = send(db, identity, "First device")
        receipt(db, direct_host, run, first)
        resume(db, identity)
        release = db.scalar(select(AgentCommand).where(AgentCommand.op_id == automation.state(run)["release_op_id"]))
        _, _, second = send(db, identity, "Second device")
        assert automation.state(run)["revision"] == 3
        coordinator.on_outcome(db, release, "executed")
        assert automation.held(run)
        with pytest.raises(DomainError) as caught:
            automation.resume(db, OWNER, identity, automation.ResumeInput(action_id=uuid4(), expected_automation_revision=1), actor=HUMAN)
        assert caught.value.code == "REVISION_CONFLICT"
        assert len(control.public_run(db, run, detail=True)["message_deliveries"]) == 2
        receipt(db, direct_host, run, second, "uncertain")
        with pytest.raises(DomainError) as caught:
            resume(db, identity)
        assert caught.value.code == "DELIVERY_UNCERTAIN"


def test_explicit_plan_approval_does_not_release_hold(direct_host):
    identity = running(direct_host)
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        plan = coordinator.frozen(db, run)
        run.state, run.state_reason, run.plan_pending, plan.approval_status = "needs_you", "plan_approval", True, "pending"
        send(db, identity, "Approved")
        automation.defer(run, "review_done", review_id="obsolete-planning-continuation")
        coordinator.approve_plan(db, run, actor=HUMAN)
        assert plan.approval_status == "approved" and automation.held(run)
        assert not run.sync_state.get("automation_pending")
        prompt = db.scalars(select(AgentCommand).where(AgentCommand.run_id == identity, AgentCommand.kind == "input")
            .order_by(AgentCommand.created_at.desc())).first()
        assert prompt.payload["origin"] == "human" and prompt.status == "queued"


def test_old_judge_result_cannot_restart_work_after_new_direct_turn(direct_host):
    identity = running(direct_host)
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        run.state = "judging"
        old = AgentAttempt(owner_id=OWNER, run_id=identity, generation=1, turn_seq=run.turn_seq,
            plan_revision=run.plan_revision, kind="judge", status="running", result={})
        db.add(old); db.flush()
        _, _, command = send(db, identity)
        db.get(AgentSession, run.session_pk).state = "working"
        old.status = "done"
        coordinator.on_verdict(db, old, {"verdict": "not_met", "gap": "OBSOLETE", "criteria": [], "steps": []})
        assert old.status == "stale" and run.state == "executing"
        receipt(db, direct_host, run, command)
        resume(db, identity)
        release = db.scalar(select(AgentCommand).where(AgentCommand.op_id == automation.state(run)["release_op_id"]))
        coordinator.on_outcome(db, release, "executed")
        assert not automation.held(run)
        assert not any((row.payload or {}).get("text") == "OBSOLETE" for row in db.scalars(select(AgentCommand).where(AgentCommand.run_id == identity)))
        assert not run.sync_state.get("automation_pending")


@pytest.mark.parametrize("limit", ["wall_clock", "turn_cap"])
def test_direct_does_not_override_hard_limits_or_hold_on_rejection(direct_host, limit):
    identity = running(direct_host)
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        run.state, run.state_reason = "needs_you", limit
        with pytest.raises(DomainError) as caught:
            send(db, identity)
        assert caught.value.code == "RUN_LIMIT" and not automation.held(run)


def test_published_output_is_bounded_fenced_ordered_and_never_a_final_turn(direct_host):
    identity = running(direct_host)
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        bot = SimpleNamespace(owner_id=OWNER)
        def frame(sequence, **changes):
            return agent_link.LiveOutput(run_id=identity, generation=run.generation, surf=run.surf,
                incarnation=run.incarnation, session_id="sess-1", turn_id="ongoing", message_id="commentary",
                sequence=sequence, at=now(), kind="commentary", text="Published progress", complete=False, **changes)
        assert agent_link.record_live_output(db, bot, direct_host.id, frame(2)) == {"saved": True}
        assert agent_link.record_live_output(db, bot, direct_host.id, frame(1)) == {"duplicate": True}
        bad = frame(3).model_copy(update={"session_id": "another-session"})
        assert agent_link.record_live_output(db, bot, direct_host.id, bad) == {"ignored": True}
        with pytest.raises(ValidationError):
            agent_link.LiveOutput(**{**frame(4).model_dump(), "kind": "reasoning"})
        for seq in range(3, 42):
            value = frame(seq).model_copy(update={"message_id": str(seq), "text": "x" * 32768})
            agent_link.record_live_output(db, bot, direct_host.id, value)
        live = control.public_run(db, run, detail=True)["live_messages"]
        assert len(live) <= 32 and sum(len(row["text"].encode()) for row in live) <= 131072
        assert live[-1]["sequence"] == 41
        assert not db.scalar(select(AgentTurn.id).where(AgentTurn.run_id == identity))
        assert run.turn_count == 0
        run.state = "completed"
        agent_retention.redact(db, run, actor={"kind": "human"}, attempts=True)
        assert not run.sync_state.get("live_messages") and not run.sync_state.get("live_sequences")


def test_explicit_native_answer_while_held_is_saved_and_does_not_unhold(direct_host):
    identity = running(direct_host)
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        question, _ = coordinator.record_question(db, OWNER, direct_host.id, agent_link.Question(
            run_id=identity, generation=1, provider="claude", provider_session_id="sess-1", surf=run.surf,
            incarnation=1, question_id="held-question", provider_request_id="request", durable=True, hold_until=0,
            items=[{"id": "scope", "text": "Which scope?"}]))
        send(db, identity, "A separate literal message")
        coordinator.answer(db, question, [{"item_id": "scope", "text": "Approved scope"}], actor=HUMAN)
        command = db.scalar(select(AgentCommand).where(AgentCommand.op_id == question.answer_op_id))
        assert command.status == "queued" and command.payload["origin"] == "human"
        assert automation.held(run) and question.status == "submitted"


def test_resume_reconciles_only_matching_host_release_snapshot_after_restart(direct_host):
    identity = running(direct_host)
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        _, _, command = send(db, identity)
        receipt(db, direct_host, run, command)
        resume(db, identity)
        revision = automation.state(run)["revision"]
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        observed = agent_link.SnapshotSurface(run_id=identity, generation=1, surf=run.surf,
            incarnation=run.incarnation, tool="claude", session_id="sess-1", state="idle",
            automation_hold=False, automation_hold_revision=revision - 1)
        coordinator.reconcile(db, OWNER, direct_host.id, [observed])
        assert automation.held(run)
        coordinator.reconcile(db, OWNER, direct_host.id, [observed.model_copy(update={"automation_hold_revision": revision})])
        assert not automation.held(run) and not automation.public(run)["releasing"]


@pytest.mark.parametrize("model_result", ["verdict", "retry"])
def test_inflight_judge_records_result_without_gap_or_retry_after_direct_hold(direct_host, model_result):
    from jarvis import agent_judge
    from test_agent_judge import criteria_run, job_for, verdict
    from test_agent_plan import turn

    identity = criteria_run(direct_host)
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        run.sync_state = {**run.sync_state, "execution_mode": "managed"}
    turn(identity, 1)
    calls = []
    def model(_evidence):
        calls.append(True)
        with session_scope() as db:
            send(db, identity, "A new instruction while the judge was running")
        if model_result == "retry":
            raise agent_judge.NonVerdict("invalid", "retry would be automatic")
        return verdict("not_met", "OBSOLETE MODEL GAP", criteria=(("c1", "not_met"),))
    agent_judge.process(job_for(identity), model=model)
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        assert automation.held(run) and run.state == "executing" and len(calls) == 1
        attempt = db.scalars(select(AgentAttempt).where(AgentAttempt.run_id == identity,
            AgentAttempt.kind == "judge")).first()
        assert attempt.result.get("automation_obsolete_revision") == 1
        assert not any("OBSOLETE MODEL GAP" in (row.payload or {}).get("text", "") for row in db.scalars(
            select(AgentCommand).where(AgentCommand.run_id == identity)))


def test_canonical_step_notes_remain_reported_progress_while_held(direct_host):
    from jarvis import agent_plan
    from jarvis.models import Task
    from test_agent_plan import call

    identity = running(direct_host)
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        task_row = db.get(Task, run.task_id)
        task_row.steps = [{"id": "step1", "title": "Implement", "status": "todo", "revision": 0}]
        task_id = task_row.id
        send(db, identity)
    result = call(agent_plan.task_step, agent_plan.StepInput, identity, task_id=task_id,
                  step_id="step1", status="doing", expected_revision=0, note="Input validation implemented; testing the receipt path")
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        reported = control.public_run(db, run, detail=True)["plan"]["steps"][0]
        assert reported == result["step"] and reported["status"] == "doing"
        assert reported["note"] and reported["updated_at"] and reported["by"] and reported["revision"] == 1
        assert automation.held(run)
        summary = control.public_run(db, run)
        assert "live_messages" not in summary and "message_deliveries" not in summary


def test_first_provider_ack_after_generic_handoff_outcome_updates_executed_input(direct_host):
    identity = running(direct_host)
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        body, _, command = send(db, identity)
        coordinator.on_outcome(db, command, "executed", result={"receipt": "handed_to_harness", "provider_ack": "pending"})
        assert receipt(db, direct_host, run, command, "provider_acknowledged") == {"saved": True}
        coordinator.on_outcome(db, command, "executed", result={"receipt": "handed_to_harness", "provider_ack": "pending"})
        assert command.outcome["result"]["receipt_stage"] == "provider_acknowledged"
        assert command.outcome["result"]["provider_ack"] == "confirmed"
        assert actions.lookup(db, OWNER, body.action_id)["delivery_stage"] == "provider_acknowledged"


def test_reconnect_supersedes_older_hold_without_failing_saved_human_messages(direct_host):
    identity = running(direct_host)
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        first, _, first_input = send(db, identity, "Offline message one")
        second, _, second_input = send(db, identity, "Offline message two")
        old_hold = db.scalars(select(AgentCommand).where(AgentCommand.run_id == identity,
            AgentCommand.kind == "automation.hold").order_by(AgentCommand.created_at)).first()
        coordinator.reconcile(db, OWNER, direct_host.id, [agent_link.SnapshotSurface(
            run_id=identity, generation=1, surf=run.surf, incarnation=1, tool="claude", session_id="sess-1", state="idle")])
        assert old_hold.status == "superseded"
        assert first_input.status == second_input.status == "queued"
        receipt(db, direct_host, run, first_input)
        receipt(db, direct_host, run, second_input)
        assert actions.lookup(db, OWNER, first.action_id)["status"] == "delivered"
        assert actions.lookup(db, OWNER, second.action_id)["delivery_stage"] == "handed_to_harness"


@pytest.mark.parametrize("spawn_fails", [False, True])
def test_explicit_resume_can_restore_suppressed_approved_worker_without_releasing_hold(direct_host, spawn_fails):
    from test_agent_hyperplan import WORKER, hyper, submit
    from test_agent_plan import turn
    with session_scope() as db:
        row = db.get(AgentHost, direct_host.id)
        row.caps = {**row.caps, "caps": [*row.caps["caps"], "spawn.roles"]}
    created, identity = hyper(direct_host, reviewer=None, worker=WORKER)
    submit(identity, created["id"])
    turn(identity, 1)
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        run.sync_state = {**run.sync_state, "execution_mode": "managed"}
        coordinator.approve_plan(db, run, actor=HUMAN)
        original = coordinator.run_spawn(db, run)
        coordinator.on_outcome(db, original, "rejected", "target_busy", {"limit": 1, "running": 1})
        coordinator.send_worker_spawn(db, run, coordinator.frozen(db, run))
        suppressed = coordinator.run_spawn(db, run)
        _, _, direct = send(db, identity, "Keep this instruction for the worker")
        assert suppressed.status == "superseded" and direct.status == "deferred"
        response = resume(db, identity)
        assert response["automation"]["held"] and response["automation"]["recovery"] == "worker_starting"
        replacement = coordinator.run_spawn(db, run)
        assert replacement.id != suppressed.id and replacement.payload["origin"] == "human" and replacement.status == "queued"
        assert automation.state(run)["revision"] == 1
        if spawn_fails:
            coordinator.on_outcome(db, replacement, "failed", "provider unavailable")
            assert automation.held(run) and run.state == "failed"
        else:
            worker = coordinator.handoff_surf(run)
            coordinator.attach(db, direct_host.id, identity, 1, "claude", "worker-restored", surf=worker, incarnation=1)
            agent_link.drain(db, db.get(AgentHost, direct_host.id), 1)
            assert direct.payload["surf"] == worker and automation.held(run)
            frame = agent_link.InputReceipt(run_id=identity, generation=1, surf=worker, incarnation=1,
                input_id=direct.op_id, session_id="worker-restored", stage="handed_to_harness", at=now())
            assert agent_link.record_input_receipt(db, SimpleNamespace(owner_id=OWNER), direct_host.id, frame) == {"saved": True}
            resume(db, identity)
            release = db.scalar(select(AgentCommand).where(AgentCommand.op_id == automation.state(run)["release_op_id"]))
            coordinator.on_outcome(db, release, "executed")
            assert not automation.held(run) and run.state == "executing"


def test_second_message_preserves_current_completed_review_for_one_incorporation(direct_host):
    from test_agent_hyperplan import hyper, submit, reviewer_up
    from test_agent_plan import turn
    from jarvis.session_models import AgentPlanReview
    with session_scope() as db:
        row = db.get(AgentHost, direct_host.id)
        row.caps = {**row.caps, "caps": [*row.caps["caps"], "spawn.roles"]}
    created, identity = hyper(direct_host)
    submit(identity, created["id"])
    turn(identity, 1)
    reviewer_up(identity)
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        run.sync_state = {**run.sync_state, "execution_mode": "managed"}
        _, _, first = send(db, identity, "Keep planning read-only")
        review = db.scalar(select(AgentPlanReview).where(AgentPlanReview.run_id == identity))
        review.verdict, review.summary = "revise", "Incorporate this current review"
        coordinator.finish_review(db, run, review)
        assert run.sync_state["automation_pending"]["kind"] == "review_done"
        _, _, second = send(db, identity, "Another planning note")
        assert run.sync_state["automation_pending"]["review_id"] == review.id
        receipt(db, direct_host, run, first)
        receipt(db, direct_host, run, second)
        resume(db, identity)
        release = db.scalar(select(AgentCommand).where(AgentCommand.op_id == automation.state(run)["release_op_id"]))
        coordinator.on_outcome(db, release, "executed")
        assert run.state == "planning" and not automation.held(run)
        critique = [row for row in db.scalars(select(AgentCommand).where(AgentCommand.run_id == identity,
            AgentCommand.kind == "input")) if "Incorporate this current review" in (row.payload or {}).get("text", "")]
        assert len(critique) == 1 and critique[0].status == "queued"
        coordinator.on_outcome(db, release, "executed")
        assert not run.sync_state.get("automation_pending")


def test_live_output_retains_distinct_messages_delivered_out_of_order(direct_host):
    identity = running(direct_host)
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        bot = SimpleNamespace(owner_id=OWNER)
        at = now()
        def frame(sequence, message_id, text):
            return agent_link.LiveOutput(run_id=identity, generation=run.generation, surf=run.surf,
                incarnation=run.incarnation, session_id="sess-1", turn_id="ongoing", message_id=message_id,
                sequence=sequence, at=at + timedelta(seconds=sequence), kind="commentary", text=text, complete=False)
        assert agent_link.record_live_output(db, bot, direct_host.id, frame(3, "second", "Second published message")) == {"saved": True}
        assert agent_link.record_live_output(db, bot, direct_host.id, frame(2, "first", "First published message")) == {"saved": True}
        assert [(row["message_id"], row["sequence"]) for row in run.sync_state["live_messages"]] == [("first", 2), ("second", 3)]
        assert agent_link.record_live_output(db, bot, direct_host.id, frame(1, "first", "Stale first snapshot")) == {"duplicate": True}
        assert agent_link.record_live_output(db, bot, direct_host.id, frame(4, "first", "Updated first snapshot")) == {"saved": True}
        assert agent_link.record_live_output(db, bot, direct_host.id, frame(3, "first", "Stale update")) == {"duplicate": True}
        assert {row["message_id"]: row["text"] for row in run.sync_state["live_messages"]} == {
            "first": "Updated first snapshot", "second": "Second published message"}


def test_live_output_eviction_floor_survives_restart_without_old_snapshot_churn(direct_host):
    identity = running(direct_host)
    at = now()
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        bot = SimpleNamespace(owner_id=OWNER)
        def frame(sequence, message_id=None):
            return agent_link.LiveOutput(run_id=identity, generation=1, surf=run.surf, incarnation=1,
                session_id="sess-1", turn_id="ongoing", message_id=message_id or str(sequence), sequence=sequence,
                at=at + timedelta(seconds=sequence), kind="commentary", text=str(sequence), complete=False)
        for sequence in range(1, 41):
            agent_link.record_live_output(db, bot, direct_host.id, frame(sequence))
        assert [row["sequence"] for row in run.sync_state["live_messages"]] == list(range(9, 41))
        assert run.sync_state["live_floors"]["sess-1"] == 8
        old = frame(1).model_copy(update={"at": at + timedelta(days=1)})
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        before = list(run.sync_state["live_messages"])
        assert agent_link.record_live_output(db, bot, direct_host.id, old) == {"duplicate": True}
        assert agent_link.record_live_output(db, bot, direct_host.id, frame(7, "different-old-id")) == {"duplicate": True}
        assert run.sync_state["live_messages"] == before
        assert agent_link.record_live_output(db, bot, direct_host.id, frame(42)) == {"saved": True}
        assert agent_link.record_live_output(db, bot, direct_host.id, frame(41)) == {"saved": True}
        assert [row["sequence"] for row in run.sync_state["live_messages"]] == list(range(11, 43))
        assert run.sync_state["live_floors"]["sess-1"] == 10
        run.state = "completed"
        agent_retention.redact(db, run, actor={"kind": "human"}, attempts=True)
        assert not run.sync_state.get("live_floors")


def test_retained_live_message_uses_own_sequence_below_eviction_floor(direct_host):
    identity = running(direct_host)
    at = now()
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        bot = SimpleNamespace(owner_id=OWNER)
        def publish(sequence, message_id, stamp, text="Published"):
            return agent_link.record_live_output(db, bot, direct_host.id, agent_link.LiveOutput(
                run_id=identity, generation=1, surf=run.surf, incarnation=1, session_id="sess-1",
                turn_id="ongoing", message_id=message_id, sequence=sequence, at=stamp,
                kind="commentary", text=text, complete=False))
        publish(2, "retained", at + timedelta(days=1))
        publish(50, "evicted", at)
        for sequence in range(51, 82):
            publish(sequence, str(sequence), at + timedelta(seconds=sequence))
        assert run.sync_state["live_floors"]["sess-1"] == 50
        assert publish(3, "retained", at + timedelta(days=1, seconds=1), "New retained snapshot") == {"saved": True}
        assert publish(2, "retained", at + timedelta(days=1, seconds=2), "Stale") == {"duplicate": True}
        assert publish(49, "old-unknown", at + timedelta(days=1, seconds=3)) == {"duplicate": True}
        assert next(row for row in run.sync_state["live_messages"] if row["message_id"] == "retained")["text"] == "New retained snapshot"
        assert len(run.sync_state["live_messages"]) == 32
