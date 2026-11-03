"""Durable automation holds. Human input and ongoing work remain independent of the loop."""
from contextlib import contextmanager
from contextvars import ContextVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from .domain import DomainError, emit
from .models import Job, now
from .session_models import AgentAttempt, AgentCommand

authorized = ContextVar("agent_explicit_action", default=False)


@contextmanager
def explicit():
    token = authorized.set(True)
    try:
        yield
    finally:
        authorized.reset(token)


def state(run):
    return (run.sync_state or {}).get("automation") or {"held": False, "revision": 0}


def held(run):
    return bool(state(run).get("held"))


def public(run):
    value = state(run)
    return {"held": bool(value.get("held")), "revision": value.get("revision", 0),
            "reason": value.get("reason"), "held_at": value.get("held_at"),
            "releasing": bool(value.get("release_op_id")),
            **({"recovery": value["recovery"]} if value.get("recovery") else {})}


def deliveries(db, run):
    rows = list(db.scalars(select(AgentCommand).where(AgentCommand.run_id == run.id,
        AgentCommand.kind == "input").order_by(AgentCommand.created_at.desc()).limit(80)))
    result = []
    for row in reversed(rows):
        payload = row.payload or {}
        if payload.get("mode") != "direct":
            continue
        receipt = ((row.outcome or {}).get("result") or {})
        stage = receipt.get("receipt_stage") or receipt.get("receipt") or ("uncertain" if row.status == "uncertain" else
            "provider_acknowledged" if receipt.get("provider_ack") == "confirmed" else
            "handed_to_harness" if row.status == "executed" else "saved")
        result.append({"action_id": row.action_id, "op_id": row.op_id, "input_id": row.op_id,
                       "text": payload.get("text", ""), "status": row.status, "stage": stage,
                       "at": row.created_at.isoformat(), "updated_at": row.updated_at.isoformat(),
                       "session_id": receipt.get("session_id"), "turn_id": receipt.get("turn_id")})
    return result


def origin(actor=None):
    from .agent_actor import bound
    from .agent_receipts import action_context

    actor = actor or bound.get()
    return "human" if authorized.get() or action_context.get() or actor and actor.kind == "human" else "automation"


def blocked(run, actor=None):
    return held(run) and origin(actor) != "human"


def defer(run, kind, **facts):
    run.sync_state = {**(run.sync_state or {}), "automation_pending": {
        "kind": kind, "generation": run.generation, "turn_seq": run.turn_seq,
        "plan_revision": run.plan_revision, **facts}}


def supersede(db, run):
    """Never rewrite or replay an operation already handed to the host/provider."""
    from . import run_coordinator as coordinator

    for row in db.scalars(select(AgentCommand).where(AgentCommand.run_id == run.id,
            AgentCommand.status.in_(("queued", "deferred"))).with_for_update()):
        if (row.payload or {}).get("origin") == "automation" and row.kind in ("input", "spawn", "verify", "worktrees.scan"):
            coordinator.settle_command(db, row, "superseded", "automation held by a direct message")
    # A queued judge has not started any model work. Its existing job will observe the hold.
    for attempt in db.scalars(select(AgentAttempt).where(AgentAttempt.run_id == run.id,
            AgentAttempt.kind == "judge", AgentAttempt.status == "queued")):
        attempt.result = {**(attempt.result or {}), "automation_held": True}


def hold(db, run, *, actor):
    from . import run_coordinator as coordinator

    previous = state(run)
    value = {"held": True, "revision": previous.get("revision", 0) + 1,
             "reason": "direct_message", "held_at": now().isoformat()}
    previous_pending = (run.sync_state or {}).get("automation_pending") or {}
    run.sync_state = {**{key: item for key, item in (run.sync_state or {}).items() if key != "automation_pending"}, "automation": value,
                     **({"automation_pending": previous_pending} if previous_pending.get("kind") in ("review_done", "planning", "handoff") else {})}
    run.state_revision += 1
    run.updated_at = now()
    supersede(db, run)
    for attempt in db.scalars(select(AgentAttempt).where(AgentAttempt.run_id == run.id,
            AgentAttempt.status.in_(("queued", "running")), AgentAttempt.kind.in_(("judge", "verify")))):
        attempt.result = {**(attempt.result or {}), "automation_obsolete_revision": value["revision"]}
        if attempt.kind == "judge" and attempt.status == "queued":
            attempt.status = "superseded"
    coordinator.command(db, run, "automation.hold", {"held": True, "automation_hold": True,
        "automation_hold_revision": value["revision"], "origin": "human"})
    coordinator.note(run, "automation_held", revision=value["revision"], actor=actor.record())
    emit(db, run.owner_id, "agent.changed", run.id)


class ResumeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action_id: UUID
    expected_automation_revision: int = Field(ge=0)


def resume(db, owner, run_id, body, *, actor):
    from . import agent_control as control
    from . import run_coordinator as coordinator

    run = coordinator.load(db, owner, run_id)
    if actor.kind != "human" or not actor.verified:
        raise DomainError("HUMAN_REQUIRED", "Resuming automation requires an authenticated human.", 403)
    coordinator.ensure_eridani_control(run)
    value = state(run)
    if body.expected_automation_revision != value.get("revision", 0):
        raise DomainError("REVISION_CONFLICT", "Automation changed; review the run before resuming.", 409)
    if run.state in coordinator.TERMINAL or run.state == "stopping":
        raise DomainError("RUN_SETTLED", "This run is stopping or finished.", 409)
    if run.paused:
        raise DomainError("RUN_PAUSED", "Resume the paused run before resuming automation.", 409)
    if run.state_reason in ("wall_clock", "turn_cap") or coordinator.past_wall_clock(run, now()) or (
            run.turn_cap and run.turn_count >= run.turn_cap):
        raise DomainError("RUN_LIMIT", "Extend the configured run limit explicitly before resuming automation.", 409)
    if not held(run):
        return control.public_run(db, run, detail=True)
    if not coordinator.host_cap(db, run.host_id, "input.direct") or not coordinator.host_cap(db, run.host_id, "commands.durable"):
        raise DomainError("DIRECT_UNSUPPORTED", "Update Remotea before resuming automation for this held run.", 409)
    if value.get("release_op_id"):
        return control.public_run(db, run, detail=True)
    uncertain = db.scalar(select(AgentCommand.id).where(AgentCommand.run_id == run.id,
        AgentCommand.status == "uncertain", AgentCommand.kind.in_(("input", "answer", "spawn", "stop", "verify", "run.pause", "run.resume"))).limit(1))
    if uncertain:
        raise DomainError("DELIVERY_UNCERTAIN", "Reconcile uncertain delivery before resuming automation.", 409)
    pending = list(db.scalars(select(AgentCommand).where(AgentCommand.run_id == run.id,
        AgentCommand.kind.in_(("input", "answer")), AgentCommand.status.in_(("deferred", "queued", "sent", "acked", "uncertain")))))
    if pending:
        uncertain = any(row.status == "uncertain" for row in pending)
        if not uncertain and run.state == "handing_off" and not run.session_pk:
            plan = coordinator.frozen(db, run)
            spawn = coordinator.run_spawn(db, run)
            handoff = (run.sync_state or {}).get("handoff") or {}
            inflight = any(row.status in ("queued", "sent", "acked", "executed", "uncertain") for row in db.scalars(
                select(AgentCommand).where(AgentCommand.run_id == run.id, AgentCommand.kind == "spawn"))
                if (row.payload or {}).get("role") == "worker" and (row.payload or {}).get("attempt") == handoff.get("attempt"))
            if plan and plan.approval_status == "approved" and spawn and not inflight and spawn.status in (
                    "superseded", "rejected", "failed", "expired", "canceled"):
                with explicit():
                    coordinator.retry_handoff(db, run)
                run.sync_state = {**(run.sync_state or {}), "automation": {**value, "recovery": "worker_starting"}}
                coordinator.note(run, "automation_recovery", actor=actor.record(), revision=value["revision"])
                return control.public_run(db, run, detail=True)
        raise DomainError("DELIVERY_UNCERTAIN" if uncertain else "DELIVERY_PENDING",
                          "Reconcile message delivery before resuming automation.", 409)
    supersede(db, run)
    revision = value.get("revision", 0) + 1
    command = coordinator.command(db, run, "automation.hold", {
        "held": False, "automation_hold": False, "automation_hold_revision": revision, "origin": "human"})
    run.sync_state = {**(run.sync_state or {}), "automation": {**value, "revision": revision, "release_op_id": command.op_id, "recovery": None}}
    run.state_revision += 1
    run.updated_at = now()
    emit(db, run.owner_id, "agent.changed", run.id)
    return control.public_run(db, run, detail=True)


def released(db, run, command):
    """Only a matching, executed release can clear the effective hold, including after restart."""
    value = state(run)
    if not value.get("held") or value.get("release_op_id") != command.op_id or (
            command.payload or {}).get("automation_hold_revision") != value.get("revision"):
        return
    run.sync_state = {**(run.sync_state or {}), "automation": {**value, "held": False, "release_op_id": None, "recovery": None}}
    run.state_revision += 1
    run.updated_at = now()
    reconcile(db, run)
    emit(db, run.owner_id, "agent.changed", run.id)


def reconcile(db, run):
    """Continue from current evidence; never replay saved prompts or judge an earlier completed turn."""
    from . import agent_decisions
    from . import run_coordinator as coordinator

    if run.paused or held(run) or run.state in coordinator.TERMINAL or run.state == "stopping":
        return
    pending = (run.sync_state or {}).get("automation_pending") or {}
    review_continuation = pending.get("kind") == "review_done" and (
        pending.get("generation"), pending.get("plan_revision")) == (run.generation, run.plan_revision)
    if any(d["status"] in ("open", "pending", "submitted") and not (
            review_continuation and d["kind"] in ("plan_approval", "plan_review"))
            for d in agent_decisions.for_run(db, run)):
        return
    run.sync_state = {key: value for key, value in (run.sync_state or {}).items() if key != "automation_pending"}
    if pending and (pending.get("generation"), pending.get("turn_seq"), pending.get("plan_revision")) != (
            run.generation, run.turn_seq, run.plan_revision):
        pending = {}
    session = db.get(coordinator.AgentSession, run.session_pk) if run.session_pk else None
    busy = bool(session and session.state == "working")
    if busy and pending.get("kind") in ("verdict", "after_turn", "judge"):
        return
    if pending.get("kind") == "verdict":
        attempt = db.get(AgentAttempt, pending.get("attempt_id"))
        if attempt and coordinator.current(run, attempt) and not (attempt.result or {}).get("automation_obsolete_revision"):
            return coordinator.on_verdict(db, attempt, pending["final"])
    if pending.get("kind") == "review_done":
        from .agent_prompts import incorporate
        from .session_models import AgentPlanReview
        review = db.get(AgentPlanReview, pending.get("review_id"))
        if review and review.status == "done" and run.plan_pending and run.state in coordinator.PLANNING and not (
                (run.sync_state or {}).get("handoff") or {}).get("done"):
            if run.state == "plan_reviewing":
                coordinator.transition(db, run, "planning", reason="review_done")
            return coordinator.say(db, run, incorporate(review, run.plan_round, coordinator.rounds_of(run),
                run.plan_round >= coordinator.rounds_of(run) or coordinator.clean_approve(review)), "plan_critique")
    if pending.get("kind") == "after_turn" and run.state in ("executing", "verifying", "judging"):
        # A newer live provider turn means the old turn is no longer a safe basis for follow-up.
        return coordinator.after_turn(db, run)
    if run.state in coordinator.PLANNING or run.plan_pending:
        if busy:
            return
        plan = coordinator.frozen(db, run)
        if plan and plan.approval_status == "approved":
            return coordinator.begin_execution(db, run)
        return coordinator.settle_planning(db, run)
    if run.state == "handing_off":
        spawn = coordinator.run_spawn(db, run)
        if spawn and spawn.status in ("queued", "sent", "acked", "executed", "uncertain"):
            return
        return coordinator.retry_handoff(db, run)
    if run.state == "judging":
        if busy:
            return
        from .domain import enqueue_job
        for attempt in db.scalars(select(AgentAttempt).where(AgentAttempt.run_id == run.id,
                AgentAttempt.kind == "judge", AgentAttempt.status == "queued")):
            if coordinator.current(run, attempt):
                existing = db.scalar(select(Job.id).where(Job.kind == "agent_judge", Job.status.in_(("queued", "running")),
                    Job.payload["attempt"].astext == attempt.id).limit(1))
                if not existing and not (attempt.result or {}).get("automation_obsolete_revision"):
                    enqueue_job(db, run.owner_id, "agent_judge", {"attempt": attempt.id})

