"""Recover planning handoffs without inventing approval or replaying uncertain writes."""
from . import run_coordinator as coordinator
from .domain import DomainError, emit
from .models import now


def retry(db, run, *, actor, automatic=False):
    coordinator.ensure_eridani_control(run)
    state = run.sync_state or {}
    failure = state.get("coordination_failure") or {}
    if run.state not in ("needs_you", "interviewing", "planning", "replanning") or run.plan_pending or run.plan_revision:
        raise DomainError("RECOVERY_UNAVAILABLE", "Review the current run before retrying planning.", 409)
    if run.paused and state.get("operator_paused") or (state.get("automation") or {}).get("held"):
        raise DomainError("RUN_PAUSED", "Resume your automation hold or pause before retrying planning.", 409)
    if coordinator.open_interview(db, run):
        raise DomainError("INTERVIEW_OPEN", "Your interview is already saved; answer it instead.", 409)
    from .agent_decisions import for_run
    if any(d["status"] in ("open", "pending") for d in for_run(db, run)):
        raise DomainError("DECISION_OPEN", "Resolve the pending decision before retrying.", 409)
    if failure.get("outcome") == "uncertain":
        raise DomainError("DELIVERY_UNCONFIRMED", "Reconcile the saved result before retrying this submission.", 409)
    if state.get("automatic_pause") not in (None, "receipt_uncertain") or (state.get("host_attention") or {}).get("kind") == "permission_denied":
        raise DomainError("RECOVERY_UNAVAILABLE", "Review the current attention request first.", 409)
    if not run.surf or not coordinator.host_cap(db, run.host_id, "run.pause"):
        raise DomainError("HOST_TOO_OLD", "Connect an updated Remotea host before retrying.", 409)
    key = f"{run.generation}:{failure.get('request_id', 'legacy')}"
    attempts = dict(state.get("planning_recovery_attempts") or {})
    if automatic and (attempts.get(key) or state.get("planning_auto_generation") == run.generation):
        return False
    if automatic and (not any(t in str(failure.get("tool", "")) for t in ("task_plan", "task_interview", "submit_plan")) or failure.get("code") not in ("RUN_TOKEN_REQUIRED", "RUN_NOT_BOUND") or not failure.get("binding")):
        return False
    attempts[key] = attempts.get(key, 0) + 1
    run.sync_state = {**state, "planning_recovery_attempts": attempts,
                      "coordination_failure": {**failure, "recovering": True}, "operator_paused": False,
                      **({"planning_auto_generation": run.generation} if automatic else {})}
    run.sync_state = {k: v for k, v in run.sync_state.items() if k not in ("automatic_pause", "host_attention")}
    run.paused = False
    coordinator.resume_paused_timers(db, run)
    coordinator.transition(db, run, "interviewing", actor=actor, reason="planning_retry")
    coordinator.command(db, run, "run.resume", {"surf": run.surf, "incarnation": run.incarnation})
    if failure.get("code") in ("RUN_TOKEN_REQUIRED", "RUN_NOT_BOUND"):
        message = "Your previous planning submission was rejected and saved nothing. The run-scoped Eridani tool connection now supplies your run identity automatically. First read task_plan and verify actor.verified is true. Then retry the SAME prepared interview questions (including their IDs and options), or the same rejected plan submission with its original submission_id. Do not change code, invent answers, or approve anything. Stop after a confirmed saved interview or plan. If this retry fails, explain the error and stop."
    else:
        message = ("Your previous planning submission was rejected and saved no plan. First read task_plan. "
                   "Correct the server-reported validation error and resubmit the prepared plan. Preserve the operator's "
                   "interview answers and success criteria. A verify criterion must name the plan's selected verify_id. "
                   "Use a new submission_id if you change the submission body. Stop after a confirmed saved plan and "
                   "wait for human approval. Do not implement or approve anything.")
    coordinator.say(db, run, message, "planning_retry", actor=actor)
    run.last_progress_at = now()
    coordinator.note(run, "planning_retry", automatic=automatic, attempt=attempts[key], actor=actor.record())
    emit(db, run.owner_id, "agent.changed", run.id)
    return True
