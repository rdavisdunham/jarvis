"""Human decisions over the existing domain rows, with atomic, stable action receipts.

No second decision table: the original question/interview/control/plan is the source of truth.
An answer being saved and an adapter confirming consumption are deliberately separate facts.
"""
import hashlib
from typing import Literal
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from . import agent_receipts as actions
from . import run_coordinator as coordinator
from .domain import DomainError, advisory, encode
from .models import now
from .session_models import (
    AgentActionReceipt,
    AgentCommand,
    AgentInterview,
    AgentPendingControl,
    AgentQuestion,
    AgentRunPlan,
)


class RespondInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action_id: UUID
    expected_revision: int = Field(ge=1)
    action: Literal["approve", "revise", "answer", "confirm", "decline", "review"]
    answers: list[dict] | dict[str, str | list[str]] | None = None
    text: str = Field(default="", max_length=4000)
    execution_target_id: str | None = Field(default=None, max_length=120)
    decisions: list[dict] = Field(default_factory=list, max_length=50)
    verify: Literal["accept", "keep"] = "keep"
    completion_policy: Literal["accept", "keep"] = "keep"


class PauseInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action_id: UUID
    expected_revision: int = Field(ge=1)


def present(db, row, kind, run=None):
    from . import agent_control as control
    from . import agent_plan
    if kind == "final_acceptance":
        return encode({"id": f"acceptance:{row.id}:{row.plan_revision}:{row.turn_seq}", "kind": kind,
                       "run_id": row.id, "generation": row.generation, "plan_revision": row.plan_revision,
                       "revision": row.state_revision, "status": "pending" if row.state == "awaiting_approval" else "superseded",
                       "payload": {"summary": row.summary}, "actions": ["approve"]})
    if kind == "plan_approval":
        current = bool(run and run.plan_revision == row.revision and run.plan_pending)
        status = row.approval_status if current or row.approval_status != "pending" else "superseded"
        if run and (run.state in coordinator.TERMINAL or run.state == "stopping"):
            status = "superseded" if status == "pending" else status
        return encode({"id": f"plan:{row.run_id}:{row.revision}", "kind": "plan_review" if row.proposals else kind,
                       "run_id": row.run_id, "generation": run.generation if run else None,
                       "plan_revision": row.revision, "revision": row.revision, "status": status,
                       "payload": {"revision": row.revision, "title": row.title, "prd_text": row.prd_text,
                                   "steps": row.steps, "criteria": row.criteria, "verify": row.verify,
                                   "proposals": row.proposals, "completion_policy": row.completion_policy},
                       "actions": ["review"] if row.proposals else ["approve", "revise"]})
    payload = (control.public_question(row) if kind == "question" else agent_plan.public_interview(row)
               if kind == "interview" else {**control.public_pending(row), **control.readiness(db, row.owner_id, row)})
    status = row.status
    if run and kind != "control" and (row.generation != run.generation or row.plan_revision != run.plan_revision):
        status = "superseded"
    if run and (run.state in coordinator.TERMINAL or run.state == "stopping") and status in ("open", "pending"):
        status = "superseded"
    op_id = getattr(row, "answer_op_id", None)
    command = db.scalar(select(AgentCommand).where(AgentCommand.owner_id == row.owner_id,
                                                  AgentCommand.op_id == op_id)) if op_id else None
    return encode({"id": row.id, "kind": kind, "run_id": row.run_id,
                   "generation": getattr(row, "generation", None),
                   "plan_revision": getattr(row, "plan_revision", run.plan_revision if run else None), "revision": 1,
                   "status": status, "payload": payload,
                   "delivery": {"status": command.status, "op_id": op_id} if command else None,
                   "actions": ["confirm", "decline"] if kind == "control" else ["answer"]})


def for_run(db, run):
    result = []
    if run.state == "awaiting_approval":
        result.append(present(db, run, "final_acceptance", run))
    for cls, kind, statuses in ((AgentQuestion, "question", ("open", "submitted")),
                                (AgentInterview, "interview", ("open", "answered")),
                                (AgentPendingControl, "control", ("pending",))):
        for row in db.scalars(select(cls).where(cls.run_id == run.id, cls.owner_id == run.owner_id,
                                                cls.status.in_(statuses)).order_by(cls.created_at)):
            decision = present(db, row, kind, run)
            if decision["status"] != "superseded":
                result.append(decision)
    plan = coordinator.frozen(db, run)
    if plan and plan.approval_status == "pending" and run.plan_pending:
        result.append(present(db, plan, "plan_approval", run))
    return result


def load(db, owner, identity):
    if identity.startswith("acceptance:"):
        try:
            _, run_id, revision, turn_seq = identity.split(":")
            run = coordinator.load(db, owner, run_id)
            if int(revision) != run.plan_revision or int(turn_seq) != run.turn_seq:
                raise ValueError()
        except ValueError:
            raise DomainError("DECISION_CLOSED", "This acceptance decision was superseded.", 409) from None
        return run, "final_acceptance", run
    if identity.startswith("plan:"):
        try:
            _, run_id, revision = identity.split(":")
            revision = int(revision)
        except ValueError:
            raise DomainError("NOT_FOUND", "That decision was not found.", 404) from None
        run = coordinator.load(db, owner, run_id)
        row = db.scalar(select(AgentRunPlan).where(AgentRunPlan.run_id == run_id,
                                                  AgentRunPlan.revision == revision).with_for_update())
        if row:
            return row, "plan_approval", run
    else:
        for cls, kind in ((AgentQuestion, "question"), (AgentInterview, "interview"), (AgentPendingControl, "control")):
            row = db.scalar(select(cls).where(cls.id == identity, cls.owner_id == owner).with_for_update())
            if row:
                run = coordinator.load(db, owner, row.run_id) if row.run_id else None
                return row, kind, run
    raise DomainError("NOT_FOUND", "That decision was not found.", 404)


def respond(db, owner, identity, body, *, actor, journal=True):
    from . import agent_control as control
    from . import agent_plan
    # Serialize before taking domain row locks; recover a receipt even after the decision was superseded.
    kind_key = "decision:" + hashlib.sha256(identity.encode()).hexdigest()[:20]
    scope_id = str(uuid5(NAMESPACE_URL, "eridani-decision:" + identity))
    if journal:
        advisory(db, f"agent-action:{owner}:{body.action_id}")
        previous = db.get(AgentActionReceipt, (owner, str(body.action_id)))
        if previous:
            if previous.kind != kind_key or previous.digest != actions.digest(kind_key, scope_id, body):
                raise DomainError("ACTION_MISMATCH", "This action id already belongs to a different request.", 409)
            return {**previous.result, "receipt": actions.lookup(db, owner, body.action_id)}
    def act():
        row, kind, run = load(db, owner, identity)
        decision = present(db, row, kind, run)
        if decision["revision"] != body.expected_revision:
            raise DomainError("REVISION_CONFLICT", "This decision changed; review it again.", 409)
        if decision["status"] not in ("open", "pending"):
            raise DomainError("DECISION_CLOSED", "This decision is already resolved or superseded.", 409)
        if body.action not in decision["actions"]:
            raise DomainError("INVALID_ARGUMENT", "Choose an action offered by this decision.")
        if kind == "final_acceptance":
            result = control.approve(db, owner, run.id, control.ApproveInput(expected_revision=run.state_revision), actor=actor)
        elif kind == "plan_approval":
            if run is None or run.plan_revision != row.revision:
                raise DomainError("STALE_GENERATION", "A newer plan superseded this decision.", 409)
            if body.action == "review":
                result = agent_plan.review_plan(db, owner, run.id, agent_plan.PlanReviewInput(
                    expected_revision=run.state_revision, decisions=body.decisions, verify=body.verify,
                    completion_policy=body.completion_policy), actor=actor)
            else:
                if body.action == "approve":
                    coordinator.approve_plan(db, run, actor=actor)
                else:
                    coordinator.revise_plan(db, run, body.text, actor=actor)
                result = control.public_run(db, run, detail=True)
        elif kind == "question":
            if not isinstance(body.answers, list):
                raise DomainError("INVALID_ARGUMENT", "Answer this question with the offered answer items.")
            result = control.answer_question(db, owner, row.id, control.AnswerInput(answers=body.answers), actor=actor)
        elif kind == "interview":
            if not isinstance(body.answers, dict):
                raise DomainError("INVALID_ARGUMENT", "Answer this interview by question id.")
            result = agent_plan.answer_interview(db, owner, row.id, agent_plan.InterviewAnswers(answers=body.answers), actor=actor)
        else:
            result = control.decide(db, owner, row.id, confirm=body.action == "confirm", via="app", actor=actor,
                                    execution_target_id=body.execution_target_id)
        return {"action_id": str(body.action_id), "decision": present(db, row, kind, run), "result": result}
    # Include decision identity in the digest without requiring a synthetic run.
    if not journal:
        return act()
    row, _kind, run = load(db, owner, identity)
    result = actions.once(db, owner, run.id if run else row.id, kind_key, body, act, scope_id=scope_id)
    return {**result, "receipt": actions.lookup(db, owner, body.action_id)}


def set_paused(db, owner, run_id, body, paused, *, actor):
    from .agent_control import public_run
    run = coordinator.load(db, owner, run_id)
    coordinator.ensure_eridani_control(run)
    coordinator.check_revision(run, body.expected_revision)
    if run.state in coordinator.TERMINAL or run.state == "stopping":
        raise DomainError("RUN_SETTLED", "This run is stopping or finished.", 409)
    if not coordinator.host_cap(db, run.host_id, "run.pause"):
        raise DomainError("HOST_TOO_OLD", "Update Remotea to pause and resume runs.", 409)
    automatic = (run.sync_state or {}).get("automatic_pause")
    if not paused and automatic:
        raise DomainError(automatic.upper(), coordinator.REASONS[automatic], 409)
    state = run.sync_state or {}
    run.sync_state = {**state, "operator_paused": paused,
                      **({"paused_since": state.get("paused_since") or now().isoformat()} if paused else {})}
    if not paused:
        coordinator.resume_paused_timers(db, run)
    session = db.get(coordinator.AgentSession, run.session_pk) if run.session_pk else None
    if session and session.surf == run.surf:
        run.incarnation = max(run.incarnation, session.incarnation)
    run.paused = paused
    coordinator.sync_wait(db, run)
    run.state_revision += 1
    run.updated_at = now()
    coordinator.note(run, "paused" if paused else "resumed", actor=actor.record())
    if run.surf:
        coordinator.command(db, run, "run.pause" if paused else "run.resume",
                            {"surf": run.surf, "incarnation": run.incarnation})
    for review in coordinator.live_reviews(db, run):
        if review.surf:
            coordinator.command(db, run, "run.pause" if paused else "run.resume",
                                {"surf": review.surf, "incarnation": review.incarnation, "role": "reviewer"})
    if not paused:
        plan = coordinator.frozen(db, run)
        if run.plan_pending and plan and plan.approval_status == "approved":
            coordinator.begin_execution(db, run, reason="plan_approved", actor=actor)
        else:
            coordinator.continue_safely(db, run, "operator_resumed")
    return public_run(db, run, detail=True)
