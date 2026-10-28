"""In-session agent tools (scope agents:session): read the task plan, tick steps.

The SessionStart context carries a short-lived run token; a call that presents it is bound to
that run (Actor run_session, verified). Without it the call falls back to the reporting tool and
session id, and the actor is marked unverified. submit_plan and task_interview arrive in stage D.
"""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from . import agent_sessions as sessions
from . import run_coordinator as coordinator
from .agent_actor import Actor
from .domain import PLAN_ID, DomainError, advisory, emit, encode, owned
from .models import Note, Task, now


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    run_token: str | None = Field(default=None, max_length=200)
    tool: sessions.Tool | None = None
    session_id: str | None = Field(default=None, pattern=sessions.SESSION_ID.pattern)


class PlanQuery(Input):
    task_id: UUID | None = None


class StepInput(Input):
    task_id: UUID
    step_id: str = Field(pattern=PLAN_ID)
    status: Literal["todo", "doing", "done", "skipped"]
    note: str | None = Field(default=None, max_length=2000)
    expected_revision: int = Field(ge=0)


class Later(BaseModel):
    model_config = ConfigDict(extra="allow")


def binding(db, bot, body):
    """(actor, run) for this call: run token first, then the reporting session, else an unbound bot."""
    run = coordinator.run_for_token(db, bot.owner_id, body.run_token)
    if run is not None:
        return Actor("run_session", credential_id=bot.id, run_id=run.id, generation=run.generation), run
    if body.run_token:
        raise DomainError("RUN_TOKEN_INVALID", "That run token expired or the run has finished.", 401)
    if body.tool and body.session_id:
        row = sessions.find(db, bot.owner_id, body.tool, body.session_id)
        run = coordinator.run_for_session(db, row, lock=False)
        if run is not None and run.state not in coordinator.TERMINAL:
            return Actor("run_session", credential_id=bot.id, run_id=run.id, generation=run.generation, verified=False), run
    return Actor("bot", credential_id=bot.id, verified=False), None


def task_plan(db, bot, body):
    actor, run = binding(db, bot, body)
    task_id = str(body.task_id) if body.task_id else (run.task_id if run else None)
    if not task_id:
        raise DomainError("INVALID_ARGUMENT", "Name the task_id, or pass the run token from your session context.")
    task = owned(db, Task, task_id, bot.owner_id)
    note = db.get(Note, task.prd_note_id) if task.prd_note_id else None
    result = {
        "task_id": task.id, "title": task.title, "status": task.status, "revision": task.revision,
        "plan_revision": task.plan_revision, "prd": note.content if note and not note.archived else task.notes,
        "steps": [{k: s.get(k) for k in ("id", "title", "status", "note", "task_id", "revision")} for s in task.steps or []],
        "success_criteria": task.success_criteria or [], "actor": actor.record(),
    }
    if run is not None and run.task_id == task.id:
        plan = coordinator.frozen(db, run)
        result["run"] = {"id": run.id, "state": run.state, "generation": run.generation, "branch": run.branch,
                         "plan_revision": run.plan_revision,
                         "frozen_criteria": plan.criteria if plan else [], "turn_count": run.turn_count,
                         "turn_cap": run.turn_cap}
    return encode(result)


def task_step(db, bot, body):
    """Per-step compare-and-set. A linked step is a checklist entry: ticking it never completes the child task."""
    actor, run = binding(db, bot, body)
    if run is not None and actor.verified and run.task_id != str(body.task_id):
        raise DomainError("RUN_NOT_BOUND", "This run works on a different task.", 409, data={"task_id": run.task_id})
    advisory(db, f"workspace:{bot.owner_id}")
    task = owned(db, Task, str(body.task_id), bot.owner_id, lock=True)
    steps = list(task.steps or [])
    index = next((i for i, s in enumerate(steps) if s.get("id") == body.step_id), None)
    if index is None:
        raise DomainError("NOT_FOUND", "No step with that id on this task.", 404)
    step = steps[index]
    if (step.get("revision") or 0) != body.expected_revision:
        raise DomainError("REVISION_CONFLICT", "This step changed; read the plan again.", 409,
                          data={"step": {k: step.get(k) for k in ("id", "status", "note", "revision")}})
    who = "agent:" + (actor.run_id or actor.credential_id or "unbound")
    steps[index] = {**step, "status": body.status, **({"note": body.note} if body.note is not None else {}),
                    "by": who if actor.verified else who + ":unverified", "updated_at": now().isoformat(),
                    "revision": (step.get("revision") or 0) + 1}
    task.steps = steps  # Step progress is not a plan change: plan_revision stays.
    task.revision += 1
    task.updated_at = now()
    emit(db, bot.owner_id, "task.changed", task.id, task.revision)
    if run is not None and run.task_id == task.id:
        run.last_progress_at = now()
        emit(db, bot.owner_id, "agent.changed", run.id)
    return encode({"task_id": task.id, "task_revision": task.revision, "step": steps[index], "actor": actor.record()})


def not_yet(name):
    raise DomainError("NOT_YET", f"{name} arrives with planning runs; use task_plan and task_step for now.", 501)
