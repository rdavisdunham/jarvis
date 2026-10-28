"""In-session agent tools (scope agents:session): read the task plan, tick steps, interview, submit a plan.

The SessionStart context carries a short-lived run token; a call that presents it is bound to
that run (Actor run_session, verified). Without it the call falls back to the reporting tool and
session id, and the actor is marked unverified. task_interview and submit_plan are run-scoped:
they need the token and a run in its interview or planning phase.
"""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select

from . import agent_sessions as sessions
from . import run_coordinator as coordinator
from .agent_actor import Actor, acting
from .domain import PLAN_ID, DomainError, advisory, emit, encode, execute, owned
from .models import Note, Task, now
from .session_models import AgentHost, AgentInterview, AgentRunPlan, ExecutionTarget


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


class Option(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    label: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=500)


class InterviewQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    id: str = Field(pattern=PLAN_ID)
    question: str = Field(min_length=1, max_length=1000)
    header: str = Field(default="", max_length=40)
    options: list[Option] = Field(default_factory=list, max_length=8)
    multi: bool = False
    free_text: bool = False


class InterviewInput(Input):
    task_id: UUID | None = None
    questions: list[InterviewQuestion] = Field(min_length=1, max_length=12)


class PlanStepInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    id: str | None = Field(default=None, pattern=PLAN_ID)
    title: str = Field(min_length=1, max_length=500)
    note: str = Field(default="", max_length=2000)
    task_id: str | None = Field(default=None, max_length=36)


class CriterionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    id: str | None = Field(default=None, pattern=PLAN_ID)
    text: str = Field(min_length=1, max_length=1000)
    kind: Literal["text", "verify"] = "text"
    verify_id: str | None = Field(default=None, max_length=64)


class VerifyInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    target_id: str = Field(min_length=1, max_length=100)
    verify_id: str = Field(pattern=r"^[A-Za-z0-9._-]{1,64}$")


class SubmitPlanInput(Input):
    task_id: UUID | None = None
    prd: str = Field(min_length=1, max_length=28_000)
    steps: list[PlanStepInput] = Field(default_factory=list, max_length=100)
    success_criteria: list[CriterionInput] = Field(min_length=1, max_length=50)
    verify: VerifyInput | None = None
    expected_plan_revision: int = Field(ge=0)


class InterviewAnswers(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # question id -> the chosen option label(s) or free text
    answers: dict[str, str | list[str]] = Field(max_length=12)

    @model_validator(mode="after")
    def bounded(self):
        for value in self.answers.values():
            items = value if isinstance(value, list) else [value]
            if len(items) > 8 or any(len(v) > 4000 for v in items):
                raise ValueError("Answers are too long.")
        return self


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


def run_scoped(db, bot, body, states):
    """task_interview and submit_plan: only the run's own session (run token), only in these phases."""
    actor, run = binding(db, bot, body)
    if run is None or not actor.verified:
        raise DomainError("RUN_TOKEN_REQUIRED", "Pass the run_token from your Eridani session context.", 401)
    if body.task_id and str(body.task_id) != run.task_id:
        raise DomainError("RUN_NOT_BOUND", "This run works on a different task.", 409, data={"task_id": run.task_id})
    run = coordinator.load(db, run.owner_id, run.id)
    if run.state not in states:
        raise DomainError("PLAN_LOCKED", f"The run is {run.state}; ask for a re-plan to change the plan.", 409,
                          data={"state": run.state})
    return actor, run


# ---------------------------------------------------------------- the interview


def task_interview(db, bot, body):
    """Questions for the operator; they arrive as one form on the remotea page."""
    actor, run = run_scoped(db, bot, body, ("interviewing",))
    if coordinator.open_interview(db, run):
        raise DomainError("INTERVIEW_OPEN", "Your questions are already with the operator; stop and wait.", 409)
    ids = [q.id for q in body.questions]
    if len(set(ids)) != len(ids):
        raise DomainError("INVALID_ARGUMENT", "Each question needs its own id.")
    questions = []
    for q in body.questions:
        item = q.model_dump()
        item["free_text"] = q.free_text or not q.options  # A question without options takes free text.
        questions.append(item)
    row = AgentInterview(owner_id=run.owner_id, run_id=run.id, generation=run.generation, questions=questions,
                         answers={}, status="open")
    db.add(row)
    db.flush()
    coordinator.note(run, "interview", interview_id=row.id, questions=len(questions))
    coordinator.notify(db, run, "agent_question", "An agent has questions before it plans",
                       " / ".join(q["question"] for q in questions)[:300], key=f"{run.id}:interview:{row.id}")
    emit(db, run.owner_id, "agent.changed", run.id)
    return encode({"interview_id": row.id, "status": "open", "actor": actor.record(),
                   "next": "Stop now. The answers arrive as your next prompt."})


def public_interview(row):
    return encode({"id": row.id, "run_id": row.run_id, "generation": row.generation, "questions": row.questions,
                   "answers": row.answers, "status": row.status, "answered_at": row.answered_at,
                   "answered_via": row.answered_via, "created_at": row.created_at})


def clean_answers(questions, given):
    clean = {}
    for question in questions:
        value = given.get(question["id"])
        values = [v.strip() for v in (value if isinstance(value, list) else [value] if value else []) if v and v.strip()]
        if not values:
            raise DomainError("INVALID_ARGUMENT", "Answer every question.", data={"question_id": question["id"]})
        labels = {o["label"] for o in question.get("options") or []}
        if len(values) > 1 and not question.get("multi"):
            raise DomainError("INVALID_ARGUMENT", "Choose one answer.", data={"question_id": question["id"]})
        if not question.get("free_text") and any(v not in labels for v in values):
            raise DomainError("INVALID_ARGUMENT", "Choose from the offered options.", data={"question_id": question["id"]})
        clean[question["id"]] = values if question.get("multi") else values[0]
    if set(given) - {q["id"] for q in questions}:
        raise DomainError("INVALID_ARGUMENT", "Answer the questions that were asked.")
    return clean


def answer_interview(db, owner, interview_id, body, *, actor):
    """Store the answers and send them as the planning prompt: interviewing → planning."""
    from . import agent_prompts as prompts

    row = db.scalar(select(AgentInterview).where(AgentInterview.id == interview_id, AgentInterview.owner_id == owner)
                    .with_for_update())
    if row is None:
        raise DomainError("NOT_FOUND", "That interview was not found.", 404)
    if row.status != "open":
        raise DomainError("INTERVIEW_CLOSED", "These questions were already answered.", 409, data={"status": row.status})
    run = coordinator.load(db, owner, row.run_id)
    if run.state != "interviewing" or row.generation != run.generation:
        raise DomainError("STALE_GENERATION", "The run moved on since these questions were asked.", 409,
                          data={"state": run.state})
    row.answers = clean_answers(row.questions, body.answers)
    row.status, row.answered_at, row.answered_via, row.updated_at = "answered", now(), actor.kind, now()
    coordinator.transition(db, run, "planning", actor=actor, reason="interview_answered")
    coordinator.command(db, run, "input", {"surf": run.surf, "incarnation": run.incarnation, "mode": "queue",
                                           "text": prompts.planning(row)}, source="eridani")
    return public_interview(row)


# ---------------------------------------------------------------- submit_plan


def seeded_steps(db, task, steps, kids):
    """The plan's steps: the agent's order, every direct child as a linked step, existing ids and statuses kept."""
    previous = {s.get("id"): s for s in task.steps or []}
    by_child = {s.get("task_id"): s for s in task.steps or [] if s.get("task_id")}
    child_ids = {c.id for c in kids}
    result, seen = [], set()
    for step in steps:
        if step.task_id and step.task_id not in child_ids:
            raise DomainError("INVALID_ARGUMENT", "A linked step must name a direct sub-task of this task.",
                              data={"task_id": step.task_id})
        sid = step.id or (by_child.get(step.task_id) or {}).get("id")
        old = previous.get(sid) or {}
        result.append({"id": sid, "title": step.title, "note": step.note, "task_id": step.task_id,
                       "status": old.get("status", "todo")})
        seen.add(step.task_id)
    for child in kids:
        if child.id not in seen:
            old = by_child.get(child.id) or {}
            result.append({"id": old.get("id"), "title": child.title[:500], "note": "", "task_id": child.id,
                           "status": old.get("status", "todo")})
    return [{k: v for k, v in s.items() if v is not None} for s in result]


def verify_ids(db, run):
    """(target names, verify ids) the run's execution target offers: its saved list and what the host advertised."""
    target = db.get(ExecutionTarget, run.execution_target_id) if run.execution_target_id else None
    if target is None:
        return set(), set()
    ids = {str(v.get("id") if isinstance(v, dict) else v) for v in target.verify or []}
    host = db.get(AgentHost, target.host_id) if target.host_id else None
    for item in (host.caps or {}).get("targets") or [] if host else []:
        if isinstance(item, dict) and item.get("id") == target.host_target:
            ids |= {str(v.get("id") if isinstance(v, dict) else v) for v in item.get("verify") or []}
    return {target.id, target.host_target}, ids


def prd_content(prd, kids):
    if not kids:
        return prd
    lines = [prd.rstrip(), "", "## Sub-tasks"] + [f"- {c.title} (task {c.id})" for c in kids]
    return "\n".join(lines)[:30_000]


def save_prd(db, run, task, content):
    """Create or replace the PRD note: title `PRD: <task>`, tag prd, linked to the task."""
    title = f"PRD: {task.title}"[:200]
    note = db.get(Note, task.prd_note_id) if task.prd_note_id else None
    if note is not None and note.owner_id == task.owner_id and not note.archived:
        tags = sorted({*(note.tags or []), "prd"})[:20]
        execute(db, run.owner_id, f"prd-update:{note.id}:{note.revision}", "note.update",
                {"note_id": note.id, "expected_revision": note.revision, "title": title, "content": content,
                 "tags": tags})
        return note.id
    created = execute(db, run.owner_id, f"prd-create:{run.id}:{run.generation}:{task.plan_revision}", "note.create",
                      {"title": title, "content": content, "tags": ["prd"], "task_ids": [task.id]})
    return created["data"]["id"]


def submit_plan(db, bot, body):
    """Planning only: write the PRD note and the task's template plan, then freeze run plan revision N+1."""
    actor, run = run_scoped(db, bot, body, coordinator.PLANNING)
    task = owned(db, Task, run.task_id, run.owner_id, lock=True)
    if (task.plan_revision or 0) != body.expected_plan_revision:
        raise DomainError("REVISION_CONFLICT", "The task's plan changed; read it again with task_plan.", 409,
                          data={"plan_revision": task.plan_revision})
    targets, offered = verify_ids(db, run)
    verify = None
    if body.verify:
        if body.verify.target_id not in targets or body.verify.verify_id not in offered:
            raise DomainError("VERIFY_UNKNOWN", "Name a verify id this run's execution target offers.", 400,
                              data={"verify_ids": sorted(offered)})
        verify = body.verify.model_dump()
    for criterion in body.success_criteria:
        if criterion.kind == "verify" and (not verify or criterion.verify_id != verify["verify_id"]):
            raise DomainError("INVALID_ARGUMENT", "A verify criterion names the plan's verify_id.")
    kids = coordinator.children(db, task)
    steps = seeded_steps(db, task, body.steps, kids)
    open_row = coordinator.open_interview(db, run)
    if open_row is not None:
        open_row.status, open_row.updated_at = "superseded", now()  # The agent went ahead without the answers.
    with acting(coordinator.judge_actor(run)):
        prd_id = save_prd(db, run, task, prd_content(body.prd, kids))
        db.refresh(task)
        execute(db, run.owner_id, f"submit-plan:{run.id}:{run.generation}:{task.revision}", "task.update", {
            "task_id": task.id, "expected_revision": task.revision, "prd_note_id": prd_id, "steps": steps,
            "success_criteria": [c.model_dump(exclude_none=True) for c in body.success_criteria], "verify": verify})
    db.refresh(task)
    revision = (run.plan_revision or 0) + 1
    note = db.get(Note, prd_id)
    db.add(AgentRunPlan(owner_id=run.owner_id, run_id=run.id, revision=revision,
                        task_plan_revision=task.plan_revision or 0, title=task.title, prd_text=note.content if note else "",
                        steps=list(task.steps or []), criteria=list(task.success_criteria or []), verify=task.verify,
                        required_children=[c.id for c in kids]))
    run.plan_revision, run.plan_pending = revision, True
    if run.state in ("interviewing", "replanning"):
        coordinator.transition(db, run, "planning", actor=actor, reason="plan_submitted")
    coordinator.note(run, "plan_submitted", plan_revision=revision, actor=actor.record())
    emit(db, run.owner_id, "agent.changed", run.id)
    return encode({"plan_revision": revision, "task_plan_revision": task.plan_revision, "prd_note_id": prd_id,
                   "steps": task.steps, "success_criteria": task.success_criteria, "verify": task.verify,
                   "actor": actor.record(), "next": "Stop now. Eridani starts execution when this turn ends."})
