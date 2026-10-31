"""In-session agent tools (scope agents:session): read the task plan, tick steps, interview, submit a plan.

The SessionStart context carries a short-lived run token; a call that presents it is bound to
that run (Actor run_session, verified). Without it the call falls back to the reporting tool and
session id, and the actor is marked unverified. task_interview and submit_plan are run-scoped:
they need the token and a run in its interview or planning phase.
"""

import re
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select

from . import agent_sessions as sessions
from . import migration_numbers as numbers
from . import run_coordinator as coordinator
from .agent_actor import Actor, acting
from .domain import PLAN_ID, DomainError, advisory, emit, encode, execute, owned
from .models import Note, Task, now
from .session_models import AgentHost, AgentInterview, AgentPlanReview, AgentRunPlan, ExecutionTarget


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
    # Proposing a change to a human-authored criterion (plan_proposals): the agent's reason, and drop to propose removal.
    because: str | None = Field(default=None, max_length=600)
    drop: bool = False


class VerifyInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    target_id: str = Field(min_length=1, max_length=100)
    verify_id: str = Field(pattern=r"^[A-Za-z0-9._-]{1,64}$")


class ReviewResolution(BaseModel):
    """The planner's decision on one reviewer finding (hyperplan): every blocking or major finding of the last review
    needs one before submit_plan accepts the next revision (REVIEW_UNRESOLVED)."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    finding_id: str = Field(min_length=1, max_length=64)
    decision: Literal["accepted", "rejected", "deferred"]
    note: str = Field(default="", max_length=1000)


class SubmitPlanInput(Input):
    task_id: UUID | None = None
    prd: str = Field(min_length=1, max_length=28_000)
    steps: list[PlanStepInput] = Field(default_factory=list, max_length=100)
    success_criteria: list[CriterionInput] = Field(min_length=1, max_length=50)
    verify: VerifyInput | None = None
    # What "done" means; omitted keeps the task's policy (default code_ready).
    completion_policy: Literal["code_ready", "pr_open", "pr_merged"] | None = None
    # The plan adds a schema migration: Eridani reserves its number at freeze (migration_numbers.py).
    migration: bool = False
    expected_plan_revision: int = Field(ge=0)
    # Hyperplan: answers to the last review's findings (stored on that review as `resolutions`).
    review_resolutions: list[ReviewResolution] | None = Field(default=None, max_length=100)


class ReviewFinding(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    id: str = Field(min_length=1, max_length=64)
    severity: Literal["blocking", "major", "minor"]
    area: str = Field(default="", max_length=200)
    issue: str = Field(min_length=1, max_length=2000)
    suggestion: str = Field(default="", max_length=2000)


class SubmitReviewInput(Input):
    """submit_plan_review (the reviewer's one call per round)."""
    verdict: Literal["approve", "revise"]
    findings: list[ReviewFinding] = Field(default_factory=list, max_length=50)
    questions_for_user: list[str] = Field(default_factory=list, max_length=12)
    summary: str = Field(default="", max_length=4000)

    @model_validator(mode="after")
    def distinct_ids(self):
        ids = [f.id for f in self.findings]
        if len(set(ids)) != len(ids):
            raise ValueError("Each finding needs its own id.")
        if any(len(q) > 1000 for q in self.questions_for_user):
            raise ValueError("Questions are too long.")
        return self


class PlanDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    id: str = Field(pattern=PLAN_ID)
    accept: bool
    text: str | None = Field(default=None, min_length=1, max_length=1000)  # The operator's own wording, if any.


class PlanReviewInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decisions: list[PlanDecision] = Field(default_factory=list, max_length=50)
    verify: Literal["accept", "keep"] = "keep"
    completion_policy: Literal["accept", "keep"] = "keep"
    # Hyperplan: the final-plan stop (needs_you(plan_approval)) takes approve, or revise with a note for the planner.
    action: Literal["approve", "revise"] | None = None
    note: str = Field(default="", max_length=4000)
    # Required: the decisions are about the proposals the reviewer saw (REVISION_CONFLICT if the run moved on).
    expected_revision: int = Field(ge=1)


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


REVIEWER = "plan_reviewer"  # Actor capability of a reviewer token: task_plan and submit_plan_review only.


def binding(db, bot, body):
    """(actor, run) for this call: run token first (a reviewer's token binds with capability plan_reviewer), then the
    reporting session, else an unbound bot."""
    run = coordinator.run_for_token(db, bot.owner_id, body.run_token)
    if run is not None:
        return Actor("run_session", credential_id=bot.id, run_id=run.id, generation=run.generation), run
    bound = coordinator.review_for_token(db, bot.owner_id, body.run_token)
    if bound is not None:
        _, run = bound
        return Actor("run_session", credential_id=bot.id, run_id=run.id, generation=run.generation, capability=REVIEWER), run
    if body.run_token:
        raise DomainError("RUN_TOKEN_INVALID", "That run token expired or the run has finished.", 401)
    if body.tool and body.session_id:
        row = sessions.find(db, bot.owner_id, body.tool, body.session_id)
        run = coordinator.run_for_session(db, row, lock=False)
        if run is not None and run.state not in coordinator.TERMINAL:
            return Actor("run_session", credential_id=bot.id, run_id=run.id, generation=run.generation, verified=False), run
    return Actor("bot", credential_id=bot.id, verified=False), None


def task_plan(db, bot, body):
    from .agent_link import gate

    gate(db, bot.owner_id)  # With remotea off, no in-session tool answers (stage F, L1).
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
        "success_criteria": task.success_criteria or [], "completion_policy": task.completion_policy,
        "actor": actor.record(),
    }
    if run is not None and run.task_id == task.id:
        plan = coordinator.frozen(db, run)
        result["run"] = {"id": run.id, "state": run.state, "generation": run.generation, "branch": run.branch,
                         "plan_revision": run.plan_revision,
                         "frozen_criteria": plan.criteria if plan else [], "turn_count": run.turn_count,
                         "turn_cap": run.turn_cap,
                         **({"migration": {k: plan.migration[k] for k in ("number", "dir")},
                             "migration_note": numbers.render(plan.migration)} if plan and plan.migration else {})}
    return encode(result)


def task_step(db, bot, body):
    """Per-step compare-and-set. A linked step is a checklist entry: ticking it never completes the child task."""
    from .agent_link import gate

    gate(db, bot.owner_id)
    actor, run = binding(db, bot, body)
    reviewer_refused(actor)
    if run is not None and actor.verified and run.task_id != str(body.task_id):
        raise DomainError("RUN_NOT_BOUND", "This run works on a different task.", 409, data={"task_id": run.task_id})
    if not (run is not None and actor.verified) and "tasks:write" not in bot.scopes:
        # Without the run's token only a key that may edit tasks anyway ticks steps (stage F, L1).
        raise DomainError("RUN_TOKEN_REQUIRED", "Pass the run_token from your Eridani session context.", 401)
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


def reviewer_refused(actor):
    """A reviewer token reads (task_plan) and reviews (submit_plan_review); it never changes the plan, the steps or
    the run (docs/AGENT_CONTROL.md, "Hyperplan and roles")."""
    if actor.capability == REVIEWER:
        raise DomainError("REVIEWER_SCOPE", "A plan reviewer's token can only call task_plan and submit_plan_review.", 403)


def run_scoped(db, bot, body, states):
    """task_interview and submit_plan: only the run's own session (run token), only in these phases."""
    from .agent_link import gate

    gate(db, bot.owner_id)
    actor, run = binding(db, bot, body)
    if run is None or not actor.verified:
        raise DomainError("RUN_TOKEN_REQUIRED", "Pass the run_token from your Eridani session context.", 401)
    reviewer_refused(actor)
    if body.task_id and str(body.task_id) != run.task_id:
        raise DomainError("RUN_NOT_BOUND", "This run works on a different task.", 409, data={"task_id": run.task_id})
    run = coordinator.load(db, run.owner_id, run.id)
    if run.state not in states:
        raise DomainError("PLAN_LOCKED", f"The run is {run.state}; ask for a re-plan to change the plan.", 409,
                          data={"state": run.state})
    return actor, run


def review_scoped(db, bot, body):
    """submit_plan_review: only a live reviewer's token, while the run is in plan_reviewing."""
    from .agent_link import gate

    gate(db, bot.owner_id)
    bound = coordinator.review_for_token(db, bot.owner_id, body.run_token)
    if bound is None:
        if body.run_token and coordinator.run_for_token(db, bot.owner_id, body.run_token) is not None:
            raise DomainError("REVIEWER_ONLY", "Only the plan reviewer's session calls submit_plan_review.", 403)
        raise DomainError("RUN_TOKEN_REQUIRED", "Pass the reviewer run_token from your Eridani session context.", 401)
    review, run = bound
    run = coordinator.load(db, run.owner_id, run.id)
    review = db.get(AgentPlanReview, review.id, with_for_update=True)
    if run.state != "plan_reviewing" or review.status not in ("spawning", "reviewing"):
        raise DomainError("REVIEW_CLOSED", f"This review round is over (the run is {run.state}).", 409,
                          data={"state": run.state, "review_status": review.status})
    return Actor("run_session", credential_id=bot.id, run_id=run.id, generation=run.generation, capability=REVIEWER), run, review


# ---------------------------------------------------------------- the interview


def task_interview(db, bot, body):
    """Questions for the operator; they arrive as one form on the remotea page. A hyperplan planner may also ask
    while planning (between review rounds): the run goes back to interviewing until the answers come."""
    actor, run = run_scoped(db, bot, body, ("interviewing", "planning"))
    if run.state == "planning":
        if not run.roles:
            raise DomainError("PLAN_LOCKED", "The run is planning; ask for a re-plan to change the plan.", 409,
                              data={"state": run.state})
        coordinator.transition(db, run, "interviewing", actor=actor, reason="interview")
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
    coordinator.say(db, run, prompts.planning(row, verify_ids(db, run)[1], round=run.plan_round, roles=run.roles),
                    "interview_answers", actor=actor)
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


POLICY_ORDER = ("code_ready", "pr_open", "pr_merged")


# A criterion that asks for test evidence: without a verify command the judge can never see it (run 75661a23).
EVIDENCE = re.compile(r"\b(tests?|test suite|suites?|pytest|vitest|ci|lint|typecheck|type-check|compiles?|builds?)\b", re.IGNORECASE)
# Verify ids a host commonly names for "the whole suite"; the first one offered wins, else the alphabetically first.
PREFERRED_VERIFY = ("all", "full", "tests", "test", "check", "ci")


def needs_evidence(criteria):
    """Does any criterion (dicts or CriterionInput) ask for tests, a suite, CI, lint or a build to pass?"""
    for c in criteria or []:
        kind = c.get("kind") if isinstance(c, dict) else getattr(c, "kind", None)
        text = c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "")
        if kind == "verify" or EVIDENCE.search(text or ""):
            return True
    return False


def pick_verify(offered, ready=None):
    """The verify id Eridani names by default. `ready` ({id: bool}, the host's hello `verify_status`) narrows the
    choice to the ids that are ready when at least one is; ids the host did not report count as ready. When none is
    ready the choice is today's (the preflight says why: agent_control.target_readiness)."""
    ready = ready or {}
    usable = {v for v in offered if ready.get(v) is not False} or set(offered)
    for name in PREFERRED_VERIFY:
        if name in usable:
            return name
    return sorted(usable)[0] if usable else None


def verify_ready(db, run):
    """{verify id: ready} for the run's execution target, as its host last advertised (empty for an older host)."""
    from .agent_control import run_verify_readiness

    return {k: v["ready"] for k, v in run_verify_readiness(db, run).items()}


def default_verify(db, run, criteria, verify):
    """When the plan has no verify but a criterion wants test evidence, Eridani names the target's verify command
    itself (`by: default`, not a human floor), so the judge gets a recorded result instead of the agent's claims.
    Returns (verify, proposal): the proposal, when the target offers several, lets the operator pick or decline."""
    if verify or not needs_evidence(criteria):
        return verify, None
    targets, offered = verify_ids(db, run)
    chosen = pick_verify(offered, verify_ready(db, run))
    if not chosen:
        return None, None
    picked = {"target_id": run.execution_target_id, "verify_id": chosen, "by": "default"}
    proposal = {"current": None, "proposed": {"target_id": run.execution_target_id, "verify_id": chosen},
                "options": sorted(offered), "because": "A success criterion asks for test evidence; without a verify "
                "command the judge cannot record a result."} if len(offered) > 1 else None
    return picked, proposal


def human_verify(task):
    """The task's verify when a human set it (the floor); agent- or default-chosen ones may be replaced."""
    return task.verify if task.verify and task.verify.get("by") not in ("agent", "default") else None


def plan_floor(task, criteria, verify, policy):
    """What a human set is a floor (stage F, M1, decision D2): the agent may add criteria, never remove,
    reword or weaken a human-authored one, clear or swap the verify, or lower the completion policy.
    Agent-authored criteria from an earlier plan are the agent's own and may be replaced."""
    kept = [c for c in task.success_criteria or [] if c.get("by") != "agent"]
    taken = {c.get("id") for c in kept}
    added = [{**c.model_dump(exclude_none=True, exclude={"because", "drop"}), "by": "agent"}
             for c in criteria if not c.id or c.id not in taken]
    for c in added:
        if c.get("id") in taken:
            c.pop("id")
    if human_verify(task):
        verify = task.verify
    current = task.completion_policy or "code_ready"
    if policy is None or POLICY_ORDER.index(policy) < POLICY_ORDER.index(current):
        policy = None  # Keep the task's own.
    return kept + added, verify, policy


def plan_proposals(task, criteria, verify, policy):
    """What the agent wants to change about what a human set. The floor (plan_floor) never applies these; they wait
    for the operator on the run page (review_plan). Only explicit changes count: a human criterion the agent sends
    back with different text, or with drop=true. Human criteria the agent simply does not list are kept, not
    proposed for removal, so an agent that only adds never pauses the run."""
    human = {c.get("id"): c for c in task.success_criteria or [] if c.get("by") != "agent" and c.get("id")}
    changes = []
    for given in criteria:
        current = human.get(given.id) if given.id else None
        if current is None:
            continue
        if given.drop:
            changes.append({"id": given.id, "change": "remove", "text": current.get("text", ""), "because": given.because or ""})
        elif given.text.strip() != (current.get("text") or "").strip():
            changes.append({"id": given.id, "change": "reword", "text": current.get("text", ""), "proposed": given.text.strip(),
                            "because": given.because or ""})
    result = {}
    if changes:
        result["criteria"] = changes
    if verify and human_verify(task) and verify.get("verify_id") != task.verify.get("verify_id"):
        result["verify"] = {"current": task.verify, "proposed": {k: v for k, v in verify.items() if k != "by"}}
    current_policy = task.completion_policy or "code_ready"
    if policy and POLICY_ORDER.index(policy) < POLICY_ORDER.index(current_policy):
        result["completion_policy"] = {"current": current_policy, "proposed": policy}
    return result or None


def submit_plan(db, bot, body):
    """Planning only: write the PRD note and the task's template plan, then freeze run plan revision N+1.
    Changes the agent proposes to human-authored criteria are stored beside the frozen plan, not applied
    (plan_proposals); the coordinator asks the operator before execution starts."""
    actor, run = run_scoped(db, bot, body, coordinator.PLAN_SUBMIT)
    task = owned(db, Task, run.task_id, run.owner_id, lock=True)
    if (task.plan_revision or 0) != body.expected_plan_revision:
        raise DomainError("REVISION_CONFLICT", "The task's plan changed; read it again with task_plan.", 409,
                          data={"plan_revision": task.plan_revision})
    last_review = resolve_findings(db, run, body.review_resolutions)
    targets, offered = verify_ids(db, run)
    verify = None
    if body.verify:
        if body.verify.target_id not in targets or body.verify.verify_id not in offered:
            raise DomainError("VERIFY_UNKNOWN", "Name a verify id this run's execution target offers.", 400,
                              data={"verify_ids": sorted(offered)})
        verify = {**body.verify.model_dump(), "by": "agent"}
    if human_verify(task):
        verify = task.verify  # The floor (see plan_floor); a verify criterion must name the human's verify.
    verify, default_proposal = default_verify(db, run, body.success_criteria, verify)
    for criterion in body.success_criteria:
        if criterion.kind == "verify" and (not verify or criterion.verify_id != verify["verify_id"]):
            raise DomainError("INVALID_ARGUMENT", "A verify criterion names the plan's verify_id.")
    kids = coordinator.children(db, task)
    steps = seeded_steps(db, task, body.steps, kids)
    open_row = coordinator.open_interview(db, run)
    if open_row is not None:
        open_row.status, open_row.updated_at = "superseded", now()  # The agent went ahead without the answers.
    agent_verify = {**body.verify.model_dump(), "by": "agent"} if body.verify else None
    proposals = plan_proposals(task, body.success_criteria, agent_verify, body.completion_policy)
    if default_proposal and not (proposals or {}).get("verify"):
        proposals = {**(proposals or {}), "verify": default_proposal}
    criteria, verify, policy = plan_floor(task, [c for c in body.success_criteria if not c.drop], verify, body.completion_policy)
    writer = Actor("run_session", credential_id=actor.credential_id, run_id=run.id, generation=run.generation,
                   capability="plan_write")
    with acting(writer):
        prd_id = save_prd(db, run, task, prd_content(body.prd, kids))
        db.refresh(task)
        execute(db, run.owner_id, f"submit-plan:{run.id}:{run.generation}:{task.revision}", "task.update", {
            "task_id": task.id, "expected_revision": task.revision, "prd_note_id": prd_id, "steps": steps,
            "success_criteria": criteria, "verify": verify, **({"completion_policy": policy} if policy else {})})
    db.refresh(task)
    revision = (run.plan_revision or 0) + 1
    note = db.get(Note, prd_id)
    previous = coordinator.frozen(db, run)
    if body.migration:
        migration = numbers.reserve(db, run)
    else:
        migration = None
        if previous is not None and previous.migration:
            numbers.release(db, previous.migration.get("reservation_id"))  # The new plan no longer adds one.
    db.add(AgentRunPlan(owner_id=run.owner_id, run_id=run.id, revision=revision,
                        task_plan_revision=task.plan_revision or 0, title=task.title, prd_text=note.content if note else "",
                        steps=list(task.steps or []), criteria=list(task.success_criteria or []), verify=task.verify,
                        completion_policy=task.completion_policy or "code_ready", required_children=[c.id for c in kids],
                        proposals=proposals, migration=migration))
    run.plan_revision, run.plan_pending, run.completion_policy = revision, True, task.completion_policy or "code_ready"
    if run.state in ("interviewing", "replanning"):
        coordinator.transition(db, run, "planning", actor=actor, reason="plan_submitted")
    coordinator.note(run, "plan_submitted", plan_revision=revision, actor=actor.record(),
                     **({"proposals": coordinator.proposal_summary(proposals)} if proposals else {}),
                     **({"review_round": last_review.round} if last_review is not None else {}))
    emit(db, run.owner_id, "agent.changed", run.id)
    if proposals:
        next_step = ("Stop now. The operator will accept or keep each change you proposed to their criteria; Eridani "
                     "continues after that.")
    elif coordinator.reviewer_role(run) and run.plan_round < coordinator.rounds_of(run) \
            and not coordinator.review_settled(last_review):
        next_step = "Stop now. A reviewer reads this revision next; its findings arrive as your next prompt."
    elif run.roles and run.roles.get("approve_final", True):
        next_step = "Stop now. The operator approves the final plan before the work starts."
    else:
        next_step = "Stop now. Eridani starts execution when this turn ends."
    return encode({"plan_revision": revision, "task_plan_revision": task.plan_revision, "prd_note_id": prd_id,
                   "steps": task.steps, "success_criteria": task.success_criteria, "verify": task.verify,
                   "completion_policy": task.completion_policy, "proposals": proposals,
                   **({"migration": {k: migration[k] for k in ("number", "dir")}} if migration else {}),
                   **({"migration_note": "No migrations folder is configured for this target, so no number was "
                                         "reserved; pick the next number after the chain head."}
                      if body.migration and not migration else {}),
                   "actor": actor.record(), "next": next_step})


def resolve_findings(db, run, resolutions):
    """Hyperplan: the planner's review_resolutions answer the last review's findings. Every blocking or major finding
    of a `revise` review needs one (REVIEW_UNRESOLVED); the decisions are stored on that review. Returns the review."""
    last = coordinator.last_review(db, run) if run.roles else None
    if last is None or last.status != "done":
        if resolutions:
            raise DomainError("INVALID_ARGUMENT", "There is no review to resolve findings for.")
        return None
    given = {r.finding_id: r for r in resolutions or []}
    ids = {f.get("id") for f in last.findings or [] if isinstance(f, dict)}
    unknown = sorted(set(given) - ids)
    if unknown:
        raise DomainError("INVALID_ARGUMENT", "Resolve the findings the reviewer raised.", data={"unknown": unknown})
    unresolved = [f["id"] for f in last.findings or [] if isinstance(f, dict) and f.get("severity") in ("blocking", "major")
                  and f.get("id") not in given]
    if unresolved and last.verdict == "revise" and last.plan_revision >= run.plan_revision:
        raise DomainError("REVIEW_UNRESOLVED", "Answer every blocking and major finding of the last review in "
                          "review_resolutions (accepted, rejected or deferred, with a note).", 409,
                          data={"round": last.round, "unresolved": unresolved})
    if resolutions:
        last.resolutions = [r.model_dump() for r in resolutions]
        last.updated_at = now()
    return last


def submit_plan_review(db, bot, body):
    """The reviewer's one call: its verdict and findings go on the review row; the round completes when its turn
    ends (run_coordinator.finish_review), which sends the planner the incorporate prompt."""
    actor, run, review = review_scoped(db, bot, body)
    if review.verdict is not None:
        raise DomainError("REVIEW_SUBMITTED", "This review was already recorded; stop now.", 409)
    review.verdict, review.via = body.verdict, "tool"
    review.findings = [f.model_dump() for f in body.findings]
    review.questions = list(body.questions_for_user)
    review.summary = body.summary
    review.updated_at = now()
    counts = coordinator.severity_counts(review)
    coordinator.note(run, "review_submitted", review_id=review.id, round=review.round, verdict=body.verdict, **counts)
    emit(db, run.owner_id, "agent.changed", run.id)
    return encode({"review_id": review.id, "round": review.round, "verdict": review.verdict, "counts": counts,
                   "actor": actor.record(), "next": "Stop now. Your review reaches the planner when this turn ends."})


def review_plan(db, owner, run_id, body, *, actor):
    """The operator's decisions on the agent's proposals: accepted changes are written to the task (a human edit,
    allowed by the gate), the plan is frozen again as the next revision without proposals, and execution starts.
    For a hyperplan run waiting in needs_you(plan_approval), `action` approves the final plan (execution or the
    worker handoff starts) or sends it back to the planner with a note."""
    from .domain import execute

    run = coordinator.load(db, owner, run_id)
    coordinator.check_revision(run, body.expected_revision)
    plan = coordinator.frozen(db, run)
    if run.state == "needs_you" and run.state_reason == "plan_approval":
        from .agent_control import public_run

        if body.action is None:
            raise DomainError("INVALID_ARGUMENT", "This run waits for the final plan's approval: pass action approve or revise.")
        if body.action == "approve":
            coordinator.approve_plan(db, run, actor=actor)
        else:
            coordinator.revise_plan(db, run, body.note, actor=actor)
        return public_run(db, run, detail=True)
    if body.action is not None:
        raise DomainError("NO_PLAN_APPROVAL", "This run is not waiting for the final plan's approval.", 409,
                          data={"state": run.state, "state_reason": run.state_reason})
    if run.state != "needs_you" or run.state_reason != "plan_review" or plan is None or not plan.proposals:
        raise DomainError("NO_PLAN_REVIEW", "This run is not waiting for a plan review.", 409, data={"state": run.state})
    task = owned(db, Task, run.task_id, run.owner_id, lock=True)
    proposals = plan.proposals
    decided = {d.id: d for d in body.decisions}
    accepted, kept, criteria = [], [], []
    for criterion in task.success_criteria or []:
        change = next((c for c in proposals.get("criteria") or [] if c["id"] == criterion.get("id")), None)
        decision = decided.get(criterion.get("id")) if change else None
        if change is None or decision is None or not decision.accept:
            criteria.append(criterion)
            if change is not None:
                kept.append(change["id"])
            continue
        accepted.append(change["id"])
        if change["change"] == "remove" and not decision.text:
            continue
        criteria.append({**criterion, "text": (decision.text or change.get("proposed") or criterion["text"]).strip()})
    changes = {"success_criteria": criteria}
    if proposals.get("verify") and body.verify == "accept":
        changes["verify"] = {**proposals["verify"]["proposed"], "by": "human"}
        accepted.append("verify")
    elif proposals.get("verify"):
        kept.append("verify")
        if proposals["verify"].get("current") is None:
            changes["verify"] = None  # Eridani's default pick was declined: no verification for this plan.
    if proposals.get("completion_policy") and body.completion_policy == "accept":
        changes["completion_policy"] = proposals["completion_policy"]["proposed"]
        accepted.append("completion_policy")
    elif proposals.get("completion_policy"):
        kept.append("completion_policy")
    with acting(Actor("human", credential_id=actor.credential_id, reason="plan_review")):
        execute(db, owner, f"plan-review:{run.id}:{plan.revision}", "task.update",
                {"task_id": task.id, "expected_revision": task.revision, **changes})
    db.refresh(task)
    revision = plan.revision + 1
    db.add(AgentRunPlan(owner_id=run.owner_id, run_id=run.id, revision=revision, task_plan_revision=task.plan_revision or 0,
                        title=task.title, prd_text=plan.prd_text, steps=list(task.steps or []),
                        criteria=list(task.success_criteria or []), verify=task.verify,
                        completion_policy=task.completion_policy or "code_ready", required_children=plan.required_children,
                        proposals=None, migration=plan.migration))
    run.plan_revision, run.plan_pending, run.completion_policy = revision, True, task.completion_policy or "code_ready"
    coordinator.note(run, "plan_reviewed", plan_revision=revision, accepted=accepted, kept=kept, actor=actor.record())
    preface = ("The operator reviewed your proposed changes: accepted " + (", ".join(accepted) or "none") +
               "; kept as written " + (", ".join(kept) or "none") + ". The plan below is what you are judged against.")
    coordinator.begin_execution(db, run, reason="plan_reviewed", preface=preface, actor=actor)
    from .agent_control import public_run

    return public_run(db, run, detail=True)


class SetVerifyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verify_id: str = Field(pattern=r"^[A-Za-z0-9._-]{1,64}$")
    expected_revision: int | None = Field(default=None, ge=1)
    action_id: UUID | None = None  # agent_receipts.once


VERIFY_SETTABLE = {"executing", "needs_you", "awaiting_approval", "awaiting_integration", "verifying", "judging"}


def set_verify(db, owner, run_id, body, *, actor):
    """The operator adds (or swaps) the plan's verify command mid-run without a re-plan: the task's verify is set
    as a human edit, the plan is frozen again as the next revision with nothing else changed, and a run that was
    waiting because no result could be recorded goes back to work so the next turn is verified."""
    from . import agent_prompts as prompts
    from .agent_control import public_run

    run = coordinator.load(db, owner, run_id)
    coordinator.check_revision(run, body.expected_revision)
    plan = coordinator.frozen(db, run)
    if run.state not in VERIFY_SETTABLE or plan is None:
        raise DomainError("RUN_SETTLED", f"A run that is {run.state} cannot change its verify command.", 409)
    targets, offered = verify_ids(db, run)
    if body.verify_id not in offered:
        raise DomainError("VERIFY_UNKNOWN", "Name a verify id this run's execution target offers.", 400,
                          data={"verify_ids": sorted(offered)})
    task = owned(db, Task, run.task_id, run.owner_id, lock=True)
    verify = {"target_id": run.execution_target_id, "verify_id": body.verify_id, "by": "human"}
    with acting(Actor("human", credential_id=actor.credential_id, reason="verify_set")):
        execute(db, owner, f"set-verify:{run.id}:{plan.revision}:{body.verify_id}", "task.update",
                {"task_id": task.id, "expected_revision": task.revision, "verify": verify})
    db.refresh(task)
    revision = plan.revision + 1
    db.add(AgentRunPlan(owner_id=run.owner_id, run_id=run.id, revision=revision, task_plan_revision=task.plan_revision or 0,
                        title=plan.title, prd_text=plan.prd_text, steps=list(task.steps or []), criteria=list(plan.criteria or []),
                        verify=verify, completion_policy=plan.completion_policy, required_children=plan.required_children,
                        proposals=None, migration=plan.migration))
    run.plan_revision = revision
    coordinator.note(run, "verify_set", plan_revision=revision, verify_id=body.verify_id, actor=actor.record())
    if run.state == "needs_you" and run.surf:
        coordinator.transition(db, run, "executing", actor=actor, reason="verify_set")
        coordinator.say(db, run, prompts.verify_set(body.verify_id), "verify_set", actor=actor)
    emit(db, run.owner_id, "agent.changed", run.id)
    return public_run(db, run, detail=True)


class EditedCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    id: str | None = Field(default=None, pattern=PLAN_ID)  # Omitted for a new criterion (task.update assigns one).
    text: str = Field(min_length=1, max_length=1000)
    kind: Literal["text", "verify"] = "text"


class SetCriteriaInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    criteria: list[EditedCriterion] = Field(min_length=1, max_length=50)
    # Also choose the verify command in the same revision (set_verify would resume a needs_you run at once).
    verify_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9._-]{1,64}$")
    # Required, like approve: the edit changes what the run is judged against, so it must be the state you reviewed.
    expected_revision: int = Field(ge=1)
    action_id: UUID | None = None  # agent_receipts.once


CRITERIA_EDITABLE = {"needs_you", "awaiting_approval"}


def set_criteria(db, owner, run_id, body, *, actor):
    """The operator edits the success criteria (and optionally the verify command) of a run that is parked for them
    (needs_you or awaiting_approval): a human edit of the task's gate fields, frozen as the next plan revision with
    nothing else changed. The run does not move; Judge again (agent_control.judge_again) checks the last turn
    against the new revision. An unchanged agent criterion stays the agent's; an edited or new one is the human's."""
    from .agent_control import public_run

    run = coordinator.load(db, owner, run_id)
    coordinator.check_revision(run, body.expected_revision)
    plan = coordinator.frozen(db, run)
    if run.state not in CRITERIA_EDITABLE or plan is None or run.state_reason == "plan_review":
        raise DomainError("RUN_NOT_PARKED", f"Criteria can be edited here while the run needs you or awaits approval "
                          f"(not during a plan review); this run is {run.state}.", 409, data={"state": run.state})
    verify = plan.verify
    if body.verify_id and body.verify_id != (plan.verify or {}).get("verify_id"):
        if body.verify_id not in verify_ids(db, run)[1]:
            raise DomainError("VERIFY_UNKNOWN", "Name a verify id this run's execution target offers.", 400,
                              data={"verify_ids": sorted(verify_ids(db, run)[1])})
        verify = {"target_id": run.execution_target_id, "verify_id": body.verify_id, "by": "human"}
    verify_id = (verify or {}).get("verify_id")
    previous = {c.get("id"): c for c in plan.criteria or []}
    criteria = []
    for edited in body.criteria:
        item = edited.model_dump(exclude_none=True)
        if item["kind"] == "verify":
            if not verify_id:
                raise DomainError("INVALID_ARGUMENT", "A test criterion needs a verify command on this run; choose one.")
            item["verify_id"] = verify_id  # A verify criterion is met by a fresh pass of the plan's verify command.
        old = previous.get(edited.id) if edited.id else None
        if old and old.get("by") == "agent" and all(old.get(k) == item.get(k) for k in ("text", "kind", "verify_id")):
            item["by"] = "agent"
        criteria.append(item)
    changes = {"success_criteria": criteria, **({"verify": verify} if verify is not plan.verify else {})}
    task = owned(db, Task, run.task_id, run.owner_id, lock=True)
    with acting(Actor("human", credential_id=actor.credential_id, reason="criteria_set")):
        execute(db, owner, f"set-criteria:{run.id}:{plan.revision}", "task.update",
                {"task_id": task.id, "expected_revision": task.revision, **changes})
    db.refresh(task)
    revision = plan.revision + 1
    db.add(AgentRunPlan(owner_id=run.owner_id, run_id=run.id, revision=revision, task_plan_revision=task.plan_revision or 0,
                        title=plan.title, prd_text=plan.prd_text, steps=list(task.steps or []),
                        criteria=list(task.success_criteria or []), verify=verify,
                        completion_policy=plan.completion_policy, required_children=plan.required_children,
                        proposals=None, migration=plan.migration))
    # The state did not change, but what an approver reviewed did: an Approve sent against the old criteria conflicts.
    run.plan_revision, run.state_revision = revision, run.state_revision + 1
    coordinator.note(run, "criteria_set", plan_revision=revision, criteria=[c.get("id") for c in task.success_criteria or []],
                     verify_id=verify_id, actor=actor.record())
    emit(db, run.owner_id, "agent.changed", run.id)
    return public_run(db, run, detail=True)
