"""The command service for agent runs: start, message/steer, stop, approve, answer and execution targets.

Browser routes (remotea_routes.py) and, from stage E, voice call these; each authenticates the actor,
resolves the owned task and host, checks the expected revision and records the state change plus
its outbound command in one transaction (run_coordinator.py owns the transitions).
"""

import re

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from . import agent_link
from . import agent_sessions as sessions
from . import run_coordinator as coordinator
from .agent_actor import Actor, acting
from .config import get_settings
from .domain import DomainError, emit, encode, owned
from .models import Note, StructureRecord, Task, now
from .session_models import (
    TERMINAL_RUN_STATES,
    AgentAttempt,
    AgentCommand,
    AgentHost,
    AgentInterview,
    AgentQuestion,
    AgentRun,
    AgentRunPlan,
    AgentSession,
    AgentSessionRule,
    AgentTurn,
    ExecutionTarget,
)

HARNESSES = {"claude": "opus", "codex": "gpt-5.6-sol"}
DISABLED_HARNESSES = {"opencode": "OpenCode control stays off until its adapter is tested."}
MODEL = r"^[A-Za-z0-9][A-Za-z0-9._:/@\[\]-]{0,79}$"
BRANCH = re.compile(r"^(?!-)(?!.*\.\.)(?!.*//)(?!.*\.lock$)[A-Za-z0-9._/-]{1,200}$")
PROMPT_CHARS = 32_000


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class StartInput(Input):
    task_id: str = Field(max_length=36)
    harness: str = Field(default="claude", max_length=20)
    model: str | None = Field(default=None, pattern=MODEL)
    execution_target_id: str | None = Field(default=None, max_length=36)
    # None: interview when the task has no PRD. False: run on the plan the task already has (stage C's manual run).
    interview: bool | None = None


class ReplanInput(Input):
    reason: str = Field(default="", max_length=2000)
    expected_revision: int | None = Field(default=None, ge=1)


class MessageInput(Input):
    text: str = Field(min_length=1, max_length=PROMPT_CHARS)
    mode: str = Field(default="queue", pattern=r"^(queue|steer)$")
    expected_revision: int | None = Field(default=None, ge=1)


class StopInput(Input):
    mode: str = Field(default="interrupt", pattern=r"^(interrupt|kill)$")
    expected_revision: int | None = Field(default=None, ge=1)


class ApproveInput(Input):
    expected_revision: int | None = Field(default=None, ge=1)


class AnswerItem(Input):
    item_id: str = Field(min_length=1, max_length=64)
    choice_ids: list[str] = Field(default_factory=list, max_length=32)
    text: str = Field(default="", max_length=4000)


class AnswerInput(Input):
    answers: list[AnswerItem] = Field(min_length=1, max_length=16)


class TargetInput(Input):
    host_id: str = Field(max_length=36)
    host_target: str = Field(pattern=r"^[A-Za-z0-9._-]{1,64}$")
    repo: str = Field(default="", max_length=1000)
    worktree_root: str = Field(default="", max_length=1000)
    name: str = Field(default="", max_length=200)
    # Save the choice for this task, or once for a record in the task's chain (its client or project).
    task_id: str | None = Field(default=None, max_length=36)
    record_id: str | None = Field(default=None, max_length=36)


# ---------------------------------------------------------------- execution targets


def chain(db, owner, task):
    """The task's record and its ancestors, nearest first."""
    from .structure import parent_chain

    record_id = sessions.record_for_task(db, owner, task.id)
    record = db.get(StructureRecord, record_id) if record_id else None
    return [record, *parent_chain(db, owner, record.parent_id)] if record else []


def public_target(target):
    return encode({"id": target.id, "host_id": target.host_id, "host_target": target.host_target, "name": target.name,
                   "repo": target.repo, "worktree_root": target.worktree_root, "record_id": target.record_id})


def advertised(db, owner):
    """Targets hosts listed in hello (optional on the wire); the picker offers them."""
    found = []
    for host in db.scalars(select(AgentHost).where(AgentHost.owner_id == owner)):
        for item in (host.caps or {}).get("targets") or []:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                found.append({"host_id": host.id, "host_name": host.name, "host_target": item["id"][:64],
                              "repo": str(item.get("repo") or "")[:1000],
                              "worktree_root": str(item.get("worktree_root") or item.get("worktreeRoot") or "")[:1000]})
    return found


def resolve_target(db, owner, task, explicit=None):
    """Explicit → the task's own target → a saved record selection → folder rules → NEEDS_TARGET."""
    if explicit:
        return owned(db, ExecutionTarget, explicit, owner)
    if task.execution_target_id:
        return owned(db, ExecutionTarget, task.execution_target_id, owner)
    records = chain(db, owner, task)
    ids = [r.id for r in records]
    if ids:
        saved = {t.record_id: t for t in db.scalars(select(ExecutionTarget).where(
            ExecutionTarget.owner_id == owner, ExecutionTarget.record_id.in_(ids)))}
        for record in records:
            if record.id in saved:
                return saved[record.id]
    targets = list(db.scalars(select(ExecutionTarget).where(ExecutionTarget.owner_id == owner)))
    rules = list(db.scalars(select(AgentSessionRule).where(AgentSessionRule.owner_id == owner,
                                                           AgentSessionRule.record_id.in_(ids or [""]))))
    candidates = []
    for record in records:
        prefixes = [r.path_prefix for r in rules if r.record_id == record.id]
        matches = {}
        for target in targets:
            repo = sessions.normalize_path(target.repo)
            if repo and any(sessions.under(repo, p) or sessions.under(p, repo) for p in prefixes):
                matches.setdefault((target.host_id, target.host_target, repo), target)
        if len(matches) == 1:
            return next(iter(matches.values()))
        if matches:
            candidates = list(matches.values())
            break
    raise DomainError("NEEDS_TARGET", "Choose where this task's agent should run. The choice is saved.", 409, data={
        "task_id": task.id,
        "candidates": [public_target(t) for t in candidates or targets],
        "advertised": advertised(db, owner),
        "records": [{"id": r.id, "title": r.title} for r in records],
    })


def save_target(db, owner, body):
    host = owned(db, AgentHost, body.host_id, owner)
    if body.record_id:
        owned(db, StructureRecord, body.record_id, owner)
    match = next((a for a in advertised(db, owner) if a["host_id"] == host.id and a["host_target"] == body.host_target), {})
    repo = sessions.normalize_path(body.repo or match.get("repo", ""))
    query = select(ExecutionTarget).where(ExecutionTarget.owner_id == owner, ExecutionTarget.host_id == host.id,
                                          ExecutionTarget.host_target == body.host_target)
    query = query.where(ExecutionTarget.record_id == body.record_id) if body.record_id else \
        query.where(ExecutionTarget.record_id.is_(None))
    target = db.scalar(query)
    if body.record_id and target is None:
        target = db.scalar(select(ExecutionTarget).where(ExecutionTarget.owner_id == owner,
                                                         ExecutionTarget.record_id == body.record_id))
    if target is None:
        target = ExecutionTarget(owner_id=owner, host_id=host.id, host_target=body.host_target, record_id=body.record_id)
        db.add(target)
    target.host_id, target.host_target = host.id, body.host_target
    target.repo = repo or target.repo or ""
    target.worktree_root = body.worktree_root or match.get("worktree_root", "") or target.worktree_root or ""
    target.name = body.name or target.name or body.host_target
    target.updated_at = now()
    db.flush()
    if body.task_id and not body.record_id:
        from .domain import execute

        task = owned(db, Task, body.task_id, owner)
        execute(db, owner, f"execution-target:{task.id}:{target.id}", "task.update",
                {"task_id": task.id, "expected_revision": task.revision, "execution_target_id": target.id})
    emit(db, owner, "agent.changed", target.id)
    return public_target(target)


# ---------------------------------------------------------------- start


def slug(text):
    words = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return words[:40].rstrip("-") or "task"


def branch_for(db, owner, task):
    """The issue branch Linear names (user/<key>-<slug>), else task/<id>."""
    from .linear_sync import link_for_task

    link = link_for_task(db, owner, task.id)
    snapshot = (link.snapshot or {}) if link else {}
    if isinstance(snapshot.get("branchName"), str) and BRANCH.match(snapshot["branchName"]):
        return snapshot["branchName"]
    if snapshot.get("identifier"):
        user = re.sub(r"[^a-z0-9-]+", "", get_settings().owner_id.lower()) or "eridani"
        return f"{user}/{str(snapshot['identifier']).lower()}-{slug(task.title)}"[:200]
    return f"task/{task.id}"


def base_for(db, task):
    """A child task's worktree starts from its parent run's branch; everything else from main."""
    if task.parent_task_id:
        branch = db.scalar(select(AgentRun.branch).where(AgentRun.task_id == task.parent_task_id, AgentRun.branch != "")
                           .order_by(AgentRun.created_at.desc()).limit(1))
        if branch:
            return branch
    return "main"


def freeze(db, owner, run, task):
    """The minimal plan revision (F05): whatever PRD, steps and criteria the task already has.

    Required children are the direct sub-tasks its linked steps name (F02)."""
    prd = ""
    if task.prd_note_id:
        note = db.get(Note, task.prd_note_id)
        prd = note.content if note and note.owner_id == owner and not note.archived else ""
    plan = AgentRunPlan(owner_id=owner, run_id=run.id, revision=1, task_plan_revision=task.plan_revision or 0,
                        title=task.title, prd_text=prd or task.notes or "", steps=list(task.steps or []),
                        criteria=list(task.success_criteria or []), verify=task.verify,
                        required_children=[c.id for c in coordinator.children(db, task)
                                           if any(s.get("task_id") == c.id for s in task.steps or [])])
    db.add(plan)
    run.plan_revision = 1
    return plan


def spawn_prompt(db, run, plan):
    text = coordinator.render_plan(db, run, plan)
    return ("Work on this Eridani task in the current worktree.\n\n" + text)[:PROMPT_CHARS]


def host_harness(host, harness):
    caps = host.caps or {}
    return "spawn" in caps.get("caps", []) and bool((caps.get("harnesses", {}).get(harness) or {}).get("spawn"))


def start_run(db, owner, account_id, body, *, actor):
    agent_link.gate(db, owner)
    if body.harness in DISABLED_HARNESSES:
        raise DomainError("HARNESS_DISABLED", DISABLED_HARNESSES[body.harness], 400)
    if body.harness not in HARNESSES:
        raise DomainError("INVALID_ARGUMENT", "Choose Claude Code or Codex.")
    task = owned(db, Task, body.task_id, owner, lock=True)
    if task.status in {"completed", "cancelled"} or task.archived or task.is_template:
        raise DomainError("INVALID_ARGUMENT", "Reopen the task before starting an agent on it.")
    active = coordinator_active(db, task.id)
    if active is not None:
        raise DomainError("RUN_ACTIVE", "An agent run is already working on this task.", 409, data={"run_id": active.id})
    target = resolve_target(db, owner, task, body.execution_target_id)
    host = db.get(AgentHost, target.host_id, with_for_update=True) if target.host_id else None
    if host is None or host.owner_id != owner:
        raise DomainError("NEEDS_TARGET", "That execution target has no remotea server.", 409, data={"task_id": task.id})
    if not agent_link.connected(host):
        raise DomainError("HOST_OFFLINE", "The remotea server is not connected.", 409, data={"host_id": host.id})
    if not host_harness(host, body.harness):
        raise DomainError("HARNESS_UNAVAILABLE", f"This remotea server cannot start {body.harness} yet.", 409)
    branch = branch_for(db, owner, task)
    interview = (not task.prd_note_id) if body.interview is None else body.interview
    run = AgentRun(owner_id=owner, account_id=account_id, task_id=task.id, host_id=host.id,
                   execution_target_id=target.id, generation=1, attempt=1, state="spawning", state_revision=1,
                   state_reason="start", state_since=now(), branch=branch, base_branch=base_for(db, task), tool=body.harness,
                   model=body.model or HARNESSES[body.harness], turn_cap=task.turn_cap or 15,
                   worktree="", completion_policy="approve", sync_state={},
                   history=[{"at": now().isoformat(), "from": "queued", "to": "spawning", "reason": "start",
                             "revision": 1, "actor": actor.record()}])
    db.add(run)
    db.flush()
    verify = None
    if interview:
        from . import agent_prompts as prompts

        prompt = prompts.interview(task, coordinator.children(db, task), task.notes or "")[:PROMPT_CHARS]
    else:
        plan = freeze(db, owner, run, task)
        prompt, verify = spawn_prompt(db, run, plan), coordinator.verify_config(plan)
    token = coordinator.mint_token(run)
    coordinator.command(db, run, "spawn", {
        "attempt": run.attempt, "harness": run.tool, "model": run.model,
        "target": {"id": target.host_target, **({"repo": target.repo} if target.repo else {}), "branch": branch,
                   "base": run.base_branch},
        "task_id": task.id, "title": task.title[:200].replace("\n", " "), "group": "Eridani",
        "prompt": prompt, "run_token": token, "env": {}, **({"verify": verify} if verify else {}),
    })
    emit(db, owner, "agent.changed", run.id, run.state_revision)
    return public_run(db, run)


def coordinator_active(db, task_id):
    from .agent_actor import active_run

    return active_run(db, task_id, lock=True)


# ---------------------------------------------------------------- control


def send_message(db, owner, run_id, body, *, actor):
    run = coordinator.load(db, owner, run_id)
    coordinator.check_revision(run, body.expected_revision)
    if run.state in TERMINAL_RUN_STATES or run.state == "stopping":
        raise DomainError("RUN_SETTLED", "This run is stopping or finished.", 409)
    if not run.surf:
        raise DomainError("NOT_STARTED", "The agent has not started yet; send this once it is running.", 409)
    if body.mode == "steer":
        run.generation += 1  # The steered turn belongs to the new generation (F07).
    if run.state in ("verifying", "judging"):
        # Davin's message overrides this turn's check: the next turn is verified and judged instead.
        for attempt in db.scalars(select(AgentAttempt).where(AgentAttempt.run_id == run.id,
                                                             AgentAttempt.status.in_(("queued", "running")))):
            attempt.status, attempt.updated_at = "superseded", now()
        coordinator.transition(db, run, "executing", actor=actor, reason=body.mode)
    elif run.state in coordinator.RESUMABLE or (run.state == "waiting_answer" and body.mode == "steer"):
        coordinator.transition(db, run, coordinator.phase(run), actor=actor, reason=body.mode)
    coordinator.command(db, run, "input", {"surf": run.surf, "incarnation": run.incarnation, "text": body.text,
                                           "mode": body.mode})
    run.last_progress_at = now()
    return public_run(db, run)


def replan(db, owner, run_id, body, *, actor):
    """Re-plan advances the generation (in-flight attempts go stale) and asks the agent for a new submit_plan."""
    from . import agent_prompts as prompts

    run = coordinator.load(db, owner, run_id)
    coordinator.check_revision(run, body.expected_revision)
    if run.state not in {"executing", "needs_you", "awaiting_approval", "awaiting_integration", "planning"}:
        raise DomainError("RUN_SETTLED", f"A run that is {run.state} cannot re-plan.", 409)
    if not run.surf:
        raise DomainError("NOT_STARTED", "The agent has not started yet.", 409)
    busy = run.state == "executing"
    run.generation += 1
    run.plan_pending = False
    for attempt in db.scalars(select(AgentAttempt).where(AgentAttempt.run_id == run.id,
                                                         AgentAttempt.status.in_(("queued", "running")))):
        attempt.status, attempt.updated_at = "superseded", now()
    if run.state != "planning":
        coordinator.transition(db, run, "replanning", actor=actor, reason="replan")
    coordinator.command(db, run, "input", {"surf": run.surf, "incarnation": run.incarnation,
                                           "text": prompts.replan(body.reason), "mode": "steer" if busy else "queue"},
                        source="eridani")
    if run.state == "replanning":
        coordinator.transition(db, run, "planning", actor=actor, reason="replan")
    return public_run(db, run)


def verify_now(db, owner, run_id, *, actor):
    """On-demand verification: the same attempt machinery, bound to the current tuple; it never moves the run."""
    run = coordinator.load(db, owner, run_id)
    if run.state in TERMINAL_RUN_STATES or run.state == "stopping":
        raise DomainError("RUN_SETTLED", "This run is stopping or finished.", 409)
    plan = coordinator.frozen(db, run)
    if plan is None or not (plan.verify or {}).get("verify_id"):
        raise DomainError("NO_VERIFY", "This run's plan has no verification.", 409)
    if db.scalar(select(AgentAttempt.id).where(AgentAttempt.run_id == run.id, AgentAttempt.kind == "verify",
                                               AgentAttempt.status == "running")):
        raise DomainError("VERIFY_RUNNING", "A verification is already running.", 409)
    coordinator.note(run, "verify_requested", actor=actor.record())
    coordinator.start_verify(db, run, plan, on_demand=True)
    return public_run(db, run, detail=True)


def stop_run(db, owner, run_id, body, *, actor):
    run = coordinator.load(db, owner, run_id)
    coordinator.request_stop(db, run, actor=actor, mode=body.mode, reason="stopped_by_user",
                             expected_revision=body.expected_revision)
    return public_run(db, run)


def approve(db, owner, run_id, body, *, actor):
    """Approval is a human completing the task; the gate then settles the run as completed."""
    from .domain import execute

    run = coordinator.load(db, owner, run_id)
    coordinator.check_revision(run, body.expected_revision)
    if run.state in TERMINAL_RUN_STATES or run.state == "stopping":
        raise DomainError("RUN_SETTLED", "This run is stopping or finished.", 409)
    task = db.get(Task, run.task_id)
    with acting(Actor("human", credential_id=actor.credential_id, reason="approve")):
        execute(db, owner, f"run-approve:{run.id}", "task.complete", {"task_id": task.id, "expected_revision": task.revision})
    db.refresh(run)
    return public_run(db, run)


def answer_question(db, owner, question_id, body, *, actor):
    question = db.scalar(select(AgentQuestion).where(AgentQuestion.id == question_id, AgentQuestion.owner_id == owner)
                         .with_for_update())
    if question is None:
        raise DomainError("NOT_FOUND", "That question was not found.", 404)
    coordinator.answer(db, question, [a.model_dump() for a in body.answers], actor=actor)
    return public_question(question)


# ---------------------------------------------------------------- read models


REASON_TEXT = {**coordinator.REASONS, "stopped_by_user": "Stopped from Eridani.",
               "task_completed_externally": "The task was completed outside the run.",
               "approved": "Approved.", "stop_refused": "The host refused the stop; try Stop again.",
               "stop_unconfirmed": "The stop is not confirmed yet.", "started": "", "start": ""}


def public_question(q):
    return encode({"id": q.id, "question_id": q.question_id, "run_id": q.run_id, "generation": q.generation,
                   "items": q.items, "status": q.status, "answers": q.answers, "answered_via": q.answered_via,
                   "remote_answerable": q.remote_answerable, "hold_until": q.hold_until, "detail": q.detail,
                   "created_at": q.created_at, "resolved_at": q.resolved_at})


def public_attempt(a):
    result = a.result or {}
    return encode({"id": a.id, "kind": a.kind, "status": a.status, "generation": a.generation, "turn_seq": a.turn_seq,
                   "plan_revision": a.plan_revision, "verify_id": a.verify_id, "initiator": a.initiator,
                   "verdict": result.get("verdict"), "gap": result.get("gap"), "summary": result.get("summary"),
                   "criteria": result.get("criteria"), "rules": result.get("rules"),
                   "non_verdict": result.get("non_verdict"), "exit_code": result.get("exit_code"),
                   "tail": (result.get("tail") or "")[-4000:] or None, "on_demand": result.get("on_demand"),
                   "model": a.model, "cost": a.cost, "latency_ms": a.latency_ms, "workspace_fingerprint": a.workspace_fingerprint,
                   "created_at": a.created_at, "updated_at": a.updated_at})


def public_command(c):
    payload = c.payload or {}
    return encode({"id": c.id, "op_id": c.op_id, "kind": c.kind, "status": c.status, "generation": c.generation,
                   "mode": payload.get("mode"), "text": (payload.get("text") or "")[:2000] if c.kind == "input" else None,
                   "outcome": c.outcome, "created_at": c.created_at, "updated_at": c.updated_at})


def public_run(db, run, *, detail=False):
    task = db.get(Task, run.task_id)
    host = db.get(AgentHost, run.host_id)
    session = db.get(AgentSession, run.session_pk) if run.session_pk else None
    harness = ((host.caps or {}).get("harnesses", {}).get(run.tool) or {}) if host else {}
    result = {
        "id": run.id, "task": {"id": task.id, "title": task.title, "status": task.status} if task else None,
        "state": run.state, "state_revision": run.state_revision, "state_reason": run.state_reason,
        "reason_text": REASON_TEXT.get(run.state_reason, run.state_reason), "state_since": run.state_since,
        "generation": run.generation, "plan_revision": run.plan_revision, "tool": run.tool, "model": run.model,
        "branch": run.branch, "base_branch": run.base_branch, "worktree": run.worktree,
        "effective_permission_mode": run.effective_permission_mode, "turn_count": run.turn_count,
        "turn_cap": run.turn_cap, "started_at": run.started_at, "finished_at": run.finished_at,
        "stop_requested_at": run.stop_requested_at, "stop_mode": run.stop_mode, "surf": run.surf,
        "turn_seq": run.turn_seq, "plan_pending": run.plan_pending,
        "integration": (run.sync_state or {}).get("integration") or [],
        "host": {"id": host.id, "name": host.name, "connected": agent_link.connected(host)} if host else None,
        "structured_answers": bool(harness.get("question")),
        "session": {"id": session.id, "state": session.state, "since": session.state_since} if session else None,
        "created_at": run.created_at,
    }
    if detail:
        plan = coordinator.frozen(db, run)
        turns = db.scalars(select(AgentTurn).where(
            (AgentTurn.run_id == run.id) | (AgentTurn.session_pk == (run.session_pk or ""))).order_by(AgentTurn.created_at))
        result.update(
            plan={"revision": plan.revision, "title": plan.title, "prd_text": plan.prd_text, "steps": task.steps if task else plan.steps,
                  "criteria": plan.criteria, "verify": plan.verify, "required_children": plan.required_children}
            if plan else None,
            attempts=[public_attempt(a) for a in db.scalars(
                select(AgentAttempt).where(AgentAttempt.run_id == run.id).order_by(AgentAttempt.created_at.desc()).limit(40))],
            interviews=[agent_plan_public(i) for i in db.scalars(
                select(AgentInterview).where(AgentInterview.run_id == run.id).order_by(AgentInterview.created_at))],
            history=run.history or [],
            turns=[{"id": t.id, "seq": t.seq, "text": t.text[:4000], "clipped": len(t.text) > 4000, "truncated": t.truncated,
                    "generation": t.generation, "at": t.created_at, "prompt": t.prompt[:500]} for t in turns],
            commands=[public_command(c) for c in db.scalars(
                select(AgentCommand).where(AgentCommand.run_id == run.id).order_by(AgentCommand.created_at))],
            questions=[public_question(q) for q in db.scalars(
                select(AgentQuestion).where(AgentQuestion.run_id == run.id).order_by(AgentQuestion.created_at))],
        )
    return encode(result)


def agent_plan_public(row):
    from .agent_plan import public_interview

    return public_interview(row)


def list_runs(db, owner, *, task_id=None, active=False, limit=50):
    query = select(AgentRun).where(AgentRun.owner_id == owner)
    if task_id:
        query = query.where(AgentRun.task_id == task_id)
    if active:
        query = query.where(AgentRun.state.not_in(TERMINAL_RUN_STATES))
    return [public_run(db, r) for r in db.scalars(query.order_by(AgentRun.created_at.desc()).limit(limit))]


def unmerged(db, owner):
    """Worktrees the host reports with unmerged or unpushed work (worktrees.report, optional on the wire)."""
    items = []
    for host in db.scalars(select(AgentHost).where(AgentHost.owner_id == owner)):
        report = (host.caps or {}).get("worktrees") or {}
        for row in report.get("worktrees") or []:
            if isinstance(row, dict) and not row.get("candidate"):
                items.append({"host_id": host.id, "host_name": host.name, **{k: row.get(k) for k in (
                    "target", "path", "branch", "base", "ahead", "behind", "unpushed", "uncommitted", "untracked",
                    "merged", "active", "reasons")}})
    return encode({"items": items})
