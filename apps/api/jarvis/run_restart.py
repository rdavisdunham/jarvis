"""Restart run: continue a finished run as a new run linked to it (docs/AGENT_CONTROL.md, "Restart run").

A finished run (completed, stopped, capped, failed) stays terminal as the record. Restart starts a new run on the same
task, branch, worktree and target, with the same harness, model, effort and permission profile, and links it to the
old one (`agent_runs.restarted_from_run_id`). Two modes:

- `plan`: the owner's note (required) goes to planning; the agent proposes a plan for the added work and waits for
  approval like any planned run. A hyperplan run keeps its reviewer and separate worker.
- `pilot`: a freestyle run (no interview, plan or judge); the owner talks to the agent directly. The note, when given,
  is the agent's first message.

Where the host can (`spawn.resume`), the old run's Claude conversation is resumed: the spawn carries
`resume_session_id` (the old session's provider id) when the session was the same harness. The host reports
`result.resumed` on the spawn outcome; it is recorded on the new run's `restarted` timeline note. Without resume the
agent starts fresh, and a plan restart's prompt summarizes the old run's final plan.

A completed task is reopened through the domain's `task.reopen` first (so Linear and the task timeline follow); a
cancelled, archived or template task is refused (`task_closed`).
"""

from uuid import uuid4

from sqlalchemy import select

from . import agent_link
from . import run_coordinator as coordinator
from .agent_sessions import SESSION_ID
from .domain import DomainError, emit, owned
from .models import Task
from .session_models import TERMINAL_RUN_STATES, AgentHost, AgentRun, AgentRunPlan, AgentSession, ExecutionTarget

CAP = "spawn.resume"
NOTE_EXCERPT = 300
WHY = {
    "not_finished": "Only a finished run can be restarted.",
    "task_closed": "The task was cancelled or archived. Reopen it before restarting the run.",
    "run_active": "Another run is already working on this task.",
    "no_target": "This run's execution target no longer exists.",
    "target_missing": "The remotea server no longer lists this run's target.",
    "host_offline": "The remotea server is offline.",
    "note_required": "Say what the plan should add or change.",
}




def target_of(db, run):
    target = db.get(ExecutionTarget, run.execution_target_id) if run.execution_target_id else None
    if target is not None and target.owner_id != run.owner_id:
        target = None
    host = db.get(AgentHost, target.host_id) if target is not None and target.host_id else None
    return target, host


def unavailable(db, run, *, lock=False):
    """Why this run cannot be restarted now (a WHY key), or None. The note check is the route's (it needs the body)."""
    from .agent_actor import active_run
    from .agent_control import hello_target

    if run.state not in TERMINAL_RUN_STATES:
        return "not_finished"
    task = db.get(Task, run.task_id)
    if task is None or task.status == "cancelled" or task.archived or task.is_template:
        return "task_closed"
    if active_run(db, run.task_id, lock=lock) is not None:
        return "run_active"
    target, host = target_of(db, run)
    if target is None or host is None or host.owner_id != run.owner_id:
        return "no_target"
    if not agent_link.connected(host):
        return "host_offline"
    if (host.caps or {}).get("targets") and hello_target(host, target.host_target) is None:
        return "target_missing"
    return None


def public(db, run):
    """The drawer's Restart run button: {available, why}. `why` is null when available, and for a run that has not
    finished (the button is not shown then)."""
    why = unavailable(db, run)
    return {"available": why is None, "why": WHY[why] if why and why != "not_finished" else None}


def restarted_from(db, run):
    """The finished run this one continues: {id, state, finished_at, mode, resumed}, or None. `resumed` is the host's
    answer (null until the spawn outcome, or from a host that does not say)."""
    if not run.restarted_from_run_id:
        return None
    old = db.get(AgentRun, run.restarted_from_run_id)
    if old is None:
        return None
    info = (run.sync_state or {}).get("restart") or {}
    return {"id": old.id, "state": old.state, "finished_at": old.finished_at, "mode": info.get("mode"),
            "resumed": info.get("resumed")}


def restarted_as(db, run):
    """The latest run that continues this one: {id, state}, or None. Only a finished run can have one."""
    if run.state not in TERMINAL_RUN_STATES:
        return None
    newer = db.scalar(select(AgentRun).where(AgentRun.restarted_from_run_id == run.id, AgentRun.owner_id == run.owner_id)
                      .order_by(AgentRun.created_at.desc()).limit(1))
    return {"id": newer.id, "state": newer.state} if newer is not None else None


def resume_id(db, run, host, harness):
    """The old conversation's provider session id when the new spawn can resume it: the host advertises `spawn.resume`,
    the old run's session was the same harness, and its id is known and well-formed. Otherwise None (fresh start)."""
    if host is None or CAP not in ((host.caps or {}).get("caps") or []):
        return None
    session = db.get(AgentSession, run.session_pk) if run.session_pk else None
    if session is None or session.owner_id != run.owner_id:
        return None
    if session.tool != agent_link.HARNESS_TOOLS.get(harness, harness):
        return None
    return session.session_id if SESSION_ID.match(session.session_id or "") else None


def earlier_summary(db, run):
    """What the old run planned (its latest plan revision: title, PRD, steps) and its last summary line, for a plan
    restart that cannot resume the conversation. "" when nothing was stored."""
    plan = db.scalar(select(AgentRunPlan).where(AgentRunPlan.run_id == run.id)
                     .order_by(AgentRunPlan.revision.desc()).limit(1))
    lines = [f"The earlier run ended {run.state}" + (f" ({run.state_reason})" if run.state_reason else "") + "."]
    if plan is not None:
        lines.append(f"Its plan: {plan.title}")
        if plan.prd_text:
            lines += ["PRD:", plan.prd_text[:6000]]
        steps = [str(s.get("title") or s.get("text") or "") for s in plan.steps or [] if isinstance(s, dict)]
        if any(steps):
            lines += ["Steps:"] + [f"- {s[:200]}" for s in steps if s][:40]
    if run.summary:
        lines.append(f"Last status: {run.summary[:500]}")
    return "\n".join(lines) if plan is not None or run.summary else ""


def start_body(run, mode):
    """The StartInput for the new run: the old run's target, harness, model, effort and permission profile. A plan
    restart of a hyperplan run keeps its planner, reviewer and separate worker (a worker the same as the planner, as a
    managed run synthesizes, is left to start_run)."""
    from .agent_control import RoleInput, StartInput

    roles = run.roles if mode == "plan" and isinstance(run.roles, dict) else None
    planner = (roles or {}).get("planner") or {}
    lead = {"harness": planner.get("harness") or run.tool, "model": planner.get("model") or run.model,
            "effort": planner.get("effort") if planner else run.effort,
            "permission_profile": planner.get("permission_profile") if planner else run.permission_profile}

    def role(spec):
        if not isinstance(spec, dict) or not spec.get("harness"):
            return None
        return RoleInput(**{k: spec[k] for k in ("harness", "model", "effort", "permission_profile") if spec.get(k)})

    reviewer = role((roles or {}).get("reviewer"))
    worker = (roles or {}).get("worker")
    worker = role(worker) if isinstance(worker, dict) and (worker.get("harness"), worker.get("model")) != (
        lead["harness"], lead["model"]) else None
    return StartInput(task_id=run.task_id, execution_target_id=run.execution_target_id, interview=False,
                      freestyle=mode == "pilot", turn_cap=min(run.turn_cap, 100) if run.turn_cap else None,
                      wall_clock_limit=run.wall_clock_limit or 0, reviewer=reviewer, worker=worker,
                      rounds=max(1, min(int((roles or {}).get("rounds") or 2), 10)),
                      approve_final=bool((roles or {}).get("approve_final", True)),
                      **{k: v for k, v in lead.items() if v})


def reopen_task(db, owner, task, actor, key):
    """A completed task is reopened through the domain (task.reopen), so Linear and the task timeline follow."""
    from .agent_actor import acting
    from .domain import execute

    with acting(actor):
        execute(db, owner, f"run-restart:{key}", "task.reopen", {"task_id": task.id, "expected_revision": task.revision})


def restart(db, owner, account_id, run_id, body, *, actor):
    """The owner's Restart run: a new run linked to this finished one (see the module docstring)."""
    from . import agent_control as control

    run = coordinator.load(db, owner, run_id)
    coordinator.check_revision(run, body.expected_revision)
    why = unavailable(db, run, lock=True)
    if why == "run_active":
        active = control.coordinator_active(db, run.task_id)
        raise DomainError("RUN_ACTIVE", "An agent run is already working on this task.", 409,
                          data={"run_id": active.id if active else None})
    if why is None and body.mode == "plan" and not body.note.strip():
        why = "note_required"
    if why:
        raise DomainError("RESTART_UNAVAILABLE", WHY[why], 409, data={"run_id": run.id, "reason": why})
    task = owned(db, Task, run.task_id, owner, lock=True)
    if task.status == "completed":
        reopen_task(db, owner, task, actor, f"{run.id}:{body.action_id or uuid4()}")
    start = start_body(run, body.mode)
    _, host = target_of(db, run)
    resume = resume_id(db, run, host, start.harness)
    result = control.start_run(db, owner, account_id, start, actor=actor, restart={
        "from": run, "mode": body.mode, "note": body.note, "resume_session_id": resume,
        "earlier": "" if resume else earlier_summary(db, run)})
    new = db.get(AgentRun, result["id"])
    spawn = coordinator.run_spawn(db, new)
    excerpt = body.note[:NOTE_EXCERPT]
    new.sync_state = {**(new.sync_state or {}), "restart": {
        "from": run.id, "mode": body.mode, "resume": bool(resume), "resumed": None,
        "op_id": spawn.op_id if spawn is not None else None}}
    coordinator.note(new, "restarted", actor=actor.record(), mode=body.mode, restarted_from=run.id,
                     resume=bool(resume), **({"text": excerpt} if excerpt else {}))
    coordinator.note(run, "restarted", actor=actor.record(), mode=body.mode, restarted_as=new.id,
                     **({"text": excerpt} if excerpt else {}))
    emit(db, owner, "agent.changed", run.id, run.state_revision)
    emit(db, owner, "agent.changed", new.id, new.state_revision)
    return control.public_run(db, new, detail=True)


def on_spawn(run, row, result):
    """The restart spawn's outcome: record the host's `resumed` on the run and its `restarted` note (run_coordinator
    calls this for an executed spawn of a restarted run)."""
    info = (run.sync_state or {}).get("restart") or {}
    if not info or info.get("op_id") != row.op_id or not isinstance((result or {}).get("resumed"), bool):
        return
    resumed = result["resumed"]
    run.sync_state = {**(run.sync_state or {}), "restart": {**info, "resumed": resumed}}
    history = list(run.history or [])
    for index in range(len(history) - 1, -1, -1):
        entry = history[index]
        if isinstance(entry, dict) and entry.get("event") == "restarted":
            history[index] = {**entry, "resumed": resumed}
            run.history = history
            break
