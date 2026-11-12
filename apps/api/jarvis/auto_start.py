"""Auto-start: tasks on the Auto route start their own run (docs/END_TO_END_UPGRADE_PLAN.md, phase 1;
docs/AGENT_CONTROL.md "Auto-start").

Off until the owner turns it on and names the repositories it may use (owner setting `remotea_auto_start`:
`{enabled, targets: [execution target ids], daily_cap, host_cap, harness}`; both the switch and a non-empty
target list are needed). The supervisor sweep then starts one run per task when all of these hold:

- the task is open, its route is `auto` (readiness.py), the readiness pass found it ready, and it has criteria;
- its target is on the owner's list, its host is connected, and fewer than `host_cap` runs are working there;
- no run is active on it and it was never auto-started before (one attempt: a second run is the owner's);
- fewer than `daily_cap` runs were auto-started for this owner in the last 24 hours.

An auto-started run plans without an interview and its plan is approved without the owner (`approves`) unless
the plan adds a migration, proposes changing something a human set, or the task has left the Auto route: then it
stops for plan approval as any run does. Everything after the plan is unchanged: verify, the judge, CI, and the
owner's approval or the existing auto-complete and auto-merge settings. Changing the task's route is the per-task
"do not auto-start".
"""

import logging
from datetime import timedelta

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select

from . import agent_link
from .domain import DomainError, advisory, emit
from .models import OwnerSettings, Task, now
from .session_models import AgentHost, AgentRun, ExecutionTarget

log = logging.getLogger(__name__)

KEY = "remotea_auto_start"
DEFAULTS = {"enabled": False, "targets": [], "daily_cap": 5, "host_cap": 2, "harness": "claude"}
SWEEP_BATCH = 20
MAX_ATTEMPTS = 3
RETRY = timedelta(minutes=30)


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    targets: list[str] = Field(default_factory=list, max_length=50)
    daily_cap: int = Field(default=5, ge=1, le=50)
    host_cap: int = Field(default=2, ge=1, le=10)
    harness: str = Field(default="claude", max_length=20)


def settings(db, owner) -> dict:
    row = db.get(OwnerSettings, owner)
    saved = (row.values if row else {}).get(KEY) or {}
    return {**DEFAULTS, **{k: v for k, v in saved.items() if k in DEFAULTS}}


def save(db, owner, body: Settings) -> dict:
    """Store the owner's auto-start settings; unknown or foreign targets are refused."""
    from .agent_control import HARNESSES

    if body.harness not in HARNESSES:
        raise DomainError("INVALID_ARGUMENT", "Choose Claude Code or Codex.")
    known = set(db.scalars(select(ExecutionTarget.id).where(ExecutionTarget.owner_id == owner)))
    if not set(body.targets) <= known:
        raise DomainError("INVALID_ARGUMENT", "Choose repositories from your saved targets.")
    advisory(db, f"budget:{owner}")  # The lock every owner-settings write takes.
    row = db.get(OwnerSettings, owner)
    value = {**body.model_dump(), "targets": list(dict.fromkeys(body.targets))}
    if row:
        row.values = {**row.values, KEY: value}
    else:
        db.add(OwnerSettings(owner_id=owner, values={KEY: value}))
    db.flush()
    emit(db, owner, "settings.changed", owner)
    return public(db, owner)


def public(db, owner) -> dict:
    """The settings, the repositories that can be chosen, and how many runs auto-start has used today."""
    value = settings(db, owner)
    targets = list(db.scalars(select(ExecutionTarget).where(ExecutionTarget.owner_id == owner)
                              .order_by(ExecutionTarget.name, ExecutionTarget.id)))
    return {**value, "targets": [t for t in value["targets"] if t in {x.id for x in targets}],
            "choices": [{"id": t.id, "name": t.name or t.host_target, "repo": t.repo} for t in targets],
            "started_today": started_recently(db, owner)}


def auto_started(run) -> bool:
    return bool((run.sync_state or {}).get("auto_start"))


def started_recently(db, owner) -> int:
    return db.scalar(select(func.count()).select_from(AgentRun).where(
        AgentRun.owner_id == owner, AgentRun.created_at > now() - timedelta(hours=24),
        AgentRun.sync_state["auto_start"].astext.is_not(None))) or 0


def approves(db, run, plan) -> bool:
    """May this run's plan go ahead without the owner? Only an auto-started run whose task is still on the Auto
    route, with auto-start still on, and a plan that adds no migration."""
    if not auto_started(run) or plan is None or plan.migration or (run.sync_state or {}).get("plan_declares_migration"):
        return False
    task = db.get(Task, run.task_id)
    return task is not None and task.run_route == "auto" and settings(db, run.owner_id)["enabled"]


def waiting(db, owner):
    """Open Auto-route tasks the pass found ready that have not used their attempts, oldest first."""
    rows = db.scalars(select(Task).where(
        Task.owner_id == owner, Task.run_route == "auto", Task.status == "open", Task.archived.is_(False),
        Task.is_template.is_(False), Task.readiness["state"].astext == "ready")
        .order_by(Task.created_at).limit(SWEEP_BATCH * 5))
    out = []
    for task in rows:
        tried = (task.readiness or {}).get("auto_start") or {}
        if tried.get("run_id") or tried.get("attempts", 0) >= MAX_ATTEMPTS or not task.success_criteria:
            continue
        if tried.get("at") and tried["at"] > (now() - RETRY).isoformat():
            continue
        out.append(task)
    return out[:SWEEP_BATCH]


def sweep(db):
    """Supervisor: start runs for waiting Auto-route tasks, within the daily and per-host caps."""
    from .swarm import host_load

    for row in db.scalars(select(OwnerSettings)):
        value = {**DEFAULTS, **((row.values or {}).get(KEY) or {})}
        owner = row.owner_id
        if not value["enabled"] or not value["targets"] or not agent_link.enabled(db, owner):
            continue
        budget = value["daily_cap"] - started_recently(db, owner)
        if budget <= 0:
            continue
        load = host_load(db, owner)
        for task in waiting(db, owner):
            if budget <= 0:
                break
            started = start_one(db, owner, task, value, load)
            if started:
                budget -= 1


def start_one(db, owner, task, value, load) -> bool:
    """Start one task's run if its target is allowed and has room. A refusal is recorded on the task and retried
    later, MAX_ATTEMPTS times in all; a start is never repeated."""
    from . import agent_control as control
    from . import run_coordinator as coordinator
    from .agent_actor import active_run

    if active_run(db, task.id):
        return False
    target = control.proposed_target(db, owner, task)
    if target is None or target.id not in value["targets"]:
        return False
    host = db.get(AgentHost, target.host_id) if target.host_id else None
    if host is None or not agent_link.connected(host) or load.get(host.id, 0) >= value["host_cap"]:
        return False  # Offline or full: not an attempt; the next sweep looks again.
    tried = dict((task.readiness or {}).get("auto_start") or {})
    try:
        with db.begin_nested():
            result = control.start_run(db, owner, host.account_id,
                                       control.StartInput(task_id=task.id, harness=value["harness"]),
                                       actor=coordinator.COORDINATOR, auto=True)
    except DomainError as error:
        tried = {"attempts": tried.get("attempts", 0) + 1, "at": now().isoformat(),
                 "error": {"code": error.code, "message": error.message[:300]}}
        task.readiness = {**(task.readiness or {}), "auto_start": tried}
        log.info("auto-start refused for task %s: %s", task.id, error.code)
        return False
    task = db.get(Task, task.id)
    task.readiness = {**(task.readiness or {}), "auto_start": {"run_id": result["id"], "at": now().isoformat()}}
    load[host.id] = load.get(host.id, 0) + 1
    emit(db, owner, "task.changed", task.id, task.revision)
    return True

