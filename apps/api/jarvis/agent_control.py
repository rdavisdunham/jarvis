"""The command service for agent runs: start, message/steer, stop, approve, answer and execution targets.

Browser routes (remotea_routes.py) and voice (`control.*` through domain.mutate, `control_status`) call
these; each authenticates the actor,
resolves the owned task and host, checks the expected revision and records the state change plus
its outbound command in one transaction (run_coordinator.py owns the transitions).
"""

import json
import re
from datetime import datetime, timedelta
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import and_, func, not_, or_, select

from . import agent_link
from . import agent_retention as retention
from . import agent_sessions as sessions
from . import migration_numbers as numbers
from . import run_coordinator as coordinator
from . import run_summaries
from .agent_actor import Actor, acting
from .config import get_settings
from .domain import DomainError, emit, encode, owned
from .models import Note, StructureRecord, Task, now
from .agent_control_schema import COMMANDS  # noqa: F401  (voice commands; domain.COMMANDS includes them)
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
    # Start in freestyle (no interview, no judge): the only way to start on a target whose agents have no Eridani
    # tools (MCP_UNCONFIGURED).
    freestyle: bool = False


class FreestyleInput(Input):
    enabled: bool
    expected_revision: int | None = Field(default=None, ge=1)
    action_id: UUID | None = None  # agent_receipts.once


class ReplanInput(Input):
    reason: str = Field(default="", max_length=2000)
    expected_revision: int | None = Field(default=None, ge=1)
    action_id: UUID | None = None  # agent_receipts.once


class MessageInput(Input):
    text: str = Field(min_length=1, max_length=PROMPT_CHARS)
    mode: str = Field(default="queue", pattern=r"^(queue|steer)$")
    expected_revision: int | None = Field(default=None, ge=1)
    action_id: UUID | None = None  # agent_receipts.once


class StopInput(Input):
    mode: str = Field(default="interrupt", pattern=r"^(interrupt|kill)$")
    # Optional on purpose: Stop is the emergency path and must work from a stale view.
    expected_revision: int | None = Field(default=None, ge=1)
    action_id: UUID | None = None  # agent_receipts.once


class ApproveInput(Input):
    # Required: approving completes the task, so it must be the state the approver reviewed (REVISION_CONFLICT if not).
    expected_revision: int = Field(ge=1)
    action_id: UUID | None = None  # agent_receipts.once


class VerifyNowInput(Input):
    expected_revision: int | None = Field(default=None, ge=1)
    action_id: UUID | None = None  # agent_receipts.once


class JudgeAgainInput(Input):
    # Required: the verdict that follows moves the run, so it must be the state (and plan) you reviewed.
    expected_revision: int = Field(ge=1)
    action_id: UUID | None = None  # agent_receipts.once


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


TARGET_FACTS = ("harnesses", "verify", "enabled", "launch", "verify_status", "mcp")


def public_target(target, adverts=None):
    """A saved target; with `adverts` (advertised() keyed by (host_id, host_target)) it also carries what its host
    advertised for it (harnesses, verify ids and, from caps_version 2 hosts, enabled/launch/verify_status/mcp),
    when the host said."""
    out = {"id": target.id, "host_id": target.host_id, "host_target": target.host_target, "name": target.name,
           "repo": target.repo, "worktree_root": target.worktree_root, "record_id": target.record_id}
    advert = (adverts or {}).get((target.host_id, target.host_target))
    if advert:
        out.update({k: advert[k] for k in TARGET_FACTS if advert.get(k) is not None})
    return encode(out)


def clip(value, limit=300):
    return " ".join(value.split())[:limit] if isinstance(value, str) else None


def target_enabled(item):
    """caps_version 2: the harnesses the target's `launch` enables (permission, installed or not); None if unsaid."""
    value = item.get("enabled") if isinstance(item, dict) else None
    return sorted({v for v in value if isinstance(v, str)}) if isinstance(value, list) else None


def target_launch(item):
    """caps_version 2: each enabled harness's launcher probe on the target, {harness: {ready, version, error?}}."""
    value = item.get("launch") if isinstance(item, dict) else None
    if not isinstance(value, dict):
        return None
    out = {}
    for name, probe in value.items():
        if name in HARNESSES and isinstance(probe, dict) and isinstance(probe.get("ready"), bool):
            out[name] = {"ready": probe["ready"], "version": clip(probe.get("version"), 100),
                         **({"error": clip(probe["error"])} if clip(probe.get("error")) else {})}
    return out


def target_verify_status(item):
    """caps_version 2: each verify id's readiness on the target, [{id, ready, reason?, timeout?, isolated?,
    isolation?, isolation_reason?}]; None when unsaid. `timeout` (seconds) is the entry's own limit when the host
    says it (verify_timeout); the isolation facts (remotea PR #24) are verify_isolation's."""
    value = item.get("verify_status") if isinstance(item, dict) else None
    if not isinstance(value, list):
        return None
    out = []
    for v in value[:50]:
        if not (isinstance(v, dict) and isinstance(v.get("id"), str) and isinstance(v.get("ready"), bool)):
            continue
        entry = {"id": str(v["id"])[:64], "ready": v["ready"], **({"reason": clip(v["reason"])} if clip(v.get("reason")) else {})}
        timeout = verify_timeout(item, entry["id"])
        if timeout:
            entry["timeout"] = timeout
        entry.update(verify_isolation(v))
        out.append(entry)
    return out


# A host-advertised verify timeout outside (0, this] is ignored: the coordinator's default deadline applies.
VERIFY_TIMEOUT_MAX = 4 * 3600


def verify_timeout(item, verify_id):
    """The verify entry's timeout in seconds as the host advertised it for this target: `timeout` on the entry's
    `verify_status` row, or on a `verify` list item sent as an object. None when the host did not say (remotea up
    to PR #24 sends `verify` as plain ids and no timeout)."""
    if not isinstance(item, dict):
        return None
    for key in ("verify_status", "verify"):
        for v in item.get(key) or []:
            if isinstance(v, dict) and v.get("id") == verify_id:
                value = v.get("timeout")
                if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 < value <= VERIFY_TIMEOUT_MAX:
                    return int(value)
    return None


def verify_isolation(entry):
    """The isolation facts of one verify_status entry, only those the host sent and in their wire shape."""
    out = {}
    if isinstance(entry.get("isolated"), bool):
        out["isolated"] = entry["isolated"]
    if entry.get("isolation") in ("bwrap", "none"):
        out["isolation"] = entry["isolation"]
    if clip(entry.get("isolation_reason")):
        out["isolation_reason"] = clip(entry["isolation_reason"])
    return out


def target_mcp(item):
    """Whether an agent spawned on the target gets Eridani's MCP server (remotea PR #20), {configured, reason?}."""
    value = item.get("mcp") if isinstance(item, dict) else None
    if not isinstance(value, dict) or not isinstance(value.get("configured"), bool):
        return None
    return {"configured": value["configured"], **({"reason": clip(value["reason"])} if clip(value.get("reason")) else {})}


def target_harnesses(item):
    """The harnesses one hello target enables (remotea sends a list of names, from its `launch`), or None when the
    host did not say (older hosts: the host-wide `caps.harnesses` flags decide)."""
    value = item.get("harnesses") if isinstance(item, dict) else None
    if isinstance(value, list):
        return sorted({v for v in value if isinstance(v, str)})
    if isinstance(value, dict):
        return sorted(k for k, v in value.items() if isinstance(k, str) and v
                      and (not isinstance(v, dict) or v.get("spawn", True)))
    return None


def hello_target(host, host_target):
    """The host's hello entry for one of its targets, or None."""
    for item in ((host.caps or {}).get("targets") or []) if host is not None else []:
        if isinstance(item, dict) and item.get("id") == host_target:
            return item
    return None


def advertised(db, owner):
    """Targets hosts listed in hello (optional on the wire); the picker offers them. `harnesses` is the list the
    target can start (null when the host did not say), `verify` its verify ids; caps_version 2 hosts add `enabled`
    (permission), `launch` (launcher probes), `verify_status` and `mcp` (null when the host did not say)."""
    found = []
    for host in db.scalars(select(AgentHost).where(AgentHost.owner_id == owner)):
        for item in (host.caps or {}).get("targets") or []:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                found.append({"host_id": host.id, "host_name": host.name, "host_target": item["id"][:64],
                              "repo": str(item.get("repo") or "")[:1000],
                              "worktree_root": str(item.get("worktree_root") or item.get("worktreeRoot") or "")[:1000],
                              "harnesses": target_harnesses(item),
                              "verify": [str(v.get("id") if isinstance(v, dict) else v)[:64]
                                         for v in item.get("verify") or [] if isinstance(v, (str, dict))][:50],
                              "enabled": target_enabled(item), "launch": target_launch(item),
                              "verify_status": target_verify_status(item), "mcp": target_mcp(item)})
    return found


def advert_index(adverts):
    """advertised() keyed by (host_id, host_target), for public_target."""
    return {(a["host_id"], a["host_target"]): a for a in adverts}


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
    adverts = advertised(db, owner)
    index = advert_index(adverts)
    raise DomainError("NEEDS_TARGET", "Choose where this task's agent should run. The choice is saved.", 409, data={
        "task_id": task.id,
        "candidates": [public_target(t, index) for t in candidates or targets],
        "advertised": adverts,
        # type_id lets the picker default to the nearest record that is not the task itself (its project).
        "records": [{"id": r.id, "title": r.title, "type_id": r.type_id} for r in records],
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
    elif target is None and body.task_id:
        # "Only this task" with a target already saved for a record: bind the task to it, do not copy it.
        target = db.scalar(select(ExecutionTarget).where(ExecutionTarget.owner_id == owner,
                                                         ExecutionTarget.host_id == host.id,
                                                         ExecutionTarget.host_target == body.host_target)
                           .order_by(ExecutionTarget.created_at).limit(1))
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
    from .agent_plan import default_verify

    verify, _ = default_verify(db, run, task.success_criteria or [], task.verify)
    plan = AgentRunPlan(owner_id=owner, run_id=run.id, revision=1, task_plan_revision=task.plan_revision or 0,
                        title=task.title, prd_text=prd or task.notes or "", steps=list(task.steps or []),
                        criteria=list(task.success_criteria or []), verify=verify,
                        completion_policy=task.completion_policy or "code_ready",
                        required_children=[c.id for c in coordinator.children(db, task)
                                           if any(s.get("task_id") == c.id for s in task.steps or [])],
                        migration=numbers.carry(db, run))
    db.add(plan)
    run.plan_revision, run.completion_policy = 1, plan.completion_policy
    return plan


def spawn_prompt(db, run, plan):
    text = coordinator.render_plan(db, run, plan)
    return ("Work on this Eridani task in the current worktree.\n\n" + text)[:PROMPT_CHARS]


HARNESS_LABELS = {"claude": "Claude Code", "codex": "Codex"}


def host_harness(host, harness, host_target=None):
    """Can this host start `harness` (on `host_target`)? The host must advertise `spawn` and the harness's host-wide
    spawn flag; when its hello entry for `host_target` lists `harnesses`, the target must enable it too (remotea
    refuses a harness missing from the target's `launch`). Hosts without per-target flags keep the host-wide rule."""
    caps = (host.caps or {}) if host is not None else {}
    if "spawn" not in caps.get("caps", []) or not ((caps.get("harnesses") or {}).get(harness) or {}).get("spawn"):
        return False
    enabled = target_harnesses(hello_target(host, host_target)) if host_target else None
    return enabled is None or harness in enabled


def harness_problem(host, harness, host_target=None):
    """Why `harness` cannot start on this host/target (None when it can); the message names the target. A harness
    the target enables but whose launcher failed its probe (caps_version 2) says so, with the host's error."""
    label = HARNESS_LABELS.get(harness, harness)
    if host is None:
        return "That execution target has no remotea server."
    if not agent_link.connected(host):
        return "The remotea server is not connected."
    if host_harness(host, harness, host_target):
        return None
    item = hello_target(host, host_target) if host_target else None
    probe = (target_launch(item) or {}).get(harness)
    if harness in (target_enabled(item) or []) and (probe is None or not probe["ready"]):
        why = (probe or {}).get("error") or "its launcher did not answer"
        return f"{label} is enabled on {host_target} but not installed: {why}"
    if not host_harness(host, harness):
        why = clip(((host.caps or {}).get("harnesses") or {}).get(harness, {}).get("launch_error"))
        return f"This remotea server cannot start {label} yet" + (f": {why}" if why else ".")
    return f"The {host_target} target on {host.name or 'this remotea server'} does not enable {label}."


def check_host(owner, task, host, harness, host_target=None):
    """The refusals between a resolved target and a spawn; start_run raises them, a proposal reports them."""
    if host is None or host.owner_id != owner:
        raise DomainError("NEEDS_TARGET", "That execution target has no remotea server.", 409, data={"task_id": task.id})
    if not agent_link.connected(host):
        raise DomainError("HOST_OFFLINE", "The remotea server is not connected.", 409, data={"host_id": host.id})
    if not host_harness(host, harness, host_target):
        raise DomainError("HARNESS_UNAVAILABLE", harness_problem(host, harness, host_target), 409,
                          data={"harness": harness, **({"host_target": host_target} if host_target else {})})


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
    check_host(owner, task, host, body.harness, target.host_target)
    tools = next((p for p in target_problems(task, hello_target(host, target.host_target))
                  if p["code"] == "MCP_UNCONFIGURED"), None)
    if tools is not None and not body.freestyle:
        # Without Eridani's tools the agent cannot submit a plan; only a freestyle run (no plan, no judge) can start.
        raise DomainError("MCP_UNCONFIGURED", tools["message"], 409, data={"host_target": target.host_target})
    branch = branch_for(db, owner, task)
    # A task whose criteria and verify a human already wrote skips the interview unless asked for (stage F, M1);
    # an interview that does run only adds to them (agent_plan.plan_floor).
    planned = bool(task.success_criteria) and bool(task.verify)
    interview = (not task.prd_note_id and not planned) if body.interview is None else body.interview
    if body.freestyle:
        interview = False  # Freestyle talks to the agent directly: nothing to interview for or submit.
    run = AgentRun(owner_id=owner, account_id=account_id, task_id=task.id, host_id=host.id,
                   execution_target_id=target.id, generation=1, attempt=1, state="spawning", state_revision=1,
                   state_reason="start", state_since=now(), branch=branch, base_branch=base_for(db, task), tool=body.harness,
                   model=body.model or HARNESSES[body.harness], turn_cap=task.turn_cap or 15,
                   worktree="", completion_policy=task.completion_policy or "code_ready", sync_state={},
                   freestyle=bool(body.freestyle),
                   history=[{"at": now().isoformat(), "from": "queued", "to": "spawning", "reason": "start",
                             "revision": 1, "actor": actor.record()}])
    db.add(run)
    db.flush()
    verify, migration = None, None
    if interview:
        from . import agent_prompts as prompts

        prompt = prompts.interview(task, coordinator.children(db, task), task.notes or "")[:PROMPT_CHARS]
    else:
        plan = freeze(db, owner, run, task)
        prompt, verify = spawn_prompt(db, run, plan), coordinator.verify_config(plan)
        # A reused worktree behind its base: the host tells the agent to re-point this migration (optional on the wire).
        migration = {k: plan.migration[k] for k in ("number", "dir")} if plan.migration else None
    token = coordinator.mint_token(run, fresh=True)
    coordinator.command(db, run, "spawn", {
        "attempt": run.attempt, "harness": run.tool, "model": run.model,
        "target": {"id": target.host_target, **({"repo": target.repo} if target.repo else {}), "branch": branch,
                   "base": run.base_branch},
        "task_id": task.id, "title": task.title[:200].replace("\n", " "), "group": "Eridani",
        "prompt": prompt, "run_token": token, "env": {}, **({"verify": verify} if verify else {}),
        **({"migration": migration} if migration else {}),
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
    sent(run, actor, body.text, mode=body.mode)
    run.last_progress_at = now()
    return public_run(db, run)


def sent(run, actor, text, **values):
    """Every text that leaves for the agent, with who sent it, on the run timeline (stage F, M5)."""
    coordinator.note(run, "sent", actor=actor.record(), text=text[:2000], **values)


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
    coordinator.say(db, run, prompts.replan(body.reason), "replan", mode="steer" if busy else "queue", actor=actor)
    if run.state == "replanning":
        coordinator.transition(db, run, "planning", actor=actor, reason="replan")
    return public_run(db, run)


def verify_now(db, owner, run_id, body=None, *, actor):
    """On-demand verification: the same attempt machinery, bound to the current tuple; it never moves the run."""
    run = coordinator.load(db, owner, run_id)
    coordinator.check_revision(run, body.expected_revision if body else None)
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


JUDGE_AGAIN_STATES = {"needs_you", "awaiting_approval"}


def judge_again(db, owner, run_id, body, *, actor):
    """Judge again: verify (when the plan has a verify command) and judge the last turn against the plan as frozen
    now, typically after Edit criteria (agent_plan.set_criteria). The verdict moves the run as after any turn."""
    run = coordinator.load(db, owner, run_id)
    coordinator.check_revision(run, body.expected_revision)
    if run.state not in JUDGE_AGAIN_STATES or run.state_reason == "plan_review":
        raise DomainError("RUN_NOT_PARKED", f"Judge again works while the run needs you or awaits approval (not during a "
                          f"plan review); this run is {run.state}.", 409, data={"state": run.state})
    if run.freestyle:
        raise DomainError("FREESTYLE", "Freestyle is on, so the judge is off; turn it off first.", 409)
    plan = coordinator.frozen(db, run)
    if plan is None or not plan.criteria:
        raise DomainError("NO_CRITERIA", "This plan has no success criteria to judge against; edit them first.", 409)
    if not run.turn_seq:
        raise DomainError("NO_TURN", "The agent has not finished a turn to judge yet.", 409)
    if db.scalar(select(AgentAttempt.id).where(AgentAttempt.run_id == run.id, AgentAttempt.status.in_(("running", "queued")))):
        raise DomainError("VERIFY_RUNNING", "A verification or verdict is still in progress.", 409)
    coordinator.judge_again(db, run, actor=actor)
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
    from . import agent_sync
    from .linear_sync import for_run

    task = db.get(Task, run.task_id)
    with acting(Actor("human", credential_id=actor.credential_id, reason="approve")), for_run(run.id, "done") as outcome:
        execute(db, owner, f"run-approve:{run.id}", "task.complete", {"task_id": task.id, "expected_revision": task.revision})
    db.refresh(run)
    if outcome.get("status"):
        agent_sync.save(run, "done", outcome)
    return public_run(db, run)


def set_freestyle(db, owner, run_id, body, *, actor):
    """Freestyle: the operator talks to the agent directly. Turns are recorded but not verified, judged or counted
    toward the cap, and no gap prompts are sent. A run the judge parked goes back to executing so the conversation
    can continue; turning freestyle off makes the next turn judged again."""
    run = coordinator.load(db, owner, run_id)
    coordinator.check_revision(run, body.expected_revision)
    if run.state in TERMINAL_RUN_STATES or run.state == "stopping":
        raise DomainError("RUN_SETTLED", "This run is stopping or finished.", 409)
    if bool(run.freestyle) != body.enabled:
        run.freestyle = body.enabled
        coordinator.note(run, "freestyle", enabled=body.enabled, actor=actor.record())
        if body.enabled and run.state == "needs_you" and run.state_reason in FREESTYLE_RESUMES and run.surf:
            coordinator.transition(db, run, "executing", actor=actor, reason="freestyle")
        emit(db, owner, "agent.changed", run.id)
    return public_run(db, run)


# needs_you reasons that only mean "the loop is parked": freestyle resumes the conversation from them.
FREESTYLE_RESUMES = {"judge_needs_you", "no_progress_gap", "verify_missing", "verify_infra", "no_progress", "turn_cap",
                     "awaiting_merge", "pr_unknown", "no_plan", "verify_incomplete"}


def answer_question(db, owner, question_id, body, *, actor):
    question = db.scalar(select(AgentQuestion).where(AgentQuestion.id == question_id, AgentQuestion.owner_id == owner)
                         .with_for_update())
    if question is None:
        raise DomainError("NOT_FOUND", "That question was not found.", 404)
    coordinator.answer(db, question, [a.model_dump() for a in body.answers], actor=actor)
    run = db.get(AgentRun, question.run_id) if question.run_id else None
    if run is not None:
        sent(run, actor, json.dumps(question.answers, ensure_ascii=False), kind="answer", question_id=question.id)
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
                   # Why a verification did not pass and what to do (run_coordinator.classify_verify).
                   "category": result.get("category"), "action": result.get("action"),
                   "duration_ms": result.get("duration_ms"),
                   "tail": (result.get("tail") or "")[-4000:] or None, "on_demand": result.get("on_demand"),
                   "model": a.model, "cost": a.cost, "latency_ms": a.latency_ms, "workspace_fingerprint": a.workspace_fingerprint,
                   # Verify only (remotea PR #24; null from older hosts): sandboxed or not, separate from the exit
                   # code, and whether the fingerprint covered the whole workspace. `deadline_at` follows the
                   # entry's advertised `timeout_s` (run_coordinator.verify_deadline).
                   **(({k: result.get(k) for k in ("isolated", "isolation", "isolation_reason", "workspace_complete",
                                                   "workspace_reasons", "timeout_s", "stale_workspace")}
                       | {"deadline_at": a.deadline_at}) if a.kind == "verify" else {}),
                   "created_at": a.created_at, "updated_at": a.updated_at})


# ---------------------------------------------------------------- open the run's terminal (terminal.focus)

FOCUS_WAIT_SECONDS = 6.0  # How long the route waits for the host's answer before giving up.
FOCUS_POLL_SECONDS = 0.2
# The host's rejection detail -> the reason the page shows.
FOCUS_REASONS = {"no_desktop": "no_desktop", "unknown_surf": "no_surface"}


def focus_terminal(db, owner, run_id):
    """Queue `terminal.focus` for the run's current surface, or say at once why it cannot be sent.

    Returns `(op_id, None)` when queued, or `(None, reason)`: `no_surface` (the run never got a terminal),
    `host_offline` (no host, or its link is down), `host_too_old` (the host does not advertise `terminal.focus`).
    Nothing about the run changes: the command is quiet (no timeline entry, no history note).
    """
    run = coordinator.load(db, owner, run_id, lock=False)
    if not run.surf:
        return None, "no_surface"
    host = db.get(AgentHost, run.host_id) if run.host_id else None
    if host is None or not agent_link.connected(host):
        return None, "host_offline"
    if "terminal.focus" not in ((host.caps or {}).get("caps") or []):
        return None, "host_too_old"
    row = coordinator.command(db, run, "terminal.focus", {"surf": run.surf}, source="human")
    return row.op_id, None


def focus_result(db, owner, op_id):
    """What became of a queued focus: `{sent: true}`, `{sent: false, reason}`, or None while the host has not
    answered."""
    row = db.scalar(select(AgentCommand).where(AgentCommand.owner_id == owner, AgentCommand.op_id == op_id))
    if row is None:
        return {"sent": False, "reason": "no_answer"}
    if row.status == "executed":
        return {"sent": True, "desktops": ((row.outcome or {}).get("result") or {}).get("desktops")}
    if row.status in ("queued", "sent", "acked"):
        return None
    detail = str((row.outcome or {}).get("detail") or "")
    return {"sent": False, "reason": FOCUS_REASONS.get(detail, "no_answer" if row.status == "expired" else "failed"),
            **({"detail": detail[:300]} if detail and detail not in FOCUS_REASONS else {})}


def give_up_focus(db, owner, op_id):
    """The wait is over: a focus still queued is canceled so it never raises a tab later."""
    row = db.scalar(select(AgentCommand).where(AgentCommand.owner_id == owner, AgentCommand.op_id == op_id)
                    .with_for_update())
    if row is not None and row.status == "queued":
        coordinator.settle_command(db, row, "canceled", "nobody is waiting any more")


def public_command(c):
    payload = c.payload or {}
    return encode({"id": c.id, "op_id": c.op_id, "kind": c.kind, "status": c.status, "generation": c.generation,
                   "mode": payload.get("mode"), "text": (payload.get("text") or "")[:2000] if c.kind == "input" else None,
                   "outcome": c.outcome, "created_at": c.created_at, "updated_at": c.updated_at})


DETAIL_PAGE = 100


TURN_CLIP = 4000  # Detail clips each turn's text here (`clipped`); GET /runs/{id}/turns/{turn_id} returns it whole.


def page(db, query, column, limit, offset):
    """The newest `limit` rows after skipping `offset` newest, oldest first; whether older ones remain; and how many
    older ones remain (counted only when there are some)."""
    rows = list(db.scalars(query.order_by(column.desc()).offset(offset).limit(limit + 1)))
    more = len(rows) > limit
    remaining = 0
    if more:
        total = db.scalar(select(func.count()).select_from(query.order_by(None).subquery()))
        remaining = max(0, total - offset - limit)
    return list(reversed(rows[:limit])), more, remaining


def verification_fact(db, run):
    """The run's latest verification as a decision fact: {verify_id, status, at, evidence, isolated, isolation_reason,
    workspace_reasons}. `evidence` says how far it proves the current turn: `fresh` (this generation, turn and plan
    revision, an unchanged and completely fingerprinted workspace), `stale` (an older turn or plan, or the checkout
    changed around it), `incomplete` (remotea's `workspace_complete: false`) or `pending` (still running). None
    before the first one. A host attempt Eridani did not match (`unmatched`) is not counted."""
    rows = db.scalars(select(AgentAttempt).where(AgentAttempt.run_id == run.id, AgentAttempt.kind == "verify")
                      .order_by(AgentAttempt.updated_at.desc()).limit(5))
    attempt = next((a for a in rows if not (a.result or {}).get("unmatched")), None)
    if attempt is None:
        return None
    result = attempt.result or {}
    current = (attempt.generation, attempt.turn_seq, attempt.plan_revision) == \
        (run.generation, run.turn_seq, run.plan_revision)
    if attempt.status == "running":
        evidence = "pending"
    elif not current or result.get("stale_workspace") or attempt.status == "stale":
        evidence = "stale"
    elif result.get("workspace_complete") is False:
        evidence = "incomplete"
    else:
        evidence = "fresh"
    return {"verify_id": attempt.verify_id, "status": attempt.status, "at": attempt.updated_at, "evidence": evidence,
            "turn_seq": attempt.turn_seq, "isolated": result.get("isolated"),
            "isolation_reason": result.get("isolation_reason"), "workspace_complete": result.get("workspace_complete"),
            "workspace_reasons": result.get("workspace_reasons") or [], "deadline_at": attempt.deadline_at}


def pending_controls(db, run):
    """Commands to the host whose effect is not known yet: `pending` (queued or sent, the host has not taken them)
    and `uncertain` (the host could not tell whether they took effect) since the run entered its current state.
    Newest first, at most five: [{kind, status, mode?, at}]."""
    rows = db.scalars(select(AgentCommand).where(
        AgentCommand.run_id == run.id, AgentCommand.status.in_(("queued", "sent", "uncertain")),
        *([AgentCommand.created_at >= run.state_since] if run.state_since else []))
        .order_by(AgentCommand.created_at.desc()).limit(5))
    return [{"kind": c.kind, "status": "uncertain" if c.status == "uncertain" else "pending",
             **({"mode": (c.payload or {}).get("mode")} if c.kind == "input" else {}), "at": c.created_at}
            for c in rows]


def freshness_text(item, clock=None):
    """The run's freshness facts as one line for Eri (control_status), e.g. "verify tests passed 3 min ago on a
    complete workspace; host seen 5 s ago; last turn 2 min ago; in needs you for 4 min"."""
    clock = clock or now()

    def ago(value):
        if not value:
            return None
        at = datetime.fromisoformat(value) if isinstance(value, str) else value
        seconds = max(0, int((clock - at).total_seconds()))
        if seconds < 60:
            return f"{seconds} s"
        if seconds < 3600:
            return f"{seconds // 60} min"
        if seconds < 172800:
            return f"{seconds // 3600} h"
        return f"{seconds // 86400} d"

    parts = []
    verify = item.get("verification")
    if verify:
        when = ago(verify["at"])
        if verify["evidence"] == "pending":
            parts.append(f"verify {verify['verify_id']} running for {when}")
        else:
            where = {"fresh": "for the current turn"
                              + (" on a complete workspace" if verify.get("workspace_complete") else ""),
                     "stale": "but it is stale (an older turn, or the checkout changed around it)",
                     "incomplete": "but the workspace fingerprint was incomplete ("
                                   + ("; ".join(verify["workspace_reasons"]) or "the host did not say what it missed")
                                   + ")"}[verify["evidence"]]
            sandbox = "" if verify["isolated"] is None else (
                "; sandboxed" if verify["isolated"] else
                f"; not sandboxed ({verify['isolation_reason'] or 'the host did not say why'})")
            parts.append(f"verify {verify['verify_id']} {verify['status']} {when} ago {where}{sandbox}")
    host = item.get("host")
    if host:
        seen = ago(host.get("last_seen_at"))
        parts.append(f"host seen {seen} ago" if host.get("connected") else
                     f"host silent for {seen}" if seen else "host never seen")
    last = ago(item.get("last_turn_at"))
    if last:
        parts.append(f"last turn {last} ago")
    since = ago(item.get("state_since"))
    if since:
        parts.append(f"in {item['state'].replace('_', ' ')} for {since}")
    for control in item.get("pending_controls") or []:
        what = control["mode"] if control.get("mode") else control["kind"]
        parts.append(f"{what} sent {ago(control['at'])} ago, "
                     + ("delivery uncertain" if control["status"] == "uncertain" else "not yet taken by the host"))
    return "; ".join(parts)


def run_turns(run):
    """Turns that belong to the run: bound to it, or recorded on its session."""
    return select(AgentTurn).where(AgentTurn.owner_id == run.owner_id,
                                   (AgentTurn.run_id == run.id) | (AgentTurn.session_pk == (run.session_pk or "")))


def reason_detail(db, run):
    """The full text behind the current state, when there is one: the judge's notes for a judge-caused needs_you,
    otherwise the detail recorded with the transition (REASON_TEXT is the one-line version). verify_infra shows the
    host setup action from the verification that could not run."""
    if run.state_reason == "verify_infra":
        recent = db.scalars(select(AgentAttempt).where(AgentAttempt.run_id == run.id, AgentAttempt.kind == "verify")
                            .order_by(AgentAttempt.created_at.desc()).limit(5))
        action = next(((a.result or {}).get("action") for a in recent if (a.result or {}).get("action")), "")
        if action:
            return action[:300]
    if run.state_reason in ("judge_needs_you", "no_progress_gap", "judge_met", "awaiting_merge", "pr_unknown",
                            "verify_missing"):
        verdict = db.scalar(select(AgentAttempt).where(AgentAttempt.run_id == run.id, AgentAttempt.kind == "judge",
                                                       AgentAttempt.status == "done")
                            .order_by(AgentAttempt.created_at.desc()).limit(1))
        if verdict is not None:
            result = verdict.result or {}
            text = result.get("gap") or result.get("summary") or ""
            if text:
                return text[:4000]
    for entry in reversed(run.history or []):
        if entry.get("to") == run.state:
            return (entry.get("detail") or "")[:300]
    return ""


def base_info(db, run, host):
    """What the run is based on (spawn outcome) and how far behind it is now (the latest worktrees.report row for
    its branch), or None for a host that did not say (older remotea)."""
    facts = run.base_facts or {}
    if not facts.get("base_sha"):
        return None
    target = db.get(ExecutionTarget, run.execution_target_id) if run.execution_target_id else None
    report = ((host.caps or {}).get("worktrees") or {}) if host else {}
    row = next((w for w in report.get("worktrees") or [] if isinstance(w, dict) and w.get("branch") == run.branch
                and (target is None or w.get("target") == target.host_target)), None)
    live = isinstance((row or {}).get("behind"), int)
    return {"sha": facts["base_sha"], "ref": facts.get("base_ref") or run.base_branch, "branch": run.base_branch,
            "fetch": facts.get("fetch"), "behind": row["behind"] if live else facts.get("behind"),
            "behind_ref": (row.get("base_ref") or row.get("base")) if live else facts.get("base_ref"),
            "as_of": report.get("at") if live else facts.get("at"), "live": live}


def changes_info(db, run, host, session):
    """What the run changed, for the review packet: the pull requests the agent's session reported (collector
    pr_urls, not a claim in its text) and the host's last worktrees report row for the run's branch (ahead, unpushed,
    uncommitted, merged), with when each was reported. Null fields: the host did not say."""
    from .agent_sessions import pull_requests

    report = ((host.caps or {}).get("worktrees") or {}) if host else {}
    target = db.get(ExecutionTarget, run.execution_target_id) if run.execution_target_id else None
    row = next((w for w in report.get("worktrees") or [] if isinstance(w, dict) and run.branch and w.get("branch") == run.branch
                and (target is None or not w.get("target") or w.get("target") == target.host_target)), None) or {}
    return {"branch": run.branch, "worktree": run.worktree or row.get("path") or "",
            "pr_urls": pull_requests(session.pr_urls if session else []),
            "pr_as_of": (run.sync_state or {}).get("session_report_at"),
            **{k: row.get(k) for k in ("ahead", "behind", "unpushed", "uncommitted", "untracked", "merged")},
            "reported_at": report.get("at") if row else None}


def public_run(db, run, *, detail=False, limit=DETAIL_PAGE, offset=0):
    task = db.get(Task, run.task_id)
    host = db.get(AgentHost, run.host_id)
    session = db.get(AgentSession, run.session_pk) if run.session_pk else None
    harness = ((host.caps or {}).get("harnesses", {}).get(run.tool) or {}) if host else {}
    result = {
        "id": run.id, "task": {"id": task.id, "title": task.title, "status": task.status} if task else None,
        "state": run.state, "state_revision": run.state_revision, "state_reason": run.state_reason,
        # Never a raw token: an unmapped reason shows nothing (the card then shows how long ago the state began).
        "reason_text": REASON_TEXT.get(run.state_reason, ""), "state_since": run.state_since,
        "generation": run.generation, "plan_revision": run.plan_revision, "tool": run.tool, "model": run.model,
        "branch": run.branch, "base_branch": run.base_branch, "worktree": run.worktree,
        "effective_permission_mode": run.effective_permission_mode, "turn_count": run.turn_count,
        "turn_cap": run.turn_cap, "started_at": run.started_at, "finished_at": run.finished_at,
        "stop_requested_at": run.stop_requested_at, "stop_mode": run.stop_mode, "surf": run.surf,
        "turn_seq": run.turn_seq, "plan_pending": run.plan_pending, "freestyle": bool(run.freestyle),
        "integration": (run.sync_state or {}).get("integration") or [],
        "completion_policy": run.completion_policy,
        "base": base_info(db, run, host),
        "reason_detail": reason_detail(db, run),
        # Linear outcomes, separate from the run's own state (agent_sync.py): in_progress, done, comment.
        "linear": {k: {f: v.get(f) for f in ("status", "code", "message", "url", "reconciled")}
                   for k, v in ((run.sync_state or {}).get("linear") or {}).items() if isinstance(v, dict)},
        "host": {"id": host.id, "name": host.name, "connected": agent_link.connected(host),
                 "degraded_reason": agent_link.build_identity(host)["degraded_reason"],
                 # The host's lease: renewed every 30 s while connected, so its age is "host seen N s ago".
                 "last_seen_at": host.last_seen_at} if host else None,
        # Whether the agent was spawned with Eridani's MCP tools (spawn outcome, remotea PR #20); null when unsaid.
        "mcp": (run.sync_state or {}).get("mcp"),
        # Freshness facts kept apart (E-S6): the transcript's age, the latest verification and how far it proves the
        # current turn, and controls sent but not yet taken by the host or whose delivery is uncertain.
        "last_turn_at": db.scalar(run_turns(run).with_only_columns(func.max(AgentTurn.created_at))),
        "verification": verification_fact(db, run),
        "pending_controls": pending_controls(db, run),
        "structured_answers": bool(harness.get("question")),
        "session": {"id": session.id, "state": session.state, "since": session.state_since} if session else None,
        "created_at": run.created_at,
        # One plain-language line on what the agent is doing now (run_summaries.py); summary_stale: a newer one is due.
        **run_summaries.public(run),
    }
    if detail:
        plan = coordinator.frozen(db, run)
        # Paginated (stage F, L7): the newest page by default; `offset` reaches older turns and commands.
        turns, more_turns, older_turns = page(db, run_turns(run), AgentTurn.created_at, limit, offset)
        commands, more_commands, older_commands = page(db, select(AgentCommand).where(
            AgentCommand.run_id == run.id, AgentCommand.kind.not_in(coordinator.QUIET_COMMANDS)),
            AgentCommand.created_at, limit, offset)
        expires = retention.run_data_expires(run)
        result.update(
            plan={"revision": plan.revision, "title": plan.title, "prd_text": plan.prd_text, "steps": task.steps if task else plan.steps,
                  "criteria": plan.criteria, "verify": plan.verify, "required_children": plan.required_children,
                  "proposals": plan.proposals}
            if plan else None,
            attempts=[public_attempt(a) for a in db.scalars(
                select(AgentAttempt).where(AgentAttempt.run_id == run.id, AgentAttempt.kind != "summary")
                .order_by(AgentAttempt.created_at.desc()).limit(40))],
            interviews=[agent_plan_public(i) for i in db.scalars(
                select(AgentInterview).where(AgentInterview.run_id == run.id).order_by(AgentInterview.created_at))],
            history=run.history or [],
            # `role` is "assistant" (a final response; `prompt` carries the typed message for older hosts) or
            # "user" (the typed message itself, sent by the host as a `prompt` frame when the turn started).
            # `clipped`: cut here for display, fetchable whole; `truncated`: the host cut it, the rest was never stored.
            turns=[{"id": t.id, "seq": t.seq, "role": t.role, "text": t.text[:TURN_CLIP], "clipped": len(t.text) > TURN_CLIP,
                    "truncated": t.truncated, "generation": t.generation, "at": t.created_at, "prompt": t.prompt[:500]}
                   for t in turns],
            commands=[public_command(c) for c in commands],
            verify_options=sorted(agent_plan_verify_ids(db, run)),
            changes=changes_info(db, run, host, session),
            # {verify id: {ready, reason?}} from the host's hello (caps_version 2); empty for an older host.
            verify_readiness=run_verify_readiness(db, run),
            more={"turns": more_turns, "commands": more_commands, "offset": offset, "limit": limit,
                  "remaining": {"turns": older_turns, "commands": older_commands}},
            # A settled run's turns and commands are deleted RUN_DATA after it ended (agent_retention.py).
            retention={"expires_at": expires, "expired": retention.run_data_expired(run)},
            questions=[public_question(q) for q in db.scalars(
                select(AgentQuestion).where(AgentQuestion.run_id == run.id).order_by(AgentQuestion.created_at))],
        )
    return encode(result)


def run_verify_readiness(db, run):
    target = db.get(ExecutionTarget, run.execution_target_id) if run.execution_target_id else None
    host = db.get(AgentHost, target.host_id) if target is not None and target.host_id else None
    status = target_verify_status(hello_target(host, target.host_target)) if host is not None else None
    return {s["id"]: {"ready": s["ready"], **({"reason": s["reason"]} if s.get("reason") else {})} for s in status or []}


def agent_plan_verify_ids(db, run):
    from .agent_plan import verify_ids

    return verify_ids(db, run)[1]


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
                    "merged", "merged_reason", "base_ref", "active", "reasons")}})
    return encode({"items": items})


# ---------------------------------------------------------------- voice (control.*, control_status)


def task_by_name(db, owner, name):
    """A task from a spoken name (navigation_find's matcher) or an exact Linear identifier; one match or ask."""
    from .navigation import find

    keys = sessions.issue_keys(name)
    if len(keys) == 1 and keys[0] == name.strip().upper():
        issue = sessions.known_issues(db, owner, keys).get(keys[0])
        if issue:
            return issue["task_id"]
    found = find(db, owner, name)
    ids = []
    for match in found["matches"]:
        if match["kind"] == "task":
            ids.append(match["id"])
        elif match["kind"] == "record":
            record = db.get(StructureRecord, match["id"])
            if record is not None and record.task_id:
                ids.append(record.task_id)
    ids = list(dict.fromkeys(ids))
    if len(ids) != 1:
        titles = [db.get(Task, i).title for i in ids[:8]]
        raise DomainError("AMBIGUOUS_TASK" if ids else "NOT_FOUND",
                          "Which task? " + "; ".join(titles) if ids else f"No task named {name!r}.",
                          409 if ids else 404, data={"candidates": [{"task_id": i, "title": t} for i, t in zip(ids, titles)]})
    return ids[0]


def device_open_run(db, owner):
    """The run the owner's screen has open (UIContext.open_run_id, sent by the remotea page), when this call runs
    under queued work from that device; None otherwise or when the id is not one of the owner's runs."""
    from .access import execution
    from .ui_control import get_context

    grant = execution.get()
    if not grant or not grant.get("device_id"):
        return None
    run_id = (get_context(owner, grant["device_id"]) or {}).get("open_run_id")
    if not isinstance(run_id, str) or not run_id:
        return None
    return db.scalar(select(AgentRun.id).where(AgentRun.id == run_id, AgentRun.owner_id == owner))


def recipient(db, run):
    """Who a control command affects, as Eri names it before acting: task · host · branch · state."""
    task = db.get(Task, run.task_id) if run.task_id else None
    host = db.get(AgentHost, run.host_id) if run.host_id else None
    title, host_name = (task.title if task else "a task")[:120], ((host.name if host else "") or "remotea")[:80]
    branch = run.branch or "no branch yet"
    return {"run_id": run.id, "task": title, "host": host_name, "branch": run.branch or "", "state": run.state,
            "label": f"{title} · {host_name} · {branch} · {run.state}"}


def ambiguous(db, runs, message):
    """Several runs fit a control command: AMBIGUOUS_RUN with each candidate, which control_command turns into a
    needs_choice result so Eri asks instead of acting."""
    unique = list({r.id: r for r in runs if r is not None}.values())[:8]
    return DomainError("AMBIGUOUS_RUN", message, 409, data={"candidates": [recipient(db, r) for r in unique]})


# Which runs a command without a run_id could mean (default: any unsettled run).
PLAUSIBLE = {"control.approve": lambda r: r.state == "awaiting_approval",
             "control.plan_review": lambda r: r.state == "needs_you" and r.state_reason == "plan_review"}


def run_for(db, owner, target, tool=None):
    """The run a voice command means: by id; by task (several tasks matching the name → every run on them is a
    candidate); else the run open on the owner's screen when it is also the newest run the command fits; else the only
    run it fits. Several plausible runs (or an open run that is not the newest) → AMBIGUOUS_RUN with candidates."""
    active = AgentRun.state.not_in(TERMINAL_RUN_STATES)
    if target.run_id:
        return coordinator.load(db, owner, target.run_id, lock=False).id
    if target.task_id or target.task:
        try:
            task_ids, named = [target.task_id or task_by_name(db, owner, target.task)], None
        except DomainError as error:
            if error.code != "AMBIGUOUS_TASK":
                raise
            task_ids, named = [c["task_id"] for c in error.data["candidates"]], error
        runs = list(db.scalars(select(AgentRun).where(AgentRun.owner_id == owner, AgentRun.task_id.in_(task_ids), active)
                               .order_by(AgentRun.created_at.desc()).limit(8)))
        if named is not None:
            if not runs:
                raise named
            raise ambiguous(db, runs, "Several tasks match that name; say which run you mean.")
        if not runs:
            raise DomainError("NOT_FOUND", "No agent is running on that.", 404)
        return runs[0].id
    fits = PLAUSIBLE.get(tool, lambda r: True)
    running = list(db.scalars(select(AgentRun).where(AgentRun.owner_id == owner, active)
                              .order_by(AgentRun.created_at.desc()).limit(10)))
    plausible = [r for r in running if fits(r)]
    open_run = device_open_run(db, owner)
    if open_run:
        if not plausible or plausible[0].id == open_run:
            return open_run
        raise ambiguous(db, [db.get(AgentRun, open_run), *plausible],
                        "The run open on your screen is not the newest one this could mean; say which.")
    if len(plausible) == 1:
        return plausible[0].id
    if plausible:
        raise ambiguous(db, plausible, "Several agents are running; name the task.")
    if len(running) == 1:
        return running[0].id  # The command then says why it does not apply to this run's state.
    if running:
        raise ambiguous(db, running, "Several agents are running; name the task.")
    raise DomainError("NOT_FOUND", "No agent is running on that.", 404)


def owner_control(db, owner):
    """control.* is personal-workspace only and owner-only (stage F, D1), whoever is calling."""
    from .access import actor as account
    from .access import role
    from .bot_access import shared

    if shared(db, owner) or role(db, owner, account(db, owner), lock=True) != "owner":
        raise DomainError("PERSONAL_WORKSPACE", "Agent control works only in your personal workspace.", 403)


# ---------------------------------------------------------------- two-phase control from Eri (stage F, M5, D3)

PENDING_TTL = timedelta(minutes=10)
# A spoken or typed yes: the request starts with an affirmation and asks for nothing else (is_yes). Joined voice
# speech ("ok cool yes", "yes start the agent") counts; a question or a changed request ("yes but …") does not.
AFFIRMATIVE = re.compile(r"^\W*(yes|yeah|yep|yup|sure|ok|okay|confirm|confirmed|do it|go ahead|go|send it|start it|"
                         r"please do)\b", re.IGNORECASE)
NEGATING = re.compile(r"\b(but|instead|actually|not|don[\u2019']?t|no|never|wait|cancel|rather|unless|except|change|"
                      r"different|however|only)\b", re.IGNORECASE)
YES_CHARS, YES_WORDS = 200, 12


def short_answer(text):
    """The bounds every spoken decision shares: at most 12 words, not a question, no but/not/instead."""
    text = (text or "")[:YES_CHARS].strip()
    if not text or "?" in text or "\uff1f" in text or len(text.split()) > YES_WORDS or NEGATING.search(text):
        return ""
    return text


def is_yes(text):
    """A short affirmation: starts with yes/ok/sure/go ahead…, at most 12 words, no question, no but/not/instead."""
    return bool(AFFIRMATIVE.match(short_answer(text)))


def names_one(text, names):
    """A short answer that names one of the offered places ("run it in jarvis", "jarvis please"): the owner's
    choice is their yes when control_confirm carries that target."""
    text = short_answer(text)
    return bool(text) and any(re.search(r"(?<![A-Za-z0-9])" + re.escape(n) + r"(?![A-Za-z0-9])", text, re.IGNORECASE)
                              for n in names if n)


def current_work(db):
    """The queued-work row this call runs under, or None (browser, legacy chat, jobs)."""
    from .access import execution
    from .models import AgentWork

    grant = execution.get()
    return db.get(AgentWork, grant["id"]) if grant else None


def work_message(row):
    from .work_crypto import unseal

    return str((unseal(row.input_ciphertext) or {}).get("message") or "") if row else ""


def proposed_target(db, owner, task):
    """resolve_target for a proposal: the target, or None when the owner still has to choose (read-only)."""
    try:
        return resolve_target(db, owner, task)
    except DomainError:
        return None  # readiness() says why; confirm resolves again.


def readiness_for(db, owner, task, harness="claude", execution_target_id=None):
    """What stands between a task and a run on it, checked without acting: no target chosen (`needs_target`, with
    the picker's data in `need`), or the target's server offline or unable to start the harness (`problems`)."""
    out = {"needs_target": False, "need": None, "target": None, "problems": []}
    try:
        if task is None or task.owner_id != owner:
            raise DomainError("NOT_FOUND", "That task is no longer available.", 404)
        target = resolve_target(db, owner, task, execution_target_id)
        out["target"] = {"id": target.id, "name": target.name or target.host_target, "repo": target.repo}
        host = db.get(AgentHost, target.host_id) if target.host_id else None
        check_host(owner, task, host, harness, target.host_target)
    except DomainError as error:
        if error.code == "NEEDS_TARGET" and isinstance(error.data, dict) and "candidates" in error.data:
            return {**out, "needs_target": True, "need": error.data}
        out["problems"].append({"code": error.code, "message": error.message, "blocking": True})
        return out
    out["problems"] += target_problems(task, hello_target(host, target.host_target) if host is not None else None)
    return out


def target_problems(task, item):
    """What the target's hello says will go wrong (caps_version 2 / remotea PR #20), as readiness problems:
    MCP_UNCONFIGURED (agents get no Eridani tools, so they cannot submit a plan: blocks a start unless it is a
    freestyle run, `freestyle_ok`) and VERIFY_NOT_READY (the task's verify command, or every one the target offers,
    is not ready on the host: a warning, `blocking: false`, since the run still starts and records the failure)."""
    problems = []
    mcp = target_mcp(item)
    if mcp is not None and not mcp["configured"]:
        problems.append({"code": "MCP_UNCONFIGURED", "blocking": True, "freestyle_ok": True,
                         "message": "Agents on this target have no Eridani tools: "
                                    f"{mcp.get('reason') or 'the host did not say why'}."})
    used, ready = verify_use(task, target_verify_status(item))
    if ready is False:
        reasons = "; ".join(f"{s['id']}: {s.get('reason') or 'not ready'}" for s in used if not s["ready"])
        problems.append({"code": "VERIFY_NOT_READY", "blocking": False,
                         "message": f"This target's verify command is not ready ({reasons})."})
    return problems


def verify_use(task, status):
    """(the verify_status rows a run would use, whether that is ready): the task's verify id when it names one, else
    any id the target offers (agent_plan.pick_verify prefers a ready one). (rows, None) when the host did not say."""
    if not status:
        return [], None
    wanted = (task.verify or {}).get("verify_id") if task is not None and isinstance(task.verify, dict) else None
    used = [s for s in status if s["id"] == wanted] if wanted else status
    if not used:
        return [], None
    return used, (used[0]["ready"] if wanted else any(s["ready"] for s in used))


def readiness(db, owner, row):
    """readiness_for a pending control.start. The proposal reports it to Eri, the card shows it; confirm checks again."""
    if row.tool != "control.start":
        return {}
    args = row.arguments or {}
    return readiness_for(db, owner, db.get(Task, args.get("task_id")), args.get("harness", "claude"), args.get("execution_target_id"))


def target_capabilities(db, owner, task, execution_target_id=None):
    """(harnesses, harness_reasons, harness_note): which harnesses the task's target can start and why not. The
    resolved target decides; while none resolves, the union over the picker's candidates (saved and advertised), with
    a note saying so. Another host's capabilities never make the resolved target look ready."""
    try:
        target = resolve_target(db, owner, task, execution_target_id)
    except DomainError as error:
        if error.code != "NEEDS_TARGET" or not isinstance(error.data, dict):
            return {name: False for name in HARNESSES}, {}, None
        pairs = {(t["host_id"], t["host_target"]) for t in error.data.get("candidates") or []} | \
            {(t["host_id"], t["host_target"]) for t in error.data.get("advertised") or []}
        hosts = {h.id: h for h in db.scalars(select(AgentHost).where(AgentHost.owner_id == owner))}
        out = {name: any(hid in hosts and agent_link.connected(hosts[hid]) and host_harness(hosts[hid], name, ht)
                         for hid, ht in pairs) for name in HARNESSES}
        reasons = {name: f"No target offered for this task can start {HARNESS_LABELS.get(name, name)}."
                   for name, ok in out.items() if not ok}
        return out, reasons, "No target is chosen yet; these are what any offered target can start."
    host = db.get(AgentHost, target.host_id) if target.host_id else None
    reasons = {name: harness_problem(host if host is not None and host.owner_id == owner else None, name,
                                     target.host_target) for name in HARNESSES}
    return ({name: reason is None for name, reason in reasons.items()},
            {name: reason for name, reason in reasons.items() if reason}, None)


def target_readiness(db, owner, task, execution_target_id=None):
    """The resolved target's verify and MCP readiness as its host last advertised it (caps_version 2 / PR #20):
    `verify_status` (all its ids), `verify_ready` for the ids a run would use (verify_use: the task's verify id, else
    whether any id the target offers is ready, since the default pick prefers a ready one; null when the host did
    not say or there are none), `verify_reasons` {id: reason} for those not ready, and `mcp` {configured, reason?}
    (null when unsaid). Nothing while no target resolves."""
    empty = {"verify_status": None, "verify_ready": None, "verify_reasons": {}, "mcp": None}
    try:
        target = resolve_target(db, owner, task, execution_target_id)
    except DomainError:
        return empty
    host = db.get(AgentHost, target.host_id) if target.host_id else None
    item = hello_target(host, target.host_target) if host is not None and host.owner_id == owner else None
    status = target_verify_status(item)
    if status is None:
        return {**empty, "mcp": target_mcp(item)}
    used, ready = verify_use(task, status)
    return {"verify_status": status, "verify_ready": ready,
            "verify_reasons": {s["id"]: s.get("reason") or "not ready" for s in used if not s["ready"]},
            "mcp": target_mcp(item)}


def preflight(db, owner, task_id, harness="claude", execution_target_id=None):
    """The Run dialog's view of a start before it happens: readiness_for plus the run already on the task, whether
    the interview would run (start_run's rule), which harnesses the task's target can spawn (`harness_reasons` says
    why not; `harness_note` is set while no target resolves), the target's verify and MCP readiness
    (target_readiness), and the task's title."""
    task = owned(db, Task, task_id, owner)
    out = readiness_for(db, owner, task, harness, execution_target_id)
    active = coordinator_active(db, task.id)
    planned = bool(task.success_criteria) and bool(task.verify)
    harnesses, reasons, note = target_capabilities(db, owner, task, execution_target_id)
    return {**out, "task": {"id": task.id, "title": task.title},
            "active_run": {"id": active.id, "state": active.state} if active is not None else None,
            "interview": not task.prd_note_id and not planned,
            "harnesses": harnesses, "harness_reasons": reasons, "harness_note": note,
            **target_readiness(db, owner, task, execution_target_id)}


# ---------------------------------------------------------------- setup readiness (E-Q1)

CONFIG_FILE = "~/.config/remotea/eridani.json"
RELOAD = "Save eridani.json; remotea reconnects within 30 seconds and its next hello carries the change."
TOO_OLD = "unknown (host too old)"
SETUP_HINT = "Settings → Connections → remotea lists every problem with its fix."
SETUP_FIX =("run `remotea eridani setup` in WSL and paste a setup code from Settings → Connections → remotea "
             "(New setup code) when asked")


def problem(code, message, fix, blocks=True):
    """One setup problem as the page, the probe, control_status and Eri all say it. `blocks`: a run cannot start
    (or cannot finish its planning turn) until it is fixed; otherwise it starts but works worse."""
    return {"code": code, "message": message, "fix": fix, "blocks": blocks}


def verify_fix(reason):
    """The plain-language fix for a verify entry the host says is not ready (reason text from remotea's hello)."""
    text = (reason or "").lower()
    if "bwrap" in text or "bubblewrap" in text:
        return ('The entry demands isolation "bwrap" and the sandbox does not work on this host: install bubblewrap '
                '(sudo apt install bubblewrap) or set isolation "none" on the entry. ' + RELOAD)
    if "path" in text:
        return ("Its program is not on the service PATH (systemd's minimal PATH; uv in ~/.local/bin is the usual "
                "case): use scripts/verify.sh or an absolute program, or set env.PATH on the entry in "
                f"{CONFIG_FILE}. {RELOAD}")
    if "cwd" in text:
        return f"Its cwd does not exist in the repository: fix cwd on the entry in {CONFIG_FILE}. {RELOAD}"
    return f"Fix the verify entry in {CONFIG_FILE} (program, cwd, env). {RELOAD}"


def host_setup(host):
    """Step 1 of the setup page: is the host connected and healthy? Its build line and its problems."""
    build = agent_link.build_identity(host)
    name = host.name or host.machine or "remotea"
    out = {"id": host.id, "name": name, "connected": agent_link.connected(host), "last_seen_at": host.last_seen_at,
           "hello_at": (host.caps or {}).get("hello_at"), "version": host.version, "build": build["build"],
           "build_dirty": build["build_dirty"], "caps_version": build["caps_version"],
           "degraded_reason": build["degraded_reason"], "problems": []}
    if not out["connected"]:
        out["problems"].append(problem(
            "HOST_OFFLINE", f"The remotea server {name} is not connected.",
            "In WSL, check it with `systemctl --user status remotea` and start it with "
            f"`systemctl --user start remotea`; if it was never linked, {SETUP_FIX}."))
    if build["degraded_reason"]:
        out["problems"].append(problem(
            "HOST_DEGRADED", f"Host journal degraded: {build['degraded_reason']}",
            "remotea refuses starts, messages and stops until its journal can write again: free disk space (or fix "
            "the journal folder's permissions) on the host, then `systemctl --user restart remotea`."))
    return encode(out)


def target_setup(db, owner, host, host_target, saved=None):
    """Steps 2 and 3 of the setup page for one target (a host's `host_target`, saved or only advertised): what the
    host's last hello said about it and what stands in a run's way, each problem with its fix. The harness checks
    are harness_problem's (the refusals start_run and the Run dialog give); facts an older host did not send are
    null and read "unknown (host too old)"."""
    label = (saved.name if saved is not None and saved.name else None) or host_target
    host = host if host is not None and host.owner_id == owner else None
    item = hello_target(host, host_target) if host is not None else None
    live = host is not None and agent_link.connected(host)
    out = {"host_id": host.id if host is not None else None, "host_name": (host.name or "remotea") if host else None,
           "host_target": host_target, "name": label,
           "repo": str((item or {}).get("repo") or (saved.repo if saved is not None else ""))[:1000],
           "execution_target_id": saved.id if saved is not None else None,
           "key": saved.id if saved is not None else f"host:{host.id if host else ''}:{host_target}",
           "advertised": item is not None, "harnesses": {}, "verify": None, "mcp": None, "problems": [], "notes": []}
    if host is None:
        out["problems"].append(problem("HOST_MISSING", f"{label} has no remotea server.",
                                       f"{SETUP_FIX[0].upper()}{SETUP_FIX[1:]}, then pick this repository again."))
        return finish_target(out)
    if not live:
        out["problems"].append(problem("HOST_OFFLINE", f"The remotea server {out['host_name']} is not connected.",
                                       "Start remotea in WSL: `systemctl --user start remotea`."))
    if item is None:
        out["problems"].append(problem(
            "TARGET_NOT_ADVERTISED", f"{out['host_name']} does not list the {host_target} target.",
            f'Add a target with id "{host_target}" to targets[] in {CONFIG_FILE} (`remotea eridani setup '
            f"--add-repo {host_target}=<path>` does it). {RELOAD}"))
        return finish_target(out)
    enabled, launch = target_enabled(item), target_launch(item)
    for name in HARNESSES:
        why = harness_problem(host, name, host_target)
        probe = (launch or {}).get(name)
        label_h = HARNESS_LABELS.get(name, name)
        out["harnesses"][name] = {
            "label": label_h, "ready": why is None, "enabled": (name in enabled) if enabled is not None else None,
            "version": (probe or {}).get("version"), "error": (probe or {}).get("error"),
            "launch": "probed" if probe is not None else TOO_OLD if launch is None
            else "not probed" if name in (enabled or []) else "not enabled",
            "reason": why}
        if live and enabled is not None and name in enabled and why is not None:
            out["problems"].append(problem(
                "HARNESS_NOT_INSTALLED", f"{label_h} is enabled but not installed: {why}",
                f"Install {label_h} where the service finds it (its login shell's PATH), or remove {name} from the "
                f"target's launch in {CONFIG_FILE}. {RELOAD}", blocks=False))
    if live and not any(h["ready"] for h in out["harnesses"].values()):
        out["problems"].append(problem(
            "NO_HARNESS", f"No coding agent can start on {label}.",
            f"Add launch.claude (or launch.codex) to the {host_target} target in {CONFIG_FILE} and install that "
            f"agent on the host. {RELOAD}"))
    status = target_verify_status(item)
    out["verify"] = status
    if status is None:
        out["notes"].append(f"Verify readiness: {TOO_OLD}.")
    elif not status:
        out["problems"].append(problem(
            "NO_VERIFY", f"{label} has no verify command, so a run cannot check its work with tests.",
            f"Add a verify[] entry to the target in {CONFIG_FILE} (for jarvis, scripts/verify.sh with an isolated "
            f"test database). {RELOAD}", blocks=False))
    for entry in status or []:
        if not entry["ready"]:
            out["problems"].append(problem(
                "VERIFY_NOT_READY",
                f"Verify {entry['id']} is not ready: {entry.get('reason') or 'the host did not say why'}",
                verify_fix(entry.get("reason")), blocks=False))
        elif entry.get("isolated") is False and entry.get("isolation") != "none":
            # Not chosen: the sandbox is unavailable on this host (an entry that sets "none" chose it).
            out["notes"].append(f"Verify {entry['id']} runs without a sandbox: "
                                f"{entry.get('isolation_reason') or 'the host did not say why'}.")
    out["mcp"] = target_mcp(item)
    if out["mcp"] is None:
        out["notes"].append(f"Eridani tools for agents (MCP): {TOO_OLD}.")
    elif not out["mcp"]["configured"]:
        out["problems"].append(problem(
            "MCP_NOT_CONFIGURED",
            f"Agents on {label} get no Eridani tools: {out['mcp'].get('reason') or 'the host did not say why'}",
            f"{SETUP_FIX[0].upper()}{SETUP_FIX[1:]}: it saves the MCP key to ~/.config/remotea/eridani-mcp-token. "
            "Without it the agent cannot submit its plan."))
    return finish_target(out)


def finish_target(out):
    out["ready"] = not any(p["blocks"] for p in out["problems"])
    out["line"] = setup_line(out["name"], out.get("host_name"), out["problems"]) if not out["ready"] else None
    return encode(out)


def setup_line(name, host_name, problems):
    """One spoken line for an unready host or target (control_status, Eri): its blocking problems, then the first
    fix."""
    blocking = [p for p in problems if p["blocks"]]
    if not blocking:
        return None
    where = name + (f" on {host_name}" if host_name else "")
    return f"{where}: " + "; ".join(p["message"].rstrip(".") for p in blocking) + f". Fix: {blocking[0]['fix']}"


def setup_report(db, owner):
    """The guided setup page (GET /remotea/setup): each host (connected? build? degraded?), then each target its hosts
    advertise or the owner saved (one entry per host and host_target), with its readiness and fixes. `lines` holds
    one line per unready host or target, the way control_status says them; `ready` is true when a connected,
    healthy host has at least one ready target."""
    hosts = list(db.scalars(select(AgentHost).where(AgentHost.owner_id == owner).order_by(AgentHost.created_at)))
    by_id = {h.id: h for h in hosts}
    saved = {}
    for row in db.scalars(select(ExecutionTarget).where(ExecutionTarget.owner_id == owner)
                          .order_by(ExecutionTarget.created_at)):
        saved.setdefault((row.host_id, row.host_target), row)
    pairs = []
    for h in hosts:
        for item in (h.caps or {}).get("targets") or []:
            if isinstance(item, dict) and isinstance(item.get("id"), str) and (h.id, item["id"]) not in pairs:
                pairs.append((h.id, item["id"]))
    pairs += [pair for pair in saved if pair not in pairs]
    targets = [target_setup(db, owner, by_id.get(hid), ht, saved.get((hid, ht))) for hid, ht in pairs]
    host_rows = [host_setup(h) for h in hosts]
    lines = []
    if not hosts:
        lines.append(f"No remotea server has connected yet. Fix: {SETUP_FIX}.")
    lines += [setup_line(h["name"], None, h["problems"]) for h in host_rows if h["problems"]]
    offline = {h["id"] for h in host_rows if not h["connected"]}
    # An offline host's targets would only repeat the host's own line.
    lines += [t["line"] for t in targets if t["line"] and not (
        t["host_id"] in offline and {p["code"] for p in t["problems"] if p["blocks"]} == {"HOST_OFFLINE"})]
    if hosts and not targets:
        lines.append("No execution target is configured. Fix: run `remotea eridani setup --scan <folder>` in WSL to "
                     f"add your repositories, or add targets[] to {CONFIG_FILE}.")
    ready = any(h["connected"] and not h["problems"] for h in host_rows) and any(t["ready"] for t in targets)
    return {"ready": ready, "hosts": host_rows, "targets": targets, "lines": lines}


def proposal_setup(db, owner, row):
    """The setup line for a pending start's resolved target when something there would stall the run (no Eridani
    tools, say) although check_host lets it start; None when the target is ready."""
    try:
        target = resolve_target(db, owner, db.get(Task, (row.arguments or {}).get("task_id")),
                                (row.arguments or {}).get("execution_target_id"))
    except DomainError:
        return None
    host = db.get(AgentHost, target.host_id) if target.host_id else None
    return target_setup(db, owner, host, target.host_target, target)["line"]


def target_probe(db, owner, key):
    """POST /remotea/targets/{key}/probe: one target's readiness. remotea has no probe command yet, so this reads the
    facts from the host's last hello and says so (`source: "hello"`, `hello_at`, `note`). `key` is a saved
    execution target id or an advertised target's `host:<host_id>:<host_target>`."""
    if key.startswith("host:"):
        _, host_id, host_target = (key.split(":", 2) + ["", ""])[:3]
        host, saved = db.get(AgentHost, host_id), None
        if host is None or host.owner_id != owner or hello_target(host, host_target) is None:
            raise DomainError("NOT_FOUND", "That remotea target is no longer advertised.", 404)
    else:
        try:
            UUID(key)
        except ValueError:
            raise DomainError("NOT_FOUND", "That execution target is not available.", 404) from None
        saved = owned(db, ExecutionTarget, key, owner)
        host, host_target = (db.get(AgentHost, saved.host_id) if saved.host_id else None), saved.host_target
    offline = host is not None and not agent_link.connected(host)
    note = ("This remotea server has no readiness probe yet, so these are the facts from its last hello"
            + (" (it is offline)" if offline else "") + ". Saving eridani.json makes remotea reconnect with fresh "
            "facts within 30 seconds; `remotea eridani status --probe` in WSL checks now.")
    return {"source": "hello", "hello_at": (host.caps or {}).get("hello_at") if host is not None else None,
            "note": note, "host": host_setup(host) if host is not None and host.owner_id == owner else None,
            "target": target_setup(db, owner, host, host_target, saved)}


def proposal(db, owner, result):
    """A control_start proposal as Eri sees it: the pending row plus what still stands in its way, and what to
    tell the owner about it. The picker's data stays on the remotea card."""
    from .session_models import AgentPendingControl

    row = db.get(AgentPendingControl, result.get("pending_id"))
    if row is None or row.tool != "control.start":
        return result
    ready = readiness(db, owner, row)
    told, extra = [], {}
    if ready["needs_target"]:
        extra["targets"] = target_options(ready["need"])
        names = ", ".join(t["name"] for t in extra["targets"][:6]) or "none listed yet"
        told.append("No repository is chosen for this task yet. Ask the owner where it should run, naming the "
                    f"options in targets ({names}), then call control_confirm with the chosen execution_target_id "
                    "(their answer naming one of them counts as their yes); an advertised one is saved for the task's "
                    "project first. They can also pick it on the card in chat or on the remotea page. A bare yes "
                    "cannot start it until a place is chosen.")
    for problem in ready["problems"]:
        if problem.get("blocking", True):
            told.append(f"Tell the owner: {problem['message']} ({problem['code']}) Confirming fails until that is "
                        "fixed; the proposal stays open until it expires.")
        else:
            told.append(f"Tell the owner: {problem['message']} ({problem['code']}) The run can still start, but its "
                        "verification will not pass until that is fixed on the host.")
    # The setup line carries the fix; it is skipped while a problem that makes confirming fail anyway stands (a
    # freestyle-lifted MCP_UNCONFIGURED or a warning does not count, so the owner still hears how to fix them).
    stuck = any(p.get("blocking", True) and not p.get("freestyle_ok") for p in ready["problems"])
    setup = proposal_setup(db, owner, row) if ready["target"] and not stuck else None
    if setup:
        told.append(f"Also tell the owner before they confirm: {setup}")
    return {**result, "task_id": row.arguments.get("task_id"), "needs_target": ready["needs_target"],
            "target": ready["target"], "problems": ready["problems"], **extra,
            **({"next": " ".join(told) + " " + result.get("next", "")} if told else {})}


def target_options(need):
    """The picker's choices as Eri reads them out: saved targets by id, then advertised ones not saved yet by the
    key control_confirm accepts (host:<host_id>:<host_target>; the same options the remotea card offers)."""
    saved = [{"execution_target_id": t["id"], "name": t.get("name") or t["host_target"], "repo": t.get("repo") or "",
              "saved": True} for t in need.get("candidates") or []]
    known = {(t["host_id"], t["host_target"]) for t in need.get("candidates") or []}
    offered = [{"execution_target_id": f"host:{t['host_id']}:{t['host_target']}", "name": t["host_target"],
                "repo": t.get("repo") or "", "host": t.get("host_name") or "", "saved": False}
               for t in need.get("advertised") or [] if (t["host_id"], t["host_target"]) not in known]
    return [*saved, *offered]


def target_scope(need):
    """Where a picked target is remembered, as the card defaults it: the nearest record that is not the task itself
    (its project), else the task's own record, else only this task (None)."""
    records = need.get("records") or []
    nearest = next((r for r in records if r.get("type_id") and r.get("type_id") != "task"), records[0] if records else None)
    return nearest["id"] if nearest else None


def chosen_target(db, owner, row, choice):
    """control_confirm's execution_target_id: a saved target's id as is; an advertised target's key is saved first
    (save_target, for the task's nearest record, else the task) exactly as the card's picker does."""
    if not choice or not choice.startswith("host:"):
        return choice
    ready = readiness(db, owner, row)
    need = ready.get("need") or {}
    host_id, _, host_target = choice[5:].partition(":")
    match = next((a for a in advertised(db, owner) if a["host_id"] == host_id and a["host_target"] == host_target), None)
    if match is None or not need:
        raise DomainError("NOT_FOUND", "That repository is not offered by a connected remotea server.", 404,
                          data={"targets": target_options(need)})
    record_id = target_scope(need)
    body = TargetInput(host_id=host_id, host_target=host_target, repo=match["repo"], worktree_root=match["worktree_root"],
                       **({"record_id": record_id} if record_id else {"task_id": need["task_id"]}))
    return save_target(db, owner, body)["id"]


def propose(db, owner, account, tool, args, actor):
    """control_start / control_message from Eri: resolve and validate now, act only after the owner confirms."""
    work = current_work(db)
    if tool == "control.start":
        task = owned(db, Task, args.task_id or task_by_name(db, owner, args.task), owner)
        arguments, run_id = {"task_id": task.id, "harness": args.harness}, None
        # Where it runs is settled now when it can be (the card then names it); without a target the card asks.
        target = proposed_target(db, owner, task)
        if target is not None:
            arguments["execution_target_id"] = target.id
        summary = f"Start {'Codex' if args.harness == 'codex' else 'Claude Code'} on \u201c{task.title[:200]}\u201d" + (
            f" in {(target.name or target.host_target)[:100]}" if target is not None else "")
    else:
        run = coordinator.load(db, owner, run_for(db, owner, args, tool), lock=False)
        task = db.get(Task, run.task_id)
        arguments, run_id = {"run_id": run.id, "text": args.text, "mode": args.mode}, run.id
        verb = "Interrupt and steer" if args.mode == "steer" else "Send to"
        summary = f"{verb} the agent on \u201c{(task.title if task else '')[:200]}\u201d: \u201c{args.text[:300]}\u201d"
    from .session_models import AgentPendingControl

    same = db.scalar(select(AgentPendingControl).where(
        AgentPendingControl.owner_id == owner, AgentPendingControl.account_id == account, AgentPendingControl.tool == tool,
        AgentPendingControl.arguments == arguments, AgentPendingControl.status == "pending",
        AgentPendingControl.expires_at > now()).order_by(AgentPendingControl.created_at.asc()).limit(1))
    if same is not None:
        # The model proposed the same thing again (a new chat turn does not carry the earlier pending_id).
        # If this request is the owner's plain yes to that proposal, it is the confirmation; otherwise hand the
        # existing proposal back rather than stacking another card on the remotea page.
        via = owners_yes(db, owner, same, work)
        if via:
            return decide(db, owner, same.id, confirm=True, via=via, actor=actor)
        return {**pending_reply(same), "next": "This was already proposed and is still waiting for the owner. " + NEXT}
    row = AgentPendingControl(owner_id=owner, account_id=account, work_id=work.id if work else None, tool=tool,
                              arguments=arguments, summary=summary, run_id=run_id, status="pending",
                              actor=actor.record(), decided_via="", expires_at=now() + PENDING_TTL)
    db.add(row)
    db.flush()
    emit(db, owner, "agent.changed", run_id or row.id)
    return {**pending_reply(row), "next": "Nothing was sent yet. " + NEXT}


NEXT = ("Read the summary to the owner and ask them to confirm. They can tap Confirm on the card in chat or on the "
        "remotea page, or answer yes in their next message (a short reply that starts with yes, okay, sure or go "
        "ahead); then call control_confirm (pending_id optional: the newest pending action). Never confirm on their "
        "behalf, and never call control_start again after their yes.")


def pending_reply(row):
    return {"status": "pending_confirmation", "pending_id": row.id, "summary": row.summary,
            "expires_at": row.expires_at.isoformat()}


def owners_yes(db, owner, row, work, *, chosen=()):
    """Is this request the owner's own plain yes to `row`? The same rule spoken_confirmation enforces: a later
    request of the same account, not the one that proposed it, whose message is a short affirmation (is_yes) or,
    when confirming with a chosen target, names it (`chosen`: that target's names), and `row` is the newest
    proposal that request could mean. Returns the channel ("voice"/"chat") or None."""
    from .models import Job
    from .session_models import AgentPendingControl

    if work is None or work.credential_id or row.work_id == work.id or row.account_id != work.account_id:
        return None
    # The yes must be unambiguous: no *different* action proposed after this one (identical twins are this one).
    later = db.scalar(select(AgentPendingControl.id).where(
        AgentPendingControl.owner_id == owner, AgentPendingControl.account_id == work.account_id,
        AgentPendingControl.status == "pending", not_proposed_by(work),
        AgentPendingControl.created_at > row.created_at,
        not_(and_(AgentPendingControl.tool == row.tool, AgentPendingControl.arguments == row.arguments))).limit(1))
    job = db.get(Job, work.id)
    message = work_message(work)
    if later or job is None or job.created_at < row.created_at or not (is_yes(message) or names_one(message, chosen)):
        return None
    return "voice" if work.voice_session_id else "chat"


def public_pending(row):
    return encode({"id": row.id, "tool": row.tool, "summary": row.summary, "run_id": row.run_id,
                   "task_id": (row.arguments or {}).get("task_id") if row.tool == "control.start" else None,
                   "status": row.status, "created_at": row.created_at, "expires_at": row.expires_at,
                   "decided_via": row.decided_via, "decided_at": row.decided_at})


def pending_list(db, owner):
    """The remotea page's open proposals, each with readiness() so a start without a target shows the picker."""
    from sqlalchemy import update

    from .session_models import AgentPendingControl

    # decide() refuses an expired row but its transaction rolls back; settle them here so they read as expired.
    db.execute(update(AgentPendingControl).where(AgentPendingControl.owner_id == owner,
                                                 AgentPendingControl.status == "pending",
                                                 AgentPendingControl.expires_at <= now()).values(status="expired"))
    rows = db.scalars(select(AgentPendingControl).where(AgentPendingControl.owner_id == owner,
                                                        AgentPendingControl.status == "pending",
                                                        AgentPendingControl.expires_at > now())
                      .order_by(AgentPendingControl.created_at.desc()).limit(20))
    return [{**public_pending(r), **readiness(db, owner, r)} for r in rows]


def decide(db, owner, pending_id, *, confirm, via, actor, execution_target_id=None):
    """Run (or drop) a pending control. `via` is "app" for the remotea page, else "voice"/"chat" (control_confirm).

    A start that still needs a target is refused with NEEDS_TARGET and stays pending (the transaction rolls
    back); the page's picker then confirms again with `execution_target_id`."""
    from .access import actor as account
    from .session_models import AgentPendingControl

    row = db.scalar(select(AgentPendingControl).where(AgentPendingControl.id == pending_id,
                                                      AgentPendingControl.owner_id == owner).with_for_update())
    if row is None:
        raise DomainError("NOT_FOUND", "That pending agent action was not found.", 404)
    if row.status != "pending":
        raise DomainError("PENDING_CLOSED", f"That agent action was already {row.status}.", 409,
                          data={"status": row.status})
    if row.expires_at <= now():
        # Refused, so this transaction rolls back: pending_list() marks the row expired.
        raise DomainError("PENDING_EXPIRED", "That agent action expired; ask again.", 409)
    row.decided_at, row.decided_via = now(), via
    if not confirm:
        row.status = "declined"
        emit(db, owner, "agent.changed", row.run_id or row.id)
        return public_pending(row)
    # The proposer stays on record; the owner's confirmation is what lets it through.
    confirmed = Actor(actor.kind, credential_id=actor.credential_id, reason="confirmed:" + row.id)
    args = row.arguments
    if row.tool == "control.start":
        result = start_run(db, owner, row.account_id or account(db, owner),
                           StartInput(task_id=args["task_id"], harness=args.get("harness", "claude"),
                                      execution_target_id=execution_target_id or args.get("execution_target_id")),
                           actor=confirmed)
        run = coordinator.load(db, owner, result["id"])
        coordinator.note(run, "confirmed", pending_id=row.id, proposed_by=row.actor, via=via)
        row.run_id = result["id"]
    else:
        run = coordinator.load(db, owner, args["run_id"])
        coordinator.note(run, "confirmed", pending_id=row.id, proposed_by=row.actor, via=via)
        result = send_message(db, owner, args["run_id"], MessageInput(text=args["text"], mode=args["mode"]),
                              actor=confirmed)
    row.status, row.result = "confirmed", {"run_id": result.get("id")}
    # Identical proposals still open (from before duplicates were folded) are settled by this one.
    for twin in db.scalars(select(AgentPendingControl).where(
            AgentPendingControl.owner_id == owner, AgentPendingControl.tool == row.tool, AgentPendingControl.arguments == row.arguments,
            AgentPendingControl.status == "pending", AgentPendingControl.id != row.id)):
        twin.status, twin.decided_at, twin.decided_via, twin.result = "superseded", now(), via, {"by": row.id}
    return {**public_pending(row), "run": result}


def spoken_confirmation(db, owner, args):
    """control_confirm from Eri: only a later request from the owner that is itself a short yes, for the newest
    pending action of that account (so the yes is unambiguous). Declining needs no proof."""
    from .session_models import AgentPendingControl

    work = current_work(db)
    if work is None or work.credential_id:
        raise DomainError("CONFIRM_IN_APP", "Confirm this on the remotea page.", 403)
    row = pending_for(db, owner, work, args.pending_id)
    if args.decision == "decline" or row.status != "pending":
        return "chat"  # Declining needs no proof; a closed one is refused by decide().
    via = owners_yes(db, owner, row, work, chosen=choice_names(db, owner, row, args.execution_target_id))
    if not via:
        raise DomainError("CONFIRMATION_REQUIRED",
                          "Only the owner's own yes confirms this: a later message of theirs that starts with yes, "
                          "okay, sure, go ahead or do it, is short, and asks for nothing else (not a question, not "
                          "\u201cyes but \u2026\u201d). Ask them, or let them tap Confirm on the remotea page.",
                          409, data={"pending_id": row.id})
    return via


def choice_names(db, owner, row, choice):
    """The names the owner may answer with for `choice` (an entry of the proposal's targets): its saved name and
    host target, so "jarvis" or "run it in jarvis" confirms a start that asked where to run."""
    if not choice or row.tool != "control.start":
        return ()
    need = (readiness(db, owner, row).get("need") or {})
    for option in target_options(need):
        if option["execution_target_id"] == choice:
            names = {option["name"], option.get("repo", "").rstrip("/").rsplit("/", 1)[-1]}
            if option["saved"]:
                target = db.get(ExecutionTarget, choice)
                if target is not None:
                    names |= {target.name, target.host_target}
            return tuple(n for n in names if n)
    return ()


def not_proposed_by(work):
    from .session_models import AgentPendingControl

    return or_(AgentPendingControl.work_id.is_(None), AgentPendingControl.work_id != work.id)


def pending_for(db, owner, work, pending_id):
    """The named pending action, or without an id the newest one of this account not proposed by this request."""
    from .session_models import AgentPendingControl

    if pending_id:
        row = db.get(AgentPendingControl, pending_id)
    else:
        row = db.scalar(select(AgentPendingControl).where(
            AgentPendingControl.owner_id == owner, AgentPendingControl.account_id == work.account_id,
            AgentPendingControl.status == "pending", AgentPendingControl.expires_at > now(),
            not_proposed_by(work)).order_by(AgentPendingControl.created_at.desc()).limit(1))
    if row is None or row.owner_id != owner:
        raise DomainError("NOT_FOUND", "That pending agent action was not found.", 404)
    return row


def journal_control(db, owner, tool, command_id, result):
    """Agent control in the action journal: visible, never undoable (stage F, M5)."""
    from uuid import uuid4

    from .access import actor as account
    from .models import ActionChange
    from .work_crypto import seal

    if not get_settings().integration_encryption_key or not isinstance(result, dict):
        return
    title = result.get("summary") or (result.get("task") or {}).get("title") or tool
    entity = result.get("run_id") or result.get("id") or command_id
    after = {"title": str(title)[:300], "status": result.get("status") or result.get("state"),
             "_undo_blocked": "Agent control cannot be undone here; stop the run on the remotea page instead."}
    db.add(ActionChange(id=str(uuid4()), owner_id=owner, account_id=account(db, owner), command_id=command_id,
                        tool=tool, entity_kind="agent_run", entity_id=str(entity)[:36], before_ciphertext=None,
                        after_ciphertext=seal(after)))


def mutate(db, owner, tool, args, command_id):
    """domain.mutate entrance for control.* (voice, chat). start and message are two-phase (propose, then the
    owner confirms); status, stop and answer stay immediate."""
    result = control_command(db, owner, tool, args)
    if tool == "control.start" and isinstance(result, dict) and result.get("status") == "pending_confirmation":
        result = proposal(db, owner, result)
    if not (isinstance(result, dict) and result.get("status") == "needs_choice"):
        journal_control(db, owner, tool, command_id, result)  # needs_choice did nothing; nothing to journal.
    return result


CHOOSE = ("Nothing was done: more than one run fits. Read each candidate to the owner as task · host · branch · state "
          "and ask which one they mean; then call {tool} again with that candidate's run_id.")
NAME_IT = "Name the task, host and branch this affects when you tell the owner ({label})."


def needs_choice(tool, error):
    return {"status": "needs_choice", "tool": tool.replace(".", "_"), "message": error.message,
            "candidates": error.data["candidates"], "next": CHOOSE.format(tool=tool.replace(".", "_"))}


def control_command(db, owner, tool, args):
    """E-Q6: a command whose run could be one of several returns needs_choice (each candidate as task · host · branch
    · state) instead of acting; one that acts names its recipient so Eri's sentence is grounded."""
    try:
        result = control_dispatch(db, owner, tool, args)
    except DomainError as error:
        if error.code == "AMBIGUOUS_RUN" and (error.data or {}).get("candidates"):
            return needs_choice(tool, error)
        raise
    return result


def named(db, result, run_id):
    """The result with its recipient: who the command affected (or, for a proposal, will affect)."""
    run = db.get(AgentRun, run_id) if run_id else None
    if run is None or not isinstance(result, dict):
        return result
    who = recipient(db, run)
    return {**result, "recipient": who, "next": (NAME_IT.format(label=who["label"]) + " " + result.get("next", "")).strip()}


def pending_choices(db, owner, work):
    """control_confirm without a pending_id: the distinct proposals of this account it could mean (newest first)."""
    from .session_models import AgentPendingControl

    rows = db.scalars(select(AgentPendingControl).where(
        AgentPendingControl.owner_id == owner, AgentPendingControl.account_id == work.account_id,
        AgentPendingControl.status == "pending", AgentPendingControl.expires_at > now(), not_proposed_by(work))
        .order_by(AgentPendingControl.created_at.desc()).limit(10))
    seen, unique = set(), []
    for row in rows:
        key = (row.tool, json.dumps(row.arguments, sort_keys=True))
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def pending_recipient(db, owner, row):
    """A proposal as a candidate: what it would affect (the run for a message; task, place and branch for a start)."""
    run = db.get(AgentRun, row.run_id) if row.run_id else None
    if row.tool == "control.message" and run is not None:
        return {**recipient(db, run), "pending_id": row.id, "summary": row.summary}
    task = db.get(Task, (row.arguments or {}).get("task_id")) if (row.arguments or {}).get("task_id") else None
    target_id = (row.arguments or {}).get("execution_target_id")
    target = db.get(ExecutionTarget, target_id) if target_id else None
    title = (task.title if task else "a task")[:120]
    place = ((target.name or target.host_target) if target else "no repository chosen yet")[:80]
    branch = branch_for(db, owner, task) if task else ""
    return {"pending_id": row.id, "run_id": None, "task": title, "host": place, "branch": branch, "state": "not started",
            "summary": row.summary, "label": f"{title} · {place} · {branch or 'new branch'} · not started"}


def control_dispatch(db, owner, tool, args):
    from .access import actor as account
    from .agent_actor import current

    agent_link.gate(db, owner)
    owner_control(db, owner)
    actor = current()
    if tool in ("control.start", "control.message"):
        result = propose(db, owner, account(db, owner), tool, args, actor)
        if tool != "control.message":
            return result
        from .session_models import AgentPendingControl

        row = db.get(AgentPendingControl, result.get("pending_id") or result.get("id") or "")
        return named(db, result, (row.run_id if row is not None else None) or (result.get("run") or {}).get("id"))
    if tool == "control.confirm":
        work = current_work(db)
        if args.decision == "confirm" and not args.pending_id and work is not None and not work.credential_id:
            choices = pending_choices(db, owner, work)
            if len(choices) > 1:
                return {"status": "needs_choice", "tool": "control_confirm",
                        "message": "Several proposals are waiting for the owner's confirmation.",
                        "candidates": [pending_recipient(db, owner, r) for r in choices],
                        "next": "Nothing was confirmed. Read each waiting proposal to the owner as task · host · branch "
                                "· state and ask which one they mean. A spoken yes confirms only the newest (call "
                                "control_confirm with its pending_id after their yes); an older one is confirmed on the "
                                "remotea page, or after declining the newer ones."}
        via = spoken_confirmation(db, owner, args)
        row = pending_for(db, owner, work, args.pending_id)
        confirm = args.decision == "confirm"
        target_id = chosen_target(db, owner, row, args.execution_target_id) if confirm else None
        result = decide(db, owner, row.id, confirm=confirm, via=via, actor=actor, execution_target_id=target_id)
        return named(db, result, (result.get("run") or {}).get("id")) if confirm else result
    if tool == "control.approve":
        run = coordinator.load(db, owner, run_for(db, owner, args, tool), lock=False)
        if run.state != "awaiting_approval":
            raise DomainError("NOT_AWAITING_APPROVAL",
                              f"This run is {run.state}" + (f" ({REASON_TEXT.get(run.state_reason) or run.state_reason})"
                                                           if run.state_reason else "") +
                              ", not awaiting approval; only a run the judge passed can be approved here. Approving "
                              "from the remotea page completes the task in any state.", 409,
                              data={"run_id": run.id, "state": run.state, "state_reason": run.state_reason})
        result = approve(db, owner, run.id, ApproveInput(expected_revision=run.state_revision), actor=actor)
        return named(db, {"status": "approved", "run_id": run.id, "task": result["task"], "state": result["state"],
                          "next": "The task is complete (approved by the owner, a human decision) and the run is "
                                  "settling; nothing else is needed."}, run.id)
    if tool == "control.plan_review":
        from .agent_plan import PlanDecision, PlanReviewInput, review_plan

        run = coordinator.load(db, owner, run_for(db, owner, args, tool), lock=False)
        plan = coordinator.frozen(db, run)
        if run.state != "needs_you" or run.state_reason != "plan_review" or plan is None or not plan.proposals:
            raise DomainError("NO_PLAN_REVIEW", f"This run is {run.state}" +
                              (f" ({run.state_reason})" if run.state_reason else "") +
                              ", not waiting for a plan review.", 409,
                              data={"run_id": run.id, "state": run.state, "state_reason": run.state_reason})
        body = PlanReviewInput(decisions=[PlanDecision(**d.model_dump()) for d in args.decisions], verify=args.verify,
                               completion_policy=args.completion_policy, expected_revision=run.state_revision)
        result = review_plan(db, owner, run.id, body, actor=actor)
        return named(db, {"status": "plan_reviewed", "run_id": run.id, "task": result["task"], "state": result["state"],
                          "plan_revision": (result.get("plan") or {}).get("revision"),
                          "next": "The plan is frozen and the agent is executing against it."}, run.id)
    if tool == "control.stop":
        run_id = run_for(db, owner, args, tool)
        return named(db, stop_run(db, owner, run_id, StopInput(mode=args.mode), actor=actor), run_id)
    if tool == "control.answer":
        # The question or interview id names its run exactly, so an answer is never ambiguous; it still names it.
        if args.interview_id:
            from .agent_plan import InterviewAnswers, answer_interview

            result = answer_interview(db, owner, args.interview_id, InterviewAnswers(answers=args.responses), actor=actor)
        else:
            result = answer_question(db, owner, args.question_id,
                                     AnswerInput(answers=[AnswerItem(**a.model_dump()) for a in args.answers]), actor=actor)
        return named(db, result, result.get("run_id") if isinstance(result, dict) else None)
    raise DomainError("INVALID_ARGUMENT", "That action is not available.")


def spoken_run(db, run):
    """A compact run for voice: what it is doing, what it waits for, and the ids to answer with."""
    from .agent_plan import public_interview

    item = public_run(db, run)
    questions = [{"question_id": q.id, "remote_answerable": q.remote_answerable, "status": q.status,
                  "items": [{"item_id": i.get("id"), "text": i.get("text"), "multi": bool(i.get("multi")),
                             "choices": [{"id": c.get("id"), "label": c.get("label")} for c in i.get("choices") or []]}
                            for i in q.items or []]}
                 for q in coordinator.open_questions(db, run)]
    interview = coordinator.open_interview(db, run)
    linear = (run.sync_state or {}).get("linear") or {}
    spoken = {"run_id": run.id, "task": item["task"], "state": run.state, "why": item["reason_text"],
              # What the agent is doing now in one line, as of summary_at; say it as "the agent is …".
              "doing": run.summary or None, "doing_as_of": run.summary_at if run.summary else None,
              "turns": f"{run.turn_count}/{run.turn_cap}", "tool": run.tool, "branch": run.branch,
              "host_connected": (item["host"] or {}).get("connected"), "since": item["state_since"],
              "questions": questions,
              "interview": {k: v for k, v in public_interview(interview).items() if k in ("id", "questions")}
              if interview else None,
              "linear": {k: (v or {}).get("status") for k, v in linear.items()} or None}
    if run.state == "needs_you" and run.state_reason == "verify_infra":
        # Not a failing test: the verification could not run, and the owner fixes the host.
        spoken["why"] = "host setup problem: " + (item["reason_detail"] or "the verification could not run on the host")
    if run.state == "needs_you" and run.state_reason == "plan_review":
        plan = coordinator.frozen(db, run)
        if plan is not None and plan.proposals:
            # What the agent wants to change, with the ids control_plan_review decides on (criteria ids; verify and
            # completion_policy are accept/keep flags).
            spoken["plan_review"] = {"plan_revision": plan.revision, "proposals": plan.proposals,
                                     "next": "Read each proposal to the owner; then call control_plan_review with "
                                             "their accept/keep decision per criterion id (text for their own "
                                             "wording), and verify / completion_policy when proposed."}
    if run.state == "awaiting_approval":
        packet = review_packet(db, run)
        spoken["awaiting_approval"] = {**judge_summary(db, run), "review_packet": packet,
                                       "next": "Read the review packet's text (and the judge's summary) to the owner, "
                                               f"naming the run ({recipient(db, run)['label']}). Approving is the owner's "
                                               "decision, not something verification established: control_approve "
                                               "completes the task on their say-so; control_message sends changes back "
                                               "instead."}
    elif run.state == "needs_you" and run.state_reason != "plan_review":
        spoken["review_packet"] = review_packet(db, run)
    # How old each fact is (E-S6): say it before a decision ("verify passed 3 min ago …; host seen 5 s ago").
    spoken["freshness"] = freshness_text(item)
    return spoken


PR_NUMBER = re.compile(r"/(?:pull|merge_requests)/(\d+)")


def spoken_ago(at):
    if at is None:
        return ""
    seconds = max(0, (now() - at).total_seconds())
    if seconds < 90:
        return "just now"
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"{minutes} min ago"
    hours = round(minutes / 60)
    return f"{hours} h ago" if hours < 36 else f"{round(hours / 24)} days ago"


def review_packet(db, run):
    """E-Q3: what a reviewer needs beside Approve, as control_status reads it out (the page's remotea-model
    reviewPacket builds the same from GET /runs/{id}): the frozen criteria with the latest verdict per criterion, the
    latest verification (exit code, sandboxed or not, workspace complete or not, age), the pull requests and branch
    state, open questions, and the judge's full reason. `text` is one sentence, e.g. "2 of 3 criteria met; tests
    passed 4 min ago in a sandbox; PR #12 open; 1 open question"."""
    plan = coordinator.frozen(db, run)
    attempts = list(db.scalars(select(AgentAttempt).where(AgentAttempt.run_id == run.id)
                               .order_by(AgentAttempt.created_at.desc()).limit(40)))
    judge = next((a for a in attempts if a.kind == "judge" and a.status == "done"), None)
    verify = next((a for a in attempts if a.kind == "verify" and a.initiator != "host"), None)
    verdicts = {c.get("id"): c for c in ((judge.result or {}).get("criteria") or [])} if judge else {}
    status_word = {"met": "met", "not_met": "not met", "unclear": "unknown"}
    criteria = [{"id": c.get("id"), "text": c.get("text"),
                 "status": status_word.get((verdicts.get(c.get("id")) or {}).get("status"), "unknown")}
                for c in (plan.criteria if plan else []) or []]
    met = sum(c["status"] == "met" for c in criteria)
    earlier = judge is not None and judge.plan_revision != run.plan_revision
    parts = [f"{met} of {len(criteria)} criteria met" + (" (judged against an earlier plan)" if earlier else "")
             if criteria else "no success criteria"]
    tests = None
    if verify is None:
        parts.append("tests not run yet" if (plan and (plan.verify or {}).get("verify_id"))
                     else "no verify command, so no test result is recorded")
    else:
        result = verify.result or {}
        word = {"passed": "tests passed", "failed": "tests failed" + (f" (exit {result['exit_code']})"
                                                                     if result.get("exit_code") is not None else ""),
                "stale": "tests stale (the code changed during the run)", "timed_out": "tests timed out",
                "running": "tests running"}.get(verify.status, f"tests {verify.status.replace('_', ' ')}")
        if str(result.get("category") or "").startswith("infra_"):
            # The check could not run (host setup): it says nothing about the code; name the fix instead.
            word = f"verification could not run ({result.get('action') or verify.status.replace('_', ' ')})"
        sandbox = {True: " in a sandbox", False: " outside a sandbox"}.get(result.get("isolated"), "")
        line = f"{word} {spoken_ago(verify.updated_at or verify.created_at)}{sandbox}".replace("  ", " ")
        if result.get("workspace_complete") is False:
            line += " (workspace incomplete: " + "; ".join(result.get("workspace_reasons") or ["not fully read"])[:200] + ")"
        if (verify.turn_seq, verify.plan_revision) != (run.turn_seq, run.plan_revision):
            line += " (for an earlier turn or plan)"
        parts.append(line)
        tests = {"verify_id": verify.verify_id, "status": verify.status, "exit_code": result.get("exit_code"),
                 "isolated": result.get("isolated"), "isolation_reason": result.get("isolation_reason"),
                 "workspace_complete": result.get("workspace_complete"),
                 "workspace_reasons": result.get("workspace_reasons") or [], "at": verify.updated_at}
    host = db.get(AgentHost, run.host_id) if run.host_id else None
    session = db.get(AgentSession, run.session_pk) if run.session_pk else None
    changes = changes_info(db, run, host, session)
    numbers = [m.group(1) for m in (PR_NUMBER.search(u) for u in changes["pr_urls"]) if m]
    if changes["pr_urls"]:
        label = ("PR #" + ", #".join(numbers)) if numbers else ("a PR" if len(changes["pr_urls"]) == 1 else "PRs")
        parts.append(f"{label} {'merged' if changes.get('merged') else 'open'}")
    else:
        pending = [f"{changes[k]} {k}" for k in ("unpushed", "uncommitted") if changes.get(k)]
        parts.append("no PR recorded" + (f" ({', '.join(pending)} on {run.branch})" if pending else ""))
    interview = coordinator.open_interview(db, run)
    open_count = len(coordinator.open_questions(db, run)) +(len(interview.questions or []) if interview else 0)
    parts.append(f"{open_count} open question{'s' if open_count != 1 else ''}" if open_count else "no open questions")
    verdict = (judge.result or {}) if judge else {}
    return {"text": "; ".join(parts), "criteria": criteria, "criteria_met": met, "criteria_total": len(criteria),
            "judged_earlier_plan": earlier, "tests": tests, "pull_requests": changes["pr_urls"], "branch": run.branch,
            "open_questions": open_count,
            "reason": "\n\n".join(t for t in (verdict.get("summary"), verdict.get("gap")) if t)[:4000]}


def judge_summary(db, run):
    """The last judge verdict on a run (summary and any gap), for Eri to read out before an approval."""
    verdict = db.scalar(select(AgentAttempt).where(AgentAttempt.run_id == run.id, AgentAttempt.kind == "judge",
                                                   AgentAttempt.status == "done")
                        .order_by(AgentAttempt.created_at.desc()).limit(1))
    result = (verdict.result or {}) if verdict is not None else {}
    return {"verdict": result.get("verdict"), "summary": (result.get("summary") or "")[:2000],
            "gap": (result.get("gap") or "")[:2000], "criteria": result.get("criteria") or []}


def status(db, owner, args):
    """control_status: active runs (or one), with open questions; recently settled runs in brief."""
    if not agent_link.enabled(db, owner):
        return {"enabled": False, "message": "remotea is off; turn it on in Settings to run agents."}
    runs = []
    if args.get("run_id") or args.get("task_id") or args.get("task"):
        from .agent_control_schema import Target

        target = Target.model_validate({k: args[k] for k in ("run_id", "task_id", "task") if args.get(k)})
        if target.run_id:
            runs = [coordinator.load(db, owner, target.run_id, lock=False)]
        else:
            task_id = target.task_id or task_by_name(db, owner, target.task)
            runs = list(db.scalars(select(AgentRun).where(AgentRun.owner_id == owner, AgentRun.task_id == task_id)
                                   .order_by(AgentRun.created_at.desc()).limit(1)))
    else:
        runs = list(db.scalars(select(AgentRun).where(AgentRun.owner_id == owner,
                                                      AgentRun.state.not_in(TERMINAL_RUN_STATES))
                               .order_by(AgentRun.created_at.desc()).limit(10)))
    recent = list(db.scalars(select(AgentRun).where(
        AgentRun.owner_id == owner, AgentRun.state.in_(TERMINAL_RUN_STATES),
        AgentRun.finished_at >= now() - timedelta(hours=24)).order_by(AgentRun.finished_at.desc()).limit(5)))
    hosts = []
    for h in db.scalars(select(AgentHost).where(AgentHost.owner_id == owner)):
        degraded = agent_link.build_identity(h)["degraded_reason"]
        # A degraded journal refuses starts, messages and stops until it clears; Eri should say so.
        hosts.append({"name": h.name, "connected": agent_link.connected(h), "last_seen_at": h.last_seen_at,
                      **({"degraded_reason": degraded, "warning": f"Host journal degraded: {degraded}"} if degraded else {})})
    # One line per unready host or target with its first fix (the setup page's wording): why a run cannot start.
    setup = setup_report(db, owner)["lines"]
    return encode({"enabled": True, "hosts": hosts, "runs": [spoken_run(db, r) for r in runs],
                   **({"setup": setup, "setup_hint": SETUP_HINT} if setup else {}),
                   "pending": spoken_pending(db, owner),
                   "recently_finished": [{"run_id": r.id, "task": (db.get(Task, r.task_id).title if r.task_id else ""),
                                          "state": r.state, "finished_at": r.finished_at} for r in recent
                                         if r not in runs]})


def spoken_pending(db, owner):
    """Open control_start / control_message proposals as Eri reads them: the summary to confirm, and for a start
    without a place, the target options control_confirm accepts."""
    items = []
    for row in pending_list(db, owner):
        item = {"pending_id": row["id"], "tool": row["tool"], "summary": row["summary"], "expires_at": row["expires_at"],
                "task_id": row.get("task_id"), "run_id": row.get("run_id")}
        if row.get("needs_target"):
            item.update(needs_target=True, targets=target_options(row.get("need") or {}))
        if row.get("problems"):
            item["problems"] = row["problems"]
        items.append(item)
    return items


def open_proposal(db, owner, account):
    """Does this account have a proposal waiting for the owner (pending, not expired)? work_runner preloads the
    confirm tools for the next turn when it does, so a bare yes needs no discovery round."""
    from .session_models import AgentPendingControl

    return db.scalar(select(AgentPendingControl.id).where(
        AgentPendingControl.owner_id == owner, AgentPendingControl.account_id == account,
        AgentPendingControl.status == "pending", AgentPendingControl.expires_at > now()).limit(1)) is not None
