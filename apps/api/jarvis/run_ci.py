"""GitHub CI as a run's final verification (docs/AGENT_CONTROL.md, "CI as the final verification").

A plan whose verify entry is only a pre-push check (its id in `agent_ci_quick_verify_ids`, default `quick`:
scripts/verify-quick.sh) is not finished by that check's pass. When the judge finds the criteria met, the run waits in
`awaiting_ci` until the pull request head's checks are terminal: green completes it as before (approval or
auto_complete), red sends the agent a gap with the failed jobs and the failing log's last lines, and nothing terminal
within `agent_ci_wait_seconds` (default 90 min) needs you (`ci_timeout`). A pass of any other verify entry (the full
`./scripts/verify.sh`) still counts on its own. Policy pr_merged is unchanged: a merge is the human's final word.

Eridani holds no GitHub credential. It asks the run's host with the read-only `ci.status` command (sent only to a host
whose hello caps list `ci.status`); the host runs `gh` with its own login in the run's worktree and answers in the
command outcome's `result`. Each answer is recorded as an attempt of kind `ci` (verify_id `github_ci`), one per run
tuple and PR head, updated in place on every poll, beside the local verify attempts.
"""

import re
from datetime import datetime, timedelta

from sqlalchemy import select

from . import agent_automation as automation
from . import run_coordinator as coordinator
from .config import get_settings
from .domain import emit
from .models import now
from .session_models import AgentAttempt, AgentCommand, AgentRun

KIND = "ci"
SOURCE = "github_ci"
COMMAND = "ci.status"
POLL = timedelta(minutes=2)
LOG_LINES = 40
LINE_CHARS = 300
CHECKS = 50
PUSH_PROMPTS = 2  # "Push and open the pull request" at most this often per run; then the wait runs out.
# Fail fast: a check that failed while others still run is a failure now. If the host has no log for it yet, wait at
# most this many more polls for one before sending the gap without it.
LOG_WAITS = 2
# needs_you reasons a green CI run on the current head un-parks, as a late local pass does (verify_infra goes to the
# judge, which counts green CI as the verification; the CI reasons were met already and go on to approval).
UNPARK = ("verify_infra", "ci_timeout", "ci_unavailable")
# Parked runs are asked less often after the first wait, and not at all after a day.
PARKED_POLL = timedelta(minutes=10)
PARKED_FOR = timedelta(hours=24)
# gh's own buckets (`gh pr checks --json bucket`); `state` is folded onto them when an older gh has no bucket.
BUCKETS = ("pass", "fail", "pending", "skipping", "cancel")
PASS_STATES = {"SUCCESS", "NEUTRAL"}
SKIP_STATES = {"SKIPPED"}
FAIL_STATES = {"FAILURE", "ERROR", "TIMED_OUT", "ACTION_REQUIRED", "STARTUP_FAILURE"}
CANCEL_STATES = {"CANCELLED"}
# What the run's CI looks like overall (`state` on the attempt result and on the read model's `ci`):
# passed and failed are terminal (failed as soon as one check failed, even while others run: fail fast); pending (some check still running), absent (a pull request without checks yet),
# no_pr (no pull request for the branch), unpushed (the worktree's HEAD is not the PR head) and cancelled (every
# check finished, at least one was cancelled and none failed) keep the run waiting.
STATES = ("passed", "failed", "pending", "absent", "no_pr", "unpushed", "cancelled")
_SHA = re.compile(r"^[0-9a-f]{7,64}$")
_RUN_URL = re.compile(r"^(https://github\.com/[^/\s]+/[^/\s]+/actions/runs/\d+)")


def quick_ids():
    return {v.strip() for v in get_settings().agent_ci_quick_verify_ids.split(",") if v.strip()}


def required(plan):
    """Whether a met verdict on this plan still needs green CI: its verify entry is a quick pre-push check and its
    completion policy is code_ready or pr_open (pr_merged ends on the merge itself)."""
    verify_id = ((plan.verify or {}).get("verify_id") if plan else None) or ""
    policy = (plan.completion_policy if plan else None) or "code_ready"
    return bool(verify_id) and verify_id in quick_ids() and policy in ("code_ready", "pr_open")


# ---------------------------------------------------------------- the host's answer (pure)


def url(value):
    text = str(value or "")[:500]
    return text if text.startswith("https://") else None


def bucket(check):
    given = str(check.get("bucket") or "").lower()
    if given in BUCKETS:
        return given
    state = str(check.get("state") or check.get("conclusion") or "").upper()
    return ("pass" if state in PASS_STATES else "skipping" if state in SKIP_STATES else "fail" if state in FAIL_STATES
            else "cancel" if state in CANCEL_STATES else "pending")


def log_lines(failed_logs, limit=LOG_LINES):
    """The last `limit` lines across the failed jobs' logs, each clipped, ANSI codes and blank lines dropped."""
    lines = []
    for item in failed_logs or []:
        if not isinstance(item, dict):
            continue
        raw = item.get("lines") if isinstance(item.get("lines"), list) else str(item.get("log") or "").splitlines()
        name = str(item.get("name") or "")[:120]
        body = [re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", str(line)).rstrip() for line in raw]
        body = [line if len(line) <= LINE_CHARS else line[:LINE_CHARS - 1] + "…" for line in body if line.strip()]
        if body:
            lines += ([f"== {name}"] if name else []) + body
    return lines[-limit:]


def parse(result):
    """The host's `ci.status` result → {state, head_sha, pr_url, pr_number, run_url, checks, failed, log,
    worktree_head}. Never raises: anything malformed reads as no pull request or no checks."""
    result = result if isinstance(result, dict) else {}
    pr = result.get("pr") if isinstance(result.get("pr"), dict) else None
    head = str((pr or {}).get("head_sha") or (pr or {}).get("headRefOid") or "").lower()[:64]
    head = head if _SHA.match(head) else ""
    local = str(result.get("worktree_head") or "").lower()[:64]
    local = local if _SHA.match(local) else ""
    checks = []
    for item in (result.get("checks") or [])[:CHECKS]:
        if isinstance(item, dict) and item.get("name"):
            checks.append({"name": str(item["name"])[:120], "bucket": bucket(item),
                           "state": str(item.get("state") or "")[:40], "link": url(item.get("link")),
                           "workflow": str(item.get("workflow") or "")[:120]})
    buckets = {c["bucket"] for c in checks}
    if pr is None:
        state = "no_pr"
    elif local and head and not local.startswith(head) and not head.startswith(local):
        state = "unpushed"
    elif not checks:
        state = "absent"
    elif "fail" in buckets:
        state = "failed"
    elif "pending" in buckets:
        state = "pending"
    elif "cancel" in buckets:
        state = "cancelled"
    else:
        state = "passed"
    runs = [m.group(1) for c in checks if c["link"] and (m := _RUN_URL.match(c["link"]))]
    failing = [c for c in checks if c["bucket"] == "fail"]
    run_url = next((m.group(1) for c in failing if c["link"] and (m := _RUN_URL.match(c["link"]))), None) or \
        (runs[0] if runs else None)
    pr_url = url((pr or {}).get("url"))
    return {"state": state, "head_sha": head, "pr_url": pr_url, "running": "pending" in buckets,
            "pr_number": (pr or {}).get("number") if isinstance((pr or {}).get("number"), int) else None,
            "run_url": run_url or (f"{pr_url}/checks" if pr_url else None), "checks": checks,
            "failed": [{"name": c["name"], "link": c["link"]} for c in failing],
            "log": log_lines(result.get("failed_logs")) if failing else [], "worktree_head": local or None}


# ---------------------------------------------------------------- records


def latest(db, run, *, current=False):
    query = select(AgentAttempt).where(AgentAttempt.run_id == run.id, AgentAttempt.kind == KIND)
    if current:
        query = query.where(AgentAttempt.generation == run.generation, AgentAttempt.turn_seq == run.turn_seq,
                            AgentAttempt.plan_revision == run.plan_revision)
    return db.scalar(query.order_by(AgentAttempt.updated_at.desc()).limit(1))


def passed_now(db, run):
    """A green CI record for the run's current (generation, turn, plan revision)."""
    attempt = latest(db, run, current=True)
    return attempt is not None and attempt.status == "passed"


def record(db, run, tuple_, facts, clock):
    """Upsert the CI attempt for this tuple and PR head (one row per head, updated on every poll)."""
    generation, turn_seq, plan_revision = tuple_
    attempt = db.scalar(select(AgentAttempt).where(
        AgentAttempt.run_id == run.id, AgentAttempt.kind == KIND, AgentAttempt.generation == generation,
        AgentAttempt.turn_seq == turn_seq, AgentAttempt.plan_revision == plan_revision,
        AgentAttempt.workspace_fingerprint == facts["head_sha"]).with_for_update())
    if attempt is None:
        attempt = AgentAttempt(owner_id=run.owner_id, run_id=run.id, generation=generation, kind=KIND,
                               turn_seq=turn_seq, plan_revision=plan_revision, verify_id=SOURCE,
                               workspace_fingerprint=facts["head_sha"], initiator="host", result={}, created_at=clock)
        db.add(attempt)
    polls = (attempt.result or {}).get("polls", 0) + 1
    attempt.status = facts["state"] if facts["state"] in ("passed", "failed") else "pending"
    attempt.result = {"source": SOURCE, **facts, "polls": polls, "checked_at": clock.isoformat()}
    attempt.updated_at = clock
    if attempt.status in ("passed", "failed") and attempt.created_at:
        attempt.latency_ms = int((clock - attempt.created_at).total_seconds() * 1000)
    db.flush()
    return attempt


def ci_state(run):
    return dict((run.sync_state or {}).get("ci") or {})


def save(run, **values):
    run.sync_state = {**(run.sync_state or {}), "ci": {**ci_state(run), **values}}


# ---------------------------------------------------------------- the gate and the poll


def gate(db, run, plan, actor):
    """Called by run_coordinator.judged_met before approval or auto_complete. None when CI is not needed or already
    green for this turn; else the run waits in awaiting_ci (or needs you when the host cannot report CI)."""
    if not required(plan) or passed_now(db, run):
        return None
    if not coordinator.host_cap(db, run.host_id, COMMAND):
        return coordinator.transition(db, run, "needs_you", actor=actor, reason="ci_unavailable",
                                      detail="The host does not report GitHub CI (no ci.status capability).")
    coordinator.transition(db, run, "awaiting_ci", actor=actor, reason="ci_pending")
    save(run, since=now().isoformat(), turn_seq=run.turn_seq, error=None)
    request(db, run)
    return run


def request(db, run, clock=None):
    """Ask the host for the PR head's checks now (`ci.status`, read-only, 2-minute expiry)."""
    clock = clock or now()
    row = coordinator.command(db, run, COMMAND, {"branch": run.branch, "turn_seq": run.turn_seq, "log_lines": LOG_LINES},
                              expires_at=clock + POLL, source="eridani")
    save(run, requested_at=clock.isoformat())
    return row


def in_flight(db, run, clock):
    return db.scalar(select(AgentCommand.id).where(
        AgentCommand.run_id == run.id, AgentCommand.kind == COMMAND, AgentCommand.status.in_(("queued", "sent")),
        AgentCommand.expires_at > clock).limit(1)) is not None


def wait_seconds():
    return get_settings().agent_ci_wait_seconds


def stamp(value):
    try:
        return datetime.fromisoformat(value) if isinstance(value, str) and value else None
    except ValueError:
        return None


def sweep(db, clock):
    """From run_coordinator.sweep: poll every POLL while a run awaits CI; past the wait, the run needs you."""
    for run in db.scalars(select(AgentRun).where(AgentRun.state == "awaiting_ci").with_for_update(skip_locked=True)):
        if run.paused or automation.blocked(run):
            continue
        since = run.state_since or clock
        if clock - since >= timedelta(seconds=wait_seconds()):
            coordinator.transition(db, run, "needs_you", reason="ci_timeout", detail=timeout_text(db, run, clock))
            continue
        asked = stamp(ci_state(run).get("requested_at"))
        if (asked is None or clock - asked >= POLL) and not in_flight(db, run, clock):
            request(db, run, clock)
    for run in db.scalars(select(AgentRun).where(AgentRun.state == "needs_you", AgentRun.state_reason.in_(UNPARK))
                          .with_for_update(skip_locked=True)):
        if not parked_poll_due(db, run, clock):
            continue
        request(db, run, clock)


def parked_poll_due(db, run, clock):
    """A run parked on a reason green CI can clear is asked every POLL for the CI wait, then every PARKED_POLL, for
    PARKED_FOR. Only plans with a verify command, on a host that reports CI, with a turn to check."""
    if run.paused or automation.blocked(run) or not run.turn_seq or not run.branch:
        return False
    if not coordinator.host_cap(db, run.host_id, COMMAND):
        return False
    plan = coordinator.frozen(db, run)
    if plan is None or not (plan.verify or {}).get("verify_id") or not plan.criteria:
        return False
    parked = clock - (run.state_since or clock)
    if parked >= PARKED_FOR:
        return False
    asked = stamp(ci_state(run).get("requested_at"))
    every = POLL if parked < timedelta(seconds=wait_seconds()) else PARKED_POLL
    return (asked is None or clock - asked >= every) and not in_flight(db, run, clock)


def timeout_text(db, run, clock):
    attempt = latest(db, run, current=True)
    facts = (attempt.result or {}) if attempt else {}
    waited = coordinator.span(max(60, (clock - (run.state_since or clock)).total_seconds()))
    head = (facts.get("head_sha") or "")[:7]
    error = ci_state(run).get("error")
    state = facts.get("state")
    if state == "pending":
        names = [c["name"] for c in facts.get("checks") or [] if c.get("bucket") == "pending"][:4]
        why = f"CI on {head} was still running after {waited}" + (f" ({', '.join(names)})" if names else "")
    elif state == "cancelled":
        why = f"CI on {head} was cancelled and has not run again in {waited}"
    elif state == "no_pr":
        why = f"No pull request for {run.branch or 'the branch'} after {waited}, so CI never ran"
    elif state == "unpushed":
        why = (f"The worktree's commit {(facts.get('worktree_head') or '')[:7]} was never pushed to the pull request "
               f"(its head is {head}) in {waited}")
    elif state == "absent":
        why = f"No CI checks reported on {head or 'the pull request'} in {waited}"
    elif error:
        why = f"The host could not read CI for {waited}: {str(error)[:160]}"
    else:
        why = f"The host reported no CI status in {waited}"
    return why + "."


def on_outcome(db, run, row, state, detail="", result=None):
    """The host's answer to `ci.status` (run_coordinator.on_outcome). Recorded always; it moves the run only while
    that run still awaits CI for the same tuple the question was asked for."""
    clock = now()
    if state != "executed":
        save(run, error=(detail or state)[:300], answered_at=clock.isoformat())
        emit(db, run.owner_id, "agent.changed", run.id)
        return None
    payload = row.payload or {}
    tuple_ = (row.generation, payload.get("turn_seq", run.turn_seq), payload.get("plan_revision", run.plan_revision))
    facts = parse(result)
    attempt = record(db, run, tuple_, facts, clock)
    save(run, error=None, answered_at=clock.isoformat())
    emit(db, run.owner_id, "agent.changed", run.id)
    if tuple_ != (run.generation, run.turn_seq, run.plan_revision) or run.paused or automation.blocked(run):
        return attempt
    if run.state == "needs_you" and run.state_reason in UNPARK:
        return unpark(db, run, attempt, facts)
    if run.state != "awaiting_ci":
        return attempt
    if facts["state"] == "passed":
        coordinator.note(run, "ci_passed", attempt_id=attempt.id, head_sha=facts["head_sha"])
        return coordinator.judged_met(db, run, coordinator.frozen(db, run), coordinator.judge_actor(run))
    if facts["state"] == "failed":
        waits = dict(ci_state(run).get("log_waits") or {})
        if not facts["log"] and facts["running"] and waits.get(facts["head_sha"], 0) < LOG_WAITS:
            # Failed already, but the host has no log for the failed job yet: ask again for it before the gap.
            save(run, log_waits={**waits, facts["head_sha"]: waits.get(facts["head_sha"], 0) + 1})
            return attempt
        return failed(db, run, attempt, facts)
    if facts["state"] in ("no_pr", "unpushed"):
        return push_prompt(db, run, facts)
    return attempt


def unpark(db, run, attempt, facts):
    """Green CI on the current head while the run is parked on a reason it clears: verify_infra (the local check could
    not run) goes to the judge, which counts the green CI record as the verification (agent_judge.ci_evidence);
    ci_timeout and ci_unavailable were met already and go on to approval or auto_complete. Anything else waits."""
    if facts["state"] != "passed":
        return attempt
    coordinator.note(run, "ci_passed", attempt_id=attempt.id, head_sha=facts["head_sha"], unparked=run.state_reason)
    if run.state_reason == "verify_infra":
        return coordinator.to_judging(db, run, None, reason="ci_passed")
    return coordinator.judged_met(db, run, coordinator.frozen(db, run), coordinator.judge_actor(run))


def failed(db, run, attempt, facts):
    """Red CI on the PR head: one notice per head, and the judge's gap to the agent (failed jobs, the log's last
    lines). The same head failing again after that gap means the agent did not push a fix: Davin decides."""
    from . import agent_prompts as prompts

    head = facts["head_sha"]
    seen = list(ci_state(run).get("gap_heads") or [])
    names = ", ".join(f["name"] for f in facts["failed"][:6]) or "a check"
    coordinator.notify(db, run, "agent_ci_failed", "CI failed on an agent's pull request",
                       f"{names} failed on {head[:7]}." + (" The agent is fixing it." if head not in seen else ""),
                       key=f"{run.id}:{head}")
    actor = coordinator.judge_actor(run)
    if head in seen:
        return coordinator.transition(db, run, "needs_you", actor=actor, reason="ci_failed_again",
                                      detail=f"CI failed again on {head[:7]} ({names}); no new commit was pushed.")
    save(run, gap_heads=[*seen, head][-10:])
    coordinator.note(run, "ci_failed", attempt_id=attempt.id, head_sha=head, failed=[f["name"] for f in facts["failed"]][:10])
    if run.turn_cap and run.turn_count >= run.turn_cap:
        return coordinator.transition(db, run, "capped", actor=actor, reason="turn_cap")
    coordinator.transition(db, run, "executing", actor=actor, reason="ci_failed")
    coordinator.say(db, run, prompts.ci_failed(head, facts["failed"], facts["log"], facts["run_url"]), "ci_failed",
                    actor=actor)
    return run


def push_prompt(db, run, facts):
    """No pull request for the branch, or the worktree's commit is not its head: tell the agent once per turn (at
    most PUSH_PROMPTS per run) to push and open or update it; otherwise keep waiting until the wait runs out."""
    from . import agent_prompts as prompts

    ci = ci_state(run)
    if ci.get("prompted_turn") == run.turn_seq or (ci.get("push_prompts") or 0) >= PUSH_PROMPTS or not run.surf:
        return run
    save(run, prompted_turn=run.turn_seq, push_prompts=(ci.get("push_prompts") or 0) + 1)
    actor = coordinator.judge_actor(run)
    if run.turn_cap and run.turn_count >= run.turn_cap:
        return coordinator.transition(db, run, "capped", actor=actor, reason="turn_cap")
    coordinator.transition(db, run, "executing", actor=actor, reason="ci_push")
    coordinator.say(db, run, prompts.ci_push(run.branch, facts["state"] == "no_pr"), "ci_push", actor=actor)
    return run


# ---------------------------------------------------------------- the read model


def public(db, run):
    """The run's CI as the drawer and the list show it: {state, current, head_sha, pr_url, run_url, checks, failed,
    log, checked_at, since, wait_until, error, required}. None when the run has no CI record and is not waiting."""
    attempt = latest(db, run)
    ci = ci_state(run)
    waiting = run.state == "awaiting_ci"
    if attempt is None and not waiting:
        return None
    facts = (attempt.result or {}) if attempt else {}
    current = bool(attempt) and (attempt.generation, attempt.turn_seq, attempt.plan_revision) == \
        (run.generation, run.turn_seq, run.plan_revision)
    since = run.state_since if waiting else None
    return {"state": facts.get("state") or ("requested" if waiting else None), "current": current,
            "attempt_id": attempt.id if attempt else None, "source": SOURCE,
            "head_sha": facts.get("head_sha") or None, "pr_url": facts.get("pr_url"), "run_url": facts.get("run_url"),
            "checks": facts.get("checks") or [], "failed": facts.get("failed") or [], "log": facts.get("log") or [],
            "checked_at": facts.get("checked_at"), "since": since,
            "wait_until": since + timedelta(seconds=wait_seconds()) if since else None,
            "error": ci.get("error") if waiting else None}
