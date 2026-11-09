"""GitHub CI as a run's final verification (docs/AGENT_CONTROL.md, "CI as the final verification").

The ruleset (Davis, 2026-11-05, revised 2026-11-07: CI is preferred wherever it exists), per plan under completion
policy code_ready or pr_open:

- a quick local check and CI: the quick check runs after every turn, green CI on the pull request head finishes the run
  (the best setup);
- CI only (the plan names no verify entry): green CI finishes the run, nothing runs locally;
- a full local check (any entry not in `agent_ci_quick_verify_ids`) and CI: the local check is skipped and CI stands in
  for it: after each turn the run waits for CI on the PR head, green goes to the judge (which counts it as the
  verification), red sends the agent the failed jobs (`stands_in`, `verify_by_ci`);
- a local check only (the pull request has no checks): the local pass finishes the run, CI is not waited for;
- neither: the run stops in `needs_you(verify_manual)`, "needs your verification": Davis checks the work and
  approves it as verified (recorded as a manual verification) or describes what is wrong and sends the agent back.

Nothing is added automatically: no quick check is written into a repository, no CI workflow. Policy pr_merged is
unchanged: a merge is the human's final word.

Whether a repository has CI is learned, not configured (`target_ci`): the newest CI answer for any run on the same
host target says so (checks reported: yes; none on `ABSENT_POLLS` polls: no). Until a first answer, a full local entry
runs locally and the run then waits for CI once; a target known to have no CI never waits.

Mechanics: a met verdict is not final while CI is required (`required`). The run waits in `awaiting_ci` until the pull
request head's checks are terminal: green completes it (approval or auto_complete), red sends the agent a gap with the
failed jobs and the failing log's last lines, a pull request with no checks on `ABSENT_POLLS` consecutive polls means
"no CI" (the local pass finishes the run, or with no local check at all it needs you; a run where CI stood in for the
local check runs that check after all), and nothing terminal within `agent_ci_wait_seconds` (default 90 min) needs you
(`ci_timeout`).

Eridani holds no GitHub credential. It asks the run's host with the read-only `ci.status` command (sent only to a host
whose hello caps list `ci.status`); the host runs `gh` with its own login in the run's worktree (or the target's
repository once the worktree is gone) and answers in the command outcome's `result`, including whether GitHub would
merge the pull request now (run_merge.py reads that). Each answer is recorded as an attempt of kind `ci` (verify_id
`github_ci`), one per run tuple and PR head, updated in place on every poll, beside the local verify attempts.
"""

import re
from datetime import datetime, timedelta

from sqlalchemy import select

from . import agent_automation as automation
from . import run_coordinator as coordinator
from .config import get_settings
from .domain import emit
from .models import now
from .session_models import AgentAttempt, AgentCommand, AgentRun, ExecutionTarget

KIND = "ci"
SOURCE = "github_ci"
COMMAND = "ci.status"
POLL = timedelta(minutes=2)
LOG_LINES = 40
LINE_CHARS = 300
CHECKS = 50
PUSH_PROMPTS = 2  # "Push and open the pull request" at most this often per run; then the wait runs out.
# A pull request whose checks are absent on this many consecutive polls (~4 min) has no CI: workflows register their
# checks within seconds of a push, so this is a repository without CI, not a slow start.
ABSENT_POLLS = 2
# Fail fast: a check that failed while others still run is a failure now. If the host has no log for it yet, wait at
# most this many more polls for one before sending the gap without it.
LOG_WAITS = 2
# How far back target_ci looks for a CI answer about the run's repository.
TARGET_CI_ROWS = 10
# needs_you reasons a green CI run on the current head un-parks, as a late local pass does (verify_infra goes to the
# judge, which counts green CI as the verification; the CI reasons were met already and go on to approval).
UNPARK = ("verify_infra", "ci_timeout", "ci_unavailable", "verify_manual")
# Parked runs are asked less often after the first wait, and not at all after a day.
PARKED_POLL = timedelta(minutes=10)
PARKED_FOR = timedelta(hours=24)
# gh's own buckets (`gh pr checks --json bucket`); `state` is folded onto them when an older gh has no bucket.
BUCKETS = ("pass", "fail", "pending", "skipping", "cancel")
SETTLED = ("completed", "stopped", "capped", "failed")  # Runs whose agent is done (run_merge shares it).
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
    """Whether a met verdict on this plan consults GitHub CI before finishing: any plan under completion policy
    code_ready or pr_open (pr_merged ends on the merge itself). CI is preferred wherever it exists; a pull request
    without CI lets a local pass stand (`no_ci`, or at once for a target known to have none: gate)."""
    if plan is None:
        return False
    return (plan.completion_policy or "code_ready") in ("code_ready", "pr_open")


def quick(plan):
    """The plan's verify entry is a quick pre-push check (or it names none)."""
    verify_id = ((plan.verify or {}).get("verify_id") or "") if plan is not None else ""
    return not verify_id or verify_id in quick_ids()


def target_ci(db, run):
    """Whether the run's repository has CI: True (checks reported on a pull request), False (none on ABSENT_POLLS
    polls), None (no answer yet). Read from the newest CI answers for any run on the same host target."""
    target = db.get(ExecutionTarget, run.execution_target_id) if run.execution_target_id else None
    if target is None:
        return None
    same = select(ExecutionTarget.id).where(ExecutionTarget.host_id == target.host_id,
                                            ExecutionTarget.host_target == target.host_target)
    rows = db.scalars(select(AgentAttempt).join(AgentRun, AgentRun.id == AgentAttempt.run_id).where(
        AgentAttempt.kind == KIND, AgentRun.execution_target_id.in_(same))
        .order_by(AgentAttempt.updated_at.desc()).limit(TARGET_CI_ROWS))
    for row in rows:
        result = row.result or {}
        if result.get("checks"):
            return True
        if result.get("state") == "absent" and (result.get("polls") or 0) >= ABSENT_POLLS:
            return False
    return None


def stands_in(db, run, plan):
    """CI stands in for the plan's local check: a full (not quick) local entry, a policy that consults CI, a host
    that reports it, and a repository known to have CI. The local check is then skipped (verify_by_ci)."""
    return bool(plan is not None and local_pass(plan) and not quick(plan) and required(plan)
                and coordinator.host_cap(db, run.host_id, COMMAND) and target_ci(db, run) is True)


def verify_by_ci(db, run, plan, actor=None):
    """After a turn (run_coordinator.after_turn), instead of the full local check: wait for CI on the PR head; green
    goes to the judge, red sends the agent the failed jobs, no CI after all runs the local check (no_ci)."""
    coordinator.transition(db, run, "awaiting_ci", actor=actor, reason="ci_verify")
    save(run, since=now().isoformat(), turn_seq=run.turn_seq, error=None, absent_polls=0, verify_turn=tuple_key(run))
    request(db, run)
    return run


def verifying(run):
    """CI stands in for the local check on this turn: green goes to the judge, not straight to approval."""
    return ci_state(run).get("verify_turn") == tuple_key(run)


def local_pass(plan):
    """The plan has a local verify entry (its pass is what brought the judge to a met verdict)."""
    return bool(plan is not None and (plan.verify or {}).get("verify_id"))


def tuple_key(run):
    return f"{run.generation}:{run.turn_seq}:{run.plan_revision}"


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


def parse(result, stale_local=()):
    """The host's `ci.status` result → {state, head_sha, pr_url, pr_number, run_url, checks, failed, log,
    worktree_head, pr_state, base, draft, mergeable, merge_state}. Never raises: anything malformed reads as no pull
    request or no checks. `stale_local`: worktree heads known to be behind the PR only because Eridani updated the
    branch on GitHub (run_merge) are not "unpushed" work."""
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
    elif local and head and not local.startswith(head) and not head.startswith(local) and local not in stale_local:
        state = "unpushed"
    else:
        state = checks_state(checks)
    runs = [m.group(1) for c in checks if c["link"] and (m := _RUN_URL.match(c["link"]))]
    failing = [c for c in checks if c["bucket"] == "fail"]
    run_url = next((m.group(1) for c in failing if c["link"] and (m := _RUN_URL.match(c["link"]))), None) or \
        (runs[0] if runs else None)
    pr_url = url((pr or {}).get("url"))
    return {"state": state, "head_sha": head, "pr_url": pr_url, "running": "pending" in buckets,
            "pr_number": (pr or {}).get("number") if isinstance((pr or {}).get("number"), int) else None,
            "run_url": run_url or (f"{pr_url}/checks" if pr_url else None), "checks": checks,
            "failed": [{"name": c["name"], "link": c["link"]} for c in failing],
            "log": log_lines(result.get("failed_logs")) if failing else [], "worktree_head": local or None,
            **pr_facts(pr)}


def checks_state(checks) -> str:
    """The pull request's own checks, ignoring the worktree: absent, failed, pending, cancelled or passed."""
    buckets = {c["bucket"] for c in checks}
    return ("absent" if not checks else "failed" if "fail" in buckets else "pending" if "pending" in buckets
            else "cancelled" if "cancel" in buckets else "passed")


def word(value, limit=20):
    text = str(value or "").upper()[:limit]
    return text if re.fullmatch(r"[A-Z_]*", text) else ""


def pr_facts(pr):
    """Whether GitHub would merge the pull request now (remotea #59; older hosts send none of it: empty strings)."""
    pr = pr or {}
    base = str(pr.get("base") or "")[:200]
    return {"pr_state": word(pr.get("state")), "base": base if re.fullmatch(r"[A-Za-z0-9._/-]*", base) else "",
            "draft": pr.get("draft") is True, "mergeable": word(pr.get("mergeable")),
            "merge_state": word(pr.get("merge_state"))}


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
    if not required(plan) or passed_now(db, run) or ci_state(run).get("no_ci") == tuple_key(run):
        return None  # Not consulted, already green, or this turn's pull request was found to have no CI.
    if local_pass(plan) and target_ci(db, run) is False:
        return None  # The repository has no CI: the local pass is the verification, nothing to wait for.
    if not coordinator.host_cap(db, run.host_id, COMMAND):
        if not local_pass(plan):
            # An older host cannot report CI, and there is no local check either: nothing to wait for, so the judge's
            # verdict stands (approval, or auto_complete) as it did before the ruleset. Updating remotea restores CI.
            return None
        if not quick(plan):
            return None  # A full local pass stands on its own when CI cannot be read.
        return coordinator.transition(db, run, "needs_you", actor=actor, reason="ci_unavailable",
                                      detail="The host does not report GitHub CI (no ci.status capability).")
    coordinator.transition(db, run, "awaiting_ci", actor=actor, reason="ci_pending")
    save(run, since=now().isoformat(), turn_seq=run.turn_seq, error=None)
    request(db, run)
    return run


def request(db, run, clock=None):
    """Ask the host for the PR head's checks now (`ci.status`, read-only, 2-minute expiry)."""
    clock = clock or now()
    target = db.get(ExecutionTarget, run.execution_target_id) if run.execution_target_id else None
    body = {"branch": run.branch, "turn_seq": run.turn_seq, "log_lines": LOG_LINES,
            **({"target": target.host_target} if target else {})}
    row = coordinator.command(db, run, COMMAND, body, expires_at=clock + POLL, source="eridani")
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
    if plan is None or not plan.criteria or not (local_pass(plan) or required(plan)):
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
    facts = parse(result, ((run.sync_state or {}).get("merge") or {}).get("stale_heads") or ())
    attempt = record(db, run, tuple_, facts, clock)
    save(run, error=None, answered_at=clock.isoformat())
    emit(db, run.owner_id, "agent.changed", run.id)
    from . import run_merge

    if run_merge.on_ci(db, run, facts):
        return attempt  # The pull request merged (an answer Eridani lost): the run completed.
    if tuple_ != (run.generation, run.turn_seq, run.plan_revision) or run.paused or automation.blocked(run):
        return attempt
    if run.state == "needs_you" and run.state_reason in UNPARK:
        return unpark(db, run, attempt, facts)
    if run.state != "awaiting_ci":
        return attempt
    if facts["state"] == "passed":
        coordinator.note(run, "ci_passed", attempt_id=attempt.id, head_sha=facts["head_sha"])
        if verifying(run):
            return coordinator.to_judging(db, run, None, reason="ci_passed")  # CI was this turn's verification.
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
    if facts["state"] == "absent":
        return absent(db, run, attempt, facts)
    save(run, absent_polls=0)
    return attempt


def absent(db, run, attempt, facts):
    """The pull request has no checks. Once on ABSENT_POLLS consecutive polls that is a repository without CI: a quick
    local pass finishes the run on its own (judged_met again, the gate now stands aside for this turn), and a plan with
    no local check at all needs Davis's own verification (needs_you(verify_manual))."""
    polls = (ci_state(run).get("absent_polls") or 0) + 1
    if polls < ABSENT_POLLS:
        save(run, absent_polls=polls)
        return attempt
    return no_ci(db, run, facts, f"no CI checks on the pull request {facts.get('pr_url') or ''}".rstrip())


def no_ci(db, run, facts, why):
    plan = coordinator.frozen(db, run)
    actor = coordinator.judge_actor(run)
    standing_in = verifying(run)
    save(run, absent_polls=0, no_ci=tuple_key(run), verify_turn=None)
    coordinator.note(run, "ci_absent", head_sha=facts.get("head_sha") or "", detail=why[:300])
    if standing_in and local_pass(plan):
        # CI was to stand in for the local check, but this pull request has none: run the local check after all.
        coordinator.transition(db, run, "verifying", actor=actor, reason="ci_absent_local")
        coordinator.start_verify(db, run, plan)
        return run
    if local_pass(plan):
        return coordinator.judged_met(db, run, plan, actor)
    verification = coordinator.manual_verification(run)
    # Opening the PR can be a new turn, but the manual approval is for the work the owner reviewed.
    if verification and facts["state"] == "absent" and verification["head"] \
            and verification["head"] == facts["head_sha"] and coordinator.manual_pr_ready(db, run, plan):
        return coordinator.judged_met(db, run, plan, actor)
    return coordinator.transition(db, run, "needs_you", actor=actor, reason="verify_manual",
                                  detail=f"The target has no verify command and there are {why}."[:300])


def unpark(db, run, attempt, facts):
    """Green CI on the current head while the run is parked on a reason it clears: verify_infra (the local check could
    not run) goes to the judge, which counts the green CI record as the verification (agent_judge.ci_evidence);
    ci_timeout and ci_unavailable were met already and go on to approval or auto_complete. Anything else waits."""
    if facts["state"] != "passed":
        return attempt
    coordinator.note(run, "ci_passed", attempt_id=attempt.id, head_sha=facts["head_sha"], unparked=run.state_reason)
    if run.state_reason == "verify_infra" or verifying(run):
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
    if (ci.get("push_prompts") or 0) >= PUSH_PROMPTS and not local_pass(coordinator.frozen(db, run)):
        # Asked twice, still no pull request, and no local check either: there is nothing to wait for.
        return no_ci(db, run, facts, "no pull request" if facts["state"] == "no_pr" else "no pushed commits on the pull request")
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
    state = facts.get("state") or ("requested" if waiting else None)
    if state == "unpushed" and run.state in SETTLED:
        state = checks_state(facts.get("checks") or [])  # Same rule as the merge panel (run_merge.public).
    return {"state": state, "current": current,
            "attempt_id": attempt.id if attempt else None, "source": SOURCE,
            "head_sha": facts.get("head_sha") or None, "pr_url": facts.get("pr_url"), "run_url": facts.get("run_url"),
            "checks": facts.get("checks") or [], "failed": facts.get("failed") or [], "log": facts.get("log") or [],
            "checked_at": facts.get("checked_at"), "since": since,
            "wait_until": since + timedelta(seconds=wait_seconds()) if since else None,
            "error": ci.get("error") if waiting else None}
