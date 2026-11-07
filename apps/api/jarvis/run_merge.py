"""Merge from Eridani (docs/MERGE_FROM_ERIDANI_PLAN.md): the run drawer's Merge and Update branch buttons.

Eridani holds no GitHub credential. Like CI (run_ci.py), it asks the run's host: `pr.merge` runs
`gh pr merge --merge --match-head-commit <sha>` there, `pr.update_branch` GitHub's update-branch API (remotea #59).
What the button may do is read from the newest `ci.status` answer, which carries the pull request's own state and
whether GitHub would merge it now (`merge_state`: CLEAN, BEHIND, DIRTY, BLOCKED, …).

Rules (Davis, 2026-11-07):

- Merge exists for every run with a pull request, in every repository. When the repository has CI, it is live only on
  green CI for the pull request's current head ("Waiting on CI" while it runs); without CI it is always live.
- Merging completes the run (an approval recorded as "Merged PR #n into <base>"); a run that already settled only
  merges.
- A pull request behind its base (a strict "up to date" rule on main) offers Update branch instead: GitHub merges the
  base in, CI runs again on the new head, and a run waiting for approval waits for that CI first (awaiting_ci).
- Conflicts are the agent's to fix: the drawer offers Send back.

Auto-merge (phase 1b): a per-run toggle (`sync_state.auto_merge`; new runs follow the owner's `remotea_auto_merge`
setting). It is offered only with a real check: a local verify entry on the plan, or CI on the repository. Without
either there is only the manual button (Davis: that nudges every repository toward a verify entry). When the run
would wait for approval (judge met, integration done, CI green), it merges by itself instead: behind its base it
updates the branch and waits for CI again, at most AUTO_TRIES times per run; anything else GitHub refuses stops the run
in needs_you(merge_blocked). A merged run completes like auto_complete (the judge completes the task).

The run's `sync_state.merge` keeps the last click: {kind, op_id, state (sent, merged, refused, failed, uncertain,
updated), head_sha, reason, message, at, actor, …}. A merge whose answer was lost is reconciled by the next CI answer
that shows the pull request MERGED (on_ci).
"""

from datetime import timedelta

from sqlalchemy import select

from . import agent_link, run_ci
from . import run_coordinator as coordinator
from .agent_actor import Actor
from .domain import DomainError, emit
from .models import now
from .session_models import AgentCommand, AgentHost, ExecutionTarget

MERGE = "pr.merge"
UPDATE = "pr.update_branch"
# Run states where the agent is still at work: Merge waits for it to finish.
WORKING = ("queued", "spawning", "interviewing", "planning", "plan_reviewing", "handing_off", "executing",
           "waiting_answer", "verifying", "judging", "replanning", "awaiting_integration", "stopping")
SETTLED = ("completed", "stopped", "capped", "failed")
# GitHub's mergeStateStatus values that mean "it would merge" (UNSTABLE: a non-required check failed; HAS_HOOKS: a
# pre-receive hook will run; UNKNOWN: GitHub has not computed it yet, it tells us on the click).
MERGEABLE = ("CLEAN", "UNSTABLE", "HAS_HOOKS", "UNKNOWN", "")
REFUSALS = {
    "behind": "The branch is behind {base}; update it, then CI runs again.",
    "conflicts": "The branch conflicts with {base}; the agent has to resolve it.",
    "head_moved": "New commits arrived since CI ran; wait for CI on the new head.",
    "blocked": "GitHub blocked the merge (a rule on {base} or a required check).",
    "draft": "The pull request is a draft.",
    "not_open": "The pull request is closed.",
    "no_pr": "There is no pull request for the branch.",
    "permission": "The host's GitHub login cannot merge in this repository.",
}
REFRESH_EVERY = timedelta(seconds=30)
AUTO_TRIES = 3  # Update branch, wait for CI, merge: this many times per run, then Davis decides.
AUTO_WAIT = ("unknown", "waiting_ci", "merging", "updating", "working")


def state(run):
    return dict((run.sync_state or {}).get("merge") or {})


def save(run, **values):
    run.sync_state = {**(run.sync_state or {}), "merge": {**state(run), **values}}


def latest_facts(db, run):
    """The newest `ci.status` answer for the run, any turn: (facts, checked_at) or (None, None)."""
    attempt = run_ci.latest(db, run)
    if attempt is None:
        return None, None
    return (attempt.result or {}), attempt.updated_at


def host_name(host):
    return (host.name or host.machine or "The host") if host else "The host"


def host_reason(db, run, kind=MERGE):
    """Why the host cannot take the command now, or ""."""
    host = db.get(AgentHost, run.host_id) if run.host_id else None
    if host is None:
        return "This run has no host."
    name = host_name(host)
    if not agent_link.connected(host):
        return f"{name} is offline."
    if not coordinator.host_cap(db, run.host_id, kind):
        return f"Update remotea on {name} to merge from here."
    return ""


def in_flight(db, run, op):
    """The last click is still with the host: its command is queued or sent and not expired."""
    if op.get("state") != "sent" or not op.get("op_id"):
        return False
    row = db.scalar(select(AgentCommand).where(AgentCommand.op_id == op["op_id"]))
    return bool(row and row.status in ("queued", "sent", "accepted") and (row.expires_at is None or row.expires_at > now()))


def public(db, run):
    """The drawer's merge panel: {status, detail, can_merge, can_update, pr, ci, checked_at, op, via}.

    status: merged, closed, merging, updating, unknown (no CI answer yet), no_pr, working (the agent is still at
    it), host (offline or too old), waiting_ci, ci_failed, draft, conflicts, behind, blocked, ready."""
    facts, checked_at = latest_facts(db, run)
    op = state(run)
    pr = None
    if facts and facts.get("pr_number"):
        pr = {"number": facts.get("pr_number"), "url": facts.get("pr_url"), "base": facts.get("base") or "",
              "head_sha": facts.get("head_sha") or "", "state": facts.get("pr_state") or ""}
    has_ci = bool(facts and facts.get("checks")) or run_ci.target_ci(db, run) is True
    ci = (facts or {}).get("state") if facts else None
    view = {"pr": pr, "checked_at": checked_at, "op": public_op(op), "ci": ci if has_ci else "none",
            "can_merge": False, "can_update": False}
    base = (pr or {}).get("base") or "main"

    def out(status, detail, **more):
        return {**view, "status": status, "detail": detail, **more}

    if op.get("state") == "merged" or (pr and pr["state"] == "MERGED"):
        number = op.get("number") or (pr or {}).get("number")
        return out("merged", f"Merged PR #{number} into {op.get('base') or base}." if number else "Merged.")
    if in_flight(db, run, op):
        return out("merging" if op.get("kind") == MERGE else "updating",
                   "Merging…" if op.get("kind") == MERGE else "Updating the branch…")
    if facts is None:
        # No answer yet. A host that answered with a failure says why, instead of "Checking…" forever.
        error = run_ci.ci_state(run).get("error")
        if error:
            name = host_name(db.get(AgentHost, run.host_id) if run.host_id else None)
            hint = " Update the GitHub CLI (gh) there." if "unknown flag" in error else ""
            return out("host", f"{name} could not read the pull request: {error}.{hint}")
        return out("unknown", "Checking the pull request…")
    if pr is None:
        return out("no_pr", "No pull request yet.")
    if pr["state"] == "CLOSED":
        return out("closed", f"PR #{pr['number']} was closed without merging.")
    if run.state in WORKING:
        return out("working", "The agent is still working.")
    why = host_reason(db, run)
    if why:
        return out("host", why)
    merge_state = facts.get("merge_state") or ""
    if facts.get("draft") or merge_state == "DRAFT":
        return out("draft", REFUSALS["draft"])
    if facts.get("mergeable") == "CONFLICTING" or merge_state == "DIRTY":
        return out("conflicts", REFUSALS["conflicts"].format(base=base))
    if has_ci and ci == "failed":
        return out("ci_failed", "CI failed on this pull request.")
    if has_ci and ci != "passed":
        return out("waiting_ci", "Waiting on CI.")
    if merge_state == "BEHIND":
        return out("behind", REFUSALS["behind"].format(base=base), can_update=not host_reason(db, run, UPDATE))
    if merge_state not in MERGEABLE:
        # BLOCKED: a rule GitHub enforces (reviews, a required check it still counts as missing). Let the click try;
        # GitHub's own words come back as the refusal.
        return out("blocked", REFUSALS["blocked"].format(base=base), can_merge=True)
    return out("ready", f"Merge PR #{pr['number']} into {base}.", can_merge=True)


def public_op(op):
    if not op:
        return None
    keys = ("kind", "state", "reason", "message", "at", "head_sha", "new_head_sha", "number", "base", "merge_commit",
            "worktree_synced")
    return {k: op.get(k) for k in keys if op.get(k) not in (None, "")}


def check(db, run, expected_head_sha, *, update=False):
    view = public(db, run)
    allowed = view["can_update"] if update else view["can_merge"]
    if not allowed:
        raise DomainError("MERGE_UNAVAILABLE", view["detail"], 409, data={"status": view["status"]})
    if expected_head_sha and expected_head_sha.lower() != (view["pr"] or {}).get("head_sha"):
        raise DomainError("HEAD_MOVED", "The pull request changed since you looked; check it again.", 409)
    return view


def send(db, run, kind, head_sha, actor):
    target = db.get(ExecutionTarget, run.execution_target_id) if run.execution_target_id else None
    body = {"branch": run.branch, "head_sha": head_sha, **({"target": target.host_target} if target else {})}
    if coordinator.host_cap(db, run.host_id, "input.direct"):
        body["origin"] = "human" if actor.kind == "human" else "automation"
    row = coordinator.command(db, run, kind, body, source="eridani")
    save(run, kind=kind, op_id=row.op_id, state="sent", head_sha=head_sha, at=now().isoformat(),
         actor=actor.record(), reason=None, message=None)
    emit(db, run.owner_id, "agent.changed", run.id)
    return row


def merge(db, owner, run_id, body, *, actor):
    """The Merge button: merge the run's pull request on its CI-tested head."""
    from .agent_control import public_run

    run = coordinator.load(db, owner, run_id)
    view = check(db, run, body.expected_head_sha)
    send(db, run, MERGE, view["pr"]["head_sha"], actor)
    return public_run(db, run)


def update_branch(db, owner, run_id, body, *, actor):
    """The Update branch button: bring the base into the pull request on GitHub; CI then runs on the new head."""
    from .agent_control import public_run

    run = coordinator.load(db, owner, run_id)
    view = check(db, run, body.expected_head_sha, update=True)
    send(db, run, UPDATE, view["pr"]["head_sha"], actor)
    return public_run(db, run)


def refresh(db, owner, run_id):
    """Ask the host for the pull request now (the drawer, on open and while it waits). At most every REFRESH_EVERY,
    never while a poll is in flight; a host that cannot report CI is left alone."""
    from .agent_control import public_run

    run = coordinator.load(db, owner, run_id)
    clock = now()
    asked = run_ci.stamp(run_ci.ci_state(run).get("requested_at"))
    if run.branch and coordinator.host_cap(db, run.host_id, run_ci.COMMAND) and not run_ci.in_flight(db, run, clock) \
            and (asked is None or clock - asked >= REFRESH_EVERY) and not host_reason(db, run, run_ci.COMMAND):
        run_ci.request(db, run, clock)
    return public_run(db, run)


def actor_of(op):
    record = op.get("actor") or {}
    return Actor(record.get("kind") or "human", credential_id=record.get("credential_id"),
                 run_id=record.get("run_id"), generation=record.get("generation"))


def on_outcome(db, run, row, outcome, detail, result):
    """The host's answer to pr.merge or pr.update_branch (run_coordinator.on_outcome)."""
    op = state(run)
    if op.get("op_id") != row.op_id:
        return None  # An older click: its outcome is on the command row, the panel follows the newest.
    clock = now().isoformat()
    if row.kind == MERGE and outcome == "executed" and result.get("merged"):
        return merged(db, run, result.get("pr") or {}, result.get("merge_commit"), already=bool(result.get("already")))
    if outcome == "executed" and row.kind == UPDATE:
        stale = [*(op.get("stale_heads") or []), op.get("head_sha")] if not result.get("worktree_synced") else []
        save(run, state="updated", stale_heads=[h for h in stale if h][-5:], new_head_sha=result.get("new_head_sha"),
             worktree_synced=bool(result.get("worktree_synced")), answered_at=clock)
        coordinator.note(run, "branch_updated", head_sha=op.get("head_sha"), new_head_sha=result.get("new_head_sha"))
        if run.state == "awaiting_approval":
            # Approval was on the old head: CI on the new one decides again (green goes back to approval).
            coordinator.transition(db, run, "awaiting_ci", actor=actor_of(op), reason="ci_update")
            run_ci.save(run, since=now().isoformat(), turn_seq=run.turn_seq, error=None, absent_polls=0)
    elif outcome == "rejected":
        reason = str(result.get("reason") or detail or "blocked")[:40]
        save(run, state="refused", reason=reason, message=str(result.get("message") or "")[:300], answered_at=clock)
        coordinator.note(run, "merge_refused" if row.kind == MERGE else "update_refused", reason=reason)
        if auto_running(run) and reason not in ("behind", "head_moved"):
            # Behind or moved: the CI answer requested below shows it, and auto_step updates or waits.
            block(db, run, f"GitHub refused the {'merge' if row.kind == MERGE else 'update'}: "
                           f"{result.get('message') or REFUSALS.get(reason, reason)}")
    else:
        save(run, state="uncertain" if outcome == "uncertain" else "failed", message=(detail or outcome)[:300],
             answered_at=clock)
    emit(db, run.owner_id, "agent.changed", run.id)
    if run.branch and coordinator.host_cap(db, run.host_id, run_ci.COMMAND) and not run_ci.in_flight(db, run, now()):
        run_ci.request(db, run)  # Show the pull request as it is now: the new head, behind, merged after all.
    return run


def merged(db, run, pr, merge_commit=None, *, already=False):
    """The pull request merged through Eridani: record it, and complete the run unless it already settled."""
    from .agent_control import complete_approved

    op = state(run)
    number, base = pr.get("number"), pr.get("base") or "main"
    save(run, state="merged", number=number, base=base, merge_commit=merge_commit, already=already,
         answered_at=now().isoformat())
    text = f"Merged PR #{number} into {base}" if number else "Merged the pull request"
    coordinator.note(run, "merged_pr", number=number, base=base, merge_commit=merge_commit)
    emit(db, run.owner_id, "agent.changed", run.id)
    if run.state in SETTLED or run.state == "stopping":
        return run
    actor = actor_of(op)
    if actor.kind != "human":
        save(run, auto_active=False)
        return coordinator.complete_by_judge(db, run, coordinator.judge_actor(run))  # Auto-merge: like auto_complete.
    try:
        complete_approved(db, run, text, actor, f"run-merge:{run.id}")
    except DomainError as error:
        # The task could not be completed (already done elsewhere, a conflict): the merge stands, Davis approves.
        coordinator.note(run, "merge_complete_failed", detail=error.message[:300])
    return run


def on_ci(db, run, facts):
    """Every CI answer (run_ci.on_outcome). The pull request MERGED while Eridani's own merge was unanswered (lost,
    uncertain, the command expired): that merge happened, and the run completes here (True). Otherwise, a run
    auto-merging takes its next step on the fresh facts."""
    op = state(run)
    if facts.get("pr_state") == "MERGED" and op.get("kind") == MERGE and op.get("state") in ("sent", "uncertain", "failed"):
        merged(db, run, {"number": facts.get("pr_number"), "base": facts.get("base")})
        return True
    if auto_running(run) and run.state == "awaiting_approval":
        auto_step(db, run)
    return False


# ---------------------------------------------------------------- auto-merge (phase 1b)


def auto_default(db, owner):
    from .models import OwnerSettings

    row = db.get(OwnerSettings, owner)
    return bool((row.values if row else {}).get("remotea_auto_merge", False))


def auto_enabled(db, run):
    """The run's toggle; a run nobody toggled follows the owner's setting."""
    value = (run.sync_state or {}).get("auto_merge")
    return auto_default(db, run.owner_id) if value is None else bool(value)


def auto_available(db, run, plan=None):
    """Why auto-merge cannot be offered, or "": it needs a real check, a local verify entry or CI."""
    plan = plan if plan is not None else coordinator.frozen(db, run)
    if run_ci.local_pass(plan) or run_ci.target_ci(db, run) is True:
        return ""
    facts, _ = latest_facts(db, run)
    if facts and facts.get("checks"):
        return ""
    return "Needs a verify entry or CI."


def auto_running(run):
    return bool(state(run).get("auto_active"))


def public_auto(db, run):
    why = auto_available(db, run)
    return {"enabled": auto_enabled(db, run) and not why, "available": not why, "why": why or None,
            "tries": state(run).get("auto_tries") or 0, "active": auto_running(run)}


def auto_on(db, run, plan):
    """judged_met asks: should this run merge by itself instead of waiting for approval?"""
    return bool(plan is not None and plan.criteria and auto_enabled(db, run) and not auto_available(db, run, plan))


def auto_start(db, run, actor):
    """The run would wait for approval: it waits there while it merges by itself (a fresh CI answer drives it)."""
    if run.state != "awaiting_approval":
        coordinator.transition(db, run, "awaiting_approval", actor=actor, reason="auto_merge")
    save(run, auto_active=True)
    if run.branch and coordinator.host_cap(db, run.host_id, run_ci.COMMAND) and not run_ci.in_flight(db, run, now()):
        run_ci.request(db, run)
    else:
        auto_step(db, run)
    return run


def auto_step(db, run):
    """One step on the newest facts: merge when it can, update when behind (at most AUTO_TRIES), wait while CI runs or
    a command is with the host, and otherwise stop for Davis (needs_you(merge_blocked))."""
    view = public(db, run)
    status = view["status"]
    actor = coordinator.judge_actor(run)
    if status == "merged":
        return merged(db, run, view["pr"] or {})
    if status in AUTO_WAIT:
        return run
    if view["can_merge"]:
        return send(db, run, MERGE, view["pr"]["head_sha"], actor)
    if status == "behind" and view["can_update"]:
        tries = state(run).get("auto_tries") or 0
        if tries >= AUTO_TRIES:
            return block(db, run, f"The branch fell behind {view['pr']['base'] or 'main'} {tries} times while "
                                  "auto-merging; update and merge it yourself.")
        save(run, auto_tries=tries + 1)
        return send(db, run, UPDATE, view["pr"]["head_sha"], actor)
    return block(db, run, view["detail"])


def block(db, run, why):
    save(run, auto_active=False)
    if run.state in SETTLED or run.state == "stopping":
        return run
    return coordinator.transition(db, run, "needs_you", actor=coordinator.judge_actor(run), reason="merge_blocked",
                                  detail=why[:300])


def set_auto(db, owner, run_id, body, *, actor):
    """The drawer's Auto-merge toggle. Turned on while the run waits for approval, it starts merging now."""
    from .agent_control import public_run

    run = coordinator.load(db, owner, run_id)
    why = auto_available(db, run)
    if body.enabled and why:
        raise DomainError("AUTO_MERGE_UNAVAILABLE", why, 409)
    run.sync_state = {**(run.sync_state or {}), "auto_merge": bool(body.enabled)}
    coordinator.note(run, "auto_merge", enabled=bool(body.enabled), actor=actor.record())
    if body.enabled and run.state == "awaiting_approval" and not auto_running(run):
        auto_start(db, run, coordinator.judge_actor(run))
    elif not body.enabled and auto_running(run):
        save(run, auto_active=False)
    emit(db, run.owner_id, "agent.changed", run.id)
    return public_run(db, run)

