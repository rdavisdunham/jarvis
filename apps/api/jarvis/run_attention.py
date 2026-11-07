"""Why a run stopped for you, in plain words: the run read model's `attention_detail`.

{reason, headline, action, detail, last_output} for a run in needs_you or capped, so the run card and drawer can lead
with the headline and the action without parsing free text. `headline` says what happened, `action` what to do next;
`detail` is the longer text behind the state (agent_control.reason_detail); `last_output` the last lines of the
verification's output for verification reasons (else []). run_coordinator.REASONS stays the one-line state text.
"""

import re

from sqlalchemy import select

from .session_models import AgentAttempt

# reason → (headline, action). Every needs_you reason run_coordinator transitions with has a row
# (tests/test_drift_guards.py::test_every_needs_you_reason_has_a_headline).
ATTENTION = {
    "permission_denied": ("The session's permissions blocked an action.",
                          "Review it in the terminal; approving here cannot override permissions."),
    "receipt_uncertain": ("Your answer was saved, but the agent has not confirmed reading it.",
                          "Wait for the receipt, or send the answer again."),
    "paused": ("You paused this run.", "Resume it when you are ready."),
    "terminal_control": ("The terminal has taken control of this session.", "Continue in the terminal."),
    "delivery_uncertain": ("The host could not tell whether your input reached the agent.",
                           "Check the terminal, then send it again if it did not arrive."),
    "host_offline": ("The remotea server went offline during the run.",
                     "Bring the server back online; the run picks up when it reconnects."),
    "surface_lost": ("The agent's terminal was lost in a host restart.", "Send the agent a message to resume, or re-plan."),
    "exited": ("The agent process exited.", "Send the agent a message to start it again, or approve the work as it is."),
    "spawn_expired": ("The host did not take the start command in time.", "Check the remotea server, then start again."),
    "spawn_failed": ("The host could not start the agent.", "Check the remotea server's log, then start again."),
    "spawn_rejected": ("The host refused the start command.", "Check the target in eridani.json, then start again."),
    "target_not_ready": ("The target is not ready: its template check failed.",
                         "Fix the target (remotea doctor shows what failed), then start again."),
    "target_busy": ("The target already runs as many agents as it allows.", "Free a run or raise the limit, then start again."),
    "spawn_uncertain": ("The host could not tell whether the agent started.", "Check the terminal, then resume or stop."),
    "wall_clock": ("The run reached its wall-clock limit.", "Resume to give it more time, approve the work, or stop."),
    "no_progress": ("The agent made no progress for a while.", "Check the terminal and send the agent a note."),
    "turn_cap": ("The run reached its turn cap.", "Approve the work as it is, or send the agent back with a note."),
    "question_timed_out": ("The agent's question timed out here.", "Answer it in the terminal."),
    "no_plan": ("The agent ended planning without questions or a plan.", "Retry planning or send the agent a note."),
    "planning_retry": ("Retrying the rejected planning handoff.", "Nothing to do yet."),
    "plan_review": ("The agent proposes changes to your success criteria.", "Accept or keep each one to start the work."),
    "plan_approval": ("The plan is final.", "Approve it to start the work, or send it back with a note."),
    "host_busy": ("The host is at its run limit, so the next session could not start.",
                  "Free a run or raise limits.maxRuns, then send the agent a message."),
    "reviewer_blocked": ("The plan reviewer could not run.", "Send the agent a note, or approve the plan as it is."),
    "no_progress_gap": ("The judge found the same gap several turns in a row.",
                        "Read the gap, then send the agent back with a note, edit the criteria, or approve."),
    "verify_missing": ("No verify command is set, so test results cannot be recorded.",
                       "Set a verify command and verify again, or approve on the evidence shown."),
    "verify_incomplete": ("The criteria look met, but the verification did not cover the whole workspace.",
                          "Review the work and approve, or verify again."),
    "verify_infra": ("Verification could not run on the host.", "Fix the setup shown, then verify again, or approve."),
    "judge_needs_you": ("The judge says this needs you.", "Read the judge's note, then approve or send the agent back."),
    "judge_unavailable": ("The judge could not reach a verdict.", "Judge again, or review the work and approve."),
    "awaiting_merge": ("The criteria are met; the pull request is not merged yet.", "Merge it; the run re-checks then."),
    "pr_unknown": ("The criteria are met, but Eridani has no current record of the pull request.",
                   "Check it, then approve or send the agent back to open it."),
    # GitHub CI as the final verification (run_ci.py). ci_timeout's headline is the specific wait (run_ci.timeout_text).
    "ci_unavailable": ("The criteria are met, but the host cannot report GitHub CI.",
                       "Update remotea to a build with ci.status, or check CI yourself and approve."),
    # The ruleset's last rung (run_ci.py): no local check on the target and no CI on the pull request.
    "verify_manual": ("Needs your verification: the target has no verify command and the pull request has no CI checks.",
                      "Check the work yourself, then approve it as verified, or describe what is wrong and send the agent back."),
    "ci_timeout": ("GitHub CI did not finish on the pull request in time.",
                   "Check the pull request's checks, then approve, or send the agent back with a note."),
    "ci_failed_again": ("GitHub CI failed again on the same commit; the agent did not push a fix.",
                        "Read the failing log, then send the agent back with a note, or approve."),
    # Auto-merge (run_merge.py): GitHub refused, the branch kept falling behind, or the pull request needs the agent.
    "merge_blocked": ("Auto-merge stopped: GitHub would not merge the pull request.",
                      "Read why below, then merge or update it yourself, or send the agent back."),
}
CI_REASONS = ("ci_timeout", "ci_failed_again", "ci_unavailable", "verify_manual")

# Per infra category (run_coordinator.classify_verify): the headline for verify_infra.
VERIFY_HEADLINES = {
    "infra_missing_binary": "Verification could not run: a program is missing on the host.",
    "infra_missing_dependency": "Verification could not run: a dependency is missing where it runs.",
    "infra_database": "Verification could not use its test database.",
    "infra_refused": "remotea did not run the verification.",
    "infra_uncertain": "The host could not tell whether the verification ran.",
}
VERIFY_REASONS = ("verify_infra", "verify_incomplete", "verify_missing")
OUTPUT_LINES, LINE_CHARS = 5, 200
# pytest -q progress: "....F..s.  [ 94%]".
_PROGRESS = re.compile(r"^([.FEsxX]+)\s*\[\s*(\d{1,3})%\]\s*$")
_FAILED = re.compile(r"\b(\d+) (?:failed|errors?)\b")


def attention(db, run, detail=""):
    """The structured stop reason of a run in needs_you or capped; None otherwise."""
    if run.state not in ("needs_you", "capped"):
        return None
    reason = run.state_reason or ""
    headline, action = ATTENTION.get(reason, ("This run needs your attention.", "Open the run and decide what is next."))
    failure = (getattr(run, "sync_state", None) or {}).get("coordination_failure")
    if reason == "receipt_uncertain" and failure:
        headline, action = ("The planning submission was rejected.", "Retry planning so the agent can correct the rejected submission.") \
            if failure.get("outcome") == "rejected" else \
            ("The planning submission result is unconfirmed.", "Reconcile the submission receipt before retrying planning.")
    last_output = []
    if reason in VERIFY_REASONS:
        attempt = latest_verify(db, run)
        result = (attempt.result or {}) if attempt is not None else {}
        last_output = tail_lines(result.get("tail"))
        if reason == "verify_infra" and attempt is not None:
            headline, action = verify_headline(attempt, result, action)
    elif reason in CI_REASONS:
        from .run_ci import latest

        ci = latest(db, run)
        last_output = tail_lines("\n".join((ci.result or {}).get("log") or []) if ci else "")
        if reason == "ci_timeout" and detail:
            headline = detail  # "CI on 1a2b3c4 was still running after 1 h 30 min (backend)."
    return {"reason": reason, "headline": headline, "action": action, "detail": detail or "", "last_output": last_output}


def latest_verify(db, run):
    return db.scalar(select(AgentAttempt).where(AgentAttempt.run_id == run.id, AgentAttempt.kind == "verify",
                                                AgentAttempt.status.not_in(("running", "queued")))
                     .order_by(AgentAttempt.created_at.desc()).limit(1))


def tail_lines(tail, count=OUTPUT_LINES):
    lines = [line.rstrip() for line in str(tail or "").splitlines() if line.strip()]
    return [line if len(line) <= LINE_CHARS else line[:LINE_CHARS - 1] + "…" for line in lines[-count:]]


def verify_headline(attempt, result, fallback_action):
    category = result.get("category") or ""
    action = result.get("action") or fallback_action
    if category != "infra_timeout":
        return VERIFY_HEADLINES.get(category, ATTENTION["verify_infra"][0]), f"{action.rstrip('.')}. Or verify again."
    if not result.get("timed_out") and not result.get("tail"):
        return swept_headline(attempt, result), ("Verify again. If it happens again, check the remotea server is online "
                                                 "and that it advertises the verify entry's timeout.")
    return timeout_headline(attempt, result), "Raise the verify timeout or verify again."


def swept_headline(attempt, result):
    """Eridani's own deadline ran out before the host reported: say how long it waited and where that came from."""
    waited = minutes((attempt.deadline_at - attempt.created_at).total_seconds()) \
        if attempt.deadline_at and attempt.created_at else None
    timeout = result.get("timeout_s")
    source = (f"the host's {span(timeout)} timeout plus 2 min" if timeout
              else "its 11-minute default: the host did not advertise the verify entry's timeout")
    return (f"Eridani stopped waiting for verification after {waited} min ({source}); the host had not reported "
            "a result." if waited else "Eridani stopped waiting for verification; the host had not reported a result.")


def timeout_headline(attempt, result):
    """"The host stopped verification after 30 min (its timeout: 30 min) at 94% of the suite, no failures." """
    seconds = (result.get("duration_ms") or 0) / 1000 or result.get("timeout_s") or (
        (attempt.updated_at - attempt.created_at).total_seconds() if attempt.updated_at and attempt.created_at else 0)
    ran = minutes(seconds) if seconds else None
    timeout = result.get("timeout_s")
    progress, failures = progress_of(str(result.get("tail") or ""))
    lines = tail_lines(result.get("tail"), 1)
    where = f"at {progress}% of the suite" if progress is not None else (f"at “{lines[0][:80]}”" if lines else "")
    failed = "no failures" if not failures else f"{failures} failure{'s' if failures != 1 else ''}"
    parts = ["The host stopped verification" + (f" after {ran} min" if ran else "")
             + (f" (its timeout: {span(timeout)})" if timeout else ""), where]
    return " ".join(p for p in parts if p) + f", {failed}."


def minutes(seconds):
    return max(1, round(seconds / 60))


def span(seconds):
    from .run_coordinator import span as spoken

    return spoken(seconds)


def progress_of(tail):
    """(last pytest percentage or None, failures seen so far) from verify output."""
    progress, failures = None, 0
    for line in tail.splitlines():
        match = _PROGRESS.match(line.strip())
        if match:
            progress = int(match.group(2))
            failures += sum(match.group(1).count(c) for c in "FE")
    return progress, max([failures, *(int(n) for n in _FAILED.findall(tail))])
