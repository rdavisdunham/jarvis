"""Linear outcomes of agent runs (decision 13, C13): In Progress at the observed start, Done and one
completion comment when the run completes.

Local run state and source sync are separate persisted outcomes: a refused or failed Linear write is
recorded on `agent_runs.sync_state["linear"]` and never undoes the start or the completion.

    in_progress | done: {status: queued|deferred|blocked|skipped, job_id?, after?, code?, message?}
    comment:            {status: queued|retrying|posted|failed|unconfirmed|skipped, job_id, marker,
                         attempted, comment_id?, url?, code?, message?}

The comment is queued once per run (job `linear_comment`, queue jarvis-linear). Linear's commentCreate has
no idempotency key, so the body carries a stable run marker and the job records `attempted` before it
writes: a retry after an accepted-but-lost write first reads the issue's comments for the marker and saves
that comment's id instead of posting a second one.
"""

from sqlalchemy import select

from .db import session_scope
from .domain import DomainError, emit, enqueue_job
from .google_auth import unseal
from .linear_client import LinearClient, LinearFailure
from .models import Job, LinearConnection, LinearIssue, Task, now
from .session_models import AgentAttempt, AgentRun, AgentSession

ATTEMPTS = 5
TERMINAL = {"succeeded", "failed", "cancelled"}
POLICY_TEXT = {"code_ready": "code ready", "pr_open": "pull request open", "pr_merged": "pull request merged"}


def marker(run_id):
    return f"eridani-run:{run_id}"


def linear_state(run):
    return dict((run.sync_state or {}).get("linear") or {})


def record_outcome(db, run_id, purpose, outcome):
    """Save one Linear outcome on the run (in_progress, done or comment)."""
    run = db.get(AgentRun, run_id, with_for_update=True) if run_id else None
    if run is None:
        return None
    clean = {k: v for k, v in outcome.items() if k not in ("purpose", "run_id")}
    linear = linear_state(run)
    linear[purpose] = {**(linear.get(purpose) or {}), **clean, "at": now().isoformat()}
    run.sync_state = {**(run.sync_state or {}), "linear": linear}
    emit(db, run.owner_id, "agent.changed", run.id)
    return linear[purpose]


def save(run, purpose, outcome):
    """Like record_outcome for a run the caller already holds."""
    linear = linear_state(run)
    linear[purpose] = {k: v for k, v in outcome.items() if k not in ("purpose", "run_id")} | {"at": now().isoformat()}
    run.sync_state = {**(run.sync_state or {}), "linear": linear}


# ---------------------------------------------------------------- the completion comment


def queue_comment(db, run):
    """On `completed`, once per run. A task without a Linear issue records `skipped`."""
    if linear_state(run).get("comment"):
        return None
    link = db.scalar(select(LinearIssue).where(LinearIssue.owner_id == run.owner_id, LinearIssue.task_id == run.task_id))
    conn = db.get(LinearConnection, run.owner_id)
    if link is None or link.sync_state == "unlinked":
        return None
    if conn is None or not conn.enabled or not conn.credentials or conn.workspace_id != link.workspace_id:
        save(run, "comment", {"status": "skipped", "message": "Linear is not connected to this issue's workspace."})
        return None
    job = enqueue_job(db, run.owner_id, "linear_comment", {
        "run_id": run.id, "link_id": link.id, "remote_id": link.remote_id, "generation": conn.generation})
    save(run, "comment", {"status": "queued", "job_id": job.id, "marker": marker(run.id), "attempted": False})
    return job


def pull_requests(db, run):
    from .agent_sessions import pull_requests as valid

    session = db.get(AgentSession, run.session_pk) if run.session_pk else None
    return valid(session.pr_urls if session else [])


def comment_body(db, run):
    """Plain facts from persisted run state; the judge's summary is quoted as its own line."""
    task = db.get(Task, run.task_id)
    verdict = db.scalar(select(AgentAttempt).where(AgentAttempt.run_id == run.id, AgentAttempt.kind == "judge",
                                                   AgentAttempt.status == "done")
                        .order_by(AgentAttempt.created_at.desc()).limit(1))
    approved = ((run.sync_state or {}).get("completion") or {}).get("approved")
    lines = ["Completed by an Eridani agent run" + (" (approved by the operator)." if approved else ".")]
    if task:
        lines.append(f"Task: {task.title[:300]}")
    if run.branch:
        lines.append(f"Branch: `{run.branch}`")
    prs = pull_requests(db, run)
    if prs:
        lines.append("Pull requests: " + ", ".join(prs))
    lines.append(f"Turns: {run.turn_count} of {run.turn_cap} · done means: "
                 + POLICY_TEXT.get(run.completion_policy, "code ready"))
    summary = ((verdict.result or {}).get("summary") or "").strip() if verdict else ""
    if summary:
        lines.append("Judge: " + " ".join(summary.split())[:1200])
    lines += ["", marker(run.id)]
    return "\n".join(lines)


def process_comment(job_id, client_factory=LinearClient):
    """Worker job. `client_factory` replaces the Linear client in tests."""
    with session_scope() as db:
        job = db.get(Job, job_id, with_for_update=True)
        if not job or job.status in TERMINAL:
            return
        payload, owner = job.payload, job.owner_id
        run = db.get(AgentRun, payload["run_id"], with_for_update=True)
        state = linear_state(run).get("comment") or {} if run else {}
        if run is None or state.get("comment_id"):
            job.status, job.finished_at = "succeeded", now()
            return
        attempt = int((job.result or {}).get("attempts", 0)) + 1
        try:
            conn = db.get(LinearConnection, owner)
            link = db.get(LinearIssue, payload["link_id"])
            if conn is None or not conn.enabled or conn.generation != payload["generation"]:
                raise DomainError("LINEAR_CHANGED", "The Linear connection changed.")
            if link is None or link.remote_id != payload["remote_id"]:
                raise DomainError("LINEAR_CHANGED", "The task's Linear issue changed.")
            key = unseal(conn.credentials)["api_key"]
        except DomainError as exc:
            job.status, job.finished_at, job.result = "failed", now(), {"message": exc.message}
            save(run, "comment", {**state, "status": "unconfirmed" if state.get("attempted") else "failed",
                                  "code": exc.code, "message": exc.message})
            emit(db, owner, "agent.changed", run.id)
            return
        body, attempted, wanted = comment_body(db, run), bool(state.get("attempted")), marker(run.id)
        job.status, job.result = "running", {"attempts": attempt}
    client = client_factory(key)
    try:
        found = None
        if attempted:
            # Reconcile first: the earlier write may have been accepted with its response lost.
            found = next((c for c in client.comments(payload["remote_id"]) if wanted in (c.get("body") or "")), None)
        if found is None:
            with session_scope() as db:
                run = db.get(AgentRun, payload["run_id"], with_for_update=True)
                save(run, "comment", {**(linear_state(run).get("comment") or {}), "attempted": True})
            found = client.comment(payload["remote_id"], body)
        with session_scope() as db:
            job, run = db.get(Job, job_id), db.get(AgentRun, payload["run_id"], with_for_update=True)
            save(run, "comment", {**(linear_state(run).get("comment") or {}), "status": "posted",
                                  "comment_id": found["id"], "url": found.get("url"), "reconciled": attempted,
                                  "attempted": True})
            job.status, job.finished_at = "succeeded", now()
            job.result = {"attempts": attempt, "comment_id": found["id"]}
            emit(db, run.owner_id, "agent.changed", run.id)
    except (LinearFailure, KeyError, TypeError, ValueError) as error:
        retry = isinstance(error, LinearFailure) and error.retry and attempt < ATTEMPTS
        code = error.code if isinstance(error, LinearFailure) else "invalid_response"
        with session_scope() as db:
            job, run = db.get(Job, job_id), db.get(AgentRun, payload["run_id"], with_for_update=True)
            current = linear_state(run).get("comment") or {}
            unknown = not isinstance(error, LinearFailure) or error.unknown
            status = "retrying" if retry else "unconfirmed" if current.get("attempted") and unknown else "failed"
            save(run, "comment", {**current, "status": status, "code": code})
            job.status = "retrying" if retry else "failed"
            job.finished_at = None if retry else now()
            job.result = {"attempts": attempt, "code": code}
            emit(db, run.owner_id, "agent.changed", run.id)
        if retry:
            raise RuntimeError("The Linear comment will retry; the run marker prevents a duplicate.") from None
    finally:
        client.close()
