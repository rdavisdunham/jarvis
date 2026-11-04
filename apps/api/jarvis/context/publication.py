"""Approval and publication of meeting proposals to Eridani tasks or Linear issues (plan_final.md section 7).

Approval (`approve`) is human-only: the decision's origin comes from `routing.human_command` through
`decisions.origin_of`, so a bot credential or a `ui-agent:` command can never approve. In one transaction it
records the `approve` decision against the exact proposal revision (which bumps the revision, so any later
regeneration or a second approval of the old revision conflicts), freezes the approved payload and destination in
a `context_publications` row with a stable `operation_id` (one per proposal revision), enqueues the
`context_publish` job and remembers an explicitly chosen destination in the owner state settings. Replaying the
same command ID returns the same publication. A changed payload always needs a new approval: the publication
never re-reads the proposal, and regeneration never edits approved or published proposals (it proposes a follow-up
update instead, `proposals.py`).

Publishing (`perform`, job `context_publish`) writes through the command path with deterministic command IDs
(`ctxpub:<publication>:<step>`), so a crash, a retry or a duplicate outbox delivery replays the stored command
result instead of writing again: zero duplicate tasks.

- create and suggestion, destination `eridani`: `task.create` with `assignee_id` (never a name) and a
  "From meeting" note naming the meeting and quoting the evidence.
- create and suggestion, destination `linear`: the same `task.create`, then `linear.publish` for that task. The
  command allocates the issue's UUID and the `LinearIssue` mapping before the remote write (`linear_write`), so
  Linear sync finds the mapping and never imports a second task, and a lost response reconciles by reading the
  issue under the same UUID before any retry (`linear_sync.process_write`). The assignee is a Linear user ID: the
  responsible actor's `linear_user` identity, the connection's viewer for the user's own work, or unassigned.
- update: `task.update` with the expected task revision recorded at proposal time; a task edited since is a
  `conflict`, never overwritten. A Linear-linked task forwards the change through `linear_sync` as any edit does.
- link: no task write; the proposal records the task, published at approval.

Publication status: queued, in_progress (Linear write pending), published, retrying, failed, conflict,
unconfirmed (Linear did not confirm; `retry` re-queues the write under the same issue UUID, which reads the remote
first). Linear outcomes are read back from the `linear_write` job and the mapping (`refresh`).

Linear matching scope. `linear_sync.process_sync` imports only issues assigned to the viewer while the
connection's `only_mine` is true, so the user's own Linear work is matched through its synced tasks. To match
other people's Linear work, widen the sync with `linear.select` and `only_mine=false`: their issues then sync as
local tasks and `proposals.generate` matches them like any task. An issue published here for someone else stays
mapped to its task while out of the narrow sync scope (`sync_state` out_of_scope), so it is never duplicated.
"""

from datetime import date

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from ..domain import DomainError, emit, enqueue_job, execute
from ..models import Actor, Job, LinearConnection, LinearIssue, Project, Task, now
from . import decisions, views
from .models import (
    ContextActorIdentity,
    ContextDecision,
    ContextOwnerState,
    ContextProposal,
    ContextPublication,
)
from .proposals import DESTINATIONS, LOCKED, _hash

STATUSES = ("queued", "in_progress", "published", "retrying", "failed", "conflict", "unconfirmed")
FINAL = ("published", "failed", "conflict", "unconfirmed")
LINEAR_STATUS = {"queued": "in_progress", "dispatched": "in_progress", "running": "in_progress",
                 "retrying": "retrying", "succeeded": "published", "conflict": "conflict",
                 "unconfirmed": "unconfirmed", "failed": "failed", "cancelled": "failed", "expired": "failed"}
JOB = "context_publish"
NOTES_LIMIT = 20000


class Edits(BaseModel):
    """The reviewer's changes to a proposal's task fields, part of what is approved."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    title: str | None = Field(default=None, min_length=1, max_length=500)
    description: str | None = Field(default=None, max_length=5000)
    assignee_id: str | None = Field(default=None, max_length=36)
    project_id: str | None = Field(default=None, max_length=36)
    due_date: date | None = None


class LinearOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    team_id: str | None = Field(default=None, max_length=100)
    assignee_id: str | None = Field(default=None, max_length=100)
    state_id: str | None = Field(default=None, max_length=100)


def operation_id(proposal_id: str, revision: int) -> str:
    return "ctxpub:" + _hash(proposal_id, revision)[:40]


def _command(publication_id: str, step: str) -> str:
    return f"ctxpub:{publication_id}:{step}"


def require_human(command_id: str | None) -> None:
    if not command_id or decisions.origin_of(command_id) != "human":
        raise DomainError("HUMAN_REQUIRED", "Only you can approve meeting proposals, from the review screen.", 403)


def _remember(db, owner: str, destination: str | None, team_id: str | None) -> None:
    """Keep the reviewer's explicit destination (and Linear team) as the default for later proposals. The owner
    state row exists: recording the approval allocated a decision seq under its lock."""
    state = db.scalar(select(ContextOwnerState).where(ContextOwnerState.owner_id == owner).with_for_update())
    values = dict(state.settings or {})
    if destination:
        values["review_destination"] = destination
    if team_id:
        values["review_linear_team_id"] = team_id
    if values != (state.settings or {}):
        state.settings, state.updated_at = values, now()


# ---------------------------------------------------------------- payloads


def _notes(proposal: ContextProposal, description: str) -> str:
    payload = proposal.payload or {}
    when = (payload.get("occurred_at") or "")[:10]
    title = payload.get("source_title") or "Untitled meeting"
    lines = [description.strip()] if description.strip() else []
    lines.append(f"From meeting: {title}" + (f" ({when})" if when else ""))
    lines += [f"> {e['quote']}" for e in (proposal.evidence or [])[:2] if e.get("quote")]
    return "\n\n".join(lines)[:NOTES_LIMIT]


def _owned(db, model, ident: str | None, owner: str, what: str):
    if ident is None:
        return None
    row = db.get(model, ident)
    if row is None or row.owner_id != owner or getattr(row, "archived", False):
        raise DomainError("INVALID_ARGUMENT", f"Choose an active {what}.")
    return row


def task_payload(db, owner: str, proposal: ContextProposal, edits: Edits | None = None) -> dict:
    """The task fields a create or suggestion approval publishes, with the reviewer's edits applied."""
    payload = proposal.payload or {}
    edits = edits or Edits()
    chosen = edits.model_dump(exclude_unset=True)
    assignee = chosen.get("assignee_id", payload.get("assignee_id")) or views.me(db, owner)
    project = chosen.get("project_id", payload.get("project_id"))
    _owned(db, Actor, assignee, owner, "assignee")
    _owned(db, Project, project, owner, "project")
    due = payload.get("due_date")
    if "due_date" in chosen:
        due = chosen["due_date"].isoformat() if chosen["due_date"] else None
    out = {"title": chosen.get("title") or proposal.title, "notes": _notes(proposal, chosen.get(
        "description", payload.get("description") or "")), "assignee_id": assignee, "project_id": project,
        "due_date": due}
    return {k: v for k, v in out.items() if v is not None}


def update_payload(db, owner: str, proposal: ContextProposal) -> dict:
    payload = proposal.payload or {}
    target, before, after = payload.get("target") or {}, payload.get("before") or {}, payload.get("after") or {}
    if not (target.get("task_id") and after):
        raise DomainError("INVALID_ARGUMENT", "This update has nothing to apply.")
    return {"task_id": target["task_id"], "expected_revision": before.get("revision") or target.get("revision"),
            **after}


def _linear_assignee(db, owner: str, conn: LinearConnection, actor_id: str | None) -> str | None:
    if not actor_id:
        return None
    if actor_id == views.me(db, owner):
        return conn.viewer_id
    i = ContextActorIdentity
    found = db.scalar(select(i.value).where(i.owner_id == owner, i.actor_id == actor_id, i.namespace == "linear_user")
                      .order_by(i.status != "confirmed", i.created_at).limit(1))
    users = {u["id"] for u in conn.directory.get("users", []) if u.get("active", True)}
    return found if found in users else None


def linear_payload(db, owner: str, task: dict, options: LinearOptions | None, remembered: str | None) -> dict:
    from ..linear_sync import connection

    conn = connection(db, owner)
    options = options or LinearOptions()
    team = options.team_id or remembered or (conn.team_ids[0] if len(conn.team_ids) == 1 else None)
    if not team or team not in conn.team_ids:
        raise DomainError("LINEAR_SCOPE", "Choose a Linear team selected in Settings.", 409)
    chosen = options.model_dump(exclude_unset=True)
    assignee = chosen["assignee_id"] if "assignee_id" in chosen else _linear_assignee(
        db, owner, conn, task.get("assignee_id"))
    out = {"team_id": team, "assignee_id": assignee}
    if options.state_id:
        out["state_id"] = options.state_id
    return out


# ---------------------------------------------------------------- approval


def view(row: ContextPublication) -> dict:
    return {"id": row.id, "proposal_id": row.proposal_id, "proposal_revision": row.proposal_revision,
            "destination": row.destination, "status": row.status, "operation_id": row.operation_id,
            "task_id": row.task_id, "remote_issue_id": row.remote_issue_id, "payload": row.payload,
            "error": (row.outcome or {}).get("error"), "message": (row.outcome or {}).get("message"),
            "created_at": row.created_at.isoformat(), "updated_at": row.updated_at.isoformat()}


def _replayed(db, owner: str, proposal_id: str, command_id: str) -> ContextPublication | None:
    d = ContextDecision
    decided = db.scalar(select(d).where(d.owner_id == owner, d.command_id == command_id, d.target_type == "proposal",
                                        d.target_id == proposal_id, d.operation == "approve").limit(1))
    if decided is None:
        return None
    return db.scalar(select(ContextPublication).where(
        ContextPublication.owner_id == owner, ContextPublication.proposal_id == proposal_id,
        ContextPublication.proposal_revision == decided.expected_revision))


def approve(db, owner: str, proposal_id: str, expected_revision: int, *, command_id: str, principal: str,
            destination: str | None = None, edits: Edits | None = None, linear: LinearOptions | None = None,
            updates: bool = False) -> dict:
    """Approve one proposal revision and enqueue its publication, atomically (see the module docstring).
    `updates` selects the separate "apply selected updates" approval: update proposals only."""
    require_human(command_id)
    if (done := _replayed(db, owner, proposal_id, command_id)) is not None:
        return view(done)
    row = db.scalar(select(ContextProposal).where(ContextProposal.id == proposal_id, ContextProposal.owner_id == owner)
                    .with_for_update())
    if row is None:
        raise DomainError("NOT_FOUND", "That proposal is no longer available.", 404)
    if (row.kind == "update") != updates:
        raise DomainError("INVALID_ARGUMENT", "Apply updates separately from new tasks." if updates is False
                          else "Only update proposals are applied here.")
    if row.revision != expected_revision:
        raise DomainError("REVISION_CONFLICT", "This proposal changed. Review the latest version.", 409)
    if row.disposition in LOCKED or row.disposition == "superseded":
        raise DomainError("REVISION_CONFLICT", "This proposal is no longer open.", 409)
    if destination is not None and destination not in DESTINATIONS:
        raise DomainError("INVALID_ARGUMENT", "Choose Eridani or Linear.")
    if row.kind in ("update", "link"):
        if destination == "linear" or edits or linear:
            raise DomainError("INVALID_ARGUMENT", "Updates and links apply to the existing task as proposed.")
        target = (row.payload or {}).get("target") or {}
        if db.get(Task, target.get("task_id") or "") is None:
            raise DomainError("NOT_FOUND", "The matched task is no longer available.", 404)
        where, payload = "eridani", update_payload(db, owner, row) if row.kind == "update" else {
            "task_id": target["task_id"]}
    else:
        where = destination or row.destination
        payload = task_payload(db, owner, row, edits)
        if where == "linear":
            state = db.get(ContextOwnerState, owner)
            remembered = ((state.settings or {}) if state else {}).get("review_linear_team_id")
            payload["linear"] = linear_payload(db, owner, payload, linear, remembered)
    decision = decisions.approve(db, owner, row.id, expected_revision, command_id=command_id, principal=principal)
    if decision.origin != "human":
        raise DomainError("HUMAN_REQUIRED", "Only you can approve meeting proposals, from the review screen.", 403)
    stamp = now()
    publication = ContextPublication(
        owner_id=owner, proposal_id=row.id, proposal_revision=expected_revision, payload=payload, destination=where,
        principal=principal or owner, command_id=command_id, operation_id=operation_id(row.id, expected_revision),
        status="queued", outcome={"decision_id": decision.id}, created_at=stamp, updated_at=stamp)
    db.add(publication)
    db.flush()
    if row.kind == "link":
        row.task_id, row.disposition = payload["task_id"], "published"
        publication.task_id, publication.status = payload["task_id"], "published"
    else:
        job = enqueue_job(db, owner, JOB, {"publication_id": publication.id})
        publication.outcome = {**publication.outcome, "job_id": job.id}
    row.destination = where
    if row.kind not in ("update", "link") and destination:
        _remember(db, owner, destination, (payload.get("linear") or {}).get("team_id"))
    emit(db, owner, "context.changed", row.id, row.revision)
    db.flush()
    return view(publication)


# ---------------------------------------------------------------- publishing


def _fail(row: ContextPublication, status: str, error: DomainError | str, **extra) -> None:
    code = error.code if isinstance(error, DomainError) else str(error)
    message = error.message if isinstance(error, DomainError) else None
    row.status, row.updated_at = status, now()
    row.outcome = {**(row.outcome or {}), "error": code, "message": message, **extra}


def _published(db, owner: str, row: ContextPublication, proposal: ContextProposal, task_id: str) -> None:
    row.task_id, row.status, row.updated_at = task_id, "published", now()
    proposal.task_id, proposal.disposition, proposal.updated_at = task_id, "published", now()
    emit(db, owner, "context.changed", proposal.id, proposal.revision)


def publish(db, owner: str, publication_id: str) -> dict:
    """Write one approved publication through the command path. Idempotent: commands replay by their IDs and a
    finished publication is left alone. Deterministic refusals (a changed task, Linear scope) end the publication
    as `conflict` or `failed`; anything else raises so the job retries."""
    row = db.scalar(select(ContextPublication).where(ContextPublication.id == publication_id,
                                                     ContextPublication.owner_id == owner).with_for_update())
    if row is None:
        return {"skipped": "unknown publication"}
    if row.status in FINAL or row.status == "in_progress":
        refresh(db, owner, row)
        return {"publication_id": row.id, "status": row.status, "replay": True}
    proposal = db.scalar(select(ContextProposal).where(ContextProposal.id == row.proposal_id).with_for_update())
    payload = dict(row.payload)
    try:
        with db.begin_nested():  # a refusal leaves no partial write (no local task without its Linear issue)
            if proposal.kind == "update":
                result = execute(db, owner, _command(row.id, "update"), "task.update", payload)
                _published(db, owner, row, proposal, result["data"]["id"])
            else:
                linear = payload.pop("linear", None)
                task = execute(db, owner, _command(row.id, "task"), "task.create", payload)["data"]
                if linear is None:
                    _published(db, owner, row, proposal, task["id"])
                else:
                    queued = execute(db, owner, _command(row.id, "linear"), "linear.publish",
                                     {"task_id": task["id"], "expected_revision": task["revision"], **linear})["data"]
                    link = db.scalar(select(LinearIssue).where(LinearIssue.owner_id == owner,
                                                               LinearIssue.task_id == task["id"]))
                    row.task_id, row.remote_issue_id = task["id"], link.remote_id if link else None
                    row.status, row.updated_at = "in_progress", now()
                    row.outcome = {**(row.outcome or {}), "linear_job_id": queued["job_id"]}
                    proposal.task_id = task["id"]
                    refresh(db, owner, row)
    except DomainError as error:
        if error.status >= 500:
            raise
        _fail(row, "conflict" if error.code in ("REVISION_CONFLICT", "LINEAR_CONFLICT", "LINEAR_PENDING")
              else "failed", error)
    db.flush()
    return {"publication_id": row.id, "status": row.status}


def refresh(db, owner: str, row: ContextPublication) -> ContextPublication:
    """Bring a Linear publication's status up to date from its `linear_write` job and mapping."""
    job_id = (row.outcome or {}).get("linear_job_id")
    if row.destination != "linear" or not job_id or row.status == "published":
        return row
    link = db.scalar(select(LinearIssue).where(LinearIssue.owner_id == owner, LinearIssue.task_id == row.task_id))
    if link is not None and link.pending_job_id and link.pending_job_id != job_id:
        job_id = link.pending_job_id  # a retry re-queued the write under the same issue UUID
    job = db.get(Job, job_id)
    status = LINEAR_STATUS.get(job.status, "in_progress") if job else "failed"
    if status == row.status:
        return row
    if status == "published":
        proposal = db.scalar(select(ContextProposal).where(ContextProposal.id == row.proposal_id).with_for_update())
        _published(db, owner, row, proposal, row.task_id)
        row.outcome = {**(row.outcome or {}), "error": None, "message": None}
    elif status in ("failed", "conflict", "unconfirmed"):
        message = (job.result or {}).get("message") if job else "The Linear operation is missing."
        row.status, row.updated_at = status, now()
        row.outcome = {**(row.outcome or {}), "error": f"LINEAR_{status.upper()}", "message": message,
                       "linear_job_id": job_id}
    else:
        row.status, row.updated_at = status, now()
    emit(db, owner, "context.changed", row.proposal_id)
    return row


def retry(db, owner: str, publication_id: str, *, command_id: str) -> dict:
    """Retry a publication that did not finish. Eridani writes re-run the same commands (they replay); a Linear
    write that is unconfirmed or failed is re-queued under the same issue UUID, so the remote is read and
    reconciled before anything is written again. Conflicts are resolved on the task (Linear comparison)."""
    require_human(command_id)
    row = db.scalar(select(ContextPublication).where(ContextPublication.id == publication_id,
                                                     ContextPublication.owner_id == owner).with_for_update())
    if row is None:
        raise DomainError("NOT_FOUND", "That publication is no longer available.", 404)
    refresh(db, owner, row)
    if row.status in ("published", "queued", "in_progress", "retrying"):
        return view(row)
    if row.status == "conflict":
        raise DomainError("PUBLICATION_CONFLICT", "The task changed since you approved this. Review it there.", 409)
    if row.destination == "linear" and row.task_id:
        from ..linear_sync import TERMINAL, connection, queue_write

        conn = connection(db, owner)
        link = db.scalar(select(LinearIssue).where(LinearIssue.owner_id == owner, LinearIssue.task_id == row.task_id))
        previous = db.get(Job, (row.outcome or {}).get("linear_job_id") or "")
        if link is None or previous is None:
            raise DomainError("NOT_FOUND", "This task has no Linear issue to retry.", 404)
        pending = db.get(Job, link.pending_job_id) if link.pending_job_id else None
        if pending is not None and pending.status not in TERMINAL:
            return view(row)
        job = queue_write(db, conn, link, previous.payload["patch"], create=previous.payload.get("create", False))
        row.status, row.updated_at = "in_progress", now()
        row.outcome = {**(row.outcome or {}), "linear_job_id": job.id, "error": None, "message": None}
    else:
        job = enqueue_job(db, owner, JOB, {"publication_id": row.id})
        row.status, row.updated_at = "queued", now()
        row.outcome = {**(row.outcome or {}), "job_id": job.id, "error": None, "message": None}
    emit(db, owner, "context.changed", row.proposal_id)
    return view(row)


def perform(work) -> dict:
    """The `context_publish` job {publication_id}. Approved publications continue while processing is paused."""
    from ..db import session_scope
    from .pipeline import MAX_ATTEMPTS

    publication_id = work.payload.get("publication_id")
    try:
        with session_scope() as db:
            return publish(db, work.owner, publication_id)
    except Exception:
        with session_scope() as db:
            row = db.get(ContextPublication, publication_id)
            if row is not None and row.status not in FINAL:
                final = work.attempts >= MAX_ATTEMPTS
                row.status, row.updated_at = ("failed" if final else "retrying"), now()
        raise
