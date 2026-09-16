"""Durable accepted work. Database state owns execution; voice only observes it."""

import hashlib
import json
from contextlib import contextmanager
from datetime import timedelta
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import select

from . import agent_models
from .access import actor, execution, person_preferences, role
from .config import get_settings
from .db import session_scope
from .domain import DomainError, advisory, emit, owned, preferences
from .models import AgentWork, Conversation, Job, Outbox, UserAccount, now
from .work_crypto import seal, unseal

ACTIVE = {"queued", "dispatched", "running"}
TERMINAL = {"succeeded", "partial", "failed", "cancelled", "expired"}


def stable_id(value):
    return str(uuid5(NAMESPACE_URL, "eridani-work:" + value))


def account_for(db, owner, device):
    from .models import AuthSession

    account = actor(db, owner)
    # Interactive creation must have an authenticated device in this namespace.
    session = db.scalar(select(AuthSession).where(AuthSession.device_id == device))
    if not session or session.owner_id != account or (session.workspace_id or account) != owner:
        raise DomainError("NOT_AUTHORIZED", "Start this request from your signed-in workspace.", 403)
    role(db, owner, account)
    return account


def enqueue(
    db,
    owner,
    account,
    device,
    conversation_id,
    request_id,
    message,
    *,
    kind="agent_intake",
    parent_id=None,
    voice_session_id=None,
    context=None,
    focus=None,
    dependencies=None,
    resources=None,
):
    conv = owned(db, Conversation, conversation_id, owner)
    if conv.device_id != device:
        raise DomainError("NOT_AUTHORIZED", "This conversation belongs to another device.", 403)
    role(db, owner, account)
    identity = hashlib.sha256(json.dumps([conversation_id, message, focus, parent_id]).encode()).hexdigest()
    advisory(db, "work:" + request_id)
    previous = db.get(AgentWork, request_id)
    if previous:
        if previous.owner_id != owner or previous.account_id != account or previous.input_hash != identity:
            raise DomainError("REVISION_CONFLICT", "This request ID already belongs to another request.", 409)
        return previous
    pending = list(
        db.scalars(select(AgentWork).join(Job).where(AgentWork.account_id == account, Job.status.in_(ACTIVE)))
    )
    if len(pending) >= get_settings().agent_queue_limit:
        raise DomainError(
            "QUEUE_FULL", "Your work queue is full. Finish or cancel pending requests first.", 429
        )
    prefs = person_preferences(db, owner, device, preferences(db, owner))
    profile = agent_models.selected(prefs, require_key=True).profile_id
    transient = conv.private or not prefs["history_enabled"]
    # Serialize intake in a conversation; action concurrency is decided after routing.
    deps = list(dependencies or [])
    if kind == "agent_intake":
        deps += [
            p.id
            for p in pending
            if p.owner_id == owner
            and p.conversation_id == conversation_id
            and db.get(Job, p.id).kind == "agent_intake"
        ]
    job = Job(id=request_id, owner_id=owner, kind=kind, status="queued", payload={"profile": profile})
    db.add(job)
    db.flush()
    row = AgentWork(
        id=request_id,
        owner_id=owner,
        account_id=account,
        device_id=device,
        conversation_id=conversation_id,
        parent_id=parent_id,
        voice_session_id=voice_session_id,
        input_hash=identity,
        input_ciphertext=seal({"message": message, "context": context or [], "focus": focus}),
        revision=1,
        dependencies=list(dict.fromkeys(deps)),
        resources=resources or [],
        result={},
        transient=transient,
        expires_at=now() + timedelta(hours=24),
        updated_at=now(),
    )
    db.add(row)
    db.add(Outbox(job_id=request_id))
    db.flush()
    emit(db, owner, "work.changed", request_id, 1)
    return row


@contextmanager
def principal_for(row):
    token = execution.set(
        {
            "id": row.id,
            "owner_id": row.owner_id,
            "account_id": row.account_id,
            "device_id": row.device_id,
            "revision": row.revision,
        }
    )
    try:
        yield
    finally:
        execution.reset(token)


def eligible(db, job):
    row = db.get(AgentWork, job.id)
    if not row:
        return True
    if job.status not in ACTIVE:
        return False
    if row.cancel_requested or row.expires_at <= now() or not db.get(UserAccount, row.account_id):
        finish(db, row, "cancelled" if row.cancel_requested else "expired", "This request did not run.")
        return False
    try:
        role(db, row.owner_id, row.account_id)
    except DomainError:
        finish(db, row, "failed", "Workspace access ended before this request finished.")
        return False
    dependencies = [db.get(Job, identity) for identity in row.dependencies]
    if any(dep and dep.status in ACTIVE for dep in dependencies):
        return False
    # Failed dependencies need a user decision, never an optimistic continuation.
    if job.kind != "agent_intake" and any(
        dep and dep.status in {"failed", "cancelled", "expired", "partial", "needs_input"}
        for dep in dependencies
    ):
        job.status = "needs_input"
        row.result = {
            **row.result,
            "message": "Earlier related work needs attention. Continue or revise this request.",
        }
        emit(db, row.owner_id, "work.changed", row.id)
        return False
    lane = "agent_intake" if job.kind == "agent_intake" else "agent_action"
    busy = list(
        db.scalars(
            select(AgentWork)
            .join(Job)
            .where(
                AgentWork.account_id == row.account_id,
                Job.kind == lane,
                Job.id != row.id,
                Job.status.in_({"running", "dispatched"}),
            )
        )
    )
    return len(busy) < (1 if lane == "agent_intake" else get_settings().agent_account_parallelism)


def finish(db, row, status, message, **result):
    job = db.get(Job, row.id)
    job.status, job.finished_at = status, now()
    row.result = {**row.result, **result, "message": message}
    row.updated_at = now()
    # Checkpoints contain retrieved data/native provider signatures; never retain after completion.
    if status in {"succeeded", "cancelled", "expired"}:
        row.checkpoint_ciphertext = None
    if row.transient and status in TERMINAL:
        row.input_ciphertext = None
        row.checkpoint_ciphertext = None
        row.result = {
            key: value
            for key, value in row.result.items()
            if key in {"actions", "tool_calls", "child_ids", "route_kinds", "quiet", "receipt_ids"}
        }
        row.result["message"] = (
            "Saved changes are available in the action details."
            if result.get("actions")
            else "Request finished."
        )
    job.result = {"work_id": row.id, "status": status}
    emit(db, row.owner_id, "work.changed", row.id, row.revision)


def checkpoint(request_id, state):
    with session_scope() as db:
        row = db.get(AgentWork, request_id)
        row.checkpoint_ciphertext = seal(state)
        row.updated_at = now()
        emit(db, row.owner_id, "work.changed", row.id, row.revision)


def require_work(db, owner, account, request_id):
    role(db, owner, account)
    row = db.get(AgentWork, request_id)
    if not row or row.owner_id != owner or row.account_id != account:
        raise DomainError("NOT_FOUND", "That request is not available.", 404)
    return row


def cancel(db, row):
    advisory(db, "work:" + row.id)
    db.flush()
    db.refresh(row)
    job = db.get(Job, row.id)
    children = list(db.scalars(select(AgentWork).where(AgentWork.parent_id == row.id)))
    for child in children:
        cancel(db, child)
    if job.status in ACTIVE or job.status == "needs_input":
        row.cancel_requested = True
        row.updated_at = now()
        if job.status != "running":
            finish(db, row, "cancelled", "Cancelled. Previously saved changes remain.")
        emit(db, row.owner_id, "work.changed", row.id, row.revision)
    return public(db, row)


def revise(db, row, message, *, continue_work=False):
    advisory(db, "work:" + row.id)
    db.flush()
    db.refresh(row)
    job = db.get(Job, row.id)
    if row.parent_id is None and row.result.get("child_ids") and job.status != "needs_input":
        children = [db.get(AgentWork, identity) for identity in row.result["child_ids"]]
        if len(children) != 1:
            raise DomainError("AMBIGUOUS_REQUEST", "Choose the specific request to correct.", 409)
        return revise(db, children[0], message, continue_work=continue_work)
    if row.input_ciphertext is None:
        raise DomainError(
            "REQUEST_EXPIRED", "Start a new request; this request's temporary input was removed.", 409
        )
    data = unseal(row.input_ciphertext)
    corrections = list(data.get("corrections", []))
    if message:
        corrections.append(message[:12000])
    data["corrections"] = corrections[-15:]
    row.input_ciphertext = seal(data)
    row.revision += 1
    if job.kind == "agent_intake":
        row.checkpoint_ciphertext = None
    saved_state = unseal(row.checkpoint_ciphertext)
    if saved_state:
        saved_state["reply"] = None
        saved_state["needs_input"] = False
        saved_state["round"] = 0
        saved_state["errors"] = []
        row.checkpoint_ciphertext = seal(saved_state)
    row.cancel_requested = False
    row.expires_at = now() + timedelta(hours=24)
    row.updated_at = now()
    row.dependencies = [] if continue_work else row.dependencies
    # A live runner observes the revision and consumes the correction itself.
    if job.status != "running":
        reschedule(db, row)
    emit(db, row.owner_id, "work.changed", row.id, row.revision)
    return public(db, row)


def reschedule(db, row):
    job = db.get(Job, row.id)
    job.status, job.finished_at = "queued", None
    job.payload = {**job.payload, "dispatch_revision": row.revision}
    outbox = db.get(Outbox, row.id)
    if outbox:
        outbox.submitted_at = None
    else:
        db.add(Outbox(job_id=row.id))


def public(db, row, *, children=True):
    from .action_history import changes_for_work

    job = db.get(Job, row.id)
    data = unseal(row.input_ciphertext)
    result = dict(row.result or {})
    items = []
    if children:
        items = [
            public(db, child, children=False)
            for child in db.scalars(
                select(AgentWork).where(AgentWork.parent_id == row.id).order_by(AgentWork.updated_at)
            )
        ]
    status = "queued" if job.status == "dispatched" else job.status
    if items and job.status != "needs_input":
        statuses = {c["status"] for c in items}
        status = next(
            (
                state
                for state in ("running", "queued", "needs_input", "waiting_sync", "failed", "partial")
                if state in statuses
            ),
            "cancelled"
            if statuses == {"cancelled"}
            else "partial"
            if statuses & {"cancelled", "expired"}
            else "succeeded",
        )
    actions = changes_for_work(db, row)
    if status == "succeeded" and any(a.get("remote_status") in {*ACTIVE, "retrying"} for a in actions):
        status = "waiting_sync"
    elif status == "succeeded" and any(
        a.get("remote_status") in {"failed", "needs_input", "conflict", "unconfirmed", "cancelled"}
        for a in actions
    ):
        status = "partial"
    return {
        "id": row.id,
        "parent_id": row.parent_id,
        "conversation_id": row.conversation_id,
        "request": data.get("message", "Request"),
        "status": status,
        "revision": row.revision,
        "message": result.get("message", ""),
        "actions": actions,
        "children": items,
        "cancel_requested": row.cancel_requested,
        "seen": bool(row.seen_at),
        "created_at": job.created_at.isoformat(),
        "updated_at": row.updated_at.isoformat(),
        "can_continue": status in {"needs_input", "failed", "partial"} and bool(row.input_ciphertext),
    }


def list_work(db, owner, account, *, limit=50):
    role(db, owner, account)
    rows = list(
        db.scalars(
            select(AgentWork)
            .where(
                AgentWork.owner_id == owner,
                AgentWork.account_id == account,
                AgentWork.parent_id.is_(None),
                AgentWork.result["quiet"].as_boolean().is_not(True),
            )
            .order_by(AgentWork.updated_at.desc())
            .limit(limit)
        )
    )
    return {"items": [public(db, row) for row in rows]}
