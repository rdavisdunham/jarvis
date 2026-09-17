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
TERMINAL = {"succeeded", "partial", "failed", "cancelled", "expired", "continued"}


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
    kind="agent_action",
    parent_id=None,
    voice_session_id=None,
    context=None,
    focus=None,
    dependencies=None,
    resources=None,
    credential_id=None,
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
    advisory(db, "work-order:" + owner)
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
    deps = list(dependencies or [])
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
        credential_id=credential_id,
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
            "credential_id": row.credential_id,
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
        if row.credential_id:
            from .bot_access import authorize
            authorize(db, row.owner_id, "work:run", credential_id=row.credential_id)
    except DomainError:
        finish(db, row, "failed", "Workspace access ended before this request finished.")
        return False
    from .work_continuation import latest
    dependencies = []
    for identity in row.dependencies:
        prior = db.get(AgentWork, identity)
        dependency = latest(db, prior).id if prior else identity
        if dependency != row.id:
            dependencies.append(db.get(Job, dependency))
    if any(dep and dep.status in ACTIVE for dep in dependencies):
        return False
    # A resumed backend receives the actual predecessor outcome, including failures.
    # Waiting releases the worker slot; no model guesses whether a predecessor finished.
    lane = "agent_action"
    busy = list(
        db.scalars(
            select(AgentWork)
            .join(Job)
            .where(
                AgentWork.account_id == row.account_id,
                Job.kind.in_({lane, "agent_intake"}),
                Job.id != row.id,
                Job.status.in_({"running", "dispatched"}),
            )
        )
    )
    return len(busy) < get_settings().agent_account_parallelism


def finish(db, row, status, message, **result):
    job = db.get(Job, row.id)
    if status == "needs_input":
        result["clarification"] = {
            "id": stable_id(f"clarification:{row.id}:{row.revision}:{result.get('tool_calls', 0)}"),
            "request_id": row.id, "revision": row.revision, "question": message,
        }
    if status=="succeeded":
        from .routing import append_offer
        message=append_offer(db,row,message)
    job.status, job.finished_at = status, now()
    row.result = {**row.result, **result, "message": message}
    row.updated_at = now()
    row.seen_at = None
    # Checkpoints contain retrieved data/native provider signatures; never retain after completion.
    if status in {"succeeded", "cancelled", "expired"}:
        row.checkpoint_ciphertext = None
    if row.transient and status in TERMINAL:
        row.input_ciphertext = None
        row.checkpoint_ciphertext = None
        row.result = {
            key: value
            for key, value in row.result.items()
            if key
            in {
                "actions",
                "tool_calls",
                "child_ids",
                "route_kinds",
                "quiet",
                "receipt_ids",
                "related_request_id", "continuation_root", "continuation_ids", "continued_as",
                "answered_clarification_id", "archived_at",
            }
        }
        row.result["message"] = (
            "Saved changes are available in the action details."
            if result.get("actions")
            else "Request finished."
        )
    from .work_continuation import touch_root
    touch_root(db, row)
    from .notices import work_finished
    work_finished(db,row,status,message)
    job.result = {"work_id": row.id, "status": status}
    emit(db, row.owner_id, "work.changed", row.id, row.revision)


def checkpoint(request_id, state):
    with session_scope() as db:
        row = db.get(AgentWork, request_id)
        row.checkpoint_ciphertext = seal(state)
        row.updated_at = now()
        from .work_continuation import touch_root
        touch_root(db, row)
        emit(db, row.owner_id, "work.changed", row.id, row.revision)


def require_work(db, owner, account, request_id):
    role(db, owner, account)
    row = db.get(AgentWork, request_id)
    from .bot_access import current_id
    if (not row or row.owner_id != owner or row.account_id != account
            or (current_id() and row.credential_id != current_id())):
        raise DomainError("NOT_FOUND", "That request is not available.", 404)
    return row


def cancel(db, row):
    advisory(db, "work-order:" + row.owner_id)
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
    advisory(db, "work-order:" + row.owner_id)
    from .work_continuation import latest
    current = latest(db, row)
    if current.id != row.id:
        revise(db, current, message, continue_work=continue_work)
        return public(db, row)
    if row.result.get("archived_at"):
        raise DomainError("REQUEST_EXPIRED", "This activity was cleared. Start a new request.", 409)
    advisory(db, "work:" + row.id)
    db.flush()
    db.refresh(row)
    job = db.get(Job, row.id)
    if job.kind == "external_command":
        raise DomainError("INVALID_ARGUMENT", "Edit the saved record to change this direct command.")
    if row.parent_id is None and row.result.get("child_ids") and job.status != "needs_input":
        children = [db.get(AgentWork, identity) for identity in row.result["child_ids"]]
        if len(children) != 1:
            raise DomainError("AMBIGUOUS_REQUEST", "Choose the specific request to correct.", 409)
        return revise(db, children[0], message, continue_work=continue_work)
    if row.input_ciphertext is None:
        raise DomainError(
            "REQUEST_EXPIRED", "Start a new request; this request's temporary input was removed.", 409
        )
    if row.credential_id:
        from .bot_access import authorize
        authorize(db, row.owner_id, "work:run", credential_id=row.credential_id)
    data = unseal(row.input_ciphertext)
    if job.status == "needs_input" and message:
        data["last_question"] = row.result.get("message", "")
    corrections = list(data.get("corrections", []))
    if message:
        corrections.append(message[:12000])
    data["corrections"] = corrections[-15:]
    row.input_ciphertext = seal(data)
    row.revision += 1
    if job.kind == "agent_intake":
        job.kind = "agent_action"
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
    from .work_continuation import touch_root
    touch_root(db, row)
    emit(db, row.owner_id, "work.changed", row.id, row.revision)
    return public(db, row)


def reschedule(db, row):
    job = db.get(Job, row.id)
    job.status, job.finished_at = "queued", None
    job.payload = {
        **job.payload,
        "dispatch_revision": max(row.revision, int(job.payload.get("dispatch_revision", 0)) + 1),
    }
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
    if children and not result.get("continuation_ids"):
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
    history = []
    from .work_continuation import question_for
    clarification = question_for(db, row) if status == "needs_input" else None
    active = row
    if result.get("continuation_ids"):
        from .work_continuation import attempts
        chain = attempts(db, row)
        active = chain[-1]
        current = public(db, active, children=False)
        status = current["status"]
        result = {**result, "message": current["message"], "waiting_for": active.result.get("waiting_for", [])}
        clarification = current.get("clarification")
        items = []  # Continuations share this card, unlike legacy independent child work.
        for attempt in chain[1:]:
            actions.extend(changes_for_work(db, attempt))
            answer = unseal(attempt.input_ciphertext).get("clarification_answer")
            if answer:
                history.append(answer)
        actions = list({action["id"]: action for action in actions}.values())
    if status == "succeeded" and any(a.get("remote_status") in {*ACTIVE, "retrying"} for a in actions):
        status = "waiting_sync"
    elif status == "succeeded" and any(
        a.get("remote_status") in {"failed", "needs_input", "conflict", "unconfirmed", "cancelled"}
        for a in actions
    ):
        status = "partial"
    from .models import BotCredential
    bot = db.get(BotCredential, row.credential_id) if row.credential_id else None
    active_job = db.get(Job, active.id)
    return {
        "id": row.id,
        "actor": {"type": "bot", "id": bot.id, "name": bot.name} if bot else None,
        "parent_id": row.parent_id,
        "related_request_id": result.get("related_request_id"),
        "waiting": bool(result.get("waiting_for")) and status == "queued",
        "conversation_id": row.conversation_id,
        "voice_session_id": active.voice_session_id,
        "navigation_only": bool(active.result.get("navigation_only")),
        "response_native_id": f"work:{active.id}:assistant:{active.revision}",
        "finished_at": active_job.finished_at.isoformat() if active_job.finished_at else None,
        "request": data.get("message", "Request"),
        "status": status,
        "revision": row.revision,
        "message": result.get("message", ""),
        "clarification": clarification,
        "clarification_history": history,
        "actions": actions,
        "children": items,
        "cancel_requested": active.cancel_requested,
        "seen": bool(row.seen_at),
        "created_at": job.created_at.isoformat(),
        "updated_at": row.updated_at.isoformat(),
        "can_revise": job.kind != "external_command",
        "can_continue": job.kind != "external_command" and status in {"needs_input", "failed", "partial"} and bool(active.input_ciphertext) and not row.result.get("archived_at"),
    }


def list_work(db, owner, account, *, limit=50, conversation_id=None, offset=0):
    role(db, owner, account)
    from .bot_access import current_id
    rows = list(
        db.scalars(
            select(AgentWork).join(Job)
            .where(
                AgentWork.owner_id == owner,
                AgentWork.account_id == account,
                AgentWork.parent_id.is_(None),
                *([AgentWork.credential_id == current_id()] if current_id() else []),
                *([AgentWork.conversation_id == conversation_id] if conversation_id else
                  [AgentWork.result["quiet"].as_boolean().is_not(True)]),
                AgentWork.result["archived_at"].as_string().is_(None),
            )
            .order_by(*( [Job.created_at, AgentWork.id] if conversation_id else [AgentWork.updated_at.desc()] ))
            .offset(offset if conversation_id else 0)
            .limit(limit + 1 if conversation_id else limit)
        )
    )
    return {"items": [public(db, row) for row in rows[:limit]],
            "next_offset": offset + limit if conversation_id and len(rows) > limit else None}


def clear_history(db, owner, account, before):
    """Hide this account's existing activity; stop unfinished work, retain saved effects/audit."""
    role(db, owner, account)
    advisory(db, "work-order:" + owner)
    rows = list(db.scalars(select(AgentWork).join(Job).where(
        AgentWork.owner_id == owner, AgentWork.account_id == account,
        Job.created_at <= before, AgentWork.result["archived_at"].as_string().is_(None),
    ).order_by(Job.created_at)))
    for row in rows:
        cancel(db, row)
        row.result = {**row.result, "archived_at": now().isoformat(), "quiet": True}
        row.seen_at = now()
        row.revision += 1  # In-flight model responses cannot execute stale plans.
        emit(db, owner, "work.changed", row.id, row.revision)
    return {"cleared": sum(row.parent_id is None for row in rows)}
