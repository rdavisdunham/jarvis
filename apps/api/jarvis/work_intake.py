"""Durable voice capture. Live owns the conversation; only explicit delegation or sending a recovered draft creates work."""

import re
import hashlib
from datetime import timedelta

from sqlalchemy import or_, select, text

from .agent_work import ACTIVE, enqueue, finish, stable_id
from .db import session_scope
from .domain import DomainError, advisory, emit
from .models import AgentWork, Job, VoiceInbox, now
from .work_crypto import seal, unseal

# A Live controller closes its own inbox; this only catches a lost process.
ABANDONED = timedelta(minutes=30)


def try_advisory(db, key):
    return db.scalar(text("SELECT pg_try_advisory_xact_lock(hashtextextended(:key, 0))"), {"key": key})


def open_voice(db, session_id, owner, account, device, conversation):
    row = db.get(VoiceInbox, session_id)
    if not row:
        row = VoiceInbox(
            id=session_id,
            owner_id=owner,
            account_id=account,
            device_id=device,
            conversation_id=conversation,
            content_ciphertext=seal({"entries": []}),
            cursor=0,
            revision=0,
            expires_at=now() + timedelta(hours=24),
        )
        db.add(row)
        db.flush()
    return row


def append_voice(db, session_id, event_id, role, delta, start, end):
    # Lock order everywhere: voice-inbox advisory, then the inbox row.
    advisory(db, "voice-inbox:" + session_id)
    row = db.get(VoiceInbox, session_id, populate_existing=True)
    if not row or row.closed or row.expires_at <= now():
        return
    data = unseal(row.content_ciphertext)
    entries = data.get("entries", [])
    if event_id and any(item["id"] == event_id for item in entries):
        return
    entries.append({"id": event_id, "role": role, "text": delta, "start": start, "end": end})
    row.content_ciphertext = seal({**data, "entries": entries})
    row.revision += 1
    row.last_input_at = now()  # Quiet means neither speaker is mid-utterance.


def mark_asked(db, session_id, clarification_id):
    """Record that this session spoke a backend question; only heard questions bind answers."""
    advisory(db, "voice-inbox:" + session_id)
    row = db.get(VoiceInbox, session_id, populate_existing=True)
    if not row or row.closed:
        return
    data = unseal(row.content_ciphertext)
    asked = [*data.get("asked", []), clarification_id][-20:]
    row.content_ciphertext = seal({**data, "asked": list(dict.fromkeys(asked))})


def grouped(entries):
    result = []
    previous_end = None
    for item in entries:
        if (
            result
            and result[-1]["role"] == item["role"]
            and (previous_end is None or item["start"] - previous_end <= 1500)
        ):
            gap = "\n" if previous_end is not None and item["start"] - previous_end > 1500 else ""
            result[-1]["content"] += gap + item["text"]
        else:
            result.append({"role": item["role"], "content": item["text"]})
        previous_end = item["end"]
    return result


# A goodbye alone needs no recovery. Everything else remains a draft, never an action guess.
CLOSING = re.compile(
    r"^\W*(?:(?:ok(?:ay)?|alright|great|cool|perfect)\W+)?"
    r"(?:(?:that'?s all(?: for now)?|that will be all|that'?ll be all|i'?m done|we'?re done|"
    r"thanks?(?: you)?(?: so much| very much)?|thank you|cheers|bye(?: bye)?|goodbye|"
    r"good night|see you|talk (?:to you )?later|end voice)\W*)+$", re.IGNORECASE)


def undelegated(turns):
    """Preserve speech without guessing intent. Only standalone closing phrases are omitted."""
    return "\n".join(t["content"].strip() for t in turns
                     if t["role"] == "user" and t["content"].strip()
                     and not CLOSING.fullmatch(t["content"].strip()))


def list_drafts(db, owner, account, device):
    rows = db.scalars(select(VoiceInbox).where(
        VoiceInbox.owner_id == owner, VoiceInbox.account_id == account,
        VoiceInbox.device_id == device, VoiceInbox.closed.is_(True),
        VoiceInbox.expires_at > now()).order_by(VoiceInbox.last_input_at.desc()).limit(100))
    items = []
    for row in rows:
        draft = unseal(row.content_ciphertext).get("draft")
        if draft:
            items.append({"id": row.id, "conversation_id": row.conversation_id,
                          "message": draft["message"], "expires_at": row.expires_at.isoformat()})
    return {"items": items}


def resolve_draft(db, owner, account, device, identity, *, message=None):
    """Same lock as capture/close; one durable disposition even after a lost HTTP response."""
    advisory(db, "voice-inbox:" + identity)
    row = db.get(VoiceInbox, identity, populate_existing=True)
    if not row or (row.owner_id, row.account_id, row.device_id) != (owner, account, device):
        raise DomainError("NOT_FOUND", "This voice draft was not found.", 404)
    if row.expires_at <= now():
        raise DomainError("DRAFT_EXPIRED", "This voice draft has expired.", 410)
    data = unseal(row.content_ciphertext)
    if "resolution" in data:
        if (message is not None and data["resolution"]["status"] == "sent"
                and data.get("sent_hash") != hashlib.sha256(message.strip().encode()).hexdigest()):
            raise DomainError("DRAFT_ALREADY_SENT", "This draft was already sent. Open Activity to edit that request.", 409)
        return data["resolution"]
    draft = data.get("draft")
    if not row.closed or not draft:
        raise DomainError("NOT_FOUND", "This voice draft was not found.", 404)
    if message is None:
        result = {"status": "discarded"}
    else:
        from .config import require_external_services
        require_external_services()
        message = message.strip()
        if not message:
            raise DomainError("INVALID_ARGUMENT", "Write a request before sending it.")
        work = enqueue(db, owner, account, device, row.conversation_id,
                       stable_id("voice-draft:" + row.id), message,
                       context=draft["context"])
        result = {"status": "sent", "work_id": work.id, "conversation_id": row.conversation_id}
    row.content_ciphertext = seal({"entries": [], "resolution": result,
        **({"sent_hash": hashlib.sha256(message.encode()).hexdigest()} if message is not None else {})})
    emit(db, owner, "voice.draft.changed", row.id)
    return result


def claim_voice(db, row, *, close=False):
    """Delegate explicitly requested work, or retain unclaimed speech as a draft on close."""
    advisory(db, "voice-inbox:" + row.id)
    db.flush()
    db.refresh(row)
    if row.closed:
        return None
    data = unseal(row.content_ciphertext)
    entries = data.get("entries", [])
    turns = grouped(entries[row.cursor :])
    history = grouped(entries[: row.cursor])
    # Pauses and Live's own detail questions stay inside one delegated request.
    if close:
        message = undelegated(turns)
    else:
        message = "\n".join(t["content"].strip() for t in turns if t["role"] == "user" and t["content"].strip())
    context = history + [t for t in turns if t["role"] == "assistant"]
    if close:
        row.closed = True
        row.cursor = 0
        row.expires_at = now() + timedelta(hours=24)
        row.content_ciphertext = seal({"entries": [], **(
            {"draft": {"message": message, "context": context[-20:]}} if message else {})})
        if message:
            emit(db, row.owner_id, "voice.draft.changed", row.id)
        return None
    result = None
    if message:
        result = enqueue(
            db,
            row.owner_id,
            row.account_id,
            row.device_id,
            row.conversation_id,
            stable_id(f"voice:{row.id}:{row.cursor}:{len(entries)}"),
            message,
            voice_session_id=row.id,
            context=context[-20:],
        )
    # A delegated turn is always a new request. Pending questions reach the backend as context,
    # and it answers one with work_answer only when this turn actually is that answer.
    row.cursor = len(entries)
    # Bounded, encrypted context; unclaimed input is never removed.
    if row.cursor > 500:
        entries = entries[-100:]
        row.cursor = 100
    row.content_ciphertext = seal({**data, "entries": entries})
    return result


def flush_voice(db):
    """Quiet speech stays retained for Live to delegate; only abandoned inboxes are closed.

    Each inbox and each expiry uses its own short transaction with the same lock order as
    capture (advisory, then row), so the worker scan never holds voice or work locks."""
    stale = list(db.scalars(select(VoiceInbox.id).where(
        VoiceInbox.closed.is_(False), VoiceInbox.expires_at > now(),
        VoiceInbox.last_input_at < now() - ABANDONED).limit(20)))
    for identity in stale:
        with session_scope() as inner:
            if not try_advisory(inner, "voice-inbox:" + identity):
                continue
            row = inner.get(VoiceInbox, identity, populate_existing=True)
            if row and not row.closed and row.last_input_at < now() - ABANDONED:
                try:
                    claim_voice(inner, row, close=True)
                except DomainError:
                    row.closed = True
                    row.content_ciphertext = seal({"entries": []})
    for row in db.scalars(select(VoiceInbox).where(VoiceInbox.expires_at <= now()).limit(100)):
        db.delete(row)
    expired = list(db.scalars(
        select(AgentWork.id)
        .join(Job)
        .where(
            AgentWork.expires_at <= now(),
            or_(
                AgentWork.input_ciphertext.is_not(None),
                AgentWork.checkpoint_ciphertext.is_not(None),
                Job.status.in_([*ACTIVE, "needs_input"]),
            ),
        )
        .limit(100)
    ))
    for identity in expired:
        with session_scope() as inner:
            row = inner.get(AgentWork, identity)
            if not row:
                continue
            advisory(inner, "work-order:" + row.owner_id)
            advisory(inner, "work:" + row.id)
            inner.refresh(row)
            job = inner.get(Job, row.id, populate_existing=True)
            if job.status == "running":
                continue  # The runner observes WORK_EXPIRED and finishes it under its own locks.
            if job.status in ACTIVE or job.status == "needs_input":
                finish(inner, row, "expired", "This request expired. Saved changes remain.")
            row.input_ciphertext = None
            row.checkpoint_ciphertext = None


async def run(request_id):
    """Compatibility for an intake job accepted before the direct-execution release."""
    from . import work_runner

    with session_scope() as db:
        job = db.get(Job, request_id)
        row = db.get(AgentWork, request_id)
        if not job or not row or job.status not in ACTIVE:
            return
        if row.result.get("child_ids"):
            # Already-routed legacy work must never be executed a second time.
            finish(db, row, "succeeded", "See saved changes below.")
            return
        job.kind = "agent_action"
        state = unseal(row.checkpoint_ciphertext)
        if state and "messages" not in state:
            row.checkpoint_ciphertext = None
    await work_runner.run(request_id)
