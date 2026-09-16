"""Durable voice capture. Captured requests go straight to the backend; no routing model."""

from datetime import timedelta

from sqlalchemy import or_, select

from .agent_work import ACTIVE, enqueue, finish, stable_id
from .domain import advisory
from .models import AgentWork, Job, VoiceInbox, now
from .work_crypto import seal, unseal


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
    advisory(db, "voice-inbox:" + session_id)
    row = db.get(VoiceInbox, session_id, populate_existing=True)
    if not row or row.closed or row.expires_at <= now():
        return
    data = unseal(row.content_ciphertext)
    entries = data.get("entries", [])
    if event_id and any(item["id"] == event_id for item in entries):
        return
    entries.append({"id": event_id, "role": role, "text": delta, "start": start, "end": end})
    row.content_ciphertext = seal({"entries": entries})
    row.revision += 1
    row.last_input_at = now()  # Quiet means neither speaker is mid-utterance.


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


def claim_voice(db, row, *, close=False):
    advisory(db, "voice-inbox:" + row.id)
    db.flush()
    db.refresh(row)
    data = unseal(row.content_ciphertext)
    entries = data.get("entries", [])
    pending = entries[row.cursor :]
    result = None
    history = grouped(entries[: row.cursor])
    for index, turn in enumerate(grouped(pending)):
        if turn["role"] == "user" and turn["content"].strip():
            accepted = enqueue(
                db,
                row.owner_id,
                row.account_id,
                row.device_id,
                row.conversation_id,
                stable_id(f"voice:{row.id}:{row.revision}:{row.cursor}:{len(entries)}:{index}"),
                turn["content"].strip(),
                voice_session_id=row.id,
                context=history[-20:],
            )
            result = result or accepted
        history.append(turn)
    row.cursor = len(entries)
    if close:
        row.closed = True
        row.content_ciphertext = seal({"entries": []})
        row.cursor = 0
    # Bounded, encrypted context; unclaimed input is never removed.
    if not close and row.cursor > 500:
        row.content_ciphertext = seal({"entries": entries[-100:]})
        row.cursor = 100
    return result


def flush_voice(db):
    for row in db.scalars(
        select(VoiceInbox)
        .where(
            VoiceInbox.closed.is_(False),
            VoiceInbox.expires_at > now(),
            VoiceInbox.last_input_at < now() - timedelta(seconds=3),
        )
        .with_for_update(skip_locked=True)
        .limit(20)
    ):
        claim_voice(db, row)
    for row in db.scalars(select(VoiceInbox).where(VoiceInbox.expires_at <= now()).limit(100)):
        db.delete(row)
    for row in db.scalars(
        select(AgentWork)
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
    ):
        job = db.get(Job, row.id)
        if job.status in ACTIVE or job.status == "needs_input":
            finish(db, row, "expired", "This request expired. Saved changes remain.")
        row.input_ciphertext = None
        row.checkpoint_ciphertext = None


async def run(request_id):
    """Compatibility for an intake job accepted before the direct-execution release."""
    from . import work_runner
    from .db import session_scope

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
