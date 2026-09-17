"""Clarification continuations: fresh attempts, one logical activity card."""

from sqlalchemy import select

from .domain import DomainError, advisory, emit
from .models import AgentWork, Command, Job, now
from .work_crypto import seal, unseal

ANSWER_TOOL = {
    "type": "function", "name": "work_answer",
    "description": "Answer one pending clarification. Use its exact request_id and clarification_id from recent work or work_list BEFORE any other effects. The server attaches this entire user turn (including any extra explicit requests) to the original work and restarts this attempt with its question and saved receipts. No approval is needed. Do not use for an unrelated request or an edit to already completed work; use work_followup for those edits. An ambiguous answer needs work_needs_input. A stale question requires rereading work_list, never replaying the original creation.",
    "parameters": {"type": "object", "properties": {
        "request_id": {"type": "string", "format": "uuid"},
        "clarification_id": {"type": "string", "format": "uuid"},
    }, "required": ["request_id", "clarification_id"], "additionalProperties": False},
}


class WorkContinued(Exception):
    """The current attempt was atomically adopted and must rebuild its context."""


def question_for(db, row):
    from .agent_work import stable_id
    if db.get(Job, row.id).status != "needs_input":
        return None
    return row.result.get("clarification") or {
        "id": stable_id(f"clarification:{row.id}:{row.revision}:{row.result.get('tool_calls', 0)}"),
        "request_id": row.id, "revision": row.revision, "question": row.result.get("message", ""),
    }


def root_for(db, row):
    identity = row.result.get("continuation_root")
    return db.get(AgentWork, identity) if identity else row


def attempts(db, row):
    root = root_for(db, row)
    return [root, *[db.get(AgentWork, identity) for identity in root.result.get("continuation_ids", [])]]


def latest(db, row):
    return attempts(db, row)[-1]


def touch_root(db, row):
    root = root_for(db, row)
    if root.id != row.id:
        root.updated_at = now()
        root.revision += 1
        root.seen_at = None
        emit(db, root.owner_id, "work.changed", root.id, root.revision)


def answer(row, arguments):
    from .agent_work import require_work, reschedule
    from .db import session_scope
    from .work_coordination import order

    if set(arguments) != {"request_id", "clarification_id"} or not all(isinstance(v, str) for v in arguments.values()):
        raise DomainError("INVALID_ARGUMENT", "Choose the exact pending question.")
    with session_scope() as db:
        advisory(db, "work-order:" + row.owner_id)
        current = require_work(db, row.owner_id, row.account_id, row.id)
        if current.result.get("answered_clarification_id") == arguments["clarification_id"]:
            return {"status": "continued", "request_id": current.parent_id}
        target = require_work(db, row.owner_id, row.account_id, arguments["request_id"])
        if target.credential_id != current.credential_id or order(db, target) >= order(db, current):
            raise DomainError("INVALID_DEPENDENCY", "Choose an earlier request from this account and agent.")
        question = question_for(db, target) or {}
        if (db.get(Job, target.id).status != "needs_input" or target.cancel_requested
                or target.result.get("archived_at") or target.expires_at <= now()
                or question.get("id") != arguments["clarification_id"]
                or question.get("revision") != target.revision
                or latest(db, target).id != target.id):
            raise DomainError("STALE_CLARIFICATION", "That question is no longer pending. Read work_list; use work_followup for a further correction, without repeating the original actions.", 409)
        if current.cancel_requested or current.result.get("archived_at") or db.get(Job, current.id).status not in {"queued", "dispatched", "running"}:
            raise DomainError("WORK_CANCELLED", "This reply was cancelled.", 409)
        if current.parent_id or current.resources or current.result.get("receipt_ids") or db.scalar(select(Command.id).where(
            Command.owner_id == row.owner_id, Command.id.startswith(row.id + ":")).limit(1)):
            raise DomainError("INVALID_DEPENDENCY", "Answer the clarification before performing any other actions.")
        root = root_for(db, target)
        data = unseal(current.input_ciphertext)
        earlier = unseal(root.input_ciphertext)
        if not earlier or not data:
            raise DomainError("REQUEST_EXPIRED", "The original context expired. Ask for a new request.", 409)
        # Keep the captured turn verbatim; the model cannot substitute its own answer.
        history = []
        for attempt in attempts(db, target):
            previous = unseal(attempt.input_ciphertext)
            if previous.get("clarification_answer"):
                history.append(previous["clarification_answer"])
        data["continuation_request"] = earlier["message"]
        data["clarification_history"] = history
        data["clarification_answer"] = {"question": question["question"], "answer": data["message"]}
        current.input_ciphertext = seal(data)
        current.parent_id = root.id
        current.result = {**current.result, "continuation_root": root.id,
            "answered_clarification_id": question["id"], "related_request_id": root.id}
        current.checkpoint_ciphertext = None
        current.revision += 1
        current.dependencies = list(dict.fromkeys([*current.dependencies, target.id]))
        # The old attempt is never executed again. Its receipts stay immutable.
        old_job = db.get(Job, target.id)
        old_job.status, old_job.finished_at = "continued", now()
        old_job.result = {"work_id": target.id, "status": "continued"}
        target.checkpoint_ciphertext = None
        target.result = {**target.result, "continued_as": current.id}
        target.updated_at = now()
        root.expires_at = max(root.expires_at, current.expires_at)
        root.result = {**root.result, "continuation_ids": [*root.result.get("continuation_ids", []), current.id]}
        touch_root(db, current)
        reschedule(db, current)
        emit(db, current.owner_id, "work.changed", current.id, current.revision)
    raise WorkContinued()
