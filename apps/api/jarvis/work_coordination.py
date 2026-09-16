"""Deterministic dependencies and record reservations for the existing backend agent.

Models identify references. Database state decides readiness. No extra model call,
rewritten speech, or exact-quote validation is involved in accepting a request.
"""

from datetime import timedelta

from sqlalchemy import select

from .agent_work import ACTIVE, require_work, reschedule
from .db import session_scope
from .domain import COMMANDS, DomainError, advisory, emit
from .models import AgentWork, Command, Job, now
from .work_crypto import seal, unseal

FOLLOWUP_TOOL = {
    "type": "function",
    "name": "work_followup",
    "description": "Link THIS request to an earlier request when the user edits a specific earlier result, answers its clarification, or asks about its result (for example 'make that call tomorrow'). Use the exact request_id from recent work or work_list BEFORE editing or recreating anything. The scheduler waits if needed and returns the earlier outcome and saved record IDs. A failed original is not proof a record exists. Never link a fully specified new creation merely because it says 'also' or follows another request. 'Add Call Alex' and 'Also add Buy milk' are independent; only 'make that call tomorrow' links to Call Alex. Ambiguous references need work_needs_input.",
    "parameters": {
        "type": "object",
        "properties": {"request_id": {"type": "string", "format": "uuid"}},
        "required": ["request_id"],
        "additionalProperties": False,
    },
}


class WorkDeferred(Exception):
    def __init__(self, dependencies):
        self.dependencies = list(dict.fromkeys(dependencies))


def order(db, row):
    return (db.get(Job, row.id).created_at, row.id)


def describe(db, row):
    receipts = list(
        db.scalars(
            select(Command)
            .where(
                Command.owner_id == row.owner_id,
                Command.account_id == row.account_id,
                Command.id.startswith(row.id + ":"),
            )
            .order_by(Command.created_at)
            .limit(30)
        )
    )
    return {
        "request_id": row.id,
        "request": unseal(row.input_ciphertext).get("message", "")[:3000],
        "status": db.get(Job, row.id).status,
        "related_request_id": row.result.get("related_request_id"),
        "outcome": row.result.get("message", "")[:1500],
        "saved_records": [
            {
                "command_id": c.id,
                "status": c.result.get("status"),
                "data": {
                    k: v
                    for k, v in c.result.get("data", {}).items()
                    if k in {"id", "title", "name", "revision", "status", "due_at", "due_date", "job_id"}
                },
            }
            for c in receipts
        ],
    }


def recent(db, row):
    candidates = list(
        db.scalars(
            select(AgentWork)
            .join(Job)
            .where(
                AgentWork.owner_id == row.owner_id,
                AgentWork.account_id == row.account_id,
                AgentWork.conversation_id == row.conversation_id,
                AgentWork.id != row.id,
                Job.created_at <= db.get(Job, row.id).created_at,
            )
            .order_by(Job.created_at.desc(), Job.id.desc())
            .limit(12)
        )
    )
    return [describe(db, item) for item in reversed(candidates) if order(db, item) < order(db, row)]


def followup(row, arguments):
    if set(arguments) != {"request_id"} or not isinstance(arguments["request_id"], str):
        raise DomainError("INVALID_ARGUMENT", "Choose an exact earlier request ID.")
    with session_scope() as db:
        advisory(db, "work-order:" + row.owner_id)
        current = db.get(AgentWork, row.id, populate_existing=True)
        target = require_work(db, row.owner_id, row.account_id, arguments["request_id"])
        if order(db, target) >= order(db, current):
            raise DomainError("INVALID_DEPENDENCY", "A follow-up can only refer to earlier work.")
        existing = current.result.get("related_request_id")
        if existing and existing != target.id:
            raise DomainError(
                "INVALID_DEPENDENCY",
                "This request already follows another request. Resolve its result first.",
            )
        current.result = {**current.result, "related_request_id": target.id}
        status = db.get(Job, target.id).status
        if status in ACTIVE:
            # Commit the link, then park outside this transaction.
            waiting = target.id
            current.dependencies = list(dict.fromkeys([*current.dependencies, target.id]))
        else:
            waiting = None
            result = describe(db, target)
    if waiting:
        raise WorkDeferred([waiting])
    return result


def tool_resources(name, arguments, row, index):
    """Stable record IDs for mutations; unknown/bulk scopes are deliberately conservative."""
    if name.startswith("ui_") and name != "ui_state":
        return {"device:" + row.device_id}
    if name == "work_revert":
        return {"*"}
    command = name.replace("_", ".", 1)
    if command not in COMMANDS:
        return set()
    if command.startswith(("settings.", "planning.", "calendar.", "linear.")):
        return {"*"}
    keys = set()

    def collect(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key.endswith("_id") and isinstance(item, str):
                    keys.add("record:" + item)
                elif key.endswith("_ids") and isinstance(item, list):
                    keys.update("record:" + identity for identity in item if isinstance(identity, str))
                elif isinstance(item, (dict, list)):
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(arguments)
    if command.endswith(".create"):
        keys.add(f"new:{row.id}:{index}")
    # Name-based references and selection tokens are unresolved until the domain lookup.
    if command == "task.selection_update" or any(
        arguments.get(key) not in (None, "", "owner")
        for key in ("project", "space", "area", "assignee", "goal", "selection_token")
    ):
        keys.add("*")
    return keys or {"*"}


def overlaps(left, right):
    return bool(left & right) or "*" in left or "*" in right


def reserve(row, name, arguments, index):
    keys = tool_resources(name, arguments, row, index)
    if not keys:
        return
    with session_scope() as db:
        advisory(db, "work-order:" + row.owner_id)
        current = db.get(AgentWork, row.id, populate_existing=True)
        prior_keys = set(current.resources)
        peers = list(
            db.scalars(
                select(AgentWork)
                .join(Job)
                .where(
                    AgentWork.owner_id == row.owner_id,
                    AgentWork.id != row.id,
                    Job.created_at >= db.get(Job, row.id).created_at - timedelta(hours=24),
                )
            )
        )
        blockers = []
        for peer in peers:
            peer_keys = set(peer.resources)
            status = db.get(Job, peer.id).status
            if order(db, peer) < order(db, current) and status in ACTIVE:
                fresh_create = all(key.startswith("new:") for key in keys)
                if overlaps(keys, peer_keys) or (not peer_keys and not fresh_create):
                    blockers.append(peer.id)
            elif (
                order(db, peer) > order(db, current)
                and peer_keys
                and current.id not in peer.dependencies
                and overlaps(keys - prior_keys, peer_keys)
            ):
                if keys - prior_keys:
                    raise DomainError(
                        "WORK_ORDER_CONFLICT",
                        "A newer request already reserved this record. Do not overwrite its changes. Read the current record and report the conflict; a fresh request can apply a new edit.",
                    )
        current.resources = sorted(prior_keys | keys)
        current.dependencies = list(dict.fromkeys([*current.dependencies, *blockers]))
    if blockers:
        raise WorkDeferred(blockers)


def park(db, row, state, dependencies):
    advisory(db, "work-order:" + row.owner_id)
    row.dependencies = list(dict.fromkeys([*row.dependencies, *dependencies]))
    row.checkpoint_ciphertext = seal(state)
    row.result = {**row.result, "waiting_for": dependencies, "message": "Waiting for earlier related work."}
    row.updated_at = now()
    # A fresh DBOS invocation is required after this invocation returns.
    reschedule(db, row)
    emit(db, row.owner_id, "work.changed", row.id, row.revision)


def record_saved_resources(db, owner, data):
    """Bind a pending creation to its real IDs in the same transaction as its receipt."""
    from .access import execution

    grant = execution.get()
    if not grant or grant["owner_id"] != owner:
        return
    keys = set()

    def collect(value):
        if isinstance(value, dict):
            if isinstance(value.get("id"), str):
                keys.add("record:" + value["id"])
            for key in ("tasks", "notes", "entries", "results", "items"):
                collect(value.get(key))
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(data)
    if keys:
        advisory(db, "work-order:" + owner)
        row = db.get(AgentWork, grant["id"], populate_existing=True)
        if row:
            row.resources = sorted(set(row.resources) | keys)
