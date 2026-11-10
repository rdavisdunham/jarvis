"""Idempotent browser run actions (docs/AGENT_CONTROL.md, "Browser routes (runs)").

The run drawer may retry an action whose response was lost. Each action body carries an optional client
`action_id`; the first request acts and stores a receipt (digest of the route and body, and the response) in
the same transaction as the effect. A repeat with the same id and body returns the stored response without
acting again; the same id with a different body is refused with 409 ACTION_MISMATCH. Without an action_id the
action runs as before. Receipts for active runs or unsettled commands survive routine retention;
settled history follows the retention policy in agent_retention.py.
"""

import hashlib
import json
from contextvars import ContextVar

from .domain import DomainError, advisory
from .session_models import AgentActionReceipt

action_context = ContextVar("agent_action_id", default=None)


def digest(kind, run_id, body):
    request = {"kind": kind, "run_id": run_id, "body": body.model_dump(mode="json", exclude={"action_id"})}
    return hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def once(db, owner, run_id, kind, body, act, *, scope_id=None):
    """Run `act()` once per (owner, body.action_id); a repeat with the same body returns the first result."""
    if body.action_id is None:
        return act()
    action_id = str(body.action_id)
    request = digest(kind, scope_id or run_id, body)
    # Serializes concurrent repeats: the second waits for the first's transaction, then reads its receipt.
    advisory(db, f"agent-action:{owner}:{action_id}")
    previous = db.get(AgentActionReceipt, (owner, action_id))
    if previous is not None:
        if previous.digest != request:
            raise DomainError("ACTION_MISMATCH", "This action id already belongs to a different request.", 409)
        return previous.result
    token = action_context.set(action_id)
    try:
        result = act()
    finally:
        action_context.reset(token)
    receipt_run_id = run_id
    if kind in ("start", "session", "restart"):
        receipt_run_id = result.get("id", run_id)
    elif kind in ("interview", "answer"):
        receipt_run_id = result.get("run_id", run_id)
    elif kind.startswith("decision:"):
        receipt_run_id = (result.get("decision") or {}).get("run_id") or run_id
    db.add(AgentActionReceipt(owner_id=owner, action_id=action_id, run_id=receipt_run_id, kind=kind,
                              digest=request, result=result))
    db.flush()
    return result


def lookup(db, owner, action_id):
    """Persisted acceptance and delivery are separate facts; uncertain effects are never called delivered."""
    from sqlalchemy import select

    from .domain import encode
    from .session_models import AgentCommand
    receipt = db.get(AgentActionReceipt, (owner, str(action_id)))
    if receipt is None:
        raise DomainError("NOT_FOUND", "This action has no saved receipt.", 404)
    commands = list(db.scalars(select(AgentCommand).where(
        AgentCommand.owner_id == owner, AgentCommand.action_id == str(action_id)).order_by(AgentCommand.created_at)))
    states = {c.status for c in commands if c.status != "superseded"}
    status = ("uncertain" if "uncertain" in states else "failed" if states & {"failed", "rejected", "expired", "canceled"}
              else "queued" if states & {"deferred", "queued", "sent", "acked"} else "delivered" if states else "saved")
    direct = [command for command in commands if (command.payload or {}).get("mode") == "direct"]
    stages = [((command.outcome or {}).get("result") or {}).get("receipt_stage") or
              ((command.outcome or {}).get("result") or {}).get("receipt") or
              ("uncertain" if command.status == "uncertain" else "handed_to_harness" if command.status == "executed" else "saved")
              for command in direct]
    return encode({"action_id": str(action_id), "run_id": receipt.run_id, "kind": receipt.kind,
                   "status": status, "terminal": status != "queued", "result": receipt.result,
                   **({"delivery_stage": stages[-1]} if stages else {}),
                   "commands": [{"op_id": c.op_id, "kind": c.kind, "status": c.status, "outcome": c.outcome} for c in commands]})
