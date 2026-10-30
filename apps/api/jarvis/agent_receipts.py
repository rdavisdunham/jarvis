"""Idempotent browser run actions (docs/AGENT_CONTROL.md, "Browser routes (runs)").

The run drawer may retry an action whose response was lost. Each action body carries an optional client
`action_id`; the first request acts and stores a receipt (digest of the route and body, and the response) in
the same transaction as the effect. A repeat with the same id and body returns the stored response without
acting again; the same id with a different body is refused with 409 ACTION_MISMATCH. Without an action_id the
action runs as before. A receipt only has to outlive a retry: it is kept 7 days (agent_retention.py).
"""

import hashlib
import json

from .domain import DomainError, advisory
from .session_models import AgentActionReceipt


def digest(kind, run_id, body):
    request = {"kind": kind, "run_id": run_id, "body": body.model_dump(mode="json", exclude={"action_id"})}
    return hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def once(db, owner, run_id, kind, body, act):
    """Run `act()` once per (owner, body.action_id); a repeat with the same body returns the first result."""
    if body.action_id is None:
        return act()
    action_id = str(body.action_id)
    request = digest(kind, run_id, body)
    # Serializes concurrent repeats: the second waits for the first's transaction, then reads its receipt.
    advisory(db, f"agent-action:{owner}:{action_id}")
    previous = db.get(AgentActionReceipt, (owner, action_id))
    if previous is not None:
        if previous.digest != request:
            raise DomainError("ACTION_MISMATCH", "This action id already belongs to a different request.", 409)
        return previous.result
    result = act()
    db.add(AgentActionReceipt(owner_id=owner, action_id=action_id, run_id=run_id, kind=kind, digest=request,
                              result=result))
    db.flush()
    return result
