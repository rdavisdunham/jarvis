"""Retention and the ingestion quota for agent data (stage F, L7; docs/AGENT_CONTROL.md, "Security model").

The worker's housekeeping pass deletes what nobody needs any more: turns of sessions no run owns after
30 days, a settled run's turns, commands, questions, attempts and interviews after 90 days (the run row,
its frozen plan and its history stay), and the payload of every settled command after 7 days (the replay
horizon: the host never asks for it again). Parity rows go after 30 days; run action receipts
(agent_receipts.py) after 7 days, long past any client retry.

Each owner may ingest QUOTA_BYTES of turn text and prompts per UTC day.
Past it the text is dropped (the turn is kept, marked truncated) and the host shows as degraded.
"""

import uuid
from datetime import timedelta

from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert

from .models import now
from .session_models import (
    TERMINAL_RUN_STATES,
    AgentActionReceipt,
    AgentAttempt,
    AgentCommand,
    AgentIngestDay,
    AgentInterview,
    AgentParity,
    AgentParityDay,
    AgentPendingControl,
    AgentQuestion,
    AgentRun,
    AgentSession,
    AgentTurn,
)

UNBOUND_TURNS = timedelta(days=30)
RUN_DATA = timedelta(days=90)
COMMAND_PAYLOADS = timedelta(days=7)
PARITY = timedelta(days=30)
ACTION_RECEIPTS = timedelta(days=7)
QUOTA_BYTES = 64 * 1_048_576
SETTLED = ("acked", "executed", "failed", "rejected", "uncertain", "expired", "canceled")


def sweep(db, clock=None):
    clock = clock or now()
    counts = {}
    bound = select(AgentSession.id).where(AgentSession.run_id.is_not(None))
    counts["unbound_turns"] = db.execute(delete(AgentTurn).where(
        AgentTurn.run_id.is_(None), AgentTurn.session_pk.not_in(bound),
        AgentTurn.created_at < clock - UNBOUND_TURNS)).rowcount
    old = select(AgentRun.id).where(AgentRun.state.in_(TERMINAL_RUN_STATES), AgentRun.finished_at < clock - RUN_DATA)
    for name, model in (("run_turns", AgentTurn), ("run_questions", AgentQuestion), ("run_attempts", AgentAttempt),
                        ("run_interviews", AgentInterview), ("run_commands", AgentCommand)):
        counts[name] = db.execute(delete(model).where(model.run_id.in_(old))).rowcount
    counts["command_payloads"] = db.execute(update(AgentCommand).where(
        AgentCommand.status.in_(SETTLED), AgentCommand.updated_at < clock - COMMAND_PAYLOADS,
        AgentCommand.payload != {}).values(payload={})).rowcount
    counts["parity"] = db.execute(delete(AgentParity).where(AgentParity.updated_at < clock - PARITY)).rowcount
    counts["parity_days"] = db.execute(delete(AgentParityDay).where(
        AgentParityDay.day < (clock - PARITY).date())).rowcount
    db.execute(delete(AgentIngestDay).where(AgentIngestDay.day < (clock - PARITY).date()))
    db.execute(delete(AgentPendingControl).where(AgentPendingControl.expires_at < clock - PARITY))
    counts["action_receipts"] = db.execute(delete(AgentActionReceipt).where(
        AgentActionReceipt.created_at < clock - ACTION_RECEIPTS)).rowcount
    return counts


def used(db, owner, clock=None):
    row = db.scalar(select(AgentIngestDay.bytes).where(AgentIngestDay.owner_id == owner,
                                                       AgentIngestDay.day == (clock or now()).date()))
    return row or 0


def degraded(db, owner):
    return used(db, owner) >= QUOTA_BYTES


def admit(db, owner, *texts):
    """Count this text toward today's quota. False when the owner is over it: the caller drops the text."""
    size = sum(len(t.encode()) for t in texts if t)
    if not size:
        return True
    if degraded(db, owner):
        return False
    table = AgentIngestDay.__table__
    statement = insert(table).values(id=str(uuid.uuid4()), owner_id=owner, day=now().date(), bytes=size,
                                     updated_at=now())
    db.execute(statement.on_conflict_do_update(
        constraint="uq_agent_ingest_day",
        set_={"bytes": table.c.bytes + statement.excluded.bytes, "updated_at": statement.excluded.updated_at}))
    return True
