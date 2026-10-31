"""Retention, redaction and the ingestion quota for agent data (stage F, L7; docs/AGENT_CONTROL.md,
"Retention and redaction").

The worker's housekeeping pass deletes what nobody needs any more: turns and settled questions of sessions
no run owns after 30 days; a settled run's turns (its own and its session's), commands, questions, attempts
and interviews once the owner's run retention window has passed (`agent_retention_days`, default 90 days);
and the payload of every settled command after 7 days (the replay horizon: the host never asks for it
again). Parity rows go after 30 days; run action receipts (agent_receipts.py) after 7 days, long past any
client retry.

The run row, its frozen plans and its bounded history outlive the window as a skeleton, so the same pass
redacts the copies of run content they hold (`redact`): history `text`/`detail`, the frozen PRD text and
the agent's proposals, the run's notification bodies and pending-control summaries; the run's action
receipts (which hold whole run responses) are deleted. Actors, times, events, states, reasons, plan titles,
steps and the criteria the judge used are kept. `POST /api/v1/remotea/runs/{id}/redact` does the same for
one settled run at once (`redact_now`) and keeps its attempts with their verdicts but not their text.

Each owner may ingest QUOTA_BYTES of turn text and prompts per UTC day.
Past it the text is dropped (the turn is kept, marked truncated) and the host shows as degraded.
"""

import uuid
from datetime import timedelta

from sqlalchemy import delete, false, or_, select, update
from sqlalchemy.dialects.postgresql import insert

from .domain import DomainError, emit
from .models import Notification, OwnerSettings, now
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
    AgentPlanReview,
    AgentQuestion,
    AgentRun,
    AgentRunPlan,
    AgentSession,
    AgentSwarm,
    AgentTurn,
)

UNBOUND_TURNS = timedelta(days=30)
RUN_DATA = timedelta(days=90)  # The default run retention window; an owner may set `agent_retention_days`.
SETTING = "agent_retention_days"
MIN_DAYS, MAX_DAYS = 1, 3650
COMMAND_PAYLOADS = timedelta(days=7)
PARITY = timedelta(days=30)
ACTION_RECEIPTS = timedelta(days=7)
QUOTA_BYTES = 64 * 1_048_576
SETTLED = ("acked", "executed", "failed", "rejected", "uncertain", "expired", "canceled")
MARKER = "[redacted after retention]"
# Free text a history entry may carry: a sent message, a transition's or an outcome's detail.
HISTORY_TEXT = ("text", "detail")
# Free text in an attempt's result: verify tails and errors, the judge's gap and summary.
ATTEMPT_TEXT = ("tail", "error", "gap", "summary", "detail")
# Free text in a pending control's arguments (control.message's text).
PENDING_TEXT = ("text", "prompt")
REDACTED_EVENT = "redacted"
REDACT_BATCH = 200


def window(db, owner):
    """The owner's run retention window: `agent_retention_days` in OwnerSettings, else 90 days."""
    row = db.get(OwnerSettings, owner)
    return days_window((row.values if row else {}).get(SETTING))


def days_window(days):
    valid = isinstance(days, int) and not isinstance(days, bool) and MIN_DAYS <= days <= MAX_DAYS
    return timedelta(days=days) if valid else RUN_DATA


def expired_runs(db, clock):
    """Settled runs past their owner's window, as a selectable of run ids."""
    overrides = {row.owner_id: days_window(row.values.get(SETTING))
                 for row in db.scalars(select(OwnerSettings).where(OwnerSettings.values.has_key(SETTING)))}
    default = AgentRun.finished_at < clock - RUN_DATA
    past = [(AgentRun.owner_id.not_in(list(overrides)) & default) if overrides else default]
    past += [(AgentRun.owner_id == owner) & (AgentRun.finished_at < clock - span) for owner, span in overrides.items()]
    return select(AgentRun.id).where(AgentRun.state.in_(TERMINAL_RUN_STATES), or_(false(), *past))


def scrub_history(history):
    """Each entry keeps its actor, time, event, states and reason; its free text becomes the marker."""
    return [{**entry, **{k: MARKER for k in HISTORY_TEXT if entry.get(k)}} if isinstance(entry, dict) else entry
            for entry in history or []]


def scrub_attempt(result):
    """An attempt keeps its verdict, rules, exit code and each criterion's id and status; not the text."""
    result = dict(result or {})
    for key in ATTEMPT_TEXT:
        if result.get(key):
            result[key] = MARKER
    if isinstance(result.get("criteria"), list):
        result["criteria"] = [{**c, "evidence": MARKER} if isinstance(c, dict) and c.get("evidence") else c
                              for c in result["criteria"]]
    return result


def is_redacted(run):
    return any(isinstance(e, dict) and e.get("event") == REDACTED_EVENT for e in run.history or [])


def redact(db, run, *, actor, reason="retention", attempts=False):
    """Scrub the copies of a run's content that outlive its child records. Idempotent; returns counts.

    Kept: the run row, every history entry's actor, time, event, states and reason, each frozen plan's title,
    steps, criteria (the ids and texts the judge used), verify and completion policy. With `attempts`, the
    attempt rows stay with their verdicts and criterion statuses and lose their text (the retention sweep
    deletes them instead)."""
    from .run_coordinator import HISTORY_LIMIT

    counts = {"history": 0, "plans": 0, "attempts": 0, "notifications": 0, "pending": 0, "receipts": 0}
    before = run.history or []
    history = scrub_history(before)
    counts["history"] = sum(a != b for a, b in zip(history, before, strict=True))
    if not is_redacted(run):
        history.append({"at": now().isoformat(), "event": REDACTED_EVENT, "reason": reason, "actor": actor})
    if history != before:
        run.history = history[-HISTORY_LIMIT:]
        run.updated_at = now()
    for plan in db.scalars(select(AgentRunPlan).where(AgentRunPlan.run_id == run.id)):
        if plan.prd_text not in ("", MARKER) or plan.proposals is not None:
            plan.prd_text = MARKER if plan.prd_text else ""
            plan.proposals = None  # The agent's proposed criterion texts and reasons.
            counts["plans"] += 1
    if attempts:
        for attempt in db.scalars(select(AgentAttempt).where(AgentAttempt.run_id == run.id)):
            scrubbed = scrub_attempt(attempt.result)
            if scrubbed != (attempt.result or {}):
                attempt.result = scrubbed
                counts["attempts"] += 1
    counts["notifications"] = db.execute(
        update(Notification)
        .where(Notification.owner_id == run.owner_id, Notification.target["run_id"].astext == run.id,
               Notification.body != MARKER)
        .values(body=MARKER)
        .execution_options(synchronize_session=False)
    ).rowcount
    for row in db.scalars(select(AgentPendingControl).where(AgentPendingControl.owner_id == run.owner_id,
                                                            AgentPendingControl.run_id == run.id)):
        arguments = {k: v for k, v in (row.arguments or {}).items() if k not in PENDING_TEXT}
        if row.summary != MARKER or arguments != (row.arguments or {}):
            row.summary, row.arguments = MARKER, arguments
            counts["pending"] += 1
    counts["receipts"] = db.execute(delete(AgentActionReceipt).where(
        AgentActionReceipt.owner_id == run.owner_id, AgentActionReceipt.run_id == run.id)).rowcount
    return counts


def run_children(db, runs, *, attempts=True):
    """Delete the child records of these runs; counts by name. Turns are the run's own and the unattributed
    ones (a host frame without a run id) of a session whose run is one of these: a session a newer run
    took over keeps its turns until that run's window passes."""
    counts = {}
    sessions = select(AgentSession.id).where(AgentSession.run_id.in_(runs))
    counts["run_turns"] = db.execute(delete(AgentTurn).where(or_(
        AgentTurn.run_id.in_(runs), AgentTurn.run_id.is_(None) & AgentTurn.session_pk.in_(sessions)))).rowcount
    models = (("run_questions", AgentQuestion), ("run_attempts", AgentAttempt), ("run_interviews", AgentInterview),
              ("run_reviews", AgentPlanReview), ("run_commands", AgentCommand))
    for name, model in models:
        if attempts or model is not AgentAttempt:
            counts[name] = db.execute(delete(model).where(model.run_id.in_(runs))).rowcount
    return counts


def redact_now(db, owner, run_id, *, actor):
    """The owner deletes one settled run's content now (POST /remotea/runs/{id}/redact): its turns, questions,
    interviews and commands go as at retention, and every copy is scrubbed; attempts keep their verdicts."""
    from . import run_coordinator as coordinator

    run = coordinator.load(db, owner, run_id)
    if run.state not in TERMINAL_RUN_STATES:
        raise DomainError("RUN_ACTIVE", "Stop this run before deleting its content.", 409,
                          data={"run_id": run.id, "state": run.state})
    counts = run_children(db, select(AgentRun.id).where(AgentRun.id == run.id), attempts=False)
    counts.update(redact(db, run, actor=actor, reason="owner", attempts=True))
    emit(db, owner, "agent.changed", run.id, run.state_revision)
    return counts


def sweep(db, clock=None):
    clock = clock or now()
    counts = {}
    bound = select(AgentSession.id).where(AgentSession.run_id.is_not(None))
    counts["unbound_turns"] = db.execute(delete(AgentTurn).where(
        AgentTurn.run_id.is_(None), AgentTurn.session_pk.not_in(bound),
        AgentTurn.created_at < clock - UNBOUND_TURNS)).rowcount
    counts["unbound_questions"] = db.execute(delete(AgentQuestion).where(
        AgentQuestion.run_id.is_(None), AgentQuestion.session_pk.not_in(bound), AgentQuestion.status != "open",
        AgentQuestion.created_at < clock - UNBOUND_TURNS)).rowcount
    old = expired_runs(db, clock)
    counts.update(run_children(db, old))
    # The skeleton stays; its copies of run content are scrubbed once (bounded per pass, the next pass continues).
    counts["redacted_runs"] = 0
    pending = (select(AgentRun).where(AgentRun.id.in_(old), ~AgentRun.history.contains([{"event": REDACTED_EVENT}]))
               .order_by(AgentRun.finished_at).limit(REDACT_BATCH).with_for_update(skip_locked=True))
    for run in db.scalars(pending):
        redact(db, run, actor={"kind": "retention"})
        counts["redacted_runs"] += 1
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


EXPORTED = (AgentRun, AgentRunPlan, AgentAttempt, AgentQuestion, AgentInterview, AgentPlanReview, AgentSession, AgentSwarm)
# Not exported: run token material; agent_turns (up to 64 MiB a day per owner, built in memory here; the host
# keeps the transcript and GET /remotea/runs/{id} pages them); agent_commands, the host-facing journal (its
# input texts are the `sent` history entries; a spawn payload carried a run token). docs/AGENT_CONTROL.md
# "Retention and redaction" lists every table.
EXPORT_OMITS = {"run_token_hash", "run_token_salt", "summary_fingerprint", "token_hash", "token_salt"}


def export_tables(db, owner):
    """The agent tables in the account export (`GET /api/v1/export`), as they are now: redacted rows export
    the marker, deleted rows are gone."""
    from .domain import serial

    return {model.__tablename__: [{k: v for k, v in serial(row).items() if k not in EXPORT_OMITS}
                                  for row in db.scalars(select(model).where(model.owner_id == owner))]
            for model in EXPORTED}


def run_data_expires(run):
    """When a settled run's turns, commands, questions, attempts and interviews are (or were) deleted; None while
    the run is unsettled."""
    if run.state not in TERMINAL_RUN_STATES or run.finished_at is None:
        return None
    return run.finished_at + RUN_DATA


def run_data_expired(run, clock=None):
    expires = run_data_expires(run)
    return expires is not None and expires <= (clock or now())


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
