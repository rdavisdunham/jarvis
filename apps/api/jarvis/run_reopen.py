"""Reopen terminal: resume a run's exited worker in place (`run.reopen`, docs/AGENT_CONTROL.md).

When a run's agent exits, the run waits in needs_you(exited). Opening a new terminal by hand and running
`claude --resume <session>` gives a terminal remotea does not bind to the run: no run token, no run commands, no
verified turns. remotea already resumes a run's terminal in place after its own restart (`resumeOnBoot`: same surf,
same `eridani` binding, secrets, MCP args and verify config, the next incarnation); `run.reopen` asks it to do the same
for a surface that exited. The resumed agent waits at its prompt; a managed-execution run is then nudged on
(`continue_safely`).

Sent only to a connected host whose hello lists `run.reopen`. Body `{surf, incarnation}`: the run's surface and the
incarnation that exited. Outcomes: `executed` with `{surf, incarnation: <new>}` puts the run back in its phase (reason
`reopened`); anything else (`rejected`, `failed`, `uncertain`, an expiry, a stale reconcile) leaves it in
needs_you(exited) with a `reopen_failed` timeline note, which the run's stop card shows (agent_control.reason_detail).
"""

from sqlalchemy import select

from . import agent_link
from . import run_coordinator as coordinator
from .domain import DomainError, emit
from .models import now
from .session_models import AgentCommand, AgentHost, AgentSession

COMMAND = "run.reopen"  # Its TTL (5 minutes) is in run_coordinator.TTL.
# A reopen the host has not answered yet (an acked one may still be respawning the terminal).
PENDING = ("queued", "sent", "acked")
WHY = {
    "not_exited": "Reopen is only for a run whose agent exited.",
    "terminal_control": "The terminal has control of this session. Continue there.",
    "no_surface": "This run has no terminal to reopen.",
    "handoff": "The plan is being handed to a new worker; its terminal cannot be reopened.",
    "reviewer": "That terminal is the plan reviewer's, not the run's worker.",
    "host_offline": "The remotea server is offline.",
    "host_too_old": "Update remotea to reopen an exited agent from here.",
    "pending": "A reopen is already on its way to the host.",
}


def pending(db, run, clock=None):
    """The unanswered reopen of the run's current generation that can still reach the host, or None."""
    clock = clock or now()
    return db.scalar(select(AgentCommand).where(
        AgentCommand.run_id == run.id, AgentCommand.kind == COMMAND, AgentCommand.generation == run.generation,
        AgentCommand.status.in_(PENDING), (AgentCommand.expires_at.is_(None)) | (AgentCommand.expires_at > clock))
        .order_by(AgentCommand.created_at.desc()).limit(1))


def unavailable(db, run):
    """Why Reopen cannot be sent now (a WHY key), or None."""
    if run.state != "needs_you" or run.state_reason != "exited":
        return "not_exited"
    if (run.sync_state or {}).get("controller") == "terminal":
        return "terminal_control"
    if not run.surf or not run.session_pk:
        return "no_surface"
    handoff = (run.sync_state or {}).get("handoff") or {}
    if handoff and not handoff.get("done"):
        return "handoff"
    if run.roles and coordinator.review_for_surface(db, run, run.surf) is not None:
        return "reviewer"
    host = db.get(AgentHost, run.host_id) if run.host_id else None
    if host is None or not agent_link.connected(host):
        return "host_offline"
    if not coordinator.host_cap(db, run.host_id, COMMAND):
        return "host_too_old"
    if pending(db, run) is not None:
        return "pending"
    return None


def public(db, run):
    """The drawer's Reopen terminal button: {available, pending, why}. `why` is null when it is available, and for a
    run that is not waiting on an exited agent (the button is not shown at all then)."""
    why = unavailable(db, run)
    return {"available": why is None, "pending": why == "pending", "why": WHY[why] if why and why != "not_exited" else None}


def reopen(db, owner, run_id, body, *, actor):
    """The owner's Reopen terminal: ask the host to resume the run's exited worker in place."""
    from .agent_control import public_run

    run = coordinator.load(db, owner, run_id)
    coordinator.check_revision(run, body.expected_revision)
    why = unavailable(db, run)
    if why:
        raise DomainError("REOPEN_UNAVAILABLE", WHY[why], 409, data={"run_id": run.id, "reason": why})
    session = db.get(AgentSession, run.session_pk)
    if session is not None and session.surf == run.surf:
        run.incarnation = max(run.incarnation, session.incarnation or 0)  # The incarnation that exited.
    coordinator.command(db, run, COMMAND, {"surf": run.surf, "incarnation": run.incarnation})
    coordinator.note(run, "reopen_requested", actor=actor.record(), surf=run.surf, incarnation=run.incarnation)
    emit(db, run.owner_id, "agent.changed", run.id)
    return public_run(db, run, detail=True)


def on_outcome(db, run, row, state, detail="", result=None):
    """What the host did with a reopen (run_coordinator.on_outcome, after its timeline entry)."""
    result = result or {}
    if state != "executed":
        failed(run, state, detail)
        emit(db, run.owner_id, "agent.changed", run.id)
        return
    surf, incarnation = result.get("surf"), result.get("incarnation")
    fresh = type(incarnation) is int and (surf is None or surf == run.surf)
    if fresh and incarnation > run.incarnation:
        run.incarnation = incarnation
    if fresh:
        from . import agent_cleanup

        agent_cleanup.remember(run, run.surf, incarnation, row.generation)  # Cleanup targets the live incarnation.
    if run.state == "stopping" and run.surf:
        coordinator.stop_command(db, run)  # A stop asked while it reopened: it must reach the new incarnation.
        return
    if run.state != "needs_you" or run.state_reason != "exited" or row.generation != run.generation:
        return  # The run moved on meanwhile (a message, a stop): the reopened agent just waits at its prompt.
    session = db.get(AgentSession, run.session_pk) if run.session_pk else None
    if fresh and session is not None and session.surf == run.surf and session.state == "exited" \
            and (session.incarnation or 0) >= incarnation:
        return failed(run, state, "the reopened agent exited again")
    coordinator.transition(db, run, coordinator.phase(run), reason="reopened")
    coordinator.continue_safely(db, run, "reopened")


def failed(run, state, detail):
    coordinator.note(run, "reopen_failed", state=state,
                     detail=(detail or "the host did not say why").strip()[:300])


def last_failure(run):
    """The detail of a reopen that failed since the run last changed state, or None (the stop card's detail)."""
    for entry in reversed(run.history or []):
        if entry.get("event") == "reopen_failed":
            return entry.get("detail") or ""
        if entry.get("to") or entry.get("event") == "reopen_requested":
            return None
    return None
