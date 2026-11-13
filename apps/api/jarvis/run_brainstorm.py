"""Brainstorm: a run that opens as a conversation about the task, and plans only when the owner says so
(docs/END_TO_END_UPGRADE_PLAN.md, phase 1b; docs/AGENT_CONTROL.md "Brainstorm").

A yes/no choice when the run starts (`StartInput.brainstorm`). On, the agent reads the task and the code, then
talks: one question per turn, what before how, approaches when more than one is plausible, and a playback of what
will be built. It works with any harness, because the conversation is ordinary turns and messages:

- each agent turn that ends without a plan stops the run at `needs_you(brainstorm)` ("your reply"), instead of
  `no_plan`; the owner's message resumes planning (`agent_control.send_message`);
- "Plan it" (`plan_it`) closes the conversation and asks for the plan; so does a `submit_plan` the agent sends
  because the owner asked for the plan in the conversation;
- the plan then takes the usual approval. It never takes the direction check: the conversation was the direction.

State: `run.sync_state["brainstorm"] = {open, started_at, turns, ended_at, ended_by}`.
"""

from .models import now

REASON = "brainstorm"


def state(run) -> dict:
    return dict((run.sync_state or {}).get("brainstorm") or {})


def had(run) -> bool:
    """Did this run start as a brainstorm (open or over)?"""
    return bool(state(run))


def is_open(run) -> bool:
    return bool(state(run).get("open"))


def begin(run):
    run.sync_state = {**(run.sync_state or {}), "brainstorm": {"open": True, "started_at": now().isoformat(), "turns": 0}}


def close(run, by):
    if is_open(run):
        run.sync_state = {**(run.sync_state or {}), "brainstorm": {
            **state(run), "open": False, "ended_at": now().isoformat(), "ended_by": by}}


def turn_ended(db, run) -> bool:
    """settle_planning: a planning turn ended without a plan. While the conversation is open that is the agent
    waiting for the owner's reply, not a failure. Returns whether it was handled here."""
    from . import run_coordinator as coordinator

    if not is_open(run):
        return False
    value = state(run)
    run.sync_state = {**(run.sync_state or {}), "brainstorm": {**value, "turns": int(value.get("turns") or 0) + 1}}
    if run.state != "needs_you":
        coordinator.transition(db, run, "needs_you", reason=REASON)
    return True


def plan_it(db, run, *, actor):
    """The owner has heard enough: close the conversation and ask the agent for the plan."""
    from . import agent_prompts as prompts
    from . import run_coordinator as coordinator
    from .domain import DomainError

    coordinator.ensure_eridani_control(run)
    if not is_open(run):
        raise DomainError("INVALID_STATE", "This run has no open conversation to plan from.", 409)
    if run.state not in ("needs_you", "planning", "interviewing"):
        raise DomainError("INVALID_STATE", f"A run that is {run.state} cannot start planning.", 409)
    close(run, "owner")
    coordinator.note(run, "brainstorm_done", actor=actor.record(), turns=state(run).get("turns"))
    if run.state != "planning":
        coordinator.transition(db, run, "planning", actor=actor, reason="brainstorm_done")
    from .agent_plan import verify_ids

    coordinator.say(db, run, prompts.brainstorm_plan(verify_ids(db, run)[1]), "brainstorm_plan", actor=actor)


def public(run):
    value = state(run)
    if not value:
        return None
    return {"open": bool(value.get("open")), "turns": int(value.get("turns") or 0),
            "started_at": value.get("started_at"), "ended_at": value.get("ended_at"), "ended_by": value.get("ended_by")}
