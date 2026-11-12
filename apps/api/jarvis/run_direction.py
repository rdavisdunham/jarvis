"""Direction check: on the `direction` route the owner answers the agent's brief once, then the plan starts
(docs/END_TO_END_UPGRADE_PLAN.md, phase 1; docs/AGENT_CONTROL.md "Direction check").

The agent sends a brief with its plan (`submit_plan.brief`: how it reads the task, its approach, what it will leave
alone, up to three questions). The run stops at the usual `needs_you(plan_approval)`, with the brief leading the
card. The owner either accepts (the plan starts) or answers once in their own words (`revise_plan`): the planner
revises, and the revised plan starts without a second approval (`approves`), unless it declares a migration or
proposes changing something a human set. Everything is kept in `run.sync_state["direction"]`:
`{brief, plan_revision, answer, answered_at, accepted_at}`; the judge reads the brief and the answer as context.

Runs with a reviewer (hyperplan) and freestyle runs keep their own flow. A run whose task is on another route, or
has none, behaves as before.
"""

from pydantic import BaseModel, ConfigDict, Field

from .models import Task, now

LINE = 400


class Brief(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    # One or two sentences each: how the agent reads the task, how it will do it, what it will not touch.
    reading: str = Field(min_length=1, max_length=LINE)
    approach: str = Field(min_length=1, max_length=LINE)
    leaves_alone: str = Field(default="", max_length=LINE)
    # Open questions whose answer would change the approach.
    questions: list[str] = Field(default_factory=list, max_length=3)


def state(run) -> dict:
    return dict((run.sync_state or {}).get("direction") or {})


def on_route(db, run) -> bool:
    """Does this run take the direction check? Its task is on the `direction` route and the run has no reviewer."""
    if run.freestyle or (run.roles or {}).get("reviewer"):
        return False
    task = db.get(Task, run.task_id)
    return task is not None and task.run_route == "direction"


def remember(run, brief: Brief | None, revision: int):
    """submit_plan: keep the latest brief beside the plan revision it came with."""
    if brief is None:
        return
    lines = [q.strip()[:LINE] for q in brief.questions if q.strip()]
    run.sync_state = {**(run.sync_state or {}), "direction": {
        **state(run), "brief": {**brief.model_dump(), "questions": lines}, "plan_revision": revision}}


def answer(db, run, text):
    """The owner sent the plan back with their direction. One answer is what the route asks for."""
    if on_route(db, run):
        run.sync_state = {**(run.sync_state or {}), "direction": {
            **state(run), "answer": (text or "")[:4000], "answered_at": now().isoformat()}}


def accept(db, run):
    if on_route(db, run):
        run.sync_state = {**(run.sync_state or {}), "direction": {**state(run), "accepted_at": now().isoformat()}}


def approves(db, run, plan) -> bool:
    """May the plan start without another stop? Only after the owner has answered once on this route, and never
    when the plan declares a migration."""
    if plan is None or plan.migration or (run.sync_state or {}).get("plan_declares_migration"):
        return False
    return on_route(db, run) and bool(state(run).get("answered_at"))


def public(db, run):
    """What the run page shows: the brief, the owner's answer, and whether the check is still open."""
    value = state(run)
    if not value and not on_route(db, run):
        return None
    return {"brief": value.get("brief"), "answer": value.get("answer"), "answered_at": value.get("answered_at"),
            "accepted_at": value.get("accepted_at"),
            "open": on_route(db, run) and not value.get("answered_at") and not value.get("accepted_at")}


def evidence(run):
    """For the judge: what the agent said it would do and what the owner told it."""
    value = state(run)
    if not value.get("brief") and not value.get("answer"):
        return None
    return {"brief": value.get("brief"), "owner_answer": value.get("answer"), "accepted": bool(value.get("accepted_at"))}
