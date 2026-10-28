"""Who is changing a task, for the run completion gate (docs/AGENT_CONTROL.md, "Actor model").

An actor is bound per request or job. Explicit bindings (the run coordinator, in-session tools,
Linear import) win; otherwise the request context decides: queued Eri work is `eri_work`, a bot
key without a run binding is `bot`, and a signed-in browser is `human`.

While a task has an unsettled run, only a human, source sync or the run judge (the coordinator)
may complete it or edit its gate fields. A human or source completion records the override and
moves the run to `stopping(task_completed_externally)`; it never asserts that the run stopped.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from .domain import DomainError

KINDS = ("human", "eri_work", "run_session", "run_judge", "source_sync", "bot")
GATE_FIELDS = ("success_criteria", "verify", "turn_cap")
# These may complete a run-owned task or edit its gate fields.
AUTHORITIES = ("human", "source_sync", "run_judge")


@dataclass(frozen=True)
class Actor:
    kind: str
    credential_id: str | None = None
    run_id: str | None = None
    generation: int | None = None
    verified: bool = True
    # Why a human completion happened (approve) so the run settles as completed, not stopped.
    reason: str = ""

    def record(self):
        return {k: v for k, v in (("kind", self.kind), ("credential_id", self.credential_id), ("run_id", self.run_id),
                                  ("generation", self.generation), ("verified", self.verified),
                                  ("reason", self.reason)) if v not in (None, "")}


bound = ContextVar("eridani_actor", default=None)


@contextmanager
def acting(actor):
    token = bound.set(actor)
    try:
        yield actor
    finally:
        bound.reset(token)


def current():
    explicit = bound.get()
    if explicit is not None:
        return explicit
    from .access import execution
    from .bot_access import principal

    grant = execution.get()
    if grant:
        return Actor("eri_work", credential_id=grant.get("credential_id"))
    if principal.get():
        return Actor("bot", credential_id=principal.get(), verified=False)
    return Actor("human")


def active_run(db, task_id, lock=False):
    from sqlalchemy import select

    from .session_models import TERMINAL_RUN_STATES, AgentRun

    query = select(AgentRun).where(AgentRun.task_id == task_id, AgentRun.state.not_in(TERMINAL_RUN_STATES))
    return db.scalar(query.with_for_update() if lock else query)


def completing(changes):
    return changes.get("status") == "completed"


def gate(db, owner, task, changes):
    """Called from task.update (every entrance) before the change is applied."""
    touches = [k for k in GATE_FIELDS if k in changes and changes[k] != getattr(task, k)]
    finishing = completing(changes) and task.status != "completed"
    if not (touches or finishing):
        return
    run = active_run(db, task.id, lock=True)
    if run is None:
        return
    actor = current()
    if actor.kind not in AUTHORITIES:
        field = "complete this task" if finishing else "change " + ", ".join(touches)
        raise DomainError("RUN_OWNS_COMPLETION", f"An agent run owns this task; only you can {field} while it runs.", 409,
                          data={"run_id": run.id, "state": run.state})
    if finishing and actor.kind in ("human", "source_sync"):
        from .run_coordinator import external_completion

        external_completion(db, run, actor)
