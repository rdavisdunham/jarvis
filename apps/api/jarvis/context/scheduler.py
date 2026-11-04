"""The context platform's housekeeping tick. `worker.housekeeping` calls `tick` (about once a minute) when
`context_enabled`.

One tick: `pipeline.schedule_due` (Granola polls and backfill, continuations); then, per owner with context
sources, each in its own savepoint so one owner's failure never blocks another: the weekly lint job when
`lint.due` says this week's pass has not run (one `context_lint` job per weekly slot, none while one is pending),
and `notices.sync` (meeting-ready, consolidated backfill and the 17:00 digest).
"""

import logging
from datetime import datetime

from sqlalchemy import select

from ..domain import enqueue_job
from ..models import Job, now
from . import notices, pipeline
from . import settings as context_settings
from .models import ContextSource

try:  # E2's weekly lint; the tick still runs without it.
    from . import lint
except ImportError:  # pragma: no cover
    lint = None

log = logging.getLogger("jarvis.context.scheduler")


def queue_lint(db, owner: str, at: datetime) -> bool:
    """Enqueue this week's lint pass once per slot."""
    if lint is None or not lint.due(db, owner, at):
        return False
    slot = lint.slot(db, owner, at)
    if db.scalar(select(Job.id).where(
            Job.owner_id == owner, Job.kind == "context_lint",
            Job.status.in_(notices.PENDING_JOBS) | (Job.payload["slot"].astext == slot)).limit(1)):
        return False
    enqueue_job(db, owner, "context_lint", {"slot": slot})
    return True


def tick(db, at: datetime | None = None) -> dict:
    at = at or now()
    out = {"pipeline": {}, "lint": 0, "ready": 0, "backfill": 0, "digest": 0, "failed": []}
    if not context_settings.enabled():
        return out
    out["pipeline"] = pipeline.schedule_due(db, at)
    for owner in sorted(set(db.scalars(select(ContextSource.owner_id).distinct()))):
        try:
            with db.begin_nested():
                out["lint"] += queue_lint(db, owner, at)
                for name, count in notices.sync(db, owner, at).items():
                    out[name] += count
        except Exception:
            log.exception("Context tick failed for an owner")
            out["failed"].append(owner)
    return out
