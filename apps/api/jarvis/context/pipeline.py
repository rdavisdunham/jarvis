"""Context job entry point. worker.perform_job sends every `context_*` job kind here (queue jarvis-context)."""

from sqlalchemy import select

from ..db import session_scope
from ..models import Job, now

DONE = {"succeeded", "cancelled", "expired"}


def _lint(job_id):
    from . import lint

    with session_scope() as db:
        job = db.get(Job, job_id, with_for_update=True)
        if not job or job.status in DONE:
            return
        job.status = "running"
        job.payload = {**job.payload, "attempts": job.payload.get("attempts", 0) + 1}
        owner, budget = job.owner_id, job.payload.get("budget_usd")
    try:
        result = lint.perform(owner, budget_usd=budget)
    except Exception as error:
        with session_scope() as db:
            job = db.get(Job, job_id)
            job.status = "failed" if job.payload.get("attempts", 0) >= 5 else "retrying"
            job.result = {"error": type(error).__name__}
        raise
    with session_scope() as db:
        job = db.get(Job, job_id)
        job.status, job.finished_at, job.result = "succeeded", now(), result


HANDLERS = {"context_lint": _lint}


def perform(job_id):
    with session_scope() as db:
        kind = db.scalar(select(Job.kind).where(Job.id == job_id))
    handler = HANDLERS.get(kind)
    if handler is None:
        raise NotImplementedError("context pipeline lands in B6")
    return handler(job_id)
