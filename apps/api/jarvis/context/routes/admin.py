"""Owner processing controls for meeting context: status, pause, resume and the rollback drill.

Browser (session cookie) routes under /api/v1/context/admin, included in `routes.owner_router` (mounted in
`api.py`), not on the bot router: these are the owner's rollback controls, never an agent's. Personal workspace
only (PERSONAL_WORKSPACE). They work whatever `context_enabled` says, so processing can be paused before the flag
is switched on or after it is switched off.

Pause takes the owner-state row lock, increments `epoch` and sets `paused`. Every automatic run started under an
older epoch fails the pipeline's apply check, and nothing automatic applies while `paused` is set. Sources, reads,
manual corrections and approved publications do not look at `paused` and continue. Pausing a paused owner changes
nothing. Resume clears `paused` and keeps the epoch, so runs rejected by the pause stay rejected; the pipeline
requeues what is still relevant.

The rollback drill pauses, exports the owner's history in the canonical format (`export.export_history`) and checks
that the document imports: it is loaded under a scratch owner inside a savepoint that is always rolled back. A
failed check is reported (`verified: false`, `problem`) and the pause stays.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ... import bot_access
from ...auth import Identity, authenticate
from ...db import session_scope
from ...domain import DomainError
from ...models import Job, now, uid
from .. import export
from ..models import ContextOwnerState, ContextRun
from ..settings import JOB_PREFIX

RECENT = 10
ERROR_CHARS = 500

User = Annotated[Identity, Depends(authenticate)]
router = APIRouter(prefix="/admin", tags=["Meeting context"])


# ---------------------------------------------------------------- state


def lock_state(db: Session, owner: str) -> ContextOwnerState:
    """The owner's state row under SELECT ... FOR UPDATE, created when absent."""
    db.execute(insert(ContextOwnerState).values(owner_id=owner).on_conflict_do_nothing(index_elements=["owner_id"]))
    return db.scalar(select(ContextOwnerState).where(ContextOwnerState.owner_id == owner).with_for_update()
                     .execution_options(populate_existing=True))


def pause(db: Session, owner: str) -> ContextOwnerState:
    """Stop automatic processing: epoch + 1 and paused, under the row lock. Idempotent while paused."""
    state = lock_state(db, owner)
    if not state.paused:
        state.epoch += 1
        state.paused, state.paused_at = True, now()
        state.updated_at = now()
    db.flush()
    return state


def resume(db: Session, owner: str) -> ContextOwnerState:
    """Allow automatic processing again. The epoch stays, so results from before the pause stay rejected."""
    state = lock_state(db, owner)
    if state.paused:
        state.paused, state.paused_at = False, None
        state.updated_at = now()
    db.flush()
    return state


def _iso(value):
    return value.isoformat() if value else None


def _run(row: ContextRun, epoch: int) -> dict:
    return {"id": row.id, "stage": row.stage, "status": row.status, "epoch": row.epoch,
            "current_epoch": row.epoch == epoch, "source_id": row.source_id, "revision_id": row.revision_id,
            "attempts": row.attempts, "error": (row.error or "")[:ERROR_CHARS] or None,
            "created_at": _iso(row.created_at), "started_at": _iso(row.started_at),
            "finished_at": _iso(row.finished_at)}


def _job(row: Job) -> dict:
    error = (row.result or {}).get("error") if isinstance(row.result, dict) else None
    return {"id": row.id, "kind": row.kind, "status": row.status, "error": str(error)[:ERROR_CHARS] if error else None,
            "created_at": _iso(row.created_at), "finished_at": _iso(row.finished_at)}


def status(db: Session, owner: str) -> dict:
    """Epoch, pause state, job and run counts by status, the latest runs and the latest failures."""
    state = db.get(ContextOwnerState, owner)
    epoch = state.epoch if state else 0
    jobs = (Job.owner_id == owner, Job.kind.startswith(JOB_PREFIX, autoescape=True))
    runs = db.scalars(select(ContextRun).where(ContextRun.owner_id == owner)
                      .order_by(ContextRun.created_at.desc(), ContextRun.id).limit(RECENT))
    failed_runs = db.scalars(select(ContextRun).where(ContextRun.owner_id == owner, ContextRun.status == "failed")
                             .order_by(ContextRun.created_at.desc(), ContextRun.id).limit(RECENT))
    failed_jobs = db.scalars(select(Job).where(*jobs, Job.status == "failed")
                             .order_by(Job.created_at.desc(), Job.id).limit(RECENT))
    return {
        "owner_id": owner, "epoch": epoch, "paused": bool(state and state.paused),
        "paused_at": _iso(state.paused_at) if state else None,
        "queue": {
            "jobs": dict(db.execute(select(Job.status, func.count()).where(*jobs).group_by(Job.status)).all()),
            "runs": dict(db.execute(select(ContextRun.status, func.count()).where(ContextRun.owner_id == owner)
                                    .group_by(ContextRun.status)).all()),
        },
        "last_runs": [_run(r, epoch) for r in runs],
        "failures": {"runs": [_run(r, epoch) for r in failed_runs], "jobs": [_job(j) for j in failed_jobs]},
    }


def rollback_drill(db: Session, owner: str) -> dict[str, Any]:
    """Pause, export the history and prove the export imports (scratch owner, savepoint rolled back)."""
    pause(db, owner)
    history = export.export_history(db, owner)
    expected = {name: len(history[name]) for name in export.SECTION_NAMES}
    savepoint = db.begin_nested()
    try:
        report = export.import_history(db, "drill:" + uid(), history, preserve_ids=False)
        imported, problem = {name: report.counts[name] for name in export.SECTION_NAMES}, None
    except (export.HistoryImportError, SQLAlchemyError) as exc:  # the pause stays; the drill reports
        imported, problem = {}, str(exc)
    finally:
        savepoint.rollback()
    return {"status": status(db, owner), "counts": expected, "verified": imported == expected, "problem": problem,
            "history": history}


# ---------------------------------------------------------------- routes


def _personal(db: Session, user: Identity) -> str:
    if bot_access.shared(db, user.owner_id):
        raise DomainError("PERSONAL_WORKSPACE", "Meeting context works only in your personal workspace.", 403)
    return user.owner_id


@router.get("/status")
def get_status(user: User):
    with session_scope() as db:
        return status(db, _personal(db, user))


@router.post("/pause")
def post_pause(user: User):
    with session_scope() as db:
        owner = _personal(db, user)
        pause(db, owner)
        return status(db, owner)


@router.post("/resume")
def post_resume(user: User):
    with session_scope() as db:
        owner = _personal(db, user)
        resume(db, owner)
        return status(db, owner)


@router.post("/rollback-drill")
def post_rollback_drill(user: User):
    with session_scope() as db:
        return rollback_drill(db, _personal(db, user))
