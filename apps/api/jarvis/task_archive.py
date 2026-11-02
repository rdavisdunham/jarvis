"""File terminal tasks after 24 elapsed hours, independently of daily view boundaries.

The worker checks each maintenance cycle. Scans process at most BATCH_SIZE records per
owner and rotate a keyset cursor so blocked parents cannot starve later records.
"""
import logging
from datetime import timedelta
from uuid import uuid4

from sqlalchemy import inspect, or_, select

from .models import Task, now

FINISHED = {"completed", "cancelled"}
GRACE = timedelta(hours=24)
BATCH_SIZE = 200
_cursors = {}
logger = logging.getLogger(__name__)


def apply_lifecycle(task, changes, *, automatic=False, external=False):
    """Call before assigning a patch, so unchanged statuses never slide the clock."""
    instant = now()
    status = changes.get("status", task.status)
    if status != task.status:
        if status in FINISHED:
            task.finished_at = instant
            task.archive_eligible_at = instant + GRACE
        else:
            task.finished_at = task.archive_eligible_at = None
            task.completed_at = None
            # Linear's explicit manual override survives remote reopen; local Reopen clears it.
            if not external or not (task.external or {}).get("local_archived"):
                changes["archived"] = False
    archived = changes.get("archived", task.archived)
    if archived and not task.archived:
        task.archived_at = instant
        task.archive_reason = "automatic" if automatic else "external" if external and not (task.external or {}).get("local_archived") else "manual"
    elif not archived and task.archived:
        task.archived_at = task.archive_reason = None
        if status in FINISHED:
            task.archive_eligible_at = instant + GRACE
    elif not archived and status not in FINISHED:
        task.archived_at = task.archive_reason = None


def observe_lifecycle(task):
    """ORM backstop for imports and direct writers, without fabricating old cancellation dates."""
    state = inspect(task)
    status = state.attrs.status.history
    changed = status.has_changes() and (state.pending or bool(status.deleted))
    if task.status in FINISHED:
        if changed and not state.attrs.finished_at.history.has_changes():
            task.finished_at = now()
            task.archive_eligible_at = task.finished_at + GRACE
        elif task.finished_at and task.archive_eligible_at is None:
            task.archive_eligible_at = task.finished_at + GRACE
        if task.status == "cancelled":
            task.completed_at = None
        elif changed and task.completed_at is None:
            task.completed_at = task.finished_at
    else:
        task.completed_at = None
        task.finished_at = task.archive_eligible_at = None
        if changed and not state.attrs.archived.history.has_changes() and not (task.external or {}).get("local_archived"):
            task.archived = False
    archived = state.attrs.archived.history
    if task.archived:
        if task.archived_at is None:
            task.archived_at = now()
        if task.archive_reason is None:
            task.archive_reason = "manual"
    else:
        task.archived_at = task.archive_reason = None
        if archived.deleted and archived.deleted[0] and task.status in FINISHED and not state.attrs.archive_eligible_at.history.has_changes():
            task.archive_eligible_at = now() + GRACE


def organizational_container(db, task):
    """Type semantics, not incidental children, protect project/custom containers."""
    from .structure import ensure, opens_as, record_type
    from .structure_models import StructureRecord
    row = db.scalar(select(StructureRecord).where(StructureRecord.owner_id == task.owner_id, StructureRecord.task_id == task.id))
    return bool(row and opens_as(record_type(ensure(db, task.owner_id), row.type_id, archived=True)) == "container")


def unfinished_descendants(db, task):
    descendants = select(Task.id).where(Task.owner_id == task.owner_id, Task.parent_task_id == task.id).cte(recursive=True)
    descendants = descendants.union(select(Task.id).join(descendants, Task.parent_task_id == descendants.c.id).where(Task.owner_id == task.owner_id))
    if db.scalar(select(Task.id).where(Task.id.in_(select(descendants.c.id)), Task.status.not_in(FINISHED)).limit(1)):
        return True
    from .structure_models import StructureRecord
    root = db.scalar(select(StructureRecord.id).where(StructureRecord.owner_id == task.owner_id, StructureRecord.task_id == task.id))
    if root is None:
        return False
    records = select(StructureRecord.id, StructureRecord.task_id).where(StructureRecord.owner_id == task.owner_id, StructureRecord.parent_id == root).cte(recursive=True)
    records = records.union(select(StructureRecord.id, StructureRecord.task_id).join(records, StructureRecord.parent_id == records.c.id).where(StructureRecord.owner_id == task.owner_id))
    return bool(db.scalar(select(Task.id).where(Task.owner_id == task.owner_id, Task.id.in_(select(records.c.task_id)), Task.status.not_in(FINISHED)).limit(1)))


def unsettled_execution(db, task):
    from .session_models import AgentInterview, AgentQuestion, AgentRun, TERMINAL_RUN_STATES
    runs = list(db.scalars(select(AgentRun).where(AgentRun.owner_id == task.owner_id, AgentRun.task_id == task.id).with_for_update()))
    if any(run.state not in TERMINAL_RUN_STATES or run.plan_pending for run in runs):
        return True
    ids = [run.id for run in runs]
    if not ids:
        return False
    return bool(db.scalar(select(AgentQuestion.id).where(AgentQuestion.owner_id == task.owner_id, AgentQuestion.run_id.in_(ids), AgentQuestion.status.in_(("open", "submitted", "timed_out")), AgentQuestion.resolved_at.is_(None)).limit(1)) or db.scalar(select(AgentInterview.id).where(AgentInterview.run_id.in_(ids), AgentInterview.status == "open").limit(1)))


def historical_finish(db, task):
    """Recover only an evidenced transition, never an unrelated updated_at value."""
    from .models import ActionChange
    from .work_crypto import unseal

    changes = db.scalars(select(ActionChange).where(
        ActionChange.owner_id == task.owner_id, ActionChange.entity_kind == "task",
        ActionChange.entity_id == task.id,
    ).order_by(ActionChange.created_at.desc()).limit(50))
    for change in changes:
        try:
            before = unseal(change.before_ciphertext) if change.before_ciphertext else None
            after = unseal(change.after_ciphertext) if change.after_ciphertext else None
        except Exception:  # Missing/rotated recovery key: start grace without inventing a date.
            return None
        if not after or after.get("status") != task.status:
            return None
        if before and before.get("status") != task.status:
            return change.created_at
    return None


def sweep_owner(db, owner, *, clock=None, limit=BATCH_SIZE):
    from .domain import TaskUpdate, advisory, emit, mutate, preferences
    from .structure import observe_core
    clock = clock or now()
    if not preferences(db, owner)["auto_archive_finished_tasks"]:
        return 0
    # Same lock order as commands: workspace, task, run. Parent moves/reopen win serially.
    advisory(db, f"workspace:{owner}")
    query = select(Task).where(Task.owner_id == owner, Task.archived.is_(False), Task.is_template.is_(False), Task.status.in_(FINISHED), or_(Task.archive_eligible_at.is_(None), Task.archive_eligible_at <= clock))
    cursor = _cursors.get(owner)
    if cursor:
        query = query.where(Task.id > cursor)
    rows = list(db.scalars(query.order_by(Task.id).limit(limit).with_for_update(skip_locked=True)))
    _cursors[owner] = rows[-1].id if len(rows) == limit else None
    count = 0
    for task in rows:
        identity = task.id
        try:
            # A malformed record/integration must not roll back valid earlier candidates.
            # Savepoints include clocks, alerts, events and backing structure mutations.
            with db.begin_nested():
                if task.archive_eligible_at is None:
                    # No trustworthy cancellation history: grace begins when policy first observes it.
                    reliable = task.finished_at or (task.completed_at if task.status == "completed" else None) or historical_finish(db, task)
                    if reliable:
                        task.finished_at = reliable
                    task.archive_eligible_at = (reliable or clock) + GRACE
                    task.revision += 1
                    task.updated_at = clock
                    emit(db, owner, "task.changed", task.id, task.revision)
                    if task.archive_eligible_at > clock:
                        continue
                if organizational_container(db, task) or unfinished_descendants(db, task) or unsettled_execution(db, task):
                    continue
                db.info["automatic_task_archive"] = True
                try:
                    # Trusted worker entrypoint uses the same domain/structure mutation as commands;
                    # it has no interactive principal and never fabricates an account or bypasses API gates.
                    command_id = "autoarchive:" + str(uuid4())
                    args = TaskUpdate(task_id=task.id, expected_revision=task.revision, archived=True)
                    result = mutate(db, owner, "task.update", args, command_id)
                    observe_core(db, owner, "task.update", result, command_id)
                finally:
                    db.info.pop("automatic_task_archive", None)
            count += 1
        except Exception:
            logger.exception("Task archive candidate failed owner=%s task=%s; will retry", owner, identity)
    return count


def sweep(db):
    owners = list(db.scalars(select(Task.owner_id).where(Task.archived.is_(False), Task.is_template.is_(False), Task.status.in_(FINISHED)).distinct().order_by(Task.owner_id)))
    count = 0
    for owner in owners:
        try:
            # A policy/schema problem in one namespace must not undo another owner's filing.
            with db.begin_nested():
                archived = sweep_owner(db, owner)
            count += archived
        except Exception:
            logger.exception("Task archive owner failed owner=%s; will retry", owner)
    return count
