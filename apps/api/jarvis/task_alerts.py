"""Tasks own completion; schedules remain the durable alert/recurrence engine."""

from sqlalchemy import select

from .models import Notification, Occurrence, Schedule, Task, now


def new_task(db, owner, title, project_id=None, *, template=False):
    from .domain import emit
    from .models import Project

    row = Task(
        owner_id=owner,
        title=title,
        project_id=project_id,
        project=db.get(Project, project_id).name if project_id else None,
        is_template=template,
    )
    db.add(row)
    db.flush()
    emit(db, owner, "task.changed", row.id, row.revision)
    return row


def finish_task(db, task, status="completed", *, sync_external=True):
    from .domain import emit

    if sync_external:
        from .linear_sync import before_task_update

        before_task_update(db, task.owner_id, task, {"status": status})
    task.status = status
    task.completed_at = (task.completed_at or now()) if status == "completed" else None
    task.revision += 1
    task.updated_at = now()
    for alert in db.scalars(
        select(Schedule).where(Schedule.task_id == task.id, Schedule.status.in_(["active", "finished"]))
    ):
        alert.status = "completed" if status == "completed" else "cancelled"
        alert.completed_at = task.completed_at
        alert.next_run_at = None
        alert.revision += 1
        emit(db, task.owner_id, "schedule.changed", alert.id, alert.revision)
    for notice in db.scalars(
        select(Notification).where(Notification.task_id == task.id, Notification.completed_at.is_(None))
    ):
        notice.completed_at = now()
        notice.read_at = notice.read_at or now()
        if notice.occurrence_id:
            occurrence = db.get(Occurrence, notice.occurrence_id)
            if occurrence:
                occurrence.status = "completed"
        emit(db, task.owner_id, "notification.changed", notice.id)
    emit(db, task.owner_id, "task.changed", task.id, task.revision)
