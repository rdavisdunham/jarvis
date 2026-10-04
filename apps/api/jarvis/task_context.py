"""Resolve references to current records; uncertain targets remain choices, not mutations."""

from sqlalchemy import delete, select

from .domain import DomainError, owned, preferences, serial
from .models import Conversation, Task, TaskReference, now


def remember(db, owner, conversation_id, ids):
    if not conversation_id:
        return
    conversation = owned(db, Conversation, conversation_id, owner)
    if conversation.private or not preferences(db, owner)["history_enabled"]:
        return
    at = now()
    for tid in set(ids):
        task = owned(db, Task, tid, owner)
        row = db.get(TaskReference, (conversation_id, task.id))
        if row:
            row.touched_at = at
        else:
            db.add(
                TaskReference(conversation_id=conversation_id, task_id=task.id, owner_id=owner, touched_at=at)
            )
    db.flush()
    old = list(
        db.scalars(
            select(TaskReference.task_id)
            .where(TaskReference.conversation_id == conversation_id, TaskReference.owner_id == owner)
            .order_by(TaskReference.touched_at.desc())
            .offset(100)
        )
    )
    if old:
        db.execute(
            delete(TaskReference).where(
                TaskReference.conversation_id == conversation_id, TaskReference.task_id.in_(old)
            )
        )


def resolve(db, owner, context, conversation_id, scope, query="", home_id=None):
    base = select(Task).where(Task.owner_id == owner, Task.archived.is_(False))
    if scope == "selected":
        ids = context.get("selected_task_ids") or (
            [context["selected_task_id"]] if context.get("selected_task_id") else []
        )
        base = base.where(Task.id.in_(ids))
    elif scope == "visible":
        if context.get("view") not in {"today", "inbox", "week", "all", "calendar", "reminders"}:
            return {"tasks": [], "ambiguous": False, "scope": scope, "truncated": False}
        base = base.where(Task.id.in_(context.get("visible_ids", [])))
    elif scope == "recent":
        if not conversation_id:
            return {"tasks": [], "ambiguous": False, "scope": scope, "truncated": False}
        conv = owned(db, Conversation, conversation_id, owner)
        if conv.private or not preferences(db, owner)["history_enabled"]:
            return {"tasks": [], "ambiguous": False, "scope": scope, "truncated": False}
        refs = list(
            db.scalars(
                select(TaskReference)
                .where(TaskReference.owner_id == owner, TaskReference.conversation_id == conversation_id)
                .order_by(TaskReference.touched_at.desc())
                .limit(60)
            )
        )
        if not query and refs:
            refs = [r for r in refs if r.touched_at == refs[0].touched_at]
        base = base.where(Task.id.in_([r.task_id for r in refs]))
    elif scope != "search":
        raise DomainError("INVALID_ARGUMENT", "Choose selected, visible, recent, or search.")
    if query.strip():
        from sqlalchemy import Text, cast

        from .text_normalize import sql_filter

        # Every word must appear in some field; spoken numbers match their digits.
        fields = (Task.title, Task.notes, Task.project, Task.assignee, Task.work_type, cast(Task.tags, Text))
        base = base.where(sql_filter(query, *fields, limit=20))
    if home_id:
        from .task_tools import under_home

        base = base.where(under_home(owner, home_id))
    rows = list(db.scalars(base.order_by(Task.updated_at.desc(), Task.id).limit(61)))
    from .task_tools import compact, with_homes

    # Compact rows keep a 60-task ambiguity check inside model context; task_get reads full notes.
    tasks = [{**compact(serial(t)), "notes_preview": t.notes[:200], "notes_truncated": len(t.notes) > 200} for t in rows[:60]]
    return {
        "tasks": with_homes(db, owner, tasks),
        "ambiguous": len(rows) > 1,
        "scope": scope,
        "truncated": len(rows) > 60,
    }
