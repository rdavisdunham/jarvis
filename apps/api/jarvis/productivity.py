"""Owned productivity graph. Goal outcomes never derive from task completion."""

from sqlalchemy import delete, select

from .models import (
    Actor,
    Area,
    Goal,
    GoalProjectLink,
    Note,
    NoteGoalLink,
    NoteNoteLink,
    NoteProjectLink,
    NoteTaskLink,
    Project,
    Space,
    Task,
    now,
)


def defaults(db, owner):
    from .domain import advisory

    advisory(db, f"workspace:{owner}")
    # Seed once; renaming/archiving the defaults must not recreate them.
    from .models import OwnerSettings

    settings = db.get(OwnerSettings, owner)
    if settings and settings.values.get("productivity_initialized"):
        return
    for name in ("Personal", "Business"):
        if not db.scalar(select(Space).where(Space.owner_id == owner, Space.name == name)):
            db.add(Space(owner_id=owner, name=name))
    for name, kind in (("owner", "person"), ("Eri", "agent")):
        if not db.scalar(select(Actor).where(Actor.owner_id == owner, Actor.name == name)):
            db.add(Actor(owner_id=owner, name=name, kind=kind))
    if settings is None:
        settings = OwnerSettings(owner_id=owner, values={})
        db.add(settings)
    settings.values = {**settings.values, "productivity_initialized": True}
    db.flush()


def home_changes(db, owner, changes, row=None):
    from .domain import DomainError, owned

    space_id = changes.get("space_id", getattr(row, "space_id", None))
    area_id = changes.get("area_id", getattr(row, "area_id", None))
    if area_id:
        area = owned(db, Area, area_id, owner)
        if area.archived and area_id != getattr(row, "area_id", None):
            raise DomainError("INVALID_ARGUMENT", "Choose an active area.")
        if "space_id" in changes and space_id != area.space_id:
            raise DomainError("INVALID_ARGUMENT", "This area belongs to a different space.")
        space_id = area.space_id
    if space_id:
        space = owned(db, Space, space_id, owner)
        if space.archived and space_id != getattr(row, "space_id", None):
            raise DomainError("INVALID_ARGUMENT", "Choose an active space.")
    changes.update(space_id=space_id, area_id=area_id)
    return changes


def task_home(db, owner, changes, row=None):
    from .domain import DomainError, owned

    project_id = changes.get("project_id", getattr(row, "project_id", None))
    if project_id:
        p = owned(db, Project, project_id, owner)
        # Project owns the organizational home. Explicit conflicting overrides
        # are rejected, while untouched homes follow a project reassignment.
        for key in ("space_id", "area_id"):
            if key in changes and changes[key] not in (None, getattr(p, key)):
                raise DomainError("INVALID_ARGUMENT", "A project task uses its project's space and area.")
            changes[key] = getattr(p, key)
    else:
        home_changes(db, owner, changes, row)
    if changes.get("assignee_id"):
        actor = owned(db, Actor, changes["assignee_id"], owner)
        if actor.archived and actor.id != getattr(row, "assignee_id", None):
            raise DomainError("INVALID_ARGUMENT", "Choose an active assignee.")
        if "assignee" in changes and changes["assignee"] != actor.name:
            raise DomainError("INVALID_ARGUMENT", "Assignee name and ID disagree.")
        changes["assignee"] = actor.name
    elif "assignee" in changes or row is None:
        name = changes.get("assignee", "owner")
        actor = db.scalar(select(Actor).where(Actor.owner_id == owner, Actor.name == name))
        if not actor:
            actor = Actor(
                owner_id=owner,
                name=name,
                kind="agent" if name.casefold() in {"eri", "agent", "eridani"} else "person",
            )
            db.add(actor)
            db.flush()
        if actor.archived and actor.id != getattr(row, "assignee_id", None):
            raise DomainError("INVALID_ARGUMENT", "Choose an active assignee.")
        changes["assignee_id"] = actor.id
    return changes


def changed(db, owner, row):
    from .domain import emit

    row.revision += 1
    if hasattr(row, "updated_at"):
        row.updated_at = now()
    emit(db, owner, row.__tablename__.rstrip("s") + ".changed", row.id, row.revision)


def replace_goals(db, owner, row, ids):
    """Update one side of the many-to-many association and revision both sides."""
    from .domain import DomainError, owned

    is_goal = isinstance(row, Goal)
    own_key, peer_key = ("goal_id", "project_id") if is_goal else ("project_id", "goal_id")
    model = Project if is_goal else Goal
    own_col, peer_col = getattr(GoalProjectLink, own_key), getattr(GoalProjectLink, peer_key)
    previous = set(db.scalars(select(peer_col).where(own_col == row.id)))
    wanted = set(ids)
    peers = {pid: owned(db, model, pid, owner) for pid in wanted | previous}
    for pid in wanted - previous:
        if peers[pid].archived:
            raise DomainError("INVALID_ARGUMENT", "Restore archived records before adding new links.")
    db.execute(delete(GoalProjectLink).where(own_col == row.id))
    for pid in sorted(wanted):
        db.add(GoalProjectLink(**{own_key: row.id, peer_key: pid}))
    for pid in wanted ^ previous:
        changed(db, owner, peers[pid])
    db.flush()
    peer_links = "goal_ids" if is_goal else "project_ids"
    return {
        "added_ids": sorted(wanted - previous),
        "removed_ids": sorted(previous - wanted),
        "peer_kind": "project" if is_goal else "goal",
        "peers": [
            {"id": pid, "revision": peers[pid].revision, peer_links: data(db, peers[pid])[peer_links]}
            for pid in sorted(wanted ^ previous)
        ],
        "message": "Both sides of these links are already updated. Use these peer revisions for later edits.",
    }


def data(db, row):
    from .domain import serial

    result = serial(row)
    if isinstance(row, Goal):
        result["project_ids"] = list(
            db.scalars(
                select(GoalProjectLink.project_id)
                .where(GoalProjectLink.goal_id == row.id)
                .order_by(GoalProjectLink.project_id)
            )
        )
        baseline, current, target = row.metric_baseline, row.metric_current, row.metric_target
        result["progress"] = (
            None
            if current is None or target is None or target == baseline
            else max(0, min(1, float((current - baseline) / (target - baseline))))
        )
        result["notes"] = related_notes(db, row.owner_id, "goal", row.id)
    if isinstance(row, Project):
        result["goal_ids"] = list(
            db.scalars(
                select(GoalProjectLink.goal_id)
                .where(GoalProjectLink.project_id == row.id)
                .order_by(GoalProjectLink.goal_id)
            )
        )
        tasks = list(
            db.scalars(
                select(Task).where(
                    Task.owner_id == row.owner_id,
                    Task.project_id == row.id,
                    Task.archived.is_(False),
                    Task.is_template.is_(False),
                )
            )
        )
        result["task_count"] = len(tasks)
        result["completed_task_count"] = sum(t.status == "completed" for t in tasks)
        result["notes"] = related_notes(db, row.owner_id, "project", row.id)
    return result


def snapshot(db, owner):
    return {
        key: [
            data(db, r)
            for r in db.scalars(select(model).where(model.owner_id == owner).order_by(model.name, model.id))
        ]
        for key, model in (
            ("spaces", Space),
            ("areas", Area),
            ("goals", Goal),
            ("projects", Project),
            ("actors", Actor),
        )
    }


def mutate(db, owner, tool, args):
    from .domain import DomainError, check_revision, emit, owned

    kind, action = tool.split(".")
    model = {"space": Space, "area": Area, "goal": Goal, "project": Project, "actor": Actor}[kind]
    key = kind + "_id"
    values = args.model_dump(exclude_unset=True, exclude={key, "expected_revision"})
    links = values.pop("project_ids" if kind == "goal" else "goal_ids", None)
    if action == "create":
        row = model(owner_id=owner)
    else:
        row = owned(db, model, getattr(args, key), owner, lock=True)
        check_revision(row, args.expected_revision)
    for field in (
        "name",
        "description",
        "success_criteria",
        "status",
        "horizon",
        "metric_baseline",
        "metric_unit",
        "archived",
    ):
        if field in values and values[field] is None:
            raise DomainError("INVALID_ARGUMENT", f"{field} cannot be null.")
    if kind in {"goal", "project"}:
        home_changes(db, owner, values, row)
    if kind == "area":
        sid = values.get("space_id", getattr(row, "space_id", None))
        if not sid:
            raise DomainError("INVALID_ARGUMENT", "An area needs a space.")
        space = owned(db, Space, sid, owner)
        if space.archived and sid != getattr(row, "space_id", None):
            raise DomainError("INVALID_ARGUMENT", "Choose an active space.")
    if kind == "goal":
        parent = values.get("parent_goal_id", row.parent_goal_id)
        seen = {row.id} if row.id else set()
        while parent:
            if parent in seen:
                raise DomainError("INVALID_ARGUMENT", "Goals cannot contain a parent cycle.")
            seen.add(parent)
            ancestor = owned(db, Goal, parent, owner)
            if ancestor.archived and parent == values.get("parent_goal_id") and parent != row.parent_goal_id:
                raise DomainError("INVALID_ARGUMENT", "Choose an active parent goal.")
            parent = ancestor.parent_goal_id
    old_home = (getattr(row, "space_id", None), getattr(row, "area_id", None))
    old_name = row.name
    for k, v in values.items():
        setattr(row, k, v)
    if kind == "project" and row.start_date and row.target_date and row.start_date > row.target_date:
        raise DomainError("INVALID_ARGUMENT", "Project target must be on or after its start.")
    if kind in {"space", "area", "project", "actor"}:
        q = select(model).where(model.owner_id == owner, model.name == row.name, model.id != (row.id or ""))
        if kind == "area":
            q = q.where(Area.space_id == row.space_id)
        if db.scalar(q):
            raise DomainError("REVISION_CONFLICT", f"A {kind} with this name already exists.", 409)
    if kind in {"goal", "project"} and "status" in values:
        row.completed_at = (row.completed_at or now()) if row.status in {"achieved", "completed"} else None
    if action == "create":
        db.add(row)
    else:
        row.revision += 1
    if hasattr(row, "updated_at"):
        row.updated_at = now()
    db.flush()
    relationships = replace_goals(db, owner, row, links) if links is not None else None
    if kind == "project" and (old_name != row.name or old_home != (row.space_id, row.area_id)):
        for child in db.scalars(select(Task).where(Task.owner_id == owner, Task.project_id == row.id)):
            child.project, child.space_id, child.area_id = row.name, row.space_id, row.area_id
            changed(db, owner, child)
        for child in db.scalars(select(Note).where(Note.owner_id == owner, Note.project_id == row.id)):
            child.space_id, child.area_id = row.space_id, row.area_id
            changed(db, owner, child)
    if kind == "area" and old_home[0] != row.space_id:
        for child_model in (Goal, Project, Task, Note):
            for child in db.scalars(
                select(child_model).where(child_model.owner_id == owner, child_model.area_id == row.id)
            ):
                child.space_id = row.space_id
                changed(db, owner, child)
    if kind == "actor" and old_name != row.name:
        for task in db.scalars(select(Task).where(Task.owner_id == owner, Task.assignee_id == row.id)):
            task.assignee = row.name
            changed(db, owner, task)
    emit(db, owner, kind + ".changed", row.id, row.revision)
    db.flush()
    result = data(db, row)
    if relationships is not None:
        result["relationship_changes"] = relationships
    return result


def related_notes(db, owner, kind, entity_id):
    from sqlalchemy import or_

    if kind == "goal":
        ids = select(NoteGoalLink.note_id).where(NoteGoalLink.goal_id == entity_id)
        condition = Note.id.in_(ids)
    elif kind == "project":
        ids = select(NoteProjectLink.note_id).where(NoteProjectLink.project_id == entity_id)
        condition = or_(Note.project_id == entity_id, Note.id.in_(ids))
    elif kind == "task":
        ids = select(NoteTaskLink.note_id).where(
            NoteTaskLink.task_id == entity_id, NoteTaskLink.linked.is_(True)
        )
        condition = Note.id.in_(ids)
    else:
        ids = select(NoteNoteLink.note_id).where(NoteNoteLink.related_note_id == entity_id)
        condition = Note.id.in_(ids)
    return [
        {"id": n.id, "title": n.title, "archived": n.archived}
        for n in db.scalars(
            select(Note).where(Note.owner_id == owner, condition).order_by(Note.title, Note.id)
        )
    ]


def note_links(db, row):
    result = {}
    for key, link, model, col in (
        ("goals", NoteGoalLink, Goal, "goal_id"),
        ("projects", NoteProjectLink, Project, "project_id"),
        ("related_notes", NoteNoteLink, Note, "related_note_id"),
    ):
        ids = select(getattr(link, col)).where(link.note_id == row.id)
        result[key] = [
            {
                "id": r.id,
                "title" if model is Note else "name": r.title if model is Note else r.name,
                "archived": r.archived,
            }
            for r in db.scalars(
                select(model).where(model.owner_id == row.owner_id, model.id.in_(ids)).order_by(model.id)
            )
        ]
    result["backlinks"] = related_notes(db, row.owner_id, "note", row.id)
    return result


def save_note_links(db, owner, row, values):
    from .domain import DomainError, emit, owned

    for key, link, model, col in (
        ("goal_ids", NoteGoalLink, Goal, "goal_id"),
        ("project_ids", NoteProjectLink, Project, "project_id"),
        ("related_note_ids", NoteNoteLink, Note, "related_note_id"),
    ):
        if key not in values:
            continue
        ids = set(values[key])
        previous = set(db.scalars(select(getattr(link, col)).where(link.note_id == row.id)))
        for rid in ids:
            target = owned(db, model, rid, owner)
            if model is Note and rid == row.id:
                raise DomainError("INVALID_ARGUMENT", "A note cannot link to itself.")
            if target.archived and rid not in previous:
                raise DomainError("INVALID_ARGUMENT", "Restore the record before adding a new link.")
        db.execute(delete(link).where(link.note_id == row.id))
        for rid in sorted(ids):
            db.add(link(**{"note_id": row.id, col: rid}))
        for rid in ids ^ previous:
            emit(db, owner, "organization.changed" if model is not Note else "note.changed", rid)
    db.flush()
