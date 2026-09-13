"""Organization rules shared by UI and assistant commands."""

from sqlalchemy import select

from .models import Project, Task


def project_changes(db, owner, changes, task=None):
    from .domain import DomainError, emit, owned

    if "project_id" in changes:
        project = owned(db, Project, changes["project_id"], owner) if changes["project_id"] else None
        if project and project.archived and (not task or task.project_id != project.id):
            raise DomainError("INVALID_ARGUMENT", "Choose an active project.")
        if "project" in changes and changes["project"] not in {None, "", project.name if project else None}:
            raise DomainError("INVALID_ARGUMENT", "Project name and ID disagree.")
        changes["project"] = project.name if project else None
    elif "project" in changes:
        name = (changes["project"] or "").strip()
        project = (
            db.scalar(select(Project).where(Project.owner_id == owner, Project.name == name))
            if name
            else None
        )
        if name and not project:
            project = Project(owner_id=owner, name=name)
            db.add(project)
            db.flush()
            emit(db, owner, "project.changed", project.id, project.revision)
        if project and project.archived and (not task or task.project_id != project.id):
            raise DomainError("INVALID_ARGUMENT", "Choose an active project.")
        changes.update(project_id=project.id if project else None, project=project.name if project else None)
    if "tags" in changes:
        if changes["tags"] is None:
            raise DomainError("INVALID_ARGUMENT", "Tags must be a list.")
        changes["tags"] = list(dict.fromkeys(t.strip() for t in changes["tags"] if t.strip()))
    if changes.get("parent_task_id"):
        seen = {task.id} if task else set()
        parent_id = changes["parent_task_id"]
        while parent_id:
            if parent_id in seen:
                raise DomainError("INVALID_ARGUMENT", "A task cannot be its own ancestor.")
            seen.add(parent_id)
            parent = owned(db, Task, parent_id, owner)
            if parent.archived:
                raise DomainError("INVALID_ARGUMENT", "Choose an unarchived parent task.")
            parent_id = parent.parent_task_id
    from .productivity import task_home

    return task_home(db, owner, changes, task)
