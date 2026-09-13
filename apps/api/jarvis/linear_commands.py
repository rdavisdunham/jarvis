from datetime import date, datetime, timedelta
from uuid import uuid4

from .db import session_scope
from .domain import DomainError, advisory, check_revision, emit, owned, serial
from .google_auth import seal, unseal
from .linear_client import LinearClient, LinearFailure
from .linear_sync import (
    TERMINAL,
    apply_remote,
    connection,
    external_state,
    link_for_task,
    queue_sync,
    queue_write,
    require_editable,
    summary,
    task_patch,
)
from .models import Job, LinearIssue, Task, now


def validate_patch(conn, patch, team):
    if team not in conn.team_ids:
        raise DomainError("LINEAR_SCOPE", "Select this Linear team in Settings first.")
    if patch.get("stateId") and not any(
        s["id"] == patch["stateId"] and s["team"]["id"] == team for s in conn.directory.get("states", [])
    ):
        raise DomainError("LINEAR_STATE", "Choose a workflow status from this team.")
    if patch.get("assigneeId") and not any(
        u["id"] == patch["assigneeId"] and u.get("active", True) for u in conn.directory.get("users", [])
    ):
        raise DomainError("LINEAR_ASSIGNEE", "Choose an active Linear member.")
    if patch.get("projectId") and not any(
        p["id"] == patch["projectId"] for p in conn.directory.get("projects", [])
    ):
        raise DomainError("LINEAR_PROJECT", "Choose a project from the synced Linear directory.")
    if patch.get("dueDate"):
        try:
            date.fromisoformat(patch["dueDate"])
        except ValueError:
            raise DomainError("INVALID_ARGUMENT", "Linear due dates use YYYY-MM-DD.") from None


def comparison(owner, task_id):
    with session_scope() as db:
        task = owned(db, Task, task_id, owner)
        conn = connection(db, owner)
        link = link_for_task(db, owner, task_id)
        if not link or link.workspace_id != conn.workspace_id:
            raise DomainError("NOT_FOUND", "This task has no connected Linear issue.", 404)
        key, remote_id, generation, revision = (
            unseal(conn.credentials)["api_key"],
            link.remote_id,
            conn.generation,
            task.revision,
        )
        local = serial(task)
        pending = db.get(Job, link.pending_job_id) if link.pending_job_id else None
        requested = pending.payload.get("patch", {}) if pending and pending.status != "succeeded" else {}
        request_labels = {}
        for field, value in requested.items():
            names = {
                "stateId": ("states", "Status"),
                "assigneeId": ("users", "Assignee"),
                "projectId": ("projects", "Project"),
            }
            if field in names:
                directory, label = names[field]
                request_labels[label] = next(
                    (r["name"] for r in conn.directory.get(directory, []) if r["id"] == value),
                    "Unassigned" if value is None else value,
                )
            elif field == "priority":
                request_labels["Priority"] = ["None", "Urgent", "High", "Medium", "Low"][value]
            elif field != "teamId":
                request_labels[field] = value
    client = LinearClient(key)
    try:
        remote = client.issue(remote_id)
    except LinearFailure:
        raise DomainError(
            "LINEAR_UNAVAILABLE", "Could not read the current Linear issue. Your task is safe here.", 503
        ) from None
    finally:
        client.close()
    token = seal(
        {
            "purpose": "linear_compare",
            "owner": owner,
            "task_id": task_id,
            "revision": revision,
            "remote_id": remote_id,
            "generation": generation,
            "remote": remote,
            "expires_at": (now() + timedelta(minutes=10)).isoformat(),
        }
    )
    return {"local": local, "linear": remote, "pending_change": request_labels, "edit_token": token}


def mutate(db, owner, tool, args):
    advisory(db, f"workspace:{owner}")
    if tool == "linear.resolve" and args.choice == "unlink":
        task = owned(db, Task, args.task_id, owner, lock=True)
        check_revision(task, args.expected_revision)
        link = link_for_task(db, owner, task.id)
        if not link:
            raise DomainError("NOT_FOUND", "This task has no Linear issue.", 404)
        pending = db.get(Job, link.pending_job_id) if link.pending_job_id else None
        if pending and pending.status not in TERMINAL:
            raise DomainError("LINEAR_PENDING", "Wait for the current Linear operation.", 409)
        link.sync_state, link.pending_job_id = "unlinked", None
        task.external = {}
        task.revision += 1
        emit(db, owner, "task.changed", task.id, task.revision)
        return serial(task)
    conn = connection(db, owner)
    if tool == "linear.select":
        check_revision(conn, args.expected_revision)
        allowed = {t["id"] for t in conn.directory.get("teams", [])}
        if not set(args.team_ids) <= allowed:
            raise DomainError("LINEAR_SCOPE", "Select an available team.")
        conn.team_ids = list(dict.fromkeys(args.team_ids))
        conn.only_mine = args.only_mine
        conn.generation += 1
        conn.revision += 1
        conn.full_sync_at = None
        job_id = queue_sync(db, owner, force=True)
        emit(db, owner, "linear.changed", owner)
        return {**summary(db, owner), "job_id": job_id}
    if tool == "linear.create":
        validate_patch(conn, {"dueDate": args.due_date}, args.team_id)
        task = Task(
            owner_id=owner,
            title=args.title,
            notes=args.description,
            due_date=date.fromisoformat(args.due_date) if args.due_date else None,
            priority={0: 0, 1: 3, 2: 3, 3: 2, 4: 1}[args.priority],
        )
        db.add(task)
        db.flush()
    else:
        task = owned(db, Task, args.task_id, owner, lock=True)
        check_revision(task, args.expected_revision)
    link = link_for_task(db, owner, task.id)
    if tool in {"linear.publish", "linear.create"}:
        if link or task.is_template or task.archived:
            raise DomainError("LINEAR_LINK", "Choose an active task that is not already linked to Linear.")
        patch = {
            "teamId": args.team_id,
            "title": task.title,
            "description": task.notes,
            "dueDate": task.due_date.isoformat() if task.due_date else None,
            "priority": args.priority if tool == "linear.create" else {0: 0, 1: 4, 2: 3, 3: 2}[task.priority],
            "assigneeId": args.assignee_id if "assignee_id" in args.model_fields_set else conn.viewer_id,
        }
        for key, remote_key in [("state_id", "stateId"), ("project_id", "projectId")]:
            if getattr(args, key):
                patch[remote_key] = getattr(args, key)
        if not patch.get("stateId"):
            from .linear_sync import STATE_TYPES

            candidate = next(
                (
                    s
                    for s in conn.directory.get("states", [])
                    if s["team"]["id"] == args.team_id and s["type"] == STATE_TYPES[task.status]
                ),
                None,
            )
            if candidate:
                patch["stateId"] = candidate["id"]
            elif task.status not in {"open", None}:
                raise DomainError(
                    "LINEAR_STATE", "Choose a Linear workflow status before publishing this task."
                )
        validate_patch(conn, patch, args.team_id)
        link = LinearIssue(
            owner_id=owner,
            workspace_id=conn.workspace_id,
            remote_id=str(uuid4()),
            task_id=task.id,
            snapshot={"team": {"id": args.team_id}},
        )
        db.add(link)
        db.flush()
    elif tool == "linear.update":
        if not link:
            raise DomainError("NOT_FOUND", "This task has no Linear issue.", 404)
        require_editable(db, conn, link)
        patch = {}
        for key, value in args.model_dump(
            exclude_unset=True, exclude={"task_id", "expected_revision"}
        ).items():
            target = {
                "state_id": "stateId",
                "assignee_id": "assigneeId",
                "project_id": "projectId",
                "due_date": "dueDate",
            }.get(key, key)
            if value is None and key in {"title", "description", "priority", "state_id"}:
                raise DomainError("INVALID_ARGUMENT", f"{key} cannot be empty.")
            patch[target] = value
        validate_patch(conn, patch, link.snapshot["team"]["id"])
    elif tool == "linear.resolve":
        if not link:
            raise DomainError("NOT_FOUND", "This task has no Linear issue.", 404)
        pending = db.get(Job, link.pending_job_id) if link.pending_job_id else None
        if pending and pending.status not in TERMINAL:
            raise DomainError("LINEAR_PENDING", "Wait for the current Linear operation.", 409)
        token = unseal(args.edit_token or "")
        if (
            token.get("purpose") != "linear_compare"
            or token.get("owner") != owner
            or token.get("task_id") != task.id
            or token.get("revision") != task.revision
            or token.get("remote_id") != link.remote_id
            or token.get("generation") != conn.generation
            or datetime.fromisoformat(token["expires_at"]) <= now()
        ):
            raise DomainError("LINEAR_STALE", "Refresh the Linear comparison.", 409)
        remote = token.get("remote")
        if not remote:
            raise DomainError(
                "LINEAR_MISSING", "The issue is unavailable. Unlink to keep working locally.", 409
            )
        if args.choice == "linear":
            apply_remote(db, conn, link, remote)
            return serial(task)
        # Republish the owner's current shared fields against the freshly reviewed version.
        link.snapshot = remote
        shadow = Task(
            title=remote["title"],
            notes=remote.get("description") or "",
            due_date=date.fromisoformat(remote["dueDate"]) if remote.get("dueDate") else None,
            priority={0: 0, 1: 3, 2: 3, 3: 2, 4: 1}.get(remote.get("priority"), 0),
            status="__remote__",
            assignee="__remote__",
            project_id="__remote__",
        )
        patch = task_patch(
            conn,
            link,
            {
                k: getattr(task, k)
                for k in ("title", "notes", "due_date", "priority", "status", "assignee", "project_id")
            },
            shadow,
        )
        if pending and pending.status != "succeeded":
            patch.update(
                {k: v for k, v in pending.payload.get("patch", {}).items() if k not in {"teamId", "id"}}
            )
        validate_patch(conn, patch, remote["team"]["id"])
    else:
        raise DomainError("INVALID_ARGUMENT", "Unknown Linear action.")
    job = queue_write(db, conn, link, patch, create=tool in {"linear.create", "linear.publish"})
    task.revision += 1
    task.updated_at = now()
    external_state(db, link)
    return {
        "task": serial(task),
        "job_id": job.id,
        "status": "queued",
        "message": "Saved locally; waiting for Linear to confirm.",
    }
