"""Owner-scoped Linear synchronization through durable jobs and stable issue identities."""

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date, datetime, timedelta
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import select

from .db import session_scope
from .domain import DomainError, advisory, emit, enqueue_job
from .google_auth import seal, unseal
from .linear_client import CREATE_QUERY, UPDATE_QUERY, LinearClient, LinearFailure
from .models import Job, LinearConnection, LinearIssue, Project, Task, now

TERMINAL = {"succeeded", "failed", "cancelled", "unconfirmed", "conflict"}
STATE_TYPES = {
    "backlog": "backlog",
    "open": "unstarted",
    "in_progress": "started",
    "completed": "completed",
    "cancelled": "canceled",
    "deferred": "backlog",
    "waiting": "started",
}


# Set while an agent run writes the task's status (agent_sync.py): the observed start (in_progress) or the
# run's completion (done). Such a write never fails the local change; its Linear outcome is recorded instead.
run_write = ContextVar("linear_run_write", default=None)


@contextmanager
def for_run(run_id, purpose):
    outcome = {"purpose": purpose, "run_id": run_id}
    token = run_write.set(outcome)
    try:
        yield outcome
    finally:
        run_write.reset(token)


def connection(db, owner, generation=None):
    row = db.get(LinearConnection, owner)
    if not row or not row.enabled or not row.credentials:
        raise DomainError("LINEAR_CONNECT", "Connect Linear in Settings.", 409)
    if generation is not None and row.generation != generation:
        raise DomainError("LINEAR_CHANGED", "Linear connection changed. Review this operation.", 409)
    return row


def summary(db, owner):
    from .linear_structure import overview

    row = db.get(LinearConnection, owner)
    if not row:
        return {"connected": False, "teams": []}
    jobs = list(
        db.scalars(
            select(Job)
            .where(Job.owner_id == owner, Job.kind == "linear_write")
            .order_by(Job.created_at.desc())
            .limit(20)
        )
    )
    return {
        "connected": row.enabled,
        "workspace": row.workspace_name,
        "viewer_id": row.viewer_id,
        "revision": row.revision,
        "team_ids": row.team_ids,
        "only_mine": row.only_mine,
        "teams": row.directory.get("teams", []),
        "states": row.directory.get("states", []),
        "users": row.directory.get("users", []),
        "projects": row.directory.get("projects", []),
        "last_sync_at": row.last_sync_at,
        "status": row.status,
        "error": row.error,
        "recent_changes": [
            {"job_id": j.id, "status": j.status, "message": (j.result or {}).get("message", "")} for j in jobs
        ],
        "structure": overview(row),
    }


def connect(owner, key):
    client = LinearClient(key)
    try:
        meta = client.identity()
    except LinearFailure as error:
        raise DomainError(
            "LINEAR_CONNECT", "Linear could not verify this key. Check its permissions.", 400
        ) from error
    finally:
        client.close()
    with session_scope() as db:
        advisory(db, f"workspace:{owner}")
        row = db.get(LinearConnection, owner)
        org = meta["organization"]
        viewer = meta["viewer"]
        if not row:
            row = LinearConnection(
                owner_id=owner, workspace_id=org["id"], workspace_name=org["name"], viewer_id=viewer["id"]
            )
            db.add(row)
            db.flush()
        if row.workspace_id != org["id"] or row.viewer_id != viewer["id"]:
            row.team_ids = []
            row.last_sync_at = row.full_sync_at = None
        row.workspace_id, row.workspace_name, row.viewer_id = org["id"], org["name"], viewer["id"]
        row.credentials = seal({"api_key": key})
        row.enabled = True
        row.generation += 1
        row.revision += 1
        row.directory = {**row.directory, "teams": meta["teams"]}
        row.status, row.error, row.next_sync_at = "ready", "", now()
        emit(db, owner, "linear.changed", owner)
        return summary(db, owner)


def queue_sync(db, owner, force=False):
    row = db.get(LinearConnection, owner)
    if not row or not row.enabled or not row.team_ids:
        return None
    if not force and row.next_sync_at and row.next_sync_at > now():
        return None
    pending = db.scalar(
        select(Job)
        .where(Job.owner_id == owner, Job.kind == "linear_sync", Job.status.in_(["queued", "running"]))
        .order_by(Job.created_at.desc())
    )
    if pending:
        if pending.payload.get("generation") == row.generation and pending.created_at > now() - timedelta(
            minutes=20
        ):
            return pending.id
        pending.status = "cancelled"
    job = enqueue_job(db, owner, "linear_sync", {"generation": row.generation})
    row.next_sync_at = now() + timedelta(minutes=5)
    return job.id


def link_for_task(db, owner, task_id):
    return db.scalar(select(LinearIssue).where(LinearIssue.owner_id == owner, LinearIssue.task_id == task_id))


def external_state(db, link):
    task = db.get(Task, link.task_id)
    snapshot = link.snapshot or {}
    task.external = {
        "provider": "linear",
        "local_archived": bool((task.external or {}).get("local_archived")),
        "identifier": snapshot.get("identifier"),
        "url": snapshot.get("url"),
        "team_id": (snapshot.get("team") or {}).get("id"),
        "state": (snapshot.get("state") or {}).get("name"),
        "state_id": (snapshot.get("state") or {}).get("id"),
        "priority": snapshot.get("priority"),
        "labels": snapshot.get("labels", {}).get("nodes", []),
        "sync_state": link.sync_state,
        "job_id": link.pending_job_id,
        "remote_updated_at": snapshot.get("updatedAt"),
    }
    emit(db, link.owner_id, "task.changed", task.id, task.revision)


def local_project(db, owner, workspace, remote):
    if not remote:
        return None
    pid = str(uuid5(NAMESPACE_URL, "linear:" + owner + ":" + workspace + ":" + remote["id"]))
    row = db.get(Project, pid)
    name = remote["name"][:170] + " · Linear"
    collision = db.scalar(
        select(Project).where(Project.owner_id == owner, Project.name == name, Project.id != pid)
    )
    if collision:
        name += " " + remote["id"][:8]
    if not row:
        row = Project(id=pid, owner_id=owner, name=name, description=remote.get("description") or "")
        db.add(row)
        db.flush()
        emit(db, owner, "project.changed", pid, row.revision)
    elif row.name != name or row.description != (remote.get("description") or ""):
        row.name = name
        row.description = remote.get("description") or ""
        row.revision += 1
        emit(db, owner, "project.changed", pid, row.revision)
    return row


def older(remote, snapshot):
    try:
        return datetime.fromisoformat(remote["updatedAt"]) < datetime.fromisoformat(snapshot["updatedAt"])
    except (KeyError, TypeError, ValueError):
        return False


def apply_remote(db, conn, link, remote, force=False):
    if not force and older(remote, link.snapshot or {}):
        return  # A reordered, older Linear read never rolls back a newer applied snapshot.
    task = db.get(Task, link.task_id)
    project = local_project(db, conn.owner_id, conn.workspace_id, remote.get("project"))
    state = remote["state"]["type"]
    changes = {
        "title": remote["title"],
        "notes": remote.get("description") or "",
        "status": {
            "started": "in_progress",
            "completed": "completed",
            "canceled": "cancelled",
            "duplicate": "cancelled",
            "backlog": "backlog",
        }.get(state, "open"),
        "priority": {0: 0, 1: 3, 2: 3, 3: 2, 4: 1}.get(remote.get("priority"), 0),
        "due_date": date.fromisoformat(remote["dueDate"]) if remote.get("dueDate") else None,
        "project_id": project.id if project else None,
        "project": project.name if project else None,
        "assignee": "owner"
        if (remote.get("assignee") or {}).get("id") == conn.viewer_id
        else (remote.get("assignee") or {}).get("name", "unassigned"),
        "archived": bool(remote.get("archivedAt")) or bool((task.external or {}).get("local_archived")),
    }
    from .productivity import task_home

    changes = task_home(db, conn.owner_id, changes, task)
    from .agent_actor import Actor, acting, gate

    with acting(Actor("source_sync")):
        # A completion arriving from Linear during an agent run stops the run (recorded override).
        gate(db, conn.owner_id, task, changes)
    changed = any(getattr(task, k) != v for k, v in changes.items())
    if changes["due_date"] is None:
        changes.update(due_time=None, due_timezone=None)
    for key, value in changes.items():
        setattr(task, key, value)
    if changed:
        if task.status in {"completed", "cancelled"}:
            from .task_alerts import finish_task

            finish_task(db, task, task.status, sync_external=False, gated=True)
        else:
            task.revision += 1
            task.updated_at = now()
            task.completed_at = None
    link.snapshot = remote
    link.latest_remote = None
    link.sync_state = "synced"
    link.pending_job_id = None
    external_state(db, link)


def process_sync(job_id):
    with session_scope() as db:
        job = db.get(Job, job_id)
        if not job or job.status in TERMINAL:
            return
        try:
            conn = connection(db, job.owner_id, job.payload["generation"])
        except DomainError:
            job.status = "cancelled"
            return
        owner, generation, key = job.owner_id, conn.generation, unseal(conn.credentials)["api_key"]
        teams, only_mine, viewer, workspace = (
            list(conn.team_ids),
            conn.only_mine,
            conn.viewer_id,
            conn.workspace_id,
        )
        full = not conn.full_sync_at or now() - conn.full_sync_at > timedelta(days=1)
        started = now()
        filter = {"team": {"id": {"in": teams}}}
        if full and only_mine:
            filter["assignee"] = {"id": {"eq": viewer}}
        if not full and conn.last_sync_at:
            filter["updatedAt"] = {"gte": (conn.last_sync_at - timedelta(minutes=2)).isoformat()}
        known = (
            set(
                db.scalars(
                    select(LinearIssue.remote_id).where(
                        LinearIssue.owner_id == owner,
                        LinearIssue.workspace_id == workspace,
                        LinearIssue.sync_state != "unlinked",
                    )
                )
            )
            if full and only_mine
            else set()
        )
        job.status = "running"
    client = LinearClient(key)
    try:
        meta = client.identity()
        if meta["organization"]["id"] != workspace:
            raise LinearFailure("workspace_changed")
        directory = client.directory(teams)
        remote_rows = client.issues(filter)
        # The assignee filter hides reassigned issues; read them by id so they become out of scope, not lost.
        missing = sorted(known - {r["id"] for r in remote_rows})
        for index in range(0, len(missing), 100):
            remote_rows += client.issues({"id": {"in": missing[index : index + 100]}})
        with session_scope() as db:
            advisory(db, f"workspace:{owner}")
            job = db.get(Job, job_id)
            try:
                conn = connection(db, owner, generation)
            except DomainError:
                job.status = "cancelled"
                return
            existing = {
                r.remote_id: r
                for r in db.scalars(
                    select(LinearIssue).where(
                        LinearIssue.owner_id == owner, LinearIssue.workspace_id == workspace
                    )
                )
            }
            seen = set()
            for remote in remote_rows:
                rid = remote["id"]
                seen.add(rid)
                included = remote["team"]["id"] in teams and (
                    not only_mine or (remote.get("assignee") or {}).get("id") == viewer
                )
                link = existing.get(rid)
                if link and link.sync_state == "unlinked":
                    continue
                if not included:
                    if link and not link.pending_job_id:
                        link.sync_state = "out_of_scope"
                        external_state(db, link)
                    continue
                if not link:
                    task = Task(owner_id=owner, title=remote["title"])
                    db.add(task)
                    db.flush()
                    link = LinearIssue(
                        owner_id=owner, workspace_id=workspace, remote_id=rid, task_id=task.id, snapshot={}
                    )
                    db.add(link)
                    db.flush()
                    existing[rid] = link
                if link.pending_job_id or link.sync_state in {"conflict", "unconfirmed"}:
                    link.latest_remote = remote
                    continue
                apply_remote(db, conn, link, remote)
            if full:
                for rid, link in existing.items():
                    if rid not in seen and not link.pending_job_id and link.sync_state != "unlinked":
                        link.sync_state = "unavailable"
                        external_state(db, link)
            # Link only parents that are also in this owner's imported scope. A Home filed in Eridani (a
            # parent that is not a Linear issue) stays; a Linear parent that went away is cleared.
            linked_tasks = {link.task_id for link in existing.values()}
            for remote in remote_rows:
                link = existing.get(remote["id"])
                parent = existing.get((remote.get("parent") or {}).get("id"))
                if link and not link.pending_job_id and link.sync_state == "synced":
                    task = db.get(Task, link.task_id)
                    if parent or task.parent_task_id in linked_tasks:
                        task.parent_task_id = parent.task_id if parent else None
            projects = {p["id"]: p for p in conn.directory.get("projects", [])}
            for remote in remote_rows:
                p = remote.get("project")
                if p:
                    projects[p["id"]] = {
                        **p,
                        "local_id": str(
                            uuid5(NAMESPACE_URL, "linear:" + owner + ":" + workspace + ":" + p["id"])
                        ),
                    }
            conn.directory = {**directory, "teams": meta["teams"], "projects": list(projects.values())}
            from .linear_structure import after_sync, sources

            # Optional structure mapping: new issues inherit their mapped Home (linear_structure.py).
            after_sync(
                db,
                owner,
                conn,
                [existing[r["id"]] for r in remote_rows if r["id"] in existing],
                sources(meta["teams"], list(projects.values())),
            )
            conn.last_sync_at = started
            if full:
                conn.full_sync_at = started
            conn.status, conn.error = "ready", ""
            job.status, job.finished_at, job.result = "succeeded", now(), {"count": len(remote_rows)}
            emit(db, owner, "linear.changed", owner)
    except (LinearFailure, KeyError, TypeError, ValueError):
        with session_scope() as db:
            job = db.get(Job, job_id)
            job.status, job.finished_at = "failed", now()
            conn = db.get(LinearConnection, owner)
            if conn and conn.generation == generation:
                conn.status = "error"
                conn.error = "Sync failed; existing tasks are preserved. Check access or retry."
                conn.next_sync_at = now() + timedelta(minutes=10)
    finally:
        client.close()


def require_editable(db, conn, link):
    if link.workspace_id != conn.workspace_id:
        raise DomainError("LINEAR_CHANGED", "Reconnect this task's Linear workspace.", 409)
    if (link.snapshot.get("team") or {}).get("id") not in conn.team_ids:
        raise DomainError("LINEAR_SCOPE", "Select this task's team in Settings.", 409)
    if link.pending_job_id:
        job = db.get(Job, link.pending_job_id)
        if job and job.status not in TERMINAL:
            raise DomainError("LINEAR_PENDING", "Wait for the current Linear change.", 409)
    if link.sync_state in {"conflict", "unconfirmed", "unavailable", "out_of_scope"}:
        raise DomainError(
            "LINEAR_CONFLICT", "Review the Linear difference before making another change.", 409
        )


def queue_write(db, conn, link, patch, create=False, origin=None):
    job = enqueue_job(
        db,
        conn.owner_id,
        "linear_write",
        {
            "generation": conn.generation,
            "workspace_id": conn.workspace_id,
            "remote_id": link.remote_id,
            "link_id": link.id,
            "create": create,
            "patch": patch,
            "base": link.snapshot,
            **({"run_id": origin["run_id"], "purpose": origin["purpose"]} if origin else {}),
        },
    )
    link.pending_job_id = job.id
    link.sync_state = "pending"
    external_state(db, link)
    return job


def state_id(conn, link, status):
    candidates = [
        s
        for s in conn.directory.get("states", [])
        if s["team"]["id"] == link.snapshot["team"]["id"] and s["type"] == STATE_TYPES[status]
    ]
    if not candidates:
        raise DomainError("LINEAR_STATE", "Choose an available Linear workflow status in the Linear editor.")
    return min(
        candidates,
        key=lambda s: (s["name"].lower() not in {"todo", "in progress", "done", "canceled", "backlog"}, s["name"]),
    )["id"]


def task_patch(conn, link, changes, task):
    patch = {}
    for key, target in [("title", "title"), ("notes", "description"), ("due_date", "dueDate")]:
        if key in changes and changes[key] != getattr(task, key):
            value = changes[key]
            patch[target] = value.isoformat() if isinstance(value, date) else value
    if "priority" in changes and changes["priority"] != task.priority:
        patch["priority"] = {0: 0, 1: 4, 2: 3, 3: 2}[changes["priority"]]
    if "status" in changes and changes["status"] != task.status:
        patch["stateId"] = state_id(conn, link, changes["status"])
    if "assignee" in changes and changes["assignee"] != task.assignee:
        name = changes["assignee"]
        matches = [u for u in conn.directory.get("users", []) if u["name"] == name]
        if name == "owner":
            patch["assigneeId"] = conn.viewer_id
        elif name == "unassigned":
            patch["assigneeId"] = None
        elif len(matches) == 1:
            patch["assigneeId"] = matches[0]["id"]
        else:
            raise DomainError("LINEAR_ASSIGNEE", "Choose the assignee in the Linear editor.")
    if "project_id" in changes and changes["project_id"] != task.project_id:
        pid = changes["project_id"]
        match = next((p for p in conn.directory.get("projects", []) if p["local_id"] == pid), None)
        if pid and not match:
            raise DomainError("LINEAR_PROJECT", "Choose a linked Linear project, or keep the project empty.")
        patch["projectId"] = match["id"] if match else None
    return patch


def before_task_update(db, owner, task, changes):
    origin = run_write.get()
    link = link_for_task(db, owner, task.id)
    if not link or link.sync_state == "unlinked":
        if origin is not None:
            origin["status"] = "skipped"
        return
    if "archived" in changes:
        task.external = {**(task.external or {}), "local_archived": changes["archived"]}
    mapped = {"title", "notes", "due_date", "priority", "status", "assignee", "project_id"}
    if not any(k in mapped and v != getattr(task, k) for k, v in changes.items()):
        return
    finishing = changes.get("status") in {"completed", "cancelled"} and changes["status"] != task.status
    try:
        conn = connection(db, owner)
        if finishing:
            in_flight = supersede_start(db, link)
            pending = db.get(Job, link.pending_job_id) if link.pending_job_id else None
            if in_flight is None and origin is not None and pending is not None and pending.status not in TERMINAL:
                in_flight = pending.id
            if in_flight is not None:
                # A run's In Progress write (or, for a run's own finish, any write) is executing: the finish
                # follows it (process_write sends it once that write settles). The local finish goes ahead.
                if origin is not None:
                    origin.update(status="deferred", after=in_flight)
                return
        require_editable(db, conn, link)
        patch = task_patch(conn, link, changes, task)
    except DomainError as exc:
        if origin is None or not finishing:
            raise  # A start Linear refuses leaves the local status alone (the caller records it).
        # A run's completion is never undone by source sync (C13); the outcome is recorded instead.
        origin.update(status="blocked", code=exc.code, message=exc.message[:300])
        return
    if patch:
        job = queue_write(db, conn, link, patch, origin=origin)
        if origin is not None:
            origin.update(status="queued", job_id=job.id)


def supersede_start(db, link):
    """Ordering guard: a run's In Progress write still pending when the task finishes.

    Not started (or waiting to retry before any write was attempted) → cancelled here, so it can never
    land after the finish (returns None). Executing or possibly written → its job id; that write settles
    first and process_write then sends the finish (finish_owed)."""
    if not link.pending_job_id:
        return None
    job = db.get(Job, link.pending_job_id, with_for_update=True)
    if not job or job.status in TERMINAL or (job.payload or {}).get("purpose") != "in_progress":
        return None
    if job.status in ("queued", "retrying") and not (job.result or {}).get("attempted"):
        job.status, job.finished_at = "cancelled", now()
        job.result = {**(job.result or {}), "message": "Superseded: the task finished before Linear showed it started."}
        link.pending_job_id, link.sync_state = None, "synced"
        external_state(db, link)
        return None
    return job.id


def finished_remotely(link, task):
    return ((link.snapshot or {}).get("state") or {}).get("type") == STATE_TYPES[task.status]


def finish_owed(db, owner, job):
    """After a write settles: a task finished locally while it was pending (a deferred run finish) gets its
    finish sent now, so a stale start never leaves Linear behind a completed task."""
    link = db.get(LinearIssue, job.payload["link_id"])
    task = db.get(Task, link.task_id) if link else None
    if task is None or task.status not in {"completed", "cancelled"} or finished_remotely(link, task):
        return
    pending = db.get(Job, link.pending_job_id) if link.pending_job_id else None
    if pending is not None and pending.id != job.id and pending.status not in TERMINAL:
        return
    try:
        conn = connection(db, owner)
        require_editable(db, conn, link)
        target = state_id(conn, link, task.status)
        if job.payload.get("patch", {}).get("stateId") == target:
            return  # This write was the finish itself; its failure shows on the task's Linear state.
        run_id = job.payload.get("run_id")
        follow = queue_write(db, conn, link, {"stateId": target},
                             origin={"run_id": run_id, "purpose": "done"} if run_id else None)
    except DomainError:
        return  # A conflict or a lost connection shows on the task's Linear state; the user resolves it.
    if run_id:
        from .agent_sync import record_outcome

        record_outcome(db, run_id, "done", {"status": "queued", "job_id": follow.id})


def target_values(remote):
    return {
        "title": remote["title"],
        "description": remote.get("description") or "",
        "dueDate": remote.get("dueDate"),
        "priority": remote.get("priority", 0),
        "stateId": remote["state"]["id"],
        "assigneeId": (remote.get("assignee") or {}).get("id"),
        "projectId": (remote.get("project") or {}).get("id"),
        "teamId": remote["team"]["id"],
    }


def matches(remote, patch):
    if not remote:
        return False
    values = target_values(remote)
    return all(values.get(k) == v for k, v in patch.items() if k not in {"id"})


def process_write(job_id):
    with session_scope() as db:
        # Locked: a completion may supersede a run's start write that has not begun (supersede_start).
        job = db.get(Job, job_id, with_for_update=True)
        if not job or job.status in TERMINAL:
            return
        owner, payload = job.owner_id, job.payload
        created_at = job.created_at
        link = db.get(LinearIssue, payload["link_id"])
        attempted = (job.result or {}).get("attempted", False)
        attempt = (job.result or {}).get("attempts", 0) + 1
        try:
            conn = connection(db, owner, payload["generation"])
            link = db.get(LinearIssue, payload["link_id"])
            if not link or link.pending_job_id != job.id:
                raise DomainError("LINEAR_CHANGED", "Operation changed.")
            team = payload["patch"].get("teamId") or payload["base"].get("team", {}).get("id")
            if team not in conn.team_ids:
                raise DomainError("LINEAR_SCOPE", "Team no longer selected.")
            key = unseal(conn.credentials)["api_key"]
        except DomainError:
            job.status = "unconfirmed" if attempted else "cancelled"
            job.result = {"message": "Connection changed; check Linear for any in-progress operation."}
            if link:
                link.sync_state = job.status
                external_state(db, link)
            return
        job.status = "running"
        job.result = {"attempts": attempt, "attempted": attempted}
    client = LinearClient(key)
    remote = None
    try:
        remote = client.issue(payload["remote_id"])
        if not matches(remote, payload["patch"]):
            if payload["create"] and remote:
                raise LinearFailure("conflict")
            if not payload["create"] and (
                not remote or remote.get("updatedAt") != payload["base"].get("updatedAt")
            ):
                raise LinearFailure("conflict")
            if now() - created_at > timedelta(hours=1):
                raise LinearFailure("expired")
            with session_scope() as db:
                connection(db, owner, payload["generation"])
                active = db.get(Job, job_id)
                active.result = {"attempts": attempt, "attempted": True}
            attempted = True
            variables = (
                {"input": {**payload["patch"], "id": payload["remote_id"]}}
                if payload["create"]
                else {"id": payload["remote_id"], "input": payload["patch"]}
            )
            result = client.query(CREATE_QUERY if payload["create"] else UPDATE_QUERY, variables)
            saved = result["issueCreate" if payload["create"] else "issueUpdate"]
            if not saved.get("success") or not saved.get("issue"):
                raise LinearFailure("invalid_response", retry=True, unknown=True)
            remote = saved["issue"]
            if remote["id"] != payload["remote_id"] or not matches(remote, payload["patch"]):
                raise LinearFailure("conflict")
        with session_scope() as db:
            advisory(db, f"workspace:{owner}")
            active = db.get(Job, job_id)
            link = db.get(LinearIssue, payload["link_id"])
            conn = db.get(LinearConnection, owner)
            active.status, active.finished_at = "succeeded", now()
            active.result = {
                "message": "Saved in Linear.",
                "task_id": link.task_id if link else None,
                "url": remote.get("url"),
            }
            task = db.get(Task, link.task_id) if link else None
            if (
                link
                and conn
                and conn.enabled
                and conn.generation == payload["generation"]
                and link.pending_job_id == job_id
                and task is not None
                and task.status in {"completed", "cancelled"}
                and remote["state"]["type"] != STATE_TYPES[task.status]
            ):
                # The task finished locally while this write (a run's start, say) was in flight: applying it
                # would reopen the task. Keep the local status, take the snapshot, send the finish next.
                link.snapshot, link.latest_remote, link.sync_state, link.pending_job_id = remote, None, "synced", None
                external_state(db, link)
                finish_owed(db, owner, active)
                conn.next_sync_at = now()
            elif (
                link
                and conn
                and conn.enabled
                and conn.generation == payload["generation"]
                and link.pending_job_id == job_id
            ):
                apply_remote(db, conn, link, remote)
                from .linear_structure import after_sync

                after_sync(db, owner, conn, [link], {})  # A confirmed project change files the issue now.
                conn.next_sync_at = now()
            elif link and link.pending_job_id == job_id:
                link.sync_state = "unconfirmed"
                link.latest_remote = remote
                external_state(db, link)
            emit(db, owner, "linear.changed", owner)
    except (LinearFailure, DomainError, KeyError, TypeError, ValueError) as error:
        retry = isinstance(error, LinearFailure) and error.retry and attempt < 5
        conflict = isinstance(error, LinearFailure) and error.code == "conflict"
        uncertain = attempted and (not isinstance(error, LinearFailure) or error.unknown or attempt > 1)
        status = "retrying" if retry else "conflict" if conflict else "unconfirmed" if uncertain else "failed"
        message = (
            "Linear changed since your last sync. Review both versions."
            if conflict
            else "Linear has not confirmed this change. Check the result before submitting again."
        )
        with session_scope() as db:
            advisory(db, f"workspace:{owner}")
            active = db.get(Job, job_id)
            active.status = status
            active.finished_at = None if retry else now()
            active.result = {"attempts": attempt, "attempted": attempted, "message": message}
            link = db.get(LinearIssue, payload["link_id"])
            if link and link.pending_job_id == job_id:
                link.sync_state = "pending" if retry else status
                link.latest_remote = remote
                external_state(db, link)
            if not retry:
                finish_owed(db, owner, active)
            emit(db, owner, "linear.changed", owner)
        if retry:
            raise RuntimeError("Linear operation will retry with its original identity.") from None
    finally:
        client.close()
