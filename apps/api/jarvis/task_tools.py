"""Exact task queries and bounded, owner-scoped immutable selection snapshots."""

import json
import time
from collections import OrderedDict
from datetime import date

from sqlalchemy import func, select

from .domain import DomainError, TaskUpdate, serial
from .models import GoalProjectLink, Task, uid

MAX_SELECTION = 1000
TTL_SECONDS = 900
MAX_SNAPSHOTS = 128
MAX_CACHE_BYTES = 16 * 1024 * 1024
snapshots = OrderedDict()
COMPACT_FIELDS = (
    "id",
    "revision",
    "title",
    "status",
    "project_id",
    "project",
    "space_id",
    "area_id",
    "assignee",
    "assignee_id",
    "tags",
    "work_type",
    "due_date",
    "due_time",
    "due_timezone",
    "planned_date",
    "priority",
    "estimate_minutes",
    "parent_task_id",
)


def compact(row):
    return {key: row.get(key) for key in COMPACT_FIELDS if key in row}


def selection(owner, selection_id):
    entry = snapshots.get(selection_id)
    if entry is None or entry["owner"] != owner or time.monotonic() >= entry["expires"]:
        raise DomainError(
            "SELECTION_EXPIRED",
            "This selection is unavailable or expired. List again to obtain a fresh selection. "
            "This does not prove a previous write failed; check its receipt before repeating work.",
            409,
        )
    return entry


def list_tasks(db, owner, args):
    limit, offset = args.get("limit", 30), args.get("offset", 0)
    if args.get("selection_id"):
        if any(k not in {"selection_id", "limit", "offset", "detail"} for k in args):
            raise DomainError(
                "INVALID_ARGUMENT", "Use only selection_id and pagination for a saved selection."
            )
        if args.get("detail") == "full":
            raise DomainError(
                "INVALID_ARGUMENT",
                "Saved selections contain compact records. Use task_get for full current details.",
            )
        entry = selection(owner, args["selection_id"])
        records, total, sid = entry["records"], entry["total"], args["selection_id"]
    else:
        q = select(Task).where(Task.owner_id == owner, Task.archived.is_(False))
        if args.get("query"):
            term = args["query"].replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            q = q.where(Task.title.ilike("%" + term + "%", escape="\\"))
        for field in ("project_id", "space_id", "area_id", "assignee_id", "work_type", "status", "assignee"):
            if field in args:
                q = q.where(getattr(Task, field) == args[field])
        if args.get("goal_id"):
            q = q.where(
                Task.project_id.in_(
                    select(GoalProjectLink.project_id).where(GoalProjectLink.goal_id == args["goal_id"])
                )
            )
        for tag in args.get("tags_all", []):
            q = q.where(Task.tags.contains([tag]))
        for tag in args.get("tags_none", []):
            q = q.where(~Task.tags.contains([tag]))
        if args.get("due_from"):
            q = q.where(Task.due_date >= date.fromisoformat(args["due_from"]))
        if args.get("due_through"):
            q = q.where(Task.due_date <= date.fromisoformat(args["due_through"]))
        if args.get("due_from") and args.get("due_through") and args["due_from"] > args["due_through"]:
            raise DomainError("INVALID_ARGUMENT", "due_from must be on or before due_through.")
        # Count and rows share one SQL snapshot, including under concurrent writes.
        matched = list(db.execute(q.add_columns(func.count().over()).order_by(Task.id).limit(MAX_SELECTION)))
        total = matched[0][1] if matched else 0
        records = [serial(row) for row, _ in matched]
        sid = None
        if total <= MAX_SELECTION:
            sid = uid()
            snapshots[sid] = {
                "owner": owner,
                "records": [compact(r) for r in records],
                "total": total,
                "expires": time.monotonic() + TTL_SECONDS,
            }
            for key in list(snapshots):
                if snapshots[key]["expires"] <= time.monotonic():
                    snapshots.pop(key, None)
            snapshots[sid]["bytes"] = len(json.dumps(snapshots[sid]).encode())
            while (
                len(snapshots) > MAX_SNAPSHOTS
                or sum(e["bytes"] for e in snapshots.values()) > MAX_CACHE_BYTES
            ):
                snapshots.popitem(last=False)
    page = records[offset : offset + limit]
    return {
        "tasks": page if args.get("detail") == "full" else [compact(r) for r in page],
        "match_count": total,
        "returned_count": len(page),
        "selection_id": sid,
        "selection_complete": total <= MAX_SELECTION,
        "selection_expires_in_seconds": max(0, int(selection(owner, sid)["expires"] - time.monotonic()))
        if sid
        else None,
        "next_offset": offset + limit if offset + limit < len(records) else None,
        "truncated": total > MAX_SELECTION,
        "pagination": "Pass selection_id with next_offset to retain this exact snapshot. task_selection_update applies ALL matches without copying IDs.",
        "scope_warning": "Narrow filters before applying; only the first 1000 matches are shown."
        if total > MAX_SELECTION
        else None,
    }


def apply_selection(db, owner, args, command_id):
    from .domain import TaskBatch, mutate

    entry = selection(owner, args.selection_id)
    changes = args.changes.model_dump(exclude_unset=True)
    if not changes:
        raise DomainError("INVALID_ARGUMENT", "Provide at least one requested change.")
    items = [
        TaskUpdate(task_id=row["id"], expected_revision=row["revision"], **changes)
        for row in entry["records"]
    ]
    if not items:
        return {
            "selection_id": args.selection_id,
            "tasks": [],
            "requested_count": 0,
            "applied_count": 0,
            "unchanged_count": 0,
            "task_ids": [],
            "applied_ids": [],
        }
    result = mutate(db, owner, "task.batch", TaskBatch(items=items), command_id)
    return {"selection_id": args.selection_id, **result}
