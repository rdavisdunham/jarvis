"""Exact task queries and bounded, owner-scoped immutable selection snapshots."""

import json
import time
from collections import OrderedDict
from datetime import date

from sqlalchemy import func, or_, select

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
    "parent_task_id", "is_quick_list", "quick_list_parent_id", "quick_section", "quick_order",
)


def compact(row):
    result = {key: row.get(key) for key in COMPACT_FIELDS if key in row}
    if row.get("source"):
        from .sources import compact_source
        result["source"] = compact_source(row["source"])
        if row["source"].get("read_only_reason"):
            result["source"]["read_only_reason"] = row["source"]["read_only_reason"]
    return result


def with_homes(db, owner, rows):
    """Attach the flexible record and its main-home chain; legacy project fields may be blank."""
    from .structure_models import StructureRecord

    ids = [r["id"] for r in rows]
    records = {
        r.task_id: r
        for r in db.scalars(
            select(StructureRecord).where(StructureRecord.owner_id == owner, StructureRecord.task_id.in_(ids))
        )
    } if ids else {}
    cache = {}

    def chain(parent_id):
        result, seen = [], set()
        while parent_id and parent_id not in seen and len(result) < 100:
            seen.add(parent_id)
            if parent_id not in cache:
                cache[parent_id] = db.get(StructureRecord, parent_id)
            parent = cache[parent_id]
            if not parent or parent.owner_id != owner:
                break
            result.append({"id": parent.id, "title": parent.title, "type_id": parent.type_id})
            parent_id = parent.parent_id
        return result[::-1]

    for row in rows:
        record = records.get(row["id"])
        row["record_id"] = record.id if record else None
        row["home"] = chain(record.parent_id) if record else []
        if record:
            from .record_contents import blocking
            from .structure import ensure
            row["blockers"] = blocking(db,record,ensure(db,owner))
    return rows


def under_home(owner, home_id):
    """Task IDs whose record sits anywhere below home_id, plus legacy project/area/space matches."""
    from .structure_models import StructureRecord

    tree = (
        select(StructureRecord.id, StructureRecord.task_id)
        .where(StructureRecord.owner_id == owner, StructureRecord.parent_id == home_id)
        .cte("home_tree", recursive=True)
    )
    child = select(StructureRecord.id, StructureRecord.task_id).where(
        StructureRecord.owner_id == owner, StructureRecord.parent_id == tree.c.id
    )
    tree = tree.union(child)
    established = select(StructureRecord.id).where(
        StructureRecord.owner_id==owner, StructureRecord.task_id==Task.id,
        StructureRecord.provenance["home_version"].as_integer()==1).exists()
    return or_(
        Task.id.in_(select(tree.c.task_id).where(tree.c.task_id.is_not(None))),
        (~established) & or_(Task.project_id==home_id,Task.area_id==home_id,Task.space_id==home_id),
    )


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
    if args.get("assignee"):
        from .assignees import resolve_assignee
        resolved = resolve_assignee(db, owner, args["assignee"])
        if args.get("assignee_id") and args["assignee_id"] != resolved:
            raise DomainError("INVALID_ARGUMENT", "Assignee name and ID refer to different actors.")
        args = {k:v for k,v in args.items() if k != "assignee"}
        args["assignee_id"] = resolved
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
            from .text_normalize import sql_filter

            q = q.where(sql_filter(args["query"], Task.title))
        for field in ("project_id", "space_id", "area_id", "assignee_id", "work_type", "status", "assignee"):
            if field in args:
                q = q.where(getattr(Task, field) == args[field])
        if args.get("home_id"):
            q = q.where(under_home(owner, args["home_id"]))
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
        "tasks": with_homes(db, owner, [dict(r) if args.get("detail") == "full" else compact(r) for r in page]),
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
