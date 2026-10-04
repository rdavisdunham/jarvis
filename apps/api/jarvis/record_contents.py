"""Scoped containment reads and atomic, revision-guarded contents operations."""

from collections import Counter
from sqlalchemy import select, func, or_, and_
from .domain import DomainError, owned, check_revision
from .models import Task
from .structure_models import StructureRecord, StructureLink
from .structure_schema import Input


from .structure_schema import ContentsPlan, ContentsApply, ContentsRestore


def subtree_query(owner, identity, *, include_root=False):
    # UNION (not UNION ALL) bounds even old corrupt cycles.
    tree = (
        select(StructureRecord.id)
        .where(StructureRecord.owner_id == owner, StructureRecord.id == identity)
        .cte(recursive=True)
    )
    tree = tree.union(
        select(StructureRecord.id)
        .join(tree, StructureRecord.parent_id == tree.c.id)
        .where(StructureRecord.owner_id == owner)
    )
    q = select(tree.c.id)
    return q if include_root else q.where(tree.c.id != identity)


def rows_under(db, owner, identity):
    return list(
        db.scalars(
            select(StructureRecord)
            .where(StructureRecord.owner_id == owner, StructureRecord.id.in_(subtree_query(owner, identity)))
            .order_by(StructureRecord.id)
        )
    )


def summary(db, row, schema):
    """Unique descendants, separate from related links; never sum parent task effort."""
    from .structure import record_type

    descendants = rows_under(db, row.owner_id, row.id)
    active = [r for r in descendants if not r.archived]
    work = [r for r in active if r.task_id]
    tasks = {
        t.id: t
        for t in db.scalars(
            select(Task).where(Task.owner_id == row.owner_id, Task.id.in_([r.task_id for r in work]))
        )
    }
    done = sum(tasks[r.task_id].status == "completed" for r in work)
    open_count = sum(tasks[r.task_id].status not in {"completed", "cancelled"} for r in work)
    starts, ends = [], []
    for r in active:
        t = record_type(schema, r.type_id, archived=True)
        for f in t["fields"]:
            value = (
                getattr(tasks[r.task_id], f["binding"], None)
                if r.task_id in tasks and f.get("binding") in {"due_date", "planned_date"}
                else r.values.get(f["id"])
            )
            if value and f.get("binding") in {"start_date", "target_date", "due_date", "planned_date"}:
                v = str(value)
                (starts if f["binding"] in {"start_date", "planned_date"} else ends).append(v)
    blockers = blocking(db, row, schema)
    return {
        "direct": sum(r.parent_id == row.id for r in active),
        "descendants": len(active),
        "work_total": len(work),
        "work_done": done,
        "work_open": open_count,
        "ready_to_complete": bool(work) and open_count == 0,
        "child_start": min(starts + ends) if starts or ends else None,
        "child_end": max(starts + ends) if starts or ends else None,
        "blockers": blockers,
    }


def blocking(db, row, schema):
    ids = [
        r["id"]
        for r in schema.definition["relationships"]
        if r.get("behavior") == "blocks" and not r["archived"]
    ]
    rows = list(
        db.scalars(
            select(StructureRecord)
            .join(StructureLink, StructureLink.source_id == StructureRecord.id)
            .where(
                StructureLink.owner_id == row.owner_id,
                StructureLink.target_id == row.id,
                StructureLink.relationship_id.in_(ids),
                StructureRecord.archived.is_(False),
            )
        )
    )
    result = []
    for blocker in rows:
        task = db.get(Task, blocker.task_id) if blocker.task_id else None
        if task and task.status not in {"completed", "cancelled"}:
            result.append({"id": blocker.id, "title": task.title, "status": task.status})
    return result


def validate_block(db, owner, schema, source, target):
    if source.id == target.id:
        raise DomainError("DEPENDENCY_CYCLE", "A record cannot block itself.")
    if not source.task_id or not target.task_id:
        raise DomainError("INVALID_DEPENDENCY", "Both ends of a blocker must be completable records.")
    ids = [
        r["id"]
        for r in schema.definition["relationships"]
        if r.get("behavior") == "blocks" and not r["archived"]
    ]
    edges = list(
        db.execute(
            select(StructureLink.source_id, StructureLink.target_id).where(
                StructureLink.owner_id == owner, StructureLink.relationship_id.in_(ids)
            )
        )
    )
    seen = set()
    todo = [target.id]
    while todo:
        current = todo.pop()
        if current == source.id:
            raise DomainError("DEPENDENCY_CYCLE", "That link would create a circular dependency.")
        if current in seen:
            continue
        seen.add(current)
        todo.extend(b for a, b in edges if a == current)


def browse(
    db,
    owner,
    *,
    parent_id=None,
    scope="children",
    section="all",
    archived=False,
    status="all",
    query="",
    limit=50,
    offset=0,
):
    from . import structure

    schema = structure.ensure(db, owner)
    if structure.core_drift(db, owner):
        structure.sync_core_records(db, owner, schema)
    parent = owned(db, StructureRecord, parent_id, owner) if parent_id else None
    q = select(StructureRecord).where(StructureRecord.owner_id == owner, StructureRecord.archived == archived)
    if scope == "related":
        if not parent:
            raise DomainError("INVALID_ARGUMENT", "Related records need a container.")
        linked = (
            select(StructureLink.target_id)
            .where(StructureLink.owner_id == owner, StructureLink.source_id == parent.id)
            .union(
                select(StructureLink.source_id).where(
                    StructureLink.owner_id == owner, StructureLink.target_id == parent.id
                )
            )
        )
        q = q.where(StructureRecord.id.in_(linked))
    elif scope == "subtree" and parent:
        q = q.where(StructureRecord.id.in_(subtree_query(owner, parent.id)))
    else:
        q = q.where(StructureRecord.parent_id == parent_id)
    if section in {"work", "content"}:
        types = [t["id"] for t in schema.definition["types"] if section in t["capabilities"]]
        q = q.where(StructureRecord.type_id.in_(types))
    elif section == "groups":
        from sqlalchemy.orm import aliased

        child = aliased(StructureRecord)
        types = schema.definition["types"]
        containers = [t["id"] for t in types if structure.opens_as(t) == "container"]
        inferred = [t["id"] for t in types if t.get("opens_as", "auto") == "auto"]
        q = q.where(
            or_(
                StructureRecord.type_id.in_(containers),
                and_(
                    StructureRecord.type_id.in_(inferred),
                    select(child.id)
                    .where(
                        child.owner_id == owner, child.parent_id == StructureRecord.id, child.archived == archived
                    )
                    .exists(),
                ),
            )
        )
    if status != "all":
        task_filter = select(Task.id).where(Task.owner_id == owner)
        task_filter = task_filter.where(
            Task.status.notin_(["completed", "cancelled"]) if status == "active" else Task.status == status
        )
        q = q.where(or_(StructureRecord.task_id.is_(None), StructureRecord.task_id.in_(task_filter)))
    if query:
        from .text_normalize import sql_filter

        q = q.where(sql_filter(query, StructureRecord.title))
    total = db.scalar(select(func.count()).select_from(q.subquery()))
    found = list(
        db.scalars(q.order_by(StructureRecord.sort_order, StructureRecord.id).offset(offset).limit(limit))
    )
    items = []
    for row in found:
        value = structure.data(db, row, schema)
        value["contents"] = summary(db, row, schema)
        items.append(value)
    if parent:
        parent_data = structure.data(db, parent, schema)
        parent_data["contents"] = summary(db, parent, schema)
    return {
        "parent": parent_data if parent else None,
        "items": items,
        "total": total,
        "next_offset": offset + limit if offset + limit < total else None,
        "scope": scope,
        "section": section,
        "schema_revision": schema.revision,
    }


def plan(db, owner, args):
    from . import structure

    schema = structure.ensure(db, owner)
    structure.assert_schema(schema, args.schema_revision)
    root = owned(db, StructureRecord, args.record_id, owner)
    structure.reconcile_core(db, root, schema)
    check_revision(root, args.expected_revision)
    if args.operation == "archive" and root.archived:
        raise DomainError("ALREADY_ARCHIVED", "This record is already archived.")
    descendants = rows_under(db, owner, root.id)
    for row in descendants:
        structure.reconcile_core(db, row, schema)
    children = [r for r in descendants if r.parent_id == root.id]
    destination = args.parent_id if args.operation == "move" else root.parent_id
    if args.operation == "archive" and args.parent_id is not None:
        raise DomainError("INVALID_ARGUMENT", "Archiving keeps or promotes children to the old home.")
    issues = []
    try:
        if args.operation == "move":
            structure.validate_parent(
                db, owner, root, structure.record_type(schema, root.type_id, archived=True), destination
            )
        if args.mode == "item":
            for child in children:
                structure.validate_parent(
                    db,
                    owner,
                    child,
                    structure.record_type(schema, child.type_id, archived=True),
                    root.parent_id,
                )
    except DomainError as exc:
        issues.append(exc.message)
    changed = (
        [root, *children]
        if args.mode == "item"
        else [r for r in [root, *descendants] if not r.archived]
        if args.operation == "archive"
        else [root]
    )
    providers = []
    for row in [root, *descendants]:
        task = db.get(Task, row.task_id) if row.task_id else None
        if task and task.external:
            providers.append(
                {"id": row.id, "title": row.title, "provider": task.external.get("provider", "external")}
            )
    chain = structure.parent_chain(db, owner, destination)
    stamp = structure.fingerprint(
        {
            "request": args.model_dump(exclude={"preview_hash"}),
            "schema": schema.revision,
            "rows": [
                (r.id, r.revision, r.parent_id, r.archived, r.provenance.get("core_revisions"))
                for r in [root, *descendants, *chain]
            ],
        }
    )
    return {
        "preview_hash": stamp,
        "record_id": root.id,
        "title": root.title,
        "operation": args.operation,
        "mode": args.mode,
        "affected_count": len(changed),
        "descendant_count": len(descendants),
        "promoted_count": len(children) if args.mode == "item" else 0,
        "types": dict(Counter(r.type_id for r in changed)),
        "providers": providers,
        "source_effect": "Local filing/archive only. Connected source records are not deleted or reparented.",
        "issues": issues,
        "schema_revision": schema.revision,
    }


def apply(db, owner, args, command_id):
    from . import structure
    from .structure_schema import RecordUpdate

    preview = plan(db, owner, args)
    if preview["preview_hash"] != args.preview_hash:
        raise DomainError("STALE_CONTENTS", "Contents changed. Review a fresh preview.", 409)
    if preview["issues"]:
        raise DomainError("INVALID_PARENT", preview["issues"][0])
    schema = structure.ensure(db, owner)
    root = owned(db, StructureRecord, args.record_id, owner)
    descendants = rows_under(db, owner, root.id)
    changed = []
    # Promote children first, preserving each child's own subtree.
    if args.mode == "item":
        for child in descendants:
            if child.parent_id == root.id:
                changed.append(
                    structure.mutate(
                        db,
                        owner,
                        "record.update",
                        RecordUpdate(
                            record_id=child.id,
                            expected_revision=child.revision,
                            schema_revision=schema.revision,
                            parent_id=root.parent_id,
                        ),
                        command_id,
                    )
                )
    targets = [root, *descendants] if args.operation == "archive" and args.mode == "subtree" else [root]
    for row in targets:
        if args.operation == "archive" and row.archived:
            continue
        patch = {"parent_id": args.parent_id} if args.operation == "move" else {"archived": True}
        changed.append(
            structure.mutate(
                db,
                owner,
                "record.update",
                RecordUpdate(
                    record_id=row.id, expected_revision=row.revision, schema_revision=schema.revision, **patch
                ),
                command_id,
            )
        )
    changed_ids = [r["id"] for r in changed]
    return {
        **preview,
        "id": root.id,
        "changed_ids": changed_ids,
        "applied": True,
        "restore_guard": guard(db, owner, changed_ids),
    }


def restore(db, owner, args, command_id):
    """Restore a grouped operation using the same guarded inverses as action cards."""
    from .models import ActionChange, Command
    from .action_history import inverse
    from .domain import execute
    from .access import actor

    receipt = db.get(Command, (owner, args.source_command_id))
    if not receipt or receipt.account_id != actor(db, owner):
        raise DomainError("NOT_FOUND", "This action is not available.", 404)
    rows = list(
        db.scalars(
            select(ActionChange)
            .where(ActionChange.owner_id == owner, ActionChange.command_id == args.source_command_id)
            .order_by(ActionChange.created_at, ActionChange.id)
        )
    )
    if rows and all(r.tool in {"record.mark_reviewed", "record.review"} for r in rows):
        # The Undo toast after Mark reviewed: each review change carries its own date guard.
        return undo_reviews(db, owner, rows, command_id)
    check_restore_guard(db, owner, receipt)
    if rows and all(r.tool == "record.instantiate" for r in rows):
        from .record_templates import restore as undo_template
        return undo_template(db, owner, receipt, rows, command_id)
    if not rows or any(r.tool != "record.contents" for r in rows):
        raise DomainError("INVALID_ARGUMENT", "Choose a contents operation to restore.")
    actions = []
    for row in rows:
        action, reason = inverse(db, row, lock=True, group=False)
        if not action:
            raise DomainError("REVISION_CONFLICT", reason, 409)
        actions.append((row, action))
    from .structure import parent_chain

    # Restore containers before children; all checks above happen before any edit.
    actions.sort(
        key=lambda item: (
            len(parent_chain(db, owner, item[1][1].get("parent_id")))
            if item[0].entity_kind == "record"
            else 100
        )
    )
    results = []
    for row, (tool, values) in actions:
        # Earlier restores may refresh compatibility caches; explicit changed fields
        # were checked as a group, so use the current revision under the workspace lock.
        from .action_history import MODELS

        current = db.get(MODELS[row.entity_kind], row.entity_id)
        values["expected_revision"] = current.revision
        from uuid import uuid5, NAMESPACE_URL

        child_command = str(uuid5(NAMESPACE_URL, command_id + ":" + str(len(results))))
        results.append(execute(db, owner, child_command, tool, values)["data"])
        row.reverted_by = command_id
    return {"restored": True, "changed_count": len(results)}


def undo_reviews(db, owner, rows, command_id):
    from uuid import NAMESPACE_URL, uuid5
    from .action_history import inverse
    from .domain import execute

    actions = []
    for row in rows:
        action, reason = inverse(db, row, lock=True, group=False)
        if not action:
            raise DomainError("REVISION_CONFLICT", reason, 409)
        actions.append((row, action))
    for i, (row, (tool, values)) in enumerate(actions):
        execute(db, owner, str(uuid5(NAMESPACE_URL, command_id + ":" + str(i))), tool, values)
        row.reverted_by = command_id
    return {"restored": True, "changed_count": len(actions)}


def guard(db, owner, identities):
    from .structure import ensure, reconcile_core, fingerprint

    schema = ensure(db, owner)
    ids = set(identities)
    for identity in identities:
        ids.update(db.scalars(subtree_query(owner, identity)))
    rows = list(
        db.scalars(
            select(StructureRecord)
            .where(StructureRecord.owner_id == owner, StructureRecord.id.in_(ids))
            .order_by(StructureRecord.id)
        )
    )
    for row in rows:
        reconcile_core(db, row, schema)
    return fingerprint([(r.id, r.revision, r.parent_id, r.archived) for r in rows])


def check_restore_guard(db, owner, receipt):
    data = receipt.result.get("data", {}) if receipt else {}
    if not data.get("restore_guard") or data["restore_guard"] != guard(db, owner, data["changed_ids"]):
        raise DomainError(
            "REVISION_CONFLICT", "The contents changed after this action. Review them before reverting.", 409
        )
