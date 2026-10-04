"""Bounded structure skeleton for the visual Atlas: ids, titles, homes, state and links only.

Details still load through the record APIs. The skeleton is breadth-first, so a truncated
workspace keeps its upper levels coherent instead of losing whole branches at random.
"""

from collections import defaultdict, deque
from sqlalchemy import select
from .models import Note, Task, now
from .structure_models import StructureLink, StructureRecord

LIMIT = 5000


def _breadth_first(pairs, limit):
    """Order (id, parent_id, sort_order) rows root-first; orphans of excluded homes become roots."""
    ids = {identity for identity, _, _ in pairs}
    children, roots = defaultdict(list), []
    for identity, parent, order in pairs:
        (children[parent] if parent in ids else roots).append((order, identity))
    queue = deque(sorted(roots))
    depth = {identity: 0 for _, identity in roots}
    chosen, seen = [], set()
    while queue and len(chosen) < limit:
        _, identity = queue.popleft()
        if identity in seen:
            continue
        seen.add(identity)
        chosen.append(identity)
        for item in sorted(children.get(identity, ())):
            depth[item[1]] = depth[identity] + 1
            queue.append(item)
    return chosen, depth


def skeleton(db, owner, *, limit=LIMIT, withhold=frozenset()):
    """Everything the Atlas needs to lay out the active workspace, capped at ``limit`` records.

    ``withhold`` names core kinds ("task", "note") whose dates a bot key may not read.
    """
    from . import structure

    schema = structure.ensure(db, owner)
    if structure.core_drift(db, owner):
        structure.sync_core_records(db, owner, schema)
    types = {t["id"]: t for t in schema.definition["types"]}
    from .record_reviews import cadence, is_due
    instant = now()
    hidden_task = select(Task.id).where(Task.owner_id == owner, (Task.archived.is_(True)) | (Task.is_template.is_(True)))
    pairs = list(db.execute(
        select(StructureRecord.id, StructureRecord.parent_id, StructureRecord.sort_order).where(
            StructureRecord.owner_id == owner,
            StructureRecord.archived.is_(False),
            StructureRecord.type_id.in_(list(types)),
            (StructureRecord.task_id.is_(None)) | (StructureRecord.task_id.notin_(hidden_task)),
        )
    ))
    chosen, depth = _breadth_first([(a, b, c or 0) for a, b, c in pairs], limit)
    included = set(chosen)
    rows = {}
    for start in range(0, len(chosen), 1000):
        batch = chosen[start:start + 1000]
        for row, task, note in db.execute(
            select(StructureRecord, Task, Note)
            .outerjoin(Task, Task.id == StructureRecord.task_id)
            .outerjoin(Note, Note.id == StructureRecord.note_id)
            .where(StructureRecord.owner_id == owner, StructureRecord.id.in_(batch))
        ):
            rows[row.id] = (row, task, note)
    with_children = {parent for _, parent, _ in pairs if parent in included}
    records = []
    for identity in chosen:
        row, task, note = rows[identity]
        t = types[row.type_id]
        if task:
            status = task.status
        else:
            status = next((s["meaning"] for s in t["statuses"] if s["id"] == row.status_id), None)
        dated = task and "task" not in withhold
        records.append({
            "id": row.id,
            "title": (task.title if task else note.title if note else row.title),
            "type_id": row.type_id,
            "parent_id": row.parent_id if row.parent_id in included else None,
            "depth": depth.get(row.id, 0),
            "revision": row.revision,
            "sort_order": row.sort_order or 0,
            "opens_as": structure.opens_as(t, row.id in with_children),
            "status_meaning": status,
            "work": bool(row.task_id) and "work" in t["capabilities"],
            "due_date": task.due_date.isoformat() if dated and task.due_date else None,
            "planned_date": task.planned_date.isoformat() if dated and task.planned_date else None,
            "review_due": is_due(row, cadence(t), instant),
            "next_review_at": row.next_review_at.isoformat() if cadence(t) and row.next_review_at and not row.review_paused else None,
        })
    behaviors = {r["id"]: r for r in schema.definition["relationships"]}
    links = []
    if included:
        for link in db.scalars(select(StructureLink).where(StructureLink.owner_id == owner)):
            relation = behaviors.get(link.relationship_id)
            if link.source_id in included and link.target_id in included and relation and not relation["archived"]:
                links.append({
                    "id": link.id,
                    "source_id": link.source_id,
                    "target_id": link.target_id,
                    "relationship_id": link.relationship_id,
                    "label": relation["name"],
                    "behavior": relation.get("behavior") or "related",
                })
    return {
        "schema_revision": schema.revision,
        "types": [
            {k: t.get(k) for k in ("id", "name", "plural", "capabilities", "parent_types", "opens_as", "review", "archived")}
            for t in schema.definition["types"]
        ],
        "records": records,
        "links": links,
        "total": len(pairs),
        "truncated": len(pairs) > len(records),
        "limit": limit,
    }
