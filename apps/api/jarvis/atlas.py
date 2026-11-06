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


def visibility_rows(db, owner, *, archived=False, show_finished=False, status="all"):
    """Visible records and minimal ancestor context, before any pagination/breadth limit.

    Context retains real containment, including archived parents of unfinished work.
    Templates never become context. Non-task organizational outcomes remain visible.
    """
    from . import structure
    schema = structure.ensure(db, owner)
    containers = {t["id"] for t in schema.definition["types"] if structure.opens_as(t) == "container"}
    rows = list(db.execute(
        select(StructureRecord.id, StructureRecord.parent_id, StructureRecord.sort_order,
               StructureRecord.archived, Task.id, Task.status, Task.archived, Task.is_template, StructureRecord.type_id)
        .outerjoin(Task, Task.id == StructureRecord.task_id)
        .where(StructureRecord.owner_id == owner)
    ))
    eligible = {r[0]: r for r in rows if not r[7]}
    visible = {r[0] for r in eligible.values()
               if bool(r[3] or r[6]) == archived
               and (archived or show_finished or not r[4] or r[8] in containers or r[5] not in {"completed", "cancelled"})}
    if not archived and status not in {"active", "all"}:
        visible = {identity for identity in visible if not eligible[identity][4] or eligible[identity][8] in containers or eligible[identity][5] == status}
    context = set()
    for identity in visible:
        parent = eligible[identity][1]
        seen = {identity}
        while parent in eligible and parent not in seen:
            seen.add(parent)
            if parent not in visible:
                context.add(parent)
            parent = eligible[parent][1]
    return eligible, visible, context


def progress(eligible):
    """Work done, open and total under each record, over all its descendants with finished work included
    whatever the Atlas filter hides: the same definition (and so the same numbers) as Browse."""
    from .record_contents import FINISHED, counts_as_work
    out = defaultdict(lambda: [0, 0, 0])
    for identity, r in eligible.items():
        if not r[4] or not counts_as_work(archived=bool(r[3] or r[6]), status=r[5], template=bool(r[7])):
            continue
        done, open_ = r[5] == "completed", r[5] not in FINISHED
        parent, seen = r[1], {identity}
        while parent in eligible and parent not in seen:
            seen.add(parent)
            counts = out[parent]
            counts[0] += done
            counts[1] += open_
            counts[2] += 1
            parent = eligible[parent][1]
    return out


def skeleton(db, owner, *, limit=LIMIT, withhold=frozenset(), archived=False, show_finished=False, status="all"):
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
    eligible, visible, context = visibility_rows(db, owner, archived=archived, show_finished=show_finished, status=status)
    work = progress(eligible)
    eligible = {identity: r for identity, r in eligible.items() if r[8] in types}
    visible &= eligible.keys()
    pairs = [(r[0], r[1], r[2] or 0) for r in eligible.values() if r[0] in visible or r[0] in context]
    ordered, depth = _breadth_first(pairs, len(pairs))
    chosen_visible = [identity for identity in ordered if identity in visible][:limit]
    included = set(chosen_visible)
    for identity in chosen_visible:
        parent = eligible[identity][1]
        seen = {identity}
        while parent in eligible and parent not in seen:
            seen.add(parent)
            included.add(parent)
            parent = eligible[parent][1]
    chosen = [identity for identity in ordered if identity in included]
    context = included - set(chosen_visible)
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
            "opens_as": "container" if row.id in context else structure.opens_as(t, row.id in with_children),
            "context_only": row.id in context,
            "archived": bool(row.archived or (task and task.archived)),
            "status_meaning": status,
            "work": row.id not in context and bool(row.task_id) and "work" in t["capabilities"],
            "due_date": task.due_date.isoformat() if dated and task.due_date else None,
            "planned_date": task.planned_date.isoformat() if dated and task.planned_date else None,
            "review_due": row.id not in context and is_due(row, cadence(t), instant),
            "next_review_at": row.next_review_at.isoformat() if cadence(t) and row.next_review_at and not row.review_paused else None,
            **dict(zip(("work_done", "work_open", "work_total"), work.get(row.id, (0, 0, 0)))),
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
        "total": len(visible),
        "context_count": len(context),
        "truncated": len(visible) > len(chosen_visible),
        "limit": limit,
    }
