"""People linked to clients and projects (docs/ORGANIZATION_KNOWLEDGE_PLAN.md, phase 4).

A link says that a person works on or around a client or project. Links are non-exclusive: one person can belong
to several. The evidence:
- tasks in the bucket's subtree assigned to that person (Linear-synced ones included);
- meeting facts about the bucket, whose evidence still survives, in which that person is the subject or object,
  responsible for a commitment or owed it, or a linked mention.

`refresh_bucket` recomputes one bucket's links from that evidence. The daily overview sweep calls it
(`record_overviews.queue_due`), with no model call. Inferred links follow the evidence: they appear, update and
disappear with it. The owner can confirm a link, which keeps it even without evidence, or reject it, which hides
it for good. Either decision survives every later refresh. The owner's own actor is never linked: they are on
everything.
"""

from collections import defaultdict

from sqlalchemy import select

from .domain import DomainError, emit, owned
from .models import Actor, Task, now
from .record_contents import subtree_query
from .structure_models import PersonBucketLink, StructureRecord

BUCKET_TYPES = ("client", "project")
STATUSES = ("inferred", "confirmed", "rejected")
SAMPLES = 5


def _facts(db, owner, record):
    """{actor_id: [statement, ...]} for supported meeting facts about this bucket."""
    from .context import briefs, views
    from .context.models import ContextEntityLink
    from .context.schema import ENTITY_TYPES, EntityRef

    if record.type_id not in ENTITY_TYPES:
        return {}
    try:
        rows = views.supported(db, owner, views.scoped(db, owner, EntityRef(entity_type=record.type_id,
                                                                           entity_id=record.id)))
    except DomainError:
        return {}
    people: dict[str, list[str]] = defaultdict(list)
    by_id = {r.id: r for r in rows}
    seen = set()
    for r in rows:
        involved = [ident for kind, ident in ((r.subject_type, r.subject_id), (r.object_type, r.object_id))
                    if kind == "actor" and ident]
        # A commitment's responsible people, and who it is owed to.
        involved += [p.get("entity_id") for p in r.responsible or []
                     if isinstance(p, dict) and p.get("entity_type") == "actor" and p.get("entity_id")]
        involved += [r.recipient] if r.recipient else []
        for ident in involved:
            if (ident, r.id) not in seen:
                seen.add((ident, r.id))
                people[ident].append(briefs.describe(r))
    if by_id:
        link = ContextEntityLink
        for assertion_id, actor_id in db.execute(select(link.assertion_id, link.target_id).where(
                link.owner_id == owner, link.assertion_id.in_(list(by_id)), link.target_type == "actor",
                link.status.in_(views.LINKED))):
            if actor_id and (actor_id, assertion_id) not in seen:
                seen.add((actor_id, assertion_id))
                people[actor_id].append(briefs.describe(by_id[assertion_id]))
    return people


def _tasks(db, owner, record):
    """{actor_id: [task title, ...]} for tasks assigned in this bucket's subtree (child buckets included)."""
    people: dict[str, list[str]] = defaultdict(list)
    for actor_id, title in db.execute(
            select(Task.assignee_id, Task.title).join(StructureRecord, StructureRecord.task_id == Task.id)
            .where(StructureRecord.owner_id == owner, StructureRecord.id.in_(subtree_query(owner, record.id)),
                   StructureRecord.type_id.not_in(BUCKET_TYPES), Task.owner_id == owner,
                   Task.assignee_id.is_not(None), Task.is_template.is_(False))
            .order_by(Task.updated_at.desc())):
        people[actor_id].append(title)
    return people


def refresh_bucket(db, owner, record) -> int:
    """Recompute this bucket's inferred links. Returns how many links it now has (rejected ones excluded)."""
    from .context.views import me

    if record is None or record.owner_id != owner or record.type_id not in BUCKET_TYPES:
        return 0
    existing = {link.actor_id: link for link in db.scalars(select(PersonBucketLink).where(
        PersonBucketLink.owner_id == owner, PersonBucketLink.record_id == record.id))}
    if record.archived:
        for link in existing.values():
            if link.status == "inferred":
                db.delete(link)
        return 0
    tasks, facts = _tasks(db, owner, record), _facts(db, owner, record)
    people = set(tasks) | set(facts)
    people.discard(me(db, owner))
    valid = set(db.scalars(select(Actor.id).where(Actor.owner_id == owner, Actor.archived.is_(False),
                                                  Actor.id.in_(people)))) if people else set()
    for actor_id in valid:
        counts = {"tasks": len(tasks.get(actor_id, [])), "meeting_facts": len(facts.get(actor_id, []))}
        evidence = {"tasks": tasks.get(actor_id, [])[:SAMPLES], "meeting_facts": facts.get(actor_id, [])[:SAMPLES]}
        link = existing.pop(actor_id, None)
        if link is None:
            db.add(PersonBucketLink(owner_id=owner, actor_id=actor_id, record_id=record.id, evidence=evidence,
                                    status="inferred", **counts))
        elif (link.tasks, link.meeting_facts, link.evidence) != (counts["tasks"], counts["meeting_facts"], evidence):
            link.tasks, link.meeting_facts, link.evidence, link.updated_at = (
                counts["tasks"], counts["meeting_facts"], evidence, now())
    for link in existing.values():  # No evidence any more.
        if link.status == "inferred":
            db.delete(link)
        elif link.tasks or link.meeting_facts:
            link.tasks, link.meeting_facts, link.evidence, link.updated_at = 0, 0, {}, now()
    db.flush()
    return len(valid) + sum(1 for link in existing.values() if link.status == "confirmed")


def public(link, *, name=None, record=None) -> dict:
    out = {"actor_id": link.actor_id, "record_id": link.record_id, "status": link.status, "tasks": link.tasks,
           "meeting_facts": link.meeting_facts, "evidence": link.evidence or {}}
    if name is not None:
        out["name"] = name
    if record is not None:
        out |= {"title": record.title, "type_id": record.type_id}
    return out


def _rank(row):
    return (row["status"] != "confirmed", -(row["tasks"] + row["meeting_facts"]))


def for_actor(db, owner, actor_id) -> list[dict]:
    """The clients and projects a person is linked to (rejected links left out), strongest first."""
    rows = db.execute(select(PersonBucketLink, StructureRecord).join(
        StructureRecord, StructureRecord.id == PersonBucketLink.record_id).where(
        PersonBucketLink.owner_id == owner, PersonBucketLink.actor_id == actor_id,
        PersonBucketLink.status != "rejected", StructureRecord.archived.is_(False))).all()
    return sorted((public(link, record=record) for link, record in rows), key=lambda r: (*_rank(r), r["title"]))


def for_bucket(db, owner, record_id) -> list[dict]:
    """The people linked to a client or project (rejected links left out), strongest first."""
    rows = db.execute(select(PersonBucketLink, Actor.name).join(Actor, Actor.id == PersonBucketLink.actor_id).where(
        PersonBucketLink.owner_id == owner, PersonBucketLink.record_id == record_id,
        PersonBucketLink.status != "rejected", Actor.archived.is_(False))).all()
    return sorted((public(link, name=name) for link, name in rows), key=lambda r: (*_rank(r), r["name"]))


def decide(db, owner, actor_id, record_id, status) -> dict:
    """The owner confirms or rejects a link (creating it if needed), or returns it to "inferred"."""
    if status not in STATUSES:
        raise DomainError("INVALID_ARGUMENT", "Choose confirmed, rejected or inferred.")
    owned(db, Actor, actor_id, owner)
    record = owned(db, StructureRecord, record_id, owner)
    if record.type_id not in BUCKET_TYPES:
        raise DomainError("NOT_A_BUCKET", "People link to clients and projects.", 400)
    link = db.scalar(select(PersonBucketLink).where(PersonBucketLink.actor_id == actor_id,
                                                    PersonBucketLink.record_id == record_id).with_for_update())
    if link is None:
        link = PersonBucketLink(owner_id=owner, actor_id=actor_id, record_id=record_id, tasks=0, meeting_facts=0,
                                evidence={}, status=status)
        db.add(link)
    link.status, link.updated_at = status, now()
    db.flush()
    if status == "inferred":
        refresh_bucket(db, owner, record)  # Back to following the evidence (gone when there is none).
        link = db.scalar(select(PersonBucketLink).where(PersonBucketLink.actor_id == actor_id,
                                                        PersonBucketLink.record_id == record_id))
    emit(db, owner, "record.people", record_id)
    return public(link, record=record) if link else {"actor_id": actor_id, "record_id": record_id, "status": None}
