"""Derived context views: open loops, meeting preparation, person pages and the context changes feed.

Everything here is computed from canonical rows at read time; nothing is stored and nothing is written.

- Open loops come from live, asserted commitment assertions with surviving evidence (plan section 3). A loop is
  `own` when the user's actor is responsible, `waiting_for` when someone else promised it to the user (or to
  nobody in particular, in the user's own meeting), `between_others` when the recipient is another person, and
  `collective` when responsibility is "we". Linked tasks are read from `tasks`; the assertion keeps no status of
  its own. A completed linked task means the work it covers is done (`task_state = "covered"`), never that the
  broader promise is fulfilled: only completion evidence on the assertion changes `status`.
- Meeting prep finds the next calendar occurrence involving an entity (or a given event), resolves attendees to
  actors through exact identities, and gathers open loops, decisions and questions that involve them.
- The person view assembles what a person page shows: identities, current context, projects, loops, tasks,
  meetings, links awaiting correction and merges.
- `changes` mirrors `changes_list`: a commit-ordered cursor over `context.changed` events, each resolved to the
  current state of the item it names, with tombstones (no content) for retracted assertions and deleted sources.

Callers authorize; every function is owner-scoped and returns JSON-ready dicts.
"""

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy import and_, func, or_, select
from sqlalchemy.dialects.postgresql import array

from ..domain import DomainError, preferences, zone
from ..models import (
    Actor,
    Area,
    Event,
    Goal,
    GoogleCalendar,
    GoogleCalendarEvent,
    LinearIssue,
    Project,
    SharedWorkspace,
    Task,
    WorkspaceMember,
    now,
)
from ..structure_models import StructureRecord
from . import evidence as survival
from . import identity, temporal
from .models import (
    ContextActorIdentity,
    ContextActorMerge,
    ContextAssertion,
    ContextBrief,
    ContextEntityLink,
    ContextEvidence,
    ContextProposal,
    ContextSegment,
    ContextSource,
    ContextSourceRevision,
)
from .schema import EntityRef, ScopePredicate

MAX_ASSERTIONS = 300
HORIZON_DAYS = 14
LINKED = ("exact", "inferred", "confirmed")
COLLECTIVE = {"we", "us", "our team", "the team", "team", "everyone", "everybody", "all of us", "both of us",
              "the group"}
NOT_COMMITTED = ("hypothetical", "negated", "question", "proposed")
CHANGE_KIND = "context.changed"


def iso(value) -> str | None:
    return value.isoformat() if value else None


def as_ref(entity) -> EntityRef | None:
    """An `EntityRef`, an `(entity_type, entity_id)` pair, or None."""
    if entity is None or isinstance(entity, EntityRef):
        return entity
    entity_type, entity_id = entity
    return EntityRef(entity_type=entity_type, entity_id=entity_id)


def me(db, owner: str) -> str | None:
    """The user's own actor, read-only (`identity.owner_actor` without creating one)."""
    workspace = db.get(SharedWorkspace, owner)
    if workspace is not None:
        member = db.get(WorkspaceMember, (owner, workspace.creator_id))
        if member is not None and member.actor_id:
            return member.actor_id
    return db.scalar(select(Actor.id).where(Actor.owner_id == owner, Actor.name == identity.OWNER_ACTOR_NAME,
                                            Actor.kind == "person")
                     .order_by(Actor.archived, Actor.created_at, Actor.id).limit(1))


# ---------------------------------------------------------------- entity lookup (read-only)


def entity_name(db, owner: str, entity_type: str, entity_id: str) -> str | None:
    """Display name of an Eridani entity the owner holds, or None when it does not exist for them."""
    named = {"actor": (Actor, "name"), "project": (Project, "name"), "area": (Area, "name"),
             "goal": (Goal, "name"), "task": (Task, "title")}
    if entity_type in named:
        model, attribute = named[entity_type]
        row = db.get(model, entity_id)
        if row is not None and row.owner_id == owner:
            return getattr(row, attribute)
    if entity_type == "calendar_event":
        event = calendar_event(db, owner, entity_id)
        return event.payload.get("summary") or "Untitled event" if event else None
    if entity_type == "linear_issue":
        issue = db.get(LinearIssue, entity_id)
        return (issue.snapshot or {}).get("title") if issue is not None and issue.owner_id == owner else None
    record = db.get(StructureRecord, entity_id)
    if record is not None and record.owner_id == owner and record.type_id == entity_type:
        return record.title
    return None


def actor_names(db, owner: str, ids: Iterable[str | None]) -> dict[str, str]:
    ids = sorted({i for i in ids if i})
    if not ids:
        return {}
    return dict(db.execute(select(Actor.id, Actor.name).where(Actor.owner_id == owner, Actor.id.in_(ids))).all())


def calendar_event(db, owner: str, event_id: str) -> GoogleCalendarEvent | None:
    """A cached Google event of the owner's, by Eridani ID or provider ID."""
    return db.scalar(select(GoogleCalendarEvent).join(GoogleCalendar, GoogleCalendar.id == GoogleCalendarEvent.calendar_id)
                     .where(GoogleCalendar.owner_id == owner,
                            or_(GoogleCalendarEvent.id == event_id, GoogleCalendarEvent.provider_id == event_id))
                     .order_by(GoogleCalendarEvent.id).limit(1))


# ---------------------------------------------------------------- assertions in scope


def scoped(db, owner: str, entity: EntityRef | None, *, kinds: Sequence[str] = (), at: datetime | None = None,
           include_superseded: bool = False, limit: int = MAX_ASSERTIONS) -> list[ContextAssertion]:
    """Current assertions about an entity (subject, object or link; for actors also commitments they are
    responsible for or owed), newest effective first. Superseded rows only when asked for."""
    at = at or now()
    a = ContextAssertion
    flags = {"owner_id": owner, "kinds": list(kinds), "include_superseded": include_superseded}
    clauses = temporal.scope_clauses(ScopePredicate(**flags, entities=[entity] if entity else []), at=at)
    where = and_(*clauses)
    if entity is not None and entity.entity_type == "actor":
        party = or_(a.responsible.contains([{"entity_type": "actor", "entity_id": entity.entity_id}]),
                    a.recipient == entity.entity_id)
        where = or_(where, and_(*temporal.scope_clauses(ScopePredicate(**flags), at=at), party))
    return list(db.scalars(select(a).where(where).order_by(a.valid_from.desc().nulls_last(), a.id).limit(limit)))


def supported(db, owner: str, rows: Sequence[ContextAssertion]) -> list[ContextAssertion]:
    """The rows whose evidence survives (`evidence.support_status == "supported"`). A row with a live,
    unconditional independent quote takes the fast path; the rest go through the survival rule."""
    if not rows:
        return []
    e = ContextEvidence
    spans = db.execute(select(e.assertion_id, e.source_id, e.derivation).where(
        e.owner_id == owner, e.assertion_id.in_([r.id for r in rows]), e.role == "support",
        e.origin == "independent")).all()
    deleted = survival.deleted_sources(db, owner, {s for _, s, _ in spans})
    fast = {aid for aid, sid, derivation in spans
            if sid not in deleted and not (derivation or {}).get("source_deleted") and not (derivation or {}).get("all_of")}
    return [r for r in rows if r.id in fast or survival.support_status(db, owner, r) == "supported"]


def quotes(db, owner: str, assertion_ids: Iterable[str], per: int = 2) -> dict[str, list[dict]]:
    """Up to `per` live supporting quotes per assertion, as citations."""
    ids = sorted(set(assertion_ids))
    if not ids:
        return {}
    e = ContextEvidence
    rows = list(db.scalars(select(e).where(e.owner_id == owner, e.assertion_id.in_(ids), e.quote != "",
                                           e.role == "support").order_by(e.created_at, e.id)))
    deleted = survival.deleted_sources(db, owner, {r.source_id for r in rows})
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.source_id in deleted or len(out[r.assertion_id]) >= per:
            continue
        out[r.assertion_id].append({"evidence_id": r.id, "source_id": r.source_id, "revision_id": r.revision_id,
                                    "segment_id": r.segment_id, "quote": r.quote})
    return out


def linked_tasks(db, owner: str, rows: Sequence[ContextAssertion]) -> dict[str, list[Task]]:
    """Tasks linked to each assertion: by subject or object, by a live entity link (to the task or to a synced
    Linear issue), or by a published proposal citing the assertion."""
    ids = {r.id for r in rows}
    if not ids:
        return {}
    pairs: set[tuple[str, str]] = set()
    issues: set[tuple[str, str]] = set()
    for r in rows:
        for kind, ident in ((r.subject_type, r.subject_id), (r.object_type, r.object_id)):
            if ident and kind == "task":
                pairs.add((r.id, ident))
            elif ident and kind == "linear_issue":
                issues.add((r.id, ident))
    link = ContextEntityLink
    for assertion_id, kind, target in db.execute(select(link.assertion_id, link.target_type, link.target_id).where(
            link.owner_id == owner, link.assertion_id.in_(ids), link.target_type.in_(("task", "linear_issue")),
            link.target_id.is_not(None), link.status.in_(LINKED))):
        (pairs if kind == "task" else issues).add((assertion_id, target))
    if issues:
        mapped = dict(db.execute(select(LinearIssue.id, LinearIssue.task_id).where(
            LinearIssue.owner_id == owner, LinearIssue.id.in_({i for _, i in issues}))).all())
        pairs |= {(a, mapped[i]) for a, i in issues if i in mapped}
    p = ContextProposal
    for task_id, cited in db.execute(select(p.task_id, p.assertion_ids).where(
            p.owner_id == owner, p.task_id.is_not(None), p.assertion_ids.has_any(array(sorted(ids))))):
        pairs |= {(a, task_id) for a in cited if a in ids}
    tasks = {t.id: t for t in db.scalars(select(Task).where(Task.owner_id == owner,
                                                            Task.id.in_({t for _, t in pairs})))} if pairs else {}
    out: dict[str, list[Task]] = defaultdict(list)
    for assertion_id, task_id in sorted(pairs):
        if task_id in tasks:
            out[assertion_id].append(tasks[task_id])
    return out


def task_view(task: Task) -> dict:
    return {"id": task.id, "revision": task.revision, "title": task.title, "status": task.status,
            "due_date": iso(task.due_date), "completed_at": iso(task.completed_at), "archived": task.archived,
            "assignee_id": task.assignee_id}


# ---------------------------------------------------------------- open loops


def _party(item: dict, names: dict[str, str], me_id: str | None) -> dict:
    actor_id = item.get("entity_id") if item.get("entity_type") == "actor" else None
    mention = item.get("mention")
    return {"actor_id": actor_id, "name": names.get(actor_id) or mention, "mention": mention,
            "resolved": actor_id is not None, "is_me": actor_id is not None and actor_id == me_id,
            "collective": actor_id is None and temporal.normalized(mention or "") in COLLECTIVE}


def _bucket(parties: list[dict], recipient: str | None, me_id: str | None) -> str:
    if any(p["is_me"] for p in parties):
        return "own"
    if any(p["collective"] for p in parties):
        return "collective"
    if recipient and recipient != me_id:
        return "between_others"
    return "waiting_for"


def _task_state(tasks: list[Task]) -> str:
    if not tasks:
        return "none"
    if any(t.status == "completed" for t in tasks):
        return "covered"
    if all(t.status == "cancelled" for t in tasks):
        return "cancelled"
    return "open"


def loops(db, owner: str, rows: Sequence[ContextAssertion]) -> list[dict]:
    """Loop views of commitment rows (already scoped and supported), soonest due first."""
    me_id = me(db, owner)
    tasks = linked_tasks(db, owner, rows)
    cited = quotes(db, owner, [r.id for r in rows])
    names = actor_names(db, owner, [*(i.get("entity_id") for r in rows for i in r.responsible),
                                    *(r.recipient for r in rows)])
    out = []
    for r in rows:
        parties = [_party(item, names, me_id) for item in r.responsible]
        bucket = _bucket(parties, r.recipient, me_id)
        linked = tasks.get(r.id, [])
        out.append({
            "assertion_id": r.id, "revision": r.current_version, "statement": r.statement,
            "lifecycle": r.lifecycle, "inferred": not r.explicit, "bucket": bucket,
            "waiting_for": bucket == "waiting_for", "responsible": parties,
            "recipient": {"actor_id": r.recipient, "name": names.get(r.recipient), "is_me": r.recipient == me_id}
            if r.recipient else None,
            "due_at": iso(r.due_at), "due_precision": r.due_precision,
            "status": "completion_observed" if r.completion_evidence else "open",
            "task_state": _task_state(linked), "tasks": [task_view(t) for t in linked],
            "subject": {"entity_type": r.subject_type, "entity_id": r.subject_id},
            "valid_from": iso(r.valid_from), "recorded_at": iso(r.updated_at), "evidence": cited.get(r.id, []),
        })
    out.sort(key=lambda loop: (loop["due_at"] is None, loop["due_at"] or "", -_stamp(loop["valid_from"]),
                               loop["assertion_id"]))
    return out


def _stamp(value: str | None) -> float:
    return datetime.fromisoformat(value).timestamp() if value else 0.0


def open_loops(db, owner: str, *, entity=None, owner_actor: str | None = None, waiting_for: bool | None = None,
               include_completed: bool = False, at: datetime | None = None) -> dict:
    """Open commitments about an entity (or all), optionally only those `owner_actor` is responsible for
    (`"me"` means the user's actor) and only (`True`) or never (`False`) waiting-for items."""
    entity = as_ref(entity)
    rows = [r for r in scoped(db, owner, entity, kinds=("commitment",), at=at) if r.modality not in NOT_COMMITTED]
    kept = supported(db, owner, rows)
    items = loops(db, owner, kept)
    if owner_actor:
        actor = me(db, owner) if owner_actor == "me" else owner_actor
        items = [i for i in items if any(p["actor_id"] == actor for p in i["responsible"])]
    if waiting_for is not None:
        items = [i for i in items if i["waiting_for"] is waiting_for]
    if not include_completed:
        items = [i for i in items if i["status"] == "open"]
    gaps = []
    if dropped := len(rows) - len(kept):
        gaps.append(f"{dropped} commitment(s) lost their supporting evidence and are not shown.")
    if unresolved := sum(1 for i in items if any(not p["resolved"] and not p["collective"] for p in i["responsible"])):
        gaps.append(f"{unresolved} item(s) name someone not yet matched to a person.")
    return {"items": items, "gaps": gaps}


# ---------------------------------------------------------------- calendar


@dataclass
class Occurrence:
    event: GoogleCalendarEvent
    start: datetime | None  # None: an event given by ID with no occurrence in the horizon
    end: datetime | None
    all_day: bool
    title: str
    attendees: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"event_id": self.event.id, "provider_id": self.event.provider_id, "title": self.title,
                "start": iso(self.start), "end": iso(self.end), "all_day": self.all_day,
                "attendees": self.attendees}


def series_key(provider_id: str | None) -> str | None:
    """Google instance IDs are `<series>_<start>`: the series part identifies a recurring meeting."""
    return provider_id.split("_", 1)[0] if provider_id else None


def upcoming(db, owner: str, *, at: datetime | None = None, days: int = HORIZON_DAYS) -> list[Occurrence]:
    """The owner's calendar occurrences that have not ended, soonest first (declined and cancelled excluded)."""
    from ..google_projection import project

    at = at or now()
    timezone = preferences(db, owner)["timezone"]
    first = at.astimezone(zone(timezone)).date()
    items, _ = project(db, owner, first, first + timedelta(days=days + 1), timezone)
    seen, picked = set(), []
    for item in items:
        start, end = datetime.fromisoformat(item["busy_start"]), datetime.fromisoformat(item["end_at"])
        key = (item["entity_id"], start)
        if end <= at or item.get("declined") or key in seen:
            continue
        seen.add(key)
        picked.append((start, end, item))
    rows = {r.id: r for r in db.scalars(select(GoogleCalendarEvent).where(
        GoogleCalendarEvent.id.in_({i["entity_id"] for _, _, i in picked})))} if picked else {}
    picked.sort(key=lambda p: (p[0], p[2]["entity_id"]))
    return [Occurrence(rows[i["entity_id"]], start, end, i["all_day"], i["title"])
            for start, end, i in picked if i["entity_id"] in rows]


def resolve_attendees(db, owner: str, event: GoogleCalendarEvent, me_id: str | None) -> list[dict]:
    """Calendar guests matched to actors: exact by email identity (or `self`), inferred by a calendar display
    name only one active actor carries, otherwise unresolved."""
    out = []
    for guest in (event.payload or {}).get("attendees", []):
        email, name = guest.get("email"), guest.get("displayName")
        actor_id, status = None, "unresolved"
        if guest.get("self") and me_id:
            actor_id, status = me_id, "exact"
        elif email and (found := identity.resolve_exact(db, owner, "email", email)):
            actor_id, status = found, "exact"
        elif name and len(named := identity.actors_for(db, owner, "calendar_name", name)) == 1:
            actor_id, status = named[0], "inferred"
        out.append({"name": name, "email": email, "actor_id": actor_id, "status": status,
                    "self": bool(guest.get("self")), "organizer": bool(guest.get("organizer")),
                    "response": guest.get("responseStatus")})
    names = actor_names(db, owner, [a["actor_id"] for a in out])
    for a in out:
        a["actor_name"] = names.get(a["actor_id"])
    return out


def _source_series(db, owner: str, rows: Sequence[ContextAssertion]) -> set[str]:
    """Series keys of the calendar events behind the sources that support these assertions."""
    if not rows:
        return set()
    e, rev = ContextEvidence, ContextSourceRevision
    keys = db.scalars(select(rev.calendar_event_id).join(e, e.revision_id == rev.id).where(
        e.owner_id == owner, e.assertion_id.in_([r.id for r in rows]), rev.calendar_event_id.is_not(None)).distinct())
    return {series_key(k) for k in keys}


def _guests(occurrence: Occurrence) -> list[dict]:
    return (occurrence.event.payload or {}).get("attendees", [])


def _matcher(db, owner: str, entity: EntityRef, at: datetime, me_id: str | None):
    """A cheap predicate over occurrences (strong identifiers only; attendees are resolved for the match)."""
    if entity.entity_type == "calendar_event":
        event = calendar_event(db, owner, entity.entity_id)
        return lambda o: event is not None and o.event.id == event.id
    if entity.entity_type == "actor":
        i = ContextActorIdentity
        emails = set(db.scalars(select(i.normalized).where(i.owner_id == owner, i.actor_id == entity.entity_id,
                                                           i.namespace == "email")))
        is_me = entity.entity_id == me_id
        return lambda o: any((g.get("self") and is_me) or identity.normalize("email", g.get("email") or "") in emails
                             for g in _guests(o))
    series = _source_series(db, owner, scoped(db, owner, entity, at=at, include_superseded=True))
    name = entity_name(db, owner, entity.entity_type, entity.entity_id) or ""
    pattern = re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", re.IGNORECASE) if len(name.strip()) >= 3 else None
    return lambda o: (series_key(o.event.provider_id) in series
                      or series_key((o.event.payload or {}).get("recurringEventId")) in series
                      or bool(pattern and pattern.search(o.title)))


def next_occurrence(db, owner: str, entity=None, *, at: datetime | None = None,
                    days: int = HORIZON_DAYS) -> Occurrence | None:
    """The next occurrence involving the entity; without one, the next meeting that has guests."""
    at = at or now()
    entity = as_ref(entity)
    me_id = me(db, owner)
    matches = _matcher(db, owner, entity, at, me_id) if entity else lambda o: bool(_guests(o))
    for occurrence in upcoming(db, owner, at=at, days=days):
        if matches(occurrence):
            occurrence.attendees = resolve_attendees(db, owner, occurrence.event, me_id)
            return occurrence
    return None


def _previous_meetings(db, owner: str, occurrence: Occurrence, emails: set[str], *, before: datetime,
                       limit: int = 3) -> list[dict]:
    s, rev = ContextSource, ContextSourceRevision
    series = {series_key(occurrence.event.provider_id),
              series_key((occurrence.event.payload or {}).get("recurringEventId"))} - {None}
    found = []
    for source, revision in db.execute(select(s, rev).join(rev, and_(rev.source_id == s.id,
                                                                      rev.revision == s.current_revision))
                                       .where(s.owner_id == owner, s.deleted_at.is_(None), s.status != "deleted",
                                              rev.occurred_at < before)
                                       .order_by(rev.occurred_at.desc()).limit(500)):
        same = series_key(revision.calendar_event_id) in series
        shared = {(p.get("email") or "").casefold() for p in revision.participants or []} & emails
        if same or shared:
            found.append({"source_id": source.id, "title": source.title, "occurred_at": iso(revision.occurred_at),
                          "match": "series" if same else "attendees"})
    found.sort(key=lambda m: m["match"] != "series")
    return found[:limit]


def meeting_prep(db, owner: str, *, calendar_event_id: str | None = None, entity=None,
                 at: datetime | None = None) -> dict:
    """Preparation for a calendar event (by ID), the next one involving `entity`, or the next meeting."""
    at = at or now()
    entity = as_ref(entity)
    me_id = me(db, owner)
    if calendar_event_id:
        event = calendar_event(db, owner, calendar_event_id)
        if event is None:
            raise DomainError("NOT_FOUND", "Calendar event is no longer available.", 404)
        occurrence = next(iter(o for o in upcoming(db, owner, at=at) if o.event.id == event.id), None)
        if occurrence is None:
            occurrence = Occurrence(event, None, None, False, (event.payload or {}).get("summary") or "Untitled event")
        occurrence.attendees = resolve_attendees(db, owner, event, me_id)
    else:
        occurrence = next_occurrence(db, owner, entity, at=at)
    gaps = []
    if occurrence is None:
        subject = "this " + entity.entity_type.replace("_", " ") if entity else "you"
        return {"event": None, "entity": _entity_view(db, owner, entity), "open_loops": [], "decisions": [],
                "open_questions": [], "previous_meetings": [],
                "gaps": [f"No upcoming calendar event involves {subject} in the next {HORIZON_DAYS} days."]}
    people = {a["actor_id"] for a in occurrence.attendees if a["actor_id"] and a["actor_id"] != me_id}
    if unresolved := [a["name"] or a["email"] for a in occurrence.attendees if not a["actor_id"]]:
        gaps.append("Not matched to a person yet: " + ", ".join(sorted(filter(None, unresolved))) + ".")
    scopes = ([entity] if entity else []) + [EntityRef(entity_type="actor", entity_id=p) for p in sorted(people)]
    rows: dict[str, ContextAssertion] = {}
    for scope in scopes:
        for row in scoped(db, owner, scope, kinds=("commitment", "decision", "open_question", "risk"), at=at):
            rows.setdefault(row.id, row)
    kept = supported(db, owner, list(rows.values()))
    commitments = [r for r in kept if r.kind == "commitment" and r.modality not in NOT_COMMITTED]
    cited = quotes(db, owner, [r.id for r in kept], per=1)

    def brief(r):
        return {"assertion_id": r.id, "revision": r.current_version, "kind": r.kind, "statement": r.statement,
                "lifecycle": r.lifecycle, "valid_from": iso(r.valid_from), "evidence": cited.get(r.id, [])}

    decided = [brief(r) for r in kept if r.kind in ("decision", "risk")][:10]
    questions = [brief(r) for r in kept if r.kind == "open_question"][:10]
    emails = {(a["email"] or "").casefold() for a in occurrence.attendees if a["email"] and not a["self"]}
    previous = _previous_meetings(db, owner, occurrence, emails, before=occurrence.start or at)
    if not previous:
        gaps.append("No earlier meeting with these people or in this series has been captured.")
    if not (commitments or decided or questions):
        gaps.append("Nothing open, decided or unanswered involves this meeting's people yet.")
    return {"event": occurrence.as_dict(), "entity": _entity_view(db, owner, entity),
            "open_loops": [i for i in loops(db, owner, commitments) if i["status"] == "open"],
            "decisions": decided, "open_questions": questions, "previous_meetings": previous, "gaps": gaps}


def _entity_view(db, owner: str, entity: EntityRef | None) -> dict | None:
    if entity is None:
        return None
    return {"entity_type": entity.entity_type, "entity_id": entity.entity_id,
            "name": entity_name(db, owner, entity.entity_type, entity.entity_id)}


# ---------------------------------------------------------------- person page


def person(db, owner: str, actor_id: str, *, at: datetime | None = None) -> dict:
    """Everything a person page shows about one actor (the user is an actor like everyone else)."""
    at = at or now()
    actor = db.get(Actor, actor_id)
    if actor is None or actor.owner_id != owner:
        raise DomainError("NOT_FOUND", "That person is no longer available.", 404)
    ref = EntityRef(entity_type="actor", entity_id=actor.id)
    rows = scoped(db, owner, ref, at=at)
    kept = supported(db, owner, rows)
    cited = quotes(db, owner, [r.id for r in kept], per=1)
    context = [{"assertion_id": r.id, "revision": r.current_version, "kind": r.kind, "statement": r.statement,
                "lifecycle": r.lifecycle, "inferred": not r.explicit, "modality": r.modality,
                "valid_from": iso(r.valid_from), "evidence": cited.get(r.id, [])}
               for r in kept if r.kind != "commitment"]
    commitments = [r for r in kept if r.kind == "commitment" and r.modality not in NOT_COMMITTED]
    all_loops = [i for i in loops(db, owner, commitments) if i["status"] == "open"]
    responsible = [i for i in all_loops if any(p["actor_id"] == actor.id for p in i["responsible"])]
    owed = [i for i in all_loops if (i["recipient"] or {}).get("actor_id") == actor.id and i not in responsible]

    i = ContextActorIdentity
    identities = [{"id": row.id, "namespace": row.namespace, "value": row.value, "status": row.status,
                   "source_scope": row.source_scope, "provenance": row.provenance}
                  for row in db.scalars(select(i).where(i.owner_id == owner, i.actor_id == actor.id)
                                        .order_by(i.namespace, i.normalized, i.id))]
    link = ContextEntityLink
    to_review = [{"link_id": row.id, "revision": row.revision, "mention": row.mention, "status": row.status,
                  "role": row.role, "assertion_id": row.assertion_id, "score": row.score}
                 for row in db.scalars(select(link).where(link.owner_id == owner, link.target_type == "actor",
                                                          link.target_id == actor.id, link.status == "inferred")
                                       .order_by(link.created_at.desc(), link.id).limit(50))]
    m = ContextActorMerge
    merges = [{"merge_id": row.id, "winner_actor_id": row.winner_actor_id, "loser_actor_id": row.loser_actor_id,
               "undone_at": iso(row.undone_at), "created_at": iso(row.created_at)}
              for row in db.scalars(select(m).where(m.owner_id == owner, or_(m.winner_actor_id == actor.id,
                                                                            m.loser_actor_id == actor.id))
                                    .order_by(m.created_at.desc(), m.id))]
    tasks = [task_view(t) for t in db.scalars(select(Task).where(
        Task.owner_id == owner, Task.assignee_id == actor.id, Task.archived.is_(False), Task.is_template.is_(False))
        .order_by(Task.status == "completed", Task.due_date.nulls_last(), Task.id).limit(50))]
    return {"actor": {"id": actor.id, "name": actor.name, "kind": actor.kind, "archived": actor.archived,
                      "revision": actor.revision, "is_me": actor.id == me(db, owner)},
            "identities": identities, "context": context, "projects": _projects(db, owner, actor.id, kept),
            "responsible_for": responsible, "owed_to_them": owed, "tasks": tasks,
            "meetings": _meetings(db, owner, actor.id, kept), "links_to_review": to_review, "merges": merges,
            "gaps": [f"{n} item(s) about this person lost their supporting evidence."]
            if (n := len(rows) - len(kept)) else []}


def _projects(db, owner: str, actor_id: str, rows: Sequence[ContextAssertion]) -> list[dict]:
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        for kind, ident in ((r.subject_type, r.subject_id), (r.object_type, r.object_id)):
            if kind == "project" and ident:
                counts[ident] += 1
    if rows:
        link = ContextEntityLink
        for target, n in db.execute(select(link.target_id, func.count()).where(
                link.owner_id == owner, link.assertion_id.in_([r.id for r in rows]), link.target_type == "project",
                link.status.in_(LINKED)).group_by(link.target_id)):
            counts[target] += n
    tasks = dict(db.execute(select(Task.project_id, func.count()).where(
        Task.owner_id == owner, Task.assignee_id == actor_id, Task.project_id.is_not(None),
        Task.archived.is_(False)).group_by(Task.project_id)).all())
    ids = set(counts) | set(tasks)
    names = dict(db.execute(select(Project.id, Project.name).where(Project.owner_id == owner,
                                                                   Project.id.in_(ids))).all()) if ids else {}
    return sorted(({"project_id": p, "name": names[p], "context_items": counts.get(p, 0), "tasks": tasks.get(p, 0)}
                   for p in ids if p in names), key=lambda x: (-x["context_items"] - x["tasks"], x["name"]))


def _meetings(db, owner: str, actor_id: str, rows: Sequence[ContextAssertion], limit: int = 20) -> list[dict]:
    """Live sources where the person spoke or that support context about them, newest first."""
    s, rev, seg = ContextSource, ContextSourceRevision, ContextSegment
    spoke = select(seg.revision_id).where(seg.owner_id == owner, seg.speaker_actor_id == actor_id)
    cited = select(ContextEvidence.revision_id).where(ContextEvidence.assertion_id.in_([r.id for r in rows]))
    found = db.execute(select(s.id, s.title, rev.occurred_at).join(rev, rev.source_id == s.id).where(
        s.owner_id == owner, s.deleted_at.is_(None), s.status != "deleted",
        or_(rev.id.in_(spoke), rev.id.in_(cited))).distinct().order_by(rev.occurred_at.desc().nulls_last(), s.id))
    out, seen = [], set()
    for source_id, title, occurred_at in found:
        if source_id not in seen:
            seen.add(source_id)
            out.append({"source_id": source_id, "title": title, "occurred_at": iso(occurred_at)})
    return out[:limit]


# ---------------------------------------------------------------- changes feed


def changes(db, owner: str, after: int | None = None, limit: int = 100) -> dict:
    """Commit-ordered context changes after a saved cursor, like `changes_list`. Each item reports the current
    state of the assertion, source, proposal, link or brief the event names; retracted assertions and deleted
    sources are tombstones without content."""
    after = after or 0
    limit = max(1, min(limit, 200))
    high = db.scalar(select(func.max(Event.id)).where(Event.owner_id == owner)) or 0
    if after > high:
        raise DomainError("CURSOR_INVALID",
                          "The change cursor is ahead of this workspace. Start a fresh synchronization.", 409)
    rows = list(db.scalars(select(Event).where(Event.owner_id == owner, Event.id > after, Event.id <= high,
                                               Event.kind == CHANGE_KIND).order_by(Event.id).limit(limit + 1)))
    more = len(rows) > limit
    rows = rows[:limit]
    states = _states(db, owner, {r.entity_id for r in rows})
    items = [{"cursor": r.id, "id": r.entity_id, "changed_at": iso(r.created_at), "event_revision": r.revision,
              **states.get(r.entity_id, {"type": "other", "deleted": False})} for r in rows]
    return {"items": items, "current_cursor": high, "next_cursor": rows[-1].id if more else high, "has_more": more}


def _states(db, owner: str, ids: set[str]) -> dict[str, dict]:
    if not ids:
        return {}
    out: dict[str, dict] = {}
    a = ContextAssertion
    assertions = list(db.scalars(select(a).where(a.owner_id == owner, a.id.in_(ids))))
    live = {r.id for r in supported(db, owner, [r for r in assertions if r.lifecycle != "retracted"])}
    successors = dict(db.execute(select(a.supersedes_id, a.id).where(
        a.owner_id == owner, a.supersedes_id.in_([r.id for r in assertions]))).all()) if assertions else {}
    for r in assertions:
        tombstone = r.id not in live
        out[r.id] = {"type": "assertion", "revision": r.current_version, "lifecycle": r.lifecycle, "kind": r.kind,
                     "deleted": tombstone, "statement": None if tombstone else r.statement,
                     "subject": None if tombstone else {"entity_type": r.subject_type, "entity_id": r.subject_id},
                     "superseded_by": successors.get(r.id) if r.lifecycle == "superseded" else None}
    s = ContextSource
    for r in db.scalars(select(s).where(s.owner_id == owner, s.id.in_(ids))):
        gone = r.deleted_at is not None or r.status == "deleted"
        out[r.id] = {"type": "source", "revision": r.current_revision, "status": r.status, "deleted": gone,
                     "title": None if gone else r.title}
    p = ContextProposal
    for r in db.scalars(select(p).where(p.owner_id == owner, p.id.in_(ids))):
        out[r.id] = {"type": "proposal", "revision": r.revision, "disposition": r.disposition, "deleted": False}
    link = ContextEntityLink
    for r in db.scalars(select(link).where(link.owner_id == owner, link.id.in_(ids))):
        out[r.id] = {"type": "link", "revision": r.revision, "status": r.status, "deleted": False,
                     "target": {"entity_type": r.target_type, "entity_id": r.target_id} if r.target_id else None}
    b = ContextBrief
    for r in db.scalars(select(b).where(b.owner_id == owner, b.id.in_(ids))):
        out[r.id] = {"type": "brief", "revision": r.version, "stale": r.stale, "deleted": False,
                     "entity": {"entity_type": r.entity_type, "entity_id": r.entity_id}}
    return out

