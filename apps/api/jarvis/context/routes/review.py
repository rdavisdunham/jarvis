"""Meeting review: brief, extracted context, speakers, proposals and publication (plan_final.md section 7).

`router` serves the signed-in owner's browser session under /api/v1/context/reviews (included in
`routes.owner_router`), in the shapes `apps/web/src/context/context-api.ts` reads:

    GET  /reviews?limit&offset               {items: [{source_id, title, occurred_at, status, pending, later,
                                              published}]}  meetings with proposals, newest first
    GET  /reviews/{source_id}                the review (see `review`)
    POST /reviews/{source_id}/approve        {command_id, destination?, linear?, items: [{proposal_id, revision,
                                              destination?, edits?, linear?}]}  new tasks, links, suggestions
    POST /reviews/{source_id}/apply-updates  {command_id, items}  updates to existing tasks: a separate approval
    POST /reviews/{source_id}/dismiss        {command_id, items, reason?}
    POST /reviews/{source_id}/later          {command_id, items, until?}  no items: every open proposal
    POST /reviews/{source_id}/done           {command_id, review_revision, items}  dismiss these; context stays
    POST /reviews/{source_id}/wrong          {command_id, assertion_id, expected_revision, reason?}
    POST /reviews/{source_id}/changed        {command_id, assertion_id, expected_revision, statement,
                                              effective_at?, due_at?}
    POST /reviews/{source_id}/speakers       {command_id, link_id, expected_revision, action, decision_id?}
    POST /reviews/publications/{publication_id}/retry   {command_id}
    POST /reviews/{source_id}/reprocess      {command_id}  a fresh extraction of the current revision

Every write is one transaction and answers `{review}` with the new state. A stale revision anywhere in a batch
refuses the whole batch with 409 REVISION_CONFLICT. A write whose command ID already recorded decisions (the UI
resends the same ID after a lost connection) changes nothing and answers the current review. Approval, update
application and retry are human-only: `publication.require_human` accepts only a plain browser command ID
(`routing.human_command`), so neither a bot key nor an agent-driven UI edit (`ui-agent:...`) can approve. Auth
follows `routes/owner.py`: `service.as_owner(user)`, then `service.authorize` (CONTEXT_DISABLED,
PERSONAL_WORKSPACE, READ_ONLY for writes), CSRF on writes through `auth.authenticate`, and the same check again
before returning.

Review state is derived, never stored. A meeting is reviewed when none of its proposals is pending (E1's
`notices.unreviewed` reads the same thing): Done dismisses what is still open, Later defers it (the decision's
`until` becomes its `effective_at`, which the 17:00 digest reads), approval publishes it. `status` is pending
while a proposal is pending or selected, later while one is deferred, done otherwise. `revision` is the sum of the
meeting's proposal revisions: it grows with every proposal change or new proposal, so Done against an older view
is refused.

Speakers. Speaker resolutions live on the segments. A correction materializes one `context_entity_links` row per
speaker label (subject: the label's first segment in the current revision, role `speaker`), so
`decisions.confirm_link`, `reject_link` (a durable, source-scoped negative that `identity.rejected_links` honours
when the meeting is processed again) and `undo` apply; the label's segments then follow the link, and
`apply.apply_speakers` leaves decided segments alone. `link_id` in the API is that first segment's ID.

Reprocess (`pipeline.reprocess`) queues a fresh extract run of the current revision under the current prompt and
model, keyed by the command ID (a resend answers the same job), then the normal downstream stages. It is
human-only like approval, refused with 409 CONTEXT_BUSY while a job for the meeting is in flight, and changes no
approved or published proposal; dismissals still hold. `busy` in the review says a job for the meeting is in
flight.

`external` serves bot keys read-only under /api/v1/external/context/reviews with `context:read` (included in
`routes.router`); task details need `tasks:read`. Agents never approve or publish; their corrections go through
`context_feedback`.
"""

from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field
from sqlalchemy import exists, func, select

from ...auth import Identity, authenticate
from ...db import session_scope
from ...domain import DomainError
from ...models import LinearConnection, Task, now
from .. import apply, decisions, pipeline, proposals, publication, service, sources, views
from ..models import (
    ContextAssertion,
    ContextDecision,
    ContextEntityLink,
    ContextEvidence,
    ContextProposal,
    ContextPublication,
    ContextSegment,
    ContextSource,
    ContextSourceRevision,
)
from .sources import Bot

User = Annotated[Identity, Depends(authenticate)]
router = APIRouter(prefix="/reviews", tags=["Meeting review"])
external = APIRouter(prefix="/context/reviews", tags=["Meeting review"])

CommandId = Annotated[str, Field(min_length=1, max_length=100)]
OPEN = ("pending", "selected")
BRIEF_SECTIONS = (("current_state_and_why", ("decision", "rationale", "fact", "procedure", "preference")),
                  ("open_loops", ("commitment",)), ("unknowns", ("open_question", "risk")))
BRIEF_CLAIMS = 8
STALE = "This meeting's proposals changed. Review the latest version."


class Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Ref(Body):
    proposal_id: UUID
    revision: int = Field(ge=1)


class ApproveItem(Ref):
    destination: Literal["eridani", "linear"] | None = None
    edits: publication.Edits | None = None
    linear: publication.LinearOptions | None = None


class Approve(Body):
    command_id: CommandId
    destination: Literal["eridani", "linear"] | None = None
    linear: publication.LinearOptions | None = None
    items: list[ApproveItem] = Field(min_length=1, max_length=100)


class Decide(Body):
    command_id: CommandId
    items: list[Ref] = Field(max_length=200)
    reason: str = Field(default="", max_length=1000)


class Later(Decide):
    until: AwareDatetime | None = None


class Done(Decide):
    review_revision: int = Field(ge=0)


class Wrong(Body):
    command_id: CommandId
    assertion_id: UUID
    expected_revision: int = Field(ge=1)
    reason: str = Field(default="", max_length=1000)


class Changed(Wrong):
    statement: str = Field(min_length=1, max_length=2000)
    effective_at: AwareDatetime | None = None
    due_at: AwareDatetime | None = None


class SpeakerAction(Body):
    command_id: CommandId
    link_id: UUID
    expected_revision: int = Field(ge=1)
    action: Literal["confirm", "reject", "undo"]
    decision_id: UUID | None = None


class Retry(Body):
    command_id: CommandId


# ---------------------------------------------------------------- reads


def _iso(value) -> str | None:
    return value.isoformat() if value else None


def _current_revision(db, source: ContextSource) -> ContextSourceRevision | None:
    return db.scalar(select(ContextSourceRevision).where(
        ContextSourceRevision.source_id == source.id, ContextSourceRevision.revision == source.current_revision))


def state(rows) -> tuple[str, int]:
    """(status, review revision) of a meeting's proposals; see the module docstring."""
    dispositions = {r.disposition for r in rows}
    status = "pending" if dispositions & set(OPEN) else "later" if "later" in dispositions else "done"
    return status, sum(r.revision for r in rows)


def _speakers(db, owner: str, revision_id: str | None, names: dict[str, str]) -> list[dict]:
    if revision_id is None:
        return []
    first: dict[str, ContextSegment] = {}
    for segment in db.scalars(select(ContextSegment).where(
            ContextSegment.revision_id == revision_id, ContextSegment.speaker_label.is_not(None))
            .order_by(ContextSegment.seq)):
        first.setdefault(segment.speaker_label, segment)
    link = ContextEntityLink
    rows = {r.subject_id: r for r in db.scalars(select(link).where(
        link.owner_id == owner, link.subject_type == "segment", link.role == "speaker",
        link.subject_id.in_([s.id for s in first.values()])))}
    out = []
    for label, segment in first.items():
        row = rows.get(segment.id)
        decided = row is not None and row.status in apply.DECIDED
        actor = segment.speaker_actor_id
        out.append({"label": label, "actor_id": actor,
                    "actor_name": names.get(f"actor:{actor}") if actor else None,
                    "status": segment.speaker_link_status, "link_id": segment.id,
                    "link_revision": row.revision if row else 1,
                    "decision_id": row.decision_id if decided else None})
    return out


def _brief(items: list[dict]) -> dict:
    """A short deterministic brief of this meeting: what holds and why, open work, unknowns. Every claim cites."""
    live = [i for i in items if i["status"] in ("current", "contested") and i.get("citation")]
    sections = []
    for name, kinds in BRIEF_SECTIONS:
        claims = [{"text": i["statement"], "assertion_ids": [i["id"]],
                   "citations": [{"assertion_id": i["id"], "source_id": i["citation"]["source_id"],
                                  "segment": i["citation"]["segment"]}]}
                  for i in live if i["kind"] in kinds][:BRIEF_CLAIMS]
        if claims:
            sections.append({"name": name, "claims": claims})
    gaps = [] if sections else ["Nothing current was decided, promised or left open in this meeting."]
    return {"sections": sections, "gaps": gaps}


def _items(db, owner: str, source_id: str, revision_id: str | None) -> tuple[list[dict], list[tuple]]:
    """The meeting's extracted context in the context service's assertion item shape, and the entities named."""
    if revision_id is None:
        return [], []
    a, e = ContextAssertion, ContextEvidence
    rows = list(db.scalars(select(a).where(a.owner_id == owner, a.lifecycle != "shadow", exists().where(
        e.assertion_id == a.id, e.source_id == source_id, e.revision_id == revision_id))
        .order_by(a.created_at, a.id)))
    ids = [r.id for r in rows]
    cites, later = service.citations(db, owner, ids), service.successors(db, owner, ids)
    link = ContextEntityLink
    people: dict[str, list[str]] = {}
    for assertion_id, actor in db.execute(select(link.assertion_id, link.target_id).where(
            link.owner_id == owner, link.assertion_id.in_(ids), link.target_type == "actor",
            link.status.in_(views.LINKED)).order_by(link.created_at, link.id)):
        people.setdefault(assertion_id, []).append(actor)
    at, items, refs = now(), [], []
    for row in rows:
        entities = [x for x in (service._entity(row.subject_type, row.subject_id),
                                service._entity(row.object_type, row.object_id))
                    if x and x["entity_type"] != "source"]
        for actor in dict.fromkeys(people.get(row.id, [])):
            if {"entity_type": "actor", "entity_id": actor} not in entities:
                entities.append({"entity_type": "actor", "entity_id": actor})
        refs += [(x["entity_type"], x.get("entity_id")) for x in entities]
        items.append(service.assertion_item(row, citation=cites.get(row.id), reference=at,
                                            superseded_by=later.get(row.id), entities=entities))
    return items, refs


def _segments(db, rows) -> dict[str, tuple[int, str | None]]:
    ids = sorted({e.get("segment_id") for r in rows for e in r.evidence or []} - {None})
    if not ids:
        return {}
    g = ContextSegment
    return {i: (seq, speaker) for i, seq, speaker in db.execute(select(g.id, g.seq, g.speaker_label)
                                                                 .where(g.id.in_(ids)))}


def _shown(values: dict | None, keys, names: dict[str, str]) -> dict | None:
    """Before or after as the reader sees it: only the fields that change, people by name."""
    if values is None:
        return None
    out = {}
    for key in keys:
        value = values.get(key)
        if key == "assignee_id":
            out["assignee"] = names.get(f"actor:{value}", "Someone") if value else None
        else:
            out[key] = value
    return out


def _publication(row: ContextPublication | None) -> dict | None:
    if row is None:
        return None
    out = publication.view(row)
    out["error"] = out.get("message") or out.get("error")
    return out


def _proposal(row: ContextProposal, names: dict[str, str], tasks: dict[str, Task], segments: dict,
              latest: dict[str, ContextPublication]) -> dict:
    payload = row.payload or {}
    target = payload.get("target") or {}
    task_id = target.get("task_id") or row.task_id
    task = tasks.get(task_id) if task_id else None
    project = payload.get("project_id")
    after = payload.get("after")
    keys = [k for k in (after or {}) if k not in ("revision", "expected_revision")]
    return {
        "id": row.id, "revision": row.revision, "kind": row.kind, "title": row.title,
        "description": payload.get("description") or "",
        "assignee": {"id": row.responsible, "name": names.get(f"actor:{row.responsible}", "Someone")}
        if row.responsible else None,
        "responsible_mention": payload.get("responsible_mention"), "collective": payload.get("collective", False),
        "project": {"id": project, "name": names.get(f"project:{project}", "Project")} if project else None,
        "due_at": _iso(row.due_at), "due_date": payload.get("due_date"), "reason": payload.get("reason") or "",
        "explanation": payload.get("explanation"), "preselected": row.preselected,
        "disposition": row.disposition, "destination": row.destination,
        "evidence": [{"quote": e.get("quote", ""), "segment": segments.get(e.get("segment_id"), (None, None))[0],
                      "speaker": segments.get(e.get("segment_id"), (None, None))[1]} for e in row.evidence or []],
        "target": {"id": task_id, "title": task.title if task else "A task", "entity_type": "task"}
        if task_id and row.kind in ("update", "link") else None,
        "before": _shown(payload.get("before"), keys, names), "after": _shown(after, keys, names),
        "match": payload.get("match"), "follows": payload.get("follows"), "assertion_ids": row.assertion_ids,
        "task": views.task_view(task) if task else None,
        "publication": _publication(latest.get(row.id)),
        "updated_at": row.updated_at.isoformat(),
    }


def _destinations(db, owner: str) -> list[str]:
    conn = db.get(LinearConnection, owner)
    linear = conn is not None and conn.enabled and conn.credentials and conn.team_ids
    return ["eridani", "linear"] if linear else ["eridani"]


def review(db, caller: service.Caller, source_id: str) -> dict:
    """Everything the meeting review shows for one source. Refreshes the Linear publication states it reports."""
    owner = caller.owner
    source = sources.owned(db, owner, source_id)
    info = sources.get_source(db, owner, source.id)
    info.pop("segments", None)
    revision = _current_revision(db, source)
    revision_id = revision.id if revision else None
    rows = list(db.scalars(select(ContextProposal).where(ContextProposal.owner_id == owner,
                                                         ContextProposal.source_id == source.id)
                           .order_by(ContextProposal.created_at, ContextProposal.id)))
    latest: dict[str, ContextPublication] = {}
    if rows:
        for pub in db.scalars(select(ContextPublication).where(
                ContextPublication.owner_id == owner, ContextPublication.proposal_id.in_([r.id for r in rows]))
                .order_by(ContextPublication.created_at)):
            latest[pub.proposal_id] = publication.refresh(db, owner, pub)
    items, refs = _items(db, owner, source.id, revision_id)
    tasks: dict[str, Task] = {}
    if caller.can("tasks:read"):
        wanted = ({((r.payload or {}).get("target") or {}).get("task_id") for r in rows}
                  | {r.task_id for r in rows}) - {None}
        tasks = {t.id: t for t in db.scalars(select(Task).where(Task.owner_id == owner, Task.id.in_(wanted)))}
    speakers = list(db.scalars(select(ContextSegment.speaker_actor_id).where(
        ContextSegment.revision_id == revision_id, ContextSegment.speaker_actor_id.is_not(None)).distinct()))
    refs += [("actor", r.responsible) for r in rows] + [("actor", a) for a in speakers]
    refs += [("actor", ((r.payload or {}).get(k) or {}).get("assignee_id")) for r in rows for k in ("before", "after")]
    refs += [("project", (r.payload or {}).get("project_id")) for r in rows]
    names = service.names(db, caller, refs)
    segments = _segments(db, rows)
    status, review_revision = state(rows)
    return {
        "source_id": source.id, "source": info, "revision": review_revision, "status": status,
        "busy": pipeline.busy(db, owner, source.id),
        "destination": proposals.destination(db, owner), "destinations": _destinations(db, owner),
        "brief": _brief(items), "items": items, "speakers": _speakers(db, owner, revision_id, names),
        "proposals": [_proposal(r, names, tasks, segments, latest) for r in rows], "names": names,
    }


def inbox(db, owner: str, limit: int, offset: int) -> dict:
    """Meetings with proposals, newest first, with their review status and counts."""
    p, s, r = ContextProposal, ContextSource, ContextSourceRevision

    def count(*states):
        return func.count(p.id).filter(p.disposition.in_(states))
    rows = db.execute(
        select(s.id, s.title, func.max(r.occurred_at), count(*OPEN), count("later"), count("published"))
        .join(p, p.source_id == s.id).join(r, (r.source_id == s.id) & (r.revision == s.current_revision))
        .where(s.owner_id == owner, s.status != "deleted", s.deleted_at.is_(None))
        .group_by(s.id).order_by(func.max(r.occurred_at).desc().nulls_last(), s.id)
        .limit(limit).offset(offset)).all()
    return {"items": [{"source_id": i, "title": t, "occurred_at": _iso(at),
                       "status": "pending" if pending else "later" if later else "done",
                       "pending": pending, "later": later, "published": published}
                      for i, t, at, pending, later, published in rows]}


def _read(source_id: str | None, limit: int = 50, offset: int = 0) -> dict:
    with session_scope() as db:
        caller = service.authorize(db)
        out = inbox(db, caller.owner, limit, offset) if source_id is None else review(db, caller, source_id)
        service._again(db, caller)
        return out


@router.get("")
def list_reviews(user: User, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0, le=100_000)):
    with service.as_owner(user):
        return _read(None, limit, offset)


@router.get("/{source_id}")
def get_review(source_id: UUID, user: User):
    with service.as_owner(user):
        return _read(str(source_id))


@external.get("")
def list_reviews_bot(identity: Bot, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0, le=100_000)):
    return _read(None, limit, offset)


@external.get("/{source_id}")
def get_review_bot(source_id: UUID, identity: Bot):
    return _read(str(source_id))


# ---------------------------------------------------------------- writes (owner session only)


def _replayed(db, owner: str, command_id: str) -> bool:
    d = ContextDecision
    return db.scalar(select(d.id).where(d.owner_id == owner, d.command_id == command_id).limit(1)) is not None


def _write(user: Identity, source_id: UUID, command_id: str, act, *, replay: bool = True) -> dict:
    """Authorize the owner's session for a write, run `act(db, caller, source)` once per command ID, and answer
    with the new review."""
    with service.as_owner(user), session_scope() as db:
        caller = service.authorize(db, service.WRITE, write=True)
        source = sources.owned(db, caller.owner, str(source_id))
        extra = {} if replay and _replayed(db, caller.owner, command_id) else act(db, caller, source) or {}
        out = {**extra, "review": review(db, caller, source.id)}
        service._again(db, caller, service.WRITE)
        return out


def _in_source(db, owner: str, source_id: str, proposal_id) -> ContextProposal:
    row = db.get(ContextProposal, str(proposal_id))
    if row is None or row.owner_id != owner or row.source_id != source_id:
        raise DomainError("NOT_FOUND", "That proposal is not part of this meeting.", 404)
    return row


def _approve(source_id: UUID, body: Approve, user: Identity, updates: bool) -> dict:
    publication.require_human(body.command_id)

    def act(db, caller, source):
        done = []
        for item in body.items:
            _in_source(db, caller.owner, source.id, item.proposal_id)
            done.append(publication.approve(
                db, caller.owner, str(item.proposal_id), item.revision, command_id=body.command_id,
                principal=caller.principal, destination=item.destination or body.destination, edits=item.edits,
                linear=item.linear or body.linear, updates=updates))
        return {"publications": done}
    # publication.approve replays per proposal, so a resent approval still answers with its publications
    return _write(user, source_id, body.command_id, act, replay=False)


@router.post("/{source_id}/approve")
def approve(source_id: UUID, body: Approve, user: User):
    """Add selected tasks: approve exact revisions, payloads and destinations, and queue their publication."""
    return _approve(source_id, body, user, updates=False)


@router.post("/{source_id}/apply-updates")
def apply_updates(source_id: UUID, body: Approve, user: User):
    """Apply selected updates to existing tasks (a separate approval from new tasks)."""
    return _approve(source_id, body, user, updates=True)


@router.post("/{source_id}/dismiss")
def dismiss(source_id: UUID, body: Decide, user: User):
    """Uncheck or finish without selection: no task; the context stays."""
    def act(db, caller, source):
        for item in body.items:
            _in_source(db, caller.owner, source.id, item.proposal_id)
            proposals.dismiss(db, caller.owner, str(item.proposal_id), item.revision, command_id=body.command_id,
                              principal=caller.principal, reason=body.reason)
    return _write(user, source_id, body.command_id, act)


@router.post("/{source_id}/later")
def later(source_id: UUID, body: Later, user: User):
    """Remind me later: these proposals (or, with no items, every open one) wait for the daily digest."""
    def act(db, caller, source):
        if body.items:
            chosen = [(_in_source(db, caller.owner, source.id, i.proposal_id), i.revision) for i in body.items]
        else:
            chosen = [(r, r.revision) for r in db.scalars(select(ContextProposal).where(
                ContextProposal.owner_id == caller.owner, ContextProposal.source_id == source.id,
                ContextProposal.disposition.in_(OPEN)).order_by(ContextProposal.created_at, ContextProposal.id))]
        for row, revision in chosen:
            proposals.later(db, caller.owner, row.id, revision, command_id=body.command_id,
                            principal=caller.principal, until=body.until, reason=body.reason)
    return _write(user, source_id, body.command_id, act)


@router.post("/{source_id}/done")
def done(source_id: UUID, body: Done, user: User):
    """Finish the review: dismiss the proposals still open (as listed); the meeting's context stays."""
    def act(db, caller, source):
        rows = list(db.scalars(select(ContextProposal).where(
            ContextProposal.owner_id == caller.owner, ContextProposal.source_id == source.id)
            .order_by(ContextProposal.id).with_for_update()))
        if state(rows)[1] != body.review_revision:
            raise DomainError("REVISION_CONFLICT", STALE, 409)
        for item in body.items:
            row = _in_source(db, caller.owner, source.id, item.proposal_id)
            if row.disposition in (*OPEN, "later"):
                proposals.dismiss(db, caller.owner, row.id, item.revision, command_id=body.command_id,
                                  principal=caller.principal, reason=body.reason or "Finished the review.")
    return _write(user, source_id, body.command_id, act)


def _assertion_in_source(db, owner: str, source_id: str, assertion_id: str) -> None:
    if not db.scalar(select(ContextEvidence.id).where(
            ContextEvidence.owner_id == owner, ContextEvidence.assertion_id == assertion_id,
            ContextEvidence.source_id == source_id).limit(1)):
        raise DomainError("NOT_FOUND", "That context item is not part of this meeting.", 404)


@router.post("/{source_id}/wrong")
def wrong(source_id: UUID, body: Wrong, user: User):
    """Extraction is wrong: retract the interpretation and supersede the open proposals resting on it."""
    def act(db, caller, source):
        assertion_id = str(body.assertion_id)
        _assertion_in_source(db, caller.owner, source.id, assertion_id)
        decision = decisions.wrong(db, caller.owner, assertion_id, body.expected_revision,
                                   command_id=body.command_id, principal=caller.principal, reason=body.reason)
        dropped = proposals.invalidate(db, caller.owner, assertion_id, "The context behind this was marked wrong.")
        return {"decision_id": decision.id, "superseded_proposals": dropped}
    return _write(user, source_id, body.command_id, act)


@router.post("/{source_id}/changed")
def changed(source_id: UUID, body: Changed, user: User):
    """Changed: record the new interpretation from its effective time and propose the task changes it implies."""
    def act(db, caller, source):
        assertion_id = str(body.assertion_id)
        _assertion_in_source(db, caller.owner, source.id, assertion_id)
        decision = decisions.changed(db, caller.owner, assertion_id, body.expected_revision,
                                     statement=body.statement, effective_at=body.effective_at, due_at=body.due_at,
                                     command_id=body.command_id, principal=caller.principal, reason=body.reason)
        summary = proposals.regenerate(db, caller.owner, source.id)
        dropped = proposals.invalidate(db, caller.owner, assertion_id,
                                       "This was replaced by a changed interpretation.")
        return {"decision_id": decision.id, "proposals": summary, "superseded_proposals": dropped}
    return _write(user, source_id, body.command_id, act)


def _speaker_link(db, owner: str, revision_id: str | None, segment_id: str) -> tuple[ContextEntityLink, list]:
    """The label's speaker link (made from its segments on first use) and the label's segments, locked."""
    first = db.get(ContextSegment, segment_id)
    missing = DomainError("NOT_FOUND", "That speaker is not part of this meeting's current transcript.", 404)
    if first is None or first.owner_id != owner or first.revision_id != revision_id or not first.speaker_label:
        raise missing
    segments = list(db.scalars(select(ContextSegment).where(
        ContextSegment.revision_id == revision_id, ContextSegment.speaker_label == first.speaker_label)
        .order_by(ContextSegment.seq).with_for_update()))
    if segments[0].id != first.id:
        raise missing
    link = ContextEntityLink
    row = db.scalar(select(link).where(link.owner_id == owner, link.subject_type == "segment",
                                       link.role == "speaker", link.subject_id == first.id).with_for_update())
    if row is None:
        stamp = now()
        row = link(owner_id=owner, subject_type="segment", subject_id=first.id, role="speaker",
                   mention=first.speaker_label[:500], target_type="actor" if first.speaker_actor_id else None,
                   target_id=first.speaker_actor_id, status=first.speaker_link_status, revision=1, evidence=[],
                   created_at=stamp, updated_at=stamp)
        db.add(row)
        db.flush()
    return row, segments


def _follow(segments: list, row: ContextEntityLink, decision_id: str | None) -> None:
    """The label's segments take the link's state; a decided segment carries the decision in its provenance."""
    for segment in segments:
        segment.speaker_actor_id = None if row.status == "rejected" else row.target_id
        segment.speaker_link_status = row.status
        provenance = {k: v for k, v in (segment.speaker_provenance or {}).items() if k != "decision_id"}
        segment.speaker_provenance = {**provenance, "decision_id": decision_id} if decision_id else provenance


@router.post("/{source_id}/speakers")
def speakers(source_id: UUID, body: SpeakerAction, user: User):
    """Confirm a speaker match, reject it (durably, for this meeting), or undo that decision."""
    def act(db, caller, source):
        revision = _current_revision(db, source)
        row, segments = _speaker_link(db, caller.owner, revision.id if revision else None, str(body.link_id))
        if row.revision != body.expected_revision:
            raise DomainError("REVISION_CONFLICT", "This speaker changed. Review the latest version.", 409)
        if body.action == "undo":
            if body.decision_id is None or str(body.decision_id) != row.decision_id \
                    or row.status not in apply.DECIDED:
                raise DomainError("REVISION_CONFLICT", "This speaker changed. Review the latest version.", 409)
            decision = decisions.undo(db, caller.owner, row.decision_id, command_id=body.command_id,
                                      principal=caller.principal)
            _follow(segments, row, None)
        else:
            run = decisions.confirm_link if body.action == "confirm" else decisions.reject_link
            decision = run(db, caller.owner, row.id, body.expected_revision, command_id=body.command_id,
                           principal=caller.principal)
            _follow(segments, row, decision.id)
        return {"decision_id": decision.id}
    return _write(user, source_id, body.command_id, act)


@router.post("/{source_id}/reprocess")
def reprocess(source_id: UUID, body: Retry, user: User):
    """Process the meeting again with the current prompt and model (see the module docstring)."""
    if decisions.origin_of(body.command_id) != "human":
        raise DomainError("HUMAN_REQUIRED", "Only you can reprocess a meeting, from the review screen.", 403)

    def act(db, caller, source):
        return {"reprocess": pipeline.reprocess(db, caller.owner, source.id, body.command_id)}
    return _write(user, source_id, body.command_id, act, replay=False)


@router.post("/publications/{publication_id}/retry")
def retry(publication_id: UUID, body: Retry, user: User):
    """Retry a failed or unconfirmed publication under its original identity (reconciles before writing)."""
    with service.as_owner(user), session_scope() as db:
        caller = service.authorize(db, service.WRITE, write=True)
        out = publication.retry(db, caller.owner, str(publication_id), command_id=body.command_id)
        source_id = db.scalar(select(ContextProposal.source_id).where(ContextProposal.id == out["proposal_id"]))
        result = {"publication": out, **({"review": review(db, caller, source_id)} if source_id else {})}
        service._again(db, caller, service.WRITE)
        return result
