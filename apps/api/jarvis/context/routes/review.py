"""Meeting review: proposals, extracted context and publication (plan_final.md section 7).

`router` serves the signed-in owner's browser session under /api/v1/context/reviews (included in
`routes.owner_router`): the review read, approve selected (new tasks, links and suggestions), apply selected
updates (a separate approval), dismiss, remind me later, Wrong and Changed on extracted context, and publication
retry. Approval, update application and retry are human-only: they need a plain browser command ID
(`routing.human_command`), so neither a bot key nor an agent-driven UI edit (`ui-agent:`) can approve. Every
write is one transaction; a stale revision anywhere in a batch refuses the whole batch (409) so the reviewer
reloads.

`external` serves bot keys read-only under /api/v1/external/context/reviews with `context:read` (included in
`routes.router`). Agents never approve or publish; their corrections go through `context_feedback`.

Both refuse when `context_enabled` is off (CONTEXT_DISABLED) and outside the personal workspace.
"""

from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field
from sqlalchemy import exists, func, select

from ...auth import Identity, authenticate
from ...db import session_scope
from ...domain import DomainError
from ...models import LinearConnection, Task
from .. import decisions, proposals, publication, sources, views
from ..models import (
    ContextAssertion,
    ContextEntityLink,
    ContextEvidence,
    ContextProposal,
    ContextPublication,
    ContextSource,
    ContextSourceRevision,
)
from .settings import owner_of
from .sources import Bot
from .sources import authorize as bot_authorize

User = Annotated[Identity, Depends(authenticate)]
router = APIRouter(prefix="/reviews", tags=["Meeting review"])
external = APIRouter(prefix="/context/reviews", tags=["Meeting review"])

CommandId = Annotated[str, Field(min_length=1, max_length=100)]


class Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Ref(Body):
    proposal_id: UUID
    expected_revision: int = Field(ge=1)


class ApproveItem(Ref):
    destination: Literal["eridani", "linear"] | None = None
    edits: publication.Edits | None = None
    linear: publication.LinearOptions | None = None


class Approve(Body):
    command_id: CommandId
    items: list[ApproveItem] = Field(min_length=1, max_length=100)


class Decide(Body):
    command_id: CommandId
    items: list[Ref] = Field(min_length=1, max_length=100)
    reason: str = Field(default="", max_length=1000)


class Later(Decide):
    until: AwareDatetime | None = None


class Wrong(Body):
    command_id: CommandId
    expected_revision: int = Field(ge=1)
    reason: str = Field(default="", max_length=1000)


class Changed(Wrong):
    statement: str = Field(min_length=1, max_length=2000)
    effective_at: AwareDatetime | None = None
    due_at: AwareDatetime | None = None


class Retry(Body):
    command_id: CommandId


# ---------------------------------------------------------------- reads


def _iso(value) -> str | None:
    return value.isoformat() if value else None


def _proposal(row: ContextProposal, names: dict[str, str], tasks: dict[str, Task],
              latest: dict[str, ContextPublication]) -> dict:
    payload = row.payload or {}
    target = (payload.get("target") or {}).get("task_id") or row.task_id
    pub = latest.get(row.id)
    return {
        "id": row.id, "revision": row.revision, "kind": row.kind, "title": row.title,
        "disposition": row.disposition, "preselected": row.preselected, "destination": row.destination,
        "responsible": {"id": row.responsible, "name": names.get(row.responsible or ""),
                        "mention": payload.get("responsible_mention"), "collective": payload.get("collective", False)},
        "project_id": payload.get("project_id"), "due_date": payload.get("due_date"),
        "description": payload.get("description") or "", "reason": payload.get("reason"),
        "explanation": payload.get("explanation"), "match": payload.get("match"),
        "before": payload.get("before"), "after": payload.get("after"), "follows": payload.get("follows"),
        "origin": payload.get("origin"), "assertion_ids": row.assertion_ids, "evidence": row.evidence,
        "task": views.task_view(tasks[target]) if target in tasks else None,
        "publication": publication.view(pub) if pub else None,
        "updated_at": row.updated_at.isoformat(),
    }


def _context(db, owner: str, source_id: str, revision_id: str | None) -> list[dict]:
    """The source's extracted context with exact quotes and entity links (speaker and person badges)."""
    if revision_id is None:
        return []
    a, e = ContextAssertion, ContextEvidence
    rows = list(db.scalars(select(a).where(a.owner_id == owner, exists().where(
        e.assertion_id == a.id, e.source_id == source_id, e.revision_id == revision_id))
        .order_by(a.created_at, a.id)))
    ids = [r.id for r in rows]
    quotes = views.quotes(db, owner, ids, per=3)
    link = ContextEntityLink
    links: dict[str, list[dict]] = {}
    for row in db.scalars(select(link).where(link.owner_id == owner, link.assertion_id.in_(ids))
                          .order_by(link.created_at, link.id)):
        links.setdefault(row.assertion_id, []).append(
            {"id": row.id, "revision": row.revision, "role": row.role, "mention": row.mention,
             "target_type": row.target_type, "target_id": row.target_id, "status": row.status})
    return [{"id": r.id, "revision": r.current_version, "kind": r.kind, "statement": r.statement,
             "lifecycle": r.lifecycle, "modality": r.modality, "subject": {"type": r.subject_type, "id": r.subject_id},
             "responsible": r.responsible, "due_at": _iso(r.due_at), "valid_from": _iso(r.valid_from),
             "valid_to": _iso(r.valid_to), "supersedes_id": r.supersedes_id, "quotes": quotes.get(r.id, []),
             "links": links.get(r.id, [])} for r in rows]


def _destinations(db, owner: str) -> dict:
    conn = db.get(LinearConnection, owner)
    linear = None
    if conn is not None and conn.enabled and conn.credentials:
        teams = {t["id"]: t.get("name") for t in conn.directory.get("teams", [])}
        linear = {"teams": [{"id": t, "name": teams.get(t)} for t in conn.team_ids], "only_mine": conn.only_mine}
    return {"default": proposals.destination(db, owner), "linear": linear}


def review(db, owner: str, source_id: str) -> dict:
    """Everything the meeting review shows for one source; refreshes Linear publication states it reports."""
    source = sources.owned(db, owner, source_id)
    info = sources.get_source(db, owner, source.id)
    info.pop("segments", None)
    revision_id = db.scalar(select(ContextSourceRevision.id).where(
        ContextSourceRevision.source_id == source.id, ContextSourceRevision.revision == source.current_revision))
    rows = list(db.scalars(select(ContextProposal).where(ContextProposal.owner_id == owner,
                                                         ContextProposal.source_id == source.id)
                           .order_by(ContextProposal.created_at, ContextProposal.id)))
    pubs = list(db.scalars(select(ContextPublication).where(
        ContextPublication.owner_id == owner, ContextPublication.proposal_id.in_([r.id for r in rows]))
        .order_by(ContextPublication.created_at)))
    latest = {}
    for pub in pubs:
        latest[pub.proposal_id] = publication.refresh(db, owner, pub)
    targets = {((r.payload or {}).get("target") or {}).get("task_id") for r in rows} | {r.task_id for r in rows}
    tasks = {t.id: t for t in db.scalars(select(Task).where(Task.owner_id == owner, Task.id.in_(targets - {None})))}
    names = views.actor_names(db, owner, [r.responsible for r in rows])
    return {"source": info, "proposals": [_proposal(r, names, tasks, latest) for r in rows],
            "context": _context(db, owner, source.id, revision_id), "destinations": _destinations(db, owner)}


def inbox(db, owner: str, limit: int) -> dict:
    """Sources with proposals still open (pending, selected or later), newest meeting first."""
    p, s, r = ContextProposal, ContextSource, ContextSourceRevision
    rows = db.execute(
        select(s.id, s.title, s.status, func.max(r.occurred_at), func.count(func.distinct(p.id)),
               func.count(func.distinct(p.id)).filter(p.preselected.is_(True)))
        .join(p, p.source_id == s.id).join(r, (r.source_id == s.id) & (r.revision == s.current_revision))
        .where(s.owner_id == owner, s.status != "deleted", p.disposition.in_(proposals.OPEN))
        .group_by(s.id).order_by(func.max(r.occurred_at).desc().nulls_last()).limit(limit)).all()
    return {"items": [{"source_id": i, "title": t, "status": st, "occurred_at": _iso(at), "open": n,
                       "preselected": pre} for i, t, st, at, n, pre in rows]}


@router.get("")
def list_reviews(user: User, limit: int = Query(50, ge=1, le=200)):
    with session_scope() as db:
        return inbox(db, owner_of(db, user), limit)


@router.get("/{source_id}")
def get_review(source_id: UUID, user: User):
    with session_scope() as db:
        return review(db, owner_of(db, user), str(source_id))


@external.get("")
def list_reviews_bot(identity: Bot, limit: int = Query(50, ge=1, le=200)):
    with session_scope() as db:
        return inbox(db, bot_authorize(db, "context:read").owner_id, limit)


@external.get("/{source_id}")
def get_review_bot(source_id: UUID, identity: Bot):
    with session_scope() as db:
        return review(db, bot_authorize(db, "context:read").owner_id, str(source_id))


# ---------------------------------------------------------------- writes (owner session only)


def _in_source(db, owner: str, source_id: str, proposal_id: str) -> None:
    row = db.get(ContextProposal, proposal_id)
    if row is None or row.owner_id != owner or row.source_id != source_id:
        raise DomainError("NOT_FOUND", "That proposal is not part of this meeting.", 404)


def _approve(source_id: UUID, body: Approve, user: Identity, updates: bool) -> dict:
    with session_scope() as db:
        owner = owner_of(db, user, write=True)
        sources.owned(db, owner, str(source_id))
        out = []
        for item in body.items:
            _in_source(db, owner, str(source_id), str(item.proposal_id))
            out.append(publication.approve(
                db, owner, str(item.proposal_id), item.expected_revision, command_id=body.command_id,
                principal=user.account_id or owner, destination=item.destination, edits=item.edits,
                linear=item.linear, updates=updates))
        return {"publications": out}


@router.post("/{source_id}/approve")
def approve(source_id: UUID, body: Approve, user: User):
    """Add selected tasks: approve exact revisions, payloads and destinations, and queue their publication."""
    return _approve(source_id, body, user, updates=False)


@router.post("/{source_id}/apply-updates")
def apply_updates(source_id: UUID, body: Approve, user: User):
    """Apply selected updates to existing tasks (a separate approval from new tasks)."""
    return _approve(source_id, body, user, updates=True)


def _decide(source_id: UUID, body: Decide, user: Identity, act) -> dict:
    with session_scope() as db:
        owner = owner_of(db, user, write=True)
        sources.owned(db, owner, str(source_id))
        out = []
        for item in body.items:
            _in_source(db, owner, str(source_id), str(item.proposal_id))
            decision = act(db, owner, str(item.proposal_id), item.expected_revision)
            out.append({"proposal_id": str(item.proposal_id), "decision_id": decision.id, **decision.after})
        return {"proposals": out}


@router.post("/{source_id}/dismiss")
def dismiss(source_id: UUID, body: Decide, user: User):
    """Uncheck or finish without selection: no task; the context stays."""
    return _decide(source_id, body, user, lambda db, owner, pid, rev: proposals.dismiss(
        db, owner, pid, rev, command_id=body.command_id, principal=user.account_id or owner, reason=body.reason))


@router.post("/{source_id}/later")
def later(source_id: UUID, body: Later, user: User):
    """Remind me later: kept pending for the daily digest."""
    return _decide(source_id, body, user, lambda db, owner, pid, rev: proposals.later(
        db, owner, pid, rev, command_id=body.command_id, principal=user.account_id or owner, until=body.until,
        reason=body.reason))


def _assertion_in_source(db, owner: str, source_id: str, assertion_id: str) -> None:
    if not db.scalar(select(ContextEvidence.id).where(
            ContextEvidence.owner_id == owner, ContextEvidence.assertion_id == assertion_id,
            ContextEvidence.source_id == source_id).limit(1)):
        raise DomainError("NOT_FOUND", "That context item is not part of this meeting.", 404)


@router.post("/{source_id}/assertions/{assertion_id}/wrong")
def wrong(source_id: UUID, assertion_id: UUID, body: Wrong, user: User):
    """Extraction is wrong: retract the interpretation and supersede the open proposals resting on it."""
    with session_scope() as db:
        owner = owner_of(db, user, write=True)
        sources.owned(db, owner, str(source_id))
        _assertion_in_source(db, owner, str(source_id), str(assertion_id))
        decision = decisions.wrong(db, owner, str(assertion_id), body.expected_revision, command_id=body.command_id,
                                   principal=user.account_id or owner, reason=body.reason)
        dropped = proposals.invalidate(db, owner, str(assertion_id), "The context behind this was marked wrong.")
        return {"decision_id": decision.id, **decision.after, "superseded_proposals": dropped}


@router.post("/{source_id}/assertions/{assertion_id}/changed")
def changed(source_id: UUID, assertion_id: UUID, body: Changed, user: User):
    """Changed: record the new interpretation from its effective time and propose the task changes it implies."""
    with session_scope() as db:
        owner = owner_of(db, user, write=True)
        sources.owned(db, owner, str(source_id))
        _assertion_in_source(db, owner, str(source_id), str(assertion_id))
        decision = decisions.changed(db, owner, str(assertion_id), body.expected_revision, statement=body.statement,
                                     effective_at=body.effective_at, due_at=body.due_at, command_id=body.command_id,
                                     principal=user.account_id or owner, reason=body.reason)
        summary = proposals.regenerate(db, owner, str(source_id))
        dropped = proposals.invalidate(db, owner, str(assertion_id), "This was replaced by a changed interpretation.")
        return {"decision_id": decision.id, **decision.after, "proposals": summary, "superseded_proposals": dropped}


@router.post("/publications/{publication_id}/retry")
def retry(publication_id: UUID, body: Retry, user: User):
    """Retry a failed or unconfirmed publication under its original identity (reconciles before writing)."""
    with session_scope() as db:
        owner = owner_of(db, user, write=True)
        return publication.retry(db, owner, str(publication_id), command_id=body.command_id)
