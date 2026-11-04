"""Person pages for the signed-in owner's browser session: /api/v1/context/people/* (on `owner_router`).

A person page reads its brief, open loops and evidence through the session routes in `owner.py`. This module adds
what those do not cover: the person view (`views.person`: identities, work observations, projects, assigned tasks,
meetings, links awaiting review, past merges) with the merge candidates that involve the person, a short people
list for choosing whom to merge with, and identity corrections.

Reads follow `owner.py`: `service.as_owner(user)`, the flag, the personal workspace, and a re-check before return
in which every assertion shown must still be in scope and cite surviving evidence (the rest is withheld and
counted in `gaps`). Writes need the CSRF header (`auth.authenticate`) and the owner role (READ_ONLY otherwise), and
they are human decisions: the client's `command_id` must pass `routing.human_command`, so an agent driving the UI
(`ui-agent:` IDs) or a bot credential cannot merge, unmerge, declare two people different or settle a link. A
retried request with a command ID that already produced a decision changes nothing and returns the current view.
"""

from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Path, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import or_, select

from ... import routing
from ...auth import Identity, authenticate
from ...db import session_scope
from ...domain import DomainError, emit
from ...models import Actor, now
from .. import decisions, identity, service, views
from ..models import (
    ContextActorIdentity,
    ContextActorMerge,
    ContextAssertion,
    ContextDecision,
    ContextEntityLink,
)
from ..schema import ScopePredicate

User = Annotated[Identity, Depends(authenticate)]
ActorId = Annotated[str, Path(max_length=100)]
router = APIRouter(prefix="/people", tags=["Meeting context (owner)"])
PEOPLE_LIMIT = 50


class Write(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    command_id: str = Field(min_length=1, max_length=100)
    reason: str = Field(default="", max_length=500)


class MergeInput(Write):
    """Merge `loser_actor_id` into `winner_actor_id`; one of them is the page's person."""

    winner_actor_id: str = Field(min_length=1, max_length=100)
    loser_actor_id: str = Field(min_length=1, max_length=100)


class NotSameInput(Write):
    other_actor_id: str = Field(min_length=1, max_length=100)


class UnmergeInput(Write):
    merge_id: str = Field(min_length=1, max_length=100)


class LinkInput(Write):
    action: Literal["confirm", "reject"]
    expected_revision: int = Field(ge=1)


# ---------------------------------------------------------------- reads


def _recheck(db, caller: service.Caller) -> service.Caller:
    """The before-return check: the same principal, still allowed."""
    current = service.authorize(db)
    if (current.owner, current.credential_id, current.session) != (caller.owner, caller.credential_id,
                                                                   caller.session):
        raise DomainError("ACCESS_REVOKED", "Access changed during the request.", 401)
    return current


def _identities(db, owner: str, actor_ids: set[str]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {i: [] for i in actor_ids}
    if actor_ids:
        i = ContextActorIdentity
        for row in db.scalars(select(i).where(i.owner_id == owner, i.actor_id.in_(sorted(actor_ids)),
                                              i.namespace.in_(("email", "alias", "calendar_name")))
                              .order_by(i.namespace, i.normalized, i.id)):
            out[row.actor_id].append({"namespace": row.namespace, "value": row.value, "status": row.status})
    return out


def person_page(db, caller: service.Caller, actor_id: str, *, at: datetime | None = None) -> dict:
    """`views.person` shaped for the page and re-checked before return. Open loops are not repeated here: the
    page reads them through `open-loops`, which applies the same check."""
    at = at or now()
    owner = caller.owner
    data = views.person(db, owner, actor_id, at=at)
    candidates = [m for m in identity.merge_candidates(db, owner) if actor_id in m.actor_ids]
    others = {i for m in candidates for i in m.actor_ids if i != actor_id}
    merged = {i for m in data["merges"] for i in (m["winner_actor_id"], m["loser_actor_id"])}
    names = views.actor_names(db, owner, others | merged | {actor_id})
    emails = _identities(db, owner, others)
    review = data["links_to_review"]
    statements = dict(db.execute(select(ContextAssertion.id, ContextAssertion.statement).where(
        ContextAssertion.owner_id == owner,
        ContextAssertion.id.in_({r["assertion_id"] for r in review if r["assertion_id"]}))).all()) if review else {}

    caller = _recheck(db, caller)
    listed = [c["assertion_id"] for c in data["context"]] + [r["assertion_id"] for r in review if r["assertion_id"]]
    shown = service.visible(db, ScopePredicate(owner_id=owner), listed, at=at)
    cited = service.citations(db, owner, shown)
    context = [{k: v for k, v in c.items() if k != "evidence"} | {"citation": cited[c["assertion_id"]]}
               for c in data["context"] if c["assertion_id"] in shown]
    links = [{**r, "statement": statements.get(r["assertion_id"]) if r["assertion_id"] in shown else None}
             for r in review]
    gaps = list(data["gaps"])
    if withheld := len(data["context"]) - len(context):
        gaps.append(f"{withheld} statement(s) withheld: the interpretation changed or its evidence is no longer "
                    "available.")
    return {
        "actor": data["actor"], "identities": data["identities"], "context": context, "projects": data["projects"],
        "tasks": data["tasks"], "meetings": data["meetings"], "links_to_review": links,
        "merges": [{**m, "winner_name": names.get(m["winner_actor_id"]), "loser_name": names.get(m["loser_actor_id"])}
                   for m in data["merges"]],
        "merge_candidates": [{"actor_id": other, "name": names.get(other), "reason": m.reason,
                              "identities": emails.get(other, [])}
                             for m in candidates for other in m.actor_ids if other != actor_id],
        "gaps": gaps,
    }


@router.get("")
def list_people(user: User, q: str = Query("", max_length=100), limit: int = Query(20, ge=1, le=PEOPLE_LIMIT)):
    """Active people, for choosing whom to merge with. `q` matches a name or an email."""
    with service.as_owner(user), session_scope() as db:
        caller = service.authorize(db)
        a, i = Actor, ContextActorIdentity
        query = select(a).where(a.owner_id == caller.owner, a.archived.is_(False), a.kind == "person")
        if term := q.strip():
            pattern = "%" + term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            emailed = select(i.actor_id).where(i.owner_id == caller.owner, i.namespace == "email",
                                               i.value.ilike(pattern, escape="\\"))
            query = query.where(or_(a.name.ilike(pattern, escape="\\"), a.id.in_(emailed)))
        rows = list(db.scalars(query.order_by(a.name, a.created_at, a.id).limit(limit)))
        found = _identities(db, caller.owner, {r.id for r in rows})
        me_id = views.me(db, caller.owner)
        _recheck(db, caller)
        return {"items": [{"id": r.id, "name": r.name, "is_me": r.id == me_id, "identities": found[r.id]}
                          for r in rows]}


@router.get("/{actor_id}")
def get_person(actor_id: ActorId, user: User):
    with service.as_owner(user), session_scope() as db:
        return person_page(db, service.authorize(db), actor_id)


# ---------------------------------------------------------------- identity corrections (human only)


def _decide(user: Identity, actor_id: str, body: Write, apply) -> dict:
    """Authorize a human write, skip it when this command already decided, apply it and return the page of
    `apply`'s returned actor (the merge winner after a merge; otherwise the page's person)."""
    with service.as_owner(user), session_scope() as db:
        caller = service.authorize(db, service.WRITE, write=True)
        if not routing.human_command(body.command_id):
            raise DomainError("NOT_AUTHORIZED", "Only you can correct who someone is; agents may only suggest it.",
                              403)
        actor = db.get(Actor, actor_id)
        if actor is None or actor.owner_id != caller.owner:
            raise DomainError("NOT_FOUND", "That person is no longer available.", 404)
        done = db.scalar(select(ContextDecision.id).where(ContextDecision.owner_id == caller.owner,
                                                          ContextDecision.command_id == body.command_id).limit(1))
        shown = actor_id
        if done is None:
            decision = identity.DecisionInput(origin="human", principal=caller.principal, reason=body.reason,
                                              command_id=body.command_id)
            shown = apply(db, caller.owner, decision) or actor_id
            emit(db, caller.owner, "context.changed", actor_id)
        return {"person": person_page(db, caller, shown)}


def _other(db, owner: str, actor_id: str) -> None:
    other = db.get(Actor, actor_id)
    if other is None or other.owner_id != owner:
        raise DomainError("NOT_FOUND", "That person is no longer available.", 404)


@router.post("/{actor_id}/merge")
def merge(actor_id: ActorId, body: MergeInput, user: User):
    """Merge two people (the page's person is one of them). Returns the page of the person kept."""
    if actor_id not in (body.winner_actor_id, body.loser_actor_id):
        raise DomainError("INVALID_ARGUMENT", "One of the two people must be this person.")

    def apply(db, owner, decision):
        _other(db, owner, body.winner_actor_id if actor_id == body.loser_actor_id else body.loser_actor_id)
        identity.merge(db, owner, body.winner_actor_id, body.loser_actor_id, decision)
        return body.winner_actor_id

    return _decide(user, actor_id, body, apply)


@router.post("/{actor_id}/not-same")
def not_same(actor_id: ActorId, body: NotSameInput, user: User):
    """Durably record that two people are different; they are never proposed for merging again."""
    if body.other_actor_id == actor_id:
        raise DomainError("INVALID_ARGUMENT", "Choose two different people.")

    def apply(db, owner, decision):
        _other(db, owner, body.other_actor_id)
        identity.no_merge(db, owner, actor_id, body.other_actor_id, decision)

    return _decide(user, actor_id, body, apply)


@router.post("/{actor_id}/unmerge")
def unmerge(actor_id: ActorId, body: UnmergeInput, user: User):
    """Undo a merge this person took part in, restoring the other person and their prior assignments."""
    def apply(db, owner, decision):
        m = ContextActorMerge
        if db.scalar(select(m.id).where(m.id == body.merge_id, m.owner_id == owner,
                                        or_(m.winner_actor_id == actor_id, m.loser_actor_id == actor_id))) is None:
            raise DomainError("NOT_FOUND", "That merge is no longer available.", 404)
        identity.unmerge(db, owner, body.merge_id, decision)

    return _decide(user, actor_id, body, apply)


@router.post("/{actor_id}/links/{link_id}")
def settle_link(actor_id: ActorId, link_id: Annotated[str, Path(max_length=100)], body: LinkInput, user: User):
    """Confirm an inferred reference to this person, or reject it durably (it is never inferred again)."""
    def apply(db, owner, decision):
        link = db.get(ContextEntityLink, link_id)
        if link is None or link.owner_id != owner or (link.target_type, link.target_id) != ("actor", actor_id):
            raise DomainError("NOT_FOUND", "That reference is no longer available.", 404)
        settle = decisions.confirm_link if body.action == "confirm" else decisions.reject_link
        settle(db, owner, link_id, body.expected_revision, command_id=decision.command_id,
               principal=decision.principal, reason=decision.reason)

    return _decide(user, actor_id, body, apply)
