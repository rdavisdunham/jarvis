"""Resolve user-facing assignee aliases before exact canonical filtering."""

from sqlalchemy import select

from .domain import DomainError, preferences
from .models import Actor


def active_actors_named(db, owner, name):
    """Non-archived actors whose display name equals ``name`` exactly (case-sensitive,
    matching the former owner-plus-name constraint). Display names may repeat."""
    return list(db.scalars(select(Actor).where(
        Actor.owner_id == owner, Actor.name == name, Actor.archived.is_(False)).order_by(Actor.created_at, Actor.id)))


def resolve_actor_by_name(db, owner, name):
    """The one active actor called ``name``, or None when there is none.

    Several active actors with that name raise AMBIGUOUS_ACTOR: the caller must pass an
    actor ID instead. Archived actors never count. Callers decide what None means.
    """
    matches = active_actors_named(db, owner, name)
    if len(matches) > 1:
        raise DomainError(
            "AMBIGUOUS_ACTOR",
            f"Several people are named {name}. Choose one by its actor ID from organization_list.",
            409,
            data={"candidates": [{"id": a.id, "name": a.name, "kind": a.kind} for a in matches]},
        )
    return matches[0] if matches else None


def resolve_assignee(db, owner, value):
    value = value.strip()
    from .access import principal
    from .models import AuthSession, SharedWorkspace, WorkspaceMember
    if value.casefold() in {"me","myself"} and db.get(SharedWorkspace,owner) and principal.get():
        session=db.get(AuthSession,principal.get())
        member=db.get(WorkspaceMember,(owner,session.owner_id)) if session else None
        if member and member.active and member.actor_id:return member.actor_id
    actors = list(db.scalars(select(Actor).where(Actor.owner_id == owner, Actor.archived.is_(False))))
    exact = [a for a in actors if a.id == value]
    if exact:
        return exact[0].id
    matches = [a for a in actors if a.name.casefold() == value.casefold()]
    preferred = preferences(db, owner)["preferred_name"]
    if value.casefold() in {"me", "myself", "owner", preferred.casefold()}:
        matches += [a for a in actors if a.name == "owner"]
    matches = {a.id: a for a in matches}
    if len(matches) != 1:
        raise DomainError(
            "AMBIGUOUS_REFERENCE" if matches else "ASSIGNEE_NOT_FOUND",
            "Choose an exact assignee from organization_list; this name is ambiguous or unavailable.",
            409,
        )
    return next(iter(matches))
