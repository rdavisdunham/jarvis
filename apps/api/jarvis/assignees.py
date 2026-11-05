"""Resolve people by name: display names and their aliases ("Also known as").

Aliases are `alias` identities on an actor (`context_actor_identities`, written by `context.identity`): the
owner's own come from Settings > Profile (`set_own_aliases`, through `settings.update` `name_aliases`), others from
meeting identity seeding. Matching is exact after normalisation (NFKC, case-insensitive, single-spaced). A
single-word alias never matches when another active actor's name or alias starts with that word: a bare first
name shared with someone else identifies nobody. Saving an alias that is another actor's name or alias, or such a
shared first name, is refused (ALIAS_TAKEN).
"""

from sqlalchemy import select

from .domain import DomainError, preferences
from .models import Actor

ALIAS_LIMIT = 10
PROFILE = {"origin": "profile"}


def name_key(value: str) -> str:
    from .context.identity import normalize

    return normalize("alias", value or "")


def _alias_rows(db, owner):
    from .context.models import ContextActorIdentity as I

    return db.execute(select(I.actor_id, I.normalized).where(
        I.owner_id == owner, I.namespace == "alias", I.source_scope.is_(None), I.status != "rejected")).all()


def _active(db, owner) -> dict[str, Actor]:
    return {a.id: a for a in db.scalars(select(Actor).where(Actor.owner_id == owner, Actor.archived.is_(False)))}


def _first_names(actors: dict[str, Actor], aliases) -> dict[str, set[str]]:
    """First word -> active actors whose name or one of whose aliases starts with it."""
    out: dict[str, set[str]] = {}
    for actor in actors.values():
        if words := name_key(actor.name).split():
            out.setdefault(words[0], set()).add(actor.id)
    for actor_id, normalized in aliases:
        if actor_id in actors and (words := normalized.split()):
            out.setdefault(words[0], set()).add(actor_id)
    return out


def alias_matches(db, owner, name) -> list[Actor]:
    """Active actors carrying `name` as an alias, under the bare-first-name rule (module docstring)."""
    key = name_key(name)
    if not key:
        return []
    aliases = _alias_rows(db, owner)
    hits = {actor_id for actor_id, normalized in aliases if normalized == key}
    if not hits:
        return []
    actors = _active(db, owner)
    hits &= set(actors)
    if len(key.split()) == 1:
        sharing = _first_names(actors, aliases).get(key, set())
        hits = {h for h in hits if not sharing - {h}}
    return sorted((actors[h] for h in hits), key=lambda a: (a.created_at, a.id))


def name_in_use(db, owner, name, *, exclude: str | None = None) -> list[Actor]:
    """Other active actors called `name` exactly or carrying it as an alias (no first-name rule): a new or renamed
    actor with this name could not be told apart by name."""
    key = name_key(name)
    actors = _active(db, owner)
    ids = {a.id for a in actors.values() if a.name == name}
    ids |= {actor_id for actor_id, normalized in _alias_rows(db, owner) if key and normalized == key}
    return sorted((actors[i] for i in ids if i in actors and i != exclude), key=lambda a: (a.created_at, a.id))


def active_actors_named(db, owner, name):
    """Non-archived actors whose display name equals ``name`` exactly (case-sensitive,
    matching the former owner-plus-name constraint). Display names may repeat."""
    return list(db.scalars(select(Actor).where(
        Actor.owner_id == owner, Actor.name == name, Actor.archived.is_(False)).order_by(Actor.created_at, Actor.id)))


def resolve_actor_by_name(db, owner, name):
    """The one active actor called ``name`` or carrying it as an alias, or None when there is none.

    Several active actors with that name raise AMBIGUOUS_ACTOR: the caller must pass an
    actor ID instead. Archived actors never count. Callers decide what None means.
    """
    matches = list({a.id: a for a in active_actors_named(db, owner, name) + alias_matches(db, owner, name)}.values())
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
    matches = [a for a in actors if a.name.casefold() == value.casefold()] + alias_matches(db, owner, value)
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


# ---------------------------------------------------------------- the owner's own aliases (Settings > Profile)


def clean_aliases(values) -> list[str]:
    """Trimmed, single-spaced, deduplicated (case-insensitively) aliases, at most ALIAS_LIMIT, each 1-80 chars."""
    out, seen = [], set()
    for raw in values or []:
        value = " ".join(str(raw).split())
        if not value:
            continue
        if len(value) > 80 or "@" in value:
            raise DomainError("INVALID_ARGUMENT", "Each name must be 80 characters or fewer and not an email address.")
        if name_key(value) in {"me", "myself", "owner", "i"}:
            raise DomainError("INVALID_ARGUMENT", f"“{value}” can't be used as a name.")
        if name_key(value) not in seen:
            seen.add(name_key(value))
            out.append(value)
    if len(out) > ALIAS_LIMIT:
        raise DomainError("INVALID_ARGUMENT", f"Add at most {ALIAS_LIMIT} other names.")
    return out


def set_own_aliases(db, owner, values) -> list[str]:
    """Replace the owner's profile aliases on their own actor (personal workspace only), refusing a name another
    actor already has, and add the sign-in email when the actor has no email identity. Returns the saved list."""
    from .context import identity
    from .context.models import ContextActorIdentity as I
    from .models import GoogleIdentity, SharedWorkspace

    if db.get(SharedWorkspace, owner) is not None:
        raise DomainError("PERSONAL_WORKSPACE", "Set the names you go by in your personal workspace.", 403)
    aliases = clean_aliases(values)
    me = identity.owner_actor(db, owner)
    actors = _active(db, owner)
    rows = _alias_rows(db, owner)
    first = _first_names(actors, rows)
    for value in aliases:
        key = name_key(value)
        taken = name_in_use(db, owner, value, exclude=me)
        taken += [a for i, a in actors.items() if i != me and name_key(a.name) == key and a not in taken]
        if taken:
            raise DomainError("ALIAS_TAKEN", f"“{value}” is already a name of {taken[0].name}. Use a name only you go by.",
                              409, data={"alias": value, "actor_id": taken[0].id})
        if len(key.split()) == 1 and (others := first.get(key, set()) - {me}):
            other = actors[min(others)]
            raise DomainError("ALIAS_TAKEN", f"“{value}” is also how {other.name}'s name starts, so it can't identify "
                              "you. Use a fuller name.", 409, data={"alias": value, "actor_id": other.id})
    keys = {name_key(v) for v in aliases}
    for row in db.scalars(select(I).where(I.owner_id == owner, I.actor_id == me, I.namespace == "alias",
                                          I.source_scope.is_(None))):
        if (row.provenance or {}).get("origin") == PROFILE["origin"] and row.normalized not in keys:
            db.delete(row)
    db.flush()
    for value in aliases:
        identity.add_identity(db, owner, me, "alias", value, status="confirmed", provenance=PROFILE)
    has_email = db.scalar(select(I.id).where(I.owner_id == owner, I.actor_id == me, I.namespace == "email").limit(1))
    account = db.get(GoogleIdentity, owner)
    if not has_email and account is not None and account.email:
        identity.add_identity(db, owner, me, "email", account.email, status="confirmed", provenance=PROFILE)
    db.flush()
    return aliases
