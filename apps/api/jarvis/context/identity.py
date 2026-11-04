"""Actor identities for meeting context (plan_final.md section 3, People).

Identifiers live in `context_actor_identities` under a namespace. Strong namespaces (email, granola_speaker,
linear_user, organization) name at most one actor per `(normalized, source_scope)`; `alias` and `calendar_name`
are display names and may name several. A diarization label (`granola_speaker`) is scoped to the source it came
from, so the same label in two meetings never shares an identity.

- `seed_candidates` turns a meeting's participants and calendar attendees into identity candidates. An email
  resolves or creates exactly one actor (`exact`); a name alone is an ambiguous alias (`inferred` when exactly
  one actor carries it, otherwise `unresolved` with the candidates). Actors are created only through the
  `actor.create` command, never by inserting rows; nothing is created from a name alone.
- `owner_actor` is the note-taker's actor, used for the owner's own lines.
- Durable negatives (`reject_link`, `no_merge`) are `context_decisions` rows; `rejected_links` and
  `merge_candidates` read them so a rejected pairing is never proposed again.
- `merge` and `unmerge` are human decisions. A merge snapshots and moves the loser's identities, task
  assignments (through `task.update`), entity links and segment speaker links, then archives the loser
  (`actor.update`). Unmerge restores the snapshot where nothing has changed it since; references made to the
  winner after the merge that could belong to either actor become unresolved.
"""

import hashlib
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from ..domain import DomainError, advisory, execute
from ..models import Actor, SharedWorkspace, Task, WorkspaceMember, now
from .models import (
    AMBIGUOUS_NAMESPACES,
    ContextActorIdentity,
    ContextActorMerge,
    ContextDecision,
    ContextEntityLink,
    ContextSegment,
)
from .schema import DecisionOrigin, EntityRef, LinkStatus, Participant

NAMESPACES = ("email", "calendar_name", "granola_speaker", "linear_user", "alias", "organization")
STRONG_NAMESPACES = tuple(n for n in NAMESPACES if n not in AMBIGUOUS_NAMESPACES)
OWNER_ACTOR_NAME = "owner"  # productivity.defaults seeds the personal owner's actor under this name
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def normalize(namespace: str, value: str) -> str:
    """Lookup form of an identifier: emails casefolded; names NFKC, casefolded, single-spaced."""
    if namespace not in NAMESPACES:
        raise ValueError(f"unknown identity namespace {namespace!r}")
    value = unicodedata.normalize("NFKC", value).strip()
    if namespace == "linear_user":
        return value
    return " ".join(value.casefold().split())


# ---------------------------------------------------------------- decisions


@dataclass(frozen=True)
class DecisionInput:
    """Who decided and why. Merges, unmerges and durable negatives need `origin="human"`."""

    origin: DecisionOrigin
    principal: str
    reason: str = ""
    command_id: str | None = None


def _require_human(decision: DecisionInput, what: str) -> None:
    if decision.origin != "human":
        raise DomainError("NOT_AUTHORIZED", f"Only a person can {what}; agents may only propose it.", 403)


def _next_decision_seq(db, owner: str) -> int:
    from .decisions import next_decision_seq

    return next_decision_seq(db, owner)


def _record(db, owner, decision: DecisionInput, operation, target_type, target_id, *, negative=False, before=None,
            after=None, undoes_id=None) -> ContextDecision:
    row = ContextDecision(owner_id=owner, seq=_next_decision_seq(db, owner), target_type=target_type,
                          target_id=target_id, operation=operation, reason=decision.reason, negative=negative,
                          origin=decision.origin, principal=decision.principal, command_id=decision.command_id,
                          before=before or {}, after=after or {}, effective_at=now(), undoes_id=undoes_id)
    db.add(row)
    db.flush()
    return row


def _live(db, owner, operation, target_type, target_id=None) -> list[ContextDecision]:
    """Decisions of one operation that no later decision undoes."""
    undone = select(ContextDecision.undoes_id).where(ContextDecision.owner_id == owner,
                                                     ContextDecision.undoes_id.is_not(None))
    q = select(ContextDecision).where(ContextDecision.owner_id == owner, ContextDecision.operation == operation,
                                      ContextDecision.target_type == target_type, ContextDecision.id.not_in(undone))
    if target_id is not None:
        q = q.where(ContextDecision.target_id == target_id)
    return list(db.scalars(q.order_by(ContextDecision.seq)))


def _link_target(ref: EntityRef) -> str:
    if not ref.entity_id:
        raise ValueError("a rejected link names an entity")
    return f"{ref.entity_type}:{ref.entity_id}"


def _mention_key(mention: str) -> str:
    return normalize("alias", mention)


def reject_link(db, owner, mention: str, ref: EntityRef, decision: DecisionInput, *, scope: str | None = None,
                link_id: str | None = None) -> ContextDecision:
    """Durably reject resolving `mention` to `ref`. `scope` (a source ID) limits the rejection to one source;
    None rejects the pairing everywhere. The decision is `context_decisions` row with operation `reject_link`,
    target `("link", "<entity_type>:<entity_id>")` and the mention and scope in `before`."""
    _require_human(decision, "reject a link")
    row = _record(db, owner, decision, "reject_link", "link", _link_target(ref), negative=True,
                  before={"mention": _mention_key(mention), "scope": scope, "entity_type": ref.entity_type,
                          "entity_id": ref.entity_id, "link_id": link_id})
    if link_id:
        link = db.scalar(select(ContextEntityLink).where(ContextEntityLink.id == link_id,
                                                         ContextEntityLink.owner_id == owner).with_for_update())
        if link is not None:
            link.status, link.decision_id, link.revision, link.updated_at = "rejected", row.id, link.revision + 1, now()
    return row


@dataclass(frozen=True)
class RejectedLinks:
    """Durable negatives as `(scope, mention, entity_type, entity_id)`; a None scope applies everywhere."""

    pairs: frozenset[tuple[str | None, str, str, str]]

    def blocks(self, mention: str, entity_type: str, entity_id: str, scope: str | None = None) -> bool:
        key = _mention_key(mention)
        return (None, key, entity_type, entity_id) in self.pairs or (
            scope is not None and (scope, key, entity_type, entity_id) in self.pairs)


def rejected_links(db, owner) -> RejectedLinks:
    return RejectedLinks(frozenset(
        (d.before.get("scope"), d.before.get("mention", ""), d.before.get("entity_type", ""),
         d.before.get("entity_id", "")) for d in _live(db, owner, "reject_link", "link")))


def _pair(a: str, b: str) -> str:
    return "|".join(sorted((a, b)))  # same key as decisions.no_merge


def no_merge(db, owner, first: str, second: str, decision: DecisionInput) -> ContextDecision:
    """Durably record that two actors are different people; `merge_candidates` never proposes the pair again."""
    _require_human(decision, "decide two people are different")
    return _record(db, owner, decision, "no_merge", "actor_pair", _pair(first, second), negative=True,
                   before={"actors": sorted((first, second))})


# ---------------------------------------------------------------- identities


def resolve_exact(db, owner, namespace: str, value: str, source_scope: str | None = None) -> str | None:
    """The one actor a strong identifier names, or None. Ambiguous namespaces have no exact resolution."""
    if namespace not in STRONG_NAMESPACES:
        raise ValueError(f"{namespace} values may name several actors; use actors_for")
    q = select(ContextActorIdentity.actor_id).where(
        ContextActorIdentity.owner_id == owner, ContextActorIdentity.namespace == namespace,
        ContextActorIdentity.normalized == normalize(namespace, value))
    q = q.where(ContextActorIdentity.source_scope.is_(None) if source_scope is None
                else ContextActorIdentity.source_scope == source_scope)
    return db.scalar(q)


def actors_for(db, owner, namespace: str, value: str) -> list[str]:
    """Every active actor carrying a (possibly ambiguous) global identifier, oldest first."""
    return list(db.scalars(
        select(ContextActorIdentity.actor_id).join(Actor, Actor.id == ContextActorIdentity.actor_id)
        .where(ContextActorIdentity.owner_id == owner, ContextActorIdentity.namespace == namespace,
               ContextActorIdentity.normalized == normalize(namespace, value),
               ContextActorIdentity.source_scope.is_(None), Actor.archived.is_(False))
        .order_by(Actor.created_at, Actor.id)))


def add_identity(db, owner, actor_id: str, namespace: str, value: str, *, source_scope: str | None = None,
                 status: LinkStatus = "inferred", provenance: Mapping | None = None) -> bool:
    """Attach an identifier; idempotent. False when a strong value already names another actor."""
    normalized = normalize(namespace, value)
    db.execute(insert(ContextActorIdentity).values(
        owner_id=owner, actor_id=actor_id, namespace=namespace, value=value.strip()[:500],
        normalized=normalized[:500], source_scope=source_scope, status=status, provenance=dict(provenance or {}),
    ).on_conflict_do_nothing())
    if namespace in AMBIGUOUS_NAMESPACES:
        return True
    return resolve_exact(db, owner, namespace, value, source_scope) == actor_id


def owner_actor(db, owner) -> str:
    """The note-taker's actor: the creator's member actor in a shared workspace, else the `owner` person.
    Created through `actor.create` only if the workspace has never been seeded."""
    workspace = db.get(SharedWorkspace, owner)
    if workspace is not None:
        member = db.get(WorkspaceMember, (owner, workspace.creator_id))
        if member is not None and member.actor_id:
            return member.actor_id
    found = db.scalar(select(Actor.id).where(Actor.owner_id == owner, Actor.name == OWNER_ACTOR_NAME,
                                             Actor.kind == "person")
                      .order_by(Actor.archived, Actor.created_at, Actor.id).limit(1))
    if found:
        return found
    return execute(db, owner, "context-owner-actor", "actor.create", {"name": OWNER_ACTOR_NAME})["data"]["id"]


def _create_actor(db, owner, name: str, email: str) -> str:
    from ..assignees import active_actors_named
    from ..models import Command

    name = name[:100]
    # A deterministic command ID per (email, name) returns the same actor if seeding repeats.
    digest = hashlib.sha256(f"{normalize('email', email)}\n{name}".encode()).hexdigest()[:40]
    command_id = f"context-identity:{digest}"
    previous = db.get(Command, (owner, command_id))
    if previous is not None:
        return previous.result["data"]["id"]
    arguments = {"name": name, "allow_duplicate": bool(active_actors_named(db, owner, name))}
    return execute(db, owner, command_id, "actor.create", arguments)["data"]["id"]


# ---------------------------------------------------------------- seeding from a meeting


@dataclass(frozen=True)
class IdentityCandidate:
    """One meeting participant resolved against actors.

    `status`: `exact` (an email identity, or the owner's own calendar entry), `inferred` (a display name carried
    by exactly one active actor) or `unresolved` (`candidates` lists the actors the name could be; empty when
    unknown). `duplicates` lists other active actors with the same display name, for merge review."""

    actor_id: str | None
    status: LinkStatus
    name: str | None = None
    email: str | None = None
    speaker_label: str | None = None
    role: str = "attendee"
    is_owner: bool = False
    created: bool = False
    candidates: tuple[str, ...] = ()
    duplicates: tuple[str, ...] = ()


@dataclass
class _Person:
    name: str | None = None
    calendar_name: str | None = None
    email: str | None = None
    speaker_label: str | None = None
    role: str = "attendee"
    is_self: bool = False


def _attendee(item) -> _Person | None:
    """A calendar attendee as Google returns it (`email`, `displayName`, `self`, `organizer`, `resource`), a
    plain mapping with `name`, or a Participant. Rooms and other resources are skipped."""
    if isinstance(item, Participant):
        return _Person(calendar_name=item.name, email=item.email, role=item.role)
    if item.get("resource"):
        return None
    return _Person(calendar_name=item.get("displayName") or item.get("name"), email=item.get("email"),
                   role="organizer" if item.get("organizer") else "attendee", is_self=bool(item.get("self")))


def _people(participants: Sequence[Participant], calendar_attendees: Sequence) -> list[_Person]:
    """Calendar attendees, then envelope participants, folded together by email."""
    people: list[_Person] = []
    by_email: dict[str, _Person] = {}
    found = [_attendee(item) for item in calendar_attendees or []]
    found += [_Person(name=p.name, email=p.email, speaker_label=p.speaker_label, role=p.role)
              for p in participants or []]
    for person in found:
        if person is None:
            continue
        email = (person.email or "").strip()
        person.email = email if EMAIL.match(email) else None
        if not (person.email or person.name or person.calendar_name):
            continue
        known = by_email.get(normalize("email", person.email)) if person.email else None
        if known is None:
            if person.email:
                by_email[normalize("email", person.email)] = person
            people.append(person)
            continue
        known.name = known.name or person.name
        known.calendar_name = known.calendar_name or person.calendar_name
        known.speaker_label = known.speaker_label or person.speaker_label
        known.is_self = known.is_self or person.is_self
        if person.role == "organizer":
            known.role = "organizer"
    return people


def _display_name(person: _Person) -> str:
    return (person.name or person.calendar_name or (person.email or "").split("@")[0] or "Unknown").strip()


def _named(db, owner, name: str) -> list[str]:
    """Active actors called `name` exactly or carrying it as an alias or calendar name."""
    from ..assignees import active_actors_named

    ids = [a.id for a in active_actors_named(db, owner, name)]
    for namespace in AMBIGUOUS_NAMESPACES:
        ids += [i for i in actors_for(db, owner, namespace, name) if i not in ids]
    return ids


def seed_candidates(db, owner, source_id, participants, calendar_attendees) -> list[IdentityCandidate]:
    """Resolve a meeting's attendees to actors (see the module docstring). Records email, alias, calendar-name
    and source-scoped speaker-label identities for resolved people. Idempotent for the same inputs."""
    me = owner_actor(db, owner)
    out = []
    for person in _people(participants, calendar_attendees):
        name = _display_name(person)
        provenance = {"source_id": source_id, "via": "seed_candidates"}
        created = False
        if person.email:
            advisory(db, f"context-identity:{owner}:{normalize('email', person.email)}")
            actor_id = resolve_exact(db, owner, "email", person.email) or (me if person.is_self else None)
            if actor_id is None:
                actor_id, created = _create_actor(db, owner, name, person.email), True
            add_identity(db, owner, actor_id, "email", person.email, status="exact", provenance=provenance)
            status: LinkStatus = "exact"
            candidates: tuple[str, ...] = ()
        else:
            candidates = tuple(_named(db, owner, name))
            actor_id = candidates[0] if len(candidates) == 1 else None
            status = "inferred" if actor_id else "unresolved"
            candidates = () if actor_id else candidates
        if actor_id:
            if person.name:
                add_identity(db, owner, actor_id, "alias", person.name, provenance=provenance)
            if person.calendar_name:
                add_identity(db, owner, actor_id, "calendar_name", person.calendar_name, provenance=provenance)
            if person.speaker_label and status == "exact":
                add_identity(db, owner, actor_id, "granola_speaker", person.speaker_label, source_scope=source_id,
                             status="exact", provenance=provenance)
        duplicates = tuple(i for i in _named(db, owner, name) if i != actor_id) if status == "exact" else ()
        out.append(IdentityCandidate(
            actor_id=actor_id, status=status, name=person.name or person.calendar_name, email=person.email,
            speaker_label=person.speaker_label, role=person.role, is_owner=actor_id is not None and actor_id == me,
            created=created, candidates=candidates, duplicates=duplicates))
    return out


# ---------------------------------------------------------------- merge proposals


@dataclass(frozen=True)
class MergeProposal:
    """Two active people who may be the same person. Agents propose; only a human decision merges."""

    actor_ids: tuple[str, str]
    reason: str


def merge_candidates(db, owner) -> list[MergeProposal]:
    """Active people sharing a display name or alias, unless both carry email identities (distinct strong
    identifiers mean distinct people), the pair has a live `no_merge` decision, or one was merged away."""
    actors = {a.id: a for a in db.scalars(select(Actor).where(Actor.owner_id == owner, Actor.archived.is_(False),
                                                               Actor.kind == "person"))}
    names: dict[str, set[str]] = {}
    for actor in actors.values():
        names.setdefault(normalize("alias", actor.name), set()).add(actor.id)
    for identity in db.scalars(select(ContextActorIdentity).where(
            ContextActorIdentity.owner_id == owner, ContextActorIdentity.namespace.in_(AMBIGUOUS_NAMESPACES))):
        if identity.actor_id in actors:
            names.setdefault(identity.normalized, set()).add(identity.actor_id)
    emailed = set(db.scalars(select(ContextActorIdentity.actor_id).where(
        ContextActorIdentity.owner_id == owner, ContextActorIdentity.namespace == "email")))
    refused = {d.target_id for d in _live(db, owner, "no_merge", "actor_pair")}
    seen, out = set(), []
    for name, ids in sorted(names.items()):
        ordered = sorted(ids, key=lambda i: (actors[i].created_at, i))
        for i, first in enumerate(ordered):
            for second in ordered[i + 1:]:
                pair = _pair(first, second)
                if pair in seen or pair in refused or (first in emailed and second in emailed):
                    continue
                seen.add(pair)
                out.append(MergeProposal((first, second), f"Both are called {name!r}."))
    return out


# ---------------------------------------------------------------- merge and unmerge


def _active_merge(db, owner, actor_id) -> ContextActorMerge | None:
    return db.scalar(select(ContextActorMerge).where(
        ContextActorMerge.owner_id == owner, ContextActorMerge.undone_at.is_(None),
        (ContextActorMerge.loser_actor_id == actor_id) | (ContextActorMerge.winner_actor_id == actor_id)).limit(1))


def _set_assignee(db, owner, command_id, task_id, actor_id):
    task = db.get(Task, task_id)
    execute(db, owner, command_id, "task.update",
            {"task_id": task_id, "expected_revision": task.revision, "assignee_id": actor_id})


def _set_archived(db, owner, command_id, actor_id, archived):
    actor = db.get(Actor, actor_id)
    if actor.archived != archived:
        execute(db, owner, command_id, "actor.update", {"actor_id": actor_id, "expected_revision": actor.revision,
                                                        "archived": archived, "allow_duplicate": True})


def merge(db, owner, winner: str, loser: str, decision: DecisionInput) -> ContextActorMerge:
    """Merge `loser` into `winner` (a human decision). The snapshot records every reassigned row so `unmerge`
    can restore it."""
    _require_human(decision, "merge people")
    if winner == loser:
        raise DomainError("INVALID_ARGUMENT", "Choose two different people to merge.")
    advisory(db, f"context-actors:{owner}")
    actors = {a.id: a for a in db.scalars(select(Actor).where(Actor.owner_id == owner, Actor.id.in_((winner, loser)))
                                          .with_for_update())}
    if set(actors) != {winner, loser}:
        raise DomainError("NOT_FOUND", "That person is no longer available.", 404)
    if actors[winner].archived or actors[loser].archived:
        raise DomainError("INVALID_ARGUMENT", "Only active people can be merged.")
    if _active_merge(db, owner, loser):
        raise DomainError("REVISION_CONFLICT", "Undo this person's earlier merge first.", 409)
    row = ContextActorMerge(owner_id=owner, winner_actor_id=winner, loser_actor_id=loser)
    db.add(row)
    db.flush()

    identities = []
    for identity in db.scalars(select(ContextActorIdentity).where(
            ContextActorIdentity.owner_id == owner, ContextActorIdentity.actor_id == loser).with_for_update()):
        saved = {k: getattr(identity, k) for k in ("id", "namespace", "value", "normalized", "source_scope",
                                                   "status", "provenance")}
        duplicate = db.scalar(select(ContextActorIdentity.id).where(
            ContextActorIdentity.owner_id == owner, ContextActorIdentity.actor_id == winner,
            ContextActorIdentity.namespace == identity.namespace,
            ContextActorIdentity.normalized == identity.normalized,
            ContextActorIdentity.source_scope.is_not_distinct_from(identity.source_scope)))
        if duplicate:
            db.delete(identity)
        else:
            identity.actor_id, identity.updated_at = winner, now()
        identities.append({**saved, "action": "deleted" if duplicate else "moved"})
    db.flush()

    links = []
    for link in db.scalars(select(ContextEntityLink).where(
            ContextEntityLink.owner_id == owner, ContextEntityLink.target_type == "actor",
            ContextEntityLink.target_id == loser).with_for_update()):
        links.append({"id": link.id, "status": link.status})
        link.target_id, link.revision, link.updated_at = winner, link.revision + 1, now()

    segments = []
    for segment in db.scalars(select(ContextSegment).where(
            ContextSegment.owner_id == owner, ContextSegment.speaker_actor_id == loser).with_for_update()):
        segments.append({"id": segment.id, "status": segment.speaker_link_status})
        segment.speaker_actor_id = winner
        segment.speaker_provenance = {**(segment.speaker_provenance or {}), "merge_id": row.id}

    tasks = list(db.scalars(select(Task.id).where(Task.owner_id == owner, Task.assignee_id == loser)
                            .order_by(Task.id)))
    for task_id in tasks:
        _set_assignee(db, owner, f"context-merge:{row.id}:{task_id}", task_id, winner)
    _set_archived(db, owner, f"context-merge:{row.id}:archive", loser, True)

    row.snapshot = {"identities": identities, "tasks": tasks, "entity_links": links, "segments": segments,
                    "loser_name": actors[loser].name}
    recorded = _record(db, owner, decision, "merge", "actor", loser,
                       before={"winner": winner, "loser": loser}, after={"merge_id": row.id, "winner": winner})
    row.decision_id = recorded.id
    db.flush()
    return row


def _names(db, owner, actor_id: str, extra: Sequence[str] = ()) -> set[str]:
    actor = db.get(Actor, actor_id)
    values = {normalize("alias", actor.name)} | {normalize("alias", v) for v in extra}
    values |= set(db.scalars(select(ContextActorIdentity.normalized).where(
        ContextActorIdentity.owner_id == owner, ContextActorIdentity.actor_id == actor_id,
        ContextActorIdentity.namespace.in_(AMBIGUOUS_NAMESPACES + ("granola_speaker",)))))
    return values


def _post_merge_target(mention: str | None, winner_names: set[str], loser_names: set[str], winner: str,
                       loser: str) -> str | None:
    """Where a reference made to the winner after the merge belongs: one actor when its mention names only
    that actor, else None (ambiguous, left unresolved)."""
    key = normalize("alias", mention) if mention else ""
    in_winner, in_loser = key in winner_names, key in loser_names
    if in_winner and not in_loser:
        return winner
    if in_loser and not in_winner:
        return loser
    return None


def unmerge(db, owner, merge_id: str, decision: DecisionInput) -> ContextActorMerge:
    """Undo a merge (a human decision): restore the loser and every snapshot assignment still pointing at the
    winner. Rows changed since the merge keep their new value. References made to the winner after the merge
    are reassigned only when their mention names one of the two actors; otherwise they become unresolved."""
    _require_human(decision, "undo a merge")
    advisory(db, f"context-actors:{owner}")
    row = db.scalar(select(ContextActorMerge).where(ContextActorMerge.id == merge_id,
                                                    ContextActorMerge.owner_id == owner).with_for_update())
    if row is None:
        raise DomainError("NOT_FOUND", "That merge is no longer available.", 404)
    if row.undone_at is not None:
        raise DomainError("REVISION_CONFLICT", "That merge was already undone.", 409)
    winner, loser, snap = row.winner_actor_id, row.loser_actor_id, row.snapshot or {}
    _set_archived(db, owner, f"context-unmerge:{row.id}:restore", loser, False)

    for saved in snap.get("identities", []):
        if saved["action"] == "moved":
            identity = db.get(ContextActorIdentity, saved["id"])
            if identity is not None and identity.actor_id == winner:
                identity.actor_id, identity.updated_at = loser, now()
        else:
            db.execute(insert(ContextActorIdentity).values(
                owner_id=owner, actor_id=loser, **{k: saved[k] for k in (
                    "id", "namespace", "value", "normalized", "source_scope", "status", "provenance")},
            ).on_conflict_do_nothing())
    db.flush()

    for task_id in snap.get("tasks", []):
        task = db.get(Task, task_id)
        if task is not None and task.assignee_id == winner:
            _set_assignee(db, owner, f"context-unmerge:{row.id}:{task_id}", task_id, loser)

    restored_links = {item["id"]: item for item in snap.get("entity_links", [])}
    restored_segments = {item["id"]: item for item in snap.get("segments", [])}
    winner_names = _names(db, owner, winner)
    loser_names = _names(db, owner, loser, [snap.get("loser_name", "")])
    merged_at: datetime = row.created_at

    for link in db.scalars(select(ContextEntityLink).where(
            ContextEntityLink.owner_id == owner, ContextEntityLink.target_type == "actor",
            ContextEntityLink.target_id == winner).with_for_update()):
        if link.id in restored_links:
            link.target_id, link.status = loser, restored_links[link.id]["status"]
        elif link.updated_at >= merged_at:
            target = _post_merge_target(link.mention, winner_names, loser_names, winner, loser)
            if target == winner:
                continue
            link.target_id, link.status = target, ("inferred" if target else "unresolved")
            if target is None:
                link.target_type = None
        else:
            continue
        link.revision, link.updated_at = link.revision + 1, now()

    for segment in db.scalars(select(ContextSegment).where(
            ContextSegment.owner_id == owner, ContextSegment.speaker_actor_id == winner).with_for_update()):
        if segment.id in restored_segments:
            segment.speaker_actor_id = loser
            segment.speaker_link_status = restored_segments[segment.id]["status"]
            segment.speaker_provenance = {k: v for k, v in (segment.speaker_provenance or {}).items()
                                          if k != "merge_id"}
        elif segment.created_at >= merged_at:
            target = _post_merge_target(segment.speaker_label, winner_names, loser_names, winner, loser)
            if target != winner:
                segment.speaker_actor_id = target
                segment.speaker_link_status = "inferred" if target else "unresolved"
                segment.speaker_provenance = {**(segment.speaker_provenance or {}), "unmerge_id": row.id}

    _record(db, owner, decision, "unmerge", "actor_merge", row.id,
                       before={"winner": winner, "loser": loser}, after={"merge_id": row.id},
                       undoes_id=row.decision_id)
    row.undone_at, row.updated_at = now(), now()
    db.flush()
    return row
