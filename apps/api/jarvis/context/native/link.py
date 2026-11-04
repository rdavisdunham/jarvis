"""Native entity linking for meeting mentions and speaker buckets (plan_final.md section 3, People).

`link_mentions` resolves each mention in order:
1. Exact identifiers: an email with an identity, a Linear key (ABC-123) of the connected workspace, or an
   Eridani record ID. These are `exact`.
2. Scored candidates (clients, projects, areas, goals, actors): the better of a lexical score over names and
   aliases and, when `LinkContext.embed` is supplied, the cosine against the record's `SearchDocument` vectors.
   Actor candidates narrow to the meeting's attendees when any attendee matches. The top candidate is
   `inferred` when it reaches AUTO_THRESHOLD and leads the runner-up by MARGIN; a candidate at or above
   CANDIDATE_FLOOR otherwise leaves the mention `unresolved` with the ranked candidates attached; below the
   floor it is an unresolved mention with none.
Durable rejections (`identity.rejected_links`) remove a candidate; if a rejected candidate would have ranked
first, nothing is inferred (as in `agent_autolink`). Whether an unresolved mention deserves a Question is the
caller's decision: `ask` is set only for `Mention.consequential` mentions that have candidates.

The constants are provisional, chosen for meeting mentions on the fixtures in tests/context/test_link.py and
to be recalibrated on adjudicated meetings (plan section 11, 95% precision for automatic links). They differ
from `agent_autolink`'s because mentions are short names, not session summaries.

`resolve_speakers` maps one source's diarization labels to actors: owner-attributed lines go to the owner's
actor (`exact`); a label with a source-scoped `granola_speaker` identity is `exact`; a label naming exactly one
attendee, or a speaker introducing themselves with one attendee's name, is `inferred`; anything else stays
unresolved. Labels never carry across sources.
"""

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from pydantic import Field
from sqlalchemy import select

from ...memory_learning import EMBEDDING_MODEL, cosine
from ...models import Actor, Area, Goal, LinearConnection, LinearIssue, Project, Task
from ...search_models import SearchDocument
from ...structure_models import StructureRecord
from .. import identity
from ..schema import Contract, EntityRef, EntityType, EvidenceRef, LinkProposal, Participant, Segment

AUTO_THRESHOLD = 0.85  # provisional
MARGIN = 0.15  # provisional
CANDIDATE_FLOOR = 0.5  # provisional
MAX_CANDIDATES = 5
SCORED_TYPES: tuple[EntityType, ...] = ("client", "project", "area", "goal", "actor")
RECORD_TYPES = ("client", "project", "area", "goal")

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
LINEAR_KEY = re.compile(r"\b[A-Z][A-Z0-9]{1,9}-\d{1,6}\b")
RECORD_ID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
WORD = re.compile(r"\w+")
INTRO = re.compile(r"\b(?:I'm|I am|this is|my name is)\s+([A-Z][\w'-]+)")
OWNER_LABEL = "me"  # bucket for owner-attributed lines that carry no speaker label


@dataclass(frozen=True)
class Mention:
    """A span an extraction named. `entity_types` limits the candidates (empty: every scored type).
    `consequential` marks mentions whose ambiguity matters (e.g. who owns a commitment)."""

    text: str
    entity_types: tuple[EntityType, ...] = ()
    evidence: tuple[EvidenceRef, ...] = ()
    consequential: bool = False


@dataclass(frozen=True)
class LinkContext:
    """`scope` is the source ID durable rejections may be scoped to; `attendees` are the meeting's
    seed_candidates; `embed` turns texts into vectors (absent or failing: lexical only)."""

    scope: str | None = None
    attendees: Sequence[identity.IdentityCandidate] = ()
    embed: Callable[[list[str]], list[list[float]]] | None = None


class RankedCandidate(Contract):
    ref: EntityRef
    name: str
    score: float


class MentionLink(LinkProposal):
    """A LinkProposal with the ranked alternatives and whether the caller should ask about it."""

    candidates: list[RankedCandidate] = Field(default_factory=list)
    ask: bool = False


@dataclass
class _Entity:
    ref: EntityRef
    name: str
    names: set[str] = field(default_factory=set)
    vectors: list[list[float]] = field(default_factory=list)


def _words(text: str) -> set[str]:
    return set(WORD.findall(text.casefold()))


def lexical(mention: str, names: Iterable[str]) -> float:
    """1.0 for an exact normalized name; 0.8 when one name's words contain the other's; else 0.7 x Jaccard."""
    key = identity.normalize("alias", mention)
    words = _words(mention)
    best = 0.0
    for name in names:
        if name == key:
            return 1.0
        other = _words(name)
        if not words or not other:
            continue
        if words <= other or other <= words:
            best = max(best, 0.8)
        else:
            best = max(best, 0.7 * len(words & other) / len(words | other))
    return best


def choose(scored: dict[str, float], rejected: frozenset[str] = frozenset()) -> tuple[str, float, float] | None:
    """scored: {key: score}. The automatic pick as (key, score, margin), or None (as agent_autolink.choose)."""
    ranked = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
    if not ranked:
        return None
    (key, top), runner = ranked[0], (ranked[1][1] if len(ranked) > 1 else 0.0)
    if key in rejected or top < AUTO_THRESHOLD or top - runner < MARGIN:
        return None
    return key, round(top, 4), round(top - runner, 4)


# ---------------------------------------------------------------- candidates


def _entities(db, owner, types: set[str]) -> list[_Entity]:
    out: dict[tuple[str, str], _Entity] = {}
    if "actor" in types:
        for actor in db.scalars(select(Actor).where(Actor.owner_id == owner, Actor.archived.is_(False))):
            out["actor", actor.id] = _Entity(EntityRef(entity_type="actor", entity_id=actor.id), actor.name,
                                             {identity.normalize("alias", actor.name)})
        for row in db.scalars(select(identity.ContextActorIdentity).where(
                identity.ContextActorIdentity.owner_id == owner,
                identity.ContextActorIdentity.namespace.in_(identity.AMBIGUOUS_NAMESPACES))):
            if ("actor", row.actor_id) in out:
                out["actor", row.actor_id].names.add(row.normalized)
    record_types = [t for t in RECORD_TYPES if t in types]
    if record_types:
        for row in db.scalars(select(StructureRecord).where(
                StructureRecord.owner_id == owner, StructureRecord.type_id.in_(record_types),
                StructureRecord.archived.is_(False))):
            out[row.type_id, row.id] = _Entity(EntityRef(entity_type=row.type_id, entity_id=row.id), row.title,
                                               {identity.normalize("alias", row.title)})
        # Planner rows not (yet) mirrored into the record registry.
        for kind, model in (("project", Project), ("area", Area), ("goal", Goal)):
            if kind in record_types:
                for row in db.scalars(select(model).where(model.owner_id == owner, model.archived.is_(False))):
                    out.setdefault((kind, row.id), _Entity(EntityRef(entity_type=kind, entity_id=row.id), row.name,
                                                           {identity.normalize("alias", row.name)}))
    return list(out.values())


def _attach_vectors(db, owner, entities: list[_Entity]) -> None:
    keys = {"record:" + e.ref.entity_id: e for e in entities if e.ref.entity_type != "actor"}
    if not keys:
        return
    for document in db.scalars(select(SearchDocument).where(
            SearchDocument.owner_id == owner, SearchDocument.embedding_model == EMBEDDING_MODEL,
            SearchDocument.target_key.in_(list(keys)))):
        keys[document.target_key].vectors = document.vectors or []


def _exact(db, owner, text: str) -> EntityRef | None:
    for email in EMAIL.findall(text):
        actor_id = identity.resolve_exact(db, owner, "email", email)
        if actor_id:
            return EntityRef(entity_type="actor", entity_id=actor_id)
    keys = LINEAR_KEY.findall(text)
    conn = db.get(LinearConnection, owner)
    if keys and conn is not None and conn.enabled:
        issue = db.scalar(select(LinearIssue).where(
            LinearIssue.owner_id == owner, LinearIssue.workspace_id == conn.workspace_id,
            LinearIssue.snapshot["identifier"].astext == keys[0]))
        if issue is not None:
            return EntityRef(entity_type="linear_issue", entity_id=issue.id)
    for record_id in RECORD_ID.findall(text):
        record = db.get(StructureRecord, record_id)
        if record is not None and record.owner_id == owner and record.type_id in RECORD_TYPES + ("task",):
            return EntityRef(entity_type=record.type_id, entity_id=record.id)
        for kind, model in (("actor", Actor), ("task", Task), ("project", Project), ("area", Area), ("goal", Goal)):
            row = db.get(model, record_id)
            if row is not None and row.owner_id == owner:
                return EntityRef(entity_type=kind, entity_id=row.id)
    return None


# ---------------------------------------------------------------- linking


def _unresolved(mention: Mention, entity_type: EntityType, candidates: list[RankedCandidate]) -> MentionLink:
    return MentionLink(mention=mention.text[:300], candidate=EntityRef(entity_type=entity_type,
                                                                       mention=mention.text[:300]),
                       status="unresolved", score=candidates[0].score if candidates else None,
                       evidence=list(mention.evidence), candidates=candidates,
                       ask=mention.consequential and bool(candidates))


def link_mentions(db, owner, mentions: Sequence[Mention | str], context: LinkContext | None = None
                  ) -> list[LinkProposal]:
    """One proposal per mention, in order (see the module docstring). Returned proposals are MentionLinks."""
    context = context or LinkContext()
    mentions = [m if isinstance(m, Mention) else Mention(text=m) for m in mentions]
    rejected = identity.rejected_links(db, owner)
    types = {t for m in mentions for t in (m.entity_types or SCORED_TYPES) if t in SCORED_TYPES}
    entities = _entities(db, owner, types)
    vectors: dict[str, list[float]] = {}
    if context.embed is not None and entities:
        _attach_vectors(db, owner, entities)
        texts = sorted({m.text for m in mentions})
        try:
            vectors = dict(zip(texts, context.embed(texts), strict=True))
        except Exception:  # noqa: BLE001 - linking falls back to lexical scores when embedding fails
            vectors = {}
    attendees = {a.actor_id for a in context.attendees if a.actor_id}

    out: list[LinkProposal] = []
    for mention in mentions:
        allowed = mention.entity_types or SCORED_TYPES
        default_type: EntityType = allowed[0] if len(allowed) == 1 else "actor"
        exact = _exact(db, owner, mention.text)
        if exact is not None and (not mention.entity_types or exact.entity_type in mention.entity_types) and not \
                rejected.blocks(mention.text, exact.entity_type, exact.entity_id, context.scope):
            out.append(MentionLink(mention=mention.text[:300], candidate=exact, status="exact", score=1.0,
                                   evidence=list(mention.evidence)))
            continue
        vector = vectors.get(mention.text)
        scored: dict[str, tuple[_Entity, float]] = {}
        for entity in entities:
            if entity.ref.entity_type not in allowed:
                continue
            score = lexical(mention.text, entity.names)
            if vector:
                score = max(score, max((cosine(vector, v) for v in entity.vectors), default=0.0))
            if score >= CANDIDATE_FLOOR:
                scored[f"{entity.ref.entity_type}:{entity.ref.entity_id}"] = (entity, score)
        present = {k for k, (e, _) in scored.items() if e.ref.entity_type == "actor" and e.ref.entity_id in attendees}
        if present:  # an attendee with this name outranks namesakes who were not in the meeting
            scored = {k: v for k, v in scored.items() if v[0].ref.entity_type != "actor" or k in present}
        blocked = frozenset(k for k, (e, _) in scored.items()
                            if rejected.blocks(mention.text, e.ref.entity_type, e.ref.entity_id, context.scope))
        picked = choose({k: s for k, (_, s) in scored.items()}, blocked)
        ranked = sorted((v for k, v in scored.items() if k not in blocked), key=lambda v: (-v[1], v[0].ref.entity_id))
        candidates = [RankedCandidate(ref=e.ref, name=e.name[:300], score=round(s, 4))
                      for e, s in ranked[:MAX_CANDIDATES]]
        if picked is None:
            entity_type = candidates[0].ref.entity_type if candidates else default_type
            out.append(_unresolved(mention, entity_type, candidates))
            continue
        entity = scored[picked[0]][0]
        out.append(MentionLink(mention=mention.text[:300], candidate=entity.ref, status="inferred", score=picked[1],
                               margin=picked[2], evidence=list(mention.evidence), candidates=candidates))
    return out


# ---------------------------------------------------------------- speaker buckets


def _evidence(segment: Segment, quote: str) -> EvidenceRef:
    start = segment.text.find(quote)
    return EvidenceRef(segment_seq=segment.seq, quote=quote, char_start=start, char_end=start + len(quote))


def resolve_speakers(db, owner, source_id: str, segments: Sequence[Segment],
                     attendees: Sequence[identity.IdentityCandidate | Participant] = ()) -> list[LinkProposal]:
    """One proposal per distinct speaker label in `segments`, in order of first appearance."""
    rejected = identity.rejected_links(db, owner)
    resolved = [a for a in attendees if isinstance(a, identity.IdentityCandidate) and a.actor_id]
    names: dict[str, set[str]] = {}
    for attendee in resolved:
        for value in (attendee.name, attendee.speaker_label):
            if value:
                key = identity.normalize("alias", value)
                names.setdefault(key, set()).add(attendee.actor_id)
                first = key.split(" ")[0]
                if first != key:
                    names.setdefault(first, set()).add(attendee.actor_id)
    me: str | None = None
    by_label: dict[str, list[Segment]] = {}
    for segment in segments:
        label = segment.speaker_label or (OWNER_LABEL if segment.attribution == "owner" else None)
        if label:
            by_label.setdefault(label, []).append(segment)

    def allowed(label, actor_id):
        return actor_id and not rejected.blocks(label, "actor", actor_id, source_id)

    out: list[LinkProposal] = []
    for label, lines in by_label.items():
        status, actor_id, evidence = "unresolved", None, []
        if all(s.attribution == "owner" for s in lines):
            me = me or identity.owner_actor(db, owner)
            if allowed(label, me):
                status, actor_id = "exact", me
        if actor_id is None:
            scoped = identity.resolve_exact(db, owner, "granola_speaker", label, source_id)
            if allowed(label, scoped):
                status, actor_id = "exact", scoped
        if actor_id is None:
            matches = names.get(identity.normalize("alias", label), set())
            if len(matches) == 1 and allowed(label, next(iter(matches))):
                status, actor_id = "inferred", next(iter(matches))
        if actor_id is None:
            for segment in lines:
                intro = INTRO.search(segment.text)
                matches = names.get(identity.normalize("alias", intro.group(1)), set()) if intro else set()
                if len(matches) == 1 and allowed(label, next(iter(matches))):
                    status, actor_id = "inferred", next(iter(matches))
                    evidence = [_evidence(segment, intro.group(0))]
                    break
        ref = EntityRef(entity_type="actor", entity_id=actor_id) if actor_id else \
            EntityRef(entity_type="actor", mention=label[:300])
        out.append(LinkProposal(mention=label[:300], candidate=ref, status=status, evidence=evidence))
    return out
