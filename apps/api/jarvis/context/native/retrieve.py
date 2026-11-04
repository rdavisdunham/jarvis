"""NativeRetriever: the native `Retriever` engine (plan_final.md section 6, "Retrieval implementation").

One SQL statement ranks two independent candidate legs and fuses them by reciprocal rank fusion (k=60):

- lexical: `websearch_to_tsquery` over assertion and (with `include_raw`) segment `tsv`, ranked by `ts_rank_cd`;
- semantic: exact pgvector cosine distance over assertion and (with `include_raw`) chunk embeddings recorded by
  the current embedding model.

The scope predicate (owner, entities, kinds, lifecycle, `valid_at`, `known_at`, deleted sources) is part of each
leg's candidate selection, so ranking and top-k only ever see authorized, in-scope rows, and neither leg filters
the other: a semantic match needs no lexical overlap. Assertion lifecycle and time use `temporal`'s clauses; with
`known_at` an assertion is searched and returned as the version recorded then (its embedding only while that
version's statement equals the head's).

Raw hits (`include_raw`) come from the source's current revision (with `known_at`, the latest revision received by
then). They are annotated by joining evidence to assertion lifecycle: a raw hit supporting a retracted or
superseded assertion carries that status and the successor's id. A segment (the exact passage) supporting an
ended interpretation the scope would not show is excluded, like the assertion itself, so raw retrieval cannot
revive it; a chunk spans many segments, so it is kept and annotated. Chunk hits cite their segment range.

The query vector comes from `memory_learning.query_embedding` under cost feature `context_embeddings`. When it
fails the lexical leg answers alone and every hit carries `rank["semantic_unavailable"] == 1.0`.
"""

from collections import defaultdict
from collections.abc import Callable
from datetime import datetime

from sqlalchemy import and_, case, exists, false, func, literal, literal_column, or_, select, union_all
from sqlalchemy.orm import aliased

from ... import memory_learning
from ...cost_features import feature
from ...models import now
from .. import temporal
from ..models import (
    DIMENSIONS,
    ContextAssertion,
    ContextAssertionVersion,
    ContextChunk,
    ContextEntityLink,
    ContextEvidence,
    ContextSegment,
    ContextSource,
    ContextSourceRevision,
)
from ..schema import ENTITY_TYPES, Citation, EntityRef, Hit, RetrieveQuery, ScopePredicate

RRF_K = 60
MIN_CANDIDATES = 50
MAX_CANDIDATES = 500
SNIPPET = 300
RESOLVED = ("exact", "inferred", "confirmed")
# Higher wins when several assertions annotate one raw hit: a stale interpretation is never hidden by a live one.
SEVERITY = {"retracted": 4, "superseded": 3, "contested": 2, "current": 1}
QueryEmbedder = Callable[[str, str], list[float]]


@feature("context_embeddings")
def query_vector(owner: str, text: str) -> list[float]:
    return memory_learning.query_embedding(owner, text[:2000])


class NativeRetriever:
    """`at` stands in for now (tests); `embed(owner, text)` returns the query vector or raises."""

    def __init__(self, db, embed: QueryEmbedder | None = None, *, at: datetime | None = None):
        self.db, self.embed, self.at = db, embed or query_vector, at

    def retrieve(self, query: RetrieveQuery) -> list[Hit]:
        scope, at = query.scope, self.at or now()
        vector = self._vector(scope.owner_id, query.text)
        cand = min(max(query.limit * 5, MIN_CANDIDATES), MAX_CANDIDATES)
        tsq = func.websearch_to_tsquery(literal_column("'english'::regconfig"), query.text)
        docs = _assertion_docs(scope, at).subquery("assertion_docs")
        lexical = [_lexical("assertion", docs.c.id, docs.c.tsv, tsq, cand)]
        semantic = [] if vector is None else [_semantic("assertion", docs, vector, cand)]
        if query.include_raw:
            segments = _segment_docs(scope, at).subquery("segment_docs")
            lexical.append(_lexical("segment", segments.c.id, segments.c.tsv, tsq, cand))
            if vector is not None:
                semantic.append(_semantic("chunk", _chunk_docs(scope, at).subquery("chunk_docs"), vector, cand))
        legs = [_ranked("lexical", lexical)] + ([_ranked("semantic", semantic)] if semantic else [])
        fused = union_all(*legs).subquery("legs")

        def leg(name, column):
            return func.min(case((fused.c.leg == name, column)))

        rrf = func.sum(1.0 / (RRF_K + fused.c.rank))
        rows = self.db.execute(
            select(fused.c.ref_type, fused.c.id, rrf.label("rrf"),
                   leg("lexical", fused.c.rank).label("lexical_rank"), leg("lexical", fused.c.score).label("lexical"),
                   leg("semantic", fused.c.rank).label("semantic_rank"),
                   leg("semantic", fused.c.score).label("semantic"))
            .group_by(fused.c.ref_type, fused.c.id)
            .order_by(rrf.desc(), fused.c.ref_type, fused.c.id).limit(query.limit)).all()
        return self._hits(scope, at, rows, semantic_available=vector is not None)

    def _vector(self, owner: str, text: str) -> list[float] | None:
        try:
            vector = self.embed(owner, text)
        except Exception:  # noqa: BLE001 - lexical retrieval survives an embedding outage
            return None
        return [float(x) for x in vector] if vector is not None and len(vector) == DIMENSIONS else None

    # ------------------------------------------------------------ hydration

    def _hits(self, scope: ScopePredicate, at: datetime, rows, *, semantic_available: bool) -> list[Hit]:
        ids = defaultdict(list)
        for row in rows:
            ids[row.ref_type].append(row.id)
        details = {**_assertion_details(self.db, scope, at, ids["assertion"]),
                   **_raw_details(self.db, scope.owner_id, at, ids["segment"], ids["chunk"])}
        hits = []
        for row in rows:
            rank = {"rrf": float(row.rrf)}
            for name in ("lexical_rank", "lexical", "semantic_rank", "semantic"):
                if getattr(row, name) is not None:
                    rank[name] = float(getattr(row, name))
            if not semantic_available:
                rank["semantic_unavailable"] = 1.0
            if (detail := details.get((row.ref_type, row.id))) is not None:
                hits.append(Hit(ref_type=row.ref_type, id=row.id, score=float(row.rrf), rank=rank, **detail))
        return hits


# ---------------------------------------------------------------- legs


def _lexical(ref_type: str, id_column, tsv, tsq, cand: int):
    score = func.ts_rank_cd(tsv, tsq)
    return (select(literal(ref_type).label("ref_type"), id_column.label("id"), score.label("score"))
            .where(tsv.op("@@")(tsq)).order_by(score.desc(), id_column).limit(cand))


def _semantic(ref_type: str, docs, vector: list[float], cand: int):
    distance = docs.c.embedding.cosine_distance(vector)
    return (select(literal(ref_type).label("ref_type"), docs.c.id, (1 - distance).label("score"))
            .where(docs.c.embedding.is_not(None), docs.c.embedding_model == memory_learning.EMBEDDING_MODEL)
            .order_by(distance, docs.c.id).limit(cand))


def _ranked(name: str, parts):
    parts = [select(*p.subquery().c) for p in parts]  # parenthesize each ordered, limited part
    u = (union_all(*parts) if len(parts) > 1 else parts[0]).subquery(f"{name}_candidates")
    order = (u.c.score.desc(), u.c.ref_type, u.c.id)
    return select(literal(name).label("leg"), u.c.ref_type, u.c.id, u.c.score.label("score"),
                  func.row_number().over(order_by=order).label("rank"))


# ---------------------------------------------------------------- scoped candidate sets


def _live_source(source):
    return [source.deleted_at.is_(None), source.status != "deleted"]


def _supported_by_live_source(assertion_id):
    """No support evidence at all (stated by a human decision), or some support from a source not deleted."""
    e, s = ContextEvidence, ContextSource
    support = [e.assertion_id == assertion_id, e.role == "support"]
    return or_(~exists().where(*support),
               exists().where(*support, s.id == e.source_id, *_live_source(s)))


def _assertion_docs(scope: ScopePredicate, at: datetime):
    a = ContextAssertion
    if scope.known_at is None:
        return select(a.id, a.tsv.label("tsv"), a.embedding.label("embedding"),
                      a.embedding_model.label("embedding_model")).where(
            *temporal.scope_clauses(scope, at=at), _supported_by_live_source(a.id))
    v = ContextAssertionVersion
    same = v.statement == a.statement
    return (temporal.select_versions(scope, at=at)
            .with_only_columns(v.assertion_id.label("id"), func.to_tsvector(
                literal_column("'english'::regconfig"), v.statement).label("tsv"),
                case((same, a.embedding)).label("embedding"), case((same, a.embedding_model)).label("embedding_model"),
                maintain_column_froms=True)
            .where(_supported_by_live_source(a.id)).order_by(None))


def _revision_in_view(revision, source, scope: ScopePredicate):
    """The source's current revision, or with `known_at` the latest revision received by then."""
    if scope.known_at is None:
        return revision.revision == source.current_revision
    r = aliased(ContextSourceRevision)
    return revision.revision == select(func.max(r.revision)).where(
        r.source_id == source.id, r.received_at <= scope.known_at).scalar_subquery()


def _segment_matches_entities(segment, scope: ScopePredicate):
    """A segment is about an entity when it, or its source, is linked to the entity, its speaker is the actor, or
    it is evidence for an assertion about the entity."""
    if not scope.entities:
        return None
    link, r = ContextEntityLink, aliased(ContextSourceRevision)
    source_id = select(r.source_id).where(r.id == segment.revision_id).correlate(segment).scalar_subquery()
    targets = or_(*[and_(link.target_type == e.entity_type, link.target_id == e.entity_id) for e in scope.entities])
    linked = exists().where(link.owner_id == scope.owner_id, link.status != "rejected", targets, or_(
        and_(link.subject_type == "segment", link.subject_id == segment.id),
        and_(link.subject_type == "source", link.subject_id == source_id)))
    actors = [e.entity_id for e in scope.entities if e.entity_type == "actor"]
    e = ContextEvidence
    about = exists().where(e.owner_id == scope.owner_id, e.segment_id == segment.id,
                           e.assertion_id == ContextAssertion.id,
                           *temporal._subject_clauses(scope.model_copy(update={"kinds": []})))
    return or_(linked, segment.speaker_actor_id.in_(actors) if actors else false(), about)


def _stale_passage(segment_id, scope: ScopePredicate, at: datetime):
    """The segment supports an ended interpretation (retracted, superseded, or past its `valid_to`) that the
    scope's lifecycle and time would not show."""
    a, e = ContextAssertion, ContextEvidence
    reference = scope.known_at or at
    return exists().where(
        e.owner_id == scope.owner_id, e.segment_id == segment_id, a.id == e.assertion_id, a.lifecycle != "shadow",
        or_(a.lifecycle.in_(("superseded", "retracted")), a.valid_to <= (scope.valid_at or reference)),
        ~temporal.lifecycle_clause(a, valid_at=scope.valid_at, reference=reference,
                                   include_superseded=scope.include_superseded,
                                   include_retracted=scope.include_retracted))


def _raw_base(columns, segment_like, scope: ScopePredicate):
    r, s = ContextSourceRevision, ContextSource
    return (select(*columns).join(r, r.id == segment_like.revision_id).join(s, s.id == r.source_id)
            .where(segment_like.owner_id == scope.owner_id, r.owner_id == scope.owner_id,
                   s.owner_id == scope.owner_id, *_live_source(s), _revision_in_view(r, s, scope)))


def _segment_docs(scope: ScopePredicate, at: datetime):
    g = ContextSegment
    query = _raw_base([g.id, g.tsv.label("tsv")], g, scope).where(~_stale_passage(g.id, scope, at))
    entities = _segment_matches_entities(g, scope)
    return query if entities is None else query.where(entities)


def _chunk_docs(scope: ScopePredicate, at: datetime):
    c = ContextChunk
    query = _raw_base([c.id, c.embedding.label("embedding"), c.embedding_model.label("embedding_model")], c, scope)
    if not scope.entities:
        return query
    g = aliased(ContextSegment)
    return query.where(exists().where(g.revision_id == c.revision_id, g.seq.between(c.segment_start, c.segment_end),
                                      _segment_matches_entities(g, scope)))


# ---------------------------------------------------------------- details


def _entity(entity_type: str | None, entity_id: str | None) -> EntityRef | None:
    if entity_type in ENTITY_TYPES and entity_id and not entity_id.startswith("mention:"):
        return EntityRef(entity_type=entity_type, entity_id=entity_id)
    return None


def _sorted(refs) -> list[EntityRef]:
    unique = {(r.entity_type, r.entity_id): r for r in refs if r is not None}
    return [unique[k] for k in sorted(unique)]


def _ended(lifecycle: str, valid_to: datetime | None, at: datetime) -> bool:
    return lifecycle in ("superseded", "retracted") or (valid_to is not None and valid_to <= at)


def _successors(db, owner: str, ids) -> dict[str, str]:
    """Each predecessor's latest live successor (`supersedes_id` points back at the predecessor)."""
    a = ContextAssertion
    found = {}
    for predecessor, successor in db.execute(
            select(a.supersedes_id, a.id).where(a.owner_id == owner, a.supersedes_id.in_(list(ids)),
                                                a.lifecycle.not_in(("shadow", "retracted")))
            .order_by(a.supersedes_id, a.valid_from.desc().nulls_last(), a.created_at.desc(), a.id)):
        found.setdefault(predecessor, successor)
    return found


def _assertion_details(db, scope: ScopePredicate, at: datetime, ids: list[str]) -> dict:
    if not ids:
        return {}
    owner, a = scope.owner_id, ContextAssertion
    rows = {r.id: r for r in db.scalars(select(a).where(a.owner_id == owner, a.id.in_(ids)))}
    shown = {i: (r.statement, r.lifecycle, r.valid_from, r.valid_to, r.current_version) for i, r in rows.items()}
    if scope.known_at is not None:
        v = ContextAssertionVersion
        for ver in db.scalars(select(v).where(v.owner_id == owner, v.assertion_id.in_(ids),
                                              temporal.recorded(scope.known_at))):
            shown[ver.assertion_id] = (ver.statement, ver.lifecycle, ver.valid_from, ver.valid_to, ver.version)
    e, g, s = ContextEvidence, ContextSegment, ContextSource
    citations = {}
    for ev, seq in db.execute(
            select(e, g.seq).join(g, g.id == e.segment_id).join(s, s.id == e.source_id)
            .where(e.owner_id == owner, e.assertion_id.in_(ids), e.role == "support", e.quote != "",
                   *_live_source(s)).order_by(e.assertion_id, e.created_at, e.id)):
        citations.setdefault(ev.assertion_id, Citation(
            source_id=ev.source_id, source_revision_id=ev.revision_id, segment_start=seq, segment_end=seq,
            quote=ev.quote))
    link = ContextEntityLink
    linked = defaultdict(list)
    for assertion_id, target_type, target_id in db.execute(
            select(link.assertion_id, link.target_type, link.target_id).where(
                link.owner_id == owner, link.assertion_id.in_(ids), link.status.in_(RESOLVED))):
        linked[assertion_id].append(_entity(target_type, target_id))
    successors = _successors(db, owner, ids)
    reference = scope.valid_at or scope.known_at or at
    details = {}
    for i, row in rows.items():
        statement, lifecycle, valid_from, valid_to, version = shown[i]
        details[("assertion", i)] = {
            "revision": version, "snippet": statement[:SNIPPET], "citation": citations.get(i), "kind": row.kind,
            "lifecycle": lifecycle, "valid_from": valid_from, "valid_to": valid_to,
            "superseded_by": successors.get(i) if _ended(lifecycle, valid_to, reference) else None,
            "entities": _sorted([_entity(row.subject_type, row.subject_id), _entity(row.object_type, row.object_id),
                                 *linked[i]])}
    return details


def _raw_details(db, owner: str, at: datetime, segment_ids: list[str], chunk_ids: list[str]) -> dict:
    """Segments and chunks with citations, entities and the evidence-joined lifecycle annotation."""
    if not (segment_ids or chunk_ids):
        return {}
    g, c, r = ContextSegment, ContextChunk, ContextSourceRevision
    members = {("segment", i): [i] for i in segment_ids}
    for chunk_id, segment_id in db.execute(
            select(c.id, g.id).join(g, and_(g.revision_id == c.revision_id,
                                            g.seq.between(c.segment_start, c.segment_end)))
            .where(c.owner_id == owner, c.id.in_(chunk_ids))):
        members.setdefault(("chunk", chunk_id), []).append(segment_id)
    all_segments = {i for ids in members.values() for i in ids}
    segments = {row.id: row for row in db.scalars(select(g).where(g.owner_id == owner, g.id.in_(all_segments)))}
    e, a = ContextEvidence, ContextAssertion
    supported = defaultdict(list)  # segment id -> [(severity, assertion id, lifecycle, ended)]
    for segment_id, lifecycle, valid_to, assertion_id in db.execute(
            select(e.segment_id, a.lifecycle, a.valid_to, a.id).join(a, a.id == e.assertion_id)
            .where(e.owner_id == owner, e.segment_id.in_(all_segments), a.lifecycle != "shadow")):
        ended = _ended(lifecycle, valid_to, at)
        severity = max(SEVERITY.get(lifecycle, 0), SEVERITY["superseded"] if ended else 0)
        supported[segment_id].append((severity, assertion_id, lifecycle, ended))
    successors = _successors(db, owner, {p[1] for pairs in supported.values() for p in pairs if p[3]})
    link = ContextEntityLink
    linked = defaultdict(list)
    for subject_id, target_type, target_id in db.execute(
            select(link.subject_id, link.target_type, link.target_id).where(
                link.owner_id == owner, link.subject_type == "segment", link.subject_id.in_(all_segments),
                link.status.in_(RESOLVED))):
        linked[subject_id].append(_entity(target_type, target_id))

    def annotate(segment_ids):
        pairs = [p for i in segment_ids for p in supported[i]]
        if not pairs:
            return None, None
        _, assertion_id, lifecycle, ended = max(pairs)
        return lifecycle, successors.get(assertion_id) if ended else None

    def entities(segment_ids):
        return _sorted([ref for i in segment_ids for ref in linked[i]]
                       + [_entity("actor", segments[i].speaker_actor_id) for i in segment_ids if i in segments])

    revisions = {row.id: row for row in db.scalars(select(r).where(r.owner_id == owner, r.id.in_(
        {s.revision_id for s in segments.values()})))}
    chunks = {row.id: row for row in db.scalars(select(c).where(c.owner_id == owner, c.id.in_(chunk_ids)))}
    details = {}
    for (ref_type, ref_id), ids in members.items():
        ids = [i for i in ids if i in segments]
        if not ids:
            continue
        lifecycle, successor = annotate(ids)
        if ref_type == "segment":
            segment = segments[ref_id]
            revision = revisions[segment.revision_id]
            citation = Citation(source_id=revision.source_id, source_revision_id=revision.id,
                                segment_start=segment.seq, segment_end=segment.seq, quote=segment.text)
            snippet = segment.text
        else:
            chunk = chunks[ref_id]
            revision = revisions[chunk.revision_id]
            citation = Citation(source_id=revision.source_id, source_revision_id=revision.id,
                                segment_start=chunk.segment_start, segment_end=chunk.segment_end)
            snippet = chunk.text
        details[(ref_type, ref_id)] = {
            "revision": revision.revision, "snippet": snippet[:SNIPPET], "citation": citation, "lifecycle": lifecycle,
            "superseded_by": successor, "entities": entities(ids)}
    return details
