"""The set-based raw entity scope (`retrieve._entity_segments`, package F2) ranks exactly as the per-segment
predicate it replaced. `Legacy` below is the replaced code verbatim (C1): a correlated OR of link, speaker and
evidence probes per candidate segment, and per segment inside each chunk. Both run on the small corpus fixture
for every raw scope the retriever accepts, and the hit lists (ids, order, scores, ranks and annotations) match."""

from dataclasses import replace

from jarvis.context import temporal
from jarvis.context.models import (
    ContextAssertion,
    ContextChunk,
    ContextEntityLink,
    ContextEvidence,
    ContextSegment,
    ContextSourceRevision,
)
from jarvis.context.native import retrieve
from jarvis.context.native.retrieve import NativeRetriever
from jarvis.context.schema import EntityRef, RetrieveQuery, ScopePredicate
from jarvis.db import session_scope
from sqlalchemy import and_, exists, false, or_, select, text
from sqlalchemy.orm import aliased

from evals.context.corpus import generate, load, queries

from .test_corpus_generation import SMALL

SPEC = replace(SMALL, seed=11)


class Legacy:
    """The replaced C1 functions."""

    @staticmethod
    def segment_matches_entities(segment, scope: ScopePredicate):
        if not scope.entities:
            return None
        link, r = ContextEntityLink, aliased(ContextSourceRevision)
        source_id = select(r.source_id).where(r.id == segment.revision_id).correlate(segment).scalar_subquery()
        targets = or_(*[and_(link.target_type == e.entity_type, link.target_id == e.entity_id)
                        for e in scope.entities])
        linked = exists().where(link.owner_id == scope.owner_id, link.status != "rejected", targets, or_(
            and_(link.subject_type == "segment", link.subject_id == segment.id),
            and_(link.subject_type == "source", link.subject_id == source_id)))
        actors = [e.entity_id for e in scope.entities if e.entity_type == "actor"]
        e = ContextEvidence
        about = exists().where(e.owner_id == scope.owner_id, e.segment_id == segment.id,
                               e.assertion_id == ContextAssertion.id,
                               *temporal._subject_clauses(scope.model_copy(update={"kinds": []})))
        return or_(linked, segment.speaker_actor_id.in_(actors) if actors else false(), about)

    @staticmethod
    def segment_docs(scope, at, _entity_segments):
        g = ContextSegment
        query = retrieve._raw_base([g.id, g.tsv.label("tsv")], g, scope).where(
            ~retrieve._stale_passage(g.id, scope, at))
        entities = Legacy.segment_matches_entities(g, scope)
        return query if entities is None else query.where(entities)

    @staticmethod
    def chunk_docs(scope, at, _entity_segments):
        c = ContextChunk
        query = retrieve._raw_base([c.id, c.embedding.label("embedding"),
                                    c.embedding_model.label("embedding_model")], c, scope)
        if not scope.entities:
            return query
        g = aliased(ContextSegment)
        return query.where(exists().where(g.revision_id == c.revision_id,
                                          g.seq.between(c.segment_start, c.segment_end),
                                          Legacy.segment_matches_entities(g, scope)))


def scopes(q):
    entity = [EntityRef(entity_type=q.entity[0], entity_id=q.entity[1])]
    yield ScopePredicate(owner_id=q.owner, entities=entity)
    yield ScopePredicate(owner_id=q.owner, entities=entity, include_superseded=True, include_retracted=True)
    yield ScopePredicate(owner_id=q.owner, entities=entity, known_at=q.known_at)
    yield ScopePredicate(owner_id=q.owner, entities=entity, kinds=["commitment", "decision"])
    yield ScopePredicate(owner_id=q.owner)


def ranking(hits):
    return [(h.ref_type, h.id, round(h.score, 12), {k: round(v, 9) for k, v in h.rank.items()}, h.lifecycle,
             h.superseded_by) for h in hits]


def test_set_based_entity_scope_ranks_like_the_per_segment_predicate(monkeypatch):
    corpus = generate(SPEC)
    with session_scope() as db:
        load(db, corpus)
    # Fresh statistics: the per-test cleanup only DELETEs, so without this the planner sees the dead rows and
    # stale estimates left by earlier tests and picks plans that took this test from seconds to many minutes.
    with session_scope() as db:
        db.execute(text("ANALYZE"))
    qs = queries(corpus, 24, seed=5)
    assert {q.entity[0] for q in qs} == {"project", "client", "actor"}
    cases = [(q, scope, vector) for q in qs for scope in scopes(q) for vector in (True, False)]

    def run():
        out, raw = [], 0
        with session_scope() as db:
            for q, scope, vector in cases:
                embed = (lambda _o, _t, v=q.vector: v.tolist()) if vector else (lambda _o, _t: None)
                hits = NativeRetriever(db, embed, at=SPEC.now).retrieve(
                    RetrieveQuery(text=q.text, scope=scope, include_raw=True, limit=30))
                raw += sum(h.ref_type != "assertion" for h in hits if scope.entities)
                out.append(ranking(hits))
        return out, raw

    new, raw = run()
    with monkeypatch.context() as patch:
        patch.setattr(retrieve, "_segment_docs", Legacy.segment_docs)
        patch.setattr(retrieve, "_chunk_docs", Legacy.chunk_docs)
        old, _ = run()
    assert raw > 50  # entity-scoped raw hits are actually exercised
    assert new == old
