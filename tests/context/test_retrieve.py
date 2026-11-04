"""Native hybrid retrieval: independent lexical and semantic legs fused by RRF, scope applied in both legs,
lifecycle defaults, raw-segment annotation and embedding outage. Vectors are inserted directly; no model calls."""

from datetime import timedelta

import pytest
from jarvis import cost_features, memory_learning
from jarvis.context import temporal
from jarvis.context.engine import Retriever
from jarvis.context.models import ContextChunk, ContextEntityLink, ContextSegment
from jarvis.context.native.retrieve import NativeRetriever
from jarvis.context.schema import EntityRef, RetrieveQuery, ScopePredicate
from jarvis.db import session_scope
from jarvis.models import now, uid

from .test_temporal import World, days

DIM = 512
MODEL = memory_learning.EMBEDDING_MODEL


def vec(*axes):
    v = [0.0] * DIM
    for axis in axes:
        v[axis] = 1.0
    return v


def embed(vector):
    def fake(owner, text):
        return vector
    return fake


def outage(owner, text):
    raise RuntimeError("embedding provider down")


def embedded(db, row, vector):
    row.embedding, row.embedding_model = vector, MODEL
    db.flush()
    return row


def search(db, owner, text, *, vector=None, embedder=None, include_raw=False, limit=20, **scope):
    retriever = NativeRetriever(db, embedder or embed(vector or vec(511)))
    return retriever.retrieve(RetrieveQuery(text=text, include_raw=include_raw, limit=limit,
                                            scope=ScopePredicate(owner_id=owner, **scope)))


def ids(hits, ref_type="assertion"):
    return [h.id for h in hits if h.ref_type == ref_type]


def owner():
    return "c1-" + uid()[:8]


def fact(w, statement, vector=None, *, subject=("project", "p1"), kind="decision", source=None):
    row = w.assertion(statement, days(-3), kind=kind, key=None, subject=subject, source=source)
    return embedded(w.db, row, vector) if vector else row


def test_native_retriever_implements_the_protocol():
    with session_scope() as db:
        assert isinstance(NativeRetriever(db), Retriever)


def test_lexical_and_semantic_legs_fuse_and_semantic_only_matches_are_returned():
    with session_scope() as db:
        me = owner()
        w = World(db, me)
        lexical = fact(w, "We launch the pilot on Postgres with pgvector.", vec(0))
        semantic = fact(w, "The database choice is settled.", vec(1, 2))
        both = fact(w, "Postgres is the database for the pilot.", vec(1))
        hits = search(db, me, "postgres pilot", vector=vec(1))
        by_id = {h.id: h for h in hits}
        assert set(ids(hits)) == {lexical.id, semantic.id, both.id}
        assert "lexical_rank" not in by_id[semantic.id].rank and by_id[semantic.id].rank["semantic_rank"] == 2
        assert {"lexical_rank", "semantic_rank", "rrf"} <= set(by_id[both.id].rank)
        assert hits[0].id == both.id  # strong in both legs
        assert all("semantic_unavailable" not in h.rank for h in hits)
        top = by_id[lexical.id]
        assert top.kind == "decision" and top.lifecycle == "current" and top.revision == 1
        assert top.snippet == lexical.statement
        assert top.citation.quote == lexical.statement and top.citation.segment_start == 0
        assert EntityRef(entity_type="project", entity_id="p1") in top.entities
        assert top.valid_from == lexical.valid_from
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)


def test_embedding_outage_leaves_the_lexical_leg_answering():
    with session_scope() as db:
        me = owner()
        w = World(db, me)
        lexical = fact(w, "We launch the pilot on Postgres.", vec(0))
        fact(w, "The database choice is settled.", vec(1))
        hits = search(db, me, "postgres pilot", embedder=outage)
        assert ids(hits) == [lexical.id]
        assert hits[0].rank["semantic_unavailable"] == 1.0 and "semantic_rank" not in hits[0].rank
        wrong_shape = search(db, me, "postgres pilot", embedder=embed([1.0, 0.0]))
        assert ids(wrong_shape) == [lexical.id] and wrong_shape[0].rank["semantic_unavailable"] == 1.0


def test_query_embedding_goes_through_the_shared_helper_under_the_context_feature(monkeypatch):
    seen = []

    def provider(owner, texts, timeout=20):
        seen.append((owner, texts, cost_features.current_feature.get()))
        return [vec(1)]

    monkeypatch.setattr(memory_learning, "embeddings", provider)
    with session_scope() as db:
        me = owner()
        w = World(db, me)
        semantic = fact(w, "The database choice is settled.", vec(1))
        hits = NativeRetriever(db).retrieve(RetrieveQuery(text="which datastore " + me,
                                                          scope=ScopePredicate(owner_id=me)))
        assert ids(hits) == [semantic.id]
    assert seen == [(me, ["which datastore " + me], "context_embeddings")]


def test_superseded_is_excluded_by_default_and_returned_as_history():
    with session_scope() as db:
        me = owner()
        w = World(db, me)
        old = embedded(db, w.assertion("Budget due 2026-11-13.", days(-20)), vec(3))
        w.reconcile([old])
        new = embedded(db, w.assertion("Budget due 2026-11-20.", days(-5)), vec(3))
        w.reconcile([new])
        assert (old.lifecycle, new.supersedes_id) == ("superseded", old.id)
        assert ids(search(db, me, "budget due", vector=vec(3))) == [new.id]
        assert ids(search(db, me, "nothing lexical", vector=vec(3))) == [new.id]  # semantic leg too
        history = {h.id: h for h in search(db, me, "budget due", vector=vec(3), include_superseded=True)}
        assert set(history) == {old.id, new.id}
        assert history[old.id].lifecycle == "superseded" and history[old.id].superseded_by == new.id
        assert history[new.id].superseded_by is None
        assert ids(search(db, me, "budget due", vector=vec(3), valid_at=days(-10))) == [old.id]


def test_shadow_and_retracted_assertions_are_never_returned_by_default():
    with session_scope() as db:
        me = owner()
        w = World(db, me)
        shadow = fact(w, "Pilot owner is Marco.", vec(4))
        retracted = fact(w, "Pilot owner is Priya.", vec(4))
        temporal.record_version(db, shadow, lifecycle="shadow")
        temporal.record_version(db, retracted, lifecycle="retracted")
        assert search(db, me, "pilot owner", vector=vec(4), include_superseded=True) == []
        assert ids(search(db, me, "pilot owner", vector=vec(4), include_retracted=True)) == [retracted.id]


def test_scope_holds_in_both_legs_and_never_leaks_across_owners():
    with session_scope() as db:
        me, them = owner(), owner()
        w = World(db, me)
        mine = fact(w, "Clinic rollout starts in March.", vec(5))
        theirs = fact(World(db, them), "Clinic rollout starts in March.", vec(5))
        other_project = fact(w, "Clinic rollout budget approved.", vec(5), subject=("project", "p2"))
        risk = fact(w, "Clinic rollout may slip.", vec(5), kind="risk")
        for text in ("clinic rollout", "zzz"):  # lexical+semantic, then semantic only
            assert theirs.id not in ids(search(db, me, text, vector=vec(5), include_raw=True))
            scoped = search(db, me, text, vector=vec(5), entities=[EntityRef(entity_type="project", entity_id="p1")])
            assert set(ids(scoped)) == {mine.id, risk.id}
            assert ids(search(db, me, text, vector=vec(5), kinds=["risk"])) == [risk.id]
        assert set(ids(search(db, me, "clinic rollout", vector=vec(5)))) == {mine.id, other_project.id, risk.id}
        assert ids(search(db, them, "clinic rollout", vector=vec(5))) == [theirs.id]


def test_entity_scope_follows_entity_links():
    with session_scope() as db:
        me = owner()
        w = World(db, me)
        linked = fact(w, "Priya owns the vendor contract.", vec(6), subject=("project", "p9"))
        fact(w, "Marco owns the vendor shortlist.", vec(6), subject=("project", "p9"))
        db.add(ContextEntityLink(owner_id=me, assertion_id=linked.id, subject_type="assertion", subject_id=linked.id,
                                 target_type="actor", target_id="actor-priya", role="mention", status="inferred"))
        db.flush()
        priya = [EntityRef(entity_type="actor", entity_id="actor-priya")]
        for text in ("vendor", "zzz"):
            hits = search(db, me, text, vector=vec(6), entities=priya)
            assert ids(hits) == [linked.id]
        assert EntityRef(entity_type="actor", entity_id="actor-priya") in hits[0].entities


def test_deleted_sources_are_excluded():
    with session_scope() as db:
        me = owner()
        w = World(db, me)
        kept = fact(w, "Vendor review happens weekly.", vec(7))
        source = w.source("Vendor review happens daily.")
        gone = fact(w, "Vendor review happens daily.", vec(7), source=source)
        source[0].deleted_at = now()  # before propagation runs: the assertion's only support is gone
        db.flush()
        hits = search(db, me, "vendor review", vector=vec(7), include_raw=True)
        assert ids(hits) == [kept.id]
        assert gone.id not in ids(hits) and len(ids(hits, "segment")) == 1


def test_raw_segments_are_annotated_and_stale_passages_stay_out_of_current_answers():
    with session_scope() as db:
        me = owner()
        w = World(db, me)
        old = w.assertion("I'll send the budget by 2026-11-13.", days(-20))
        w.reconcile([old])
        new = w.assertion("I'll send the budget by 2026-11-20.", days(-5))
        w.reconcile([new])
        unextracted, _ = w.source("The budget spreadsheet lives in the shared drive.")
        hits = search(db, me, "budget", include_raw=True)
        segments = {h.citation.quote: h for h in hits if h.ref_type == "segment"}
        assert set(segments) == {new.statement, "The budget spreadsheet lives in the shared drive."}
        assert segments[new.statement].lifecycle == "current" and segments[new.statement].superseded_by is None
        recovery = segments["The budget spreadsheet lives in the shared drive."]
        assert recovery.lifecycle is None and recovery.citation.source_id == unextracted.id
        assert recovery.revision == 1 and recovery.citation.segment_start == recovery.citation.segment_end == 0
        history = {h.citation.quote: h for h in search(db, me, "budget", include_raw=True, include_superseded=True)
                   if h.ref_type == "segment"}
        assert history[old.statement].lifecycle == "superseded" and history[old.statement].superseded_by == new.id
        assert ids(search(db, me, "budget")) == [new.id]  # raw only when asked for


def test_chunk_hits_cite_segment_ranges_and_carry_annotation():
    with session_scope() as db:
        me = owner()
        w = World(db, me)
        old = w.assertion("Launch is on 2026-12-01.", days(-20))
        w.reconcile([old])
        new = w.assertion("Launch is on 2026-12-08.", days(-5))
        w.reconcile([new])
        revision_id = db.get(ContextSegment, db.query(ContextSegment.id).filter(
            ContextSegment.owner_id == me, ContextSegment.text == old.statement).scalar()).revision_id
        db.add(ContextSegment(owner_id=me, revision_id=revision_id, seq=1, text="Marco agreed.", char_start=26,
                              char_end=39))
        chunk = ContextChunk(owner_id=me, revision_id=revision_id, seq=0, segment_start=0, segment_end=1,
                             text=old.statement + "\nMarco agreed.", embedding=vec(8), embedding_model=MODEL)
        stale_model = ContextChunk(owner_id=me, revision_id=revision_id, seq=1, segment_start=1, segment_end=1,
                                   text="Marco agreed.", embedding=vec(8), embedding_model="other-model")
        db.add_all([chunk, stale_model])
        db.flush()
        hits = search(db, me, "zzz", vector=vec(8), include_raw=True)
        assert ids(hits, "chunk") == [chunk.id]
        hit = hits[0]
        assert (hit.citation.segment_start, hit.citation.segment_end, hit.citation.quote) == (0, 1, None)
        assert hit.lifecycle == "superseded" and hit.superseded_by == new.id
        assert ids(search(db, me, "zzz", vector=vec(8)), "chunk") == []  # raw only when asked for
        assert ids(search(db, me, "zzz", embedder=outage, include_raw=True), "chunk") == []


def test_known_at_searches_the_version_recorded_then():
    with session_scope() as db:
        me = owner()
        w = World(db, me)
        row = fact(w, "Kickoff is in Lisbon.", vec(9))
        before = now()
        temporal.record_version(db, row, statement="Kickoff is in Porto.", at=before + timedelta(seconds=1))
        past = search(db, me, "lisbon", vector=vec(9), known_at=before)
        assert ids(past) == [row.id] and past[0].snippet == "Kickoff is in Lisbon." and past[0].revision == 1
        assert "semantic_rank" not in past[0].rank  # the head's vector embeds a different statement
        now_hits = search(db, me, "lisbon", vector=vec(9))  # only the semantic leg matches today's statement
        assert ids(now_hits) == [row.id] and "lexical_rank" not in now_hits[0].rank
        current = search(db, me, "porto", vector=vec(9))
        assert ids(current) == [row.id] and current[0].snippet == "Kickoff is in Porto." and current[0].revision == 2


@pytest.mark.parametrize("text", ["the and of", "!!!"])
def test_queries_without_lexical_terms_still_answer_semantically(text):
    with session_scope() as db:
        me = owner()
        row = fact(World(db, me), "Hiring freeze until Q2.", vec(10))
        assert ids(search(db, me, text, vector=vec(10))) == [row.id]
        assert search(db, me, text, embedder=outage) == []


def test_raw_entity_scope_uses_segment_links_speakers_sources_and_evidence():
    with session_scope() as db:
        me = owner()
        w = World(db, me)
        spoken, _ = w.source("Vendor call notes from Priya.")
        linked_source, _ = w.source("Vendor pricing table attached.")
        unrelated, unrelated_rev = w.source("Vendor lunch menu.")
        about = fact(w, "Vendor contract renews in May.", subject=("actor", "actor-priya"))
        segment = db.query(ContextSegment).filter(ContextSegment.owner_id == me,
                                                  ContextSegment.text == "Vendor call notes from Priya.").one()
        segment.speaker_actor_id = "actor-priya"
        db.add(ContextEntityLink(owner_id=me, subject_type="source", subject_id=linked_source.id,
                                 target_type="actor", target_id="actor-priya", role="participant", status="exact"))
        db.add(ContextChunk(owner_id=me, revision_id=unrelated_rev.id, seq=0, segment_start=0, segment_end=0,
                            text="Vendor lunch menu.", embedding=vec(12), embedding_model=MODEL))
        db.flush()
        hits = search(db, me, "vendor", vector=vec(12), include_raw=True,
                      entities=[EntityRef(entity_type="actor", entity_id="actor-priya")])
        quotes = {h.citation.quote for h in hits if h.ref_type == "segment"}
        assert quotes == {"Vendor call notes from Priya.", "Vendor pricing table attached.", about.statement}
        assert ids(hits) == [about.id] and ids(hits, "chunk") == []
        assert unrelated.id not in {h.citation.source_id for h in hits}
        spoken_hit = next(h for h in hits if h.citation.source_id == spoken.id)
        assert spoken_hit.entities == [EntityRef(entity_type="actor", entity_id="actor-priya")]
