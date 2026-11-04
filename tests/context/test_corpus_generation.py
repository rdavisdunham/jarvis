"""The plan-scale corpus generator (evals/context/corpus.py) at a small size: deterministic, internally
consistent deletion and lifecycle history, loadable, and searchable by the native retriever without leaks."""

import math
from collections import Counter, defaultdict
from dataclasses import replace
from itertools import pairwise

import pytest
from jarvis import memory_learning
from jarvis.context.models import (
    ContextAssertion,
    ContextAssertionVersion,
    ContextChunk,
    ContextEntityLink,
    ContextEvidence,
    ContextSegment,
    ContextSource,
    ContextSourceRevision,
)
from jarvis.context.native.retrieve import NativeRetriever
from jarvis.context.schema import EntityRef, RetrieveQuery, ScopePredicate
from jarvis.db import session_scope
from sqlalchemy import func, select

from evals.context.bench import Expect
from evals.context.corpus import CorpusSpec, generate, load, queries

SMALL = CorpusSpec(sources=40, segments=2_000, embedded=400, entities_per_owner=24, topics=6, seed=7,
                   deleted=0.15, revised=0.15)


@pytest.fixture(scope="module")
def corpus():
    return generate(SMALL)


def test_same_seed_same_rows_and_a_different_seed_differs(corpus):
    assert generate(SMALL).digest() == corpus.digest()
    assert generate(replace(SMALL, seed=8)).digest() != corpus.digest()


def test_sizes_and_owners_follow_the_spec(corpus):
    rows, counts = corpus.rows, corpus.counts()
    assert counts["context_sources"] == 40 and counts["context_segments"] == 2_000
    assert counts["context_chunks"] + counts["context_assertions"] == 400
    by_owner = Counter(r["owner_id"] for r in rows[ContextSource])
    assert [by_owner[o] for o in SMALL.owners] == [24, 10, 6]
    assert all(len({r["owner_id"] for r in rows[m]} - set(SMALL.owners)) == 0 for m in rows)
    targets = {r["target_id"] for r in rows[ContextEntityLink] if r["target_id"]}
    assert len(targets) > 20


def test_embeddings_are_unit_vectors_recorded_under_the_embedding_model(corpus):
    for model in (ContextAssertion, ContextChunk):
        for row in corpus.rows[model]:
            if row["embedding"] is None:
                continue
            assert len(row["embedding"]) == 512 and row["embedding_model"] == memory_learning.EMBEDDING_MODEL
            assert math.isclose(math.sqrt(sum(x * x for x in row["embedding"])), 1.0, rel_tol=1e-4)


def test_deletion_history_matches_what_the_purge_leaves(corpus):
    rows = corpus.rows
    deleted = {r["id"]: r for r in rows[ContextSource] if r["deleted_at"]}
    purged = {i for i, r in deleted.items() if r["title"] == ""}
    assert purged and set(deleted) - purged  # both purged and awaiting purge
    source_of = {r["id"]: r["source_id"] for r in rows[ContextSourceRevision]}
    for chunk in rows[ContextChunk]:
        if source_of[chunk["revision_id"]] in purged:
            assert chunk["embedding"] is None and chunk["text"] == ""
    for segment in rows[ContextSegment]:
        if source_of[segment["revision_id"]] in purged:
            assert segment["text"] == ""
    support = defaultdict(set)
    for e in rows[ContextEvidence]:
        if e["source_id"] in purged:
            assert e["quote"] == "" and e["segment_id"] is None
        if e["role"] == "support":
            support[e["assertion_id"]].add(e["source_id"])
    lifecycle = {r["id"]: r["lifecycle"] for r in rows[ContextAssertion]}
    only_purged = [a for a, sources in support.items() if sources <= purged]
    assert only_purged and all(lifecycle[a] == "retracted" for a in only_purged)
    survived = [a for a, sources in support.items() if sources & purged and sources - set(deleted)]
    assert any(lifecycle[a] != "retracted" for a in survived)  # independent support survives


def test_lifecycle_history_is_consistent(corpus):
    rows = corpus.rows[ContextAssertion]
    by_id = {r["id"]: r for r in rows}
    counts = Counter(r["lifecycle"] for r in rows)
    assert {"current", "superseded", "retracted", "contested", "shadow"} <= set(counts)
    versions = defaultdict(list)
    for v in corpus.rows[ContextAssertionVersion]:
        versions[v["assertion_id"]].append(v)
    position = {r["id"]: i for i, r in enumerate(rows)}
    for row in rows:
        history = sorted(versions[row["id"]], key=lambda v: v["version"])
        assert [v["version"] for v in history] == list(range(1, row["current_version"] + 1))
        assert history[-1]["recorded_to"] is None and history[-1]["lifecycle"] == row["lifecycle"]
        assert all(a["recorded_to"] == b["recorded_from"] for a, b in pairwise(history))
        if row["supersedes_id"]:
            old = by_id[row["supersedes_id"]]
            assert old["lifecycle"] == "superseded" and old["valid_to"] == row["valid_from"]
            assert position[old["id"]] < position[row["id"]]  # predecessors insert first
    assert sum(1 for r in rows if r["supersedes_id"]) == counts["superseded"]


def test_loads_and_retrieval_stays_inside_scope(corpus):
    with session_scope() as db:
        counts = load(db, corpus)
    with session_scope() as db:
        assert db.scalar(select(func.count()).select_from(ContextSegment)) == counts["context_segments"]
        assert db.scalar(select(func.count()).select_from(ContextAssertion)) == counts["context_assertions"]
    expect = Expect(corpus)
    qs = queries(corpus, 12, seed=3)
    assert {q.owner for q in qs} <= set(SMALL.owners)
    with session_scope() as db:
        for q in qs:
            retriever = NativeRetriever(db, lambda _o, _t, v=q.vector: v.tolist(), at=SMALL.now)
            for scope in (ScopePredicate(owner_id=q.owner),
                          ScopePredicate(owner_id=q.owner, entities=[EntityRef(entity_type=q.entity[0],
                                                                               entity_id=q.entity[1])])):
                hits = retriever.retrieve(RetrieveQuery(text=q.text, scope=scope, include_raw=True))
                assert expect.violations(q, [(h.ref_type, h.id) for h in hits], history=False) == 0
            assert retriever.retrieve(RetrieveQuery(text=q.text, scope=ScopePredicate(owner_id=q.owner)))
