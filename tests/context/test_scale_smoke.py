"""Scale smoke for native retrieval: a few thousand rows per owner, exact vector scan, latency recorded.

Two owners each get 200 meetings (2,000 segments, 200 embedded chunks) and 2,500 embedded assertions, a tenth of
them superseded. Run with `-s` to print the latencies; the bound is deliberately loose (a regression alarm, not a
benchmark; C4 owns the corpus benchmark).
"""

import random
import statistics
import time

from jarvis import memory_learning
from jarvis.context.models import (
    ContextAssertion,
    ContextChunk,
    ContextEvidence,
    ContextSegment,
    ContextSource,
    ContextSourceRevision,
)
from jarvis.context.native.retrieve import NativeRetriever
from jarvis.context.schema import RetrieveQuery, ScopePredicate
from jarvis.db import session_scope
from jarvis.models import now, uid
from sqlalchemy import insert, text

DIM = 512
MEETINGS = 200
SEGMENTS = 10
ASSERTIONS = 2_500
QUERIES = 20
P95_BOUND_S = 1.5
WORDS = ("pilot", "budget", "clinic", "rollout", "vendor", "contract", "launch", "hiring", "roadmap", "pricing",
         "migration", "postgres", "security", "audit", "onboarding", "training", "survey", "dashboard", "invoice",
         "renewal", "schedule", "staffing", "forecast", "checklist")


def unit(rng):
    v = [rng.gauss(0, 1) for _ in range(DIM)]
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v]


def sentence(rng, n=8):
    return " ".join(rng.choice(WORDS) for _ in range(n)) + "."


def populate(db, owner, rng):
    model, at = memory_learning.EMBEDDING_MODEL, now()
    sources, revisions, segments, chunks = [], [], [], []
    for m in range(MEETINGS):
        source_id, revision_id = uid(), uid()
        sources.append({"id": source_id, "owner_id": owner, "connector": "api", "external_id": f"{owner}-{m}",
                        "current_revision": 1})
        revisions.append({"id": revision_id, "owner_id": owner, "source_id": source_id, "revision": 1,
                          "content_hash": "h" * 64, "occurred_at": at})
        lines = [sentence(rng) for _ in range(SEGMENTS)]
        segments += [{"id": uid(), "owner_id": owner, "revision_id": revision_id, "seq": s, "text": line,
                      "char_start": 0, "char_end": len(line)} for s, line in enumerate(lines)]
        chunks.append({"id": uid(), "owner_id": owner, "revision_id": revision_id, "seq": 0, "segment_start": 0,
                       "segment_end": SEGMENTS - 1, "text": "\n".join(lines), "embedding": unit(rng),
                       "embedding_model": model})
    assertions, evidence = [], []
    for i in range(ASSERTIONS):
        segment = segments[i % len(segments)]
        revision = revisions[(i % len(segments)) // SEGMENTS]
        assertion_id = uid()
        kind = rng.choice(("decision", "fact", "commitment"))
        assertions.append({"id": assertion_id, "owner_id": owner, "kind": kind,
                           "statement": segment["text"], "subject_type": "project", "subject_id": f"p{i % 20}",
                           "predicate": "fact", "lifecycle": "superseded" if i % 10 == 0 else "current",
                           "valid_from": at, "embedding": unit(rng), "embedding_model": model})
        evidence.append({"id": uid(), "owner_id": owner, "assertion_id": assertion_id, "assertion_version": 1,
                         "source_id": revision["source_id"], "revision_id": revision["id"],
                         "segment_id": segment["id"], "quote": segment["text"], "char_start": 0,
                         "char_end": len(segment["text"])})
    for model_class, rows in ((ContextSource, sources), (ContextSourceRevision, revisions),
                              (ContextSegment, segments), (ContextChunk, chunks), (ContextAssertion, assertions),
                              (ContextEvidence, evidence)):
        db.execute(insert(model_class), rows)


def timed(owner, rng, **query):
    latencies, results = [], []
    vector = unit(rng)
    with session_scope() as db:
        retriever = NativeRetriever(db, lambda o, t: vector)
        for _ in range(QUERIES):
            q = RetrieveQuery(text=" ".join(rng.sample(WORDS, 2)), scope=ScopePredicate(owner_id=owner), **query)
            started = time.perf_counter()
            results.append(retriever.retrieve(q))
            latencies.append(time.perf_counter() - started)
    latencies.sort()
    return {"p50_ms": round(statistics.median(latencies) * 1000, 1),
            "p95_ms": round(latencies[int(0.95 * (len(latencies) - 1))] * 1000, 1),
            "max_ms": round(latencies[-1] * 1000, 1)}, results


def test_retrieval_latency_over_a_few_thousand_rows_per_owner():
    rng = random.Random(1104)
    me, them = "c1-scale-" + uid()[:8], "c1-scale-" + uid()[:8]
    with session_scope() as db:
        populate(db, me, rng)
        populate(db, them, rng)
    with session_scope() as db:
        for table in ("context_assertions", "context_segments", "context_chunks", "context_evidence"):
            db.execute(text(f"ANALYZE {table}"))
    with session_scope() as db:
        mine = set(db.scalars(text("SELECT id FROM context_assertions WHERE owner_id = :o AND lifecycle = 'current' "
                                   "UNION ALL SELECT id FROM context_segments WHERE owner_id = :o "
                                   "UNION ALL SELECT id FROM context_chunks WHERE owner_id = :o"), {"o": me}))
    report = {}
    for name, query in (("assertions", {"limit": 20}), ("with_raw", {"limit": 20, "include_raw": True})):
        report[name], results = timed(me, rng, **query)
        assert all(results) and all(h.id in mine for hits in results for h in hits)  # no leaks, no superseded
    print(f"\ncontext retrieval scale smoke (2 owners x {ASSERTIONS} assertions, {MEETINGS * SEGMENTS} segments, "
          f"{MEETINGS} chunks): {report}")
    assert all(r["p95_ms"] < P95_BOUND_S * 1000 for r in report.values()), report
