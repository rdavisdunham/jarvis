"""Latency benchmark for native context retrieval at the plan's corpus size (plan_final.md section 11).

    uv run --no-sync python -m evals.context.bench                       # full corpus, 200 queries, report
    uv run --no-sync python -m evals.context.bench --scale 0.1 --queries 40 --report -
    uv run --no-sync python -m evals.context.bench --pipeline 60          # one meeting: stage times, model cost

Creates a fresh database on the server named by JARVIS_DATABASE_URL (default `jarvis_bench`; only names starting
with `jarvis_bench` are accepted because the database is dropped and recreated), creates the model tables, loads
`evals.context.corpus` and times `NativeRetriever.retrieve` (one SQL statement plus hydration: what
`context_search` runs, excluding synthesis) for every query under each scope configuration. Query embeddings are
synthetic, so there are no API calls. Then it builds HNSW indexes on the embedded tables, enables
`hnsw.iterative_scan`, reruns the same queries, checks whether the planner used the indexes and measures top-k
agreement with the exact run. Every run also checks that no hit crosses owners, comes from a deleted source or
(outside `known_at`) is an ended, shadow or future assertion. The database is dropped at the end unless `--keep`.

`--pipeline MINUTES` instead loads the 1x corpus, runs one synthetic meeting (`corpus.meeting`) through
`pipeline.perform` with the deterministic engines and prints stage timings, then the tokens and cost the native
extractor would spend on it, computed with a scripted model (no calls). Reports: reports/pilot-perf.md.
"""

import argparse
import json
import os
import platform
import re
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from jarvis.config import get_settings
from jarvis.context.models import (
    ContextAssertion,
    ContextChunk,
    ContextEntityLink,
    ContextEvidence,
    ContextSegment,
    ContextSource,
    ContextSourceRevision,
)
from jarvis.context.native.retrieve import NativeRetriever
from jarvis.context.schema import EntityRef, RetrieveQuery, ScopePredicate
from jarvis.models import Base
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from evals.context.corpus import BenchQuery, Corpus, CorpusSpec, generate, load, queries

GATE_MS = 2_000
LIMIT = 20
EMBEDDING_PRICE_PER_MTOK = 0.02  # text-embedding-3-small list price (USD per million tokens), an assumption
HNSW = {"context_assertions": "ix_bench_context_assertions_hnsw", "context_chunks": "ix_bench_context_chunks_hnsw"}
REPORT = Path(__file__).parent / "reports" / "native-retrieval-benchmark.md"


def _scope(q: BenchQuery, **extra) -> ScopePredicate:
    return ScopePredicate(owner_id=q.owner, **extra)


def _entity(q: BenchQuery) -> list[EntityRef]:
    return [EntityRef(entity_type=q.entity[0], entity_id=q.entity[1])]


# name -> (description, query builder)
CONFIGS: dict[str, tuple[str, Callable[[BenchQuery], RetrieveQuery]]] = {
    "owner": ("assertions, owner scope only",
              lambda q: RetrieveQuery(text=q.text, scope=_scope(q), limit=LIMIT)),
    "owner+raw": ("assertions + segments + chunks, owner scope only",
                  lambda q: RetrieveQuery(text=q.text, scope=_scope(q), limit=LIMIT, include_raw=True)),
    "entity": ("assertions, entity scope",
               lambda q: RetrieveQuery(text=q.text, scope=_scope(q, entities=_entity(q)), limit=LIMIT)),
    "entity+raw": ("assertions + segments + chunks, entity scope",
                   lambda q: RetrieveQuery(text=q.text, scope=_scope(q, entities=_entity(q)), limit=LIMIT,
                                           include_raw=True)),
    "known_at": ("assertions as recorded at a past instant, owner scope",
                 lambda q: RetrieveQuery(text=q.text, scope=_scope(q, known_at=q.known_at), limit=LIMIT)),
}


# ---------------------------------------------------------------- database


def _bench_url(name: str) -> tuple[str, str]:
    if not re.fullmatch(r"jarvis_bench[a-z0-9_]*", name):
        raise SystemExit(f"refusing database name {name!r}: it must start with jarvis_bench")
    server = os.environ.get("JARVIS_DATABASE_URL") or get_settings().database_url
    url = make_url(server)
    return url.render_as_string(hide_password=False), url.set(database=name).render_as_string(hide_password=False)


def _engine(url: str, *, iterative: bool = False):
    engine = create_engine(url, connect_args={"options": "-c timezone=UTC"})
    if iterative:
        @event.listens_for(engine, "connect")
        def _hnsw(connection, _record):
            with connection.cursor() as cursor:
                cursor.execute("SET hnsw.iterative_scan = relaxed_order")
            connection.commit()
    return engine


def create_database(server: str, name: str) -> None:
    admin = create_engine(server, isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        connection.exec_driver_sql(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        connection.exec_driver_sql(f'CREATE DATABASE "{name}"')
    admin.dispose()


def drop_database(server: str, name: str) -> None:
    admin = create_engine(server, isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        connection.exec_driver_sql(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    admin.dispose()


def _vacuum(engine) -> None:
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
        connection.exec_driver_sql("VACUUM (ANALYZE)")


# ---------------------------------------------------------------- expectations


class Expect:
    """What each hit may be, from the generated rows (independent of the retriever's SQL)."""

    def __init__(self, corpus: Corpus):
        rows = corpus.rows
        self.owner = {r["id"]: r["owner_id"] for m in (ContextAssertion, ContextSegment, ContextChunk)
                      for r in rows[m]}
        deleted = {r["id"] for r in rows[ContextSource] if r["deleted_at"]}
        source_of = {r["id"]: r["source_id"] for r in rows[ContextSourceRevision]}
        self.dead_raw = {r["id"] for m in (ContextSegment, ContextChunk) for r in rows[m]
                         if source_of[r["revision_id"]] in deleted}
        support = defaultdict(set)
        for e in rows[ContextEvidence]:
            if e["role"] == "support":
                support[e["assertion_id"]].add(e["source_id"])
        now = corpus.spec.now
        self.hidden = {r["id"] for r in rows[ContextAssertion]
                       if r["lifecycle"] in ("superseded", "retracted", "shadow") or r["valid_from"] > now
                       or (support[r["id"]] and support[r["id"]] <= deleted)}

    def violations(self, q: BenchQuery, hits, *, history: bool) -> int:
        bad = 0
        for ref_type, ident in hits:
            bad += self.owner.get(ident) != q.owner or ident in self.dead_raw
            bad += not history and ref_type == "assertion" and ident in self.hidden
        return bad


# ---------------------------------------------------------------- timing


def _percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(p * (len(ordered) - 1))))]


def run_config(engine, corpus: Corpus, qs: list[BenchQuery], build, expect: Expect, *, history: bool,
               warmup: int = 10) -> dict:
    latencies, results, violations, empty = [], [], 0, 0
    with Session(engine) as db:
        for q in qs[:warmup]:
            NativeRetriever(db, lambda _o, _t, v=q.vector: v.tolist(), at=corpus.spec.now).retrieve(build(q))
        for q in qs:
            retriever = NativeRetriever(db, lambda _o, _t, v=q.vector: v.tolist(), at=corpus.spec.now)
            started = time.perf_counter()
            hits = retriever.retrieve(build(q))
            latencies.append((time.perf_counter() - started) * 1000)
            ids = [(h.ref_type, h.id) for h in hits]
            results.append(ids)
            violations += expect.violations(q, ids, history=history)
            empty += not ids
        db.rollback()
    return {"p50": statistics.median(latencies), "p95": _percentile(latencies, 0.95), "max": max(latencies),
            "mean_hits": statistics.mean(len(r) for r in results), "empty": empty, "violations": violations,
            "results": results}


def perf_summary(scale: float, corpus: Corpus, exact: dict) -> dict:
    """The exact-scan timings (the production plan) in the shape `evals.context.report --perf` grades."""
    return {"scale": scale, "corpus": corpus.counts(),
            "scopes": {name: {"p50_ms": round(r["p50"], 1), "p95_ms": round(r["p95"], 1), "queries": len(r["results"])}
                       for name, r in exact.items()},
            "leaks": sum(r["violations"] for r in exact.values())}


def plan_uses(engine, build, q: BenchQuery, corpus: Corpus) -> dict[str, bool]:
    """EXPLAIN the retrieval statement for one query and report which HNSW indexes the planner chose."""
    captured = []

    def capture(_conn, _cursor, statement, parameters, _context, _many):
        if "websearch_to_tsquery" in statement and not captured:
            captured.append((statement, parameters))

    event.listen(engine, "before_cursor_execute", capture)
    try:
        with Session(engine) as db:
            NativeRetriever(db, lambda _o, _t: q.vector.tolist(), at=corpus.spec.now).retrieve(build(q))
            statement, parameters = captured[0]
            plan = "\n".join(r[0] for r in db.connection().exec_driver_sql("EXPLAIN " + statement, parameters))
            db.rollback()
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    return {table: index in plan for table, index in HNSW.items()}


PROBE = ("SELECT id FROM context_assertions WHERE owner_id = :owner AND embedding_model = :model "
         "AND lifecycle IN ('current', 'contested') AND embedding IS NOT NULL "
         "ORDER BY embedding <=> CAST(:v AS vector){tie} LIMIT {k}")


def semantic_probe(engine, qs: list[BenchQuery], k: int = 100) -> dict:
    """The bare semantic leg (owner and lifecycle filter, top `k`) two ways: as the planner chooses with the
    retriever's `distance, id` ordering, and ordered by distance alone with the HNSW index forced. Measures what
    an index-friendly leg would gain and its recall against the exact top `k`."""
    from jarvis import memory_learning

    out = {}
    for name, tie, forced in (("exact (distance, id)", ", id", False), ("HNSW (distance only, forced)", "", True)):
        statement = text(PROBE.format(tie=tie, k=k))
        latencies, results = [], []
        with Session(engine) as db:
            if forced:
                db.execute(text("SET LOCAL enable_seqscan = off"))
                db.execute(text("SET LOCAL enable_bitmapscan = off"))
                db.execute(text("SET LOCAL enable_sort = off"))
            for q in qs:
                params = {"owner": q.owner, "model": memory_learning.EMBEDDING_MODEL,
                          "v": "[" + ",".join(f"{x:.6f}" for x in q.vector) + "]"}
                started = time.perf_counter()
                results.append(list(db.scalars(statement, params)))
                latencies.append((time.perf_counter() - started) * 1000)
            plan = "\n".join(db.execute(text("EXPLAIN " + PROBE.format(tie=tie, k=k)), params).scalars())
            db.rollback()
        out[name] = {"p50": statistics.median(latencies), "p95": _percentile(latencies, 0.95),
                     "index": HNSW["context_assertions"] in plan, "results": results}
    exact, approx = (r["results"] for r in out.values())
    out["recall"] = statistics.mean(len(set(a) & set(e)) / len(e) for e, a in zip(exact, approx, strict=True) if e)
    return out


def agreement(exact: list[list], approx: list[list]) -> float:
    """Mean share of the exact run's top-k (ref_type, id) pairs that the indexed run also returned."""
    shares = [len(set(a) & set(e)) / len(e) for e, a in zip(exact, approx, strict=True) if e]
    return statistics.mean(shares) if shares else 1.0


# ---------------------------------------------------------------- report


def environment(engine) -> dict[str, str]:
    with engine.connect() as connection:
        info = {
            "postgres": connection.scalar(text("SHOW server_version")),
            "pgvector": connection.scalar(text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")),
            "shared_buffers": connection.scalar(text("SHOW shared_buffers")),
            "work_mem": connection.scalar(text("SHOW work_mem")),
            "max_parallel_workers_per_gather": connection.scalar(text("SHOW max_parallel_workers_per_gather")),
            "jit (server default; the retriever turns it off for its statement)": connection.scalar(text("SHOW jit")),
            "database size": connection.scalar(text("SELECT pg_size_pretty(pg_database_size(current_database()))")),
        }
    cpu = ""
    try:
        cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                   if line.startswith("model name"))
    except (OSError, StopIteration):
        cpu = platform.processor()
    info["host"] = f"{cpu or 'unknown CPU'}, {os.cpu_count()} logical CPUs, {platform.system()} {platform.release()}"
    return info


def _ms(value: float) -> str:
    return f"{value:,.0f}" if value >= 100 else f"{value:.1f}"


def recommendation(exact: dict, indexed: dict | None, agree: dict[str, float] | None,
                   used: dict[str, dict[str, bool]] | None, probe: dict | None = None) -> str:
    worst = max(r["p95"] for r in exact.values())
    if worst < GATE_MS:
        line = (f"**HNSW recommendation: not needed.** Exact cosine scan meets the gate at corpus size (worst p95 "
                f"{_ms(worst)} ms < {GATE_MS:,} ms); keep it and rerun this benchmark before adding an index.")
        if indexed and used and not any(any(u.values()) for u in used.values()):
            line += (" With HNSW indexes present the planner still chose the exact plan for every configuration: the "
                     "semantic leg orders by `distance, id`, which an HNSW index cannot serve, and the owner and "
                     "lifecycle filters leave few enough rows that a sort is cheap.")
            if probe:
                exact_leg, hnsw_leg = probe["exact (distance, id)"], probe["HNSW (distance only, forced)"]
                line += (f" An index-friendly leg (distance only, index forced) would take the bare semantic leg from "
                         f"p95 {_ms(exact_leg['p95'])} ms to {_ms(hnsw_leg['p95'])} ms at recall@100 "
                         f"{probe['recall']:.1%}; revisit when the exact leg approaches the budget, and then drop the "
                         f"`id` tie-break from the ORDER BY (keep it only as a post-sort).")
        elif indexed and agree:
            best = max(indexed[n]["p95"] for n in indexed)
            line += (f" With HNSW the worst p95 was {_ms(best)} ms at a mean top-{LIMIT} agreement of "
                     f"{min(agree.values()):.1%}.")
        return line
    if indexed and agree and max(r["p95"] for r in indexed.values()) < GATE_MS and min(agree.values()) >= 0.95:
        return (f"**HNSW recommendation: add HNSW with `hnsw.iterative_scan`.** Exact scan misses the gate (worst p95 "
                f"{_ms(worst)} ms) and the indexed run meets it with at least 95% top-{LIMIT} agreement.")
    return (f"**HNSW recommendation: investigate.** Exact scan misses the gate (worst p95 {_ms(worst)} ms) and HNSW "
            f"does not clearly fix it; profile the statement before choosing an index.")


def render(corpus: Corpus, env: dict, timings: dict, exact: dict, indexed: dict | None, agree, used,
           qs: list[BenchQuery], probe: dict | None = None) -> str:
    counts = corpus.counts()
    rows = corpus.rows
    lifecycles = defaultdict(int)
    for r in rows[ContextAssertion]:
        lifecycles[r["lifecycle"]] += 1
    deleted = [r for r in rows[ContextSource] if r["deleted_at"]]
    purged = sum(1 for r in deleted if r["title"] == "")
    embedded_chunks = sum(1 for r in rows[ContextChunk] if r["embedding"] is not None)
    per_owner = defaultdict(lambda: defaultdict(int))
    for model, label in ((ContextSource, "sources"), (ContextSegment, "segments"), (ContextChunk, "chunks"),
                         (ContextAssertion, "assertions")):
        for r in rows[model]:
            per_owner[r["owner_id"]][label] += 1
    targets = {(r["target_type"], r["target_id"]) for r in rows[ContextEntityLink] if r["target_id"]}
    spec = corpus.spec
    scale = spec.sources / CorpusSpec().sources
    lines = [
        "# Native retrieval benchmark (corpus fixture" + ("" if scale == 1 else f", {scale:g}x the plan size") + ")",
        "",
        (f"Generated by `evals/context/bench.py` on {datetime.now(UTC):%Y-%m-%d %H:%M} UTC with corpus seed "
        f"{spec.seed}, {len(qs)} queries per configuration (after 10 warm-up queries), limit {LIMIT}. Latency is "
        "wall time of `NativeRetriever.retrieve` (the ranked SQL statement plus hydration), the database part of "
        "`context_search`; synthesis is excluded and query embeddings are synthetic (no API calls)."),
        "",
        "## Environment",
        "",
        *[f"- {k}: {v}" for k, v in env.items()],
        "- client: Python on the same host, TCP to the database container (no network hop off the machine)",
        "",
        "## Corpus",
        "",
        (f"{counts['context_sources']} sources ({len(deleted)} deleted: {purged} purged, {len(deleted) - purged} "
        f"awaiting purge; {counts['context_source_revisions'] - counts['context_sources']} with a second revision), "
        f"{counts['context_segments']:,} segments, {counts['context_chunks']:,} chunks ({embedded_chunks:,} embedded "
        f"after purges), {counts['context_assertions']:,} assertions ({counts['context_assertion_versions']:,} "
        f"versions, {counts['context_evidence']:,} evidence rows), {counts['context_entity_links']:,} entity links "
        f"to {len(targets)} distinct entity ids, {counts['context_decisions']} correction decisions. Embeddings are "
        f"512-dimension unit vectors clustered by {spec.topics} topics and meeting series."),
        "",
        "| Assertion lifecycle | Rows |",
        "|---|---:|",
        *[f"| {k} | {v:,} |" for k, v in sorted(lifecycles.items(), key=lambda kv: -kv[1])],
        "",
        "| Owner | Sources | Segments | Chunks | Assertions |",
        "|---|---:|---:|---:|---:|",
        *[f"| {o} | {c['sources']} | {c['segments']:,} | {c['chunks']:,} | {c['assertions']:,} |"
          for o, c in per_owner.items()],
        "",
        (f"Generation {timings['generate']:.1f} s, load {timings['load']:.1f} s, VACUUM ANALYZE "
        f"{timings['vacuum']:.1f} s."),
        "",
        "## Latency (ms)",
        "",
        "| Configuration | Plan | p50 | p95 | max | mean hits | empty | violations |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, (description, _) in CONFIGS.items():
        for label, table in (("exact", exact), ("HNSW", indexed)):
            if table and name in table:
                r = table[name]
                lines.append(f"| {name}: {description} | {label} | {_ms(r['p50'])} | {_ms(r['p95'])} | "
                             f"{_ms(r['max'])} | {r['mean_hits']:.1f} | {r['empty']} | {r['violations']} |")
    lines += ["", ("Violations count hits from another owner or a deleted source, and (outside `known_at`) "
              "superseded, retracted, shadow or future-effective assertions; all must be zero.")]
    if indexed:
        lines += ["", "## HNSW run", "",
                  (f"Indexes `USING hnsw (embedding vector_cosine_ops)` with default `m`/`ef_construction` on "
                  f"`context_assertions` and `context_chunks`, built serially in {timings['hnsw']:.1f} s; sessions set "
                  "`hnsw.iterative_scan = relaxed_order` (default `hnsw.ef_search`)."), "",
                  "| Configuration | HNSW index chosen by the planner | top-k agreement with exact |",
                  "|---|---|---:|"]
        for name in CONFIGS:
            chosen = ", ".join(t for t, u in used[name].items() if u) or "none (exact scan)"
            lines.append(f"| {name} | {chosen} | {agree[name]:.1%} |")
    if probe:
        slowest = max(exact, key=lambda n: exact[n]["p95"])
        semantic = probe["exact (distance, id)"]["p95"]
        lines += ["", (f"Slowest configuration: `{slowest}` (exact p95 {_ms(exact[slowest]['p95'])} ms). The bare "
                  f"semantic leg below costs {_ms(semantic)} ms p95, so the rest of that time is lexical candidate "
                  "selection, the scope predicates (for raw hits, the entity-segment set and the stale-passage check) "
                  "and hydration, none of which a vector index changes.")]
        lines += ["", "### Bare semantic leg probe", "",
                  ("Assertions only, owner plus live-lifecycle filter, top 100 by cosine distance, same query "
                   "vectors. The first row is the retriever's ordering as the planner runs it; the second orders by "
                   "distance alone with sequential scans, bitmap scans and sorts disabled so the HNSW index is used."), "",
                  "| Leg | HNSW index in plan | p50 | p95 |", "|---|---|---:|---:|"]
        for name in ("exact (distance, id)", "HNSW (distance only, forced)"):
            r = probe[name]
            lines.append(f"| {name} | {'yes' if r['index'] else 'no'} | {_ms(r['p50'])} | {_ms(r['p95'])} |")
        lines += ["", f"HNSW recall@100 against the exact leg: {probe['recall']:.1%}."]
    tokens = statistics.mean(max(1, round(len(q.text.split()) * 1.3)) for q in qs)
    lines += [
        "",
        "## Cost per query",
        "",
        (f"Database time only here. In production each `context_search` adds one query embedding "
        f"(`text-embedding-3-small`, about {tokens:.0f} tokens for these queries): roughly "
        f"${tokens * EMBEDDING_PRICE_PER_MTOK / 1e6:.7f} at an assumed ${EMBEDDING_PRICE_PER_MTOK} per million "
        "tokens, recorded under cost feature `context_embeddings`. No other model call is made before synthesis."),
        "",
        "## Recommendation",
        "",
        recommendation(exact, indexed, agree, used, probe),
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- one meeting: stage timings and model cost

PIPELINE_OWNER = "bench-owner-a"
REVIEW_STAGES = ("context_ingest", "context_extract", "context_link", "context_reconcile")
REASONING_TOKENS = 1000  # assumed hidden reasoning per call at low effort, as the extraction fixtures assume
SCRIPTED_LINE = re.compile(r"^\[(\d+)\] ([^:]+?)(?: \(continued\))?: (.*)$")
PLANTED = re.compile(r"Decision: ([^.]+)\.|I'll ([^.]+?) by (\d{4}-\d{2}-\d{2})\.")


class ScriptedModel:
    """Stands in for the extraction model behind `native.extract.call_model`: answers each pass as a well-behaved
    model would for `corpus.meeting`'s planted decisions and commitments, and reports estimated usage (prompt:
    system, user content and response schema; completion: the answer plus REASONING_TOKENS). No network."""

    def __init__(self):
        self.passes: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # name -> [calls, prompt, completion]

    def __call__(self, owner, body, allowance, timeout, prompt_tokens) -> dict:
        from jarvis.model_map import estimate_tokens

        name = body["response_format"]["json_schema"]["name"]
        content = json.dumps(getattr(self, name.removeprefix("context_"))(json.loads(body["messages"][1]["content"])))
        usage = {"prompt_tokens": sum(estimate_tokens(m["content"]) for m in body["messages"])
                 + estimate_tokens(body["response_format"]),
                 "completion_tokens": estimate_tokens(content) + REASONING_TOKENS}
        tally = self.passes[name]
        tally[0], tally[1], tally[2] = tally[0] + 1, tally[1] + usage["prompt_tokens"], tally[2] + usage[
            "completion_tokens"]
        return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}], "usage": usage,
                "service_tier": "default"}

    @staticmethod
    def detect(user: dict) -> dict:
        found = []
        for line in user["lines"].splitlines():
            if m := SCRIPTED_LINE.match(line):
                found += [{"kind": "decision" if p[1] else "commitment", "summary": p[0], "segment_seqs": [int(m[1])],
                           "quote": p[0]} for p in PLANTED.finditer(m[3])]
        return {"candidates": found}

    @staticmethod
    def enrich(user: dict) -> dict:
        speakers = {int(m[1]): m[2] for line in user["lines"].splitlines() if (m := SCRIPTED_LINE.match(line))}
        items = []
        for i, c in enumerate(user["candidates"]):
            p, seq = PLANTED.search(c["quote"]), c["segment_seqs"][0]
            item = {"candidate": i, "kind": c["kind"], "subject": None, "object": None, "key": None,
                    "modality": "asserted", "inferred": False, "responsible": [], "responsibility": "none",
                    "recipient": None, "due": None, "valid_from": None, "valid_to": None, "confidence": 0.9,
                    "evidence": [{"segment_seq": seq, "quote": c["quote"]}], "statement": f"The team decided {p[1]}."}
            if p[2]:
                who = speakers.get(seq, "Unknown speaker")
                item.update(statement=f"{who} will {p[2]} by {p[3]}.", key="due_date", responsibility="speaker",
                            responsible=[{"entity_type": "actor", "mention": who, "candidate_id": None}],
                            due={"expression": f"by {p[3]}", "value": p[3], "precision": "day", "derivation": None})
            items.append(item)
        return {"items": items}

    @staticmethod
    def verify(user: dict) -> dict:
        return {"verdicts": [{"item": it["item"], "entailed": True, "modality": it["modality"], "attribution_ok": True,
                              "reason": ""} for it in user["items"]]}


def _use_app_database(url: str) -> None:
    """Point `jarvis.db.session_scope` (the pipeline's sessions) at the benchmark database."""
    from jarvis import db

    os.environ["JARVIS_DATABASE_URL"] = url
    get_settings.cache_clear()
    db.engine.cache_clear()


def pipeline_run(engine, url: str, envelope) -> dict:
    """`sources.receive` then every queued context job through `pipeline.perform`, oldest first, as the worker would,
    with the deterministic engines (`engine: double`) and stubbed embeddings, on top of a loaded corpus owner.
    Returns wall time per job kind, the extraction request the pipeline built and the texts sent for embedding."""
    from jarvis import memory_learning
    from jarvis.context import pipeline, sources
    from jarvis.context.models import ContextOwnerState
    from jarvis.models import Job
    from sqlalchemy import select, update

    _use_app_database(url)
    embedded, requests = [], []

    def vectors(owner, texts, timeout=20):
        embedded.extend(texts)
        return [[1.0] + [0.0] * 511 for _ in texts]

    original_embeddings, original_extractor = memory_learning.embeddings, pipeline.run_extractor

    def recording(engine_name, request, *args, **kwargs):
        requests.append(request)
        return original_extractor(engine_name, request, *args, **kwargs)

    memory_learning.embeddings, pipeline.run_extractor = vectors, recording
    timings: dict[str, list[float]] = defaultdict(list)
    try:
        with Session(engine) as db, db.begin():
            db.execute(update(ContextOwnerState).where(ContextOwnerState.owner_id == PIPELINE_OWNER)
                       .values(settings={"engine": "double"}))
        from jarvis.db import session_scope
        started = time.perf_counter()
        with session_scope() as db:
            sources.receive(db, PIPELINE_OWNER, envelope)
        timings["sources.receive"].append(time.perf_counter() - started)
        for _ in range(500):
            with session_scope() as db:
                job = db.scalar(select(Job).where(Job.owner_id == PIPELINE_OWNER, Job.kind.like("context\\_%"),
                                                  Job.status.in_(("queued", "retrying")))
                                .order_by(Job.created_at, Job.id).limit(1))
                if job is None:
                    break
                job_id, kind = job.id, job.kind
            started = time.perf_counter()
            pipeline.perform(job_id)
            timings[kind].append(time.perf_counter() - started)
        with session_scope() as db:
            failed = list(db.scalars(select(Job.kind).where(Job.owner_id == PIPELINE_OWNER,
                                                            Job.status != "succeeded")))
    finally:
        memory_learning.embeddings, pipeline.run_extractor = original_embeddings, original_extractor
    return {"timings": dict(timings), "request": requests[0], "embedded": embedded, "failed": failed}


def model_cost(request, embedded: list[str]) -> dict:
    """What the native extractor would send for this request (with the candidate list filled to the pipeline's
    50-project cap), priced on the budget's Luna Standard rates; plus the stage's embeddings."""
    from jarvis.agent_models import catalog
    from jarvis.context.native import extract
    from jarvis.context.native.prompts import payload
    from jarvis.context.native.reconcile import MAX_CANDIDATES, SCHEMA, SYSTEM_PROMPT
    from jarvis.context.native.synthesize import MAX_OUTPUT_TOKENS, Synthesis, _assertion
    from jarvis.context.native.synthesize import SYSTEM as BRIEF_SYSTEM
    from jarvis.context.pipeline import EXTRACT_CANDIDATES
    from jarvis.context.schema import AssertionView, EntityCandidate, TimeBound
    from jarvis.model_map import estimate_tokens

    projects = [EntityCandidate(ref=EntityRef(entity_type="project", entity_id=f"project-{i:03d}"),
                                name=f"{TOPIC} project {i}") for i in range(EXTRACT_CANDIDATES)]
    request = request.model_copy(update={"candidates": [*request.candidates, *projects]})
    scripted, original = ScriptedModel(), extract.call_model
    extract.call_model = scripted
    try:
        result = extract.NativeExtractor().extract(request)
    finally:
        extract.call_model = original
    luna = catalog()["luna"]

    def price(prompt: int, completion: int) -> float:
        return luna.usage_cost({"prompt_tokens": prompt, "completion_tokens": completion})

    passes = {name: {"calls": c, "prompt": p, "completion": o, "usd": price(p, o)}
              for name, (c, p, o) in sorted(scripted.passes.items())}
    statements = [a.statement for a in result.assertions]
    # Reconciliation: at most one call per meeting, MAX_CANDIDATES new statements each against one earlier one.
    items = [{"item_id": f"a{i}", "statement": st, "candidates": [{"candidate_id": f"b{i}", "statement": st}]}
             for i, st in enumerate(statements[:MAX_CANDIDATES])]
    reconcile_prompt = (estimate_tokens(SYSTEM_PROMPT) + estimate_tokens(json.dumps({"items": items}))
                        + estimate_tokens(SCHEMA))
    reconcile_completion = estimate_tokens(json.dumps({"pairs": [{"item_id": i["item_id"], "candidate_id": "b",
                                                                  "relation": "unrelated"} for i in items]}))
    reconcile_completion += REASONING_TOKENS
    # One brief per touched entity when `synthesize_briefs` is on, here over every statement of this meeting.
    views = [AssertionView(**a.model_dump(exclude={"valid_from"}), id=f"a{i}", revision=1,
                           valid_from=a.valid_from or TimeBound(at=request.occurred_at), recorded_at=request.occurred_at)
             for i, a in enumerate(result.assertions)]
    brief = payload(BRIEF_SYSTEM, {"SUBJECT": {"entity_type": "project", "entity_id": "project-000", "name": TOPIC},
                                   "QUERY": None, "SECTIONS": [], "ASSERTIONS": [_assertion(v) for v in views],
                                   "HISTORY": [], "TASKS": []}, "context_brief", Synthesis, MAX_OUTPUT_TOKENS)
    brief_prompt = sum(estimate_tokens(m["content"]) for m in brief["messages"]) + estimate_tokens(
        brief["response_format"])
    brief_completion = MAX_OUTPUT_TOKENS // 4 + REASONING_TOKENS  # assumed: a quarter of the output cap
    embed_tokens = sum(estimate_tokens(t) for t in embedded)
    return {"passes": passes, "assertions": len(result.assertions),
            "partial": result.diagnostics.get("partial"), "extract_usd": sum(p["usd"] for p in passes.values()),
            "extractor_cost_usd": result.diagnostics.get("cost_usd"), "chunks": len(result.diagnostics["chunks"]),
            "reconcile": {"prompt": reconcile_prompt, "completion": reconcile_completion,
                          "usd": price(reconcile_prompt, reconcile_completion)},
            "brief": {"prompt": brief_prompt, "completion": brief_completion,
                      "usd": price(brief_prompt, brief_completion)},
            "embeddings": {"texts": len(embedded), "tokens": embed_tokens,
                           "usd": embed_tokens * EMBEDDING_PRICE_PER_MTOK / 1e6}}


TOPIC = "Clinic rollout"


def render_pipeline(envelope, run: dict, cost: dict) -> str:
    segments = envelope.segments or []
    words = sum(len(s.text.split()) for s in segments)
    chars = sum(len(s.text) for s in segments)
    timings = run["timings"]
    total = {kind: sum(v) for kind, v in timings.items()}
    review = total.get("sources.receive", 0) + sum(total.get(k, 0) for k in REVIEW_STAGES)
    lines = [
        "### Stage timings (deterministic engines, model time excluded)", "",
        (f"Synthetic {round((segments[-1].end_ms or 0) / 60_000)}-minute meeting from `corpus.meeting()`: "
         f"{len(segments)} utterances, {words:,} words, {chars:,} characters, four speakers, planted decisions and "
         f"commitments. Received for `{PIPELINE_OWNER}` on top of the loaded 1x corpus (that owner's two years of "
         "history), then every queued job run through `pipeline.perform` in order with `engine: double` and "
         "stubbed embeddings. Wall time per job, seconds:"), "",
        "| Stage | Jobs | Seconds |", "|---|---:|---:|",
        *[f"| {kind} | {len(v)} | {sum(v):.2f} |" for kind, v in timings.items()],
        "",
        (f"Review-ready (receive through reconcile, when the source is `ready` and its proposals exist): "
         f"**{review:.2f} s** of database and pipeline work. Jobs not succeeded: {run['failed'] or 'none'}."),
        "", "### Model tokens and cost per meeting (computed, no calls)", "",
        (f"The native extractor run on the request the pipeline built (its candidate list filled to the 50-project "
         f"cap) with a scripted model behind `native.extract.call_model`: {cost['chunks']} chunks, "
         f"{cost['assertions']} assertions kept, partial: {cost['partial']}. Prompt tokens are "
         "`model_map.estimate_tokens` (4 characters per token) over system prompt, user content and response "
         f"schema; completion tokens are the answer plus an assumed {REASONING_TOKENS:,} reasoning tokens per call. "
         "Priced with `agent_models.catalog()['luna'].usage_cost` (Luna Standard, $0.20 / $1.20 per million input / "
         "output tokens, no cache discount); embeddings at the budget's $0.02 per million tokens."), "",
        "| Item | Calls | Prompt tokens | Completion tokens | USD |", "|---|---:|---:|---:|---:|",
        *[f"| extract: {name} | {p['calls']} | {p['prompt']:,} | {p['completion']:,} | {p['usd']:.4f} |"
          for name, p in cost["passes"].items()],
        (f"| reconcile (at most one call, {min(20, cost['assertions'])} pairs) | 1 | {cost['reconcile']['prompt']:,} "
         f"| {cost['reconcile']['completion']:,} | {cost['reconcile']['usd']:.4f} |"),
        (f"| embeddings ({cost['embeddings']['texts']} chunks and statements) | 1 | {cost['embeddings']['tokens']:,} "
         f"| 0 | {cost['embeddings']['usd']:.5f} |"),
        (f"| **per meeting, default settings** | | | | "
         f"**{cost['extract_usd'] + cost['reconcile']['usd'] + cost['embeddings']['usd']:.4f}** |"),
        (f"| brief synthesis per touched entity (only with `synthesize_briefs` on) | 1 | "
         f"{cost['brief']['prompt']:,} | {cost['brief']['completion']:,} | {cost['brief']['usd']:.4f} |"),
        "",
        (f"The extractor's own accounting for the same run: ${cost['extractor_cost_usd']:.4f} against the "
         "$1.00 default per-source cap (`context_source_budget_usd`). The reconcile row assumes the one call "
         "pairs each new statement with one earlier statement; the brief row assumes an answer of a quarter of its "
         "6,000-token output cap. Brief synthesis is off by default (deterministic briefs, no model call)."),
    ]
    return "\n".join(lines)


def pipeline_main(server: str, name: str, minutes: int) -> str:
    """Load the 1x corpus into `name`, process one synthetic meeting, compute its model cost; drop the database."""
    from evals.context.corpus import meeting

    url = make_url(server).set(database=name).render_as_string(hide_password=False)
    create_database(server, name)
    engine = _engine(url)
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
        Base.metadata.create_all(engine)
        with Session(engine) as db, db.begin():
            load(db, generate(CorpusSpec()))
        _vacuum(engine)
        envelope = meeting(minutes)
        run = pipeline_run(engine, url, envelope)
        cost = model_cost(run["request"], run["embedded"])
    finally:
        from jarvis import db

        db.engine().dispose()
        engine.dispose()
        drop_database(server, name)
    return render_pipeline(envelope, run, cost)


# ---------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database", default="jarvis_bench")
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--seed", type=int, default=CorpusSpec.seed)
    parser.add_argument("--scale", type=float, default=1.0, help="multiply sources, segments and embedded rows")
    parser.add_argument("--report", help="Markdown output path, or - for stdout only (default: reports/"
                        "native-retrieval-benchmark.md, with a -<scale>x suffix when --scale is not 1)")
    parser.add_argument("--no-hnsw", action="store_true")
    parser.add_argument("--keep", action="store_true", help="keep the database afterwards")
    parser.add_argument("--pipeline", type=int, metavar="MINUTES",
                        help="instead: stage timings and model cost for one synthetic meeting of MINUTES")
    args = parser.parse_args(argv)
    if args.pipeline:
        server, _ = _bench_url(args.database)
        fragment = pipeline_main(server, f"{args.database}_pipeline", args.pipeline)
        print(fragment)
        if args.report not in (None, "-"):
            Path(args.report).write_text(fragment + "\n")
        return 0
    if args.report is None:
        args.report = str(REPORT if args.scale == 1 else REPORT.with_stem(f"{REPORT.stem}-{args.scale:g}x"))

    base = CorpusSpec()
    spec = CorpusSpec(seed=args.seed, sources=round(base.sources * args.scale),
                      segments=round(base.segments * args.scale), embedded=round(base.embedded * args.scale),
                      entities_per_owner=max(10, round(base.entities_per_owner * min(1.0, args.scale))))
    server, url = _bench_url(args.database)
    timings = {}
    create_database(server, args.database)
    engine = _engine(url)
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
        Base.metadata.create_all(engine)
        started = time.perf_counter()
        corpus = generate(spec)
        timings["generate"] = time.perf_counter() - started
        started = time.perf_counter()
        with Session(engine) as db, db.begin():
            load(db, corpus)
        timings["load"] = time.perf_counter() - started
        started = time.perf_counter()
        _vacuum(engine)
        timings["vacuum"] = time.perf_counter() - started
        print(f"corpus loaded: {corpus.counts()} ({timings})", file=sys.stderr)

        qs = queries(corpus, args.queries, seed=args.seed)
        expect = Expect(corpus)
        exact = {}
        for name, (_, build) in CONFIGS.items():
            exact[name] = run_config(engine, corpus, qs, build, expect, history=name == "known_at")
            print(f"exact {name}: p50 {exact[name]['p50']:.1f} ms p95 {exact[name]['p95']:.1f} ms", file=sys.stderr)
        indexed = agree = used = probe = None
        if not args.no_hnsw:
            started = time.perf_counter()
            with engine.begin() as connection:
                # A serial build: parallel workers need dynamic shared memory beyond Docker's default /dev/shm.
                connection.exec_driver_sql("SET LOCAL max_parallel_maintenance_workers = 0")
                for table, index in HNSW.items():
                    connection.exec_driver_sql(
                        f"CREATE INDEX {index} ON {table} USING hnsw (embedding vector_cosine_ops)")
            timings["hnsw"] = time.perf_counter() - started
            _vacuum(engine)
            hnsw_engine = _engine(url, iterative=True)
            indexed, agree, used = {}, {}, {}
            for name, (_, build) in CONFIGS.items():
                indexed[name] = run_config(hnsw_engine, corpus, qs, build, expect, history=name == "known_at")
                agree[name] = agreement(exact[name]["results"], indexed[name]["results"])
                used[name] = plan_uses(hnsw_engine, build, qs[0], corpus)
                print(f"hnsw {name}: p50 {indexed[name]['p50']:.1f} ms p95 {indexed[name]['p95']:.1f} ms "
                      f"agreement {agree[name]:.3f} index {used[name]}", file=sys.stderr)
            probe = semantic_probe(hnsw_engine, qs)
            print(f"probe: { {k: v if k == 'recall' else {x: y for x, y in v.items() if x != 'results'}
                             for k, v in probe.items()} }", file=sys.stderr)
            hnsw_engine.dispose()
        env = environment(engine)
        report = render(corpus, env, timings, exact, indexed, agree, used, qs, probe)
        summary = perf_summary(args.scale, corpus, exact)
    finally:
        engine.dispose()
        if not args.keep:
            drop_database(server, args.database)
    print(report)
    if args.report != "-":
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(report)
        Path(args.report).with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"report written to {args.report} (and .json for report --perf)", file=sys.stderr)
    failed = any(r["violations"] for r in [*exact.values(), *(indexed or {}).values()])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
