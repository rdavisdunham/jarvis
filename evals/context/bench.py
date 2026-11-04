"""Latency benchmark for native context retrieval at the plan's corpus size (plan_final.md section 11).

    uv run --no-sync python -m evals.context.bench                       # full corpus, 200 queries, report
    uv run --no-sync python -m evals.context.bench --scale 0.1 --queries 40 --report -

Creates a fresh database on the server named by JARVIS_DATABASE_URL (default `jarvis_bench`; only names starting
with `jarvis_bench` are accepted because the database is dropped and recreated), creates the model tables, loads
`evals.context.corpus` and times `NativeRetriever.retrieve` (one SQL statement plus hydration: what
`context_search` runs, excluding synthesis) for every query under each scope configuration. Query embeddings are
synthetic, so there are no API calls. Then it builds HNSW indexes on the embedded tables, enables
`hnsw.iterative_scan`, reruns the same queries, checks whether the planner used the indexes and measures top-k
agreement with the exact run. Every run also checks that no hit crosses owners, comes from a deleted source or
(outside `known_at`) is an ended, shadow or future assertion. The database is dropped at the end unless `--keep`.
"""

import argparse
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
    lines = [
        "# Native retrieval benchmark (corpus fixture)",
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
                  "selection, the scope predicates (for raw hits, the segment-to-entity and stale-passage checks) "
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


# ---------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database", default="jarvis_bench")
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--seed", type=int, default=CorpusSpec.seed)
    parser.add_argument("--scale", type=float, default=1.0, help="multiply sources, segments and embedded rows")
    parser.add_argument("--report", default=str(REPORT), help="Markdown output path, or - for stdout only")
    parser.add_argument("--no-hnsw", action="store_true")
    parser.add_argument("--keep", action="store_true", help="keep the database afterwards")
    args = parser.parse_args(argv)

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
    finally:
        engine.dispose()
        if not args.keep:
            drop_database(server, args.database)
    print(report)
    if args.report != "-":
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(report)
        print(f"report written to {args.report}", file=sys.stderr)
    failed = any(r["violations"] for r in [*exact.values(), *(indexed or {}).values()])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
