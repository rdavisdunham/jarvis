# Pilot performance, indexing and cost (F2)

Package F2 of the context platform, 2026-11-04. Covers the plan's performance gate (plan_final.md section 11):
p95 `context_search` under 2 s at the corpus fixture size, excluding synthesis; 95% of one-hour meetings
review-ready within 10 minutes when upstream, model and budget are available; cost by meeting, stage and query.
All numbers are from one machine (i7-9700K, 8 logical CPUs, WSL2; PostgreSQL 16.15 with pgvector 0.8.6 in Docker,
`shared_buffers` 128 MB). No real model or embedding calls were made.

**Gate status: met at 1x, 4x and 10x.** Worst p95 is 160 ms at 1x, 398 ms at 4x and 1,183 ms at 10x. No HNSW
index and no migration were added.

## Retrieval latency by scope

Source: `evals/context/bench.py`, 200 queries per configuration after 10 warm-up queries, limit 20, exact vector
scan. Full reports: `native-retrieval-benchmark.md` (1x) and `native-retrieval-benchmark-4x.md` (4x). 1x is the
plan's fixture: 500 sources, 50,000 segments, about 10,000 embedded rows, three owners at 60/25/15%. 4x and 10x
multiply sources, segments and embedded rows. The 10x run (5,000 sources, 500,000 segments, 72,358 assertions,
100 queries) was a one-off to inform the index decision, so its report is not committed.

| Scope (configuration) | 1x p50 | 1x p95 | 4x p50 | 4x p95 | 10x p50 | 10x p95 |
|---|---:|---:|---:|---:|---:|---:|
| owner: assertions | 46 | 58 | 136 | 165 | 372 | 427 |
| owner+raw: assertions, segments, chunks | 75 | 97 | 199 | 244 | 537 | 619 |
| entity: assertions | 29 | 35 | 64 | 75 | 150 | 185 |
| entity+raw: assertions, segments, chunks | 68 | 90 | 154 | 192 | 359 | 459 |
| known_at: assertions as recorded then | 108 | 160 | 263 | 398 | 770 | 1,183 |

Milliseconds of `NativeRetriever.retrieve` wall time: the ranked statement plus hydration, which is the database
part of `context_search`. Every run had zero violations: no hit from another owner or a deleted source, and none
superseded, retracted, shadow or future-effective outside `known_at`.

### What changed: the raw entity predicate and JIT

C4 measured `entity+raw` at 4x at 1,471 ms p50 and 2,706 ms p95, and traced the cost to the raw-hit entity
predicate. Profiling showed two causes.

1. **Per-segment probes.** `_segment_matches_entities` was an OR of correlated EXISTS subqueries (segment link,
   source link via the revision, speaker, evidence for an assertion about the entity). PostgreSQL cannot pull an
   OR of EXISTS up into a join, so it evaluated them for every candidate segment. The chunk leg was worse: it
   evaluated them for every segment inside every candidate chunk. These probes are now one CTE,
   `entity_segments`, the owner's segments about the scope's entities: segment links, source links joined to
   their revisions' segments, the actor's own lines, and evidence segments of assertions about the entities. The
   segment leg semi-joins it by id. The chunk leg semi-joins it on `revision_id` plus a seq range. The CTE is
   computed once per statement in about 50 ms at 4x (4,238 rows for a large project), using the existing
   `ix_context_entity_links_target`, `uq_context_segments_seq` and segment primary-key indexes. No new table or
   index was needed.
2. **JIT compilation.** The scope's OR-ed EXISTS clauses on assertions inflate the planner's cost estimate
   (about 617,000 at 4x) past PostgreSQL's `jit_above_cost` and `jit_optimize_above_cost`. `EXPLAIN ANALYZE` of a
   2.4 s query showed 2,326 ms of JIT compilation and a plan that ran in about 270 ms. The retriever now issues
   `SET LOCAL jit = off` before its statement. The setting lasts until the caller's transaction ends. JIT is
   `on` on the test server and is likely on in production's pgvector image.

Same 4x database and 30 queries for each row (`entity+raw`, ms; other agents shared the machine during these
runs):

| Variant | p50 | p95 |
|---|---:|---:|
| C4 code (per-segment predicate, JIT on) | 2,056 | 5,104 |
| per-segment predicate, JIT off | 1,037 | 1,829 |
| set-based CTE, JIT on | 351 | 2,769 |
| set-based CTE, JIT off (shipped) | 149 | 177 |

Results are unchanged. `tests/context/test_retrieve_parity.py` keeps the replaced predicate verbatim and compares
both versions on the small corpus. It covers 24 queries over project, client and actor entities, five scopes
(entity; entity with superseded and retracted; entity at `known_at`; entity with kinds; owner only), and runs each
with and without a query vector. Hit ids, order, scores, per-leg ranks and lifecycle annotations are identical. A
mutation check confirmed the test catches a change: dropping the speaker branch from the CTE fails it. The
existing retrieve tests pass unchanged.

### ANN decision: no HNSW

The semantic leg stays exact, ordered by `distance, id`. Measured on the bare semantic leg (assertions only, owner
and live-lifecycle filter, top 100):

| Scale | exact (distance, id) p95 | HNSW, distance only, index forced p95 | HNSW recall@100 vs exact |
|---|---:|---:|---:|
| 1x | 12.0 ms | 4.7 ms | 98.1% |
| 4x | 48.0 ms | 4.8 ms | 97.4% |
| 10x | 148 ms | 8.0 ms | 94.8% |

With HNSW indexes present the planner never chose them, because the leg orders by `distance, id`. Top-20
agreement with the exact run was 100% for every configuration at every scale. At 10x the exact leg costs 148 ms
of a 1,183 ms worst p95. An index would save at most that much, and recall would fall to 94.8% at default
`ef_search`. The remaining time is lexical candidate selection, scope predicates and hydration. So exact search
stays, with deterministic tie-breaks and no approximation error.

**When to revisit:** when the exact leg's p95 approaches about 500 ms, roughly 30x the fixture for one owner.
At that point:

- add `USING hnsw (embedding vector_cosine_ops)` on `context_assertions` and `context_chunks` in a migration;
- order the inner semantic leg by distance alone, so the index can serve it, and apply the `id` tie-break in
  `_ranked`'s window ordering after the candidate set (that ordering already ends in `ref_type, id`);
- set `hnsw.iterative_scan = relaxed_order` so owner and lifecycle filters do not starve the candidate set;
- rerun this benchmark's recall probe, and raise `hnsw.ef_search` if recall@100 is below 95%.

The tie at the candidate cutoff then becomes nondeterministic among equal distances. That is acceptable for ANN,
but the parity test would need to compare sets at the boundary.

### Where the remaining time goes

`known_at` is now the slowest scope (1,183 ms p95 at 10x). It reads assertion versions recorded at an instant,
and computes `to_tsvector` on each version's statement at query time because versions have no stored `tsv`. It
is the history and audit path, not the default search. If it ever approaches the gate, the fix is a stored
generated `tsv` with a GIN index on `context_assertion_versions` (a schema change outside F2's files), not a
vector index.

## Review-ready latency per one-hour meeting

Source: `python -m evals.context.bench --pipeline 60`. It loads the 1x corpus, receives one synthetic meeting for
`bench-owner-a` on top of that owner's two years of history, and runs every queued job through `pipeline.perform`
in order with the deterministic engines (`engine: double`) and stubbed embeddings. **Model time is excluded**:
the doubles make no calls.

The meeting comes from `corpus.meeting(60)`: 414 utterances, 8,600 words (150 words per minute), 69,447
characters, four speakers, 11 planted commitments and 6 planted decisions. It runs about 25% longer in characters
than typical speech, because the topic vocabulary uses long words. That is pessimistic for token counts.

| Stage | Seconds |
|---|---:|
| sources.receive | 0.22 |
| context_ingest | 0.44 |
| context_extract (double) | 0.38 |
| context_link | 0.12 |
| context_reconcile | 0.14 |
| context_embed | 0.09 |
| context_brief | 0.09 |

**Review-ready, receive through reconcile: 1.29 s** of pipeline and database work. Embedding and briefs follow
it and are not needed for review.

With the native engine, review time adds the model calls. The extractor runs 4 chunks × 3 passes = 12 calls in
sequence, plus at most one reconcile call. To finish within the 10-minute target, each call must average under
about 45 s. Queue wait and Granola polling delay (`context_poll_minutes`) also count toward the target and are not
measured here. Measure real model latency on the pilot owner's first live meetings (F1's live run). The pipeline
records `started_at` and `finished_at` on each `context_runs` row, so per-stage timings come from that table.

## Cost per meeting and per query

Source: the same `--pipeline` run. The native extractor is run on the exact `ExtractRequest` the pipeline built,
with its candidate list filled to the 50-project cap, and with a scripted model behind
`native.extract.call_model`, so there are no network calls. Token counts:

- **Prompt tokens:** `model_map.estimate_tokens` (4 characters per token) over the system prompt, user content
  and response schema.
- **Completion tokens:** the scripted answer plus an assumed 1,000 reasoning tokens per call.

Prices are from the budget table, `agent_models.catalog()['luna'].usage_cost`: Luna Standard at $0.20 per
million input and $1.20 per million output tokens, with no cache discount. Embeddings are $0.02 per million
tokens (`memory_learning.provider_request`).

| Item (one 60-minute meeting) | Calls | Prompt tokens | Completion tokens | USD |
|---|---:|---:|---:|---:|
| extract: detect | 4 | 24,556 | 4,716 | 0.0106 |
| extract: enrich | 4 | 34,959 | 6,268 | 0.0145 |
| extract: verify | 4 | 25,508 | 4,411 | 0.0104 |
| reconcile (at most one call) | 1 | 1,048 | 1,281 | 0.0017 |
| embeddings (24 chunks and statements) | 1 | 18,908 | 0 | 0.0004 |
| **per meeting, default settings** | 14 | | | **0.038** |
| brief synthesis, per touched entity, only with `synthesize_briefs` on | 1 | 2,205 | 2,500 | 0.0034 |

The extractor's own accounting for the same run is $0.0355. That is 3.6% of the $1.00 default per-source cap
(`context_source_budget_usd`), so a one-hour meeting never reaches the cap. At 20 one-hour meetings a week, the
cost is about $3.30 a month with default settings. It is about $4.15 with brief synthesis on, at roughly three
touched entities per meeting. Both fit inside the default $150 monthly budget.

**Per query:** `context_search` makes one model call, the query embedding (`context_embeddings`). That is about
4 tokens for the benchmark queries and about 20 for a natural question, roughly $0.0000004. Ranking makes no
model call. The calling agent reads the results: at most 20 hits with snippets of up to 300 characters plus
metadata, under about 2,500 input tokens. That is under $0.001 at Luna's Fast tier, charged to the calling
agent's own cost feature, not to the context platform.

## Gate deviations and remedies

None at the fixture size or at 4x or 10x. The remaining risks and their remedies:

| Risk | Remedy |
|---|---|
| Production planner settings differ (JIT, `work_mem`, parallel workers). | The retriever turns JIT off itself. Rerun `bench.py` against a Railway restore or staging database before the pilot. |
| One owner reaches about 30x the fixture. | Use the HNSW plan in "When to revisit" above. |
| `known_at` grows past the gate. | Add a stored `tsv` and a GIN index on assertion versions. |
| The review-ready target depends on model latency. | 12 sequential calls per hour of audio; measure on live meetings. The pipeline's per-stage `context_runs` timestamps already record it. |
