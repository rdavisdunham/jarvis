# Semantic search and search vocabulary

September 18, 2026.

## User behavior

The top search finds actionable tasks across custom collections. Notes keep their
own search-by-meaning control. Eri can search every accessible custom record type,
field definition, option and relationship. Exact names, current fields and hierarchy
produce deterministic matches; cloud embeddings and text retrieval supply possible
matches outside an inferred classification. Eri evaluates both lanes and chooses
what to show. Possible relevance does not silently change a saved assignment.

For example, “the pest control company” can resolve to ABC. Tasks filed under ABC
are structured results; an insect-treatment task under another client remains a
possible result, with its actual saved assignment visible. Explicit date/status/
field/home filters still constrain both lanes; strict mode excludes the possible
lane. Same-name clients stay ambiguous until resolved.

Search aliases are personal vocabulary, not a factual claim about a company and
not independent training examples for organization. Field aliases and record
aliases have distinct stable keys. Existing IDs survive renames; changed field
semantics invalidate older mappings. A forgotten mapping leaves a suppression
marker so replayed evidence cannot recreate it.

A backend selection alone teaches nothing. A user can teach a provisional alias
by opening/using a selected result or continuing within 15 minutes of actual
presentation without a correction. Browser replies need visible message elements;
selected visual results need visible cards; voice requires a captured spoken target.
Silence, hidden output and wake-word-only input stay unknown. Obvious corrections
withdraw the weak positive; Eri can attribute nuanced corrections with search_feedback.
Both account and workspace scope apply. External bots never supply implicit feedback.
Turning learning off stops new search-session capture and suppresses pending feedback.

Settings → Organization → Search aliases exposes sources, confirmation, correction,
pause, forgetting and learning preferences. Provisional aliases can help retrieval
immediately. The weekly organization review may propose a corresponding rule, but
only a separate review acceptance activates it. Explicitly accepted rules then
remain independent of the search alias. Shared workspace vocabulary does not become
another member's personal routing rule.

## Architecture

- `search_index.py`: bulk canonical projection; full text in overlapping chunks;
  batched `text-embedding-3-small` vectors with 512 dimensions through the existing
  provider adapter. Content fingerprints skip unchanged documents. Mutation events
  invalidate the workspace generation, including changes to linked/inherited names.
- `index_search`: durable outbox/DBOS work, bounded retries and incremental backfill.
  Provider waits occur outside transactions; a generation change discards stale work
  and queues another pass. Existing sources are authoritative throughout.
- `search_service.py`: fresh membership checks and canonical reads before and after
  embedding waits, PostgreSQL text ranking, exact names/fields and vector scoring.
  Missing/stale vectors cannot override current records. A provider failure retains
  text search. Pagination has no newest-1,000-record cutoff. Query vectors have a
  short bounded owner-scoped cache to avoid paying again for each result page.
- `search_documents` and `search_index_states`: derived, rebuildable index data.
- `search_sessions`, `search_aliases`, `search_preferences`: separate account/workspace
  vocabulary and reversible interaction evidence. No writes to personal memory or
  independent human routing observations. Durable-agent alias phrases are checked
  against the captured request, not just the model's rewritten search query.
- Backend tools: `record_search`, `search_select`, `search_feedback`; record/search
  groups expose detailed schemas on demand. `ui_records` can show just selected IDs.
- Browser API: `/api/v1/search/records`, `/selection`, `/feedback`, `/events`,
  `/aliases`, `/aliases/{id}`, `/preferences` under the same `/api/v1/search` prefix.
- Bots: POST `/api/v1/external/search` and MCP `record_search`, requiring `records:read`.
  Existing token expiry, revocation and workspace membership rules still apply.

Current storage is PostgreSQL JSON vectors with exact in-process scoring. This is
appropriate for the present small workspaces; pgvector indexing is a later capacity
upgrade. Existing note embeddings remain available for the legacy rollback path.
No Qdrant service or additional model runtime was introduced.

## Verification

- Full backend suite: 656 passed, one existing skipped test. Fourteen focused search
  regressions cover misfiling, strict filters, inherited fields/explicit clears,
  options, duplicate names, changed data during embedding, >1,000 records, paging,
  alias evidence/corrections/forgetting, review-only rules, account isolation,
  revocation during a provider wait, and API/MCP scopes without bot learning.
- Frontend: 130 passed, one deliberately disabled Realtime test skipped; TypeScript
  and production build pass. Existing bundle-size warning remains.
- Migration 0017 applies cleanly; Alembic reports no model/schema differences.
- All four isolated browser suites pass, including semantic search at 390, 820 and
  1440 px, selected mixed results, opening a card, alias controls, and existing
  planner/navigation/chat behavior. Screenshot: `artifacts/semantic-search/aliases.png`.
- Real-provider evaluation: six synthetic scenarios each for GPT-5.6 Luna and
  Gemini 3.8 Flash, using production tool schemas/adapters and real embeddings in
  a disposable local database. Both returned correct outcomes in all six. Two
  trajectories hit a rejected selection and recovered. No unearned aliases were
  created. Field-definition questions can be answered directly from schema data;
  those answers do not teach an alias without a selected search interpretation.
- Evaluation grades selected record identities and grounded answers, not one exact
  tool sequence or the absence of a title explicitly described as excluded. The
  script records trajectories and recovered errors; this is a small regression
  sample, not a measured production accuracy rate. Approach informed by
  [OpenAI's evaluation guidance](https://developers.openai.com/blog/eval-skills).

Reproduce with `pytest tests/test_semantic_search.py`, `npm test` in `apps/web`,
`python scripts/validate_custom_planner.py`, and, explicitly opting into paid calls,
`python scripts/evaluate_semantic_search.py --run` against a local test database.
CI includes the new backend and browser checks and uses synthetic credentials.
The paired provider eval is intentionally opt-in and never runs in CI.

## Rollout and rollback

1. Deploy migration/code on API and worker with `JARVIS_SEMANTIC_SEARCH_ENABLED=false`.
2. Run `/app/.venv/bin/python -m jarvis.search_index` in the deployed Railway worker
   to queue backfill (use the application virtualenv, not the container system Python).
3. Inspect `/app/.venv/bin/python -m jarvis.search_index --status`; all workspaces should be ready.
4. Enable `JARVIS_SEMANTIC_SEARCH_ENABLED=true` on both services. Queue a fresh pass
   to cover edits made while the flag was off, then verify readiness and index status.
5. Roll back behavior by disabling the flag on both services. Keep derived tables
   and evidence; do not downgrade away retained data. Existing keyword search stays
   available, and older note-search behavior remains available.

Production rollout verified: both Railway services have the feature enabled and
all 67 initial search documents are ready, with zero pending generations/errors.
GitHub CI passed for the implementation and the public readiness endpoint is healthy.

Physical phone/Live acceptance remains on the Todo list. Synthetic transcripts and
browser visibility tests cannot prove what was heard through a real microphone session.
