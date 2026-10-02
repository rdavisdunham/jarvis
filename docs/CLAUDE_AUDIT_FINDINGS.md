# Eridani independent audit — findings and repair sequence

Prepared 2026-10-01 by Claude against the local working tree at HEAD `35a8ec9` plus uncommitted changes, using [CLAUDE_AUDIT_HANDOFF.md](CLAUDE_AUDIT_HANDOFF.md) as a map. Read-only: no files were changed, no secrets read, no production access, no paid calls. Frontend `tsc`, `vitest` (136 pass / 1 skip) and `vite build` were run offline; backend tests were not run (they need PostgreSQL).

**Evidence labels.** *Proven* = established by reading the code path end to end (several were spot-rechecked by a second reader). *Hypothesis* = plausible from code but depends on runtime/provider behaviour; write the regression test first. Paths are relative to `apps/api/jarvis/` unless prefixed `web/` (`apps/web/src/`) or repo-root.

---

## 0. Do today (non-code)

1. **Commit the eval harness.** ~25 untracked modules in `scripts/app_eval/` and `evals/test_*.py` plus ~11.5k uncommitted lines exist only on this disk. Tracked `scripts/app_eval/contracts.py:690` already imports untracked `scenarios`; the local `ci.yml` calls `runner plan/coverage` via untracked `campaign.py`. Commit harness + evals tests + `ci.yml` + `.gitignore` together on a branch (partial commits break CI).
2. **Ignore `.codex-remote-attachments/`** — untracked, not ignored, public repo.
3. **README points at the retired stack.** `README.md:1` says Eridani runs from `compose.upgrade.yml`, whose API/worker are `restart: unless-stopped` against local `.env` — the stack the handoff says must never run as a live writer again. Rewrite README for Railway; put the worker behind a compose profile.

---

## 1. Critical / High — proven defects

| # | Area | Defect | Location | Minimal fix |
|---|---|---|---|---|
| H1 | Voice | **Quiet-inbox flush enqueues every user turn as backend work**, independent of Live delegation or completeness. Root mechanism of the Sept 24 "add this to Todo" double-question. Also: 3 s mid-utterance pauses split one request into two jobs; chit-chat and "goodbye" become paid backend runs (`close()` claims remainder). | `work_intake.py:64-115` (`claim_voice`, `flush_voice`), `live_voice.py:335-341,485` | See §4 redesign: flush retains, never enqueues. |
| H2 | Work | **Prose question ends `succeeded` (EVAL-003).** Status is `needs_input` only if `work_needs_input` was called; voice `report_work` additionally drops quiet zero-tool results, so the question is never voiced nor durable. | `work_runner.py:328-332,459-461,513`; `live_voice.py:396-400` | Deterministic guard: final text ends in `?`, no effects/errors → one corrective round, then coerce to `needs_input`. Longer term: structured final `work_reply{text, needs_answer}`. |
| H3 | Work | **First correction to a finished request always fails.** Reservation ID = work ID; successful run closes it; `revise()` reruns same ID; `ensure_room` raises "This model session has ended" (independent of enforcement flag). Existing test only checks task count, so it passes. | `work_runner.py:527`, `budget.py:130-134,157`, `agent_work.py:257-313`, `tests/test_agent_work.py:784` | Per-revision reservation ID (`work_id:revision`) or reopen on revise. |
| H4 | Work | **Same request can execute twice concurrently.** `run()` has no run-level exclusion; revise while `dispatched` bumps `dispatch_revision` and enqueues a second DBOS workflow without cancelling the first. Double spend, checkpoint clobbering, same `work_id:index` with different args. Deploy-overlap DBOS recovery is a second path (hypothesis). | `work_runner.py:197`, `agent_work.py:308`, `worker.py:119,135` | Session advisory lock `work-run:{id}` for the whole run, or store run token/dispatch_revision and exit when stale. |
| H5 | Work | **Stuck `running`/`dispatched` rows freeze an account for 24 h.** Exceptions escaping `run()` (DB errors, deadlocks, `KeyError` on a retired profile at `work_runner.py:202` outside the try) exhaust DBOS retries with no terminal status. Two stuck rows block all new agent work; a stuck `*` reservation blocks every later mutation in the workspace. | `worker.py:47,260`, `agent_work.py:156-168`, `work_coordination.py:206-209` | Catch-all → finish `failed` (budget uncertain); reaper for stale agent rows; move catalog lookup into the try. |
| H6 | Auth | **`records:write` bot scope mutates core Tasks/Notes** without `tasks:write`/`notes:write`. Mirrored records sync title/body/archived/status into core rows; record.create on work/content types creates real Tasks/Notes; `check_command` only checks `record.*` scope. `tests/test_external_agents.py:546` already does it. | `bot_access.py:170`, `structure.py:745-825,900-930` | In `check_command` for `record.*`, resolve backing task/note (or type capabilities on create) and require the core scope too. |
| H7 | Structure | **Two hierarchies diverge in both directions (root of EVAL-008).** `observe_core` never syncs `parent_id`; `sync_capabilities` never writes `project_id`; routing sets flexible home only; projects/areas/spaces created after `ensure()` have no registry row. | `structure.py:33-41,744-790,1004-1021,1055-1059`; `productivity.py` | Make `structure_records.parent_id` canonical; derive legacy columns (see §5 S1). |
| H8 | Structure | **Core read tools only see legacy hierarchy (EVAL-004/008).** `task_list` compact fields/filters, `task_resolve` (also returns full `serial()` for 60 tasks with 20k-char notes), `task_get`, `organization_list`, `project_list`. | `task_tools.py:18-38,83-91`; `task_context.py:79-96`; `tools.py:613-617,739-742,848-858` | Add `record_id`, `home[]`, `home_id` filter (recursive CTE) to task tools; compact `task_resolve`. |
| H9 | Revert | **EVAL-001: Revert compares offset-bearing timestamp strings.** Same class for schedule edits (`anchor_at`, `next_run_at` not in `SKIP`) and `domain.serial` completion strings. Engine never pins session TZ. | `action_history.py:47-54,157,167`; `domain.py:229-236`; `db.py:12-19` | Pin `-c timezone=UTC`; one UTC-normalizing encoder for `snapshot()`/`serial()`; parse old journal ISO strings before compare; keep `revision` as the stale guard. TZ-matrix test. |
| H10 | Routing | **Agent UI patches are recorded as human routing evidence.** `RecordCard` auto-saves editor patches issued by `ui_editor patch`; browser command IDs are UUIDs with no colon, so `structure.py:978` marks them human. `record.update` also drops the rule-provenance exclusion. Pollutes held-out precision. | `web/RecordCard.tsx:38-40`, `web/editor-control.tsx:160`, `web/api.ts:71`, `structure.py:972-980`, `routing.py:119` | Tag agent-originated browser commands (e.g. `ui-agent:<work>:<uuid>`), verify against active AgentWork server-side, pass `human=False`. |
| H11 | Frontend | **Every SSE event triggers full `load()`** (5 requests incl. duplicate `/bootstrap`, all task/schedule pages, notes+memory refetch). No coalescing or stale-response guard. Each write costs ≥2 full reloads; 30 s poll runs even with SSE up. | `web/App.tsx:467-505,549-571,664-666` | Short term: in-flight flag + trailing debounce + sequence numbers; skip poll while SSE connected. Then: events carry entity, patch locally. |

---

## 2. Medium — proven defects

**Voice / work**
- **Duplicate question channels.** `needs_input` question goes out as both `thinking.append` context and commentary in the same tick (test asserts both). `live_voice.py:409-429`. Fix: one spoken channel.
- **Stale/repeat announcements.** Stamp includes `root.revision`, so any later revision bump re-announces; no check that conversation moved on. `live_voice.py:387-393`.
- **Browser goodbye regex closes on bare "thanks"** while work is busy or a clarification pending. `web/voice-ending.ts:61-65`, `web/voice.ts:58-77`.
- **Recovered errors still yield `partial`** (EVAL-002 secondary). Errors clear only on same tool + identical target dict. `work_runner.py:432-445`. Key by affected entity; report recovered separately.
- **Lock-order inversion, bot cancel/reply vs running request** (`work:{id}` → `work-order` vs reverse). `external_mcp.py:179`, `external_service.py:275`, `agent_work.py:240-266`. Take `work-order` first.
- **Budget holds leak.** `close(uncertain=bool(failure))` treats cancellation/limits/timeouts with fully recorded usage as uncertain; `summary()` counts them forever. `work_runner.py:527`, `budget.py:42,58`. Uncertain only when a POST's outcome is unknown.
- **One failing scan blocks all dispatch.** Single try around all supervisor scans and `dispatch_outbox`. `worker.py:367-407`. Dispatch first, per-scan try/session.
- **Expiry finishes running work without locks.** `work_intake.py:126`.

**Mutation semantics**
- **EVAL-002 contract contradiction.** Schema `notes: str | None` + description "explicit null clears" vs validator rejecting null for title/notes/priority/archived. Same pattern in `notes.py:153`, `productivity.py:227`, `structure.py mutate`. `domain.py:72,610-612`, `tool_catalog.py:205`. Map `notes: None → ""`; remove null from schemas where not clearable; add a schema-vs-command null-policy test across `registry()`.
- **Batch/selection task edits never reach structure records.** `structure.py:990-995` only expands `note.tasks`.
- **Task status absent from custom workflow blocks later record edits.** `structure.py:931,1009-1012`, `structure_schema.py:104-106`.
- **`record_list`/`structure_schema` reads take the workspace lock** and run `sync_core_records` (N+1 `data()`), bumping revisions without events → surprise conflicts. Default limit 100/max 200 full bodies blows model context. `structure.py:384-462,1114-1141`, `tools.py:431`.

**Auth / privacy**
- **`records:read` exposes all note bodies/task descriptions** regardless of `notes:read`/`tasks:read`; bypasses `scrub()`. `structure.py:393,406`; external routes, MCP, search, change feed.
- **Viewers / revoked members cannot log out** in a shared workspace (logout goes through `request_access` → `READ_ONLY`/`ACCESS_REVOKED`). `api.py:238`, `access.py:219-225`.
- **Any workspace owner can mint server accounts** via workspace invites on first Google login (operator-paid budget). `google_auth.py:226-231`, `accounts.py:233-236`.
- Low: queued bot work can't use 5 advertised read tools (`external_service.py:404-459`); viewers can't search (`access.py:211-225`); Google link/consent leaves the old session alive (`google_auth.py:290`, `google_routes.py:107`).

**Memory / search / notes**
- **EVAL-005 root cause.** Pairing only uses alphabetic ≥3-char tokens; `fact_key` never read by the dream pass. `memory_review.py:127-151,216-235`. Fix: group active rows by non-empty `fact_key` (conflict review) + numeric mask (`<num>`) bucket. Manual capture and review resolution store `fact_key=""` — fix too.
- **EVAL-006 coupling.** `embeddings()` before `apply_facts()`; each retry re-runs paid extraction. Null-vector commit + `embed_memory` job pattern already exists elsewhere. `memory_learning.py:336`.
- **Failed extractions are never retried** (backfill treats failed v4 as done). `memory_learning.py:405-409`.
- **Reasserting a superseded value is dropped** ("moved back to Austin"). `memory_learning.py:260-264,293`.
- **EVAL-007.** No coverage check; all rejection reasons folded into one `uncertain` count. `note_lists.py:476-492,605`. Log per-entry reasons; deterministic list-split coverage check → focused second pass or `possible_omissions`.
- **Spoken numbers.** Legacy tools use literal `ilike '%q%'` (`task_tools.py:80`, `notes.py:310`, `structure.py:461`); semantic exact check and `websearch_to_tsquery` fail too. Add a shared `canon()` (NFKC, casefold, number-words↔digits, join single-digit runs) on index and query.
- Note search does one embedding SELECT per note (≤1000) and an uncached query embedding. `notes.py:356,381-390`.
- `memory_service.py:123` emits literal `\n` into the prompt.

**Integrations / notifications**
- **Timed-out sync bumps account generation and cancels all queued Google writes.** `google_calendar.py:143-148`, `google_writes.py:27,335-345`. Separate `sync_epoch`.
- **Failed/cancelled publish leaves a PlanningEntry uneditable and undeletable** (`CALENDAR_CONFLICT`). `planning.py:54-68,112-153`.
- **Morning summary marks task-less overnight reminders read but never sends them.** `notices.py:174,283-299`.
- **No `pushsubscriptionchange` handling / re-registration** — devices silently go dark after 410. `web/../public/sw.js`, `web/App.tsx:1592`.
- One bad planning row breaks the whole calendar view (no per-row guard in `planning.project`, `planning.py:204`).

**Frontend**
- Constant polling regardless of visibility: `/ui/sync` every 700 ms (`App.tsx:1332`), Activity every 3 s despite SSE (`Activity.tsx:58`), voice status every 400 ms with full-App rerender (`voice.ts:330`; each poll is an auth DB transaction).
- Global `busy` flag; quick-add silently drops captures while any mutation is in flight (`App.tsx:715`); no optimistic completion.
- Dialogs declare `aria-modal` but have no focus trap / initial / return focus (14+). Use native `<dialog>.showModal()`.
- JS mobile breakpoint 1000 px vs CSS 500–1200 px variants; foldables get mobile JS with desktop CSS.

---

## 3. Hypotheses (write the test first)

- Lock-order inversion between `flush_voice` (row lock → advisory) and `append_voice` (advisory → row); flush also holds inbox locks for the whole scan transaction, and synchronous DB calls in the asyncio loop stall all voice sessions. `work_intake.py:33-44,65,108`, `worker.py:368-399`.
- Deploy overlap: DBOS recovers PENDING workflows still executing in the old process (default executor ID) → double execution. Fixed by H4's run lock.
- Bot keys revive on member re-invite (not revoked at removal). `accounts.py:307-366`.
- Engine lacks `hide_parameters=True`; exceptions could log note content. `db.py:12`.
- Reverting a record edit that added a bound field sets it to None, which the task validator rejects (`action_history.py:184-186`).
- In-process selection dict breaks across workers/restarts (`task_tools.py:17`).
- Retryable Google `RefreshError` (5xx) mapped to `needs_reconnect` permanently. `google_calendar.py:38-41`.
- "Use Google" conflict resolution stores raw `dateTime` in calendar TZ, failing round-trip. `planning.py:143`.
- Live session keeps startup memory after a mid-session forget. `live_voice.py:95,107`.
- Model-called `search_feedback(accepted)` is recorded as explicit acceptance on prompt trust alone. `search_learning.py:108-123`.

---

## 4. Recommended redesign — voice clarification ownership

1. **Flush retains, never enqueues.** Quiet turns are marked `retained` with a TTL; the cursor does not advance.
2. **Live delegates explicitly** with `{intent_complete, utterance_event_ids}`; the server claims exactly those events once their final transcript lands (not time-based).
3. **Live asks for simple missing details itself** before delegating; the server joins retained draft + answer into one job.
4. **Fallback:** on close/long silence, retained turns become one `draft` per conversation ("Unsent voice request — Send / Discard"), never auto-run.
5. **Once a backend job is `needs_input`, it owns the question.** Live gets it as thinking context only and must not re-ask; the next delegated user turn in that session binds to `work_answer` deterministically (model may reject if clearly unrelated).
6. **Keep H2's guard** as a backstop. Don't enqueue the turn that triggered `voice_end`.
7. **Regression set:** Sept 24 flow (no work after incomplete turn, one job after "milk", no stale question); 3.5 s pause → one job; goodbye not enqueued; answer arriving while original still running binds when it reaches `needs_input`.

Push `voice.state` over SSE and reduce the 400 ms poll to a 5 s lease heartbeat. Signal the worker (`pg_notify`) on enqueue instead of the 5 s scan to cut up to 5 s from every voice request.

---

## 5. Design suggestions

- **S1. Retire legacy organization tables.** (1) `structure_records.parent_id` canonical; legacy `project_id/space_id/area_id` read-only derived. (2) Reimplement `project.*/goal.*/space.*/area.*` as `record.*` adapters with fixed type IDs. (3) Rebuild `organization_list`/`project_list` as record projections; task filters on `home_id`. (4) Backfill migration, then drop legacy columns/FKs, `ensure()` import and `legacy_kind` paths. This removes the EVAL-004/008 class at the source.
- **S2. pgvector.** Exact Python cosine over JSONB breaks voice latency around 2–5k search chunks / 5–10k memory facts. Add `vector(512)` columns, backfill, dual-write, query `ORDER BY embedding <=> q LIMIT 50` unioned with FTS; HNSW only beyond ~100k rows. `deploy.py:119` already probes the extension.
- **S3. Planner work windows.** `work_windows` are never consulted by the solver; local-only scheduling can place blocks at 02:00. Subtract out-of-window time by default. Deduplicate free/busy code (`planner.py:77-88`, `planning.py:289-299`).
- **S4. Cost/latency.** Prefilter memory extraction (skip short / no first-person cue); cache query embeddings in `memory_service`; take voice memory lookup off the critical path (`voice.py:365`); debounce `organize_note` 30–60 s; reuse one `httpx.AsyncClient`; cache `budget.summary` per round instead of rebuilding 30-day usage under lock; batch `sync_deadlines` and `prepare_deliveries`.
- **S5. Routing gate is effectively unreachable** (100 held-out at 20% split ≈ 500 matches/phrase), so every learned rule goes to review. Decide whether that is intended.
- **S6. Key rotation.** Single Fernet key for credentials, work input and device context with no `MultiFernet` path. Add rotation and escrow verification.
- **S7. Frontend architecture.** `App.tsx` is 3,430 lines with 82 `useState`/30 `useEffect`, no ESLint (no exhaustive-deps). Split into data store + `useVoiceSession` + `useUIBridge` + route components. Replace CopilotKit (used only as a local registry; pulls rxjs/protobuf/phoenix/ag-ui ≈700 KB source) with a small Map registry; lazy-load dialogs/views. PlannerApp chunk is 950 KB min / 252 KB gzip; expect >40% reduction.
- **S8. Quick lists (requested).** Quick-add "List" toggle; `Packing: passport, charger, socks` creates a checklist immediately. Items are ordinary Tasks under a hidden container of kind `quick` (reminders, search, voice, Activity work unchanged); optional chips for due date / space / share; "Promote to project" flips kind. Offer promotion only at ~10 items or a deadline.
- **S9. Conflict UX.** One shared "Changed elsewhere — Keep mine / Use theirs / Compare" banner reusing TaskDetails' comparison UI.
- **S10. Integrations.** Linear post-write `issue.history` check for non-atomic CAS; guard `apply_remote` against older `updatedAt`; accept inclusive `last_day` for all-day events in the model tool schema; ±60 s deadline/reminder dedupe window; don't backfill a day of pushes to a newly subscribed device.

---

## 6. Operations, CI, test integrity

- **No acceptance coverage on the highest-risk features:** `fully_bound=0` for accounts, workspaces, privacy, external_agents, receipts, google/linear writes, operations, live_voice; ~91% of the 98 bound cases are memory and task CRUD.
- **Shrink to a ~200-case risk-weighted core set** (5–8/feature) rather than binding all 1,001; keep the rest as a labelled exploratory catalog. `repeats≥3` for live-model cases — current runs are repeats=1 and cannot distinguish intermittent from consistent. Binding order: authority → mutation/revert → clarification/queue/live protocol → data-loss paths → search/routing/structure → UI.
- **Half-suite failure status:** clarifications.01 intermittent (EVAL-008); clarifications.03 consistent (EVAL-003); memory_dream.19 and time_deadlines.26 deterministic; memory_capture.24 consistent; task_edit.25 still present (passed only after retry). The six downgrades are evidence gaps, not defects.
- **CI:** no pip-audit / `npm audit` (explicitly `--no-audit`), no Dependabot/CodeQL/gitleaks on a public repo; coverage computed but never ratcheted; branch protection unconfirmed; ruff correctness-only.
- **Migrations:** all downgrades raise and `overlapSeconds: 0` means release N-1 serves against schema N. Add an N-1 compatibility CI job and an expand/contract rule.
- **Backups:** PITR lives in the same Railway project the whole-project `railway.ts` plan manages; R2 inactive; key escrow undocumented. Activate R2, run a restore drill, record who holds `JARVIS_BACKUP_KEY` and the integration key, test decrypt from escrow.
- **`.railway/railway.ts:6-12`** declares both PG16 and PG18 services with volumes. Confirm the active one with the provider, annotate PG18 as retired-but-preserved, require a reviewed plan with destructive-diff check.
- **Dockerfile.upgrade:** tag (not digest) pins; `COPY apps/api` before `uv sync` defeats layer cache; default `CMD` skips `deploy.py` validation. `deploy.py:208-215` swallows migration exceptions (log type + revision).
- **Single-process voice:** document as accepted; announce reconnect to clients on SIGTERM.
- **Hygiene:** archive legacy `backend/`, `client/`, GPU `docker-compose.yml`, stale root `Dockerfile.backup` (runs as root); run manifests should record a dirty-tree flag / app-source hash.

---

## 7. Prioritized repair sequence

1. **Safety of the repo and ops (today):** §0 items; reconcile `railway.ts`; enable secret scanning.
2. **Security:** H6 (records→core scope), `records:read` body exposure, logout for viewers/revoked, account-minting via workspace invites, revoke bot keys on member removal.
3. **Durable-work integrity:** H4 run lock, H5 catch-all + reaper, H3 per-revision reservation, lock-order fix, budget-hold leak, per-scan isolation in the supervisor. These are small, deterministic, and currently can freeze or double-charge users.
4. **Deterministic eval defects:** H9/EVAL-001, EVAL-002 null policy + recovered-error status, EVAL-005 fact_key/numeric review, EVAL-006 null-vector commit, failed-extraction retry.
5. **Voice clarification redesign (§4)** with H2's guard shipped first as a standalone backstop.
6. **Structure consistency:** expose `home`/`record_id` in task tools (H8) as the fast EVAL-008 fix, batch-edit sync, workflow-status fallback, lock-free compact `record_list`; then begin S1 legacy retirement.
7. **Routing provenance (H10)** before any rule auto-activation is relied upon.
8. **Integrations:** sync epoch, stranded PlanningEntry, morning summary, push resubscription, per-row calendar guard.
9. **Frontend performance:** coalesced load + visibility-aware polling (H11), voice state over SSE, optimistic capture, dialog primitive; then App split and bundle trim.
10. **Search normalization** (`canon()`), then pgvector when data size warrants.
11. **Test integrity:** ~200-case core set in the binding order above, repeats≥3, coverage ratchet, N-1 migration job.

---

## 8. Repair status (2026-10-01, branch `audit-fixes`, uncommitted)

Verified after all fixes: backend **783 passed / 1 skipped** under both `PGTZ=UTC` and the server default America/Chicago (baseline was 688 + 6 timezone failures); evals 82 passed / 2 skipped; web `tsc` clean, vitest 163 passed / 1 skipped, build OK (PlannerApp 950 KB → 435 KB min, 252 KB → 126 KB gzip); CI ruff rules clean. No migrations or model column changes. Regression tests: `tests/test_audit_*.py`, `tests/test_push_registration.py`, `tests/test_spoken_task_search.py`, and new vitest files.

**Fixed:** §0 items 2–3 (README, compose worker behind `--profile worker`, `.codex-remote-attachments/` ignored); H1 (flush retains, Live delegation claims one combined job, goodbye/chit-chat not enqueued, close-time action-verb fallback), H2 guard, H3, H4 run lock, H5 catch-all + reaper, H6, H7 (minimal two-way parent sync), H8 (`home`/`record_id`/`home_id` in task tools; compact `task_resolve`), H9/EVAL-001 (UTC session pin + UTC encoder + instant comparison), H10 (`ui-agent:` command IDs), H11 (coalesced sequenced loads, visibility-aware polling). All §2 medium items, including EVAL-002 null policy with schema/validator agreement test, EVAL-005/006/007, failed-extraction retry, superseded reassertion, spoken-number normalizer (`text_normalize.py`, applied to semantic, memory, notes, task_list, task_resolve, record_list), records-scope body exposure and body probing, viewer logout/search, account minting, session and bot-key revocation, lock orders, budget-hold leak, supervisor scan isolation, Google sync epoch, stranded PlanningEntry, morning summary, push resubscription + no backlog to new devices + VAPID 403 deactivation, per-row calendar guard. Hypotheses confirmed and fixed: bound-field revert, retryable RefreshError, "use Google" timezone. CopilotKit replaced by a local registry; heavy dialogs lazy-loaded; shared focus-managed Dialog. CI security job, Dependabot, eval coverage ratchet, Dockerfile layer caching, deploy migration-failure logging, `railway.ts` warnings, key-escrow checklist.

**Not done / needs the owner:**
- Commit: the harness and these fixes are uncommitted on `audit-fixes` (a stale empty `.git/index.lock` from Sep 20 must be removed first).
- Deliberately deferred: S1 legacy-table retirement, S2 pgvector, S3 planner work windows, S8 Quick lists, App.tsx split, conflict banner (S9), ~200-case eval core set, N-1 migration CI job, archiving legacy directories.
- Manual: apply the branch-protection ruleset (docs/CI.md), R2 credentials + restore drill + escrow, confirm the active Railway database, pin gitleaks/action SHAs and image digests, remove `@copilotkit/react-core` from package.json (needs `npm install` to update the lockfile).
- Open risks: voice close-time fallback is a verb heuristic; an answer given while the original job is still running becomes its own job; one extra pooled DB connection per running request (fits defaults); first CI security run may surface historical findings to triage.
- Not exercised: Playwright e2e (needs live API), real GPT-Live audio, real Google/Linear, physical devices.
