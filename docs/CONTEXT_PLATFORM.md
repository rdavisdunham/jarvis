# Meeting context platform

Eridani ingests completed meeting transcripts, keeps source-backed context about the people and projects in
them, prepares task proposals for the owner to review, and serves cited context to agents over MCP and REST.
It lives in `apps/api/jarvis/context/` as an extractable module, behind the `context_enabled` flag (default
false). The specification is `plan_final.md` (revision 3, 2026-11-04, kept outside the repository); this
document describes what the code does. Turning it on for one owner: [pilot release note](CONTEXT_PILOT_RELEASE_NOTE.md).
Agent-facing contract: [EXTERNAL_AGENTS.md, "Meeting context"](EXTERNAL_AGENTS.md#meeting-context).

## 1. Module map

```
apps/api/jarvis/context/
  __init__.py      boundary rules (read first)
  schema.py        Pydantic contracts: SourceEnvelope, engine requests/results, RetrieveQuery, ScopePredicate, Hit,
                   brief sections; enumerations as Literals; json_schemas() for the frontend
  engine.py        the four engine Protocols: Extractor, Reconciler, Retriever, Synthesizer
  doubles.py       deterministic engines (tests, offline development, owner setting "engine": "double")
  models.py        the context_* tables (migrations 0048-0050)
  settings.py      one accessor over config.Settings; records the pgvector decision
  sources.py       receive (the one ingestion entry point), revisions, segments, provisional chunks, reads, deletion
  imports.py       TXT, Markdown, WebVTT, SRT and versioned JSON -> SourceEnvelope
  connectors/      registry (upload, api, granola, zoom; unknown labels -> api), granola.py poller, zoom.py stub
  calendar_link.py source -> Google Calendar event association, attendee seeding
  identity.py      actor identities, seeding, negatives, merge and unmerge
  temporal.py      effective intervals, recorded versions, keyed chains, reconciliation write path
  evidence.py      support and contradiction spans, survival rule, deletion and revision propagation
  decisions.py     append-only human/agent/system decisions (Wrong, Changed, confirm, links, merges, approve...)
  apply.py         guarded application of engine results (lock, epoch, pause, revision, watermark)
  native/          chunking, prompts, extract, validate, link, reconcile, embed, retrieve, synthesize
  pipeline.py      job dispatch for every context_* kind, stage runs, continuation and scheduling
  service.py       the read and feedback application service behind REST and MCP
  mcp.py           MCP tool definitions and dispatch
  briefs.py        versioned entity briefs (deterministic v0, optional synthesis)
  views.py         open loops, meeting prep, person view, changes feed (read-time only)
  proposals.py     persistent task proposals from commitments and proposed actions
  publication.py   revision-bound approval and idempotent publication to Eridani tasks or Linear
  notices.py       meeting-ready, consolidated backfill and 17:00 digest notices
  scheduler.py     the housekeeping tick
  lint.py          weekly lint and incremental brief rebuilds
  export.py        canonical history export/import (also the fixture format)
  routes/          __init__ (router, owner_router), sources, context, review, settings, admin, owner, people
apps/web/src/context/   Meetings.tsx, MeetingReview.tsx, PersonPage.tsx, SourcesSettings.tsx, context-api.ts,
                        context-model.ts, meetings.css
scripts/               bridge_submit.py (folder submitter), context_export.py (export/import CLI)
evals/context/         fixture runner, corpus generator, retrieval benchmark, gates and shadow comparison
tests/context/         one test module per context module, recorded fixtures under fixtures/
```

Registration points outside the module (the only places that import it): `worker.py` (queue, dispatch,
housekeeping tick, notification URLs), `external_routes.py` (bot router), `api.py` (owner router, account
export, import body limit), `external_mcp.py` (definitions and dispatch), `models.py` (import block),
`cost_features.py` (labels), `bot_access.py` (scopes, read and write tools), `notices.py` (`context_*` categories),
`review_questions.py` (the `context` Questions kind), `ui_control.py`/`ui_contracts.py` (views).

## 2. Data model

Every table has `owner_id` (the tenant). Context IDs are opaque UUID strings. Foreign keys join only context
tables; Eridani entities are `(entity_type, entity_id)` strings validated in code. Enumerations are plain strings
whose value sets are the Literals in `schema.py`.

| Table | Holds | Key invariants |
|---|---|---|
| `context_connections` | Connector account (Granola): external account, Fernet credential ciphertext, checkpoint, health, last success | Unique per owner, connector, account; key never returned or logged |
| `context_cursors` | A connector's resumable position (Granola scan watermark, backfill progress) | Unique per connection and name |
| `context_sources` | One meeting: connector, external ID, title, upstream owner, current revision, status, availability | Unique per owner, connector, external ID; `deleted_at` is a tombstone |
| `context_source_revisions` | Immutable delivery: content hash, external revision, change kind, payload, occurrence and calendar metadata, participants, display-only summary | Unique per source and revision number; same content never makes a new revision |
| `context_segments` | Exact text and offsets, timestamps, speaker label and attribution, speaker actor link, kind (`transcript`, `private_note`), generated `tsv` (GIN) | The citation unit; never embedded individually |
| `context_chunks` | Extraction chunk: segment range, token estimate, text, `vector(512)` embedding with model and time, `tsv` | Every character owned by exactly one chunk |
| `context_runs` | One stage attempt: stage, status, job, source/revision or entity, engine/model/prompt/config versions, decision watermark, epoch, coverage, `stage_results`, tokens and cost, error | `stage_results` lets a retry skip completed model calls |
| `context_assertions` | Current head: kind, statement, subject/object refs, predicate and key, modality, `[valid_from, valid_to)` with precision, lifecycle, commitment fields (responsible, recipient, due, completion evidence), `vector(512)`, `tsv` | Lifecycle `current`, `contested`, `superseded`, `retracted`, `shadow` |
| `context_assertion_versions` | Immutable recorded history: statement, lifecycle, effective interval, `[recorded_from, recorded_to)`, decision or run | Closed and appended, never rewritten |
| `context_evidence` | Span on a segment: quote, offsets, role (`support`, `contradiction`), origin (`independent`, `derived`), derivation | Quotes occur verbatim; scrubbed on deletion |
| `context_entity_links` | Mention or speaker link to an entity: status (`exact`, `inferred`, `confirmed`, `rejected`, `unresolved`), score, margin, revision, decision | Human-decided links are never overwritten |
| `context_actor_identities` | Namespace (email, granola_speaker, linear_user, organization, alias, calendar_name), value, normalized value, source scope, status, provenance | Strong namespaces name one actor per value and scope; `alias` and `calendar_name` may name several |
| `context_actor_merges` | Winner, loser, snapshot of moved identities and assignments, decision, undone time | Unmerge restores the snapshot |
| `context_decisions` | Append-only decision: target and expected revision, operation, reason, negative flag, origin, principal, command ID, before/after, effective and recorded time, `undoes_id`, per-owner `seq` | Origin is derived, never claimed |
| `context_proposals` | Task proposal: kind (`create`, `update`, `link`, `suggestion`), fingerprint, title, payload, destination, responsible, due, assertions, evidence, revision, disposition, preselected, task | Unique per owner, source, fingerprint (nulls equal) |
| `context_publications` | Approved proposal revision: frozen payload and destination, principal, command ID, unique `operation_id`, task, remote issue, status, outcome | One per approved revision |
| `context_briefs` | Versioned brief: sections, dependencies, watermarks, fingerprint, version, stale flag | Identical fingerprint never makes a new version |
| `context_owner_state` | Per owner: processing `epoch`, `decision_seq`, `paused`, `settings` (JSON) | Row lock serializes decisions and pause |

Migrations: `0047_actor_name_uniqueness` (actor names may repeat; lookups by name resolve only a unique
non-archived match), `0048_context_sources` (creates the `vector` extension), `0049_context_assertions`,
`0050_context_review`. Later packages added no migration.

## 3. Ingestion and the pipeline

**Contract.** Every connector produces a `SourceEnvelope` (connector, external ID, title, occurrence time and
timezone, participants, segments or text, optional private notes, external revision or content hash) and calls
`sources.receive(db, owner, envelope)`. Receipt never calls a model. Same content: nothing new (display metadata
refreshed in place). New content: an immutable revision with segments and provisional chunks and one
`context_ingest` job. A deleted source ignores redelivery of content it already held.

**Adapters.**
- `upload`: Settings → Connections → Meeting sources file import (`POST /api/v1/context/settings/import`).
- `api`: `PUT /api/v1/external/sources/{connector}/{external_id}` with `sources:write`. Known keys (`granola`,
  `zoom`, `upload`, `api`) stay themselves; any other label (`recorder`, `other`) becomes `api` with the label kept
  and the external ID namespaced by it. `scripts/bridge_submit.py` submits a folder of VTT, SRT, Markdown, text,
  envelope JSON or Granola MCP exports.
- `granola` (optional): in-app REST poller, off unless `context_granola_enabled`. Polls every
  `context_poll_minutes`, scans from the day before the last successful scan, dedupes by `updated_at` and content
  hash, imports only notes whose `owner.email` is the connected account, advances the watermark only after a
  whole successful scan; 401 → `auth_failed`, 429/5xx/timeouts → `degraded`, 403/404 park a note (never deletion
  proof). A 90-day backfill runs only while no ingestion or extraction is waiting. `attribution: me` lines go to
  the owner's actor; `private_notes_markdown` becomes `private_note` segments; `summary_markdown` is display only.
- `zoom`: documented stub (`connectors/zoom.py`); Zoom exports go through `api` today.

**Source statuses:** `received`, `queued`, `processing`, `partial`, `ready`, `failed`, `awaiting_input`, `deleted`.

**Job kinds** (all on the `jarvis-context` DBOS queue, concurrency `context_queue_concurrency`, dispatched by
`pipeline.perform`; each records one `context_runs` row per stage and is idempotent):

| Kind | Payload | Does |
|---|---|---|
| `context_ingest` | source, revision | Extraction chunks (5,000-token target, 500-token neighbours, speaker-boundary preference), identity seeding from participants and exactly linked calendar attendees, speaker buckets |
| `context_extract` | source, revision | Native extraction per chunk (detect, enrich, deterministic validation, verify), coverage ledger, admission of candidates with exact evidence |
| `context_link` | source, revision, extract run | Mention links to actors, projects, clients, areas, goals, tasks, Linear issues; consequential ambiguity becomes a Question |
| `context_reconcile` | source, revision, extract run | Bounded reconciliation; source becomes `ready`; queues `context_embed`, `context_brief` for touched entities and `context_proposals` |
| `context_embed` | source, revision (or none: catch-up) | Chunk and assertion embeddings outside any transaction, written only where text is unchanged |
| `context_proposals` | source, revision | Deterministic proposal generation (`proposals.perform`) |
| `context_publish` | publication | Publishes one approved proposal revision (`publication.perform`) |
| `context_brief` | entity | Deterministic brief, or synthesized when the owner's `synthesize_briefs` is on |
| `context_lint` | budget (optional) | One weekly lint unit per transaction (`lint.perform`) |
| `context_delete` | source, through revision | Purge of a deleted source's content |
| `context_granola_sync` / `context_granola_backfill` | connection, days | Granola poll and 90-day backfill |

**Run rules.** Model and embedding calls happen outside transactions; their outputs are persisted on the run
before anything is applied. `apply.guard` re-takes the owner lock and the owner-state row lock and re-checks the
processing epoch, the pause flag, the source revision and the decision watermark. A stale result is marked
`stale` and requeued if still relevant, continued after resume, or dropped (newer revision, deleted source).
Every assertion version or link written emits `context.changed` in its transaction.

**Budget.** The native extractor reserves each model call through `memory_learning` (owner budget, cost feature
`context_extraction`) and caps the revision at `context_source_budget_usd` (default $1), counting earlier
attempts. Hitting the cap leaves the source `awaiting_input` (`pipeline.continue_source` resumes with a larger
cap, reusing completed chunks); an owner-budget refusal or failed chunks leave it `partial`, continued with backoff.
Cost labels: `context_extraction`, `context_linking`, `context_reconciliation`, `context_embeddings`,
`context_synthesis`. Model timeout `context_model_timeout_s` (default 90 s).

**Validation** (`native/validate.py`, deterministic, each rejection has a reason code): quotes must occur in the
cited segment inside the owning chunk; dates and validity bounds only when their expression is quoted and
recomputes from the meeting's local date; every number in a statement appears in the quotes; hypothetical and
negation cues forbid `asserted`; commitments need promise language and a responsible person who spoke, was
named, or is a collective "we"; statements the owner marked wrong are not proposed again; overlapping duplicates
dedupe.

## 4. Temporal and evidence rules

- Effective time is the half-open interval `[valid_from, valid_to)` with a precision. An engine never asserts
  "now": an unsupported start means the meeting's occurrence time.
- Recorded time lives in `context_assertion_versions`: each change closes the open version and appends one.
  `valid_at` asks what was in effect; `known_at` asks what Eridani had recorded then (default: now).
- Current means lifecycle `current` or `contested` and effective now. A future-effective successor never displaces
  today's value: the predecessor stays current with `valid_to` set and becomes `superseded` once the successor's
  start passes (`temporal.rechain`, `settle`).
- Deterministic replacement only for keyed chains: the agreed due date and the explicitly exclusive responsible
  person of a linked work item (task, Linear issue or prior commitment with a stable key). Chains order by
  effective time alone; overlapping or unknown times are `contested`, never ordered by arrival.
- Free-text decisions and rationale go through one LLM duplicate-or-contradiction check; a contradiction with
  certain order is a `supersede` applied in `shadow` unless the owner's `semantic_supersede` setting is on,
  otherwise a `conflict`. Membership, affiliation, preferences and participation are never single-valued.
- Wrong retracts and records a scoped negative decision (re-extraction cannot revive it). Changed adds a
  successor from the supplied effective time. Undo appends a compensating decision and refuses when a later
  decision depends on the undone one. Origin is `human` only when `routing.human_command` accepts the command ID
  and no bot credential is active; agents can never label themselves human.
- Survival: an assertion stays supported while one support group survives. Each live independent span is its
  own group; spans sharing `derivation["all_of"]` are one conjunctive group. Contradictions and derived spans
  never corroborate. Briefs and generated answers are never evidence. A human Changed decision supports its
  successor.
- Deleting a source keeps assertions with alternative support, retracts unsupported ones, reports partially
  supported conjunctive ones for revalidation, scrubs quotes (non-content tombstone) and marks dependent briefs
  stale. A fully extracted newer revision does the same for what only earlier revisions said.

## 5. Identity

People are Eridani `Actor` rows; `0047` lets names repeat. Identifiers are namespaced: strong (email,
`granola_speaker` scoped to its source, `linear_user`, organization) name one actor; `alias` and `calendar_name`
may name several. Seeding: an email resolves or creates exactly one actor through the `actor.create` command; a
name alone is an ambiguous alias (`inferred` only when exactly one actor carries it). The owner's own lines
resolve to the owner's actor. Speaker labels never carry across sources. Automatic links need
`AUTO_THRESHOLD` 0.85 and a `MARGIN` of 0.15 over the runner-up; candidates above `CANDIDATE_FLOOR` 0.5 stay
unresolved with candidates (provisional constants, to be calibrated on adjudicated meetings). Rejected links and
`no_merge` pairs are durable negatives. Merge and unmerge are human decisions from the person page: a merge
snapshots and moves identities, task assignments (through `task.update`), links and speaker links and archives
the loser; unmerge restores the snapshot where nothing changed since and leaves ambiguous later references
unresolved.

## 6. Retrieval

`native/retrieve.py` ranks two independent legs in one SQL statement and fuses them by reciprocal rank fusion
(k = 60):

- lexical: `websearch_to_tsquery` over assertion and (with `include_raw`) segment `tsv`, ranked by `ts_rank_cd`;
- semantic: exact pgvector cosine distance over assertion and (with `include_raw`) chunk embeddings recorded by
  the current embedding model (`text-embedding-3-small`, 512 dimensions).

The scope predicate (owner, entities, kinds, lifecycle, `valid_at`, `known_at`, deleted sources) is inside each
leg's candidate query, so top-k only sees authorized rows and neither leg filters the other. Raw hits are
annotated by joining evidence to assertion lifecycle: a passage supporting a retracted or superseded
interpretation carries that status and the successor's ID, and a segment supporting an ended interpretation the
scope would not show is excluded. When the query embedding fails, the lexical leg answers alone and hits carry
`semantic_unavailable`. There is no similarity floor on the semantic leg yet.

Measurements (`evals/context/reports/`): at the plan corpus (500 sources, 50,000 segments, three owners) on the
development machine the worst exact-scan p95 was 530 ms (entity scope with raw hits) against the 2 s gate; the
bare semantic leg costs 11.7 ms p95. HNSW indexes were not chosen by the planner (the leg orders by distance and
ID) and are not needed; no HNSW migration exists. At 4x scale only entity scope with raw hits exceeded the gate
(p95 2,706 ms), caused by the per-segment entity predicate, not vector search; F2 replaced it with one
precomputed set of the owner's segments about the entities and runs the statement with `jit = off` (see
`evals/context/reports/pilot-perf.md` for the current numbers).

## 7. Agent contract

One application service (`service.py`) serves MCP and REST. Authorization runs before assembly (flag, key and
scope, personal workspace, scope predicate in the query) and again before return (key re-read; every cited
assertion re-checked and must still have a surviving citation; anything failing is withheld and counted in
`gaps`). Statements carry `basis` (`stated`, `inferred`, `source_text`) and `status` (`current`, `contested`,
`historical`, `retracted`); superseded and retracted context needs `include_history`. Linked task state needs
`tasks:read` as well.

| MCP tool | REST twin (under `/api/v1/external`) | Scope |
|---|---|---|
| `context_search` | `POST /context/search` | `context:read` |
| `context_brief` | `POST /context/brief` | `context:read` |
| `context_evidence` | `GET /context/evidence/{assertion_id}` | `context:read` |
| `context_source` | `GET /context/sources/{source_id}` | `context:read` |
| `context_open_loops` | `POST /context/open-loops` | `context:read` |
| `context_meeting_prep` | `POST /context/meeting-prep` | `context:read` |
| `context_changes` | `GET /context/changes` | `context:read` |
| `context_feedback` | `POST /context/feedback` | `context:write` |

Other bot routes: `PUT /sources/{connector}/{external_id}` (`sources:write`, cannot read back), `GET /sources`
and `GET /sources/{id}` (`context:read`), `DELETE /sources/{id}` (`context:write`), and the read-only review
inbox `GET /context/reviews` and `GET /context/reviews/{source_id}` (`context:read`). Keys holding any of
`context:read`, `context:write`, `sources:write` exist only in the personal workspace. Feedback is an agent-origin
suggestion that appears in Questions; only the owner's answer (`context.resolve`) applies it. Agents never
approve, publish or merge.

## 8. Owner (session) routes, review, notices, digest and lint

The web app never reads `context_*` tables or the bot API; it calls session twins under `/api/v1/context`
(`routes/__init__.py` `owner_router`, mounted in `api.py`), which run the same service inside
`service.as_owner(user)`. Non-GET requests need the CSRF header.

| Routes | Purpose |
|---|---|
| `POST search`, `brief`, `open-loops`, `meeting-prep`; `GET evidence/{id}`, `changes`, `sources`, `sources/{id}`, `sources/{id}/segments`; `DELETE sources/{id}` | Session twins of the agent contract |
| `GET reviews`, `GET reviews/{source_id}`; `POST reviews/{source_id}/approve`, `apply-updates`, `dismiss`, `later`, `assertions/{id}/wrong`, `assertions/{id}/changed`; `POST reviews/publications/{id}/retry` | Meeting review (Meetings view) |
| `GET people`, `GET people/{actor_id}`; `POST people/{id}/merge`, `not-same`, `unmerge`, `links/{link_id}` | Person pages and identity corrections |
| `GET settings`; `POST settings/import`; `POST settings/granola`, `granola/{id}/test`, `granola/{id}/sync`, `granola/{id}/backfill`; `DELETE settings/granola/{id}` | Sources panel |
| `GET admin/status`; `POST admin/pause`, `admin/resume`, `admin/rollback-drill` | Operations (work even with the flag off) |

**Review and publication** (`proposals.py`, `publication.py`, plan section 7). Proposals have a stable identity
(fingerprint of responsible person, action words and target work) independent of evidence, so they survive chunk
overlap, retries and revisions. A clear unfinished action of the owner's own with no matching work is a checked
`create`; someone else's work or unclear ownership an unchecked `create`; covered work a `link`; changed existing
work an unchecked `update` with before and after; a useful next step a separate unchecked `suggestion`.
Hypothetical, proposed, negated and question commitments, completed work and retracted interpretations stay
context only. Approval is human-only (`routing.human_command`), records the decision against the exact proposal
revision, freezes payload and destination in a publication with a stable `operation_id` and queues
`context_publish` in one transaction. Publishing writes through `task.create`/`task.update` (and `linear.publish`
for Linear) with deterministic command IDs `ctxpub:<publication>:<step>`, so retries never duplicate. Updates use
the task revision recorded at proposal time; a task edited since is a `conflict`. Publication statuses: queued,
in_progress, published, retrying, failed, conflict, unconfirmed. Approved and published proposals are never
modified by regeneration; changed work becomes a follow-up `update` proposal. Linear matching covers the owner's
synced Linear tasks; widen the sync (`linear.select`, `only_mine=false`) to match others' issues.

**Notices** (`notices.py`, written by `scheduler.tick` from worker housekeeping while the flag is on):
`context_ready` once per meeting review revision (dedup `context:ready:{source}:{revision}`; a later revision
notifies again only for material new pending work); `context_backfill`, one consolidated notice per owner-local
day for historical meetings; `context_digest` during the 17:00 owner-local hour when unreviewed meetings, Later
proposals or deferred context Questions are pending (items arriving after 17:00 wait a day). Quiet hours,
priority and device rules apply through `jarvis.notices`; `notices.current` re-checks a notice before it is pushed.

**Questions.** Contested context, consequential link ambiguity and agent suggestions appear in the Questions
inbox as kind `context` and are answered with the `context.resolve` command.

**Weekly lint** (`lint.py`) runs in the owner-local memory dream slot (Sunday 03:00) when the owner has
`memory_learning` and `deep_sleep_enabled` on, processing changed neighbourhoods under a per-slot budget
(default $0.50, owner setting `lint_budget_usd`) with a resumable cursor: keyed chains settled, overdue untracked
commitments proposed, recurring unresolved mentions and likely duplicate people suggested, contradictions
reconciled, briefs brought up to date. Ten unchanged runs create nothing new.

**Briefs** (`briefs.py`) have six sections (current state and why, open loops, next meeting prep, who is who,
recent changes, unknowns). Version 0 is deterministic. With `synthesize_briefs` on, the model writes sections
outside any transaction and every claim must cite surviving evidence or is dropped and counted as a gap. A stale
brief is rebuilt or served as the deterministic current view marked stale, never as current.

## 9. Operations

**Server flags** (`config.py`, env prefix `JARVIS_`): `context_enabled` (false), `context_queue_concurrency` (2),
`context_source_budget_usd` (1.0), `context_model_timeout_s` (90), `context_poll_minutes` (5),
`context_max_source_chars` (1,000,000; at most 20,000 segments), `context_granola_enabled` (false),
`context_granola_api_key` (server-wide fallback; connections store their own sealed key).

**Per-owner settings** (`context_owner_state.settings` JSON; there is no UI or route for them, set them from a
shell as the release note shows): `synthesize_briefs` (model-written briefs; default off), `engine: "double"`
(deterministic engines, for offline trials), `semantic_supersede` (apply free-text supersessions instead of
shadow; default off), `lint_budget_usd` (default 0.50). The owner state also keeps lint progress and the last
explicitly chosen publication destination.

**Pause, resume, epoch.** `POST /api/v1/context/admin/pause` increments the epoch and sets `paused` under the row
lock: in-flight automatic results fail the apply guard and nothing automatic applies. Sources, reads, manual
corrections and approved publications continue. Resume clears `paused` and keeps the epoch; still-relevant work is
requeued once. The rollback drill pauses, exports the canonical history and proves it imports under a scratch
owner inside a rolled-back savepoint; its response contains the full history, so treat it as an export.

**Export and import.** `scripts/context_export.py --owner OWNER --out FILE [--include-embeddings]` and
`--in FILE [--new-ids] [--dry-run]`. The format (`eridani-context-history` v1) equals the fixture format; it
leaves out credentials and `tsv`, and embeddings unless asked. Import refuses an owner that already has context
rows. The account export (`GET /api/v1/export`) includes `context_history` in the same format.

**Backups.** Railway's managed PITR is not available on the pgvector image production runs since 2026-11-04
(`CLOUD_MIGRATION.md`). Until scheduled encrypted exports exist, take an encrypted `scripts/backup.py once`
export and a `scripts/context_export.py` history export before enabling or changing the pilot.

**Rollback** pauses processing and leaves canonical data readable. Turning `context_enabled` off hides every
route and tool (`CONTEXT_DISABLED`) and stops the tick; tables are never dropped.

**Evaluation.** `evals/context/` holds the fixture-history runner (doubles or recorded responses), the corpus
generator and retrieval benchmark, and the stage F gates and shadow comparison (raw-history vs native vs
combined) that need the owner's verified labels. The app eval catalog has the `meeting_context` feature
(`evals/app/features/meeting_context.json`); the owner's admin routes belong to `operations`.

## 10. Module boundary and extraction

Rules (`context/__init__.py`, plan section 2):

1. The module reaches Eridani only through entity lookup and candidate search (actors, projects, clients,
   areas, goals, tasks, calendar events, Linear issues), the command path for every planner write (`task.create`,
   `task.update`, `actor.create`, `actor.update`, `linear.publish`, notifications as ordinary rows), and auth,
   principal and bot-credential scopes.
2. Nothing outside it imports it except the registration points in section 1.
3. Every reader, the web app included, goes through the context REST and MCP API.
4. Entity references are `(entity_type, entity_id)` strings, never foreign keys. Read-side SQL joins to Eridani
   tables inside the module are allowed for performance.
5. Engines receive authorized inputs and return proposals; only core writes canonical tables.

Extracting it into a service later:

1. Move `jarvis/context/`, its tests, `evals/context/` and migrations 0048-0050 into the new service with its own
   database; `export.py` moves the data (`--new-ids` is not needed, IDs are context-owned).
2. Replace the read-side joins to Eridani tables (`tasks`, `actors`, `projects`, `google_calendar_events`,
   `linear_issues`, `jobs`) with lookups through Eridani's bot API (`records_get`, `changes_list`) or a small
   internal read API.
3. Replace command-path writes with Eridani bot commands under a scoped key (tasks, actors, Linear publish), keeping
   the deterministic command IDs for idempotency, and notifications with a push endpoint.
4. Replace the DBOS queue and `domain.enqueue_job` with the service's own worker; job kinds and payloads stay.
5. Replace `bot_access` scope checks with token introspection against Eridani, or the service's own keys with the
   same three scopes; keep the personal-workspace rule.
6. Point `external_mcp`'s `context_*` delegation and the web app's `context-api.ts` base URL at the service, or
   proxy both through Eridani.

## 11. Deliberately not built

- Granola webhooks; deleted-note visibility and daily inventory.
- Shared folders, colleagues' notes, any source the owner does not own, organization-level knowledge and sharing.
- Windows and Mac recorder clients; the Zoom cloud-recording adapter (stub only); email and AI-chat ingestion.
- Alternative or challenger memory engines, engine switching, a second database backend.
- Legacy personal-memory migration and unified legacy plus context retrieval.
- HNSW or other ANN indexes (measured unnecessary); a semantic similarity floor.
- A read-only Linear search at link time (widen the sync instead).
- UI for per-owner settings and for pause/resume/rollback drill.
- Backups other than the manual encrypted export for the pgvector image (no managed PITR).
