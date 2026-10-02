# Eridani architecture and audit handoff

Prepared October 1, 2026, for an independent Claude audit. This describes the inspected local working tree, not a freshly verified production deployment.

Eridani is a voice-enabled productivity application: a planner, task tracker, linked notebook, calendar, and configurable organizational system. Its assistant is Eridani, or Eri. The product goal is to let a person organize work in their own vocabulary and use conversation to capture, find, change, and schedule it. The assistant should be witty, playful but formal, and primarily useful. It should act on clear requests and report actual saved changes rather than require routine approval.

The repository and Python package retain the name Jarvis. The current application is a React frontend, Python FastAPI backend, PostgreSQL database, and a separate DBOS-backed worker. It is not the original local speech/Mem0/Qdrant prototype described in most of the root README.

## Scope and evidence

The checked-out HEAD is `35a8ec9bcf5fd56a53866695efcf47a8dafc89fc`, titled “Default paid evaluations to Luna and pause Gemini comparisons.” Substantial eval automation, Langfuse export, test-fixture changes, and TODO updates are present locally but uncommitted. A reviewer who clones only GitHub main will miss them.

This handoff was built by reading source, configuration, migrations, documentation, and saved evaluation findings. No production database was queried, no current cloud settings were inspected, and no fresh full test suite or paid campaign was run for this document. Historical deployment and test evidence is identified as such. Model names and prices mentioned here describe repository configuration, not an independent October 1 verification of provider availability or pricing.

Use [the accompanying source snapshot](AUDIT_SOURCE_SNAPSHOT.json) to identify the source state and local changes. It contains file names, hashes, and Git status, not credentials or user records. The source tree is authoritative when older prose disagrees.

Recommended accompanying material for Claude:

- This document and the current source tree, including local eval additions.
- [TODO](TODO.md), the work ledger. Completed implementation, pending fixes, and device acceptance are distinct.
- [Functionality inventory](APP_FUNCTIONALITY.md), covering 40 feature areas.
- [Evaluation findings](../evals/app/FINDINGS.md) and [verification plan](TESTING_PLAN.md).
- The September 23 [independent grader report](../artifacts/app-evals/half-20260923-sol-graded/grader.md) and adjacent grader.json, if sharing the locally retained synthetic evidence.
- [UX recommendations](../recommendations.md), as historical audit input rather than a claim that every suggestion is outstanding.
- Relevant feature documents linked below. Older PRDs explain intent, not necessarily current behavior.

Do not bundle `.env*`, `.runtime`, database dumps, provider keys, account exports, or personal attachments. The ignored artifacts directory contains more than the chosen audit evidence; select the synthetic report files deliberately.

## 1. Runtime and deployment

### Components

The frontend is React 19 and TypeScript, built with Vite. It uses Lucide icons, Zod for typed browser actions, CopilotKit's headless frontend-tool registry, Vitest, and Playwright. There is no separate Next.js server or hosted CopilotKit agent service.

The backend is Python 3.12 with FastAPI/Uvicorn, Pydantic validation, SQLAlchemy 2, psycopg 3, Alembic, HTTPX, WebSockets, DBOS, cryptography, Web Push, Google OAuth support, and the MCP SDK. Dependency resolution is recorded in uv.lock and the frontend package-lock.json.

PostgreSQL is the canonical store for product data, sessions, memberships, command receipts, agent work, source evidence, derived search data, and DBOS system state. Local Compose and CI pin PostgreSQL 16.15. Current search vectors are JSONB arrays; no production pgvector index or Qdrant service is used by this application.

The same container image runs in two roles:

- API: HTTP endpoints, authentication, frontend static assets, SSE events, active voice-session controllers, and browser control coordination.
- Worker: durable accepted requests, reminder delivery, integration sync/writes, extraction, indexing, and periodic review work.

The multistage [Dockerfile](../Dockerfile.upgrade) builds the web bundle with Node 22, installs the locked Python environment, copies migrations and the bundle, then runs as a non-root user. The API serves the bundle and its own routes on the same origin.

### Interconnection

```text
Browser React app
  | HTTPS: commands, reads, work submission, UI acknowledgments
  | SSE: persistent record/work change events
  | status polling: voice state and transcript fragments
  v
FastAPI API ---- SQLAlchemy ---- PostgreSQL
  |                               | application records and command receipts
  | creates provider voice call   | Job + transactional Outbox
  | attaches control WebSocket    | DBOS workflow system state
  v                               v
OpenAI GPT-Live <--- WebRTC ---> Browser microphone and audio playback
                                  ^
                                  | current-device UI action / acknowledgment
DBOS worker ----------------------+
  | selected backend model: Luna / Gemini / retained Groq option
  | validated domain tools -> records + receipts + action journal
  | Google Calendar / Linear durable sync and writes
  | personal memory / organization review / note filing / search indexing
  | Web Push delivery
  v
PostgreSQL events -> API SSE -> browser refresh
```

There is no Redis broker, separate Celery deployment, local Whisper process, or required local GPU in the current runtime.

### Hosting status and boundaries

The cloud runbook records a successful Railway migration to `https://app.eridani.app`, with a separate web/API service, worker service, and PostgreSQL service. Cloudflare handles domain services and is the intended independent R2 backup destination. Supabase is not part of the inspected runtime.

The September 16 records document PostgreSQL 16.15, private database connectivity, working native PITR, and a successful timestamp recovery drill. Those are historical observations, not a fresh assertion of current backup health.

[deploy.py](../apps/api/jarvis/deploy.py) validates cloud configuration, requires HTTPS and Google credentials, prohibits public pairing login, checks encryption configuration, and exposes migration/preflight entry points. The cloud entry point honors Railway's PORT. API and worker run the same migration head; readiness compares the database Alembic version with the release.

Two important limits:

1. Live controllers are process-local. Startup requires `WEB_CONCURRENCY=1`, and the API entry point explicitly starts one Uvicorn worker. Horizontal API scaling needs a session-routing/state design; it is not safe to assume arbitrary replication already works.
2. A PostgreSQL advisory lease allows one worker supervisor for the database. DBOS runs concurrent jobs within that supervisor. This is not a completed multi-supervisor scaling design.

Staging is deliberately constrained: both the worker and external services must be off. Provider/push credentials are cleared from in-memory settings when external services are disabled. This helps prevent a restored copy from sending reminders, syncing live accounts, or spending money.

The old local production stack was retired after cloud cutover. Do not start it as an independent live writer against its stale data.

**Infrastructure audit warning:** [.railway/railway.ts](../.railway/railway.ts) still contains both PostgreSQL 16 and 18 service declarations and template references. Treat it as a potentially stale whole-project plan, not proof of the actual active database or an instruction to apply it. Reconcile with the provider before any infrastructure change.

Sources: [cloud runbook](CLOUD_MIGRATION.md), [config.py](../apps/api/jarvis/config.py), [deploy.py](../apps/api/jarvis/deploy.py), [worker.py](../apps/api/jarvis/worker.py), [Compose](../compose.upgrade.yml).

## 2. Database model and organizational flexibility

### Operational records versus user-defined organization

The application has two deliberately connected layers.

**Operational services** provide strongly defined behavior: Task, Note, Schedule, Occurrence, PlanningEntry, Notification, and provider integration records. These own things like task completion, reminder recurrence, note bodies, and remote synchronization.

**The configurable record registry** provides the user's vocabulary and organization. It stores:

- `structure_schemas`: one versioned JSON definition per workspace namespace.
- `structure_records`: typed records with title/body, JSON custom values, workflow status, main parent, revision, provenance, and optional task_id/note_id.
- `structure_links`: named relationships between records.
- `structure_proposals`: pending schema edits with author, source revision, impact, expiry, and apply state.

Creating a custom type does not execute DDL or create a new SQL table. A “Client,” “Course,” or “Property” is a definition and a set of rows in the same registry. Descriptions are required for types, fields, and relationships. Stable IDs are separate from editable display names.

The default vocabulary includes Space, Area, Client, Project, Goal, Task, and Note. It is a starting template, not a mandatory ontology.

### Capabilities

Types can enable:

- **work**: operational tasks, statuses, planned day, deadline, priority, estimate, assignee.
- **content**: authored notes and the note pipeline.
- **timeline**: start and target dates, shown as a range.
- **metric**: baseline, current value, target, and unit.

A work record is linked to a canonical Task. A content record is linked to a canonical Note. Generic fields bound to operational behavior must retain their expected scalar types and cannot inherit. Labels can change; the meanings of completion, authority, and scheduling cannot be redefined arbitrarily.

Supported custom field kinds include text, long text, number, boolean, date, datetime, select, multiselect, and relation. Relation fields may target allowed types and allow multiple values. The schema is bounded: for example, at most 50 types and 60 fields per type in the current validation.

### Hierarchy and links

Each record has one main parent, giving it a primary organizational home. For example:

```text
Work
  ABC
    Transcript Intelligence
      Finish the central documentation
```

Allowed parent types are configurable. Optional classification inheritance follows this main-parent chain. An explicitly empty value overrides inheritance; resetting that override restores inherited behavior.

Additional named links support cross-cutting relationships and cardinalities, including many-to-many. A project can support multiple goals, and a goal can have several contributing projects. The main-parent tree is therefore only one view of a larger graph. Schema validation, current workspace checks, and cycle prevention constrain edits.

### What remains from the earlier schema

Legacy Space, Area, Project, Goal, Actor, and explicit link tables remain. Old task fields such as project_id, space_id, and area_id coexist with the configurable registry. Import/reconciliation code adopts old and integration-created core records; it does not replace every service with the generic layer.

`structure.observe_core`, `sync_capabilities`, `reconcile_core`, and related paths synchronize the two representations and invalidate stale revisions.

This compatibility boundary deserves close audit. A generic project is not necessarily a row in the legacy projects table. EVAL-008 demonstrates a real failure where the agent inspected only legacy project relationships and missed a distinguishing flexible parent.

### Schema changes versus record edits

Ordinary authorized task/note/record edits execute directly. Redesigning the schema is different: the backend creates an impact preview, binds it to schema and record state, and requires explicit confirmation before applying it. Eri cannot propose and confirm a structural change in the same request. Structural restoration also creates a new preview; it is not a blind rollback.

Archiving definitions preserves old data instead of silently dropping it. Conversions are rejected when operational behavior or existing data would become incompatible.

Sources: [structure models](../apps/api/jarvis/structure_models.py), [schema validation](../apps/api/jarvis/structure_schema.py), [structure service](../apps/api/jarvis/structure.py), [implementation guide](CUSTOM_PLANNER_IMPLEMENTATION.md), [core models](../apps/api/jarvis/models.py).

## 3. Tasks, reminders, calendar blocks, and planner views

A Task holds a title, description/notes, lifecycle status, archive state, dates, priority, estimate, responsibility, optional hierarchy/classification, revision, and integration metadata.

Stable status meanings are backlog, open, in_progress, waiting, deferred, completed, and cancelled. User-facing workflows can rename statuses while mapping them to these meanings. Archive is separate from completion. Assignment expresses responsibility; it does not grant permissions or automatically launch an agent.

Timing deliberately has separate concepts:

- **planned_date**: the day the user intends to work on it.
- **due_date / due_time / due_timezone**: a deadline.
- **Schedule**: a reminder/recurrence definition.
- **Occurrence**: a specific scheduled firing, uniquely keyed by schedule, revision, and instant.
- **PlanningEntry**: a local appointment or reserved task work block.

A deadline is not busy calendar time. Completing a task closes its outstanding alerts. Reopening does not silently replay previously closed alerts.

The task/reminder unification is behavioral, not deletion of the scheduling tables. A reminder is associated with actionable work; recurring routines use templates and concrete occurrence tasks so individual occurrences can be completed without completing the entire routine. The worker scans due schedules, materializes deduplicated occurrences, and delivers notifications through durable jobs.

Tasks has Today, Inbox, Next 7 days, and All views over the same underlying work. Today/week use due or planned dates, including overdue work. Inbox represents unfiled work. Lists, boards, timelines, filters, saved views, inline editing, and bulk operations provide alternate presentations rather than separate task stores. Legacy tab helpers still exist; verify the active configurable planner's filtering separately from old compatibility views.

The calendar combines imported Google events, local appointments/work blocks, and task-related schedule information. Month, week, day, selected-day agenda, detail cards, and editing are available. All-day display dates and Google's exclusive end dates must be translated deliberately. Recurring Google edits distinguish one occurrence from the series.

### Verified scheduling

[planner.py](../apps/api/jarvis/planner.py) implements a bounded deterministic scheduling solver behind the agent:

1. Read current task revisions and explicit durations/constraints.
2. Fetch confirmed availability from selected Google calendars, or use local-only availability when explicitly requested.
3. Merge busy intervals and compute free intervals.
4. Search eligible task orders and place non-preemptive blocks.
5. Return a short-lived proposal token.
6. Recheck task revisions, proposal expiry, and fresh availability before saving all local blocks atomically.

The request is limited to eight tasks within a seven-day window. It is a single-person fixed-duration scheduler with a search bound, not a general multi-resource optimization engine. It does not automatically publish to Google or modify task deadlines. Reusing a proposal is deduplicated.

Sources: [domain.py](../apps/api/jarvis/domain.py), [task tools](../apps/api/jarvis/task_tools.py), [task alerts](../apps/api/jarvis/task_alerts.py), [planning.py](../apps/api/jarvis/planning.py), [planner schema](../apps/api/jarvis/planner_schema.py), [time tools](../apps/api/jarvis/time_tools.py).

## 4. API, commands, consistency, and event propagation

Read endpoints use workspace-scoped service queries. Writes from the browser, backend tools, and direct external clients converge on validated domain commands wherever applicable.

A domain command includes a stable command ID, tool name, and arguments. `domain.execute`:

1. Rechecks authorization.
2. Computes a canonical hash of the requested operation and arguments.
3. Takes an owner/command advisory lock.
4. Returns the saved receipt if the exact command already ran; rejects reuse with different arguments.
5. Validates arguments with a typed Pydantic schema.
6. Takes workspace or memory locks where required.
7. Applies the mutation, records journaled changes, and reconciles operational/generic records.
8. Saves a receipt identifying the command, result, and commit time in the same transaction.

Optimistic revisions reject stale updates. Workspace locks protect graph invariants such as parent cycles. Task batch and frozen-selection edits use revision checks and all-or-nothing semantics.

PostgreSQL Events drive SSE refreshes. The API streams monotonically identified events, supports a reconnect cursor, rechecks membership during streaming, and sends heartbeat/refresh events. The browser reconnects with bounded exponential backoff. Work changes have a dedicated activity refresh signal. The event stream does not itself replace canonical record reads.

This gives strong retry protection for local effects. It does not make an arbitrary external HTTP operation exactly-once; Google/Linear write reconciliation is a separate layer.

Sources: [domain.execute](../apps/api/jarvis/domain.py), [database sessions](../apps/api/jarvis/db.py), [API](../apps/api/jarvis/api.py), [browser event client](../apps/web/src/events.ts).

## 5. Backend model, instructions, and tool exposure

The default available backend profile is **GPT-5.6 Luna with low reasoning**, using the Responses endpoint. The profile allows 8,192 output tokens and uses `store: false`. The adapter preserves native output items and encrypted reasoning content across tool rounds.

Gemini 3.8 Flash remains selectable when configured, through Google's OpenAI-compatible Chat Completions endpoint with low reasoning. A Groq GPT-OSS 120B profile remains in the catalog. The default provider fallback order is Luna, Groq, then Gemini based on available credentials; saved user preferences can override it. The retired OpenAI profile maps to Luna rather than GPT-5.4 mini.

The owner paused Gemini evaluation; that is not the same as removing its product profile. Luna without reasoning is a requested TODO, not currently a separate option.

Background extraction is also distinct from the interactive backend choice: personal-memory extraction, note organization, and organization understanding use the dedicated Luna extraction path. Selecting Gemini for interactive work does not redirect those pipelines.

### Prompt construction

- [personality.py](../apps/api/jarvis/personality.py) is the shared versioned identity/style source. The old environment system prompt is retired for this app.
- [agent_instructions.py](../apps/api/jarvis/agent_instructions.py) supplies global capability and behavior policies, preferred name, time zone, current time, work-window hints, workspace/role, and focused UI context.
- [work_runner.py](../apps/api/jarvis/work_runner.py) adds the current accepted request, previous verified receipts, recent work/dependencies, clarification history, search context, and bounded earlier conversation context.
- Personal memory is retrieved before the initial backend request in a personal workspace. It is not supplied to shared-workspace or bot work.

Earlier conversation and retrieved material are labeled as data, not new instructions. The model is instructed to use real record IDs/revisions, preserve unrequested fields, distinguish queued from completed actions, and never claim an unobserved effect.

### Progressive tools

A small core of common search/task/navigation tools is initially exposed. `tools_load` adds groups such as records, schema, notes/lists, reminders, memory, Google, Linear, planning, activity, routing, settings, and browser editors.

Detailed usage rules live with each tool definition, including descriptions and parameter schemas. They must be visible before that tool call; a tool result cannot retroactively teach the model how to construct its first call. Discovery reduces the initial schema/context burden without an extra routing model.

The backend normalizes provider tool calls, validates availability and argument schemas, dispatches reads or commands, and feeds verified results into subsequent rounds. Sparse-update semantics and strict-provider schema compatibility matter, especially for absent versus null fields.

There is no general shell tool, email-sending tool, or arbitrary web-search tool in this agent. External content, notes, memory, and UI data do not grant new capabilities.

Sources: [model catalog](../apps/api/jarvis/agent_models.py), [Responses adapter](../apps/api/jarvis/responses_adapter.py), [tool catalog](../apps/api/jarvis/tool_catalog.py), [tools](../apps/api/jarvis/tools.py).

## 6. Durable background work and concurrency

### Acceptance and dispatch

A typed chat request or captured voice request becomes a Job plus AgentWork plus Outbox entry transactionally. AgentWork records the account, workspace, device, conversation, optional voice session/bot credential, dependencies, resource reservations, revision, expiry, and encrypted input/checkpoint.

The request captures its backend profile when enqueued. An idempotent request ID prevents accidental repeated submission.

The worker dispatches Outbox rows to DBOS with stable workflow IDs. A crash between DBOS enqueue and marking the outbox submitted can reuse the same workflow identity. Redis is not required.

The worker separates queues:

- General scheduled work: concurrency 1.
- Memory, note filing, search indexing, and organization review: concurrency 2.
- Google: concurrency 1.
- Linear: concurrency 1.
- Legacy intake compatibility: concurrency 4.
- Backend agent work: configurable, default 4 globally and 2 active per account.

Other defaults are 100 queued requests per account, 600 seconds of agent execution allowance, 100 tool calls, and 30 model rounds. There is also a 250,000-byte serialized request-context guard. These are application defaults, not provider promises, and environment settings may differ.

### Checkpointed execution

The tool loop checkpoints provider messages, loaded tools, pending calls, call offset, action/error lists, and revision. Command IDs are derived from the request and tool index. If a process dies after a command commits, replay can obtain the same receipt instead of repeating its effects.

Cancellation stops future work but does not undo committed changes. The runner checks current authority and cancellation/revision around provider waits and before effects. Accepted requests have a 24-hour expiry. Transient input and completed checkpoints have bounded retention, while actual saved records and receipts remain.

A resumed execution uses saved receipts as truth. It does not assume that “the model said done” means a record exists.

### Parallelism and related work

There is no separate lightweight intake classifier anymore. The existing backend model resolves references; deterministic database state decides readiness.

For example:

- “Add Call Alex” and “Also add Buy milk” can run independently.
- “Make that call tomorrow” invokes `work_followup` referencing the first request.
- If the first is active, the new job is checkpointed and parked until its dependency resolves, releasing its worker slot.
- If it already completed, the backend receives the actual outcome and record IDs, rereads the current task, and edits it.

Before mutations, resource reservations coordinate overlapping work. Stable record IDs and device IDs are used when available; creates get provisional new-resource keys, then bind to real saved IDs. Unknown/bulk and integration scopes can reserve a conservative wildcard. Earlier conflicting requests block later ones; a late-discovered overlap with newer reserved work produces a conflict rather than overwriting it.

This deliberately trades some parallelism for safety. The agent still has to identify the intended relationship correctly; dependency checks do not solve natural-language reference resolution by themselves.

### Clarification continuations

A needed clarification should use `work_needs_input`, producing a durable question ID and request revision. A later answer uses `work_answer` before effects, passing the original captured user answer rather than a model-rewritten substitute.

The server validates identity, question, state, ordering, revision, and prior receipts. It marks the old attempt continued and creates a new execution linked to the logical root. The UI aggregates attempts into one activity card. Further questions remain on the newest execution; the root retains history and actual saved changes.

This machinery exists and has targeted tests, but model trajectories still sometimes fail to use it. The audit must inspect both deterministic guards and instruction/tool affordances.

Sources: [agent_work.py](../apps/api/jarvis/agent_work.py), [work_runner.py](../apps/api/jarvis/work_runner.py), [work_coordination.py](../apps/api/jarvis/work_coordination.py), [work_continuation.py](../apps/api/jarvis/work_continuation.py), [continuation guide](CLARIFICATION_CONTINUATIONS.md).

## 7. Voice and GPT-Live

GPT-Live is the enabled voice provider. Realtime's controller and catalog remain in source but the API rejects it through ENABLED_PROVIDERS. No current acceptance plan should silently treat Realtime as the active system.

The browser captures the microphone and negotiates WebRTC. The server creates an OpenAI Live session using its server-side key and the browser SDP, then attaches a control WebSocket. Audio travels over WebRTC; the server handles delegation, transcripts, factual context updates, results, and usage events. The application does not archive raw audio.

Live handles natural conversation. Durable backend work handles record operations, careful reasoning, navigation, and settings. Interrupting speech or closing voice is intended to leave already accepted work running.

### Actual intake behavior and a current bug boundary

Transcript deltas are appended to an encrypted VoiceInbox with event deduplication and a consumed cursor. Explicit Live delegation waits briefly, then claims unconsumed speech. Separately, the worker's `flush_voice` claims quiet inboxes after three seconds without either speaker's input, subject to the worker scan interval.

`claim_voice` groups the new transcript entries and enqueues every nonempty user turn it claims. It is not limited to a validated, fully specified action. This is an important difference between the intended conversational policy and actual acceptance mechanics.

Consequently, “add this to Todo” can be accepted as backend work while Live independently asks what to add. The backend can then ask its own question; a later answer may create/complete another action while leaving the first waiting.

This code path is a plausible mechanism for the September 24 report, not a production-log reconstruction of that exact incident. The planned fix needs to coordinate conversational clarification ownership and capture durability. Merely editing the Live prompt is insufficient if the quiet-inbox flush still delegates incomplete conversation.

### Results and live context

The controller observes AgentWork roots/continuations and sends verified results through Live commentary. Pending questions carry their exact identities as context. It deduplicates announcements and tracks whether sibling work remains busy.

Personal memory is included at session startup. User transcript updates trigger a debounced semantic lookup; a small factual bundle is appended through `session.thinking.append`, with freshness checks, acknowledgments, and bounded pending updates. This can add context during the session, but it cannot guarantee that already-spoken words will be revised or that a retrieval always arrives before speech starts.

The browser polls session status roughly every 400 ms, receives transcript fragments, and uses LiveTranscript presentation to smooth text. This is separate from the general SSE stream.

### Idle, goodbye, wake word, recovery

The user-facing idle timeout is 30 seconds after actual activity/playout, suspended while speaking or backend work is busy. This is distinct from the server's 30-second missing-client-heartbeat cutoff.

The backend exposes `voice_end` only for its bound active voice session. It is supposed to use conversational meaning for “goodbye” or a response to “anything else?”, finish any explicitly requested final action, and end the session without requiring another confirmation. Spoken goodbye reliability remains a physical-device acceptance concern.

Wake recognition uses browser SpeechRecognition/webkitSpeechRecognition, opt-in and foreground-only. It recognizes Eri/Eridani and pronunciation variants, with optional “hey,” and can retain the rest of the utterance. It does not run a downloaded local wake model and can use the browser vendor's recognition service. It is not an Android background hotword service.

Closing voice releases that session's media; enabled wake listening may resume. Process restart loses the active media controller, but durable work can survive. Test these as separate recovery properties.

Sources: [Live controller](../apps/api/jarvis/live_voice.py), [voice endpoints](../apps/api/jarvis/voice.py), [voice intake](../apps/api/jarvis/work_intake.py), [voice controls](../apps/api/jarvis/voice_control.py), [browser voice](../apps/web/src/voice.ts), [idle](../apps/web/src/voice-idle.ts), [wake word](../apps/web/src/wake-word.ts).

## 8. Action cards and Revert

Cards describe verified domain receipts and recorded before/after changes. They are not an interpretation-model approval step. Failed or waiting work can already have saved effects, so status and action history are separate.

The action journal encrypts before/after snapshots in ActionChange and associates them with the command and acting account. Public summaries derive from these records without another summarization model.

The chat timeline anchors a card to its originating user message and keeps it there across completion or continuation. Later conversation appears below it. Navigation-only work should not create a card; unresolved questions/failures can still surface attention. Activity provides broader work history. Seen/reviewed metadata acknowledges an item; it is not permission to execute.

Revert is a compensating command, not unrestricted database rollback:

- Restore supported changed fields only if they still match the recorded post-change state.
- Undo a supported creation by archiving only if unchanged and without blocking linked records.
- Check current schema and backing-record revisions.
- Refuse unsupported inverses or unsettled remote writes.
- Journal the reversal and deduplicate repeated reversal requests.

Local planning events deliberately require their own editor rather than generic automatic undo. A known open defect compares serialized timestamps too strictly, blocking Revert for equivalent instants. Fix semantic comparison without removing real stale-edit checks.

Successful agent actions no longer create notifications. Questions and failed/partial work can. Correctly rooted continuations dismiss superseded question notices.

Sources: [action_history.py](../apps/api/jarvis/action_history.py), [Activity.tsx](../apps/web/src/Activity.tsx), [chat-timeline.ts](../apps/web/src/chat-timeline.ts), [notices.py](../apps/api/jarvis/notices.py).

## 9. Browser and conversational site controls

The responsive SPA provides Tasks/planner, organization, calendar, notes/lists, memory, activity, notifications, and settings. Chat opens from a persistent bottom-right button. Profile controls contain memory/settings/logout. Detail cards support individual inline edits rather than a global edit/save mode.

Boards support mouse/touch movement; timelines display timing. Settings use sections for phone/foldable layouts. Custom browser-history handling restores pages/layers and scroll. Only opaque indices enter history.state; snapshots live in memory, so this is not persistent cross-reload history.

CopilotKit supplies the frontend registry and handler lifecycle. Eridani's existing authenticated transport delivers typed actions; there is no second model runtime, Copilot cloud agent, or arbitrary browser automation engine.

The flow is:

1. Browser publishes bounded page/filter/selection/editor context.
2. Backend requests a typed UI action.
3. A database-backed device bridge delivers it to the current authenticated device.
4. Zod/schema checks and editor guards validate it.
5. React applies it and acknowledges observed state.
6. Backend reports navigation success only after acknowledgment.

Actions cover pages, chat open/close, search, filters, layout, selections, saved views, calendar range, editor drafts, and supported device preferences. Draft patches are not database saves. Record mutations still use server commands.

Bridge context/actions expire. A disconnected page cannot truthfully be navigated. Dirty/in-flight edits can block leaving; merely opening a detail should not create that barrier.

The service worker handles push and record-aware navigation. It intentionally does not cache private API responses or transcript/task bodies. This is not an offline-first app with offline write synchronization.

Sources: [App.tsx](../apps/web/src/App.tsx), [PlannerApp.tsx](../apps/web/src/PlannerApp.tsx), [copilot.tsx](../apps/web/src/copilot.tsx), [site-actions.ts](../apps/web/src/site-actions.ts), [editor-control.tsx](../apps/web/src/editor-control.tsx), [app-history.ts](../apps/web/src/app-history.ts), [ui_control.py](../apps/api/jarvis/ui_control.py), [device_bridge.py](../apps/api/jarvis/device_bridge.py).

## 10. Personal memory and its dream pass

The active memory system is custom code backed by PostgreSQL. It does not use Mem0, Qdrant, or a local embedding model.

### Source-backed learning

Conversation text is stored as Source rows with role, kind, native identity, conversation, deletion state, and extraction version. Eligible user statements queue learning according to history/learning settings. Authored notes and organizational assignments do not automatically become personal memories.

Luna low reasoning with strict structured output proposes at most five atomic durable facts per bounded source. Each includes an exact contiguous evidence quote, confidence, tags, stable subject/attribute fact_key, and optional supersession.

Greetings, temporary requests, hypothetical examples, unsupported inference, credentials, and organization rules are excluded by instructions/validation. Stable facts embedded in requests can count: “remind me to feed my cat Hayes” supports the pet fact, not permanent storage of the reminder as memory.

Candidates are validated and embedded with OpenAI text-embedding-3-small at 512 dimensions. Before committing, the service rechecks source eligibility, learning settings, suppression, revisions, and deduplication under a memory lock.

Memory rows retain source/evidence, attribution, tags, fact key, fingerprint, vector/model, revision, suppression, supersession, and merge links. Exact and high-similarity checks reduce duplicates. Suppressed fingerprints help prevent forgotten facts from immediately reappearing.

**Current coupling:** Memory rows are committed only after embeddings succeed. An embedding failure leaves the source intact and allows retry, but no new canonical fact is yet committed. EVAL-006 records this behavior as an open architectural issue.

### Retrieval

Active facts with undeleted sources are ranked by lexical overlap and exact vector similarity computed in Python. The service rereads canonical revisions after the provider wait. Lexical retrieval remains if cloud embeddings fail.

The backend initially injects up to six relevant source-backed facts as JSON data. Without a query, startup uses recent facts. Live also receives startup context and can receive later bounded updates. Personal memory is excluded from shared-workspace and bot agents.

Correction, suppression, and source deletion must affect retrieval and prompts, not only the Memory page. User/profile statements override older learned assertions.

This is not approximate nearest-neighbor indexing. A future vector index is a capacity improvement, not something already deployed.

### Dream maintenance

The current memory dream is primarily deterministic, not a general LLM reflection system:

- Weekly periods are anchored to Sunday 03:00 in the account time zone; manual review also exists.
- Normalized exact duplicates merge, preferring explicit owner assertions and retaining source/merge lineage.
- Near-identical statements differing in a plausible alphabetic misspelling become review candidates using lexical and phonetic heuristics.
- Ambiguous identities require an owner answer.
- Reviews have revision, defer, and offer-cooldown checks.

The spelling-focused candidate generator misses some numeric contradictions. EVAL-005 documents this. Do not describe it as a completed general contradiction detector.

Sources: [memory_learning.py](../apps/api/jarvis/memory_learning.py), [memory_service.py](../apps/api/jarvis/memory_service.py), [memory_review.py](../apps/api/jarvis/memory_review.py).

## 11. Organization rules and the organization dream

Organization learning has its own FieldUnderstanding, RoutingObservation, RoutingPattern, and RoutingReview records. It does not mix task-filing rules into personal memory.

A changed type/field/relation description triggers asynchronous assessment, producing an operational understanding and at most one necessary question. Manual use remains available. Definition fingerprints prevent old understanding from validating changed semantics.

Manual classifications/corrections supply evidence. Imports and model-generated assignments do not become independent human training examples. Multiple versions of one record do not multiply support.

The weekly review defaults to Monday 03:00 local time, with an editable schedule. It proposes precise title phrases mapped to existing parents/classification values. Unchanged schema/evidence can avoid repeated inference.

Automatic learned-rule activation requires at least three independent supporting examples, 100 held-out human-labeled matches, and 95% precision. Small samples go to review; explicit owner approval can activate a rule immediately. Note rules require review even with strong support. No real-user quality gate has been claimed passed.

Active rules apply through deterministic phrase matching, remove conflicting assignments, preserve explicit choices, and leave uncertainty unassigned. They cannot invent due dates, responsibility, authority, or workspace membership. Work windows are weak context, not an unconditional “8–5 means Work” rule.

Corrections can pause conflicting rules. Forgetting suppresses evidence. Relevant schema changes invalidate understanding; reorganizing existing records requires a separate preview and confirmation.

Review offers are bounded, present one question at a time, and support deferral. They do not autonomously open the microphone. General spontaneous learning questions during ordinary conversation remain future work.

Sources: [routing.py](../apps/api/jarvis/routing.py), [routing schemas](../apps/api/jarvis/routing_schema.py), [structure models](../apps/api/jarvis/structure_models.py), [custom planner guide](CUSTOM_PLANNER_IMPLEMENTATION.md).

## 12. Authored notes and self-organizing lists

Notes preserve authored bodies, tags, links, and revisions. They link tasks/goals/projects/other notes. Append and exact-anchor replacement avoid unnecessary full rewrites. Extracting task proposals is separate from explicitly creating source-linked tasks.

Saved Notes lists are filtered views, not copies/folders. One note can appear in several lists. Lists have required descriptions, tag/custom-field filters, revision, and separate automatic-filing and item-extraction switches. Movies, Books, Shows, Restaurants, and Recipes are optional editable defaults.

A note write can queue organize_note. The worker captures source/list/schema state, calls Luna outside the transaction, then rechecks revisions, definitions, and authority before committing.

“Sam recommended Arrival and After Yang; save these movies” can produce individual saved notes linked to the unchanged source. Evidence must occur exactly in the source, and the proposed title must occur in that evidence. Passing mentions, negative requests, hypothetical content, and quoted instructions should not create saved recommendations.

Thresholds are 0.90 for classification and 0.95 for reuse, with unambiguous supplied identity. Recent-note candidate context is bounded, but global collision checks prevent blindly creating another known title outside that window.

Manual tag edits lock automatic tags, including clearing. Custom classifications fill missing values. Generated notes do not recursively extract more items. Reprocessing should preserve later edits and archive decisions. Failed jobs preserve the source and permit retry.

The newer tables are note_lists, note_organizations, and note_entry_sources. Saved content still uses Note plus the structure registry; no Movie/Book SQL tables were introduced.

“Find possible missing items” shows broader candidates without filing them merely because they were viewed. Current custom-field filters compare direct values, not inherited values. Invalidated schema fields produce a repair state.

Known limitation: real extraction has sometimes omitted one of multiple explicit items. One later passing sample does not fix that omission.

Sources: [notes.py](../apps/api/jarvis/notes.py), [note_lists.py](../apps/api/jarvis/note_lists.py), [list models](../apps/api/jarvis/note_list_models.py), [Notes guide](NOTE_LISTS.md).

## 13. Semantic search and vocabulary learning

Record search is separate from personal-memory retrieval.

The index projects current canonical records, names, descriptions, custom fields, organization context, and source evidence. SearchDocument stores fingerprints and JSON vectors. SearchIndexState tracks workspace generations/jobs.

Mutations invalidate the generation. The worker creates overlapping chunks (1,800 characters, 1,600-character stride), batches embeddings, and skips unchanged fingerprints. Provider work occurs outside the commit transaction. Changed generations invalidate stale output and trigger another pass. The derived index is rebuildable.

Retrieval combines exact name/field evidence, PostgreSQL lexical ranking, vector similarity, and eligible aliases. Membership and canonical state are checked around waits. Missing/stale vectors cannot override current records; text search remains on provider failure.

Results separate structured matches from possible misfiled/unlabeled matches. Strict filters are available when requested. Pagination avoids a newest-1,000 cutoff. Query embeddings use a bounded owner-scoped cache.

SearchSession, SearchAlias, and SearchPreference store account/workspace-specific vocabulary evidence. Field and record targets use different stable keys/fingerprints.

“Pest control company” mapping to ABC is a retrieval alias, not automatically a new personal fact or filing rule. Backend selection alone teaches nothing. Opening/using a presented result or continuing within a bounded interval after actual presentation can supply provisional evidence. Silence, hidden output, and wake-only input remain unknown. Corrections withdraw weak evidence; forgotten mappings retain suppression markers. Bots cannot create implicit human feedback.

Settings supports confirmation, correction, pause, forgetting, and learning preferences. A corresponding organization rule requires its own review.

Source defaults semantic_search_enabled to false; historical rollout docs record production enabled and indexed. No live flag check was performed for this handoff. Current vectors are JSONB with exact Python scoring, not a pgvector index.

Spoken numbers/punctuation remain an open resolution defect: “test test 123” may fail to find “test, test, one, two, three.” Search existing does not mean every legacy tool uses it correctly.

Sources: [search_index.py](../apps/api/jarvis/search_index.py), [search_service.py](../apps/api/jarvis/search_service.py), [search_learning.py](../apps/api/jarvis/search_learning.py), [search models](../apps/api/jarvis/search_models.py), [search guide](SEMANTIC_SEARCH.md).

## 14. Google Calendar and Linear

### Google Calendar

Google sign-in and Calendar consent are separate. Verified OpenID/email identity enables login; read and optional event-write grants enable Calendar. Multiple selected calendars are supported. Refresh credentials are encrypted with the integration key.

Cached events include descriptions, meeting links, organizers, guest information, and attachment links. Calendar selection/access roles and freshness constrain reads, writes, and availability.

Writes use durable jobs and remote reconciliation. Stable creation identities and conditional version checks avoid unsafe blind retries. A queued local receipt does not prove a saved Google change. Recurring edits distinguish occurrence/series, and all-day ranges translate inclusive UI dates to exclusive provider ends.

Local appointments/work blocks can stay Eridani-only or be published. Linked copies support comparison and choosing the local version, Google version, or unlinking after independent edits. Publishing a block does not change a task deadline.

Guest invitations and specialized event operations are outside the supported write surface. Google Calendar events are not a complete Google Tasks integration.

Sources: [Google guide](GOOGLE_SETUP.md), [google_calendar.py](../apps/api/jarvis/google_calendar.py), [google_writes.py](../apps/api/jarvis/google_writes.py), [google_projection.py](../apps/api/jarvis/google_projection.py).

### Linear

Linear uses direct GraphQL calls with an encrypted personal API key, not a Slack-bot relay. Users select teams and may restrict imports to their assigned issues.

Issues become linked tasks. Mapped fields include title, description, status, due date, priority, assignee, project, and in-scope parent relationships. Linear labels remain source metadata; Eridani tags, due times, reminders, blocks, and linked notes have separate semantics.

Polling is approximately every five minutes with an overlapping updated-time cursor and daily reconciliation. All pages must succeed before treating the snapshot as authoritative. Removed/inaccessible/out-of-scope issues leave local records for review.

Writes are durable and reconcile stable create IDs. Updates compare remote updated time against the last shared version before a sparse mutation. This is not atomic compare-and-swap: a remote edit can still race after that check. Conflict-resolution tokens bind reviewed versions and local revisions.

Disconnect retains local records. Full Linear detail metadata, editable source colors, and private nonsynced notes are pending. Comments, cycles, attachments, project creation, and remote issue deletion are not all implemented write tools.

Cloud cutover docs recorded no configured Linear connection at that time; current connection state was not inspected.

Sources: [Linear guide](LINEAR_SETUP.md), [linear_sync.py](../apps/api/jarvis/linear_sync.py), [linear_commands.py](../apps/api/jarvis/linear_commands.py).

## 15. Notifications

Schedules/occurrences and timed deadlines feed Notification records, distinct from work queue/action history. Date-only deadlines remain planner/summary information. Coincident explicit reminders should suppress duplicate deadline alerts.

Quiet hours default to 22:00–08:00 local time. Explicit urgent alerts can bypass them; task priority alone cannot. An optional daily summary defaults to 08:00 and skips empty days. Snoozing changes notification delivery, not deadlines or recurrence definitions.

Delivery attempts have retry, lease, and generation state. Before sending, the worker checks eligibility, completion, archive/rescheduling, quiet hours, and subscriptions. VAPID Web Push reaches the browser service worker and opens workspace-aware targets.

Success stays in chat/Activity. Questions and failures can notify. Shared workspaces and bots have extra restrictions; full per-member shared push routing remains future work.

Synthetic browser tests cannot prove locked-phone delivery, operating-system behavior, real microphone wake/shutdown, or fold-device ergonomics.

Sources: [notices.py](../apps/api/jarvis/notices.py), [worker.py](../apps/api/jarvis/worker.py), [service worker](../apps/web/public/sw.js).

## 16. Authentication, sharing, and external agents

### Human sessions and authorization

The browser session is an opaque random token in a cookie, stored hashed in auth_sessions. It is not a JWT. Sessions carry account, device, authentication method, expiry, and active workspace. Default lifetime is 30 days.

Browser mutations require CSRF and origin checks. HTTPS controls Secure-cookie behavior. Cloud startup disables the old pairing/PIN flow and requires Google login. Accounts require invitations rather than public self-service signup.

Each person has a personal namespace; shared workspaces have owner/editor/viewer membership. Invites are bound to verified email. Assignment/classification never grants access.

The owner_id field often means record namespace, which may be an account or shared workspace; account_id identifies the actor. This distinction must survive every join, delayed job, and endpoint. Authorization is application-enforced; do not assume a Supabase-style RLS layer.

Membership is rechecked on requests, commands, around provider waits, and during SSE/Live activity. Workspace switches rotate context. Revocation stops future access without deleting previously committed shared work.

Personal memory, private integration credentials, and external calendar data stay personal. Shared conversations are transient and excluded from personal-memory learning. Saved views are private to a person within a workspace. Removing the private-chat UI did not remove every historical private/transient field.

Sources: [auth.py](../apps/api/jarvis/auth.py), [access.py](../apps/api/jarvis/access.py), [accounts.py](../apps/api/jarvis/accounts.py), [account guide](ACCOUNTS_AND_SHARING.md).

### External HTTP and MCP

Connected agents receive revocable, expiring, hashed bearer credentials tied to one creating account and workspace. Browser workspace switching does not move a token.

HTTP lives under /api/v1/external. MCP uses Streamable HTTP at /api/v1/external/mcp/. Clients need a configured Bearer header; Google website OAuth is not an MCP authorization server.

Scopes cover tasks, notes, organization, schema, and records. Optional work:run submits natural-language requests to the durable backend without broadening data scopes. Direct API/MCP commands do not require a model.

Schema writes require ownership. Generic record/content access can expose configured note bodies; its scope must be treated accordingly. Existing tokens do not automatically receive newly added scopes.

Bots cannot use this surface to read personal memory/conversations, manipulate browsers/settings, or obtain Google/Linear credentials/tools. Membership, revocation, expiration, and scopes are rechecked for queued work. Documented rate limiting is 120 requests per minute per key.

Direct bot writes use domain commands, revisions, and action receipts identifying the bot. They do not insert an interpretation/review model.

Sources: [bot_access.py](../apps/api/jarvis/bot_access.py), [external_service.py](../apps/api/jarvis/external_service.py), [external_mcp.py](../apps/api/jarvis/external_mcp.py), [external guide](EXTERNAL_AGENTS.md).

## 17. Privacy, accounting, and backup recovery

### Retention and encryption

Tasks, notes, and retained conversation sources are ordinary database fields. Integration credentials, agent input/checkpoints, device context, and action snapshots use application encryption. This is not end-to-end encryption against the server operator.

History and memory-learning settings are separate. history_days=0 means no automatic age-based deletion. History-off/shared work can temporarily retain encrypted input for durable execution until completion/expiry; actual saved actions remain.

Provider store:false settings do not establish a universal provider retention guarantee. Relevant user records can be sent to cloud models to fulfill requests. Browser wake recognition can use a separate vendor service.

Deleting current data does not immediately delete prior backup copies. Database recovery also needs the relevant encryption keys.

### Usage and budgets

Recording and enforcement are separate. September deployment records say cost tracking was on and enforcement off during development. Both source defaults are true, so actual deployed settings must be checked before assuming that remains the case.

BudgetReservation holds estimated/uncertain usage. Usage rows store provider usage and feature attribution. Repeated usage IDs must not double-charge. Incomplete answers may still have known charges; uncertain outcomes retain holds instead of silently being forgiven.

Settings exposes rolling 7-day/30-day estimates and calendar-month totals. A tracking-start marker labels incomplete periods and suppresses misleading projections. Hosting, taxes, credits, and the isolated eval campaign are excluded.

Code estimates Luna/Gemini by tokens and GPT-Live by session seconds. Rate constants are dated configuration, not an invoice. Audit accounting across retries/timeouts as well as happy-path calls.

Sources: [budget.py](../apps/api/jarvis/budget.py), [cost guide](COST_TRACKING.md), [work_crypto.py](../apps/api/jarvis/work_crypto.py).

### Backups

Railway native PITR was historically enabled and tested through a timestamp restore. Independent encrypted exports to Cloudflare R2 have implementation/runbook support, but credentialed activation and real remote restore verification remain pending.

The separate backup service creates encrypted PostgreSQL exports, verifies checksums and uploaded objects before recording success, and retains 30 days of daily plus 12 weeks of weekly copies. Its cloud schedule is daily at 09:00 UTC.

Bucket credentials and backup key belong to the backup service, not API/worker. The application integration-encryption key separately enables recovery of encrypted tokens/work. A restored copy must not start live workers alongside the original.

Sources: [R2 guide](R2_BACKUPS.md), [cloud runbook](CLOUD_MIGRATION.md), [backup image](../Dockerfile.backup).

## 18. CI, evaluation, and Langfuse

### Ordinary checks

GitHub CI has two independent jobs:

1. Backend/migrations: locked dependencies, correctness lint, catalog/harness checks, empty-database upgrade, model/schema agreement, and regression tests.
2. Frontend/browser: locked dependencies, frontend tests, TypeScript/build, Chromium, and isolated desktop/mobile acceptance with synthetic providers and external services off.

It runs on main pushes, PRs to main, and manual requests. CI does not itself deploy or run paid inference. TODO records Railway “wait for CI” enabled and read back for both production services in September. This is separate from GitHub branch protection and was not remotely rechecked here.

The local workflow contains uncommitted harness enhancements. What is described in the working tree may exceed the current GitHub workflow.

Unit, component, browser fixture, real-model, connected-service, physical-device, and disaster-recovery evidence answer different questions. Mock success is not proof of calendar consent, locked-phone push, or spoken goodbye.

### Eval scaffolding

The catalog has 1,001 scenarios across 40 features: generally 25 each plus an additional timing case. A fictional Rowan Chen persona supplies clients/projects, roles, preferences, notes, ambiguous names, and cross-account canaries.

The runner separates contracts, agent, pipeline, integration, browser, and voice types. Current scaffolding expands the full plan to 245 reusable jobs, including 65 paid trials, using a dedicated synthetic PostgreSQL corpus and per-job cloned databases/processes.

Coverage is incomplete:

- 98 cases have complete declared acceptance bindings.
- 224 have component bindings, overlapping the 98.
- 900 still need full individual assertions.
- Three explicitly require physical-device evidence.

That does not mean 1,001 tests passed.

Commands support validate, plan, coverage, run, STOP/resume, report, compare, and manual evidence import. Shared suites are deduplicated without sharing mutable trial state. No schedule was installed.

A transactional SQLite ledger reserves conservative costs before requests and enforces one $10 campaign ceiling across workers, embeddings, judges, retries, and uncertain charges. Built-in judging has a $1 subcap. External semantic grading avoids a Luna judge call; external Codex grader usage is outside app API cost.

Paid campaigns currently use Luna. Gemini comparisons are paused. The runner does not yet call real GPT-Live/audio. Dedicated Google/Linear/R2 probes require separate synthetic resources; they do not prove every integration or full restore scenario.

### Recorded results

September 22's full available 245-job plan ended with 233 passes, eight failures, four blocked. Raw case classification was 92 passed, six failed, 146 partial, 755 blocked, and two component failures. Jobs and cases are different denominators.

September 23 selected 501 cases with prior failures deliberately included. All 153 available jobs finished: 139 passed, five failed, five deferred semantic judgments, four blocked. Supporting suites included 634 backend passes and six browser fixture passes.

Independent GPT-6 Sol review classified the 501 cases as 42 passed, four failed, six needing review, 75 partial, one component failure, and 373 unassessable.

Six raw acceptance claims were downgraded because retrieval/prompt removal, real scheduler ticks, microphone behavior, or active-view/recovery evidence was missing. Missing evidence is not automatically a new product defect.

The half-run recorded $0.055128104 across 104 application API requests with no unresolved reservations. It is not a price prediction for fully implementing every missing scenario or real voice testing.

### Langfuse

The new local exporter is opt-in post-run export of synthetic evidence. The runner owns execution, isolation, assertions, budget, and original reports. Langfuse does not currently trace ordinary production interactions or run scheduled/cloud judges.

It uses OTLP/HTTP JSON plus the Scores API, creates an item per selected case, deduplicates shared job traces, separates automated and external verdicts, and preserves unassessable cases. Bounded business evidence is exported; full auth/database/config snapshots are not.

Receipts and evidence hashes protect resume/readback after uncertain uploads. Provider cost comes from the ledger once; embedding-cache bookkeeping is not billed as a new call. Reconstructed child timelines are not valid live latency measurements.

The corrected import was verified at 501 items, 937 observations, and 256 scores. The first superseded import was explicitly removed. Exporter/harness verification last recorded 82 passes and two optional skips.

Live tracing, worker correlation, production redaction, and trace-based evaluation remain future work.

Sources: [workflow](../.github/workflows/ci.yml), [CI guide](CI.md), [eval README](../evals/app/README.md), [persona](../evals/app/personas/rowan-v1.md), [runner](../scripts/app_eval/runner.py), [bindings](../evals/app/bindings.json), [Langfuse](LANGFUSE.md), [exporter](../scripts/app_eval/langfuse_export.py).

## 19. Known defects and planned work

Recorded findings to reproduce rather than assume fixed:

- EVAL-001: equivalent timestamps produce false Revert conflicts.
- EVAL-002: notes:null clearing calls are rejected; recovery/status behavior varies.
- EVAL-003: missing or misused durable clarification state leaves duplicate/stranded cards.
- EVAL-004: historical project-disambiguation/context-limit behavior remains variable.
- EVAL-005: numeric memory contradictions escape spelling-focused review.
- EVAL-006: embedding failure blocks committing extracted facts, while source survives.
- EVAL-007: multi-item recommendation extraction can omit an explicit item.
- EVAL-008: legacy project lookup misses flexible organization and cannot resolve a clarified task.

The September 24 report adds the need for Live to gather minimum actionable detail before accepting backend work and to avoid stale repeated questions. Speech-number/punctuation resolution remains pending too.

Requested features that are not yet complete:

- Luna without reasoning as a faster selectable backend option.
- Quick lists/quick projects: immediate checklists without mandatory organizational setup, optional grouping/deadline, possible later promotion.
- A broader drag-and-drop visual Structure tree; the current editor is not that full requested redesign.
- Editable source colors, complete Linear details, private nonsynced task notes.
- Cleaner standalone-account invitation/sharing UX.
- General spontaneous learning clarification during ordinary conversation.
- A vector index beyond JSON/Python similarity.
- Live Langfuse tracing and completion of acceptance bindings.
- Credentialed R2 activation/restore verification.
- Native Android and true background hotword behavior.

Still important operational acceptance: physical phone/foldable interaction, real wake/goodbye/reconnect, voice staying active during work, locked-phone push, integration writes/conflicts, second-account access/revocation, restarts/recovery, and backups. The seven-day/50-interaction pilot is planned after hardening and device/operations checks.

Production is still treated as online development. That tolerance does not prove reliability or authorize an auditor to perform destructive tests on personal records.

## 20. Audit priorities and reading order

The most useful audit challenges boundaries between existing mechanisms rather than starting with a framework rewrite.

1. **Authority and scope:** access/auth, commands, bots, and workspace tests. Follow revocation across provider waits, delayed mutations, SSE, and device actions.
2. **Voice clarification ownership:** transcript capture, quiet flush, delegation, work_needs_input, work_answer, and announcements. Retain speech durably without prematurely accepting incomplete intent.
3. **Ordering and recovery:** provisional resources, wildcards, unknown predecessors, continuation order, checkpoints, cancellation, remote pending state, and retries. Test actual interleavings.
4. **Generic/core consistency:** custom parents through Task/Note backing rows, imports, recurrence, search, and undo. Find legacy lookups blind to flexible structure.
5. **Mutation/inverse semantics:** absent/null/empty fields, equivalent times, revisions, links, and external conflicts.
6. **Learning evidence:** source validation, suppression, human/model provenance, repeated examples, activation gates, extraction completeness, stale responses, and deletion.
7. **Search:** spoken-number normalization, inheritance, possible/strict results, stale indexes, scale, revocation, and unearned alias learning.
8. **Latency and accounting:** queue acceptance delay, polling/SSE, workspace locks, exact vector scans, memory waits, schema/context growth, extraction fan-out, retry charges. Non-reasoning Luna is one lever, not proof of the main bottleneck.
9. **Operations:** stale infrastructure declarations, deployed hashes, single-process voice, migration compatibility, key recovery, PITR/R2 health, and provider-side CI gating.
10. **Test integrity:** distinguish supporting/component/acceptance/device evidence and reproduce the findings. Do not present the selected half-suite as an unbiased reliability percentage.

The main product constraint is simplicity for the user despite flexible internals. Consider shared mechanics for Tasks, Notes lists, organization, and Quick lists without forcing schema design before ordinary capture.


## 21. Evaluation outputs for the auditor

Read these saved outputs before drawing conclusions about reliability. They contain actual results, not just test plans. The artifacts are local and ignored by Git, so a GitHub-only checkout will not contain them. Provide Claude the selected campaign directories if it cannot access this workspace. No new evaluation was run for this handoff.

### Start with the independently reviewed half-suite

The most useful first stop is [Sol's readable grading report](../artifacts/app-evals/half-20260923-sol-graded/grader.md). It explains the reproduced failures and where the automated assertions claimed more than their evidence supported.

Use these together:

- [Independent case-by-case grades](../artifacts/app-evals/half-20260923-sol-graded/grader.json): all 501 selected IDs, criterion judgments, reasons, scope disagreements, and evidence paths.
- [Interactive automated report](../artifacts/app-evals/half-20260923-sol-graded/report.html): browse outcomes by feature/type/case.
- [Authoritative automated JSON](../artifacts/app-evals/half-20260923-sol-graded/report.json): raw case outcomes, jobs, regression summaries, cost attribution, and coverage.
- [Frozen selection](../artifacts/app-evals/half-20260923-selection.json): which cases were selected; prior failures were deliberately included.
- [Run manifest](../artifacts/app-evals/half-20260923-sol-graded/manifest.json): model, modes, campaign limits, and source/catalog/harness/corpus fingerprints.
- [Usage ledger summary](../artifacts/app-evals/half-20260923-sol-graded/spending.json): actual estimated usage and uncertain reservations, separate from Codex grader usage.
- [Execution results](../artifacts/app-evals/half-20260923-sol-graded/results/): reusable job outcomes and attempt references.
- [Raw attempts](../artifacts/app-evals/half-20260923-sol-graded/attempts/): model/tool trajectories, saved state, assertion results, logs, and supporting suite/browser evidence.

Raw code outcomes were 44 passed, four failed, four needing review, 75 partial, 373 blocked, and one component failure. Sol's separate assessment was 42 passed, four failed, six needing review, 75 partial, 373 unassessable, and one component failure. Preserve both: external review did not rewrite the automated report.

Prioritize the four acceptance failures: clarifications.01, clarifications.03, memory_dream.19, and time_deadlines.26. Then inspect the memory_capture.24 component failure and the six scope downgrades listed in grader.md. Check task_edit.25 and note_organization.01 against earlier failures even though their final saved outcomes passed this sample.

### Compare with the full available campaign

The [September 22 full report](../artifacts/app-evals/full-20260922-localpg-luna/report.html) covers the entire 1,001-case catalog using all then-available adapters. This is broader selection, not full automated acceptance coverage.

- [Full report JSON](../artifacts/app-evals/full-20260922-localpg-luna/report.json).
- [Manifest and fingerprints](../artifacts/app-evals/full-20260922-localpg-luna/manifest.json).
- [Spending and uncertain charges](../artifacts/app-evals/full-20260922-localpg-luna/spending.json).
- [Job results](../artifacts/app-evals/full-20260922-localpg-luna/results/).
- [Raw attempts](../artifacts/app-evals/full-20260922-localpg-luna/attempts/).
- [JUnit output](../artifacts/app-evals/full-20260922-localpg-luna/junit.xml).

Raw outcomes: 92 passed, six failed, 146 partial, 755 blocked, two component failures. Execution jobs: 233 passed, eight failed, four blocked. Compare failures and traces with the later sample to identify intermittent behavior rather than declaring a defect fixed after one pass.

The original task_capture.05 oracle incorrectly rejected an equivalent offset-aware time. Its [targeted corrected-oracle report](../artifacts/app-evals/due-time-oracle-20260922/report.json) passed. The [separate manifest](../artifacts/app-evals/due-time-oracle-20260922/manifest.json) preserves the changed oracle fingerprint; the original full report was not silently rewritten.

### How to follow a result to the underlying evidence

1. Find the case ID in report.json and, for the half-suite, grader.json.
2. Follow its bound execution/evidence references into results/ and attempts/.
3. An agent/pipeline attempt commonly contains input.json, trace.json, state.json, summary.json, and result.json. Read the actual tool responses and final saved state, not only assistant prose.
4. A supporting regression/browser job may instead contain suite.log, junit.xml, and browser artifacts. It proves only the assertions it exercised.
5. Compare expected/invariant criteria with [bindings.json](../evals/app/bindings.json), [coverage](../evals/app/automation-coverage.json), and [execution protocols](../evals/app/protocols.md).
6. Check manifest fingerprints before comparing runs. A changed oracle/harness is not a pure product improvement.
7. Retain blocked/unassessable cases in the explanation. They are neither demonstrated passes nor demonstrated product failures.

Campaign state and logs use synthetic fixtures, but manifests include local test connection/configuration metadata. They should be inspected within the audit workspace, not blindly published as a public artifact bundle. Do not upload budget/cache databases or unrelated artifacts merely to read outcomes.

### Cross-reference findings and historical experiments

[Findings](../evals/app/FINDINGS.md) is the dated record of EVAL-001 through EVAL-008, including reproduced, intermittent, and component-only evidence.

Older experiments provide tool/model design history:

- [Expert agent results](EXPERT_AGENT_RESULTS.md) and [expert report](evals/expert-agent-report-2026-09-13.html).
- [Tool refinement results](TOOL_REFINEMENT_RESULTS.md) and [tool refinement report](evals/tool-refinement-report-2026-09-13.html).
- [Held-out reliability results](RELIABILITY_HELDOUT_RESULTS.md).

These older paired-model experiments used earlier interfaces/fixtures and must not be treated as measurements of the current full application or evidence that Gemini evaluation is currently enabled.

### Langfuse mirror

The corrected half-suite is also in [the Langfuse project](https://us.cloud.langfuse.com/project/cmtz0kofn00mrad0cqoefpl6h), experiment ID eri-eval-82fb37d8bf2e95a717fcad0b. Claude may need its own authorized project access; the local outputs do not require Langfuse credentials.

The [experiment verification](../artifacts/app-evals/half-20260923-sol-graded/langfuse-experiment-verification.json) and [readback/replay verification](../artifacts/app-evals/half-20260923-sol-graded/langfuse-replay-verification.json) document the uploaded mirror. Local raw evidence remains authoritative and more complete; historical reconstructed spans should not be read as live provider latency.

## Suggested prompt for Claude

> Audit this Eridani repository and handoff against the actual code. Treat the handoff as a map, not proof. Check the source snapshot and distinguish committed code, local changes, historical deployment evidence, and TODO-only work.
>
> Review correctness, consistency, permissions/privacy, concurrency/recovery, voice clarification ownership, custom-schema/core synchronization, memory/routing/search evidence, external sync, UX, costs/performance, and test coverage.
>
> Start with the evaluation outputs in section 21: read Sol's grader.md/grader.json alongside the raw half-suite report, compare the full-run failures and corrected oracle, and follow important cases to tool traces and final database state. Preserve code and reviewer verdicts separately and distinguish failed behavior from missing evidence.
>
> Reproduce findings in isolated synthetic databases where possible. Do not read or transmit secrets, start the retired local production stack, mutate production, contact users, deploy, or run paid evaluations without explicit authorization. A later approved paid campaign should preserve the existing $10 total limit and uncertain-charge accounting.
>
> Return concrete findings with severity, file/line, scenario, expected versus actual behavior, evidence strength, minimal fix, and targeted regression. Separate proven defects, hypotheses, operational unknowns, coverage gaps, and design suggestions. Provide a prioritized repair sequence. Preserve existing work; this is an independent audit to hand back to Codex for implementation.
