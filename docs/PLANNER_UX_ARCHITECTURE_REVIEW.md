# Planner UX and architecture review

Reviewed September 13, 2026 against the implemented application, schema 0011,
the upgrade PRD, and the owner's subsequent decisions. This is a code and
rendered-browser review, not a user study.

## Direction

Keep one planner with several ways to see the same work. Task capture should
require a title, not a classification exercise. Organization is optional and can
be added later. Eri should use the same validated record operations and view
state as the person using the site.

The data model is broadly appropriate. The largest complexity costs are the
number of controls shown simultaneously, ambiguous terminology, and application
code that still combines several page controllers. Adding another agent runtime,
microservice, or canonical task store would increase those costs.

## UX decisions and application

- **One work area:** Today, Inbox and This week now reuse Work's list, board and
  timeline. Search, filters, sort and record actions have the same semantics.
  Inbox means no project, space or area. Today/Week include overdue work.
- **Compact by default:** reduce record padding and decorative headings; keep a
  comfortable density option under Settings → Profile. Mobile touch controls
  retain usable height.
- **Progressive disclosure:** put the full filter set behind Filters & sort.
  Applied filter chips remain visible while the controls are collapsed.
  Project descriptions, success criteria and relationship links expand on demand.
- **Project portfolio:** show status, target, task counts and relationship counts
  in compact rows/cards. Add status boards and start-to-target timelines.
  Changing a project's status does not claim its outcome goal was achieved.
- **Task timing stays honest:** planned dates and deadlines have distinct timeline
  markers. A task's dates do not imply continuous work across that interval.
  Project lifecycle dates can form a bar. Undated and out-of-range records remain
  discoverable, rather than silently disappearing.
- **Settings has sections:** Profile, Voice, Integrations, Privacy and System.
  Voice shortcuts open Voice; integration recovery opens Integrations.
  Device voice/density/wake settings stay distinct from account preferences.
- **Notes use clearer relationships:** Home project controls classification;
  related projects/goals/notes are cross-links. Expand relationship detail when
  needed. A title-only save sends a title-only change, preserving exact authored
  whitespace, quoted evidence and unmentioned links.
- **Conversational controls share the UI:** typed page/filter/layout actions,
  editor reads/patches/saves, bulk selection and device preferences use CopilotKit's
  local registry and the existing authenticated transport. Eri receives actual
  acknowledgement and bounded current screen state.
- **Drafts remain drafts:** reading and filling a form does not save it. Navigation
  refuses an open editor; close refuses unsaved changes. Explicit discard is
  separate. Local save receipts and pending Google writes are distinct outcomes.
- **Keep voice unobtrusive:** opening requested content on mobile closes the
  overlay while leaving active voice running. The transcript and global voice
  indicator remain available. Microphone permission and OAuth consent still use
  the browser's own interaction.

Rendered acceptance covers desktop 1440×1000 and mobile 390×844, with isolated
synthetic records and actual authenticated API/domain commands. It is not proof
of every physical phone, browser microphone, screen reader or network condition.

## Product model

Space is context (Personal, Business). Area is an ongoing responsibility (Health,
Operations). Project is a finite initiative (Website release). Goal is a desired
outcome (Five paying customers). Task is actionable work. A project can support
several goals and a goal can have several supporting projects.

Keep those records distinct, but avoid making their fields mandatory for every
task. A single task can be unclassified, belong to a space/area, or belong to a
project. Subtasks use a task parent; they are not project records. Responsibility
uses an Actor ID; assigning a task to an agent does not launch an execution.

A goal's metric is independent of completed task counts. Completing a project is
not sufficient evidence that the goal was achieved. Archive is retention and
visibility, not completion.

## Dates, alerts and calendars

Task planned_date means intended work on a floating local date. Its due_date,
optional due_time and due_timezone mean deadline. estimate_minutes is effort.

Schedule is the durable alert/recurrence definition, not a competing task list.
Occurrence is a scheduled instance. Notification and Delivery record attention
and transport outcomes. Task completion remains canonical; dismissing a
notification is not completing a task.

PlanningEntry is either an appointment or a task work block. A work block reserves
time and may link to a task. A task can have several blocks. GoogleCalendarEvent
is an external projection, not another local task. Optional Google publication
uses an explicit link and durable write job. The same separation applies to
LinearIssue and its linked Task.

The new planner computes a single-person, non-preemptive schedule for at most
eight tasks in a seven-day window, with release times, deadlines, fixed busy
intervals and within-plan dependencies. It labels proof of earliest finish
separately from merely feasible output and from unknown availability. Commit
rechecks task revisions and current availability and saves all local blocks in
one transaction. A short owner-bound reference resolves an encrypted proposal kept server-side
for fifteen minutes; expired proposals are pruned on subsequent proposal creation.
Committed plans have a separate durable replay receipt, preventing a
retry with a new command ID from duplicating blocks.

This is not a multi-person resource scheduler, and Google free/busy cannot be
locked atomically with a local database transaction. Availability is freshly
checked, but an external calendar can change immediately afterward. Imported
calendar freshness and pending/conflict states therefore remain visible.

## Notes, memory and retrieval

Note is authored material. NoteTaskLink stores evidence and source revision for
extracted commitments. NoteGoalLink, NoteProjectLink and NoteNoteLink connect
records; reverse note links are backlinks. Search does not create tasks.

Memory is a learned assertion with a Source, evidence, attribution, revision and
suppression/correction state. It is not a second notes editor. Explicit correction
can supersede a fact without rewriting the original source.

Eligible saved user text enters a durable extraction job. Luna performs
structured extraction, and OpenAI text-embedding-3-small produces 512-dimensional
embeddings. Memory embeddings and note chunks are currently stored in PostgreSQL.
Hybrid lexical/cosine retrieval rechecks canonical revisions before returning
facts. Relevant facts enter the backend prompt; Live also receives small,
deduplicated context updates during a session. Private/deleted sources and
suppressed facts are excluded.

Weekly deep sleep consolidates clear duplicates and queues uncertain names or
identities for an owner answer. It must not equate similar spelling with identity.
Mem0 and legacy Qdrant are not the active canonical memory path. The existing
Qdrant storage is retained for a possible future vector index; this release does
not delete it or migrate personal memories.

## Application boundaries

- React/TypeScript/Vite owns presentation and ephemeral editor/view state.
- CopilotKit owns the local frontend tool registry. Existing authenticated
  chat/voice transports deliver controls; no second cloud agent runtime is needed.
- GPT-Live handles conversation/audio and delegates record work. Luna or Gemini
  runs the backend tool loop. Common tools are initially available; other
  capability groups load on demand. Tool-specific instructions stay with tools.
- FastAPI/Pydantic validate input; domain modules own authorization, revisions and
  transactions. SQLAlchemy/Alembic manage PostgreSQL and migrations.
- DBOS/worker jobs and the transactional outbox provide durable reminders,
  extraction, sync and retries. Commands and events record accepted outcomes.
- Google and Linear adapters own remote identities, cursors, receipts and
  conflict handling. A queued job is not remote confirmation.
- Authentication is owner/device scoped. Spaces classify records; they do not
  grant shared membership. Tailscale provides the private HTTPS entry.
- Encrypted backups cover canonical records and integration state. Raw audio is
  not an Eridani recording archive. Development cost recording remains disabled.

The number of tables mainly reflects source history, reliable delivery and
external synchronization. Merging these into a generic entity JSON table would
weaken validation and make edits, search and recovery harder to reason about.

## What to simplify next

1. Extract App.tsx's page/view controller and session controller into focused
   modules. The new typed action schema, validation, editor bridge and work-view
   components are a start; App.tsx still carries too much coordination.
2. Generate or parity-check browser contracts against API schemas. Record types
   and tool/UI enums are currently maintained in both Python and TypeScript.
   Keep end-to-end acknowledgement tests because schema generation alone does
   not establish actual browser behavior.
3. Treat Task.project and Task.assignee text as compatibility/display fields;
   project_id and assignee_id remain identities. Remove redundant writes only
   after all legacy clients and external adapters are migrated.
4. Keep Home project distinct from related projects on notes. Do not merge these
   fields while they encode ownership versus cross-link semantics.
5. Centralize generic encryption/token helpers currently located in google_auth.
   Planner proposal payloads and integration tokens share the configured server encryption key;
   that does not imply a Google account is required.
6. Retain the existing scheduler/occurrence tables. Renaming them to sound more
   unified would create migration risk without simplifying the user experience.

## Missing capabilities worth adding deliberately

- Saved personal views, remembered filters/layouts and shareable/deep view links.
- Persistent task dependencies and explicit blocked reasons. Current scheduling
  dependencies apply to a proposal, not a stored task dependency graph.
- Direct goal-to-task links if real one-off goal work routinely has no project.
  Do not create placeholder projects solely to satisfy a relationship.
- Scalable vector indexing and retrieval benchmarks. Current memory retrieval
  scans canonical vectors, and note semantic search has a bounded candidate set.
  Preserve owner filters, deletion/revision checks and lexical fallback on upgrade.
- An authenticated external-bot API/MCP with scoped credentials, stable receipts
  and change cursors. Reuse the domain service; do not allow direct database writes.
- Richer notification priority, bundling and snooze policy without conflating
  notification state with task completion.
- Native Android background/wake behavior, offline capture and conflict handling.
  Foreground browser voice does not establish background microphone reliability.
- Shared workspace membership only with a separate authorization design.
- Metric history/check-ins if outcome trends become useful; a single current
  numeric value is sufficient for today's goal progress.

These are product expansions, not prerequisites for the current planner release.
Home Assistant, finance and research agents in the original PRD remain separate
later domains. Langfuse and the seven-day usage pilot remain deferred as agreed.

## PRD reconciliation

The PRD's modular API/worker/PostgreSQL architecture and authoritative local task
database still fit. Later owner decisions supersede its initial Realtime-first,
local embedding/reranking and read-only Google assumptions: Live is active,
Realtime is paused, embeddings are cloud-generated, and Google/Linear support
explicit writes. Notes and productivity organization have advanced ahead of the
PRD's home-control/finance phases.

No database migration or new service is required for boards, timelines, typed
site controls or verified local planning. They operate on the existing records.

## Validation and remaining tool risks

The full held-out run and the separate short-reference diagnostic are retained in
[RELIABILITY_HELDOUT_RESULTS.md](RELIABILITY_HELDOUT_RESULTS.md). Both models
passed all three repeated scheduling cases after the short-reference change.
That targeted result does not replace the overall held-out scores.

The review also caught full-note retranscription damage, preferred-name versus
actor-ID ambiguity, and model replies claiming an unverified view change.
The next refinements are revision-guarded append/anchored edits and stronger
entity/layout grounding. The tools reject malformed writes and preserve drafts,
but model-authored replacement text still requires these focused improvements.
Completed verification and its limits are in [PLANNER_VALIDATION.md](PLANNER_VALIDATION.md).
