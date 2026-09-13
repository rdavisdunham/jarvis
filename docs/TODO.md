# Eridani / Jarvis — progress and next steps

Updated September 12, 2026. Eridani (Eri) is the assistant's name.
Jarvis remains the repository and infrastructure project name.

## Current batch: mobile Calendar and Google editing — deployed

- [x] Owner confirmed real Google Calendar sync works on mobile.
- [x] Fixed the roughly 30-second scroll jump: background refresh retains the
      agenda instead of clearing it; assistant highlights scroll only once.
- [x] Month, Week and Day views, remembered on this browser. Double-tap a date
      to open Day, with an explicit Open day button and previous/next navigation.
- [x] Google event creation, editing/rescheduling and deletion through the website
      and Eri's shared tools. Choose a writable selected calendar; timed/all-day
      events, title, notes, location, availability and basic repeating events.
      Recurring edits explicitly target one occurrence or the entire series.
- [x] Separate optional Calendar editing consent, fresh Google permissions,
      conditional edits to prevent stale overwrites, durable write receipts and
      retry reconciliation to avoid duplicate events. Recent calendar changes
      shows confirmed, pending, failed or unconfirmed outcomes.
- [x] Automated validation: 183 backend tests plus the additional uncertain-outcome
      regression pass; 74 frontend tests, production build, scoped Ruff and isolated
      browser/migration acceptance pass. Browser checks cover a real 30-second
      refresh on a long mobile agenda, all views and create/edit/delete with a
      lost-response retry. No synthetic writes were made to the owner's calendar.
- [x] Deployed schema 0009 with healthy API/worker/PostgreSQL and the verified HTTPS
      bundle. Google credentials/selections, 2,680 cached events and the historical
      cost ledger are preserved; cost tracking remains off. Encrypted backup
      jarvis-20260913T002615Z.pgdump.enc restored successfully in an isolated database.
- [ ] Owner: Settings → Google → Enable Calendar editing, grant the additional
      Google permission, then try a real personal event. See [GOOGLE_SETUP.md](GOOGLE_SETUP.md).
      Events with guests and special Google event types remain managed in Google.
- [ ] Verify multiple selected calendars and combined availability on the real
      account, including shared calendars. Source selection and simultaneous sync
      are implemented; owner confirmed general sync, not this whole matrix.
- [ ] Next expansion: smarter notifications (priority, snooze controls and bundling),
      then authenticated scoped API/MCP access for external bots.
- [ ] Real-use voice recovery, device/operations checks and the seven-day pilot stay
      after expansion. Internal cost tracking stays disabled during development.

## Previous batch: Google sign-in and Calendar — deployed and connected

- [x] Owner-only Google sign-in linked from an authenticated session, separate
      read-only Calendar consent and PIN pairing retained for recovery.
- [x] Encrypted refresh credentials, selected calendars, durable incremental sync,
      recurring/deleted-event handling, availability and disconnect/unlink controls.
- [x] Deployed schema 0008 with healthy API/worker/PostgreSQL. 157 backend and 71
      frontend tests, browser/migration acceptance and encrypted restore passed.
      Backup: jarvis-20260911T231255Z.pgdump.enc; historical cost ledger preserved.
- [x] Owner created the Google Cloud project/client and saved credentials locally.
      API/worker loaded the configuration; owner subsequently confirmed real sync.
- [ ] Verify Google sign-in from a second device. Pairing remains the recovery route.

## Previous batch: development mode, contextual tasks and notes — deployed

Owner order: keep expanding before the large testing round. Internal cost recording
and budget enforcement are disabled locally during development; the owner monitors
OpenAI Usage. Historical holds remain preserved and no longer block work.

- [x] Development switch in ignored .env.upgrade: JARVIS_COST_TRACKING_ENABLED=false.
      No new cost events/reservations, headroom checks or reconciliation writes while
      disabled. Settings shows the disabled state and links to OpenAI Usage.
- [x] Contextual task lookup from current selection, visible records and recently
      discussed tasks in the same conversation, with fresh revisions and ambiguity.
- [x] Select tasks and bulk-edit status, project, assignee, priority and due date.
      One stale task rejects the whole batch; no partial update. Eri shares the API.
- [x] Notes workspace with editable text, tags, project/task/conversation links,
      archive/restore, keyword search and explicit search by meaning.
- [x] Cloud note embeddings and durable indexing; extraction previews with exact
      source quotes, linked tasks and duplicate prevention. Authored notes do not
      silently become personal memory assertions.
- [x] Eri can read/search/edit notes, propose or create requested to-dos, select
      task groups and open linked records through the CopilotKit bridge.
- [x] Network-response-loss regression: repeat Save reuses the original command
      receipt. Search changes cannot mix old pagination or trigger meaning queries.
- [x] Deployed schema 0007; API/worker/PostgreSQL healthy; live HTTPS bundle checked.
      130 backend and 71 frontend tests, isolated browser/migration checks, and
      post-deploy encrypted backup/restore passed. Historical cost ledger unchanged.
- [x] Next expansion implemented: Google owner sign-in and read-only Calendar;
      real account sync has since been confirmed by the owner.
- [ ] Complete fine-grained CopilotKit field/settings controls, archived-note
      filter parity, broader contextual-reference quality and project board/timeline.
- [ ] After expansion: real-use voice recovery, physical devices/operations,
      then the seven-day owner pilot. Use OpenAI Usage for development costs.
- [ ] Before re-enabling internal accounting, reconcile historical holds and account
      for the unrecorded development period. Disabled tracking cannot backfill usage.

## Previous batch: unified workspace and calendar — deployed

Previous owner order: accounting fixes, workspace/calendar expansion, then acceptance.
The development-mode decision above now defers accounting reconciliation.

- [x] One Work list for tasks and standalone reminders, with linked reminders under
      their task; searchable metadata, status/project/kind filters and retained history.
- [x] Month calendar and selected-day agenda for date-only/timed deadlines, delivered
      reminders and future recurrence previews. Owner timezone, DST-aware schedules,
      completed items, month navigation and mobile layout. Reading never dispatches work.
- [x] Real projects with rename/archive, migrated existing project labels, subtasks
      with cycle checks, assignee labels, work types, tags and priority.
      Assignment metadata does not launch an agent.
- [x] Task details can open/create linked reminders. Reminder details support edits,
      rescheduling, completing one occurrence and cancelling a series.
      A metadata edit preserves a reminder already queued for delivery.
- [x] Shared Eri tools for projects, organization fields, calendar reads and calendar
      navigation/filtering. CopilotKit acknowledges the selected date and filters;
      an open editor blocks navigation.
- [x] Budget lifecycle: stale activity becomes unconfirmed rather than active; explicit
      text-model rejections release unused allowances, unknown outcomes remain held.
      Missing Realtime usage cannot silently count as zero; separate duration-based
      transcription usage is recorded once. Settings lists unconfirmed sessions.
- [x] Audited reconciliation operator command from provider evidence, retaining the
      original ledger and an idempotent adjustment. No historical charge is guessed.
- [ ] Deferred until accounting is re-enabled: historical reconciliation needs billing evidence. The existing
      project API key received HTTP 403 from the organization Costs endpoint.
      The 25 older uncertain Realtime sessions plus one abandoned active session
      retain their unknown headroom. This is held allowance, not confirmed spending.
      No budget limit was raised and no old hold was automatically forgiven.
- [x] Deployment, current encrypted restore, 116 backend tests, 36 frontend tests,
      migration round-trip and isolated desktop/mobile/CopilotKit checks.
- [x] Conversational voice sign-off: Eri may offer a natural closing question.
      The next complete reply follows the question's meaning (no to "anything else",
      yes to "will that be all"). Added requests/continued speech keep voice open.
      Live uses raw captions for context before animated text finishes appearing.
      Both providers retain their microphone cleanup, wake listening and quiet timeout.
- [ ] Verify the new natural sign-off with the owner's microphone on Live and Realtime.
- [ ] Before accounting is re-enabled, reconcile allowances against a Jarvis-only OpenAI cost export for September 11
      UTC. The current command settles individual sessions; project/day aggregates
      need a matching period-level adjustment, not invented per-session costs.
      Preserve original usage, exclude unrelated project activity, and avoid double counting.
- [ ] Project board/timeline, deeper contextual references, Google sign-in/Calendar,
      notification bundling and scoped bot API/MCP remain later. The first notes and
      contextual editing release is implemented in the current batch.

## Current position

The daily-use task/reminder foundation is running in Docker at
https://davispc.tail957c2.ts.net:9443. OpenAI text, Realtime, and selectable GPT-Live voice are connected.
The owner reports that real voice conversations and “Hey, Eri” work very well. This is the working
foundation, not completion of every integration in the upgrade PRD.

Detailed implementation, recovery instructions, evidence, and limitations:
[JARVIS_IMPLEMENTATION.md](JARVIS_IMPLEMENTATION.md).

## PRD comparison and current delivery order — September 11

The [PRD release boundaries](JARVIS_UPGRADE_PRD.md#3-release-boundaries) describe
the original sequence. Current implementation and owner feedback change that order:

- **R0, durable voice/text tasks:** implemented, with successful real-provider checks.
- **R1, daily use:** most features are implemented. Actual locked-phone delivery,
  Windows reboot recovery, a seven-day pilot and the full measured acceptance suite
  remain open. Passing automated tests is not full R1 acceptance.
- **R2, Home Assistant:** not implemented; defer behind the connected personal
  workspace because that is the owner's current priority.
- **R3, memory/context:** source-backed cloud learning, embeddings, retrieval,
  correction/deletion and the first weekly review are active. Retrieval quality,
  clarification follow-through, richer context brokering and indexed search remain.
- **R4, richer personal work:** projects and linked notes are implemented; durable
  research/planner jobs, finance and broader integrations remain expansions. GPT-Live's short task delegation
  does not implement the durable research system.

Later owner decisions supersede the PRD's local GPU/legacy-memory migration
proposals: use cloud inference, retire the active legacy Qdrant bridge, and consider
pgvector after base hardening. Keep the host awake, retain the 15-second quiet
timeout without the old total-session caps, and use the configured fixed pairing
PIN until Google sign-in. GPT-Live, full conversational app control, linked notes,
unified work items and weekly memory review extend the original PRD.

**Execution order:** development cost tracking off; contextual task editing and
linked notes (deployed); Google sign-in and read-only Calendar (implemented, real account setup pending); smarter
notifications; scoped bot access and bounded agent work. Then complete real-use
voice recovery and device/operations acceptance, followed by the seven-day pilot.
Android and broader integrations follow the core workspace. Expand CopilotKit
coverage alongside every feature. Accounting reconciliation and vector indexing
are deferred until base hardening; do not treat missing development costs as zero.

## Done

- [x] GPT-Live (gpt-live-1) alongside Realtime, selectable in Settings.
      GPT-Live delegates tasks to the existing gpt-5.4-mini backend; domain tools,
      owner checks, receipts, and budget accounting are shared.
- [x] Provider-specific voices in Settings, remembered per browser and validated
      on both client and server. Chat stays minimal.
- [x] New chat, a simplified conversation window, and a microphone-reactive
      iridescent voice mode with both speaker transcripts.
- [x] Remove ownership callout blobs and replace raw “unresolved” with a useful,
      plain-language voice status.
- [x] Complete reminders without deleting them. Delivered reminders remain
      pending until completed; completed recurring occurrences preserve the routine.
- [x] Eri can open app pages and open/highlight a requested task or reminder.
      This is the initial typed UI tool, not the full CopilotKit expansion below.
- [x] Foreground wake phrases “Eri,” “Eridani,” and “Hey, Eri” using browser
      recognition. Owner confirmed the existing Hey Eri behavior works.
- [x] A 15-second quiet window after responses, paused/reset by speech and task work.
      Actual audio playout extends the window beyond early transcript arrival.
      Wake-word listening resumes after voice has fully closed.
- [x] Global iridescent voice glow and a mobile/desktop voice dock outside chat.
- [x] CopilotKit frontend tool registry with an authenticated, per-device bridge:
      chat open/close, smart mobile visibility, searches, status/project filters,
      task/reminder forms, current screen context and UI acknowledgements.
- [x] Editable preferred name in Settings; shared by greetings and model prompts.
      No hard-coded owner name remains in the personality file.
- [x] Cloud fact extraction, verbatim source evidence, tags, deduplication,
      corrections, embeddings, semantic retrieval and early context injection.
- [x] Gradual assistant caption display for GPT-Live, keeping exact provider text.
- [x] Realtime close drains cancelled-response usage and releases idle headroom;
      an already-hung-up call no longer creates a spurious budget hold.

- [x] FastAPI domain API, PostgreSQL storage, and a separate durable DBOS worker.
- [x] Responsive web app with task capture, editing, completion, reminders,
      recurring schedules, notification Inbox, and export.
- [x] Persistent pairing sessions: secure HTTP-only cookie, 30-day expiry.
      The owner's chosen fixed PIN is set in ignored local configuration until
      OAuth replaces pairing. Setup preserves it; session tokens remain random.
- [x] OpenAI project key loaded from the ignored .env; live text actions verified.
- [x] Realtime browser voice with server-owned tools, silence gate, interruption,
      and spoken confirmation after a successful save.
- [x] Browser microphone capture waits for provider readiness; brief connection
      drops have a recovery window.
- [x] Live cloud speech captions in the conversation: partial text while speaking,
      reconciled final transcripts, and both speaker labels.
- [x] Clarify the manual voice control as “Respond now”; prevent repeating a
      completed action with it.
- [x] Clear recovered voice errors, ignore late responses from stopped sessions,
      and retry transient voice status failures.
- [x] Reconnect task event streams explicitly after HTTP failures, preserve their
      cursor, and keep task-sync warnings separate from voice status.
- [x] Private conversation mode and source-backed memory capture/correction/deletion.
      The legacy Qdrant bridge was retired at the owner's request on September 11.
- [x] Command retry deduplication, durable reminder recovery, and budget tracking.
- [x] Private Tailscale HTTPS access, Docker startup helper, encrypted backups,
      and a populated backup restore check.
- [x] Owner has tried a real microphone conversation successfully.
- [x] Replace competing Buster/Jarvis personality instructions with Eridani/Eri:
      witty and lightly playful, polished and formal, with assistance as her purpose.
- [x] Share one personality source between text and voice:
      [personality.py](../apps/api/jarvis/personality.py).
      Remove the old JARVIS_SYSTEM_PROMPT entry from environment configuration.
- [x] Remove the old five-minute and twenty-turn voice session caps.
      Keep browser-disconnect cleanup, provider failure handling, explicit End
      voice, and the existing budget controls. The new owner-requested 15-second
      quiet timeout is separate from the removed total-session limits.

## Latest validation

September 11 hardening: **99 backend tests and 33 frontend tests pass** (one
optional paid-provider test skipped). Ruff and the production build pass.
Migration 0005_task_due_time is deployed; API, worker and
PostgreSQL are healthy. Backend regression coverage includes bulk edits beyond the
old caps, partial results, correction/cancellation races, stale memory reads,
review failure/cooldown behavior, task-time/DST rules and budget deferral/resume.
Frontend coverage includes standalone endings and microphone release for both
providers. Mobile task-time editing and the budget display passed browser checks,
alongside the existing memory-review flow. Screenshots were visually inspected.
A current encrypted backup restored successfully with dispatch disabled.

The live snapshot at 19:16 UTC has 1 visible canonical memory; older assertions
remain as history/suppressed records. No test tasks or memories were written to the
live owner in this batch. The physical-phone/reboot checks and seven-day owner
pilot remain open. Use this deployment as the pilot baseline; do not count historic
synthetic acceptance records as owner interactions.


Earlier September 11 baseline: **76 backend tests and 18 frontend tests pass** (one optional
paid-provider test is skipped in the normal suite). Ruff and the production build
pass. Migration `0004_memory_review` preserved a populated memory during an
isolated upgrade/downgrade/upgrade check. Mobile browser review/correction and
Review now passed in a disposable database. Earlier opt-in real OpenAI
extraction/embedding/retrieval checks also passed; those were not repeated here.

Weekly review is deployed and enabled. The first live pass scanned 2 facts,
merged 0, and queued the Hayes/Haze clarification. Next run: September 13 at
03:00 America/Chicago. The active legacy Qdrant service has been removed, and
API/worker/PostgreSQL are healthy. Database checks show the new review queue.

Real-provider GPT-Live: one saved task, both transcripts, audible confirmation,
global mobile glow/dock, quiet timeout, final usage, New chat and switch to Realtime.
Realtime: one task, audio, interruption without losing the save, restart, partial
captions and injected SSE/status recovery. Its latest reservation closed correctly.

Real text-agent browser checks: chat closing, project/status filters, search,
mobile navigation closing the overlay, visible learned memories, editable-name
Settings and no mobile overflow. Both mobile screenshots were inspected.
The live history pass processed 65 eligible user sources and produced **2 embedded,
source-backed memories**. Most prior messages are task requests or conversational
backchannels and correctly do not become memories. Source transcripts contain a
name spelling variation; entity/alias reconciliation remains a quality follow-up.
Synthetic task fixtures were archived; no test memories were added to the live owner.

## Prioritized next work — owner usage feedback

Owner-requested feature expansions are also recorded in the app's “Eridani
roadmap” project. The code-review findings and release gates added here on
September 11 are tracked in this document; they have not been copied into new
app tasks. This order takes precedence over the older thematic lists below.
The first hardening implementation batch is now deployed; remaining acceptance and expansion work stays open.

### Completed: current experience

- [x] Preserve working Hey Eri behavior; add standalone Eri and the 15-second timeout.
- [x] CopilotKit chat-panel controls, mobile behavior and current app context.
- [x] Page/capability map, selected/visible IDs, search/filter state and UI acknowledgements.
- [x] Global voice glow and controls while chat is closed.
- [x] Automatic extraction, embeddings, semantic retrieval and visible memory controls.
- [x] Editable preferred name across text and voice.
- [x] Smoother GPT-Live assistant captions.

### First: harden the daily-use foundation

Owner reordered the work: finish connected-workspace expansions before the large
voice/device/operations acceptance round, then the seven-day owner pilot. Internal
cost tracking and enforcement are off for development. Reconciliation before
re-enabling accounting and vector-index work remain later.
The first three code findings were confirmed by reading the current implementation;
device acceptance items are outstanding checks, not claims of observed failures.

- [x] **Checkpoint the current deployed work before the next coding batch.**
      Committed and pushed the memory/UI/deep-sleep baseline to main as
      b9d1a4cd63216790a473f1b94b7101ee93e205fa before changing app code.
- [x] **Raise the multi-action allowance and explain limits.** Replaced four
      calls with 100 tool calls and 30 planning rounds per request, configurable
      through JARVIS_MAX_TOOL_CALLS_PER_REQUEST and
      JARVIS_MAX_MODEL_ROUNDS_PER_REQUEST. Text/GPT-Live have one final summary
      call after the planning allowance. Reads count toward the tool allowance.
      Task reads paginate; limits preserve receipts and expose partial results.
      Larger allowances trade more possible model work for fewer interrupted
      batches; explicit cancellation still applies. Budget checks are disabled in local development.
- [x] **End voice by speaking.** Standalone "goodbye," "thank you" or "that's all"
      should end voice, release the microphone and return to wake listening for
      "Hey, Eri" or "Eri." Exclude quoted phrases and thanks followed by another
      request. Preserve committed actions and make pending work status clear.
- [x] **Optional task due times.** Add a time and timezone with the due date in
      storage, tools, forms and displays. Preserve date-only tasks; a due time
      does not create a reminder. Validate timezone and date/time edits.
- [x] **Complete Eri's access to existing actions.** Expose the existing
      memory.correct, memory.forget, notification.read, notification.snooze and
      notification.dismiss commands through the shared model-tool registry.
      Preserve current owner checks, revisions, idempotency and truthful results.
      Cover text, Realtime and GPT-Live delegation through the same contract.
      PRD: T01 and the shared tool gateway.
- [x] **Prevent stale memory context.** Retrieval currently snapshots facts before
      awaiting the cloud query embedding. Recheck canonical source visibility,
      suppression and memory revisions after that await and before returning or
      injecting context. Test correction/deletion during a delayed lookup.
      PRD: M05 and section 11.4.
- [x] **Weekly-review failure visibility and clarification cooldown.** Failed
      reviews now show their state, last successful run and Retry review.
      Preparing context no longer consumes the daily offer. The cooldown starts
      when an assistant question contains the candidate spellings; transcript
      production is not proof that audio was heard. Broader semantic follow-through
      remains part of the later memory-quality work.
- [x] **Cost visibility and optional-work controls.** Show active reservations,
      uncertain holds, calendar-month spend and a pace-based projection separately.
      Add the 80% warning, 95% optional-memory deferral and a version label on new
      usage estimates. Deferred memory jobs resume durably when room returns.
      Cancellation during initial retrieval releases the unused chat allowance;
      cancelling an in-flight paid request keeps its outcome marked uncertain.
- [ ] **Deferred: reconcile holds before re-enabling accounting.** September 11,
      19:16 UTC snapshot: $2.551407 recorded estimated spend and $124.267111 held
      for unconfirmed sessions. Do not treat held amounts as confirmed charges.
      Retain them until provider evidence supports settlement. Complete the
      provider-price/transcription accounting audit. The seven-day comparison moves to the post-expansion pilot.
- [ ] **Finish voice and site-control recovery acceptance.** Exercise both voice
      providers through long conversations/provider session endings, interruptions,
      a correction arriving during a pending action, close/reopen, network loss and
      API restart. Confirm receipts restore committed outcomes without replaying
      actions. Check standalone Eri, 15-second mic handoff, current selection,
      browser action acknowledgments and recovery on the actual mobile device.
      Extend existing record editors/settings coverage after these paths are sound;
      search, filters, navigation and chat control already exist.
- [x] **Restore the current encrypted backup in isolation.** Verified
      jarvis-20260911T202655Z.pgdump.enc at migration 0006_workspace_accounting,
      including tasks, projects, schedules, notifications and memory reviews.
      No worker or reminder dispatch ran against the restored database.
- [ ] **Prove operational recovery and reminder delivery.** Test actual Windows
      reboot startup and locked-phone Web Push, including permission denial and
      opening the notification. Verify an encrypted backup of the current schema
      restores in an isolated stack with dispatch disabled. Keep an encrypted
      recovery copy and its separately stored recovery key off this PC.


The hardening exit is evidence of reliable saved actions, current memory context,
visible failures, recoverable state and understood spending. Automated checks,
real-device results and the pilot each establish different parts of that evidence.
The narrow GPT-Live context enhancement below follows correctness fixes and can
be exercised during the pilot.

- [x] **Weekly deep sleep, first version.** Durable DBOS review each Sunday at
      3 AM in the owner's home timezone, with one catch-up after downtime.
      Merge exact duplicate assertions while preserving source links. Queue
      similar-spelling candidates for clarification; never select a name just
      because it sounds similar. Memory shows review questions, a corrected-fact
      editor, “These are different,” “Ask next week,” and Review now.
      Eri receives at most one optional review offer per day and can resolve an
      explicit answer with the shared command tools. Settings can disable it.
- [x] **Retire the legacy Qdrant bridge.** Remove its active search adapter, UI,
      configuration, Compose service and backup mount. Old volumes/backups remain
      offline recovery artifacts; the running app does not read them.
- [x] **Quiet memory updates during GPT-Live.** Feed relevant retrieved facts
      through session.thinking.append while speech continues, with debouncing,
      small payloads, freshness/deletion checks and cancellation on close.
      Implemented with a one-second debounce, bounded factual payloads, revision
      checks and append acknowledgments. New protocol/lifecycle tests pass; the
      owner's real-provider pilot still needs to assess the resulting conversation.
      An append acknowledgment is estimated context delivery, not a guarantee
      that the end of the current response or the next words use the entire update.

### Next: a connected personal workspace

The following batch covers linked/searchable notes, contextual task editing,
projects/work types/subtasks/assignees, Google Calendar availability, smarter
notifications, and an authenticated API/MCP connection for other bots.

Build these as small successive batches in the listed order, carrying the same
actions into the web interface and Eri's tools with each release.

- [x] **Unified work-item experience and organization, first release.** Work combines
      tasks and standalone reminders while preserving schedules, delivery records
      and completion history. Real projects, work types, subtasks, assignee labels,
      tags, priorities and status filters are implemented. Tasks sort by priority
      and deadline; reminders sort by scheduled time. Existing project labels are
      migrated. Scheduled agent execution and additional sort modes remain later.
- [x] **Contextual task editing, first release.** Resolve selected, visible and
      recently discussed task references within a conversation, re-read current
      records, preserve ambiguity and revision checks, and edit groups atomically.
      The web selection/bulk editor and shared Eri tools use the same commands.
- [ ] **Full field/settings control and richer references.** Extend CopilotKit to
      every editor field, setting and multi-step workflow; measure reference quality
      with real conversations. Archived-note filter controls still need UI parity.
- [x] **Calendar and day agenda.** Tasks and reminder occurrences share a month
      calendar, selected-day agenda, filters and conversational controls without
      requiring Google sync.
- [ ] **Project board and timeline views.** Follow the calendar/workspace release.
- [x] **Linked notes, first release.** Editable notes linked to tasks/projects and conversations,
      tags, Eri-readable source content, and to-do extraction with provenance and
      duplicate prevention. Cloud embeddings and hybrid semantic/keyword search
      use PostgreSQL JSONB chunks; keyword lookup remains available during provider
      failures. Authored notes stay separate from personal memory assertions.
- [ ] Measure note retrieval and extraction quality on real owner content before
      the later pgvector migration. Add richer source lifecycle and note-to-memory
      review only if the owner wants authored notes to supply learned facts.
- [x] **Google sign-in and read-only Calendar implementation.** Persistent owner
      sign-in, separate Calendar consent, current availability, deadline conflicts,
      source links/freshness and private-host-compatible polling are deployed.
      Supported recurring exceptions, deletions and invalid cursors are covered.
      Pairing remains a recovery route.
- [ ] Configure the real Google client and complete first account-link/sync checks.
      Add Google event edits only after read/sync behavior is proven.
- [ ] **Smarter notifications.** Flexible/natural-language snoozing, bundling or
      digests, priority levels and preferences. Extend the existing simple snooze
      while preserving delivery/completion history and recurring behavior.
- [ ] **External bot API/MCP.** Personal-Linear-style task tracking for other AI
      agents. Reuse the current REST/domain command path, add revocable scoped
      bot access, documented contracts, idempotency, actor/audit attribution and
      an MCP adapter. Share the capability contracts with Android.
      Start after the work-item contracts stabilize, so outside bots use the same
      rules and recorded results as the website and Eri.
- [ ] **Bounded agent execution, then native Android.** Build durable jobs with
      allowed actions, budgets, progress, cancellation and stored results before
      an agent assignment can launch work. Reuse established app/tool contracts
      for Android; native background wake and notifications need their own device
      acceptance. Home Assistant, finance and broader research/capture integrations
      follow the core workspace unless the owner changes priorities.

### After expansion: verify, then pilot

- [ ] Complete real-use voice recovery and actual-device/operations checks listed above
      after the unified workspace and calendar expansion.
- [ ] **Run the PRD owner pilot and record measured results.** Seven days with at
      least 50 successful task/reminder interactions; report voice failures,
      unwanted speech, interruptions, delivery results and per-category costs.
      Use section 16's fixed command/failure suites for duplicate effects,
      truthful confirmation, latency and source attribution. Mark unmeasured
      targets explicitly; cloud memory supersedes the old GPU-specific setup.

### Accepted design boundaries

- [x] **Notes and memory:** share source links, tags and retrieval infrastructure,
      but retain authored notes as editable source records and personal memory as
      derived assertions. Define revisions, back-links, deletion and reindexing.
      Owner accepted this direction. The authored-notes model is not yet implemented.
- [x] **Tasks, reminders and agent work:** owner accepted one work-item interface with
      owner/agent assignment, optional schedules and notifications. Keep due dates,
      recurring schedules, execution attempts, delivery and completion as distinct
      concepts underneath. Preserve standalone reminders and existing history.
      Do not delete the reminder system merely to reduce the number of UI sections.
- [ ] **Agent assignees:** define execution scope, allowed actions, running/failed/
      completed status and result references before assignments launch automation.

### Memory audit before this batch — September 11 (historical)

Read-only inspection of the live deployment found history and memory learning
enabled, a fresh worker heartbeat, **0 visible canonical memory entries**, and
**66 completed extraction jobs**. The saved sources included **65 user messages**
and **59 assistant messages**. None of the saved user messages matched the
old extractor's exact opening phrases, so the jobs completed without creating
memories. The worker is running; extraction is a deliberately narrow initial
implementation, not the full R3 memory system.

Explicit “remember this” tool capture, manual memory capture, correction and
forgetting exist. The old automatic capture checked only prefixes such as
“remember that,” “I prefer,” and “I live in.” Normal conversational wording may
never match. Private conversations are excluded from automatic history/learning.
At that audit, canonical search was lexical/full-text only. Legacy memory is
a separate read-only search bridge and is only queried when search text is given;
its records do not populate an empty canonical-memory view.

This batch replaced that implementation and backfilled eligible saved history.
The live system now has cloud extraction and embeddings, semantic search and
pre-response context. Saved counts and processing counts are reported separately.

## Requested next: full app control and Android

- [ ] Expand beyond today's open/highlight tool into full conversational site control:
      searches, filters, sort order, views, record editors, forms, and settings.
      CopilotKit's frontend tools and app context are integrated now. Extend richer
      editors/settings, multi-step recovery and Android contracts; adding another
      agent runtime remains explicitly acceptable to the owner.
- [x] Use typed, authorized app actions with completion acknowledgments and
      visible outcomes. Carry current page/filter/selection state into agent context.
      Further multi-step/editor/settings coverage remains in the expansion task.
- [ ] Design those capabilities as shared contracts for the web app and a future
      native Android app, so both can be operated by talking to Eri.
- [x] Move voice model and voice selectors out of chat into Settings; match the
      voice list to the selected model and validate it again on the server.

## Next: daily-use polish and account access

- [x] Make the owner's preferred/display name editable in Settings and use it
      consistently in greetings, text, GPT-Live, and Realtime. The current saved
      name stays until edited; prompts no longer hard-code Davin.
      Also saved in the app as “Make my preferred name editable in Settings.”

- [x] Implement Google OAuth/OpenID Connect restricted to the account linked by
      the owner, with persistent sessions and device revocation.
- [ ] Complete the real Google client setup and account-link acceptance before
      closing the app task “Add Google account sign-in to Jarvis.”
- [ ] Verify the added standalone “Eri” phrase on the owner's actual device,
      including mic handoff after the timeout. Existing Hey Eri is owner-confirmed;
      parser, idle behavior and automated voice checks pass.
- [ ] Compare GPT-Live and Realtime across long conversations, interruptions,
      late task corrections, and different voices. Basic real-provider checks pass.
- [ ] Test locked-phone Web Push on the actual phone, including permission,
      delivery, opening the notice, and recovery after a connection gap.
- [ ] Validate long conversations on the actual device, varied speech, pauses,
      and interruption timing. Provider/network interruptions remain possible
      with no total-duration/turn cap and the new 15-second quiet timeout.
- [ ] After expansion and voice/device/operations acceptance, run the seven-day usage/cost and voice-quality pilot; record issues and
      tune the gate and personality from actual use.
- [ ] Test startup after an actual Windows reboot.
- [ ] Keep a recovery copy of encrypted backups and the recovery key off this PC.

The CopilotKit and Android requests are also saved under the “Eridani roadmap”
project in the app:
“Expand Eri's site control with CopilotKit” and
“Plan Eridani's Android app with shared voice controls.”

## Later PRD work

- [ ] **PostgreSQL vector index after base hardening.** Use pgvector with a
      cosine HNSW index for canonical memories and later note chunks. Migrate the
      existing 512-dimensional cloud embeddings, retain lexical fallback, enforce
      owner/source/deletion filters, and benchmark recall and latency before rollout.
      PostgreSQL remains canonical storage; Qdrant is not the planned future store.
      Reference: [pgvector](https://github.com/pgvector/pgvector).
- [ ] Extend deep sleep beyond exact duplicates and spelling candidates:
      cross-sentence entity/alias reconciliation, linked-fact corrections,
      contradiction review, measured semantic candidate retrieval/reranking and
      reliable clarification follow-through. The first version does not silently
      merge merely similar facts or run an LLM over the whole memory collection.
- [ ] Connect Home Assistant with explicit device mappings.
- [x] Google Calendar and availability code is deployed with outbound polling;
      real credentials and owner consent remain open in the current batch above.
- [ ] Add bounded, durable research/planner jobs with progress, cancellation,
      saved artifacts, sourced web research, and replay-safe child delegation.
      GPT-Live's short conversational task delegation is not this R4 job system.
- [ ] Add read-only finance imports with idempotent ingestion and reconciled
      aggregates, then broader integrations.
- [ ] Expand memory context brokering, source/relevance feedback, and measured
      retrieval quality. The current capture/search layer is only part of R3.

## Git checkpoint

Before the caption/recovery changes, the working foundation was committed as
`8d41337de35ef5ad6dc01d5f01f2f9daade1c7c1` and pushed to `origin/main`.
The remote previously used `master`; `main` was created at the owner's request.
Before this batch, caption/recovery, GPT-Live, daily-use polish and the roadmap
were committed and pushed to `origin/main` as
`404c230ddd9b7dae48fab4c78d6a5c2e87ea3707`. Remote equality was verified before
new edits. The memory/UI/idle/name and weekly-review work was subsequently
committed and pushed to main as b9d1a4cd63216790a473f1b94b7101ee93e205fa before
this hardening batch. The following source commit records the hardening work.

## Maintenance rule

Record completed work and concrete next steps here after substantive changes.
Keep detailed operational evidence in JARVIS_IMPLEMENTATION.md; the PRD and
upgrade decisions remain the design references. Environment files hold secrets
and deployment settings, not an alternate personality.
