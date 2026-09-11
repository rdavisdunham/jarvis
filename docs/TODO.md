# Eridani / Jarvis — progress and next steps

Updated September 11, 2026. Eridani (Eri) is the assistant's name.
Jarvis remains the repository and infrastructure project name.

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
- **R4, richer personal work:** projects, durable research/planner jobs, finance
  and broader integrations remain expansions. GPT-Live's short task delegation
  does not implement the durable research system.

Later owner decisions supersede the PRD's local GPU/legacy-memory migration
proposals: use cloud inference, retire the active legacy Qdrant bridge, and consider
pgvector after base hardening. Keep the host awake, retain the 15-second quiet
timeout without the old total-session caps, and use the configured fixed pairing
PIN until Google sign-in. GPT-Live, full conversational app control, linked notes,
unified work items and weekly memory review extend the original PRD.

**Execution order:** checkpoint the current work; close existing capability and
correctness gaps; validate recovery, devices and costs; build unified work items
and contextual editing; add linked notes; connect Google Calendar and smarter
notifications; expose bot access and bounded agent work; then build Android and
other integrations. Expand CopilotKit coverage with each feature, so voice and
the website do not drift apart. The checklists below define the work.

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

Complete the following in order before larger feature or vector-index work.
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
      batches; budget checks and explicit cancellation still apply.
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
- [ ] **Reconcile older holds and finish accounting validation.** September 11,
      19:16 UTC snapshot: $2.551407 recorded estimated spend and $124.267111 held
      for unconfirmed sessions. Do not treat held amounts as confirmed charges.
      Retain them until provider evidence supports settlement. Complete the
      provider-price/transcription accounting audit and seven-day comparison.
- [ ] **Finish voice and site-control recovery acceptance.** Exercise both voice
      providers through long conversations/provider session endings, interruptions,
      a correction arriving during a pending action, close/reopen, network loss and
      API restart. Confirm receipts restore committed outcomes without replaying
      actions. Check standalone Eri, 15-second mic handoff, current selection,
      browser action acknowledgments and recovery on the actual mobile device.
      Extend existing record editors/settings coverage after these paths are sound;
      search, filters, navigation and chat control already exist.
- [x] **Restore the current encrypted backup in isolation.** Verified
      jarvis-20260911T191619Z.pgdump.enc at migration 0005_task_due_time, including
      tasks, schedules, notifications, memory reviews and the new time columns.
      No worker or reminder dispatch ran against the restored database.
- [ ] **Prove operational recovery and reminder delivery.** Test actual Windows
      reboot startup and locked-phone Web Push, including permission denial and
      opening the notification. Verify an encrypted backup of the current schema
      restores in an isolated stack with dispatch disabled. Keep an encrypted
      recovery copy and its separately stored recovery key off this PC.
- [ ] **Run the PRD owner pilot and record measured results.** Seven days with at
      least 50 successful task/reminder interactions; report voice failures,
      unwanted speech, interruptions, delivery results and per-category costs.
      Use section 16's fixed command/failure suites for duplicate effects,
      truthful confirmation, latency and source attribution. Mark unmeasured
      targets explicitly; cloud memory supersedes the old GPU-specific setup.

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

- [ ] **Unified work-item experience and organization.** Bring tasks, standalone
      reminders and scheduled agent work into one coherent interface while retaining
      separate due dates, schedules, delivery records and completion underneath.
      Add real projects, work types, subtasks, owner/agent assignees, tags, priority,
      richer states and sorting. The current project field is a simple label.
      Ship assignment metadata separately from automatic agent execution.
- [ ] **Contextual task editing and full control of those views.** Resolve “move
      that thing I mentioned earlier to Friday” from conversation and selected-record
      context; retain revisions, date semantics and clarification for ambiguous
      targets. Extend CopilotKit to the new fields, editors, settings and multi-step
      workflows. A feature is complete when direct UI use and Eri can both operate it.
- [ ] **Calendar, project, timeline and day views.** Build these during the task
      tracker expansion, sharing filters, selection and conversational controls.
      A task calendar uses local task dates/times without requiring Google sync.
      These views are intentionally deferred from the hardening batch.
- [ ] **Linked notes.** Editable notes linked to tasks/projects and conversations,
      tags, Eri-readable source content, and to-do extraction with provenance and
      duplicate prevention. Add cloud embeddings and semantic search.
      Keep the authored note distinct from derived memory assertions. After
      hardening, evaluate the pgvector migration alongside the growing note corpus;
      retain source/deletion checks and measure retrieval before switching indexes.
- [ ] **Google sign-in and read-only Calendar first.** Add persistent owner sign-in
      and separately authorized calendar access. Show availability, conflicts,
      timezone-aware planning, source links and sync freshness. Use private-host-
      compatible outbound polling; handle recurring exceptions, deletions and invalid
      sync cursors. Add explicit event edits only after read/sync behavior is proven.
      The temporary pairing PIN remains usable until this batch.
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

### Accepted design boundaries — implementation is next

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

- [ ] Add Google OAuth/OpenID Connect sign-in, restricted to Davin's authorized
      Google account. Preserve persistent sessions and device revocation.
      Also saved in Eridani's task list as “Add Google account sign-in to Jarvis.”
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
- [ ] Run the seven-day usage/cost and voice-quality pilot; record issues and
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
- [ ] Google Calendar and availability integration is now in the prioritized
      connected-workspace batch above; retain private-host-compatible sync.
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
