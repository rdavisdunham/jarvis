# Eridani / Jarvis — progress and next steps

Updated September 11, 2026. Eridani (Eri) is the assistant's name.
Jarvis remains the repository and infrastructure project name.

## Current position

The daily-use task/reminder foundation is running in Docker at
https://davispc.tail957c2.ts.net:9443. OpenAI text, Realtime, and selectable GPT-Live voice are connected.
Davin reports that a real voice conversation works very well. This is the working
foundation, not completion of every integration in the upgrade PRD.

Detailed implementation, recovery instructions, evidence, and limitations:
[JARVIS_IMPLEMENTATION.md](JARVIS_IMPLEMENTATION.md).

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
- [x] Initial foreground “Hey, Eri” implementation using browser recognition.
      **Owner reports activation is not working.** Repair and actual-device
      acceptance are priority work below; do not treat this feature as verified.
      No local Whisper model or always-on closed-app microphone.

- [x] FastAPI domain API, PostgreSQL storage, and a separate durable DBOS worker.
- [x] Responsive web app with task capture, editing, completion, reminders,
      recurring schedules, notification Inbox, and export.
- [x] Persistent pairing sessions: secure HTTP-only cookie, 30-day expiry.
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
- [x] Private conversation mode, source-backed memory capture/correction/deletion,
      and read-only access to an isolated copy of legacy memory.
- [x] Command retry deduplication, durable reminder recovery, and budget tracking.
- [x] Private Tailscale HTTPS access, Docker startup helper, encrypted backups,
      and a populated backup restore check.
- [x] Owner has tried a real microphone conversation successfully.
- [x] Replace competing Buster/Jarvis personality instructions with Eridani/Eri:
      witty and lightly playful, polished and formal, with assistance as her purpose.
- [x] Share one personality source between text and voice:
      [personality.py](../apps/api/jarvis/personality.py).
      Remove the old JARVIS_SYSTEM_PROMPT entry from environment configuration.
- [x] Remove application voice session duration, speech-turn, and idle-silence caps.
      Keep browser-disconnect cleanup, provider failure handling, explicit End
      voice, and the existing budget controls.

## Latest validation

September 11: 51 backend tests and 14 frontend tests passed; Ruff and the
production build passed. The additive migration ran successfully in Docker.

Real GPT-Live WebRTC acceptance created exactly one task, streamed user and Eri
captions, received audible completion after the backend save, collected final
usage on close, started a New chat, and switched back to Realtime.
The Realtime regression created one task, received audio, interrupted without
losing the save, restarted successfully, and recovered from injected SSE and
status-poll failures. Partial speech captions arrived before end-of-speech.
Browser checks also verified model/voice compatibility, agent-directed reminder
navigation/highlighting, retained completion history, and the minimal mobile
chat without horizontal overflow. Synthetic fixtures were cleaned up.

Actual microphone/wake-word and long-conversation quality checks are still
listed below; browser automation does not establish those physical behaviors.

## Prioritized next work — owner usage feedback

All items here are recorded in the app's “Eridani roadmap” project as well.
“Now” means the recommended next implementation batch, not already implemented.

### Now: repair and make the current experience trustworthy

- [ ] **Repair “Hey, Eri.”** Reproduce the reported activation failure on the
      actual browser/device. Check permissions, recognition service, wake phrase
      plus request, accurate armed/paused/error status, and microphone handoff.
- [ ] **Chat-panel tools and app context.** Let CopilotKit tools explicitly open
      and close chat. Add context-sensitive behavior when showing a requested
      record, especially on mobile, with user overrides and easy reopening.
      Keep voice running independently of panel visibility.
- [ ] **Eri understands the site.** Maintain a capability/page map and provide
      current page, selected/visible record IDs, search, filters, panel state,
      viewport and permitted controls. Acknowledge tool completion from the UI.
      Extend this alongside the panel tools before broad multi-step UI actions.
- [ ] **Global voice glow.** Render the reactive iridescent animation in the app
      shell whenever voice is active, beyond the chat window. Keep accessible
      End voice, reopen-chat and caption controls available.
- [ ] **Useful, visible memory.** Replace exact prefix matching with source-backed
      extraction from natural conversation. Show saved/updated memory feedback,
      learning status and extraction failures. Preserve private mode and
      correction/forgetting semantics. See the live audit below.
- [ ] **Editable preferred name.** Include the previously requested Settings
      change in this batch; use the chosen name consistently across modes.

### Next: a connected personal workspace

- [ ] **Natural-language task editing.** Resolve “move that thing I mentioned
      earlier to Friday” from conversation and selected-record context; retain
      revisions, clear date semantics and clarification for ambiguous targets.
- [ ] **Task organization.** Richer projects, work types, subtasks, owner/agent
      assignees, tags, priority, filtering and sorting. The current project field
      is a simple label, not the full project experience.
- [ ] **Linked notes.** Editable notes linked to tasks/projects and conversations,
      tags, Eri-readable source content, and to-do extraction with provenance and
      duplicate prevention. Add cloud embeddings and semantic search.
- [ ] **Google Calendar.** Calendar availability, conflicts, timezone-aware
      planning and links to tasks/notes; explicit event edits after read/sync
      semantics are established. Coordinate Google account connection with OAuth.
- [ ] **Smarter notifications.** Flexible/natural-language snoozing, bundling or
      digests, priority levels and preferences. Extend the existing simple snooze
      while preserving delivery/completion history and recurring behavior.
- [ ] **External bot API/MCP.** Personal-Linear-style task tracking for other AI
      agents. Reuse the current REST/domain command path, add revocable scoped
      bot access, documented contracts, idempotency, actor/audit attribution and
      an MCP adapter. Share the capability contracts with Android.

### Design decisions before changing the data model

- [ ] **Notes and memory:** share source links, tags and retrieval infrastructure,
      but retain authored notes as editable source records and personal memory as
      derived assertions. Define revisions, back-links, deletion and reindexing.
      This is the recommended direction, not an implemented migration.
- [ ] **Tasks, reminders and agent work:** evaluate one work-item interface with
      owner/agent assignment, optional schedules and notifications. Keep due dates,
      recurring schedules, execution attempts, delivery and completion as distinct
      concepts underneath. Preserve standalone reminders and existing history.
      Do not delete the reminder system merely to reduce the number of UI sections.
- [ ] **Agent assignees:** define execution scope, allowed actions, running/failed/
      completed status and result references before assignments launch automation.

### Memory audit — September 11

Read-only inspection of the live deployment found history and memory learning
enabled, a fresh worker heartbeat, **0 visible canonical memory entries**, and
**66 completed extraction jobs**. The saved sources included **65 user messages**
and **59 assistant messages**. None of the saved user messages matched the
extractor's current exact opening phrases, so the jobs completed without creating
memories. The worker is running; extraction is a deliberately narrow initial
implementation, not the full R3 memory system.

Explicit “remember this” tool capture, manual memory capture, correction and
forgetting exist. Automatic capture currently checks only prefixes such as
“remember that,” “I prefer,” and “I live in.” Normal conversational wording may
never match. Private conversations are excluded from automatic history/learning.
Canonical search is lexical/full-text, not semantic embeddings. Legacy memory is
a separate read-only search bridge and is only queried when search text is given;
its records do not populate an empty canonical-memory view.

Record whether backfilling eligible saved history is wanted when implementing
better extraction; do not confuse a successful processing job with a saved memory.

## Requested next: full app control and Android

- [ ] Expand beyond today's open/highlight tool into full conversational site control:
      searches, filters, sort order, views, record editors, forms, and settings.
      Evaluate and integrate CopilotKit's frontend tools and shared app state;
      adding its agent runtime is explicitly acceptable to Davin.
- [ ] Use typed, authorized app actions with completion acknowledgments and
      visible outcomes. Carry current page/filter/selection state into agent context.
      Test multi-step interactions, permission-sensitive settings, and recovery.
- [ ] Design those capabilities as shared contracts for the web app and a future
      native Android app, so both can be operated by talking to Eri.
- [x] Move voice model and voice selectors out of chat into Settings; match the
      voice list to the selected model and validate it again on the server.

## Next: daily-use polish and account access

- [ ] Make the owner's preferred/display name editable in Settings and use it
      consistently in greetings, text, GPT-Live, and Realtime. Eri currently
      calls the owner Davin; remove hard-coded name references from prompts.
      Also saved in the app as “Make my preferred name editable in Settings.”

- [ ] Add Google OAuth/OpenID Connect sign-in, restricted to Davin's authorized
      Google account. Preserve persistent sessions and device revocation.
      Also saved in Eridani's task list as “Add Google account sign-in to Jarvis.”
- [ ] Repair the reported “Hey, Eri” failure, then verify actual-device
      microphone handoff, hidden-tab pause and permission-denial recovery.
      The wake-phrase/parser logic is tested; recognition is not validated.
- [ ] Compare GPT-Live and Realtime across long conversations, interruptions,
      late task corrections, and different voices. Basic real-provider checks pass.
- [ ] Test locked-phone Web Push on the actual phone, including permission,
      delivery, opening the notice, and recovery after a connection gap.
- [ ] Validate long conversations on the actual device, varied speech, pauses,
      and interruption timing. Provider/network interruptions remain possible
      even though application session caps are removed.
- [ ] Run the seven-day usage/cost and voice-quality pilot; record issues and
      tune the gate and personality from actual use.
- [ ] Test startup after an actual Windows reboot.
- [ ] Keep a recovery copy of encrypted backups and the recovery key off this PC.

The CopilotKit and Android requests are also saved under the “Eridani roadmap”
project in the app:
“Expand Eri's site control with CopilotKit” and
“Plan Eridani's Android app with shared voice controls.”

## Later PRD work

- [ ] Improve canonical and legacy memory retrieval: semantic search,
      embeddings/reranking, correction/deletion overlays, and quality benchmarks.
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
The caption/recovery, GPT-Live, and daily-use polish work follows that checkpoint and is currently uncommitted.

## Maintenance rule

Record completed work and concrete next steps here after substantive changes.
Keep detailed operational evidence in JARVIS_IMPLEMENTATION.md; the PRD and
upgrade decisions remain the design references. Environment files hold secrets
and deployment settings, not an alternate personality.
