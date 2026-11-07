# Eridani functionality and evaluation map

Verified against the current repository on September 19, 2026. The machine-readable
[surface inventory](../evals/app/surface-inventory.json) lists 68 mutation commands,
52 read tools, five loop controls and 112 HTTP/WebSocket routes. It is an ownership
inventory, not a claim that every operation has already passed an independent test.

## How the app works

The React frontend talks to a Python FastAPI service. PostgreSQL holds the canonical
records, revisions, source evidence, queued work, receipts and membership. A separate
durable worker runs accepted work and background jobs. This is not a frontend-only app.

User-defined types, fields, relationships, statuses and record hierarchies provide the
organizational layer. Types with work capabilities have canonical task backing rows;
content types have note backing rows. A generic project need not be a legacy Project
row. Renaming a label changes organization, not permissions or the scheduling engine.
Spaces and clients classify work; workspace membership grants access.

Tasks separate lifecycle status, archive state, planned date, deadline, reminder and
reserved calendar time. Assignee is responsibility; it neither grants access nor
automatically launches an agent. Notes preserve authored content and can link records,
produce explicit to-dos, and appear in saved filtered lists without being copied.

GPT-Live is the active voice frontend. Requests typed in the app's chat run on their own
model, Settings → Task agent → Typed chat (GPT-6.1 Terra at medium reasoning by default:
slower and smarter, at Standard processing). Voice-delegated, bot and other background work
uses the selected task agent, Settings → Task agent → Background work (Luna by default, or
Gemini; the code also retains a Groq profile). The model is fixed when a request is accepted:
revisions and clarification answers continue on the model the request started with, so a
typed answer to a question asked during voice stays on the task agent. Accepted requests survive
voice interruption. Independent requests can proceed concurrently; references and
clarification answers carry dependencies and lineage. Commands validate permissions,
arguments and revisions before transactionally saving effects and action receipts.
Cards report verified changes; Revert checks for intervening edits and links.

## Learning and search are separate systems

Personal-memory learning extracts source-backed facts using Luna and embeds eligible
facts with OpenAI text-embedding-3-small. Retrieval combines meaning and lexical evidence,
checks current visibility/revision, and injects relevant facts as data into agent context.
Voice sessions open with a short, background-maintained "About you" summary instead (§33).
The memory dream pass merges safe duplicates and queues ambiguous spellings/facts for
review. It does not silently pick between Miso and Mizo. It also queues semantic
paraphrases (embedding similarity of the current model, newest 400 facts) and keyed
name/timezone facts that disagree with the profile. Each question carries its provenance
(method, similarity or profile value, and every fact's source and evidence quote). Only the
owner's answer merges or replaces a fact; a profile question never edits the profile.
Facts the agent saved through a tool are `agent_capture`, not owner statements, so the
owner's own wording wins between exact duplicates.

Organization learning uses human task/note classifications, field descriptions and
corrections in separate tables. Its weekly dream pass proposes patterns, validates
targets against the current schema and holds ambiguous/insufficient patterns for review.
Automatic assignments do not count as independent human evidence. Work hours are a weak
hint. The review and routing state report the evidence split (`eligible`, `training`,
`held_out`); "no training examples" with eligible examples means they all landed in the
stable held-out split, not that capture is missing. Rules may classify records; they may not invent deadlines, change responsibility
or grant access. Note-organization rules require explicit approval.

Search can retrieve by meaning, identify likely organizational targets, then show exact
structured matches alongside possible misfiled/unlabeled matches. Search vocabulary
learning remains distinct from personal facts. A passive lack of correction is weaker
evidence than explicit confirmation; aliases must not silently become powerful rules.

Self-organizing notes use saved filters (tags and current custom values), optional
confident filing and separate extraction of clearly intended saved items. Source
notes and evidence remain intact. Passing mentions, negative requests and quoted
instructions must not create recommendations or become personal memories.

## Test layers and boundaries

[Rowan Chen](../evals/app/personas/rowan-v1.md) provides a fictional automation-delivery
job, three clients, four projects, personal interests and deliberately ambiguous data.
The [eval README](../evals/app/README.md) explains the persistent corpus and per-case clones.

Each feature below has 25 acceptance cases; date/time has 26 after the new non-UTC
receipt finding. The complete catalog has 1,001 scenarios. Tests cover successful,
ambiguous, invalid, stale, concurrent, permission and recovery behavior as applicable.
The existing regression files are supporting evidence; they do not automatically pass
the new language or device scenarios.

Realtime remains disabled and is tested as a disabled boundary. Cost controls exist
with cost recording restored September 19 and budget enforcement off in development.
R2 backup machinery exists; credentialed restore/drill checks remain an operations task. No real OAuth, push,
microphone, R2 or PITR success is inferred from fake transports. Android-native features
and other unimplemented roadmap ideas are outside this current-functionality baseline.

## Feature-by-feature coverage

### 1. Capture tasks

Create tasks with precise titles, notes, ownership and initial classifications.

- Acceptance set: [25 cases](../evals/app/features/task_capture.json).
- Implementation: [domain.py](../apps/api/jarvis/domain.py), [task_tools.py](../apps/api/jarvis/task_tools.py).
- Existing regression evidence: [test_domain.py](../tests/test_domain.py), [test_tool_refinement.py](../tests/test_tool_refinement.py).

### 2. Edit and resolve tasks

Resolve references and change only requested fields, preserving concurrent edits.

- Acceptance set: [25 cases](../evals/app/features/task_edit.json).
- Implementation: [task_tools.py](../apps/api/jarvis/task_tools.py), [task_context.py](../apps/api/jarvis/task_context.py).
- Existing regression evidence: [test_tool_refinement.py](../tests/test_tool_refinement.py), [test_batch1.py](../tests/test_batch1.py).

### 3. Task status and bulk changes

Backlog, open, active, waiting, completion, reopening, archive and atomic bulk operations.

Completed and canceled tasks hide immediately from ordinary Tasks and Organization
views, with a **Show completed and canceled** filter. Both Today surfaces keep a daily
checklist below unfinished work. The worker files eligible tasks after 24 elapsed hours;
Archived preserves outcomes and history. Restore grants another 24 hours; Reopen
returns work to active. Settings → Organization exposes a personal/workspace archive
policy toggle, restricted to owners in shared workspaces. Organizational containers and
tasks with unfinished descendants or unsettled execution are protected. See
[finished tasks and automatic archiving](TASK_ARCHIVING.md) for lifecycle and validation details.

- Acceptance set: [25 cases](../evals/app/features/task_lifecycle.json).
- Implementation: [domain.py](../apps/api/jarvis/domain.py), [workspace.py](../apps/api/jarvis/workspace.py).
- Existing regression evidence: [test_backlog.py](../tests/test_backlog.py), [test_workspace.py](../tests/test_workspace.py), [test_recurrence_edges.py](../tests/test_recurrence_edges.py).

### 4. Dates, deadlines and time zones

Separate planned days, deadlines, local times and notification schedules.

- Acceptance set: [26 cases](../evals/app/features/time_deadlines.json).
- Implementation: [time_tools.py](../apps/api/jarvis/time_tools.py), [task_alerts.py](../apps/api/jarvis/task_alerts.py).
- Existing regression evidence: [test_recurrence_edges.py](../tests/test_recurrence_edges.py), [test_tool_refinement.py](../tests/test_tool_refinement.py).

### 5. Reminders and recurring routines

Durable alerts, occurrence tasks, schedule changes and recurrence boundaries.

- Acceptance set: [25 cases](../evals/app/features/routines.json).
- Implementation: [domain.py](../apps/api/jarvis/domain.py), [worker.py](../apps/api/jarvis/worker.py).
- Existing regression evidence: [test_domain.py](../tests/test_domain.py), [test_daily_polish.py](../tests/test_daily_polish.py), [test_recurrence_edges.py](../tests/test_recurrence_edges.py).

### 6. Goals, projects and responsibility

Spaces, areas, clients, goals, projects, assignees, subtasks and linked records.

- Acceptance set: [25 cases](../evals/app/features/organization.json).
- Implementation: [organization.py](../apps/api/jarvis/organization.py), [productivity.py](../apps/api/jarvis/productivity.py).
- Existing regression evidence: [test_productivity.py](../tests/test_productivity.py), [test_planner_release.py](../tests/test_planner_release.py).

### 7. Custom types and fields

Versioned structure proposals, required descriptions, workflows, relationships and restoration.

- Acceptance set: [25 cases](../evals/app/features/custom_schema.json).
- Implementation: [structure.py](../apps/api/jarvis/structure.py), [structure_schema.py](../apps/api/jarvis/structure_schema.py).
- Existing regression evidence: [test_structure.py](../tests/test_structure.py).

### 8. Custom records and inheritance

Typed records, parent trees, custom values, inherited classifications and graph links.

- Acceptance set: [25 cases](../evals/app/features/custom_records.json).
- Implementation: [structure.py](../apps/api/jarvis/structure.py), [structure_routes.py](../apps/api/jarvis/structure_routes.py).
- Existing regression evidence: [test_structure.py](../tests/test_structure.py), [test_accounts.py](../tests/test_accounts.py).

### 9. Views, boards and timelines

Task tabs, saved filters, selection, drag/drop status and calendar projections.

- Acceptance set: [25 cases](../evals/app/features/planner_views.json).
- Implementation: [saved_views.py](../apps/api/jarvis/saved_views.py), [workspace.py](../apps/api/jarvis/workspace.py), [calendar_details.py](../apps/api/jarvis/calendar_details.py).
- Existing regression evidence: [test_workspace.py](../tests/test_workspace.py), [test_planner_release.py](../tests/test_planner_release.py), [test_backlog.py](../tests/test_backlog.py).

### 10. Scheduling work blocks

Constraint-aware proposals, atomic commits, calendar publication and conflict resolution.

- Acceptance set: [25 cases](../evals/app/features/planning.json).
- Implementation: [planner.py](../apps/api/jarvis/planner.py), [planning.py](../apps/api/jarvis/planning.py).
- Existing regression evidence: [test_planning.py](../tests/test_planning.py), [test_reliability_eval_cases.py](../tests/test_reliability_eval_cases.py).

### 11. Notes and linked context

Authored content, exact edits, linked tasks/projects/goals, sources and archive behavior.

- Acceptance set: [25 cases](../evals/app/features/notes.json).
- Implementation: [notes.py](../apps/api/jarvis/notes.py), [productivity.py](../apps/api/jarvis/productivity.py).
- Existing regression evidence: [test_notes_context.py](../tests/test_notes_context.py), [test_productivity.py](../tests/test_productivity.py).

### 12. Extract tasks from notes

Evidence-backed task suggestions with explicit acceptance and duplicate prevention.

- Acceptance set: [25 cases](../evals/app/features/note_tasks.json).
- Implementation: [notes.py](../apps/api/jarvis/notes.py).
- Existing regression evidence: [test_notes_context.py](../tests/test_notes_context.py), [test_expert_eval_cases.py](../tests/test_expert_eval_cases.py).

### 13. Saved Notes lists

Descriptions, deterministic tags/custom-field filters, multiple memberships and list lifecycle.

- Acceptance set: [25 cases](../evals/app/features/note_lists.json).
- Implementation: [note_lists.py](../apps/api/jarvis/note_lists.py).
- Existing regression evidence: [test_note_lists.py](../tests/test_note_lists.py).

### 14. Automatic note organization

Conservative filing, source-linked recommendations, correction locks and recovery.

- Acceptance set: [25 cases](../evals/app/features/note_organization.json).
- Implementation: [note_lists.py](../apps/api/jarvis/note_lists.py).
- Existing regression evidence: [test_note_lists.py](../tests/test_note_lists.py).

### 15. Semantic and lexical search

Hybrid retrieval across tasks, notes, record bodies, fields, aliases and misfiled candidates.

- Acceptance set: [25 cases](../evals/app/features/search.json).
- Implementation: [search_service.py](../apps/api/jarvis/search_service.py), [search_index.py](../apps/api/jarvis/search_index.py).
- Existing regression evidence: [test_semantic_search.py](../tests/test_semantic_search.py), [test_accounts.py](../tests/test_accounts.py).

### 16. Search vocabulary learning

Separate provisional vocabulary, confirmation, correction, pause, forget and review.

- Acceptance set: [25 cases](../evals/app/features/search_aliases.json).
- Implementation: [search_learning.py](../apps/api/jarvis/search_learning.py).
- Existing regression evidence: [test_semantic_search.py](../tests/test_semantic_search.py).

### 17. Automatic memory capture

Extract durable facts from eligible user sources with provenance and embeddings.

- Acceptance set: [25 cases](../evals/app/features/memory_capture.json).
- Implementation: [memory_learning.py](../apps/api/jarvis/memory_learning.py).
- Existing regression evidence: [test_memory_learning.py](../tests/test_memory_learning.py), [test_memory_cloud.py](../tests/test_memory_cloud.py).

### 18. Memory use and correction

Capture, retrieve, inject, correct, forget and preserve source/privacy constraints.

- Acceptance set: [25 cases](../evals/app/features/memory_management.json).
- Implementation: [memory_service.py](../apps/api/jarvis/memory_service.py), [memory_learning.py](../apps/api/jarvis/memory_learning.py).
- Existing regression evidence: [test_memory_cloud.py](../tests/test_memory_cloud.py), [test_privacy.py](../tests/test_privacy.py), [test_memory_learning.py](../tests/test_memory_learning.py).

### 19. Dream sequence: memories

Weekly duplicate cleanup, similar-name questions, explicit resolution and scheduling.

- Acceptance set: [25 cases](../evals/app/features/memory_dream.json).
- Implementation: [memory_review.py](../apps/api/jarvis/memory_review.py).
- Existing regression evidence: [test_memory_review.py](../tests/test_memory_review.py).

### 20. Organization rules

Explicit rule creation, explanations, preview/apply, user overrides and field understanding.

- Acceptance set: [25 cases](../evals/app/features/routing_rules.json).
- Implementation: [routing.py](../apps/api/jarvis/routing.py), [structure.py](../apps/api/jarvis/structure.py).
- Existing regression evidence: [test_routing.py](../tests/test_routing.py).

### 21. Dream sequence: organization

Review human categorization evidence, propose task/note rules and offer bounded interviews.

- Acceptance set: [25 cases](../evals/app/features/routing_dream.json).
- Implementation: [routing.py](../apps/api/jarvis/routing.py).
- Existing regression evidence: [test_routing.py](../tests/test_routing.py), [test_note_lists.py](../tests/test_note_lists.py).

### 22. Smarter notifications

Quiet hours, priority levels, bundles and digests, deadline alerts, summaries, snooze in words ("tomorrow morning", "after lunch"), dismiss and completion, with delivered, read, snoozed and completed kept distinct.

- Acceptance set: [25 cases](../evals/app/features/notifications.json).
- Implementation: [notices.py](../apps/api/jarvis/notices.py), [snooze_words.py](../apps/api/jarvis/snooze_words.py), [task_alerts.py](../apps/api/jarvis/task_alerts.py), [worker.py](../apps/api/jarvis/worker.py).
- Existing regression evidence: [test_smart_notifications.py](../tests/test_smart_notifications.py), [test_notification_bundles.py](../tests/test_notification_bundles.py), [test_snooze_words.py](../tests/test_snooze_words.py), [test_daily_polish.py](../tests/test_daily_polish.py).

### 23. Google Calendar read and sync

Consent, selected calendars, event details, coverage, availability and timezone handling.

- Acceptance set: [25 cases](../evals/app/features/google_sync.json).
- Implementation: [google_calendar.py](../apps/api/jarvis/google_calendar.py), [google_auth.py](../apps/api/jarvis/google_auth.py), [google_projection.py](../apps/api/jarvis/google_projection.py).
- Existing regression evidence: [test_google_calendar.py](../tests/test_google_calendar.py).

### 24. Google Calendar writes

Durable idempotent event creation/edit/deletion and honest remote status.

- Acceptance set: [25 cases](../evals/app/features/google_writes.json).
- Implementation: [google_writes.py](../apps/api/jarvis/google_writes.py), [google_routes.py](../apps/api/jarvis/google_routes.py).
- Existing regression evidence: [test_google_writes.py](../tests/test_google_writes.py), [test_eval_integrations.py](../tests/test_eval_integrations.py).

### 25. Linear read and sync

Credentials, selected teams/projects, imported issue mapping, pagination and reconciliation.
Optional structure onboarding maps Linear teams/projects onto Homes with a dry-run preview
([Linear setup](LINEAR_SETUP.md#optional-file-linear-teams-and-projects-into-homes)).

- Acceptance set: [25 cases](../evals/app/features/linear_sync.json).
- Implementation: [linear_sync.py](../apps/api/jarvis/linear_sync.py), [linear_client.py](../apps/api/jarvis/linear_client.py), [linear_structure.py](../apps/api/jarvis/linear_structure.py).
- Existing regression evidence: [test_linear.py](../tests/test_linear.py), [test_linear_structure.py](../tests/test_linear_structure.py).

### 26. Linear writes

Create/update/publish issues, pending status, revisions and deliberate conflict resolution.

- Acceptance set: [25 cases](../evals/app/features/linear_writes.json).
- Implementation: [linear_commands.py](../apps/api/jarvis/linear_commands.py), [linear_sync.py](../apps/api/jarvis/linear_sync.py).
- Existing regression evidence: [test_linear.py](../tests/test_linear.py).

### 27. Durable parallel agent work

Accept requests independently of voice, serialize conflicting entities, retry and cancel safely.

- Acceptance set: [25 cases](../evals/app/features/queue.json).
- Implementation: [work_intake.py](../apps/api/jarvis/work_intake.py), [work_runner.py](../apps/api/jarvis/work_runner.py), [work_coordination.py](../apps/api/jarvis/work_coordination.py).
- Existing regression evidence: [test_agent_work.py](../tests/test_agent_work.py), [test_action_limits.py](../tests/test_action_limits.py).

### 28. Clarification continuation

Stable question identity, one continued work item, late answers and concurrent replies.

- Acceptance set: [25 cases](../evals/app/features/clarifications.json).
- Implementation: [work_continuation.py](../apps/api/jarvis/work_continuation.py).
- Existing regression evidence: [test_work_continuation.py](../tests/test_work_continuation.py).

### 29. Action cards, history and revert

Actual saved changes, compact anchored cards, field-level reversion and history cleanup. While a request
runs, its card shows the backend's latest step, such as "Looking that up…" or "Saving Call Alex…", in
place of "Working on your request…"; the note goes away when the request settles. A clarification
continuation shares its original card; there are no nested child cards.

- Acceptance set: [25 cases](../evals/app/features/receipts.json).
- Implementation: [action_history.py](../apps/api/jarvis/action_history.py), [agent_work.py](../apps/api/jarvis/agent_work.py).
- Existing regression evidence: [test_agent_work.py](../tests/test_agent_work.py), [test_work_continuation.py](../tests/test_work_continuation.py), [test_external_agents.py](../tests/test_external_agents.py).

### 30. Conversation and backend agents

Production tool discovery, context, personality, provider continuity and honest reporting.

- Acceptance set: [25 cases](../evals/app/features/chat_agent.json).
- Implementation: [work_runner.py](../apps/api/jarvis/work_runner.py), [agent_models.py](../apps/api/jarvis/agent_models.py), [responses_adapter.py](../apps/api/jarvis/responses_adapter.py).
- Existing regression evidence: [test_agent_models.py](../tests/test_agent_models.py), [test_responses_agent.py](../tests/test_responses_agent.py), [test_tool_refinement.py](../tests/test_tool_refinement.py).

### 31. Conversational site controls

Acknowledged navigation, filters, selection, editor control and draft protection. Named-item
navigation treats the kind the owner says ("task", "meeting", "note", "client") as a hint: one match
of any kind opens its details without correcting the wording; several matches get a question.

- Acceptance set: [26 cases](../evals/app/features/site_controls.json); `site_controls.26` (kind mismatch) has a component contract.
- Implementation: [ui_control.py](../apps/api/jarvis/ui_control.py), [ui_contracts.py](../apps/api/jarvis/ui_contracts.py), [tools.py](../apps/api/jarvis/tools.py), [navigation.py](../apps/api/jarvis/navigation.py).
- Existing regression evidence: [test_ui_control.py](../tests/test_ui_control.py), [test_batch1.py](../tests/test_batch1.py), [test_tool_refinement.py](../tests/test_tool_refinement.py), [test_navigation.py](../tests/test_navigation.py).

### 32. Browser navigation and settings

Back/Forward, persistent chat affordance, fold layouts, profile menu and preferences.

- Acceptance set: [25 cases](../evals/app/features/browser_settings.json).
- Implementation: [api.py](../apps/api/jarvis/api.py), [saved_views.py](../apps/api/jarvis/saved_views.py).
- Existing regression evidence: [test_planner_release.py](../tests/test_planner_release.py), [test_api.py](../tests/test_api.py).

### 33. GPT-Live sessions

Live creation, captions, delegation, interruption, context updates and session recovery.

Instructions. Live's prompt (`live_instructions` in
[agent_instructions.py](../apps/api/jarvis/agent_instructions.py)) follows OpenAI's
[GPT-Live prompting guide](https://developers.openai.com/api/docs/guides/live-prompting) and holds only
what the voice model acts on: Dani's identity and shared personality
([personality.py](../apps/api/jarvis/personality.py)), how Dani sounds, the Backchannel, Interruption
and Delegation policies, and a small Profile DATA line (preferred name, timezone, current time). Dani's pronouns follow the chosen voice
(she/her or he/him, from OpenAI's published presentation for each voice; Marin when none is chosen),
in voice and in typed chat, which reads the device's voice from the screen context. The
Delegation policy lists the backend's capabilities and says when to "Generate <|SpawnThinking|>" and
when not to, the wording the guide gives for gpt-live-1. While work runs, Dani acknowledges briefly and
gives a short update as each verified result arrives. The screen, focused record IDs, scheduling
windows and the closing controller's timers stay with the backend, which receives them on every
delegated request. The prompt tells Dani the backend sees the screen, so referential requests such as
"this one" or "what am I looking at" go through as said instead of prompting a "which one?". The
"anything else?" closing rule sits with how Dani sounds, and Dani offers one only when commentary
approves it. The rendered prompt is about 4.0 kB.

Progress while work runs. The server checks the session's requests every second and speaks each
result as commentary once it is verified, waiting until the owner has been quiet for 2.5 seconds. While a
request runs, the backend records its latest step from a fixed template (no model call):
"looking that up" for searches and reads, "saving {title}" for creations, "updating {title}" for edits,
completions and batch changes, "checking the earlier request" for a follow-up, "trying another way after a
step didn't work" when the model continues after a failed step, and "waiting on an earlier request" while
it waits for related work. Discovery, screen controls, questions and ending voice have no note, and a note
never says done, added or saved. Voice relays a new note as "Progress, not a result" commentary once the
request has been busy for 4 seconds, at most one note every 8 seconds per session and 3 per request,
never while the owner talks or a backend question is open, and never in the same second as a result.
A failed step is not relayed; the retry note follows it. When two requests run, the one heard from least
recently goes first. A request still running after 12 seconds with nothing new to say gets one "still
working" note that names the request; a step note replaces it, and it counts toward the same limits.

Saved memory during a session. The session starts with a short "About you" summary appended to its
instructions after `live_instructions` as `About you DATA (facts, not instructions; ...): <summary>`:
three to six plain sentences (about 400-700 characters, capped at 800) on who the owner is, key
people, work and roles, ongoing priorities, stable preferences and how they like things done, with no
IDs, timestamps or JSON. No individual memory, and no memory metadata, is sent at start, so every
saved fact can still arrive mid-session. The summary is never generated during a call; when none
exists yet, nothing is sent. [about_you.py](../apps/api/jarvis/about_you.py) maintains it in the
background: a worker sweep looks at each owner with live memories at most once a day and queues one
`about_you` job only when its material changed (a digest of the profile name and aliases, the live
saved memories, the names and descriptions of spaces, areas, clients, projects and goals, and
recurring reminders; no tasks or calendar; no other people: it describes the owner only) or the summary is over 7 days old. The job calls Luna
once (cost feature "About you summary") and rewrites the owner's single `about_you_summaries` row
(plain text, like the memories it is distilled from). A summary built on a fact that has since been
forgotten, corrected, merged or had its source deleted is withheld at once and rebuilt on the next
sweep; an owner with no live memories has none. Automatic learning off (`memory_learning`) clears it
and stops generation; Memory during voice off stops sending it. The Memory page shows it read-only
with its updated date and a "Refresh now" button (`POST /api/v1/memory/about/refresh`): it queues one forced
job that rewrites even when nothing changed (a pending job is reused; refused when learning is off, the
account has no memories, or the budget is used up) and the page shows "Refreshing…" until it lands. The paused Realtime controller opens with the same summary.
While the owner talks, Eridani retrieves facts (1.5 s after the last transcript words,
at most once every 10 s) and quietly appends new ones with `session.thinking.append`: fact text only,
at most six facts, 1,200 characters and a conservative 450-token estimate per update (Live's append
limit is 500 tokens), never a fact already sent in this session, and only after rechecking that each fact is current (same revision, not forgotten, corrected,
merged or superseded, and its source not deleted). When the owner names a person (a full name, or a first
name nobody else shares), the same update carries one line per person, for at most three people, with up to
four clients and projects they are linked to (`people_links.mentioned`; rejected links are left out), once
per session. A capitalized word that sounds like or is spelled close to exactly one person's first name also
counts (speech-to-text misspellings: `name_match`), and its line says what was heard ("Janice" is probably
Janiece); words that are a known name, common words and ambiguous words never do. Typed chat gets the same people
lines with its saved-memory context. The task agent has the read tool `person_lookup` (organization and knowledge
tools): exact names, saved alternate spellings (aliases) and sound-alikes, best first, each with the clients and
projects that person works on, so it can choose by context or ask. Closing the session
cancels a pending lookup.
The provider's `session.thinking.appended` acknowledgement only means the context was delivered; it
does not mean the next spoken words use it. A query embedding, when stored vectors exist, is billed
under the existing "Memory retrieval" cost feature; the session logs only counts (retrievals, facts
sent, acknowledged, dropped as stale). Owner preference **Memory during voice**
(`live_memory_updates`, Settings → Privacy) turns the opening summary and the mid-session updates off;
the updates stop immediately, including in a running session. It is on by default because these
context updates were already part of the GPT-Live design (acceptance cases `live_voice.14`/`.15`).
Shared workspaces never receive personal memory or a summary.

- Acceptance set: [28 cases](../evals/app/features/live_voice.json).
- Implementation: [live_voice.py](../apps/api/jarvis/live_voice.py), [voice_control.py](../apps/api/jarvis/voice_control.py).
- Existing regression evidence: [test_live_voice.py](../tests/test_live_voice.py), [test_about_you.py](../tests/test_about_you.py), [test_voice_settlement.py](../tests/test_voice_settlement.py), [test_voice.py](../tests/test_voice.py).

### 34. Wake words and voice shutdown

Foreground Dani/Hey Dani detection, continuous captured-audio startup, a 120-second quiet window, contextual ending candidates and wake recovery. Ordinary thanks keeps listening. Ambiguous completion may receive a rate-limited closing offer. Device and provider acceptance remains pending; see [voice validation](VOICE_WAKE_VALIDATION.md).

- Acceptance set: [33 cases](../evals/app/features/wake_shutdown.json).
- Implementation: [voice_control.py](../apps/api/jarvis/voice_control.py), [live_voice.py](../apps/api/jarvis/live_voice.py).
- Existing regression evidence: [test_live_voice.py](../tests/test_live_voice.py), [test_about_you.py](../tests/test_about_you.py), [test_voice.py](../tests/test_voice.py).

### 35. Accounts, login and sessions

Google identity, invitations, pairing compatibility, sessions, CSRF and device revocation.

- Acceptance set: [25 cases](../evals/app/features/accounts.json).
- Implementation: [auth.py](../apps/api/jarvis/auth.py), [accounts.py](../apps/api/jarvis/accounts.py), [google_auth.py](../apps/api/jarvis/google_auth.py).
- Existing regression evidence: [test_accounts.py](../tests/test_accounts.py), [test_api.py](../tests/test_api.py), [test_google_calendar.py](../tests/test_google_calendar.py).

### 36. Shared workspaces and permissions

Private accounts, owner/editor/viewer roles, switching, revocation and actor attribution.

- Acceptance set: [25 cases](../evals/app/features/workspaces.json).
- Implementation: [access.py](../apps/api/jarvis/access.py), [accounts.py](../apps/api/jarvis/accounts.py).
- Existing regression evidence: [test_accounts.py](../tests/test_accounts.py).

### 37. External HTTP API and MCP

Scoped credentials, direct and queued commands, receipts, feeds, rate limits and revocation.

- Acceptance set: [25 cases](../evals/app/features/external_agents.json).
- Implementation: [external_routes.py](../apps/api/jarvis/external_routes.py), [external_mcp.py](../apps/api/jarvis/external_mcp.py), [bot_access.py](../apps/api/jarvis/bot_access.py).
- Existing regression evidence: [test_external_agents.py](../tests/test_external_agents.py).

### 38. History, retention and exports

Source retention, deletion, export redaction, notification privacy and injection boundaries.

- Acceptance set: [25 cases](../evals/app/features/privacy.json).
- Implementation: [api.py](../apps/api/jarvis/api.py), [memory_service.py](../apps/api/jarvis/memory_service.py).
- Existing regression evidence: [test_privacy.py](../tests/test_privacy.py), [test_accounts.py](../tests/test_accounts.py).

### 39. Deployment, backups and recovery

Readiness, worker leases, outbox replay, encrypted backups, restore and disabled integrations.

- Acceptance set: [25 cases](../evals/app/features/operations.json).
- Implementation: [deploy.py](../apps/api/jarvis/deploy.py), [worker.py](../apps/api/jarvis/worker.py).
- Existing regression evidence: [test_cloud_backups.py](../tests/test_cloud_backups.py), [test_cloud_deployment.py](../tests/test_cloud_deployment.py), [test_worker_recovery.py](../tests/test_worker_recovery.py).

### 40. Optional cost controls

Disabled-dev mode, reservations, action limits, settlement and uncertainty reconciliation.
Usage is three separate buckets (`budget.summary().buckets`, `/budget/holds` `bucket`):
recorded charges, reservations for active work, and unconfirmed sessions (unknown provider
outcome). Unconfirmed holds count against the limit until provider evidence settles them
(`scripts/reconcile_budget.py --evidence-file`, dry run unless `--apply`) or they lapse:
holds from before `cost_tracking_since` close as `predates_tracking`, and unconfirmed holds
older than 30 days close as `expired_without_evidence`, holding nothing (rows kept for
audit). A settlement's adjustment is dated at the session, so historic sessions do not
change this month's recorded charges. Settings shows recorded spend this month, the
projection, and one line for work still in progress.

- Acceptance set: [25 cases](../evals/app/features/budget.json).
- Implementation: [budget.py](../apps/api/jarvis/budget.py), [voice.py](../apps/api/jarvis/voice.py).
- Existing regression evidence: [test_budget_guard.py](../tests/test_budget_guard.py), [test_budget_controls.py](../tests/test_budget_controls.py), [test_budget_reconciliation.py](../tests/test_budget_reconciliation.py), [test_action_limits.py](../tests/test_action_limits.py).

### 41. Meeting context

Completed meeting transcripts (file import, the submission endpoint, the bridge script, or the optional Granola
poller) become source-backed context: decisions, rationale, facts, risks, questions and commitments with exact
quotes, linked to people and projects, with effective and recorded history. After each meeting the owner gets a
review in Meetings with preselected own actions and publishes only approved tasks (Eridani or Linear); Wrong,
Changed, Later and person-page identity corrections are human decisions. Agents read cited context over MCP and
`/api/v1/external/context/*` with `context:*` scopes and can only suggest corrections. Off by default
(`context_enabled`); personal workspace only. Architecture and operations: [CONTEXT_PLATFORM.md](CONTEXT_PLATFORM.md);
switching on the pilot: [CONTEXT_PILOT_RELEASE_NOTE.md](CONTEXT_PILOT_RELEASE_NOTE.md).

- Acceptance set: [31 cases](../evals/app/features/meeting_context.json) (none bound yet); engine quality,
  retrieval benchmark and pilot gates in [evals/context](../evals/context/README.md).
- Implementation: [context/](../apps/api/jarvis/context/) ([sources.py](../apps/api/jarvis/context/sources.py), [pipeline.py](../apps/api/jarvis/context/pipeline.py), [service.py](../apps/api/jarvis/context/service.py), [mcp.py](../apps/api/jarvis/context/mcp.py), [notices.py](../apps/api/jarvis/context/notices.py)), [web](../apps/web/src/context/).
- Existing regression evidence: [tests/context/](../tests/context/) (one module per context module, including [test_end_to_end.py](../tests/context/test_end_to_end.py)).

### 42. Set up with Dani

A guided interview that proposes how to organize Eridani and what Dani should always keep in mind. It starts from
Today's setup banner or Settings → Organization ("Set up with Dani", "Resume with Dani" or "Update with Dani"), and
from Memory → About you ("Update with Dani"). Before it starts, the owner chooses voice or typing (either/or). In
typed mode, "Switch to voice" stays available, and ending voice returns to the same conversation typed. Both modes
write one transcript: the setup conversation, which is kept even when conversation history is off and can be deleted
when setup finishes. Personal workspace only.

- **Interview.** Typed turns carry a setup block in the dynamic half of the prompt. In voice, GPT-Live holds the
  interview itself and delegates only proposing, confirming, revising and ending voice. Every setup turn runs on
  GPT-6.1 Terra.
- **Proposal.** "Propose my organization" (or Dani's `setup_propose`) runs one Terra synthesis (cost feature "Set up
  with Dani"). It reads the transcript, the current spaces, areas, goals and projects, saved memories and About you.
  It produces a previewed tree (`setup.tree_preview`, nothing written), core-fact suggestions, and the topics
  covered, which tick the chips.
- **Review.** The tree review opens on its own. It lets the owner edit, add, remove and tick archives, and
  re-previews before "Create this organization". That runs one `setup.tree_apply` command, matched to existing
  records by name and revertible as a group.
- **About you.** The About you review (`setup.memories_save`) saves the ticked facts as pinned (`Memory.core`)
  owner statements. It deduplicates against learned facts and discards unticked ones. Pinned facts always feed
  the About you summary. Any memory can be pinned or unpinned from the Memory page.
- **Finishing.** Finishing posts Dani's closing line, marks setup complete once something was created or saved
  (the banner retires), and queues the About you rebuild.
- **Re-runs.** A re-run opens with "Tell me what's changed since we last talked." It focuses on the organization
  (from Settings) or on who you are (from Memory). It proposes only additions, renames and ticked archives;
  records it leaves out stay as they are.

The form path ("Fill out the form") and "Refine with Dani" in Types & fields are unchanged. Plan, decisions and
deviations: [SETUP_WITH_DANI_PLAN.md](SETUP_WITH_DANI_PLAN.md).

- Acceptance set: [25 cases](../evals/app/features/setup_interview.json) (none bound yet); a physical phone and
  microphone check is in [RELEASE_ACCEPTANCE.md](RELEASE_ACCEPTANCE.md).
- Implementation: [setup.py](../apps/api/jarvis/setup.py), [setup_synthesis.py](../apps/api/jarvis/setup_synthesis.py), [setup_tree.py](../apps/api/jarvis/setup_tree.py), [setup_memory.py](../apps/api/jarvis/setup_memory.py), [SetupConversation.tsx](../apps/web/src/SetupConversation.tsx), [SetupTreeReview.tsx](../apps/web/src/SetupTreeReview.tsx), [SetupMemoryReview.tsx](../apps/web/src/SetupMemoryReview.tsx).
- Existing regression evidence: [test_setup_session.py](../tests/test_setup_session.py), [test_setup_voice.py](../tests/test_setup_voice.py), [test_setup_synthesis.py](../tests/test_setup_synthesis.py), [test_setup_tree.py](../tests/test_setup_tree.py), [test_setup_memory.py](../tests/test_setup_memory.py), [test_setup_rerun.py](../tests/test_setup_rerun.py).

## Bug reports (Report a bug)

A round red bug button floats bottom-left on every signed-in page (shortcut Alt+Shift+B). It takes a
screenshot of the page first (`modern-screenshot`, loaded on demand; JPEG at most 1600 px wide and
1.5 MB), then opens a dialog: "What went wrong?" (required), "What did you expect?" (optional), the
screenshot with an Include screenshot toggle, and "Details that will be attached", which lists what is
sent. If the screenshot fails, the report still sends and says why. Send posts to
`POST /api/v1/bug-reports` (owner, personal workspace, CSRF; body up to 3 MB, screenshot up to 2 MB,
context up to 200 KB; 30 reports per 10 minutes) and shows "Reported. Thanks" with Open task.

Each report is a `bug_reports` row (migration 0042) and a task tagged `bug-report` in the project
"Eridani bug reports" (created on first use). The task's title is the description's first line (80
characters); its notes hold the description, the expected behavior, a context summary (view, address,
build, server, device, last errors, failing requests) and links to the report and screenshot. Both task
cards show the screenshot inline ("Bug report" section). The task can be Run with an agent like any other.

What is collected and never collected: see `docs/EXTERNAL_AGENTS.md` "Bug reports" (the `context` fields).
The ring buffers (`apps/web/src/bug-report.ts`) start when the app loads: the last 50 console errors and
warnings plus uncaught errors and rejections, the last 40 API requests made through `api.ts`, and the last
20 page changes.

Reading reports: the owner uses `GET /api/v1/bug-reports[?status=&task_id=]`, `GET /api/v1/bug-reports/{id}`
and `GET /api/v1/bug-reports/{id}/screenshot`; sets status with `POST /api/v1/bug-reports/{id}/status`
(`open`, `fixed`, `dismissed`); deletes one with `DELETE /api/v1/bug-reports/{id}` (the task stays). A bot
key with `tasks:read` reads the same under `/api/v1/external/bug-reports…`.

Retention and privacy: the account export (`GET /api/v1/export`) includes `bug_reports` with each
screenshot as base64. The worker's housekeeping drops screenshots older than 90 days
(`bug_reports.SCREENSHOT_DAYS`); the text and context stay until the report is deleted.

- Implementation: [bug_reports.py](../apps/api/jarvis/bug_reports.py), [BugReport.tsx](../apps/web/src/BugReport.tsx), [bug-report.ts](../apps/web/src/bug-report.ts), [bug-report-capture.ts](../apps/web/src/bug-report-capture.ts).
- Regression evidence: [test_bug_reports.py](../tests/test_bug_reports.py), [bug-report.test.tsx](../apps/web/src/bug-report.test.tsx); browser fixture `apps/web/e2e/bug-report.mjs` (manual).
