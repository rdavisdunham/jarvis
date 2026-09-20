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

GPT-Live is the active voice frontend. Durable backend work uses the selected task
agent (Luna or Gemini; the code also retains a Groq profile). Accepted requests survive
voice interruption. Independent requests can proceed concurrently; references and
clarification answers carry dependencies and lineage. Commands validate permissions,
arguments and revisions before transactionally saving effects and action receipts.
Cards report verified changes; Revert checks for intervening edits and links.

## Learning and search are separate systems

Personal-memory learning extracts source-backed facts using Luna and embeds eligible
facts with OpenAI text-embedding-3-small. Retrieval combines meaning and lexical evidence,
checks current visibility/revision, and injects relevant facts as data into agent context.
The memory dream pass merges safe duplicates and queues ambiguous spellings/facts for
review. It does not silently pick between Miso and Mizo.

Organization learning uses human task/note classifications, field descriptions and
corrections in separate tables. Its weekly dream pass proposes patterns, validates
targets against the current schema and holds ambiguous/insufficient patterns for review.
Automatic assignments do not count as independent human evidence. Work hours are a weak
hint. Rules may classify records; they may not invent deadlines, change responsibility
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

Quiet hours, urgency, deadline alerts, summaries, snooze, dismiss and completion.

- Acceptance set: [25 cases](../evals/app/features/notifications.json).
- Implementation: [notices.py](../apps/api/jarvis/notices.py), [task_alerts.py](../apps/api/jarvis/task_alerts.py), [worker.py](../apps/api/jarvis/worker.py).
- Existing regression evidence: [test_smart_notifications.py](../tests/test_smart_notifications.py), [test_daily_polish.py](../tests/test_daily_polish.py).

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

- Acceptance set: [25 cases](../evals/app/features/linear_sync.json).
- Implementation: [linear_sync.py](../apps/api/jarvis/linear_sync.py), [linear_client.py](../apps/api/jarvis/linear_client.py).
- Existing regression evidence: [test_linear.py](../tests/test_linear.py).

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

Actual saved changes, compact anchored cards, field-level reversion and history cleanup.

- Acceptance set: [25 cases](../evals/app/features/receipts.json).
- Implementation: [action_history.py](../apps/api/jarvis/action_history.py), [agent_work.py](../apps/api/jarvis/agent_work.py).
- Existing regression evidence: [test_agent_work.py](../tests/test_agent_work.py), [test_work_continuation.py](../tests/test_work_continuation.py), [test_external_agents.py](../tests/test_external_agents.py).

### 30. Conversation and backend agents

Production tool discovery, context, personality, provider continuity and honest reporting.

- Acceptance set: [25 cases](../evals/app/features/chat_agent.json).
- Implementation: [conversation.py](../apps/api/jarvis/conversation.py), [agent_models.py](../apps/api/jarvis/agent_models.py), [responses_adapter.py](../apps/api/jarvis/responses_adapter.py).
- Existing regression evidence: [test_agent_models.py](../tests/test_agent_models.py), [test_responses_agent.py](../tests/test_responses_agent.py), [test_tool_refinement.py](../tests/test_tool_refinement.py).

### 31. Conversational site controls

Acknowledged navigation, filters, selection, editor control and draft protection.

- Acceptance set: [25 cases](../evals/app/features/site_controls.json).
- Implementation: [ui_control.py](../apps/api/jarvis/ui_control.py), [ui_contracts.py](../apps/api/jarvis/ui_contracts.py), [tools.py](../apps/api/jarvis/tools.py).
- Existing regression evidence: [test_ui_control.py](../tests/test_ui_control.py), [test_batch1.py](../tests/test_batch1.py), [test_tool_refinement.py](../tests/test_tool_refinement.py).

### 32. Browser navigation and settings

Back/Forward, persistent chat affordance, fold layouts, profile menu and preferences.

- Acceptance set: [25 cases](../evals/app/features/browser_settings.json).
- Implementation: [api.py](../apps/api/jarvis/api.py), [saved_views.py](../apps/api/jarvis/saved_views.py).
- Existing regression evidence: [test_planner_release.py](../tests/test_planner_release.py), [test_api.py](../tests/test_api.py).

### 33. GPT-Live sessions

Live creation, captions, delegation, interruption, context updates and session recovery.

- Acceptance set: [25 cases](../evals/app/features/live_voice.json).
- Implementation: [live_voice.py](../apps/api/jarvis/live_voice.py), [voice_control.py](../apps/api/jarvis/voice_control.py).
- Existing regression evidence: [test_live_voice.py](../tests/test_live_voice.py), [test_voice_settlement.py](../tests/test_voice_settlement.py), [test_voice.py](../tests/test_voice.py).

### 34. Wake words and voice shutdown

Hey Eri/Eri detection, 30-second quiet timeout, natural goodbye and wake-up recovery.

- Acceptance set: [25 cases](../evals/app/features/wake_shutdown.json).
- Implementation: [voice_control.py](../apps/api/jarvis/voice_control.py), [live_voice.py](../apps/api/jarvis/live_voice.py).
- Existing regression evidence: [test_live_voice.py](../tests/test_live_voice.py), [test_voice.py](../tests/test_voice.py).

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

- Acceptance set: [25 cases](../evals/app/features/budget.json).
- Implementation: [budget.py](../apps/api/jarvis/budget.py), [voice.py](../apps/api/jarvis/voice.py).
- Existing regression evidence: [test_budget_guard.py](../tests/test_budget_guard.py), [test_budget_controls.py](../tests/test_budget_controls.py), [test_budget_reconciliation.py](../tests/test_budget_reconciliation.py), [test_action_limits.py](../tests/test_action_limits.py).
