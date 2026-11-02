# Finished tasks and automatic archiving

Implemented from the 2026-11-02 plan, based on main `bc554e4` and the user's ER-Q1 review decisions. The validation gate below records verification separately from deployment.

## Product behavior

Completed and canceled are outcomes. Archived means filed away; the task keeps its outcome, notes, relationships, and run-history links.

- Tasks outside the Today tab and Organization, including Atlas, hide finished tasks immediately. **Show completed and canceled** includes finished unarchived tasks. **Archived** opens filed-away history.
- Both Today views show unfinished work first, followed by checked **Completed today** items and a separate **Canceled today** section. The local calendar day uses the configured timezone. Finished rows are not duplicated in the schedule. Undated and future-due work completed today still appears in the daily checklist.
- The background worker archives eligible tasks after 24 elapsed hours, independently of whether the browser or desktop is open. It checks each maintenance cycle rather than waiting for a 4 AM sweep. Under normal capacity, eligible tasks should be filed within 15 minutes; downtime, a large backlog, or deferred execution/descendant checks can delay filing.
- Reopening returns a task to active work and clears archive eligibility. Restoring a finished archived task retains its outcome and gives it another 24 hours before automatic filing. Ordinary edits and repeated identical status updates do not restart the clock.
- Automatic archiving is on by default, with a personal/workspace policy toggle. Pausing it does not change visibility rules. Only shared-workspace owners can change that workspace's archive policy.
- Archiving does not stop execution or delete logs. Tasks with unfinished descendants or nonterminal/pending-review runs are deferred. Context-only ancestors keep active descendants reachable without changing their stored relationships.
- Organizational containers such as projects and custom container types remain visible in Organization and are excluded from automatic archiving, including those with task backing. An ordinary task with children still follows task visibility rules, with ancestor context retained for unfinished descendants.

## Work distribution and contracts

All implementation agents use GPT-6.1 Sol, with isolated worktrees from the same base.

| Owner | Files and responsibility | Integration contract |
|---|---|---|
| Backend agent | Task model/migration, lifecycle and Linear integration, worker, task API and settings authorization, backend regression tests | Nullable `finished_at`, `archive_eligible_at`, `archived_at`, `archive_reason`; preference `auto_archive_finished_tasks`; daily task query below |
| Tasks/Today agent | `App.tsx`, `Today.tsx`, daily checklist component/helper/styles, task types and legacy task card, frontend tests | Separate daily-finished state; shared `FinishedToday`; `todayFinished` prop for the shared task-list component |
| Organization/Atlas agent | `Structure.tsx`, Atlas/Browse UI and backend queries, graph projection/types, record card and content summaries, relevant tests | Archive/finished filters before pagination/graph limits; context-only ancestor projection; historical completion totals retained |
| Coordinating agent | Settings control, documentation, surface inventory, integration fixes, real browser acceptance, cross-review and CI/PR | Combine committed changes; validate complete flow on synthetic data; merge only after all required checks pass |

Task API: `GET /api/v1/tasks?finished_on=YYYY-MM-DD&timezone=<IANA zone>` uses the existing `before`/`limit` cursor pagination and `{items, next_cursor}` shape. This mode returns completed/canceled tasks finished during that local day, including automatic archives but excluding manual/external archives, templates, and quick-list children. Ordinary task loads keep their existing active/archive behavior.

Atlas API: `/structure/atlas` accepts `archived`, `show_finished`, and the canonical `status` filter. Apply visibility before graph limits and counts. Required hidden ancestor context is labeled with `context_only`, does not consume the visible record budget, and does not contribute to work/search totals. A context projection may open as a container without changing the saved record.

The same visibility semantics apply to organization browsing. Existing intentional saved views remain supported; a status filter from another view must not erase Today's separate checklist.

## Lifecycle and integration details

- Retain `completed_at` for compatibility. `finished_at` records the actual completion/cancellation transition; `archive_eligible_at` separately handles the 24-hour threshold and explicit restore grace.
- Legacy completion dates may be backfilled from reliable existing data. Unknown cancellation dates remain unknown; eligibility begins when the policy first observes the record. Do not invent historical activity or use unrelated last-edit times.
- Use bounded, idempotent worker batches, current-state/revision checks, and owner-scoped domain mutations/change events. Blocked early candidates must not starve eligible later tasks. Multiple workers and reopen-versus-sweep races must remain safe.
- Candidate and workspace savepoints isolate failures: a malformed task rolls back its own changes, logs the error for retry, and does not prevent other candidates or workspaces from being processed.
- Preserve PR #130's Linear cancellation/out-of-scope handling and run-stop behavior while replacing unconditional cancellation archiving with the 24-hour policy. Preserve explicit local/manual archive intent on remote reopen; automatic archives may clear when the remote issue reopens. A normal sync must not resurrect an automatically archived still-finished task.
- Preserve archived completion history in summaries. Recurring templates, organizational projects/goals/notes, and execution history are not cascaded into task archiving.
- Archive policy is separate from agent-content retention/deletion and from native Windows Eridani control.

## Validation and merge gate

- Backend regression coverage includes completion/cancellation age boundaries, unchanged-status edits, restore/reopen, legacy data, active descendants/runs, policy permissions, shared namespaces, blocked batches, concurrent sweep behavior, failure isolation, and Linear source changes.
- Migration upgrade and model agreement are verified on disposable PostgreSQL (`0044_task_archive`; no model drift).
- Daily query coverage includes owner isolation, pagination, configured timezone, local-day boundaries, canonical cancellation timestamps, and archive-source distinctions.
- Frontend tests cover daily checklist grouping and query pagination, archived outcomes, hidden ancestor context, and saved setting/error behavior. The integrated suite passes 496 tests, with one existing skip.
- `apps/web/e2e/task-archive.mjs` passes the real Tasks, Today, Atlas, worker, Restore/Reopen button, policy toggle, and mobile acceptance flow. It runs as part of the existing isolated planner browser suite in CI.
- Merge requires successful **Backend and migrations**, **Frontend and browser**, and **Security audit** checks on the final PR commit; the full regression suite and eval coverage ratchet remain required.

Local tests use a disposable PostgreSQL instance and `JARVIS_ENV_FILE=""`; no production task is archived as a test. The project has automatic deployment triggers on main after successful checks; merging does not itself establish that a deployment is healthy.
