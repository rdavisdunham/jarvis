# Configurable planner, organization learning and notifications

Status: deployed to production September 17, 2026, in `8e63e95`. Both Railway
services report SUCCESS for that commit. The local development database has not
been migrated in this batch. Database and browser acceptance used disposable databases and synthetic
records. This document supersedes the fixed classification design in the original
[TASK_ROUTING_PRD.md](TASK_ROUTING_PRD.md).

## What is available

### Shape the planner

Organization → Structure edits record types, singular/plural labels, required
descriptions, properties, field order/visibility, workflows and relationships.
Space, Area, Client, Project, Goal, Task and Note are the initial template. Users
can replace that vocabulary, add types and choose their behaviors:

- Actionable work: completion, deadlines, planned day, priority, effort and assignee.
- Content: authored notes using the existing note service and retrieval pipeline.
- Timeline: start and target dates, displayed as a range rather than a busy appointment.
- Metric: baseline, current value, target and unit, independent of task counts.

A record has one main home, which can be another allowed record type. For example,
Work → ABC → Transcript Intelligence → a task. Named links support additional
many-to-many relationships such as projects supporting several goals. Relation
fields can be single or multiple; optional inheritance follows the main home only.
Explicit empty values override inheritance; “Use inherited value” removes the override.
Cycles and cross-workspace links are rejected. Organization does not grant access.

Custom status names map to stable meanings such as backlog, active work, completed
and cancelled. Field labels and structure are customizable; identities, revisions,
permissions and operational meanings remain system responsibilities. Task dates
never become busy calendar blocks automatically.

Tasks includes every type with actionable behavior. Today, Inbox, Next 7 days and
All use the same records: Inbox is unfiled work; Today/week include due or planned
work through their cutoff, including overdue work. The workspace provides inline
cards, list/board/timeline, drag or touch-grip moves, home and custom-field filters,
saved views, completion and task multi-select. Bulk editing preserves the existing
operational task controls; custom classification is edited on record cards or with
Eri. Cards have copyable links and navigation history. Scheduling and linked notes
remain available through the task's “Schedule & linked notes” control.

Structure changes always produce an impact preview. Apply checks the schema,
record snapshot, author and expiry; a later request confirms changes proposed by
Eri. Structure history can preview a reversal. Existing saved fields/types are
archived rather than silently discarded; incompatible conversions explain what
must be resolved first. Ordinary record edits execute directly.

### Learn organization separately from personal memory

Type, field and relationship descriptions are assessed asynchronously. Eri asks
one necessary question when scope is ambiguous; sufficient descriptions need no
interview. Manual use remains available while assessment is pending or unavailable.
Questions appear in Profile → Settings → Organization learning → Review and can
be answered conversationally through Eri's routing tools. A pending interview can
receive the same once-daily offer as the weekly pattern review.

Routing evidence, rules, reviews and field understanding have their own tables.
They do not become personal-memory facts or embeddings. The worker's weekly review
runs Monday at 03:00 in the account time zone by default; day/hour are editable.
It proposes precise title phrases with an existing main home and/or classification
values. Review is also available manually. Repeated reviews skip model calls when
neither the human evidence nor schema changed.

Only explicit manual organization/corrections count as independent training
evidence. Agent guesses, imports and repeated versions of the same record do not
multiply support. The stable held-out split is by record identity. A discovered
rule needs at least three supporting examples, 100 held-out human-labeled matches
and 95% precision before automatic activation. Smaller samples become review
questions. A rule explicitly confirmed by the owner can start immediately.
**No real-user learned-rule quality gate has been claimed as passed.**

Strong, non-conflicting matches can apply automatically to new work. Uncertain
fields remain unassigned; explicit choices win. Suggest-only and Off are available.
Work hours are editable weak context, never a rule on their own. Rules cannot infer
deadlines, assignees, permissions or a different workspace. Corrections pause
conflicting rules; changes to relevant definitions require review. Unrelated schema
changes retain valid rules. Pause, confirm and forget are available; forgetting
suppresses the supporting evidence. Existing records only change after a separate
reorganization preview and confirmation.

The weekly interview offers at most once per account-local day after ordinary
successful work. It does not open the microphone or interrupt necessary task
clarification. Seen/heard acknowledgment records delivery across devices. The panel
shows one question at a time and offers a stopping point after three answers.
Later today, tomorrow and next week are available for pattern questions. Spontaneous
mid-conversation learning questions remain a later feature.

### Smarter notifications

Timed task deadlines can generate alerts, with per-task On/Off/default and an
explicit urgent override. Date-only deadlines stay in the planner/summary. A
coincident explicit reminder avoids a second deadline alert. Completed, archived,
rescheduled and already-seen work is checked again before delivery.

Quiet hours default to 22:00–08:00 local time. Only an explicitly urgent alert
bypasses them; task priority alone does not. The optional daily summary defaults
to 08:00, skips empty days and consolidates eligible held notices. Snooze changes
the same notification's next delivery, not the task deadline or recurring schedule;
choose a duration or exact time. Delivery generations prevent old retries from
sending a superseded snooze or deadline.

Unseen successful background work groups for 30 seconds; questions and failures
have their own actions and open Activity. Task notifications open the correct
record using the existing workspace-aware link flow. Real locked-phone delivery
and provider behavior still require the device checks below.

## Architecture and compatibility

- PostgreSQL migrations: `0015_custom_structure`, `0016_smart_notifications`.
- Registry: `structure_schemas`, `structure_records`, `structure_links`,
  `structure_proposals`; bounded JSON definitions with stable IDs and revisions.
- Learning: `field_understandings`, `routing_observations`, `routing_patterns`,
  `routing_reviews`; existing durable jobs perform assessment and weekly review.
- The existing Task/Note services remain the operational source for completion,
  scheduling, recurrence, external sync and note indexing. Generic records link to
  those identities. Recurring/imported core rows are adopted; external edits
  invalidate stale open cards and unsafe undo. Retired capabilities are not revived.
- Existing spaces, areas, projects, goals, tasks and notes get a best-effort import.
  Source rows are retained; no account, credential, memory or external record is
  intentionally deleted. Legacy organization endpoints remain for compatibility;
  new organization should use schema/record tools rather than those fixed types.
- Eri discovers current definitions and uses record tools plus `ui_records` and
  bounded `ui_editor` patches. Custom links/fields produce receipts with guarded
  undo. Structural undo is a new preview, never a blind rollback.
- External HTTP/MCP supports `schema:read/write` and `records:read/write`.
  Structure writes require workspace ownership. Custom-record access includes
  configured content/note bodies, explicitly labeled in Connected agents settings.
  Existing credentials gain no additional scopes automatically.
- Shared workspaces retain member roles and ownership rules; personal learning,
  memory and push policies remain outside shared organization.

## Verification and rollout

Automated checks cover schema validation, cycles, owner boundaries, cardinality,
stale previews, inheritance/clears, capability mapping, retry deduplication, core
sync, action/link undo, rule conflicts/evidence/held-out gating, provider failure,
weekly scheduling, deadline deduplication, quiet hours and snooze generations.

The final clean-environment backend suite passed (638 tests plus one intentional
skip), with local environment files and provider credentials excluded. Frontend
tests passed (123 plus one intentional skip). The production
build passes; the existing large-bundle advisory remains. Isolated browser
acceptance exercises an empty-database upgrade, preview/apply, inline editing,
Eri navigation, timeline, task bulk editing, record deep links, desktop/mobile
layouts and settings. The tests use synthetic credentials; live model behavior
and actual push delivery are not simulated as proof of production acceptance.

Reproduce from the repository:

```sh
.venv/bin/pytest -q
cd apps/web && npm test && npm run build && cd ../..
PYTHONPATH=apps/api .venv/bin/python scripts/validate_custom_planner.py
```

Deployment releases the API and worker together using existing Railway PITR
for recovery. Recent PostgreSQL logs confirm successful continuous archive uploads.
An additional local database export was blocked by automatic approval review and
was not retried; this release does not create or claim an independent backup. Run the normal Alembic upgrade before serving the new code; restart the
worker to register the new job kinds. Destructive downgrades are deliberately not
provided: test rollback through an isolated restored database. No R2 credentials
were required or changed for this batch.

Still to verify before a broader release:

- [ ] Physical phone: keyboard, touch board moves, inline saves, wake/goodbye and
  voice recovery while opening custom cards.
- [ ] Live description assessment and weekly interview with the actual model,
  including user corrections and a later-day deferred review.
- [ ] Locked-phone push, urgent versus quiet-hours delivery, exact-time snooze,
  daily summary, and a notification opened while another workspace is selected.
- [ ] Real Google/Linear and recurring-task edits appear once with correct custom
  organization; real external client uses the new scopes and loses access on revoke.
- [ ] Second-account permissions; cloud API/worker restart; backup/restore drill.

The complete pending owner/device checklist remains in [TODO.md](TODO.md).

## Production verification — September 17, 2026

- Web deployment: `40ebef44-f218-4dee-8b3f-139dd6a9e607`, SUCCESS.
- Worker deployment: `a45cf519-c4c6-4b6d-ab36-5331e9e3826d`, SUCCESS.
- Both deployment metadata records identify commit `8e63e95907b131ad6b3e60a8ef9a3ea0970850ff`.
- Public page and `/health/ready` return HTTP 200. Frontend JavaScript
  `index-DMcsGTZi.js` and CSS `index-98bLRaCn.css` hashes match the tested local build.
- API startup and DBOS worker launch appear in recent logs, with no error/traceback
  lines in the inspected window. Native PITR logs show successful archive uploads.
- These are rollout checks, not proof of real-device voice, push delivery, live
  learning interviews, or connected-account behavior. Those checks remain open.
