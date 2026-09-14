# Eridani productivity schema

Status: deployed as schema 0012; release and backup validation are recorded in TODO.md.

## Organization and outcomes

- **Space** is a private organizational context, initially Personal and Business.
  Personal and shared workspace scopes control record access. Legacy spaces do not automatically
  provide shared membership or collaboration permissions.
- **Area** is an ongoing responsibility within a space, such as Health or
  Operations. It has no completion state; it can be archived.
- **Goal** records an outcome, success criteria, optional parent goal, horizon,
  target date, and optional numeric baseline/current/target/unit.
  Short-term and long-term goals share one model. Parent cycles are rejected.
- **Project** records a finite initiative, success criteria, start/target dates,
  status, and optional space/area.
- **GoalProjectLink** is a many-to-many relationship. Projects may support several
  goals and goals may have several projects. Editing either side updates the
  reverse relationship and peer revisions in the same transaction.
- **Task** remains the actionable unit, with optional project, parent task, tags,
  priority, work type, and assignment. Tasks without a project can have a space
  and area or remain unclassified in Inbox. Goal assignment is currently through
  a supporting project.
- **Actor** gives a stable local ID to each assignee name and distinguishes people
  from agents. Assignment records responsibility; it does not authorize or launch
  an agent execution.

Goals use planned, active, on_hold, achieved and abandoned states. Projects use
planned, active, on_hold, completed and cancelled. Task states retain their
existing meanings. Completion is recorded separately from archive state.

A goal's progress comes from its outcome metric, including decreasing targets;
it never comes from task counts. Marking an outcome achieved is explicit.
Projects show completed task counts without changing their goals' statuses.

## Dates, alerts and recurrence

Task planned_date is a floating date for intended work. due_date is the actual
deadline, with optional due_time and due_timezone. estimate_minutes describes
expected effort. Planned dates appear in the Today and Next 7 days tabs and Calendar.
Moving a planned date changes neither the deadline nor any existing reminder.

PlanningEntry continues to represent appointments and task work blocks. A work
block reserves an interval. Planned dates and date-only deadlines reserve no busy
time and do not automatically create Google events.

Tasks own completion. Schedule remains the durable alert/recurrence definition:
a reminder repeats attention for its linked task; recurring_task creates separately
completable occurrences from a template. Occurrence and Notification retain
delivery/completion history. Dismiss and snooze do not mark a task complete.
Completing a task closes its alerts. This release retains the existing proven
scheduler rather than migrating its internal records solely to rename tables.

## Notes and links

Note retains its authored content, revision, tags, optional conversation and primary
project/home. NoteTaskLink preserves source evidence for extracted tasks.

NoteGoalLink and NoteProjectLink attach one note to multiple goals and projects.
NoteNoteLink stores explicit directional links; incoming links are exposed as
backlinks. A note is stored once. Removing a link never deletes either record.
Goal/project details expose their linked notes; note details expose related notes
and backlinks. Keyword and semantic note search can filter by space, area, goal,
project and task. A goal filter also includes notes attached to its supporting
projects.

Authored notes remain separate from learned Memory assertions. Embeddings help
retrieve source material; they do not replace explicit graph relationships.
Existing source/evidence and memory privacy rules remain intact.

## Consistency and integrations

All commands go through the existing owner-scoped command service, revision
checks, commit-order locks, durable receipts and events. A repeated command ID
replays its prior result. Cross-owner IDs and cyclic parents are rejected.
Archiving preserves records and existing relationships; new relationships to an
archived target are rejected.

Project space/area changes carry the same home to its tasks and primary-project
notes. Area moves carry their space change to contained work. Cross-links do not
grant access or move a record to another home. Creating tasks remains possible
without completing organization fields.

Linear and Google bindings remain separate from the productivity model.
Local planning dates, goals, note links and classification do not silently become
provider fields. Existing provider write/conflict/recovery behavior is retained.

The authenticated /api/v1/organization snapshot and Eri's organization_list tool
expose current IDs and revisions. Eri can create/edit spaces, areas, goals,
projects and assignees, update note relationships, filter the workspace and open
or highlight organization records through the existing site controls.

Migration 0011 adds these records and fields without guessing classifications.
It preserves existing task contents, deadlines, alert identities, integration
credentials and cost records, and backfills stable assignee IDs. Default spaces
are seeded once. Backup restore verification includes all new graph tables.

## Later decisions

Shared workspace membership, invitations, scoped retrieval and saved task views
are implemented. Private-to-shared record moves and combined workspace views
remain extensions. Independent agent execution jobs remain distinct from task
assignment. Richer metric check-ins, dependencies and notification policy can
extend this foundation.


## Planner workspace projections (September 13)

Boards and timelines reuse Task and Project records; they introduce no tables.
Task planned/deadline markers are separate from project lifecycle bars and
PlanningEntry work blocks. The deterministic planner checks up to eight tasks
within seven days, then atomically commits verified local blocks.

Short proposal references use owner-scoped internal Command receipt storage.
The encrypted proposal expires after fifteen minutes; subsequent proposals prune
expired proposal rows. Committed plan receipts are retained independently for
idempotent retries. A proposal is not a saved work block or a Google publication.

The full schema/UX review, including intentional boundaries and later candidates,
is in [PLANNER_UX_ARCHITECTURE_REVIEW.md](PLANNER_UX_ARCHITECTURE_REVIEW.md).

## Tasks page and calendar details (September 13)

Tasks is one collection. The tabs select overlapping views of that collection:

- Today includes tasks planned or due through today, including overdue work.
- Inbox includes tasks with no project, space or area. A task can have dates
  and still be in Inbox; this is unfiled work, not necessarily unscheduled work.
- Next 7 days includes tasks planned or due through today plus six days,
  including overdue work. This replaces the ambiguous “This week” label.
- All replaces the old Work page and includes all non-archived tasks.

Search, status and organization filters apply to every tab and persist when
switching tabs. The default status filter hides completed and cancelled tasks.
List, Board and Timeline display the same filtered tasks. Dragging a board grip
changes status, project or assignee according to the selected grouping. Card
order follows the chosen sort; there is no stored manual card rank.

“Task reminder” adds an alert to an existing task, or creates a task if no task
is selected. It is not another type of actionable record. For example:
“Pay the invoice” can have a Friday 3 PM deadline and a Friday 1 PM reminder.
A 10–10:30 AM work block can reserve time for it. Completing the invoice task
closes its alerts; the reserved calendar interval does not complete the task.

Calendar items identify tasks, task reminders, repeating tasks, work blocks,
local events and Google events. Clicking an item opens saved details. Edit at
the upper right opens its actual editor. Tasks and created routine occurrences
can be completed; appointments, Google events and work blocks have no independent
completion. Future routine projections become completable when their occurrence
task is created. A repeating alert for one task and a repeating task with separate
occurrences remain different recurrence modes.

Eri uses the same controls and domain commands. ui_calendar accepts
open_details=true with a date and entity_id. ui_editor.read identifies local
saved detail cards as mode=detail; ui_form opens an editable draft. Navigation
can leave a saved detail card, while unsaved forms retain their existing guard.


## Invited accounts and shared workspaces (September 14, 2026)

See [Accounts and sharing](ACCOUNTS_AND_SHARING.md) for the current access model.
A shared space/project is a distinct record namespace with explicit membership;
existing personal records are not made shared by assigning them to another actor.
Personal memory and integration credentials remain separate. Command receipts
record the acting account. This supersedes the earlier single-owner limitations.
