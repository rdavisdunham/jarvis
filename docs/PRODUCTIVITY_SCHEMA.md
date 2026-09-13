# Eridani productivity schema

Status: deployed as schema 0011; release and backup validation are recorded in TODO.md.

## Organization and outcomes

- **Space** is a private organizational context, initially Personal and Business.
  Existing owner authorization still controls every record. Spaces do not yet
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
expected effort. Planned dates appear in Today, This week and Calendar.
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

Shared workspace membership, invitations and scoped retrieval need a dedicated
authorization design before enabling collaboration. Independent agent execution
jobs remain distinct from task assignment. Saved views, richer metric check-ins,
dependencies and notification policy can extend this foundation.
