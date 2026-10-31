/**
 * The sections both task cards show, by accessible name. `RecordCard.tsx` (the quick view) and
 * `TaskDetails.tsx` (the legacy task card) render each of these through the same shared component
 * (`TaskCardSections.tsx`, `SourceDetails.tsx`, `record-links.tsx`, `Remotea.tsx` `RunTask`), and
 * `task-card-parity.test.tsx` renders both cards for one task and fails if either card lacks one.
 * Add a section here when it belongs on both cards; see docs/TASK_CARDS.md.
 */
export const TASK_CARD_SECTIONS = {
  /** `SourceDetails`; only when the task has a source. */
  source: "Source",
  /** Eridani-only notes (the task record's `local_notes`). */
  localNotes: "Local notes",
  /** `RecordTools`: Copy record link and Previous record. */
  recordLink: "Record link",
  /** `BugReportDetails`: the screenshot and status behind a Report a bug task; only for tasks tagged `bug-report`. */
  bugReport: "Bug report",
  /** remotea `RunTask` (owned by Remotea.tsx; its section label must stay this). */
  run: "Agent run",
  /** Parent task and subtasks. */
  relatedTasks: "Related tasks",
  /** The new-subtask form. ("New subtask" is the input; "Add subtask" the button.) */
  addSubtask: "Add a subtask",
  /** Notes linked to the task, and New linked note. */
  linkedNotes: "Linked notes",
  /** Reminders, Add reminder and Reserve time. */
  schedule: "Schedule",
  /** The Save as template button. */
  saveTemplate: "Save as template",
} as const;

export type TaskCardSection = keyof typeof TASK_CARD_SECTIONS;
