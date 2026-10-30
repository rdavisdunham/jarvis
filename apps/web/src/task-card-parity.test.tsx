/// <reference types="vite/client" />
// Drift guard for the two task cards (docs/TASK_CARDS.md): RecordCard (the quick view) and
// TaskDetails (the legacy card) render the same task, and each must show every section in
// TASK_CARD_SECTIONS. Add a section to the constant when it belongs on both cards.
import { describe, expect, it, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import type { ReactElement } from "react";

// RunTask needs the remotea provider and a live server; stand in a section with the real label
// (checked against Remotea.tsx below) so the test sees whether each card mounts it.
vi.mock("./Remotea", () => ({
  RunTask: ({ taskId }: { taskId: string }) => <section aria-label="Agent run" data-task={taskId}/>,
}));
vi.mock("./Linear", () => ({ LinearTask: () => null, LinearSettings: () => null }));

import { RecordCard } from "./RecordCard";
import { TaskDetails } from "./TaskDetails";
import { TaskCardHostContext, type TaskCardHost } from "./TaskCardSections";
import { TASK_CARD_SECTIONS } from "./task-card-sections";
import type { Schedule, Task } from "./types";
import type { CustomRecord, Schema } from "./structure-types";
import { emptyOrganization } from "./productivity";
import remoteaSource from "./Remotea.tsx?raw";
import recordCardSource from "./RecordCard.tsx?raw";
import taskDetailsSource from "./TaskDetails.tsx?raw";

const source = { provider: "linear", label: "Linear", identifier: "ENG-1", url: "https://linear.app/x/issue/ENG-1", sync_state: "synced" };
const task = (id: string, title: string, extra: Partial<Task> = {}): Task => ({
  id, title, notes: "", status: "open", priority: 1, project: null, project_id: null, parent_task_id: null,
  assignee: "owner", work_type: "", tags: [], due_date: null, due_time: null, due_timezone: null, revision: 1,
  created_at: "2026-10-30T00:00:00Z", updated_at: "2026-10-30T00:00:00Z", completed_at: null, archived: false,
  occurrence_id: null, ...extra,
});
const parent = task("11111111-1111-4111-8111-111111111111", "Ship the release");
const subject = task("22222222-2222-4222-8222-222222222222", "Write release notes", { parent_task_id: parent.id, source });
const child = task("33333333-3333-4333-8333-333333333333", "Draft changelog", { parent_task_id: subject.id });
const tasks = [parent, subject, child];
const schedules: Schedule[] = [{
  id: "44444444-4444-4444-8444-444444444444", project_id: null, title: "Nudge me", timezone: "America/Chicago", recurrence: null,
  kind: "reminder", next_run_at: "2026-10-31T15:00:00Z", anchor_at: "2026-10-31T15:00:00Z", status: "active", revision: 1,
  task_id: subject.id, completed_at: null,
}];
const schema = {
  revision: 1, relationships: [], understandings: [],
  types: [{
    id: "task", name: "Task", plural: "Tasks", description: "", capabilities: ["work"], parent_types: ["task"], archived: false,
    statuses: [{ id: "open", name: "Open", meaning: "open" }, { id: "done", name: "Done", meaning: "completed" }],
    fields: [{ id: "due", name: "Due date", description: "", kind: "date", options: [], target_types: [], multiple: false, inherit: false, visible: true, archived: false, binding: "due_date" }],
  }],
} as unknown as Schema;
const record: CustomRecord = {
  id: "55555555-5555-4555-8555-555555555555", revision: 1, schema_revision: 1, type_id: "task", type_name: "Task",
  title: subject.title, body: "", values: {}, inherited: {}, parent_id: null, status_id: "open", archived: false,
  task_id: subject.id, note_id: null, capabilities: ["work"], home: [], links: [], source, local_notes: "",
};
const host: TaskCardHost = { tasks, schedules, openNote: () => {}, newNote: () => {}, reminder: () => {}, reserveTime: () => {} };

const quickView = () => <TaskCardHostContext.Provider value={host}>
  <RecordCard schema={schema} initial={record} choices={[]} canEdit onClose={() => {}} onChanged={async () => {}} onOpen={() => {}}/>
</TaskCardHostContext.Provider>;
const legacy = () => <TaskDetails id={subject.id} canEdit remotea onOpenRun={() => {}} organization={emptyOrganization}
  tasks={tasks} schedules={schedules} zone="America/Chicago" mutate={async () => undefined} onClose={() => {}}
  onTask={() => {}} onNote={() => {}} onNewNote={() => {}} onReminder={() => {}} onBlock={() => {}}/>;

const labels = (html: string) => new Set(Array.from(html.matchAll(/aria-label="([^"]*)"/g), (m) => m[1]));
/** Render twice: the first pass starts the lazy RunTask import, the second shows it. */
async function render(card: () => ReactElement) {
  renderToStaticMarkup(card());
  await import("./Remotea");
  await new Promise((resolve) => setTimeout(resolve, 0));
  return renderToStaticMarkup(card());
}

describe("task card parity", () => {
  it.each([["RecordCard", quickView], ["TaskDetails", legacy]] as const)("%s shows every shared section", async (_name, card) => {
    const shown = labels(await render(card));
    const missing = Object.values(TASK_CARD_SECTIONS).filter((label) => !shown.has(label));
    expect(missing).toEqual([]);
  });

  it("the Run stand-in uses RunTask's real section label", () => {
    const runTask = remoteaSource.slice(remoteaSource.indexOf("export function RunTask"));
    expect(runTask).toContain(`aria-label="${TASK_CARD_SECTIONS.run}"`);
  });

  it("both cards take the labels from the shared constant, not their own copies", () => {
    for (const [file, text] of [["RecordCard.tsx", recordCardSource], ["TaskDetails.tsx", taskDetailsSource]]) {
      for (const label of Object.values(TASK_CARD_SECTIONS)) expect(text, file).not.toContain(`aria-label="${label}"`);
    }
  });
});
