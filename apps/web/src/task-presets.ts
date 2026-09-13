import type { Task, View } from "./types";
import { shiftDate } from "./workspace";

export const taskTabs = [
  {
    id: "today",
    label: "Today",
    description: "Planned or due today, including overdue work.",
  },
  {
    id: "inbox",
    label: "Inbox",
    description: "Unfiled tasks: no project, space, or area yet.",
  },
  {
    id: "week",
    label: "Next 7 days",
    description:
      "Planned or due in the next seven days, including overdue work.",
  },
  {
    id: "all",
    label: "All",
    description: "All non-archived tasks. Your search and filters still apply.",
  },
] as const;
export type TaskTab = (typeof taskTabs)[number]["id"];
export function isTaskTab(view: string): view is TaskTab {
  return taskTabs.some((tab) => tab.id === view);
}
export function matchesTaskTab(task: Task, tab: TaskTab, today: string) {
  if (task.archived) return false;
  if (tab === "inbox")
    return !task.project_id && !task.space_id && !task.area_id;
  if (tab === "all") return true;
  const through = tab === "today" ? today : shiftDate(today, 6);
  return !!(
    (task.planned_date && task.planned_date <= through) ||
    (task.due_date && task.due_date <= through)
  );
}
export function initialView(search: string): View {
  const params = new URLSearchParams(search);
  const view = params.get("view");
  if (view === "tasks")
    return isTaskTab(params.get("tab") ?? "")
      ? (params.get("tab") as TaskTab)
      : "today";
  return [
    "today",
    "inbox",
    "week",
    "all",
    "organize",
    "calendar",
    "notes",
    "memory",
    "notifications",
    "settings",
    "reminders",
  ].includes(view ?? "")
    ? (view as View)
    : "today";
}
