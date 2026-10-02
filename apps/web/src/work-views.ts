import { humanLabel } from "./ux";
import type { Task } from "./types";
import { shiftDate } from "./workspace";

export type WorkLayout = "list" | "board" | "timeline";
export type WorkSort = "priority" | "due" | "planned" | "title" | "updated";
export type WorkGroup = "status" | "project" | "assignee";
export type TimelineSpan = 14 | 30 | 90;
export const statuses = [
  "backlog",
  "open",
  "in_progress",
  "waiting",
  "deferred",
  "completed",
  "cancelled",
] as const;
export type TaskFilters = {
  assignee: string;
  work_type: string;
  tag: string;
  due_from: string;
  due_through: string;
};
export const emptyTaskFilters: TaskFilters = {
  assignee: "",
  work_type: "",
  tag: "",
  due_from: "",
  due_through: "",
};
export function matchesTaskFilters(task: Task, filter: TaskFilters) {
  return (
    (!filter.assignee ||
      (task.assignee_id ?? task.assignee) === filter.assignee ||
      task.assignee === filter.assignee) &&
    (!filter.work_type || task.work_type === filter.work_type) &&
    (!filter.tag || task.tags.includes(filter.tag)) &&
    (!filter.due_from ||
      (!!task.due_date && task.due_date >= filter.due_from)) &&
    (!filter.due_through ||
      (!!task.due_date && task.due_date <= filter.due_through))
  );
}
export function sortTasks(tasks: Task[], sort: WorkSort) {
  return [...tasks].sort((a, b) => {
    const primary =
      sort === "title"
        ? a.title.localeCompare(b.title)
        : sort === "updated"
          ? b.updated_at.localeCompare(a.updated_at)
          : sort === "planned"
            ? (a.planned_date ?? "9999").localeCompare(b.planned_date ?? "9999")
            : sort === "due"
              ? (a.due_date ?? "9999").localeCompare(b.due_date ?? "9999") ||
                (a.due_time ?? "99:99").localeCompare(b.due_time ?? "99:99")
              : b.priority - a.priority;
    return (
      primary ||
      (a.due_date ?? "9999").localeCompare(b.due_date ?? "9999") ||
      a.title.localeCompare(b.title) ||
      a.id.localeCompare(b.id)
    );
  });
}
export function groupedTasks(tasks: Task[], group: WorkGroup) {
  const groups = new Map<
    string,
    { key: string; label: string; tasks: Task[] }
  >();
  if (group === "status")
    statuses.forEach((key) =>
      groups.set(key, { key, label: humanLabel(key), tasks: [] }),
    );
  for (const task of tasks) {
    const key =
      group === "status"
        ? task.status
        : group === "project"
          ? (task.project_id ?? "")
          : (task.assignee_id ?? task.assignee);
    if (!groups.has(key))
      groups.set(key, {
        key,
        label:
          group === "project"
            ? task.project || "No project"
            : humanLabel(task.assignee || "Unassigned"),
        tasks: [],
      });
    groups.get(key)!.tasks.push(task);
  }
  return [...groups.values()];
}
export function timelineWindow(start: string, span: TimelineSpan) {
  return Array.from({ length: span }, (_, index) => shiftDate(start, index));
}
export function dayOffset(start: string, date: string) {
  return Math.round(
    (Date.parse(date + "T12:00:00Z") - Date.parse(start + "T12:00:00Z")) /
      86400000,
  );
}
export function taskDates(task: Task) {
  return { planned: task.planned_date ?? null, deadline: task.due_date };
}

// ---- Date presentation shared by task rows, boards and Today --------------------------
export type DueBucket = "overdue" | "today" | "upcoming" | "none";
export const dueBuckets: { key: DueBucket; label: string }[] = [
  { key: "overdue", label: "Overdue" },
  { key: "today", label: "Today" },
  { key: "upcoming", label: "Upcoming" },
  { key: "none", label: "No date" },
];
export function dueBucket(date: string | null | undefined, today: string): DueBucket {
  if (!date) return "none";
  return date < today ? "overdue" : date === today ? "today" : "upcoming";
}
/** "Today", "Tomorrow", "Yesterday", "Oct 5", or "Oct 5, 2027" outside this year. */
export function shortDate(date: string, today: string) {
  const offset = dayOffset(today, date);
  if (offset === 0) return "Today";
  if (offset === 1) return "Tomorrow";
  if (offset === -1) return "Yesterday";
  return new Date(date + "T12:00:00Z").toLocaleDateString("en-US", {
    timeZone: "UTC",
    month: "short",
    day: "numeric",
    ...(date.slice(0, 4) !== today.slice(0, 4) ? { year: "numeric" } : {}),
  });
}
/** "15:00" or "15:00:00" -> "3:00 pm". */
export function clockLabel(time: string) {
  const [h, m] = time.split(":").map(Number);
  if (Number.isNaN(h)) return time;
  return (h % 12 || 12) + ":" + String(m || 0).padStart(2, "0") + (h < 12 ? " am" : " pm");
}
/** "Oct 5" without relative words or year. */
export function monthDay(date: string) {
  return new Date(date + "T12:00:00Z").toLocaleDateString("en-US", { timeZone: "UTC", month: "short", day: "numeric" });
}
