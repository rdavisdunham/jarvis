import { api } from "./api";
import type { CalendarEntry, Task } from "./types";
import { localDateTime } from "./workspace";

export const isFinishedTask = (task: Pick<Task, "status">) => ["completed", "cancelled"].includes(task.status);
/** A finish a sync wrote (a Linear issue closed there, set by linear_sync.apply_remote), not one made in Eridani. */
export const finishedBySync = (task: Pick<Task, "external">) => !!task.external?.finished_via;
export function finishedToday(tasks: Task[], today: string, zone: string) {
  return tasks.filter(task => {
    const at = task.finished_at ?? (task.status === "completed" ? task.completed_at : null);
    return isFinishedTask(task) && !task.is_template && !task.quick_list_parent_id &&
      (!task.archived || task.archive_reason === "automatic") && !!at &&
      Number.isFinite(Date.parse(at)) && localDateTime(at, zone).slice(0, 10) === today;
  }).sort((a, b) => (b.finished_at ?? b.completed_at ?? "").localeCompare(a.finished_at ?? a.completed_at ?? "") || a.title.localeCompare(b.title));
}
export function activeTodayEntry(entry: CalendarEntry, tasks: Task[]) {
  if (entry.kind !== "task") return true;
  const task = tasks.find(task => task.id === entry.task_id);
  return !isFinishedTask(entry) && (!task || (!isFinishedTask(task) && !task.archived));
}
/** Current workspace rows win over history, including optimistic reopen/status changes. */
export function mergeFinishedTasks(history: Task[], current: Task[]) {
  return [...new Map([...history, ...current].map(task => [task.id, task])).values()];
}

/** Daily history is a separate, paginated read; ordinary task loads remain unchanged. */
export async function readFinishedTasks(today: string, zone: string, current: () => boolean = () => true) {
  const items: Task[] = [];
  let cursor: string | null = null;
  do {
    const params = new URLSearchParams({ limit: "200", finished_on: today, timezone: zone });
    if (cursor) params.set("before", cursor);
    const page: { items: Task[]; next_cursor: string | null } = await api("/tasks?" + params);
    items.push(...page.items);
    cursor = page.next_cursor;
  } while (cursor && current());
  return items;
}
