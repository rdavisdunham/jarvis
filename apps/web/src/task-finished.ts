import type { CalendarEntry, Task } from "./types";
import { localDateTime } from "./workspace";

export const isFinishedTask = (task: Pick<Task, "status">) => ["completed", "cancelled"].includes(task.status);
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
