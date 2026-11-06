import type { Task } from "./types";
import { finishedBySync, finishedToday } from "./task-finished";

/** Why an open task is on Today's "Due and overdue" list. Each task lands in exactly one bucket, so the
 *  parts always add up to the card's count: a deadline wins over a planned day, the past over today. */
export type DueReason = "overdue" | "due_today" | "slipped" | "planned";
export function dueReason(task: Pick<Task, "due_date" | "planned_date">, today: string): DueReason | null {
  if (task.due_date && task.due_date < today) return "overdue";
  if (task.due_date === today) return "due_today";
  if (task.planned_date && task.planned_date < today) return "slipped";
  if (task.planned_date === today) return "planned";
  return null;
}

export type DueCounts = Record<DueReason, number>;
export function dueCounts(due: Pick<Task, "due_date" | "planned_date">[], today: string): DueCounts {
  const counts: DueCounts = { overdue: 0, due_today: 0, slipped: 0, planned: 0 };
  for (const task of due) { const reason = dueReason(task, today); if (reason) counts[reason]++; }
  return counts;
}

/** The phrases both the header and the card use, in one order: "16 overdue, 5 slipped". */
export function dueParts(counts: DueCounts): string[] {
  return [
    counts.overdue && `${counts.overdue} overdue`,
    counts.due_today && `${counts.due_today} due today`,
    counts.slipped && `${counts.slipped} slipped`,
    counts.planned && `${counts.planned} planned for today`,
  ].filter((part): part is string => !!part);
}

/** Completions the owner made in Eridani today; Linear closures that synced in today do not count. */
export function doneToday(tasks: Task[], today: string, zone: string) {
  return finishedToday(tasks, today, zone).filter(task => task.status === "completed" && !finishedBySync(task)).length;
}
