import type { CalendarEntry, Task } from "./types";
export function calendarKind(event: CalendarEntry) {
  if (event.kind === "task")
    return {
      label: event.timing === "planned" ? "Task · planned" : "Task · deadline",
      icon: "task",
      hint: "Complete the task when the work is done.",
    };
  if (event.kind === "reminder")
    return {
      label: "Task reminder",
      icon: "reminder",
      hint: "An alert for a task. Completing the task closes its alerts.",
    };
  if (event.kind === "routine")
    return {
      label: "Repeating task",
      icon: "routine",
      hint: "Complete this occurrence; the routine continues.",
    };
  if (event.kind === "block")
    return {
      label: "Work block",
      icon: "block",
      hint: "Reserved work time. The linked task has its own completion.",
    };
  return {
    label: event.kind === "google" ? "Google event" : "Event",
    icon: "event",
    hint: "A calendar event, with no task checkbox.",
  };
}
export function completableCalendarTask(
  event: CalendarEntry,
  task: Task | null,
) {
  return (
    !!task &&
    !task.is_template &&
    !task.archived &&
    task.status !== "cancelled" &&
    (event.kind === "task" ||
      event.kind === "reminder" ||
      (event.kind === "routine" && !event.projected))
  );
}
export function dateLabel(date: string) {
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeZone: "UTC",
  }).format(new Date(date + "T12:00:00Z"));
}
