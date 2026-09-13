import type { Task, Schedule, Project } from "./types";
export function dateKey(date: Date) {
  return date.toISOString().slice(0, 10);
}
export function monthDays(selected: string) {
  const first = new Date(selected.slice(0, 7) + "-01T12:00:00Z");
  const begin = new Date(first);
  begin.setUTCDate(1 - first.getUTCDay());
  return Array.from({ length: 42 }, (_, i) => {
    const day = new Date(begin);
    day.setUTCDate(begin.getUTCDate() + i);
    return dateKey(day);
  });
}
export function shiftMonth(selected: string, amount: number) {
  const date = new Date(selected.slice(0, 7) + "-01T12:00:00Z");
  date.setUTCMonth(date.getUTCMonth() + amount);
  return dateKey(date);
}
export function matchesStatus(status: string, filter: string) {
  return (
    filter === "all" ||
    (filter === "active"
      ? !["completed", "cancelled", "superseded", "suppressed"].includes(status)
      : status === filter)
  );
}
export function localDateTime(value: string, zone: string) {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: zone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
  }).formatToParts(new Date(value));
  const p = Object.fromEntries(parts.map((p) => [p.type, p.value]));
  return p.year + "-" + p.month + "-" + p.day + "T" + p.hour + ":" + p.minute;
}
export function scheduleProject(
  s: Schedule,
  tasks: Task[],
  projects: Project[],
) {
  const t = tasks.find((t) => t.id === s.task_id);
  return t?.project ?? projects.find((p) => p.id === s.project_id)?.name ?? "";
}

export type CalendarMode = "month" | "week" | "day";
export function shiftDate(day: string, amount: number) {
  const date = new Date(day + "T12:00:00Z");
  date.setUTCDate(date.getUTCDate() + amount);
  return dateKey(date);
}
export function weekDays(day: string) {
  const date = new Date(day + "T12:00:00Z");
  const first = shiftDate(day, -date.getUTCDay());
  return Array.from({ length: 7 }, (_, i) => shiftDate(first, i));
}
export function calendarRange(day: string, mode: CalendarMode) {
  const days =
    mode === "month" ? monthDays(day) : mode === "week" ? weekDays(day) : [day];
  return { days, from: days[0], end: shiftDate(days[days.length - 1], 1) };
}
