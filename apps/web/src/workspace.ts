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
    (filter === "open"
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
