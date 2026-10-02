import type { CalendarEntry, Task } from "./types";
export function calendarKind(event: CalendarEntry) {
  if (event.kind === "task")
    return {
      label: event.timing === "planned" ? "Task (planned)" : "Task (deadline)",
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

/** Hours and minutes since local midnight of an instant in a time zone. */
export function minutesInZone(value: string | Date, zone: string) {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: zone,
    hour: "numeric",
    minute: "numeric",
    hourCycle: "h23",
  }).formatToParts(typeof value === "string" ? new Date(value) : value);
  const get = (type: string) =>
    Number(parts.find((part) => part.type === type)?.value ?? 0);
  return (get("hour") % 24) * 60 + get("minute");
}
export function clockLabel(value: string, zone: string) {
  return new Intl.DateTimeFormat(undefined, {
    timeZone: zone,
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(value));
}
/** True for entries that occupy a span of time rather than a single moment. */
export function spansTime(event: CalendarEntry) {
  return ["google", "event", "block"].includes(event.kind);
}
export type Placed<T> = { item: T; start: number; end: number; lane: number; lanes: number };
/** Side-by-side lanes for overlapping timed items (minutes since midnight). */
export function layoutLanes<T>(items: { item: T; start: number; end: number }[]) {
  const sorted = [...items].sort((a, b) => a.start - b.start || b.end - a.end);
  const placed: Placed<T>[] = [];
  let cluster: Placed<T>[] = [],
    laneEnds: number[] = [],
    clusterEnd = -1;
  const close = () => {
    for (const entry of cluster) entry.lanes = laneEnds.length;
    cluster = [];
    laneEnds = [];
  };
  for (const entry of sorted) {
    if (entry.start >= clusterEnd) close();
    let lane = laneEnds.findIndex((end) => end <= entry.start);
    if (lane < 0) lane = laneEnds.push(entry.end) - 1;
    else laneEnds[lane] = entry.end;
    const next = { ...entry, lane, lanes: 1 };
    cluster.push(next);
    placed.push(next);
    clusterEnd = Math.max(clusterEnd, entry.end);
  }
  close();
  return placed;
}

export type Cascaded<T> = { item: T; start: number; end: number; left: number; width: number; depth: number; shared: boolean; overlaps: boolean };
/** Calendar-style placement for overlapping timed items (minutes since midnight). Items that start
 * within `together` minutes of each other share a row side by side; an item that starts later
 * cascades over the earlier ones, indented by `indent` percent per level, so every title keeps
 * most of the column width instead of shrinking to a sliver. Returns left/width in percent. */
export function layoutCascade<T>(items: { item: T; start: number; end: number }[], indent = 12, together = 30) {
  const sorted = [...items].sort((a, b) => a.start - b.start || b.end - a.end);
  type Group = { start: number; end: number; depth: number; members: Cascaded<T>[] };
  const groups: Group[] = [];
  const placed: Cascaded<T>[] = [];
  for (const entry of sorted) {
    const current = groups.at(-1);
    const next = { ...entry, left: 0, width: 100, depth: 0, shared: false, overlaps: false };
    if (current && entry.start < current.end && entry.start - current.start < together) {
      current.members.push(next);
      current.end = Math.max(current.end, entry.end);
    } else {
      // Depth = how many earlier groups are still running when this one starts.
      const active = groups.filter((g) => g.end > entry.start);
      const depth = active.length ? Math.max(...active.map((g) => g.depth)) + 1 : 0;
      groups.push({ start: entry.start, end: entry.end, depth, members: [next] });
    }
    placed.push(next);
  }
  for (const group of groups) {
    const left = Math.min(group.depth * indent, 60);
    const overlaps = group.members.length > 1 || groups.some((g) => g !== group && g.start < group.end && group.start < g.end);
    const share = (100 - left) / group.members.length;
    group.members.forEach((member, index) => {
      member.depth = group.depth;
      member.left = left + index * share;
      member.width = share;
      member.shared = group.members.length > 1;
      member.overlaps = overlaps;
    });
  }
  return placed;
}
