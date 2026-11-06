import { describe, expect, it } from "vitest";
import { doneToday, dueCounts, dueParts, dueReason } from "./today-counts";
import { scheduledBy } from "./productivity";
import type { Task } from "./types";

const TODAY = "2026-11-05";
const task = (id: string, extra: Partial<Task> = {}) => ({
  id, title: id, status: "open", archived: false, due_date: null, planned_date: null, ...extra,
}) as Task;

describe("Today's due counts", () => {
  const due = [
    task("late", { due_date: "2026-11-01" }),
    task("late-and-planned", { due_date: "2026-11-02", planned_date: "2026-11-05" }),
    task("deadline-today", { due_date: TODAY, planned_date: "2026-11-01" }),
    task("slipped", { planned_date: "2026-11-03" }),
    task("slipped-future-deadline", { planned_date: "2026-11-04", due_date: "2026-11-20" }),
    task("planned", { planned_date: TODAY }),
  ];
  it("puts every task on the card in exactly one bucket, deadline first", () => {
    expect(due.every(t => scheduledBy(t, TODAY))).toBe(true);
    expect(due.map(t => dueReason(t, TODAY))).toEqual(["overdue", "overdue", "due_today", "slipped", "slipped", "planned"]);
    const counts = dueCounts(due, TODAY);
    expect(counts).toEqual({ overdue: 2, due_today: 1, slipped: 2, planned: 1 });
    expect(Object.values(counts).reduce((a, b) => a + b, 0)).toBe(due.length);
  });
  it("words the header and the card the same way", () => {
    expect(dueParts(dueCounts(due, TODAY))).toEqual(["2 overdue", "1 due today", "2 slipped", "1 planned for today"]);
    expect(dueParts({ overdue: 16, due_today: 0, slipped: 5, planned: 0 }).join(", ")).toBe("16 overdue, 5 slipped");
    expect(dueParts({ overdue: 0, due_today: 0, slipped: 0, planned: 0 })).toEqual([]);
  });
});

describe("the day ring's done count", () => {
  const at = "2026-11-05T15:00:00Z";
  it("counts completions made in Eridani today, not Linear closures that synced in", () => {
    const tasks = [
      task("mine", { status: "completed", finished_at: at, completed_at: at }),
      task("linked-but-mine", { status: "completed", finished_at: at, completed_at: at, external: { provider: "linear", finished_via: null } }),
      task("closed-in-linear", { status: "completed", finished_at: at, completed_at: at, external: { provider: "linear", finished_via: "linear" } }),
      task("canceled", { status: "cancelled", finished_at: at }),
      task("yesterday", { status: "completed", finished_at: "2026-11-04T15:00:00Z" }),
    ];
    expect(doneToday(tasks, TODAY, "UTC")).toBe(2);
  });
});
