import { describe, expect, it } from "vitest";
import { rangeView, rememberRange, type CalendarRangeData } from "./calendar-cache";
import { calendarRange } from "./workspace";
import type { CalendarEntry } from "./types";

const entry = (id: string, date: string): CalendarEntry => ({
  id, entity_id: id, kind: "task", title: id, date, at: null, status: "active",
  project_id: null, task_id: null, revision: 1, projected: false, notification_id: null,
});
const loaded = (mode: "month" | "week" | "day", day: string, items: CalendarEntry[], over: Partial<CalendarRangeData> = {}) => {
  const { from, end } = calendarRange(day, mode);
  return { from, end, zone: "America/Chicago", items, truncated: false, ...over };
};

describe("calendar range cache", () => {
  const month = loaded("month", "2026-11-05", [entry("a", "2026-11-03"), entry("b", "2026-11-05"), entry("c", "2026-11-20")]);

  it("draws a week or day inside an already-loaded month immediately", () => {
    const cache = rememberRange([], month);
    const week = calendarRange("2026-11-05", "week");
    const view = rangeView(cache, week.from, week.end, "America/Chicago");
    expect(view?.provisional).toBe(true);
    expect(view?.items.map((e) => e.id)).toEqual(["a", "b"]);
    const day = calendarRange("2026-11-05", "day");
    expect(rangeView(cache, day.from, day.end, "America/Chicago")?.items.map((e) => e.id)).toEqual(["b"]);
  });

  it("only shows a loading state for a range nothing loaded covers", () => {
    const cache = rememberRange([], month);
    const later = calendarRange("2027-02-10", "week");
    expect(rangeView(cache, later.from, later.end, "America/Chicago")).toBeNull();
    const week = calendarRange("2026-11-05", "week");
    expect(rangeView(cache, week.from, week.end, "Europe/London")).toBeNull();
  });

  it("prefers the newest load and replaces a refetched range", () => {
    const week = loaded("week", "2026-11-05", [entry("b", "2026-11-05"), entry("new", "2026-11-06")]);
    let cache = rememberRange(rememberRange([], month), week);
    expect(rangeView(cache, week.from, week.end, week.zone)).toMatchObject({ provisional: false, items: week.items });
    cache = rememberRange(cache, { ...month, items: [] });
    expect(cache).toHaveLength(2);
    expect(rangeView(cache, week.from, week.end, week.zone)?.items).toEqual([]);
    // A newer month that was cut short does not hide the week's own complete rows.
    cache = rememberRange(cache, { ...month, items: [], truncated: true });
    expect(rangeView(cache, week.from, week.end, week.zone)?.items).toEqual(week.items);
  });

  it("does not carry a month's too-many-items notice into a smaller range, but keeps source warnings", () => {
    const week = calendarRange("2026-11-05", "week");
    expect(rangeView([{ ...month, truncated: true }], week.from, week.end, month.zone)?.truncated).toBe(false);
    const warned = { ...month, truncated: true, warnings: [{ calendar: "Work", reason: "token_expired" }] };
    expect(rangeView([warned], week.from, week.end, month.zone)?.truncated).toBe(true);
  });
});
