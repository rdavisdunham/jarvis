import { expect, test } from "vitest";
import { matchesTaskTab, initialView } from "./task-presets";
import { calendarKind, completableCalendarTask } from "./calendar-presentation";
import type { Task, CalendarEntry } from "./types";
const task = {
  id: "task",
  status: "open",
  archived: false,
  is_template: false,
  project_id: null,
  space_id: null,
  area_id: null,
  due_date: null,
  planned_date: null,
} as unknown as Task;
test("task tabs overlap and preserve floating dates across month/year boundaries", () => {
  const due = { ...task, due_date: "2026-12-31" };
  expect(matchesTaskTab(due, "today", "2027-01-01")).toBe(true);
  expect(matchesTaskTab(due, "week", "2027-01-01")).toBe(true);
  expect(matchesTaskTab(due, "inbox", "2027-01-01")).toBe(true);
  expect(
    matchesTaskTab(
      { ...task, planned_date: "2027-01-07" },
      "week",
      "2027-01-01",
    ),
  ).toBe(true);
  expect(
    matchesTaskTab(
      { ...task, planned_date: "2027-01-08" },
      "week",
      "2027-01-01",
    ),
  ).toBe(false);
  expect(
    matchesTaskTab({ ...task, space_id: "personal" }, "inbox", "2027-01-01"),
  ).toBe(false);
  expect(matchesTaskTab({ ...due, archived: true }, "all", "2027-01-01")).toBe(
    false,
  );
});
test("task links accept new tabs and existing bookmarked view names", () => {
  expect(initialView("?view=tasks&tab=week")).toBe("week");
  expect(initialView("?view=inbox")).toBe("inbox");
  expect(initialView("?view=calendar")).toBe("calendar");
  expect(initialView("?view=tasks&tab=unknown")).toBe("today");
  expect(initialView("?view=settings&google=connected")).toBe("settings");
});
test("completion belongs to actionable tasks, never appointments/blocks or future routine templates", () => {
  for (const kind of ["task", "reminder"] as const)
    expect(completableCalendarTask({ kind } as CalendarEntry, task)).toBe(true);
  for (const kind of ["event", "block", "google"] as const)
    expect(completableCalendarTask({ kind } as CalendarEntry, task)).toBe(
      false,
    );
  expect(
    completableCalendarTask(
      { kind: "routine", projected: true } as CalendarEntry,
      task,
    ),
  ).toBe(false);
  expect(
    completableCalendarTask(
      { kind: "routine", projected: false } as CalendarEntry,
      { ...task, is_template: true },
    ),
  ).toBe(false);
  expect(
    completableCalendarTask(
      { kind: "routine", projected: false } as CalendarEntry,
      task,
    ),
  ).toBe(true);
  expect(
    completableCalendarTask({ kind: "task" } as CalendarEntry, {
      ...task,
      status: "cancelled",
    }),
  ).toBe(false);
  expect(
    calendarKind({ kind: "task", timing: "planned" } as CalendarEntry).label,
  ).toContain("planned");
  expect(calendarKind({ kind: "google" } as CalendarEntry).label).toBe(
    "Google event",
  );
});
