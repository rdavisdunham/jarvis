import { describe, it, expect } from "vitest";
import { actionSchema } from "./site-actions";
import { validateSiteAction } from "./site-validation";
import { emptyOrganization } from "./productivity";
import {
  dayOffset,
  timelineWindow,
  matchesTaskFilters,
  emptyTaskFilters,
  sortTasks,
} from "./work-views";
import type { Task } from "./types";
describe("site control contract", () => {
  it("rejects the retired private-chat control", () => {
    expect(() => actionSchema.parse({ id: "private", kind: "device", private_chat: true })).toThrow();
  });
  it("keeps day view and rejects invalid dates and executable extras", () => {
    expect(
      actionSchema.parse({
        id: "x",
        kind: "calendar",
        date: "2028-02-29",
        calendar_view: "day",
      }).calendar_view,
    ).toBe("day");
    for (const date of ["2026-02-29", "2026-13-01"])
      expect(() =>
        actionSchema.parse({ id: "x", kind: "calendar", date }),
      ).toThrow();
    expect(() =>
      actionSchema.parse({
        id: "x",
        kind: "show",
        view: "all",
        script: "alert(1)",
      }),
    ).toThrow();
  });
  it("rejects unsupported combinations before UI state changes", () => {
    for (const action of [
      { kind: "filter", view: "notes", status: "completed" },
      { kind: "workspace", view: "calendar", layout: "board" },
      {
        kind: "workspace",
        view: "organize",
        organization_tab: "goal",
        layout: "timeline",
      },
      { kind: "workspace", view: "all", show_archived: true },
      { kind: "editor", operation: "save", changes: { title: "unrequested" } },
      { kind: "chat", mode: "open", query: "not used" },
    ])
      expect(() =>
        validateSiteAction(
          actionSchema.parse({ id: "x", ...action }),
          "all",
          emptyOrganization,
          "project",
        ),
      ).toThrow();
  });
  it("compares calendar days across DST without changing spacing", () => {
    expect(dayOffset("2026-03-07", "2026-03-10")).toBe(3);
    expect(dayOffset("2026-10-31", "2026-11-03")).toBe(3);
    expect(timelineWindow("2028-02-28", 14).slice(0, 3)).toEqual([
      "2028-02-28",
      "2028-02-29",
      "2028-03-01",
    ]);
  });
  it("filters exact tags and inclusive due dates, sorts without mutating records", () => {
    const a = {
      id: "a",
      title: "A",
      tags: ["launch"],
      due_date: "2030-01-05",
      due_time: "12:00",
      priority: 3,
      assignee: "Eri",
      assignee_id: "eri",
    } as Task;
    const b = { ...a, id: "b", title: "B", due_date: "2030-01-04" };
    expect(
      matchesTaskFilters(a, {
        ...emptyTaskFilters,
        tag: "launch",
        assignee: "eri",
        due_from: "2030-01-05",
        due_through: "2030-01-05",
      }),
    ).toBe(true);
    expect(matchesTaskFilters(a, { ...emptyTaskFilters, tag: "laun" })).toBe(
      false,
    );
    const input = [a, b];
    expect(sortTasks(input, "due").map((t) => t.id)).toEqual(["b", "a"]);
    expect(input[0]).toBe(a);
  });
});
