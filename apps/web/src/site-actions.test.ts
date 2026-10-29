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
  it("accepts the remotea page Eri can name (ui_show, ui_workspace, ui_search)", () => {
    for (const action of [
      { id: "show", kind: "show", view: "remotea" },
      { id: "run", kind: "show", view: "remotea", entity_id: "0b6f0f43-6a39-4f4a-9d0e-2f1c1f0a7b11" },
      { id: "page", kind: "workspace", view: "remotea" },
    ])
      expect(() => validateSiteAction(actionSchema.parse(action), "today", emptyOrganization, "project")).not.toThrow();
  });
  it("scopes saved note lists to Notes", () => {
    const list = actionSchema.parse({id:"list",kind:"workspace",view:"notes",note_list_id:"uncategorized"});
    expect(() => validateSiteAction(list, "all", emptyOrganization, "project")).not.toThrow();
    expect(() => validateSiteAction({...list,view:"all"}, "all", emptyOrganization, "project")).toThrow();
  });
  it("accepts bounded custom-field patches and rejects arbitrary nesting", () => {
    expect(actionSchema.parse({id:"custom",kind:"editor",operation:"patch",changes:{values:{client:"abc",labels:["one","two"],active:true}}}).changes?.values).toEqual({client:"abc",labels:["one","two"],active:true});
    expect(()=>actionSchema.parse({id:"custom",kind:"editor",operation:"patch",changes:{values:{nested:{too:"deep"}}}})).toThrow();
  });
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


it("accepts exact calendar entries only for a bounded calendar detail command", () => {
  const action = actionSchema.parse({id: "exact", kind: "calendar", date: "2026-10-27",
    entity_id: "series", open_details: true, calendar_entry_id: "series:17"});
  expect(() => validateSiteAction(action, "all", emptyOrganization, "project")).not.toThrow();
  for (const invalid of [
    {...action, entity_id: undefined},
    {...action, open_details: false},
    {...action, kind: "show" as const},
  ]) expect(() => validateSiteAction(invalid, "all", emptyOrganization, "project")).toThrow();
  expect(() => actionSchema.parse({...action, calendar_entry_id: "x".repeat(301)})).toThrow();
});
