import { describe, expect, it } from "vitest";
import { emptyOrganization, emptyFilter, matchesOrganization, scheduledBy } from "./productivity";

describe("productivity views", () => {
  it("includes planned work without moving its deadline", () => {
    const task = { planned_date: "2030-01-02", due_date: "2030-01-10" };
    expect(scheduledBy(task, "2030-01-01")).toBe(false);
    expect(scheduledBy(task, "2030-01-02")).toBe(true);
    expect(task.due_date).toBe("2030-01-10");
    expect(scheduledBy({ planned_date: null, due_date: null }, "2030-01-02")).toBe(false);
  });
  it("resolves a project home and multiple supporting goals", () => {
    const org = { ...emptyOrganization, projects: [
      { id: "p", name: "Project", revision: 1, description: "", archived: false, space_id: "business", area_id: "operations", goal_ids: ["g1", "g2"] },
    ] };
    const task = { project_id: "p", space_id: "old" };
    expect(matchesOrganization(task, { ...emptyFilter, space: "business", goal: "g2" }, org)).toBe(true);
    expect(matchesOrganization(task, { ...emptyFilter, space: "personal" }, org)).toBe(false);
    expect(matchesOrganization({ space_id: "personal" }, { ...emptyFilter, space: "personal" }, org)).toBe(true);
    expect(matchesOrganization({ space_id: "personal" }, { ...emptyFilter, goal: "g1" }, org)).toBe(false);
  });
});
