import { describe, expect, it } from "vitest";
import type { Schema, SchemaType } from "./structure-types";
import {
  addAllowedHome, allowAndPlace, alsoAllowedIn, diagramParent, fieldRole, fieldRoleLabel, recordsLivingIn, removeAllowedHome, typeMoveChoice,
} from "./blueprint-model";

const type = (id: string, parent_types: string[], extra: Partial<SchemaType> = {}): SchemaType => ({
  id, name: id[0].toUpperCase() + id.slice(1), plural: id[0].toUpperCase() + id.slice(1) + "s", description: id, capabilities: [], parent_types, fields: [], statuses: [], archived: false, ...extra,
});
const schema = (): Schema => ({
  revision: 1, relationships: [], understandings: [], field_library: [],
  types: [type("space", []), type("client", ["space"]), type("project", ["client", "space"]), type("task", ["project", "client", "space", "task"]), type("goal", ["space"])],
  type_layout: [
    { type_id: "space", parent_type_id: null }, { type_id: "client", parent_type_id: "space" }, { type_id: "project", parent_type_id: "client" },
    { type_id: "task", parent_type_id: "project" }, { type_id: "goal", parent_type_id: "space" },
  ],
});

describe("blueprint move choice", () => {
  it("enables only the rearrangement when the target is already an allowed home", () => {
    const c = typeMoveChoice(schema(), "task", "client");
    expect(c.rearrange.enabled).toBe(true);
    expect(c.allow.enabled).toBe(false);
    expect(c.allow.note).toBe("Already allowed");
  });
  it("offers to also allow a home that isn't allowed yet, without enabling a silent rearrangement", () => {
    const c = typeMoveChoice(schema(), "goal", "client");
    expect(c.rearrange).toEqual({ enabled: false, note: "Client isn't an allowed home for goals yet" });
    expect(c.allow).toMatchObject({ enabled: true, label: "Also allow goals inside clients" });
    const next = allowAndPlace(schema(), "goal", "client");
    expect(next.types.find(t => t.id === "goal")!.parent_types).toEqual(["space", "client"]);
    expect(diagramParent(next, "goal")).toBe("client");
  });
  it("refuses cycles in both options", () => {
    const c = typeMoveChoice(schema(), "client", "task");
    expect(c.cycle).toBe(true);
    expect(c.rearrange.enabled || c.allow.enabled).toBe(false);
    expect(typeMoveChoice(schema(), "task", "task").cycle).toBe(true);
  });
});

describe("allowed homes", () => {
  it("lists other allowed homes beyond the diagram parent and itself", () => {
    expect(alsoAllowedIn(schema(), "task").map(t => t.id)).toEqual(["client", "space"]);
    expect(alsoAllowedIn(schema(), "client")).toEqual([]);
  });
  it("adds and removes homes in the draft only", () => {
    const s = schema();
    expect(removeAllowedHome(s, "task", "space").types.find(t => t.id === "task")!.parent_types).toEqual(["project", "client", "task"]);
    expect(addAllowedHome(s, "goal", "client").types.find(t => t.id === "goal")!.parent_types).toEqual(["space", "client"]);
    expect(s.types.find(t => t.id === "task")!.parent_types).toHaveLength(4);
  });
  it("finds the records that removing a home would strand", () => {
    const records = [
      { id: "p", type_id: "project", parent_id: null }, { id: "t1", type_id: "task", parent_id: "p" }, { id: "t2", type_id: "task", parent_id: "t1" }, { id: "t3", type_id: "task", parent_id: null },
    ];
    expect(recordsLivingIn(records, "task", "task").map(r => r.id)).toEqual(["t2"]);
    expect(recordsLivingIn(records, "task", "project").map(r => r.id)).toEqual(["t1"]);
  });
  it("marks fields as operational or inheritable", () => {
    expect(fieldRoleLabel[fieldRole({ binding: "due_date", inherit: false })]).toBe("Operational, never inherited");
    expect(fieldRoleLabel[fieldRole({ binding: null, inherit: false })]).toBe("Can inherit from home");
    expect(fieldRole({ binding: null, inherit: true })).toBe("inherits");
  });
});
