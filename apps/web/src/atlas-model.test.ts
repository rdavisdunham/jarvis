import { describe, expect, it } from "vitest";
import {
  ROOT, chooseLabels, flyTarget, indexAtlas, itemState, layoutAtlas, legalTargets, markKind, markName, moveChoices, movesWhole, placedNode, scaleOf,
  searchRecords, trimLabel, viewFor, visibleNodes, type AtlasRecord, type AtlasType,
} from "./atlas-model";

const types: AtlasType[] = [
  { id: "space", name: "Space", plural: "Spaces", capabilities: [], parent_types: [], opens_as: "container" },
  { id: "client", name: "Client", plural: "Clients", capabilities: [], parent_types: ["space"], opens_as: "container" },
  { id: "project", name: "Project", plural: "Projects", capabilities: ["work", "timeline"], parent_types: ["client", "space"], opens_as: "container" },
  { id: "task", name: "Task", plural: "Tasks", capabilities: ["work"], parent_types: ["project", "client", "space", "task"], opens_as: "item" },
  { id: "note", name: "Note", plural: "Notes", capabilities: ["content"], parent_types: ["project", "client", "space", "task"], opens_as: "item" },
  { id: "goal", name: "Goal", plural: "Goals", capabilities: ["metric"], parent_types: ["space"], opens_as: "auto" },
];
let order = 0;
const rec = (id: string, type: string, parent: string | null, extra: Partial<AtlasRecord> = {}): AtlasRecord => ({
  id, title: extra.title ?? id, type_id: type, parent_id: parent, depth: 0, revision: 1, sort_order: order++,
  opens_as: ["space", "client", "project"].includes(type) ? "container" : "item", status_meaning: ["task", "project"].includes(type) ? "open" : null,
  work: ["task", "project"].includes(type), due_date: null, planned_date: null, ...extra,
});
const records = [
  rec("work", "space", null, { title: "Work" }), rec("abc", "client", "work", { title: "ABC Home & Commercial" }), rec("beacon", "client", "work", { title: "Beacon Dispatch" }),
  rec("ti", "project", "abc", { title: "Transcript Intelligence" }), rec("web", "project", "abc", { title: "Website refresh" }),
  rec("fcd", "task", "ti", { title: "Finish the central docs", status_meaning: "in_progress" }), rec("ce", "task", "fcd", { status_meaning: "completed" }), rec("wg", "task", "fcd"),
  rec("swd", "task", "ti", { due_date: "2026-10-01" }), rec("dn", "note", "ti"), rec("w1", "task", "web", { status_meaning: "completed" }),
  rec("g1", "goal", "work"), rec("cpr", "project", "beacon"),
];
const index = indexAtlas({ records, types, links: [] });

describe("legal targets", () => {
  it("follows allowed homes and excludes the record, its own subtree and its current home", () => {
    const fcd = index.byId.get("fcd")!;
    const legal = legalTargets(fcd, index);
    expect(legal.has("web")).toBe(true);
    expect(legal.has("abc")).toBe(true);
    expect(legal.has("swd")).toBe(true); // a task may live in another task
    expect(legal.has("ti")).toBe(false); // current home
    expect(legal.has("fcd")).toBe(false); // itself
    expect(legal.has("ce")).toBe(false); // own subtree
    expect(legal.has("dn")).toBe(false); // notes are not allowed homes for tasks
    expect(legal.has("g1")).toBe(false);
  });
  it("keeps a project out of its own descendants and out of disallowed types", () => {
    const legal = legalTargets(index.byId.get("ti")!, index);
    expect([...legal].sort()).toEqual(["beacon", "work"]);
  });
});

describe("move choices", () => {
  it("offers both contents choices when there are children, naming the old home", () => {
    const options = moveChoices(index.byId.get("fcd")!, index);
    expect(options.map(o => o.mode)).toEqual(["subtree", "item"]);
    expect(options[0].note).toBe("2 items inside move along");
    expect(options[1]).toEqual({ mode: "item", label: "Move only this record", note: "Its 2 items stay in Transcript Intelligence" });
  });
  it("moves leaves and tasks with subtasks without asking, but asks for containers with contents", () => {
    expect(movesWhole(index.byId.get("swd")!, index)).toBe(true); // a task with nothing inside
    expect(movesWhole(index.byId.get("dn")!, index)).toBe(true); // a note
    expect(movesWhole(index.byId.get("fcd")!, index)).toBe(true); // a task with subtasks
    expect(movesWhole(index.byId.get("ti")!, index)).toBe(false); // a project with contents
    expect(movesWhole(index.byId.get("abc")!, index)).toBe(false); // a client with contents
  });
  it("offers only a plain move for a leaf", () => {
    expect(moveChoices(index.byId.get("swd")!, index)).toEqual([{ mode: "subtree", label: "Move with contents", note: "Nothing inside to move" }]);
  });
});

describe("open-as decisions and marks", () => {
  const layout = layoutAtlas(index);
  it("lays containers out as regions and keeps items with children as items with moons", () => {
    expect(layout.byId.get("abc")!.region).toBe(true);
    const fcd = layout.byId.get("fcd")!;
    expect(fcd.region).toBe(false);
    expect(fcd.moons.map(m => m.id)).toEqual(["ce", "wg"]);
    expect(layout.byId.has("ce")).toBe(false);
    expect(placedNode(layout, index, "ce").id).toBe("fcd");
  });
  it("flies to a region itself, and to an item's nearest region with the item selected", () => {
    expect(flyTarget(layout, index, "abc")).toEqual({ focus: "abc", select: "abc" });
    expect(flyTarget(layout, index, "fcd")).toEqual({ focus: "ti", select: "fcd" });
    expect(flyTarget(layout, index, "ce")).toEqual({ focus: "ti", select: "ce" });
  });
  it("styles marks by kind and state", () => {
    expect(markKind(index.byId.get("dn")!, index.types.get("note"))).toBe("note");
    expect(markKind(index.byId.get("g1")!, index.types.get("goal"))).toBe("goal");
    expect(itemState(index.byId.get("swd")!, "2026-10-03")).toBe("overdue");
    expect(itemState(index.byId.get("fcd")!, "2026-10-03")).toBe("in_progress");
    expect(itemState(index.byId.get("ce")!, "2026-10-03")).toBe("done");
  });
  it("counts work in each subtree without the record's own status", () => {
    expect(index.progress.get("abc")).toEqual({ done: 2, total: 7 });
    expect(index.progress.get("fcd")).toEqual({ done: 1, total: 2 });
    expect(markName(index.byId.get("abc")!, index, "2026-10-03")).toBe("Client: ABC Home & Commercial, 2 of 7 done");
    expect(markName(index.byId.get("fcd")!, index, "2026-10-03")).toBe("Task: Finish the central docs, in progress, 2 items inside");
  });
});

describe("semantic zoom and the label budget", () => {
  const layout = layoutAtlas(index);
  it("shows only nodes large enough at the current zoom", () => {
    const whole = visibleNodes(layout, viewFor(layout.root), 800, 600);
    expect(whole.some(n => n.id === "work")).toBe(true);
    const zoomed = visibleNodes(layout, viewFor(layout.byId.get("ti")!), 800, 600);
    expect(zoomed.some(n => n.id === "fcd")).toBe(true);
    expect(zoomed.some(n => n.id === "cpr")).toBe(false); // off screen
    expect(scaleOf(viewFor(layout.byId.get("ti")!), 800, 600)).toBeGreaterThan(scaleOf(viewFor(layout.root), 800, 600));
  });
  it("labels the focused level's children first, always the selection, within budget and without overlap", () => {
    const focus = layout.byId.get("abc")!, view = viewFor(focus);
    const nodes = visibleNodes(layout, view, 800, 600);
    const labels = chooseLabels(nodes, focus, view, 800, 600, "swd");
    expect(labels.map(l => l.id)).toContain("ti");
    expect(labels.map(l => l.id)).toContain("swd");
    expect(labels.some(l => l.id === "abc")).toBe(false);
    expect(chooseLabels(nodes, focus, view, 800, 600, null, 1)).toHaveLength(1);
    const many = chooseLabels(visibleNodes(layout, viewFor(layout.root), 800, 600), layout.root, viewFor(layout.root), 800, 600, null);
    for (const a of many) for (const b of many) if (a !== b) expect(Math.abs(a.x - b.x) > 1 || Math.abs(a.y - b.y) > 1).toBe(true);
  });
  it("trims long titles and finds records by title", () => {
    expect(trimLabel("Transcript Intelligence", 10)).toBe("Transcrip…");
    expect(searchRecords(index, "web").map(r => r.id)).toEqual(["web"]);
    expect(searchRecords(index, "")).toEqual([]);
    expect(layout.root.id).toBe(ROOT);
  });
});

describe("unfiled records", () => {
  it("gather in one synthetic region that is never a move target", async () => {
    const { UNFILED } = await import("./atlas-model");
    const loose = indexAtlas({ records: [...records, rec("inbox1", "task", null), rec("inbox2", "note", null)], types, links: [] });
    const layout = layoutAtlas(loose);
    expect(layout.root.children.map(n => n.id)).toContain(UNFILED);
    expect(layout.byId.get("inbox1")!.parent!.id).toBe(UNFILED);
    expect(flyTarget(layout, loose, "inbox1")).toEqual({ focus: UNFILED, select: "inbox1" });
    expect(legalTargets(loose.byId.get("fcd")!, loose).has(UNFILED)).toBe(false);
    expect(markName(layout.byId.get(UNFILED)!.record!, loose, "2026-10-03")).toBe("Unfiled: 2 records without a home");
  });
});
