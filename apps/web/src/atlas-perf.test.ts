import { describe, expect, it } from "vitest";
import { chooseLabels, indexAtlas, layoutAtlas, viewFor, visibleNodes, type AtlasRecord, type AtlasType } from "./atlas-model";

/** Dev check: a 2,000-record workspace lays out, culls and labels well inside an interactive budget. */
const types: AtlasType[] = [
  { id: "space", name: "Space", plural: "Spaces", capabilities: [], parent_types: [] },
  { id: "client", name: "Client", plural: "Clients", capabilities: [], parent_types: ["space"] },
  { id: "project", name: "Project", plural: "Projects", capabilities: ["work"], parent_types: ["client"] },
  { id: "task", name: "Task", plural: "Tasks", capabilities: ["work"], parent_types: ["project", "task"] },
];
function synthetic(total: number): AtlasRecord[] {
  const out: AtlasRecord[] = [];
  const add = (type: string, parent: string | null, container: boolean) => {
    const id = type + out.length;
    out.push({ id, title: `${type} ${out.length}`, type_id: type, parent_id: parent, depth: 0, revision: 1, sort_order: out.length, opens_as: container ? "container" : "item",
      status_meaning: container ? null : ["open", "completed", "in_progress"][out.length % 3], work: !container, due_date: out.length % 7 ? null : "2026-10-01", planned_date: null });
    return id;
  };
  for (let s = 0; s < 3 && out.length < total; s++) {
    const space = add("space", null, true);
    for (let c = 0; c < 10 && out.length < total; c++) {
      const client = add("client", space, true);
      for (let p = 0; p < 6 && out.length < total; p++) {
        const project = add("project", client, true);
        for (let t = 0; t < 10 && out.length < total; t++) { const task = add("task", project, false); if (t % 4 === 0 && out.length < total) add("task", task, false); }
      }
    }
  }
  return out;
}

describe("atlas performance", () => {
  it("lays out 2,000 records in under 100 ms", () => {
    const records = synthetic(2000);
    expect(records.length).toBe(2000);
    layoutAtlas(indexAtlas({ records, types })); // warm up the JIT
    const started = performance.now();
    const index = indexAtlas({ records, types });
    const layout = layoutAtlas(index);
    const view = viewFor(layout.root);
    const nodes = visibleNodes(layout, view, 1100, 760);
    chooseLabels(nodes, layout.root, view, 1100, 760, null);
    const elapsed = performance.now() - started;
    expect(elapsed).toBeLessThan(100);
    // Semantic zoom keeps the top-level view far smaller than the workspace.
    expect(nodes.length).toBeLessThan(records.length);
    // A zoomed-in frame (culling + labels) stays cheap enough for 60 fps animation.
    const focus = layout.byId.get(records.find(r => r.type_id === "client")!.id)!;
    const frame = performance.now();
    for (let i = 0; i < 10; i++) { const v = viewFor(focus); chooseLabels(visibleNodes(layout, v, 1100, 760), focus, v, 1100, 760, null); }
    expect((performance.now() - frame) / 10).toBeLessThan(16);
  });
});
