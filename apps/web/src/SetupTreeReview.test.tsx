// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SetupTreeReview } from "./SetupTreeReview";
import { addNode, editNode, removeNode, treeImpact, type OrganizationProposal, type TreeNode } from "./setup-model";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const tree = (): TreeNode[] => [{ id: "s1", kind: "space", title: "Work", description: "", existing_id: "space-1", action: "keep", children: [
  { id: "a1", kind: "area", title: "Teaching", description: "Classes", action: "create", children: [
    { id: "p1", kind: "project", title: "Lab safety", description: "", action: "create", children: [] }] },
  { id: "a2", kind: "area", title: "Old club", description: "", existing_id: "area-9", action: "archive", children: [] }] }];
const proposal = (nodes = tree(), id = "prop-1"): OrganizationProposal => ({ id, tree: nodes, preview_hash: "h".repeat(64),
  applied_at: null, expires_at: "2026-11-09T00:00:00Z",
  impact: { create: { space: 0, area: 1, goal: 0, project: 1 }, rename: 0, archive: 0, keep: 2, issues: [], blocking_count: 0 } });

describe("tree helpers", () => {
  it("describes what will be created, renamed and archived", () => {
    expect(treeImpact(tree())).toBe("1 area and 1 project will be created. Nothing is created until you confirm.");
    const renamed = editNode(tree(), "s1", { title: "Job" });
    expect(renamed[0].action).toBe("rename");
    const ticked = editNode(renamed, "a2", { archive_confirmed: true });
    expect(treeImpact(ticked)).toBe("1 area and 1 project will be created. 1 record will be renamed. 1 record will be archived. Nothing is created until you confirm.");
    expect(treeImpact(removeNode(tree(), "a1"))).toBe("Nothing new will be created. Nothing is created until you confirm.");
    expect(addNode(tree(), "a1", "goal")[0].children[0].children.at(-1)?.kind).toBe("goal");
  });
});

describe("the proposed organization review", () => {
  let host: HTMLDivElement, root: Root;
  beforeEach(() => { host = document.createElement("div"); document.body.append(host); root = createRoot(host); });
  afterEach(() => { act(() => root.unmount()); host.remove(); vi.unstubAllGlobals(); });
  const button = (name: string) => [...host.querySelectorAll("button")].find(b => b.textContent === name || b.getAttribute("aria-label") === name)!;

  it("shows existing rows, leaves archive unticked, and re-previews edits before creating", async () => {
    const calls: { tool: string; arguments: Record<string, unknown> }[] = [];
    vi.stubGlobal("fetch", async (_url: string, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body));
      calls.push(body);
      const data = body.tool === "setup.tree_preview" ? proposal(body.arguments.tree, "prop-2") : { created_count: 2, changed_count: 0 };
      return new Response(JSON.stringify({ data, command_id: body.command_id }));
    });
    const applied: unknown[] = [];
    await act(async () => { root.render(<SetupTreeReview proposal={proposal()} onClose={() => {}} onKeepTalking={() => {}}
      onByHand={() => {}} onApplied={result => applied.push(result)}/>); });
    expect(host.textContent).toContain("Already exists");
    expect(host.querySelector<HTMLInputElement>(".setup-tree-archive input")!.checked).toBe(false);
    expect(host.querySelector("[role=status]")!.textContent).toContain("1 area and 1 project will be created");
    const title = host.querySelector<HTMLInputElement>("input[aria-label='Project title']")!;
    await act(async () => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!.call(title, "Lab safety module");
      title.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await act(async () => { button("Create this organization").click(); });
    expect(calls.map(c => c.tool)).toEqual(["setup.tree_preview", "setup.tree_apply"]);
    expect(JSON.stringify(calls[0].arguments)).toContain("Lab safety module");
    expect(calls[1].arguments).toEqual({ proposal_id: "prop-2", preview_hash: "h".repeat(64) });
    expect(applied).toEqual([{ created_count: 2, changed_count: 0 }]);
    expect(button("Keep talking")).toBeTruthy();
    expect(button("Fill in by hand")).toBeTruthy();
  });

  it("applies an unedited preview directly and blocks one with issues", async () => {
    const tools: string[] = [];
    vi.stubGlobal("fetch", async (_url: string, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body)); tools.push(body.tool);
      return new Response(JSON.stringify({ data: { created_count: 2, changed_count: 0 }, command_id: body.command_id }));
    });
    await act(async () => { root.render(<SetupTreeReview proposal={proposal()} onClose={() => {}} onKeepTalking={() => {}}
      onByHand={() => {}} onApplied={() => {}}/>); });
    await act(async () => { button("Create this organization").click(); });
    expect(tools).toEqual(["setup.tree_apply"]);
    const blocked = proposal();
    blocked.impact = { ...blocked.impact, issues: [{ node_id: "p1", message: "Another project is already called \"Lab safety\"." }], blocking_count: 1 };
    await act(async () => { root.render(<SetupTreeReview key="b" proposal={blocked} onClose={() => {}} onKeepTalking={() => {}}
      onByHand={() => {}} onApplied={() => {}}/>); });
    expect(host.textContent).toContain("Another project is already called");
    expect(button("Create this organization").disabled).toBe(true);
  });
});
