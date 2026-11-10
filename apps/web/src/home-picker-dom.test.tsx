// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { HomePicker } from "./HomePicker";
import { groupByType, moveRecord, offersSection, pathOf, recentHomes, refusal, rememberHome, undoMove, type HomeRef } from "./home-picks";
import { api, command, post } from "./api";
import type { CustomRecord, Schema, SchemaType } from "./structure-types";

vi.mock("./api", async importOriginal => ({
  ...await importOriginal<typeof import("./api")>(), api: vi.fn(), post: vi.fn(), command: vi.fn(),
}));

const type = (id: string, name: string, capabilities: string[], parent_types: string[]): SchemaType =>
  ({ id, name, plural: name + "s", description: "", capabilities, parent_types, fields: [], statuses: [], archived: false });
const schema = { revision: 7, relationships: [], understandings: [], types: [
  type("space", "Space", [], []), type("client", "Client", [], ["space"]), type("project", "Project", ["work"], ["space", "client"]),
  type("task", "Task", ["work"], ["space", "client", "project", "task"]), type("note", "Note", ["content"], ["space", "client", "project", "task", "note"]),
] } as Schema;
const rec = (id: string, title: string, type_id: string, home: { id: string; title: string; type_id: string }[] = [], extra: Partial<CustomRecord> = {}) => ({
  id, title, type_id, type_name: schema.types.find(t => t.id === type_id)!.name, home, parent_id: home.at(-1)?.id ?? null, revision: 3,
  schema_revision: 7, body: "", values: {}, inherited: {}, status_id: null, archived: false, task_id: type_id === "task" ? id : null, note_id: null,
  capabilities: schema.types.find(t => t.id === type_id)!.capabilities, links: [], ...extra,
}) as CustomRecord;
const business = { id: "biz", title: "Business", type_id: "space" }, brainforge = { id: "bf", title: "Brainforge", type_id: "client" };
const row = rec("t1", "Send the invoice", "task", [business, brainforge]);

describe("home picker helpers", () => {
  it("names the path, refuses itself, its insides and types that can't hold it", () => {
    expect(pathOf(row)).toBe("Business › Brainforge");
    expect(refusal(schema, row, rec("t1", "Send the invoice", "task"))).toMatch(/inside/);
    expect(refusal(schema, row, rec("t2", "Sub", "task", [business, brainforge, { id: "t1", title: "Send the invoice", type_id: "task" }]))).toMatch(/inside/);
    expect(refusal(schema, row, rec("n1", "Notes", "note"))).toBe("A note can't hold a task.");
    expect(refusal(schema, row, rec("abc", "ABC", "client", [business]))).toBe("");
    expect(offersSection(schema, row, "work")).toBe(true);
    expect(offersSection(schema, row, "content")).toBe(false);
  });
  it("groups search results by type in order and keeps five recent homes, newest first", () => {
    expect(groupByType([rec("a", "A", "client"), rec("b", "B", "task"), rec("c", "C", "client")]).map(g => [g.type, g.rows.map(r => r.id)]))
      .toEqual([["Client", ["a", "c"]], ["Task", ["b"]]]);
    const h = (id: string): HomeRef => ({ id, title: id, type_id: "client", path: "" });
    let list: HomeRef[] = [];
    for (const id of ["a", "b", "c", "d", "e", "f", "b"]) list = rememberHome(list, h(id));
    expect(list.map(x => x.id)).toEqual(["b", "f", "e", "d", "c"]);
  });
  it("moves a record whole after a clean preview and undoes by command", async () => {
    vi.mocked(post).mockResolvedValue({ preview_hash: "hash", issues: [] });
    const send = vi.fn().mockResolvedValue({});
    vi.mocked(command).mockReturnValue({ id: "cmd-1", send });
    await expect(moveRecord(row, 7, "abc")).resolves.toBe("cmd-1");
    expect(post).toHaveBeenCalledWith("/structure/contents/preview", expect.objectContaining({ record_id: "t1", expected_revision: 3, schema_revision: 7, operation: "move", mode: "subtree", parent_id: "abc" }));
    expect(command).toHaveBeenCalledWith("record.contents", expect.objectContaining({ preview_hash: "hash", parent_id: "abc" }));
    await undoMove("cmd-1");
    expect(command).toHaveBeenLastCalledWith("record.restore_contents", { source_command_id: "cmd-1" });
    vi.mocked(post).mockResolvedValue({ preview_hash: "hash", issues: ["That home is archived."] });
    await expect(moveRecord(row, 7, "old")).rejects.toThrow("That home is archived.");
  });
});

describe("HomePicker", () => {
  let host: HTMLDivElement, root: Root;
  beforeEach(() => {
    (globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
    host = document.createElement("div"); document.body.append(host); root = createRoot(host);
    localStorage.clear();
    vi.mocked(api).mockImplementation(async (url: string) => {
      if (url.includes("/placement")) return { suggestion: { home_id: "abc", title: "ABC Home & Commercial", type_id: "client", confidence: "high", reason: "Mentions ABC.", at: null }, filed: null, pending: false };
      if (url.startsWith("/structure/browse") && url.includes("parent_id=biz")) return { items: [rec("bf", "Brainforge", "client", [business]), rec("abc", "ABC Home & Commercial", "client", [business])] };
      if (url.startsWith("/structure/browse")) return { items: [rec("biz", "Business", "space", [], { contents: { direct: 2 } as CustomRecord["contents"] })] };
      if (url.includes("capability=work")) return { items: [rec("t9", "Quarterly review", "task", [business])], next_offset: null };
      if (url.includes("query=")) return { items: [rec("abc", "ABC Home & Commercial", "client", [business]), rec("t9", "ABC follow-up", "task", [business, { id: "abc", title: "ABC Home & Commercial", type_id: "client" }]), rec("n1", "ABC notes", "note")] };
      throw new Error("unexpected " + url);
    });
  });
  afterEach(async () => { await act(async () => root.unmount()); host.remove(); vi.resetAllMocks(); });
  const panel = () => document.querySelector<HTMLElement>(".floating-panel.home-picker");
  const flush = () => act(async () => { await new Promise(r => setTimeout(r, 0)); });
  async function open(onChoose = vi.fn()) {
    await act(async () => root.render(<HomePicker row={row} schema={schema} disabled={false} onChoose={onChoose}/>));
    expect(host.textContent).toContain("Brainforge");
    await act(async () => host.querySelector<HTMLButtonElement>('[aria-label="Choose main home"]')!.click());
    await flush();
    return onChoose;
  }

  it("floats outside the card with the current path, a suggestion, the homes tree and collapsed Tasks", async () => {
    await open();
    expect(panel()).not.toBeNull();
    expect(host.contains(panel())).toBe(false);
    expect(panel()!.querySelector('[aria-label="Now"]')!.textContent).toContain("Business › Brainforge");
    expect(panel()!.querySelector('[aria-label="Suggested"]')!.textContent).toContain("ABC Home & Commercial");
    expect(panel()!.querySelector('[aria-label="Tasks"] .home-section-head')!.getAttribute("aria-expanded")).toBe("false");
    expect(panel()!.querySelector('[aria-label="Notes"]')).toBeNull();  // A note can't hold a task.
    expect(vi.mocked(api).mock.calls.some(([url]) => url.includes("capability="))).toBe(false);  // Collapsed: not loaded.
    const space = panel()!.querySelector<HTMLElement>("[role=treeitem]")!;
    expect(space.getAttribute("aria-expanded")).toBe("false");
    await act(async () => panel()!.querySelector<HTMLButtonElement>('[aria-label="Expand Business"]')!.click());
    await flush();
    expect(vi.mocked(api)).toHaveBeenCalledWith("/structure/browse?section=groups&limit=100&parent_id=biz");
    const current = Array.from(panel()!.querySelectorAll<HTMLElement>("[role=treeitem][aria-level=\"2\"]")).find(n => n.textContent?.includes("Brainforge"))!;
    expect(current.getAttribute("aria-selected")).toBe("true");
    await act(async () => panel()!.querySelector<HTMLButtonElement>('[aria-label="Tasks"] .home-section-head')!.click());
    await flush();
    expect(panel()!.querySelector('[aria-label="Tasks"]')!.textContent).toContain("Quarterly review");
  });

  it("searches everything, grouped by type with paths, and a choice remembers the home", async () => {
    const onChoose = await open();
    const search = panel()!.querySelector<HTMLInputElement>('input[type=search]')!;
    expect(document.activeElement).toBe(search);
    await act(async () => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!.call(search, "abc");
      search.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await act(async () => { await new Promise(r => setTimeout(r, 220)); });
    const groups = Array.from(panel()!.querySelectorAll<HTMLElement>(".home-results section")).map(s => s.getAttribute("aria-label"));
    expect(groups).toEqual(["Client", "Task", "Note"]);
    expect(panel()!.querySelector('[aria-label="Use ABC follow-up as home"]')!.textContent).toContain("Business › ABC Home & Commercial");
    expect(panel()!.querySelector<HTMLButtonElement>('[aria-label="Use ABC notes as home"]')!.disabled).toBe(true);
    await act(async () => panel()!.querySelector<HTMLButtonElement>('[aria-label="Use ABC Home & Commercial as home"]')!.click());
    expect(onChoose).toHaveBeenCalledWith("abc", { id: "abc", title: "ABC Home & Commercial", type_id: "client", path: "Business" });
    expect(panel()).toBeNull();
    expect(recentHomes().map(h => h.id)).toEqual(["abc"]);
  });

  it("removes the task from its home", async () => {
    const onChoose = await open();
    const remove = Array.from(panel()!.querySelectorAll("button")).find(b => b.textContent === "Remove from home")!;
    await act(async () => remove.click());
    expect(onChoose).toHaveBeenCalledWith(null, null);
  });
});
