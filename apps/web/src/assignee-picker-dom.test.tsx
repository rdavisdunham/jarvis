// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AssigneePicker } from "./AssigneePicker";
import { assigneeOptions, currentAssignee, filterOptions, type AccountContext } from "./assignees";
import { TaskCardHostContext, type TaskCardHost } from "./task-card-host";
import { api } from "./api";
import type { Actor } from "./productivity";

vi.mock("./api", async importOriginal => ({ ...await importOriginal<typeof import("./api")>(), api: vi.fn() }));

const actor = (id: string, name: string, kind: Actor["kind"] = "person", archived = false): Actor => ({ id, name, kind, archived, revision: 1 });
const actors = [actor("a-owner", "owner"), actor("a-dani", "Dani", "agent"), actor("a-sreya", "Sreya"), actor("a-old", "Old bot", "agent", true)];
const me: AccountContext = { name: "Davis", accountId: "acc-1", workspaceId: null };

describe("assignee options", () => {
  it("offers me, the assistant and Unassigned in a personal workspace", () => {
    const options = assigneeOptions({ actors, account: me, members: null });
    expect(options.map(o => [o.label, o.actorId, o.name])).toEqual([
      ["Davis (me)", "a-owner", "owner"], ["Dani", "a-dani", "Dani"], ["Unassigned", null, "unassigned"],
    ]);
    // People from outside (a Linear assignee) are shown as the current value, not offered.
    expect(currentAssignee(options, { id: "a-sreya", name: "Sreya" }, actors)).toEqual({ option: null, label: "Sreya" });
    expect(currentAssignee(options, { id: "a-owner" }, actors).label).toBe("Davis (me)");
    expect(currentAssignee(options, { name: "unassigned" }, actors).option?.label).toBe("Unassigned");
    expect(filterOptions(options, "dav").map(o => o.label)).toEqual(["Davis (me)"]);
  });
  it("offers a shared workspace's active members, and the assistant by name before its actor exists", () => {
    const options = assigneeOptions({ actors: [actor("m1", "Davis"), actor("m2", "Jo")], account: { ...me, workspaceId: "ws" }, members: [
      { account_id: "acc-1", name: "Davis", role: "owner", active: true, actor_id: "m1" },
      { account_id: "acc-2", name: "Jo", role: "editor", active: true, actor_id: "m2" },
      { account_id: "acc-3", name: "Gone", role: "editor", active: false, actor_id: "m3" },
    ] });
    expect(options.map(o => o.label)).toEqual(["Davis (me)", "Jo", "Dani", "Unassigned"]);
    expect(options.find(o => o.label === "Dani")).toMatchObject({ actorId: null, name: "Dani" });
  });
});

describe("AssigneePicker", () => {
  let host: HTMLDivElement, root: Root;
  beforeEach(() => {
    (globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
    host = document.createElement("div"); document.body.append(host); root = createRoot(host);
  });
  afterEach(async () => { await act(async () => root.unmount()); host.remove(); vi.resetAllMocks(); });
  const card = (value: { id?: string; name?: string }, onChange = vi.fn(), account: AccountContext = me) =>
    <TaskCardHostContext.Provider value={{ account, actors } as TaskCardHost}><AssigneePicker value={value} onChange={onChange}/></TaskCardHostContext.Provider>;
  const panel = () => document.querySelector<HTMLElement>(".floating-panel.assignee-panel");

  it("searches, moves with the arrow keys and picks with Enter", async () => {
    const onChange = vi.fn();
    await act(async () => root.render(card({ id: "a-owner", name: "owner" }, onChange)));
    expect(host.textContent).toContain("Davis (me)");
    await act(async () => host.querySelector<HTMLButtonElement>('[aria-label="Change assignee"]')!.click());
    expect(host.contains(panel())).toBe(false);
    const search = panel()!.querySelector<HTMLInputElement>('input[aria-label="Find an assignee"]')!;
    expect(document.activeElement).toBe(search);
    expect(Array.from(panel()!.querySelectorAll("[role=option]")).map(o => [o.textContent, o.getAttribute("aria-selected")]))
      .toEqual([["Davis (me)", "true"], ["Dani", "false"], ["Unassigned", "false"]]);
    await act(async () => { search.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowDown", bubbles: true })); });
    expect(search.getAttribute("aria-activedescendant")).toBe(panel()!.querySelectorAll("[role=option]")[1].id);
    await act(async () => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!.call(search, "unas");
      search.dispatchEvent(new Event("input", { bubbles: true }));
    });
    expect(Array.from(panel()!.querySelectorAll("[role=option]")).map(o => o.textContent)).toEqual(["Unassigned"]);
    await act(async () => { search.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true })); });
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ label: "Unassigned", name: "unassigned", actorId: null }));
    expect(panel()).toBeNull();
  });

  it("shows a Linear assignee on the button without offering them", async () => {
    await act(async () => root.render(card({ id: "a-sreya", name: "Sreya" })));
    expect(host.textContent).toContain("Sreya");
    await act(async () => host.querySelector<HTMLButtonElement>('[aria-label="Change assignee"]')!.click());
    expect(panel()!.textContent).not.toContain("Sreya");
    expect(panel()!.querySelector("[aria-selected=true]")).toBeNull();
  });

  it("loads a shared workspace's members once", async () => {
    vi.mocked(api).mockResolvedValue({ members: [{ account_id: "acc-1", name: "Davis", role: "owner", active: true, actor_id: "a-owner" }], invitations: [] });
    await act(async () => root.render(card({ id: "a-owner" }, vi.fn(), { ...me, workspaceId: "ws-9" })));
    await act(async () => { await new Promise(r => setTimeout(r, 0)); });
    expect(api).toHaveBeenCalledWith("/accounts/members/ws-9");
    expect(host.textContent).toContain("Davis (me)");
  });
});
