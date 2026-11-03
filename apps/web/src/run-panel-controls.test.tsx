// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { RemoteaProvider, useRemoteaActions } from "./remotea-context";
import { RunPanel } from "./RunPanel";
import { setDurableScope } from "./api";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
let host: HTMLDivElement, root: Root;
const id = "11111111-1111-4111-8111-111111111111";
function Harness({ phone }: { phone: boolean }) {
  const actions = useRemoteaActions();
  return <div className="app-shell"><button onClick={() => actions.openRun(id)}>Open run</button><RunPanel phone={phone}/></div>;
}
const find = (label: string) => host.querySelector<HTMLButtonElement>(`[aria-label="${label}"]`)!;
beforeEach(() => {
  history.replaceState(null, "", "/"); sessionStorage.clear(); localStorage.clear(); setDurableScope("panel-controls:personal");
  host = document.createElement("div"); document.body.append(host); root = createRoot(host);
});
afterEach(() => { act(() => root.unmount()); host.remove(); vi.unstubAllGlobals(); });
it.each([false, true])("preserves the mounted drawer, draft and focus through collapse/reopen; close only changes the view (phone=%s)", async phone => {
  const writes: string[] = [];
  vi.stubGlobal("fetch", async (url: string, init: RequestInit = {}) => {
    if (init.method && init.method !== "GET") writes.push(String(url));
    if (String(url).includes("/drafts/")) return new Response(JSON.stringify({ value: null, revision: 0 }));
    return new Response(JSON.stringify({ id, task: { id: "task", title: "A long running task" }, state: "executing", stage: "executing", state_revision: 3,
      state_reason: "", generation: 1, plan_revision: 1, tool: "codex", model: "gpt-6.1-sol", surf: "", turn_count: 1,
      created_at: "2026-11-03T10:00:00Z", history: [], turns: [], commands: [], questions: [], direct_supported: true,
      automation: { held: false, revision: 0, held_at: null, reason: null, releasing: false } }));
  });
  await act(async () => root.render(<RemoteaProvider><Harness phone={phone}/></RemoteaProvider>));
  const opener = host.querySelector<HTMLButtonElement>("button")!; opener.focus();
  await act(async () => { opener.click(); });
  // Flush the lazy chunk once so subsequent assertions cover the real drawer.
  await act(async () => { await import("./Remotea"); await new Promise(resolve => setTimeout(resolve, 0)); });
  const drawer = host.querySelector(".run-view")!;
  const show = [...host.querySelectorAll<HTMLButtonElement>("button")].find(button => button.textContent === "Show chat")!;
  if (show) act(() => show.click());
  const input = host.querySelector<HTMLTextAreaElement>("textarea")!;
  act(() => { Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")!.set!.call(input, "Keep this unsent instruction"); input.dispatchEvent(new Event("input", { bubbles: true })); });
  const collapse = find("Collapse run panel");
  expect(collapse.getAttribute("aria-expanded")).toBe("true");
  expect(document.getElementById(collapse.getAttribute("aria-controls")!)).toBe(host.querySelector(".run-panel-body"));
  await act(async () => collapse.click());
  expect(host.querySelector(".run-panel-body")?.hasAttribute("hidden")).toBe(true);
  expect(host.querySelector(".run-view")).toBe(drawer);
  const expand = host.querySelector<HTMLButtonElement>(".run-panel-rail")!;
  expect(document.activeElement).toBe(expand);
  await act(async () => expand.click());
  expect(host.querySelector("textarea")).toBe(input);
  expect(input.value).toBe("Keep this unsent instruction");
  expect(document.activeElement).toBe(find("Close run"));
  await act(async () => find("Close run").click());
  expect(host.querySelector(".run-panel")).toBeNull();
  expect(document.activeElement).toBe(opener);
  expect(writes).toEqual([]);
});
