// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { RemoteaDataContext, type RemoteaData } from "./remotea-context";
import { PlanApproval } from "./Remotea";
import { RunTaskDialog } from "./RunTaskDialog";
import { DRAFT_STORAGE_KEY, ROLES_UNAVAILABLE, type RunDetail } from "./remotea-model";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const ID = "11111111-1111-4111-8111-111111111111";
let host: HTMLDivElement, root: Root;
beforeEach(() => { host = document.createElement("div"); document.body.append(host); root = createRoot(host); localStorage.clear(); });
afterEach(() => { act(() => root.unmount()); host.remove(); vi.unstubAllGlobals(); });

/** A fetch that answers by method and path, and records every call. */
function stubFetch(answer: (method: string, url: string, body: unknown) => { status?: number; body: unknown }) {
  const calls: { method: string; url: string; body: unknown }[] = [];
  vi.stubGlobal("fetch", async (url: string, init: RequestInit = {}) => {
    const body = init.body ? JSON.parse(String(init.body)) : undefined, method = init.method ?? "GET";
    calls.push({ method, url, body });
    const out = answer(method, url, body);
    return new Response(JSON.stringify(out.body), { status: out.status ?? 200 });
  });
  return calls;
}
const settle = () => act(async () => { await new Promise(r => setTimeout(r, 0)); });
const byText = (selector: string, text: string) => [...host.querySelectorAll<HTMLElement>(selector)].find(e => e.textContent?.includes(text))!;
function type(el: HTMLTextAreaElement, value: string) {
  Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")!.set!.call(el, value);
  el.dispatchEvent(new Event("input", { bubbles: true }));
}

describe("the Run dialog's Start", () => {
  const data: RemoteaData = { enabled: true, status: { enabled: true, key: null, hosts: [], link_url: "", connect_command: "" }, activeRuns: [],
    runsByTask: new Map(), pending: [], cards: {}, error: "", panel: { run: null, collapsed: false }, start: null };
  const preflight = (roles: boolean) => ({ body: { target: { id: "x", name: "jarvis", repo: "/r" }, needs_target: false, need: null, problems: [],
    active_run: null, interview: false, harnesses: { claude: true, codex: true }, roles } });
  const open = async () => {
    act(() => root.render(<RemoteaDataContext.Provider value={data}><RunTaskDialog taskId="t1" onClose={() => {}} onStarted={() => {}}/></RemoteaDataContext.Provider>));
    await settle();
  };

  it("starts a hyperplan run, asks the preflight about the reviewer, and remembers the choices", async () => {
    const calls = stubFetch((method, url) => method === "POST" ? { body: { id: ID } } : preflight(true));
    await open();
    act(() => byText("label", "Hyperplan").querySelector("input")!.click());
    await settle();
    expect(calls.at(-1)!.url).toContain("&reviewer_harness=codex");
    act(() => byText("button", "Start").click());
    await settle();
    expect(calls.at(-1)).toMatchObject({ method: "POST", url: "/api/v1/remotea/runs",
      body: { task_id: "t1", harness: "claude", reviewer: { harness: "codex" }, rounds: 2, approve_final: true } });
    expect(JSON.parse(localStorage.getItem(DRAFT_STORAGE_KEY)!)).toMatchObject({ hyperplan: true, rounds: 2 });
  });

  it("starts classic and turns the roles off while the host cannot spawn them", async () => {
    localStorage.setItem(DRAFT_STORAGE_KEY, JSON.stringify({ hyperplan: true, worker: { harness: "claude", model: "sonnet" } }));
    const calls = stubFetch((method) => method === "POST" ? { body: { id: ID } } : preflight(false));
    await open();
    expect(host.textContent).toContain(ROLES_UNAVAILABLE);
    expect(host.querySelector<HTMLSelectElement>('select[aria-label="Worker"]')!.disabled).toBe(true);
    // The re-read for the classic draft no longer names the roles.
    expect(calls.at(-1)!.url).not.toContain("reviewer_harness");
    act(() => byText("button", "Start").click());
    await settle();
    expect(calls.at(-1)!.body).toEqual({ task_id: "t1", harness: "claude" });
  });

  it("shows a HOST_NO_ROLES refusal inline and offers a classic start", async () => {
    const calls = stubFetch((method) => method === "POST"
      ? { status: 409, body: { error: { code: "HOST_NO_ROLES", message: "Update remotea first.", data: {} } } } : preflight(true));
    await open();
    act(() => byText("label", "Hyperplan").querySelector("input")!.click());
    await settle();
    act(() => byText("button", "Start").click());
    await settle();
    expect(host.textContent).toContain("Update remotea first.");
    expect(host.querySelector('[role="alert"]')).toBeNull();
    expect(byText("label", "Hyperplan").querySelector("input")!.disabled).toBe(true);
    act(() => byText("button", "Start").click());
    await settle();
    expect(calls.at(-1)!.body).toEqual({ task_id: "t1", harness: "claude" });
  });
});

describe("the final-plan approval card", () => {
  const run = { id: ID, state: "needs_you", state_reason: "plan_approval", state_revision: 7, plan_revision: 2, reviews: [], roles: null,
    plan: { revision: 2, title: "Ship it", prd_text: "", steps: [], criteria: [] } } as unknown as RunDetail;

  it("posts Revise with the note, and Approve, at the reader's revision", async () => {
    const calls = stubFetch(() => ({ body: { id: ID } }));
    const done = vi.fn();
    act(() => root.render(<PlanApproval run={run} onDone={done}/>));
    act(() => type(host.querySelector("textarea")!, "Split step 2."));
    act(() => byText("button", "Revise with note").click());
    await settle();
    expect(calls[0]).toMatchObject({ method: "POST", url: `/api/v1/remotea/runs/${ID}/plan-review`,
      body: { action: "revise", note: "Split step 2.", expected_revision: 7 } });
    act(() => byText("button", "Approve").click());
    await settle();
    expect(calls[1].body).toEqual({ action: "approve", expected_revision: 7 });
    expect(done).toHaveBeenCalledTimes(2);
  });

  it("reloads like the other controls on REVISION_CONFLICT", async () => {
    stubFetch(() => ({ status: 409, body: { error: { code: "REVISION_CONFLICT", message: "This run changed.", data: {} } } }));
    const conflict = vi.fn();
    act(() => root.render(<PlanApproval run={run} onDone={() => {}} onConflict={conflict}/>));
    act(() => byText("button", "Approve").click());
    await settle();
    expect(conflict).toHaveBeenCalledOnce();
    expect(host.querySelector('[role="alert"]')).toBeNull();
  });
});
