// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { RunDrawer } from "./Remotea";
import { setDurableScope } from "./api";
import { chatByDefault, conversationEntries } from "./RunPresentation";
import type { RunDetail } from "./remotea-model";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
let host: HTMLDivElement, root: Root, current: RunDetail;
let posts: { url: string; body: Record<string, unknown> }[];
let offline = false;
const fixture = (over: Partial<RunDetail> = {}): RunDetail => ({ id: "presentation-run", task: { id: "task", title: "Implement the widgets", status: "in_progress" },
  state: "executing", stage: "executing", state_revision: 3, state_reason: "", reason_text: "", state_since: "2026-11-03T10:00:00Z", generation: 1, plan_revision: 1,
  tool: "codex", model: "gpt-6.1-sol", branch: "task/widgets", base_branch: "main", worktree: "/fixture/worktree", effective_permission_mode: "", turn_count: 1, turn_cap: 0,
  started_at: null, finished_at: null, stop_requested_at: null, stop_mode: "", surf: "surface", host: { id: "host", name: "Fixture", connected: true },
  execution_mode: "managed", direct_supported: true, structured_answers: true, session: null, created_at: "2026-11-03T10:00:00Z", history: [], turns: [], commands: [], questions: [],
  automation: { held: false, revision: 0, held_at: null, reason: null, releasing: false },
  plan: { revision: 1, title: "Widget plan", approval_status: "approved", prd_text: "Full detailed plan", criteria: [], steps: [
    { id: "one", title: "Investigate widgets", status: "done", phase: "Investigate" }, { id: "two", title: "Implement focused fix", status: "todo", note: "Preserve current behavior", phase: "Fix" }] }, ...over });
const button = (label: string) => [...host.querySelectorAll<HTMLButtonElement>("button")].find(node => node.textContent === label)!;
const composer = () => host.querySelector<HTMLTextAreaElement>('textarea[aria-label="Message the agent"]');
async function mount() { await act(async () => { root.render(<RunDrawer id={current.id} preloaded={current} onClose={() => {}}/>); }); }
async function refresh() { await act(async () => { window.dispatchEvent(new Event("eri-agents-changed")); await new Promise(resolve => setTimeout(resolve, 320)); }); }
function type(value: string) { const input = composer()!; Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")!.set!.call(input, value); input.dispatchEvent(new Event("input", { bubbles: true })); }
beforeEach(() => {
  localStorage.clear(); setDurableScope("presentation-owner:personal"); current = fixture(); posts = []; offline = false;
  host = document.createElement("div"); document.body.append(host); root = createRoot(host);
  vi.stubGlobal("fetch", async (url: string, init: RequestInit = {}) => {
    if (String(url).includes("/drafts/")) return new Response(JSON.stringify({ value: null, revision: 0 }));
    if (init.method === "POST") {
      posts.push({ url, body: JSON.parse(String(init.body)) });
      if (offline) throw new TypeError("offline fixture");
      if (url.endsWith("/message")) current = { ...current, automation: { held: true, revision: 1, held_at: new Date().toISOString(), reason: "direct_message", releasing: false } };
      if (url.endsWith("/automation/resume")) current = { ...current, automation: { ...current.automation!, revision: current.automation!.revision + 1, releasing: true } };
      return new Response(JSON.stringify({ ...current, receipt: { status: "saved", terminal: true } }));
    }
    return new Response(JSON.stringify(current));
  });
});
afterEach(() => { act(() => root.unmount()); host.remove(); vi.unstubAllGlobals(); });

describe("drawer view and automation boundaries", () => {
  it("defaults approved work to the Run section with a truthful stepper; Show/Hide chat sends no control", async () => {
    await mount(); expect(composer()).toBeNull(); expect(host.querySelectorAll(".run-stage")).toHaveLength(3);
    const toggle = (name: string) => host.querySelector(`.run-stage-${name} .run-stage-toggle`)!;
    expect([toggle("plan"), toggle("run"), toggle("verify")].map(t => t.getAttribute("aria-expanded"))).toEqual(["false", "true", "false"]);
    expect(host.querySelector(".run-step.is-next")?.textContent).toContain("Implement focused fix");
    expect(host.querySelector('.run-step[aria-current="step"]')).toBeNull();
    expect([...host.querySelectorAll(".run-step-label")].map(l => l.textContent)).toEqual(["1a", "2a"]);
    expect(toggle("run").textContent).toContain("1 of 2 steps done, next is step 2a");
    expect(host.querySelector(".run-ring")?.getAttribute("aria-label")).toBe("50% of steps done");
    act(() => button("Show chat").click()); expect(composer()).not.toBeNull(); expect(host.querySelectorAll(".run-stage")).toHaveLength(0);
    expect(host.querySelector(".run-stage-track")).not.toBeNull();
    act(() => button("Hide chat").click()); expect(composer()).toBeNull(); expect(posts).toEqual([]);
  });
  it("sends Approved literally in direct mode and holds automation without pausing the worker", async () => {
    await mount(); act(() => button("Show chat").click()); act(() => type("Approved"));
    await act(async () => button("Send").click());
    expect(posts).toHaveLength(1); expect(posts[0]).toMatchObject({ url: "/api/v1/remotea/runs/presentation-run/message", body: { mode: "direct", text: "Approved", action_id: expect.any(String) } });
    expect(posts.some(post => post.url.includes("/decisions/") || post.url.endsWith("/pause"))).toBe(false);
    expect(host.querySelector(".run-drawer-status")!.textContent).toContain("Automation held");
    expect(composer()).not.toBeNull(); expect(current.paused).toBeUndefined();
  });
  it("keeps chat open while release is pending and collapses only on newer server-confirmed release", async () => {
    current = fixture({ automation: { held: true, revision: 4, held_at: "2026-11-03T10:00:00Z", reason: "direct_message", releasing: false } });
    await mount(); act(() => button("Show chat").click());
    await act(async () => button("Resume automation").click());
    expect(posts[0].body).toMatchObject({ expected_automation_revision: 4 }); expect(composer()).not.toBeNull();
    expect(host.textContent).toContain("Reconciling before automation resumes");
    current = { ...current, automation: { ...current.automation!, held: false, releasing: false } };
    await refresh(); expect(composer()).toBeNull(); expect(button("Show chat")).toBeTruthy();
  });
  it("retains open chat and held state when resume cannot reach the API", async () => {
    current = fixture({ automation: { held: true, revision: 2, held_at: "2026-11-03T10:00:00Z", reason: "direct_message", releasing: false } });
    await mount(); act(() => button("Show chat").click()); offline = true;
    await act(async () => button("Resume automation").click());
    expect(composer()).not.toBeNull(); expect(host.querySelector(".run-drawer-status")!.textContent).toContain("Automation held");
    expect(host.textContent).toContain("Delivery unconfirmed");
  });
  it("keeps pending decisions visible without opening chat or releasing a hold", async () => {
    current = fixture({ automation: { held: true, revision: 1, held_at: null, reason: "direct_message", releasing: false }, decisions: [{ id: "control", run_id: "presentation-run", kind: "control", revision: 1, status: "pending", actions: ["confirm", "decline"], payload: { summary: "Send the requested instructions" }, generation: 1, plan_revision: 1 }] });
    await mount(); expect(composer()).toBeNull(); expect(button("Approve once")).toBeTruthy();
    await act(async () => button("Approve once").click()); expect(posts[0].url).toContain("/decisions/control/respond");
    expect(current.automation!.held).toBe(true); expect(composer()).toBeNull();
    expect(host.querySelector(".run-drawer-decisions .run-delivery")?.textContent).toBe("Applied by Eridani");
  });
  it("collapses a planning view at approval while preserving execution Show chat across polls", async () => {
    current = fixture({ state: "planning", stage: "planning", plan_pending: true }); await mount(); expect(composer()).not.toBeNull();
    act(() => button("Hide chat").click()); act(() => button("Show chat").click());
    current = { ...current, state: "executing", stage: "executing", plan_pending: false }; await refresh(); expect(composer()).toBeNull();
    act(() => button("Show chat").click()); await refresh(); expect(composer()).not.toBeNull();
  });
});

it("keeps distinct published commentary and replaces duplicated final snapshots using provider identities", () => {
  const base = { run_id: "presentation-run", generation: 1, surf: "surface", incarnation: 1, session_id: "provider", turn_id: "turn", sequence: 1, at: "2026-11-03T10:00:00Z", complete: true, truncated: false };
  const run = fixture({ live_messages: [{ ...base, message_id: "comment", kind: "commentary", text: "Reading the public files" }, { ...base, message_id: "answer", kind: "commentary", text: "Implemented the fix" }],
    turns: [{ id: "final", seq: 1, text: "Implemented the fix", clipped: false, truncated: false, generation: 1, at: base.at, prompt: "", provider_session_id: "provider", provider_turn_id: "turn" }] });
  expect(conversationEntries(run).map(entry => entry.detail)).toEqual(["Reading the public files", "Implemented the fix"]);
  expect(chatByDefault(fixture({ plan: null, execution_mode: "legacy", freestyle: true }))).toBe(true);
});

it("uses one literal direct action for Enter and preserves Shift+Enter/IME composition", async () => {
  await mount(); act(() => button("Show chat").click()); act(() => type("An explicit instruction"));
  act(() => composer()!.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", shiftKey: true, bubbles: true, cancelable: true })));
  act(() => composer()!.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", isComposing: true, bubbles: true, cancelable: true })));
  expect(posts).toHaveLength(0);
  await act(async () => composer()!.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true, cancelable: true })));
  expect(posts).toHaveLength(1); expect(posts[0].body).toMatchObject({ text: "An explicit instruction", mode: "direct" });
  expect(composer()!.value).toBe("");
});

it("does not fall back to a queued message when the host lacks direct support", async () => {
  current = fixture({ direct_supported: false }); await mount(); act(() => button("Show chat").click());
  expect(composer()!.disabled).toBe(true); expect(button("Send").disabled).toBe(true);
  expect(composer()!.placeholder).toContain("Direct messages are unavailable"); expect(posts).toHaveLength(0);
});

it("keeps a visible load error and retry action when there is no run yet", async () => {
  vi.stubGlobal("fetch", async () => new Response(JSON.stringify({ error: { code: "UNAVAILABLE", message: "Fixture unavailable" } }), { status: 503 }));
  await act(async () => root.render(<RunDrawer id="unavailable" onClose={() => {}}/>));
  expect(host.querySelector('[role="alert"]')?.textContent).toBeTruthy(); expect(button("Retry loading run")).toBeTruthy();
});

it("upgrades a settled runtime receipt from server message deliveries without another Send", async () => {
  await mount(); act(() => button("Show chat").click()); act(() => type("One direct instruction"));
  await act(async () => button("Send").click());
  const action_id = String(posts[0].body.action_id);
  current = { ...current, message_deliveries: [{ action_id, op_id: "direct-op", text: "One direct instruction", status: "delivered", stage: "handed_to_harness" }] };
  await refresh(); expect(host.querySelector(".run-delivery")?.textContent).toContain("provider acknowledgement pending");
  current = { ...current, message_deliveries: [{ ...current.message_deliveries![0], stage: "provider_acknowledged" }] };
  await refresh(); expect(host.querySelector(".run-delivery")?.textContent).toBe("Provider acknowledged the message");
  expect(posts).toHaveLength(1);
});


it("keeps distinct response messages from a completed provider turn and removes only its final duplicate", () => {
  const base = { run_id: "presentation-run", generation: 1, surf: "surface", incarnation: 1, session_id: "provider", turn_id: "completed-turn", sequence: 1, at: "2026-11-03T10:00:00Z", complete: true, truncated: false, kind: "response" as const };
  const run = fixture({ live_messages: [{ ...base, message_id: "earlier", text: "The first check found an issue" }, { ...base, message_id: "later", text: "I am correcting the affected widget" }, { ...base, message_id: "final-snapshot", text: "The fix is complete" }],
    turns: [{ id: "final", seq: 1, text: "The fix is complete", clipped: false, truncated: false, generation: 1, at: base.at, prompt: "", provider_session_id: "provider", provider_turn_id: base.turn_id }] });
  expect(conversationEntries(run).map(entry => entry.detail)).toEqual(["The first check found an issue", "I am correcting the affected widget", "The fix is complete"]);
});

it("opens chat at newest output and preserves a reader's scroll-up on later snapshots", async () => {
  await mount();
  const middle = host.querySelector<HTMLElement>(".run-drawer-middle")!;
  Object.defineProperty(middle, "scrollHeight", { configurable: true, value: 1000 });
  Object.defineProperty(middle, "clientHeight", { configurable: true, value: 100 });
  act(() => button("Show chat").click()); expect(middle.scrollTop).toBe(1000);
  middle.scrollTop = 400; act(() => middle.dispatchEvent(new Event("scroll", { bubbles: true })));
  current = { ...current, live_messages: [{ run_id: current.id, generation: 1, surf: "surface", incarnation: 1, session_id: "provider", turn_id: "turn", message_id: "comment", sequence: 1, at: "2026-11-03T11:00:00Z", kind: "commentary", text: "New published update", complete: false, truncated: false }] };
  await refresh(); expect(middle.scrollTop).toBe(400);
  act(() => button("Hide chat").click()); act(() => button("Show chat").click()); expect(middle.scrollTop).toBe(1000);
});

it("does not let a planning-to-handoff refresh collapse chat before a pending automation release succeeds", async () => {
  current = fixture({ state: "planning", stage: "planning", plan_pending: true, automation: { held: true, revision: 8, held_at: "2026-11-03T10:00:00Z", reason: "direct_message", releasing: false } });
  await mount(); expect(composer()).not.toBeNull();
  await act(async () => button("Resume automation").click());
  current = { ...current, state: "handing_off", stage: "handing_off", plan_pending: false };
  await refresh(); expect(composer()).not.toBeNull(); expect(current.automation!.held).toBe(true);
  current = { ...current, automation: { ...current.automation!, held: false, releasing: false } };
  await refresh(); expect(composer()).toBeNull();
});


it("keeps an approved-worker recovery held and chat open until a later actual release", async () => {
  current = fixture({ state: "handing_off", stage: "handing_off", automation: { held: true, revision: 3, held_at: "2026-11-03T10:00:00Z", reason: "direct_message", releasing: false, recovery: "worker_starting" } });
  await mount(); act(() => button("Show chat").click());
  expect(host.querySelector(".run-drawer-status")?.textContent).toContain("Restoring worker");
  expect(host.textContent).toContain("Automation remains held"); expect(button("Resume automation").disabled).toBe(false);
  await refresh(); expect(composer()).not.toBeNull(); expect(posts).toHaveLength(0);
});

it("includes this browser's offline state in the single top status without changing the run", async () => {
  await mount(); const online = vi.spyOn(navigator, "onLine", "get").mockReturnValue(false);
  try {
    act(() => window.dispatchEvent(new Event("offline")));
    expect(host.querySelector(".run-drawer-status")?.textContent).toContain("Offline");
    expect(posts).toHaveLength(0); expect(current.host!.connected).toBe(true);
  } finally { online.mockRestore(); }
});

it("shows a planning recovery card even when chat is hidden", async () => {
  current = fixture({ state: "needs_you", stage: "interviewing", state_reason: "no_plan", plan: null, plan_revision: 0,
    reason_text: "The planner stopped without saving questions or a plan." });
  await mount(); act(() => button("Hide chat").click());
  expect(composer()).toBeNull(); expect(host.textContent).toContain("Planning needs recovery");
  await act(async () => button("Retry planning").click());
  expect(posts[0].url).toContain("/retry-planning");
  expect(posts[0].body).toMatchObject({ expected_revision: 3 });
});


it.each([false, true])("permits a confirmed rejected-call retry only outside an operator pause (%s)", async operatorPaused => {
  current = fixture({ state: "needs_you", stage: "interviewing", state_reason: "no_plan", plan: null, plan_revision: 0,
    paused: true, operator_paused: operatorPaused, pause_reason: "receipt_uncertain",
    coordination_failure: { code: "RUN_TOKEN_REQUIRED", outcome: "rejected", recovering: false } });
  await mount();
  expect(button("Retry planning").disabled).toBe(operatorPaused);
});
