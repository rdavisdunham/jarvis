// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { activityStatus, approvalShortcut, DecisionCard, DeliveryReceipt, RunActivity, useRunDraft, workflowAttention, workflowStage, type RunDecision } from "./RunWorkflow";
import { setDurableScope } from "./api";
import { PlanApproval, QuestionCard } from "./Remotea";
import { modelOptions, unsupportedRole } from "./RunModels";
import { runChip, railDot, restoreDraft, startBody, storedDraft, type RunDetail, type RunQuestion } from "./remotea-model";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
const run = (values: Partial<RunDetail> = {}) => ({ id: "r1", state: "planning", state_reason: "", plan_revision: 1, host: { connected: true }, ...values } as RunDetail);
const decision = (values: Partial<RunDecision> = {}): RunDecision => ({ id: "p1", kind: "plan_approval", run_id: "r1", revision: 7, status: "pending", payload: {}, actions: ["approve", "revise"], generation: 1, plan_revision: 2, ...values });

it("keeps stage, attention and connection independent", () => {
  const planning = run({ stage: "planning", attention: "decision" });
  expect(workflowStage(planning)).toBe("Planning");
  expect(workflowAttention(planning)).toBe("Waiting for you");
  expect(runChip(planning, [])).toMatchObject({ label: "Planning", attention: "Waiting for you" });
  expect(railDot(planning).label).toBe("Planning · Waiting for you");
  expect(workflowStage(run({ state: "needs_you", state_reason: "plan_approval" }))).toBe("Reviewing plan");
  expect(workflowStage(run({ state: "verifying" }))).toBe("Checking results");
  expect(workflowStage(run({ state: "awaiting_approval" }))).toBe("Ready for review");
  expect(workflowStage(run({ state: "completed" }))).toBe("Complete");
});

it("animates only fresh reported work and keeps waiting/reconnection separate", () => {
  const now = Date.parse("2026-11-02T12:00:00Z");
  const active = run({ activity: { kind: "thinking", at: "2026-11-02T11:59:50Z" } });
  expect(activityStatus(active, now)).toEqual({ label: "Thinking…", animate: true });
  expect(activityStatus({ ...active, attention: "decision" }, now)).toEqual({ label: "Waiting for you", animate: false });
  expect(activityStatus({ ...active, connectivity: "offline" }, now)).toEqual({ label: "Reconnecting…", animate: false });
  expect(activityStatus(active, now + 100_000)).toEqual({ label: "Activity unconfirmed", animate: false });
  expect(activityStatus(run(), now).animate).toBe(false);
  expect(activityStatus({ ...active, paused: true, pause_reason: "receipt_uncertain" }, now)).toEqual({ label: "Delivery unconfirmed", animate: false });
  expect(activityStatus({ ...active, paused: true, pause_reason: "permission_denied" }, now)).toEqual({ label: "Permission needs review", animate: false });
  expect(renderToStaticMarkup(<RunActivity run={active} now={now}/>)).toContain('aria-live="polite"');
});

it("Approved resolves a sole primary request and never invents answers or permanent permission", () => {
  expect(approvalShortcut(run({ controller: "terminal", decisions: [decision()] }), "Approved")).toBeNull();
  expect(approvalShortcut(run({ decisions: [decision()] }), "Approved")?.action).toBe("approve");
  expect(approvalShortcut(run({ decisions: [decision(), decision({ id: "p2" })] }), "Approved")).toBeNull();
  expect(approvalShortcut(run({ decisions: [decision({ kind: "interview", actions: ["answer"] })] }), "Approved")).toBeNull();
  const permission = decision({ kind: "question", status: "open", actions: ["answer"], payload: { kind: "permission", items: [{ id: "perm", choices: [{ id: "allow_once" }, { id: "always" }] }] } });
  expect(approvalShortcut(run({ decisions: [permission] }), "Approved.")?.values).toEqual({ answers: [{ item_id: "perm", choice_ids: ["allow_once"] }] });
  permission.payload = { kind: "permission", items: [{ id: "perm", choices: [{ id: "always" }] }] };
  expect(approvalShortcut(run({ decisions: [permission] }), "Approved")).toBeNull();
});

it("shares task, agent, reason and honest delivery on every decision", () => {
  const html = renderToStaticMarkup(<DecisionCard title="Plan approval" task="Build widget" agent="Planner · Sol" reason="Ready to start"><button>Approve</button></DecisionCard>);
  expect(html).toContain("Build widget · Planner · Sol"); expect(html).toContain("Ready to start");
  const receipt = { id: "a", url: "/test", body: {}, updatedAt: "", status: "saved" as const };
  expect(renderToStaticMarkup(<DeliveryReceipt receipt={receipt}/>)).toContain("waiting for agent acknowledgement");
  expect(renderToStaticMarkup(<DeliveryReceipt receipt={{ ...receipt, status: "uncertain" }}/>)).toContain("Retry delivery");
});

it("uses host model catalogs and keeps unsupported remembered choices visible", () => {
  const models = [{ host_id: "h", harness: "codex", model: "gpt-6.1-sol", label: "GPT-6.1 Sol", efforts: ["low", "medium", "high"], permission_profiles: [] }];
  expect(modelOptions(models, "codex").map(m => m.model)).toEqual(["gpt-6.1-sol"]);
  expect(unsupportedRole(models, { harness: "codex", model: "gpt-6.1-sol", effort: "ultra" })).toContain("does not support ultra");
  const remembered = restoreDraft(storedDraft({ harness: "codex", model: "retired", effort: "ultra", interview: "auto", target: null }));
  expect(remembered.model).toBe("retired"); expect(remembered.effort).toBe("ultra");
  expect(unsupportedRole(models, remembered)).toContain("does not support retired");
  expect(startBody("t", { ...remembered, model: "gpt-6.1-sol", effort: "high" })).toMatchObject({ model: "gpt-6.1-sol", effort: "high" });
});

describe("durable drafts", () => {
  let host: HTMLDivElement, root: Root;
  const values: ((value: string) => void)[] = [];
  function Draft({ name }: { name: string }) { const [text, setText] = useRunDraft(name, ""); values.push(setText); return <output>{text}</output>; }
  beforeEach(() => { localStorage.clear(); setDurableScope("test-owner:personal"); host = document.createElement("div"); root = createRoot(host); vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("offline"))); });
  afterEach(() => { act(() => root.unmount()); vi.unstubAllGlobals(); });
  it("preserves text through run navigation and reload while offline", () => {
    act(() => root.render(<Draft name="composer:r1"/>));
    act(() => values.at(-1)!("Keep my instructions"));
    act(() => root.render(<Draft name="composer:r2"/>));
    expect(host.textContent).toBe("");
    act(() => root.render(<Draft name="composer:r1"/>));
    expect(host.textContent).toBe("Keep my instructions");
    act(() => root.unmount()); root = createRoot(host);
    act(() => root.render(<Draft name="composer:r1"/>));
    expect(host.textContent).toBe("Keep my instructions");
  });
});


describe("native permission decisions", () => {
  let host: HTMLDivElement, root: Root;
  beforeEach(() => { localStorage.clear(); setDurableScope("test-owner:personal"); host = document.createElement("div"); root = createRoot(host); });
  afterEach(() => { act(() => root.unmount()); vi.unstubAllGlobals(); });
  it("shows the complete action and submits the explicit allow-once choice immediately", async () => {
    const request = { id: "q1", kind: "permission", status: "open", generation: 1, remote_answerable: true, detail: "The tool needs permission",
      items: [{ id: "permission", text: "Run command?", choices: [{ id: "allow_once", label: "Allow" }, { id: "deny", label: "Deny" }, { id: "always", label: "Allow always" }] }],
      action: { tool: "shell", cwd: "/repo", input: { command: "npm test", nested: { full: "z".repeat(5000) } }, permissions: { network: false }, digest: "sha256:abc" } } as unknown as RunQuestion;
    const permission = decision({ id: "q1", kind: "question", status: "open", actions: ["answer"], payload: { kind: "permission", items: request.items, action: request.action } });
    const posts: unknown[] = [];
    vi.stubGlobal("fetch", async (_url: string, init: RequestInit = {}) => {
      if (init.method === "POST") { posts.push(JSON.parse(String(init.body))); return new Response(JSON.stringify({ receipt: { status: "saved" } })); }
      return new Response(JSON.stringify({ value: null, revision: 0 }));
    });
    act(() => root.render(<QuestionCard question={request} run={run({ state: "executing", generation: 1, structured_answers: true, decisions: [permission] })} onDone={() => {}}/>));
    expect(host.textContent).toContain("npm test"); expect(host.textContent).toContain("z".repeat(5000));
    expect(host.textContent).toContain("/repo"); expect(host.textContent).toContain("sha256:abc");
    const approve = [...host.querySelectorAll("button")].find(button => button.textContent === "Approve once")!;
    await act(async () => { approve.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(posts).toEqual([{ action_id: expect.any(String), expected_revision: 7, action: "answer", answers: [{ item_id: "permission", choice_ids: ["allow_once"] }] }]);
    expect(host.textContent).toContain("waiting for agent acknowledgement");
  });
});

it("keeps an approved paused plan visible without offering a second approval", () => {
  const paused = run({ paused: true, state: "needs_you", state_reason: "plan_approval", plan_revision: 1,
    plan: { revision: 1, title: "Widget plan", prd_text: "Preserve this plan", steps: [], criteria: [], approval_status: "approved" } });
  const html = renderToStaticMarkup(<PlanApproval run={paused} onDone={() => {}}/>);
  expect(html).toContain("Plan approved · paused"); expect(html).toContain("Preserve this plan");
  expect(html).not.toContain("Request changes");
});
