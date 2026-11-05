// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { RunDrawer } from "./Remotea";
import { setDurableScope } from "./api";
import type { Attempt, RunDetail } from "./remotea-model";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
let host: HTMLDivElement, root: Root, current: RunDetail;
const steps = [
  { id: "read", title: "Read the mount code", status: "done", phase: "Investigate" },
  { id: "repro", title: "Reproduce the 405", status: "skipped", phase: "Investigate" },
  { id: "route", title: "Accept the slashless route", status: "doing", phase: "Fix", note: "Patched the mount, running tests", updated_at: "2026-11-04T10:00:00Z" },
  { id: "test", title: "Add a regression test", status: "todo", phase: "Fix" },
  { id: "pr", title: "Open the pull request", status: "disputed", phase: "Ship", note: "No PR found" },
];
const fixture = (over: Partial<RunDetail> = {}): RunDetail => ({ id: "stages-run", task: { id: "task", title: "MCP endpoint", status: "in_progress" },
  state: "executing", stage: "executing", state_revision: 3, state_reason: "", reason_text: "", state_since: "2026-11-04T10:00:00Z", generation: 1, plan_revision: 1,
  tool: "claude", model: "fable", branch: "task/mcp", base_branch: "main", worktree: "/fixture", effective_permission_mode: "", turn_count: 1, turn_cap: 0,
  started_at: null, finished_at: null, stop_requested_at: null, stop_mode: "", surf: "surface", host: { id: "host", name: "Fixture", connected: true },
  execution_mode: "managed", direct_supported: true, structured_answers: true, session: null, created_at: "2026-11-04T10:00:00Z", history: [], turns: [], commands: [], questions: [],
  automation: { held: false, revision: 0, held_at: null, reason: null, releasing: false },
  plan: { revision: 1, title: "Endpoint plan", approval_status: "approved", prd_text: "Mount the app at both paths.", steps,
    criteria: [{ id: "c1", text: "No 405 without the slash" }, { id: "c2", text: "Tests pass", kind: "verify", verify_id: "tests" }, { id: "c3", text: "Docs mention it" }] }, ...over });
const judge: Attempt = { id: "judge", kind: "judge", status: "done", generation: 1, turn_seq: 2, plan_revision: 1, verify_id: "", initiator: "eridani", verdict: "not_met",
  gap: "The docs do not mention it.", summary: "Two of three met.", rules: null, non_verdict: null, exit_code: null, tail: null, on_demand: null,
  criteria: [{ id: "c1", status: "met", evidence: "curl returns 200" }, { id: "c2", status: "met", evidence: "" }, { id: "c3", status: "unclear", evidence: "" }],
  created_at: "2026-11-04T10:00:00Z", updated_at: "2026-11-04T10:00:00Z" };
const q = (selector: string) => host.querySelector<HTMLElement>(selector);
const toggle = (name: string) => q(`.run-stage-${name} .run-stage-toggle`)!;
const expanded = () => ["plan", "run", "verify"].map(name => toggle(name).getAttribute("aria-expanded"));
const button = (label: string) => [...host.querySelectorAll<HTMLButtonElement>("button")].find(node => node.textContent === label)!;
async function mount() { await act(async () => { root.render(<RunDrawer id={current.id} preloaded={current} onClose={() => {}}/>); }); }
async function refresh() { await act(async () => { window.dispatchEvent(new Event("eri-agents-changed")); await new Promise(resolve => setTimeout(resolve, 320)); }); }
beforeEach(() => {
  localStorage.clear(); setDurableScope("stages-owner:personal"); current = fixture();
  host = document.createElement("div"); document.body.append(host); root = createRoot(host);
  vi.stubGlobal("fetch", async (url: string) => new Response(JSON.stringify(String(url).includes("/drafts/") ? { value: null, revision: 0 } : current)));
});
afterEach(() => { act(() => root.unmount()); host.remove(); vi.unstubAllGlobals(); delete (Element.prototype as { scrollIntoView?: unknown }).scrollIntoView; });

describe("the sectioned run drawer", () => {
  it("opens Run and draws the phased stepper with every status", async () => {
    await mount();
    expect(expanded()).toEqual(["false", "true", "false"]);
    expect([...host.querySelectorAll(".run-phase-name")].map(n => n.textContent)).toEqual(["Investigate", "Fix", "Ship"]);
    expect([...host.querySelectorAll(".run-step-label")].map(n => n.textContent)).toEqual(["1a", "1b", "2a", "2b", "3a"]);
    expect(q('.run-step[data-step="read"]')!.className).toContain("status-done");
    expect(q('.run-step[data-step="read"] .run-step-marker svg')).not.toBeNull();
    const skipped = q('.run-step[data-step="repro"]')!;
    expect(skipped.className).toContain("status-skipped"); expect(skipped.textContent).toContain("Skipped"); expect(skipped.textContent).not.toContain("To do");
    const active = q('.run-step[aria-current="step"]')!;
    expect(active.dataset.step).toBe("route"); expect(active.textContent).toContain("Patched the mount, running tests"); expect(active.textContent).toContain("Reported");
    expect(q('.run-step[data-step="pr"]')!.className).toContain("status-disputed");
    expect(q('.run-step[data-step="pr"]')!.textContent).toContain("No PR found");
    expect(q('.run-step[data-step="repro"]')!.hasAttribute("data-prev-done")).toBe(true);
    expect(q(".run-phase.is-done .run-phase-name")!.textContent).toBe("Investigate");
    expect(q(".run-ring")!.getAttribute("aria-label")).toBe("40% of steps done");
    expect(toggle("run").textContent).toContain("2 of 5 steps done, working on step 2a");
    expect(q('.run-stage-track [aria-current="step"]')!.textContent).toBe("Run");
    expect(q(".run-stage-track .is-done")!.textContent).toBe("Plan");
  });

  it("numbers flat plans 1..N without phase headings", async () => {
    current = fixture({ plan: { ...fixture().plan!, steps: steps.map(({ phase: _phase, ...rest }) => rest) } }); await mount();
    expect(host.querySelectorAll(".run-phase-title")).toHaveLength(0);
    expect([...host.querySelectorAll(".run-step-label")].map(n => n.textContent)).toEqual(["1", "2", "3", "4", "5"]);
  });

  it("follows the run to Verify and keeps a manual toggle only within a stage", async () => {
    await mount();
    act(() => toggle("plan").click()); expect(expanded()).toEqual(["true", "true", "false"]);
    expect(q(".run-stage-plan")!.textContent).toContain("Mount the app at both paths.");
    await refresh(); expect(expanded()).toEqual(["true", "true", "false"]);
    current = fixture({ state: "awaiting_approval", stage: "awaiting_approval", attempts: [judge] }); await refresh();
    expect(expanded()).toEqual(["false", "false", "true"]);
    expect(q('.run-stage-verify [aria-label="Final acceptance"]')).not.toBeNull();
    expect(toggle("verify").textContent).toContain("2 of 3 criteria met, awaiting your acceptance");
    expect(q(".run-stage-verify .run-stage-flag")!.textContent).toBe("Needs you");
  });

  it("shows criteria with the judge's status, the verify line and the judge's notes", async () => {
    current = fixture({ state: "needs_you", stage: "executing", state_reason: "not_met", attempts: [judge], plan: { ...fixture().plan!, verify: { target_id: "t", verify_id: "tests" } } });
    await mount();
    expect(expanded()).toEqual(["false", "false", "true"]);
    const verify = q(".run-stage-verify")!;
    expect(verify.querySelector(".run-criteria-count")!.textContent).toContain("2 of 3 met");
    expect(verify.querySelectorAll(".criterion-met")).toHaveLength(2);
    expect(verify.querySelectorAll(".criterion-unclear")).toHaveLength(1);
    expect(verify.textContent).toContain("curl returns 200");
    expect(verify.textContent).toContain("needs tests to pass");
    expect(verify.textContent).toContain("Verified by tests after each turn");
    expect(verify.querySelector(".run-judge")!.textContent).toContain("The docs do not mention it.");
  });

  it("states a settled run's outcome", async () => {
    current = fixture({ state: "completed", stage: "completed", state_reason: "judge_met", finished_at: "2026-11-04T11:00:00Z",
      attempts: [{ ...judge, verdict: "met", criteria: judge.criteria!.map(c => ({ ...c, status: "met" as const })) }] });
    await mount();
    expect(expanded()).toEqual(["false", "false", "true"]);
    expect(q(".run-outcome")!.textContent).toContain("Completed");
    expect(q(".run-stage-verify")!.textContent).toContain("Completed by verification");
    expect(host.querySelectorAll(".run-stage-track .is-done")).toHaveLength(3);
  });

  it("swaps the sections for the conversation and keeps the header", async () => {
    await mount();
    act(() => button("Show chat").click());
    expect(host.querySelectorAll(".run-stage")).toHaveLength(0);
    expect(q(".run-stage-track")).not.toBeNull(); expect(q(".run-ring")).not.toBeNull();
    expect(q('[aria-label="Conversation"]')).not.toBeNull();
    act(() => button("Hide chat").click());
    expect(host.querySelectorAll(".run-stage")).toHaveLength(3);
  });

  it("scrolls the active step into view when it moves, only while Run is open", async () => {
    const scroll = vi.fn(); Element.prototype.scrollIntoView = scroll;
    await mount(); expect(scroll).toHaveBeenCalledTimes(1);
    current = fixture({ plan: { ...fixture().plan!, steps: steps.map(s => s.id === "route" ? { ...s, status: "done" } : s.id === "test" ? { ...s, status: "doing" } : s) } });
    await refresh(); expect(scroll).toHaveBeenCalledTimes(2);
    expect((scroll.mock.contexts.at(-1) as HTMLElement).dataset.step).toBe("test");
    act(() => toggle("run").click());
    current = fixture({ plan: { ...fixture().plan!, steps: steps.map(s => ["route", "test"].includes(s.id) ? { ...s, status: "done" } : s) } });
    await refresh(); expect(scroll).toHaveBeenCalledTimes(2);
  });

  it("flags a waiting request in the current stage and reopens a section closed by hand", async () => {
    await mount();
    act(() => toggle("run").click()); expect(expanded()).toEqual(["false", "false", "false"]);
    current = fixture({ decisions: [{ id: "control", run_id: "stages-run", kind: "control", revision: 1, status: "pending", actions: ["confirm", "decline"], payload: { summary: "Run the migration" }, generation: 1, plan_revision: 1 }] });
    await refresh();
    expect(expanded()).toEqual(["false", "true", "false"]);
    expect(q(".run-stage-run .run-stage-flag")!.textContent).toBe("Needs you");
    expect(q(".run-stage-run .run-drawer-decisions")!.textContent).toContain("Run the migration");
  });

  it("puts planning questions and the plan awaiting approval in Plan", async () => {
    current = fixture({ state: "interviewing", stage: "interviewing", plan: null, plan_revision: 0,
      interviews: [{ id: "interview", run_id: "stages-run", generation: 1, status: "open", answers: {}, answered_at: null, created_at: "2026-11-04T10:00:00Z",
        questions: [{ id: "scope", question: "Which endpoints?", options: [], multi: false, free_text: true }] }] });
    await mount(); act(() => button("Hide chat").click());
    expect(expanded()).toEqual(["true", "false", "false"]);
    expect(q(".run-stage-plan")!.textContent).toContain("Which endpoints?");
    expect(q(".run-stage-plan .run-stage-flag")).not.toBeNull();
    expect(toggle("run").textContent).toContain("Starts once the plan is approved");
    current = fixture({ state: "needs_you", stage: "planning", state_reason: "plan_approval", plan: { ...fixture().plan!, approval_status: "pending" } }); await refresh();
    expect(expanded()).toEqual(["true", "false", "false"]);
    const card = q('.run-stage-plan [aria-label="Final plan approval"]')!;
    expect(card.className).not.toContain("is-reference");
    expect([...card.querySelectorAll(".run-step-label")].map(n => n.textContent)).toEqual(["1a", "1b", "2a", "2b", "3a"]);
    expect(card.querySelector("details:has(.run-steps)")!.hasAttribute("open")).toBe(true);
  });
  const ciBase = { current: true, attempt_id: "ci", source: "github_ci" as const, head_sha: "1a2b3c4d5e", pr_url: "https://github.com/o/r/pull/9",
    run_url: "https://github.com/o/r/actions/runs/1", checked_at: "2026-11-04T10:00:00Z", error: null };
  it("puts GitHub CI in Verify while the run awaits it, and a failed CI card in Run while the agent fixes it", async () => {
    current = fixture({ state: "awaiting_ci", stage: "awaiting_ci", state_reason: "ci_pending", attempts: [judge],
      ci: { ...ciBase, state: "pending", checks: [{ name: "backend", bucket: "pending", link: null }], failed: [], log: [],
        since: new Date(Date.now() - 6 * 60_000).toISOString(), wait_until: new Date(Date.now() + 84 * 60_000).toISOString() } });
    await mount();
    expect(expanded()).toEqual(["false", "false", "true"]);
    expect(toggle("verify").textContent).toContain("2 of 3 criteria met, waiting for GitHub CI");
    expect(q(".run-stage-verify .run-ci-card")!.textContent).toContain("Waiting 6 min for GitHub CI on 1a2b3c4");
    expect(host.querySelectorAll(".run-ci-card, .run-ci-line")).toHaveLength(1);
    current = fixture({ state: "executing", stage: "executing", state_reason: "ci_failed", attempts: [judge],
      ci: { ...ciBase, state: "failed", checks: [{ name: "backend", bucket: "fail", link: null }], failed: [{ name: "backend", link: null }],
        log: ["FAILED tests/test_x.py::test_y"], since: null, wait_until: null } });
    await refresh();
    expect(expanded()).toEqual(["false", "true", "false"]);
    expect(q(".run-stage-run .run-ci-card.run-ci-failed")!.textContent).toContain("FAILED tests/test_x.py::test_y");
    expect(toggle("verify").textContent).toContain("2 of 3 criteria met, CI failing");
  });
  it("puts a parked run's stop card in its stage's section and flags it", async () => {
    current = fixture({ state: "needs_you", stage: "executing", state_reason: "ci_timeout", attempts: [judge],
      attention_detail: { reason: "ci_timeout", headline: "CI on 1a2b3c4 was still running after 1 h 30 min (backend).", action: "Check the checks, then approve.", detail: "", last_output: [] } });
    await mount();
    expect(expanded()).toEqual(["false", "false", "true"]);
    expect(q(".run-stage-verify .run-stop-card")!.textContent).toContain("still running after 1 h 30 min");
    expect(toggle("verify").textContent).toContain("Needs you");
  });
});
