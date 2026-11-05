import { describe, expect, it } from "vitest";
import type { Attempt, RunDetail } from "./remotea-model";
import {
  activeStep, attentionCount, groupSteps, letters, openSections, planSummary, runStage, runSummary, sectionEpoch, stepProgress,
  stepWord, toggleSections, verifySummary, type PlanStep,
} from "./run-stages";

const step = (id: string, status = "todo", phase?: string, over: Partial<PlanStep> = {}): PlanStep => ({ id, title: "Step " + id, status, ...(phase === undefined ? {} : { phase }), ...over });
const plan = (steps: PlanStep[], over: Partial<NonNullable<RunDetail["plan"]>> = {}): NonNullable<RunDetail["plan"]> =>
  ({ revision: 2, title: "Plan", prd_text: "", steps, criteria: [{ id: "c1", text: "Tests pass" }, { id: "c2", text: "Docs" }], approval_status: "approved", ...over });
const run = (over: Partial<RunDetail> = {}): RunDetail => ({
  id: "run", task: { id: "task", title: "Task", status: "in_progress" }, state: "executing", stage: "executing", state_revision: 1, state_reason: "",
  reason_text: "", state_since: "2026-11-04T10:00:00Z", generation: 1, plan_revision: 2, tool: "claude", model: "fable", branch: "b", base_branch: "main",
  worktree: "/w", effective_permission_mode: "", turn_count: 1, turn_cap: 0, started_at: null, finished_at: null, stop_requested_at: null, stop_mode: "",
  surf: "s", host: { id: "h", name: "Host", connected: true }, structured_answers: true, session: null, created_at: "2026-11-04T10:00:00Z",
  history: [], turns: [], commands: [], questions: [], plan: plan([step("a", "done", "Investigate"), step("b", "doing", "Fix"), step("c", "todo", "Fix")]), ...over,
});
const judge = (criteria: Attempt["criteria"]): Attempt => ({ id: "j", kind: "judge", status: "done", generation: 1, turn_seq: 3, plan_revision: 2, verify_id: "",
  initiator: "eridani", verdict: "not_met", gap: "Docs missing", summary: "One left", criteria, rules: null, non_verdict: null, exit_code: null, tail: null,
  on_demand: null, created_at: "2026-11-04T10:00:00Z", updated_at: "2026-11-04T10:00:00Z" });

describe("groupSteps", () => {
  it("numbers flat when no step has a phase", () => {
    const { flat, groups } = groupSteps([step("a"), step("b"), step("c", "todo", "  ")]);
    expect(flat).toBe(true);
    expect(groups.flatMap(g => g.steps.map(s => s.label))).toEqual(["1", "2", "3"]);
  });
  it("groups consecutive phases as 1a 1b 2a, and a phase that comes back starts a new group", () => {
    const { flat, groups } = groupSteps([step("a", "done", "Investigate"), step("b", "todo", " Investigate "), step("c", "todo", "Fix"), step("d", "todo", "Investigate")]);
    expect(flat).toBe(false);
    expect(groups.map(g => [g.index, g.label])).toEqual([[1, "Investigate"], [2, "Fix"], [3, "Investigate"]]);
    expect(groups.flatMap(g => g.steps.map(s => s.label))).toEqual(["1a", "1b", "2a", "3a"]);
  });
  it("gives steps without a phase among phased ones an unlabeled numbered group", () => {
    const { groups } = groupSteps([step("a", "todo", "Fix"), step("b"), step("c"), step("d", "todo", "Ship")]);
    expect(groups.map(g => g.label)).toEqual(["Fix", "", "Ship"]);
    expect(groups.flatMap(g => g.steps.map(s => s.label))).toEqual(["1a", "2a", "2b", "3a"]);
  });
  it("letters past z and marks the connector behind settled steps", () => {
    expect([0, 25, 26, 27, 51, 52].map(letters)).toEqual(["a", "z", "aa", "ab", "az", "ba"]);
    const { groups } = groupSteps(Array.from({ length: 28 }, (_, i) => step("s" + i, i < 2 ? (i ? "skipped" : "done") : "todo", "Big")));
    expect(groups[0].steps.at(-1)!.label).toBe("1ab");
    expect(groups[0].steps.slice(0, 4).map(s => s.prevDone)).toEqual([false, true, true, false]);
  });
});

describe("progress and the active step", () => {
  it("counts skipped steps toward completion", () => {
    expect(stepProgress([step("a", "done"), step("b", "skipped"), step("c")])).toEqual({ done: 2, total: 3, percent: 67 });
    expect(stepProgress([])).toEqual({ done: 0, total: 0, percent: 0 });
  });
  it("fills the ring for a completed run even when a step was left open (approved from needs-you)", () => {
    const steps = [step("a", "done"), step("b", "done"), step("c", "done"), step("d")];
    expect(stepProgress(steps, { state: "completed" })).toEqual({ done: 4, total: 4, percent: 100 });
    expect(stepProgress(steps, { state: "needs_you" })).toEqual({ done: 3, total: 4, percent: 75 });
  });
  it("prefers the most recently reported doing step, else the first unfinished one", () => {
    const steps = [step("a", "doing", "P", { updated_at: "2026-11-04T09:00:00Z" }), step("b", "doing", "P", { updated_at: "2026-11-04T10:00:00Z" })];
    expect(activeStep(steps)).toMatchObject({ kind: "doing", label: "1b", step: { id: "b" } });
    expect(activeStep([step("a", "done"), step("b", "skipped"), step("c", "disputed")])).toMatchObject({ kind: "next", label: "3", step: { id: "c" } });
    expect(activeStep([step("a", "done"), step("b", "skipped")])).toBeNull();
  });
  it("names every status, so skipped and disputed never read as to do", () => {
    expect(["done", "doing", "skipped", "disputed", "todo", "odd"].map(stepWord)).toEqual(["Done", "Doing", "Skipped", "Disputed", "To do", "To do"]);
  });
});

describe("runStage", () => {
  it.each<[Partial<RunDetail>, string]>([
    [{ state: "completed", stage: "completed" }, "verify"],
    [{ state: "failed", stage: "failed" }, "verify"],
    [{ state: "verifying", stage: "verifying" }, "verify"],
    [{ state: "judging", stage: "checking_results" }, "verify"],
    [{ state: "awaiting_approval", stage: "awaiting_approval" }, "verify"],
    [{ state: "awaiting_integration", stage: "awaiting_integration" }, "verify"],
    [{ state: "awaiting_ci", stage: "awaiting_ci", state_reason: "ci_pending" }, "verify"],
    [{ state: "needs_you", stage: "executing", state_reason: "ci_timeout" }, "verify"],
    [{ state: "needs_you", stage: "executing", state_reason: "ci_failed_again" }, "verify"],
    [{ state: "executing", stage: "executing", state_reason: "ci_failed" }, "run"],
    [{ state: "interviewing", stage: "interviewing", plan: null }, "plan"],
    [{ state: "planning", stage: "planning", plan_pending: true }, "plan"],
    [{ state: "plan_reviewing", stage: "plan_reviewing" }, "plan"],
    [{ state: "needs_you", stage: "interviewing", state_reason: "no_plan", plan: null }, "plan"],
    [{ state: "needs_you", stage: "planning", state_reason: "plan_approval", plan: plan([], { approval_status: "pending" }) }, "plan"],
    [{ state: "needs_you", stage: "executing", state_reason: "plan_approval", paused: true }, "plan"],
    [{ state: "needs_you", stage: "executing", state_reason: "judge_needs_you" }, "verify"],
    [{ state: "needs_you", stage: "executing", state_reason: "verify_missing" }, "verify"],
    [{ state: "needs_you", stage: "executing", state_reason: "not_met" }, "verify"],
    [{ state: "needs_you", stage: "executing", state_reason: "wall_clock" }, "run"],
    [{ state: "waiting_answer", stage: "executing", state_reason: "question" }, "run"],
    [{ state: "executing", stage: "handing_off" }, "run"],
    [{ state: "executing", stage: "executing" }, "run"],
    [{ state: "stopping", stage: "stopping", state_reason: "approved" }, "verify"],
    [{ state: "stopping", stage: "stopping", state_reason: "stopped_by_user" }, "run"],
  ])("%o is %s", (over, stage) => expect(runStage(run(over))).toBe(stage));
});

describe("summaries", () => {
  it("summarizes the plan by stage", () => {
    expect(planSummary(run({ state: "interviewing", stage: "interviewing", plan: null, interviews: [{ id: "i", run_id: "run", generation: 1, questions: [], answers: {}, status: "open", answered_at: null, created_at: "" }] }))).toBe("The planner has questions for you");
    expect(planSummary(run({ state: "planning", stage: "planning", plan: null }))).toBe("The planner is writing the plan");
    expect(planSummary(run({ state: "needs_you", state_reason: "plan_approval", plan: plan([], { approval_status: "pending" }) }))).toBe("Revision 2 needs your approval");
    expect(planSummary(run())).toBe("Approved revision 2, 2 phases");
    expect(planSummary(run({ plan: plan([step("a"), step("b")]), reviews: [{} as never] }))).toBe("Approved revision 2, 2 steps, reviewed 1 time");
  });
  it("summarizes where the work is", () => {
    expect(runSummary(run())).toBe("1 of 3 steps done, working on step 2a");
    expect(runSummary(run({ plan: plan([step("a", "done"), step("b")]) }))).toBe("1 of 2 steps done, next is step 2");
    expect(runSummary(run({ plan: plan([step("a", "done")]) }))).toBe("1 of 1 steps done, every step finished");
    expect(runSummary(run({ state: "completed", state_reason: "approved", plan: plan([step("a", "done"), step("b")]) }))).toBe("1 of 2 steps done, the rest accepted with your approval");
    expect(runSummary(run({ state: "completed", state_reason: "judge_met", plan: plan([step("a", "done"), step("b")]) }))).toBe("1 of 2 steps done, the rest settled when the run completed");
    expect(runSummary(run({ state: "planning", stage: "planning", plan_pending: true, plan: plan([], { approval_status: "pending" }) }))).toBe("Starts once the plan is approved");
    expect(runSummary(run({ state: "completed", stage: "completed", plan: plan([step("a", "done"), step("b")]) }))).toBe("1 of 2 steps done, the rest settled when the run completed");
  });
  it("summarizes verification and the outcome", () => {
    const judged = [judge([{ id: "c1", status: "met", evidence: "" }, { id: "c2", status: "not_met", evidence: "" }])];
    expect(verifySummary(run())).toBe("2 criteria, not judged yet");
    expect(verifySummary(run({ attempts: judged }))).toBe("1 of 2 criteria met");
    expect(verifySummary(run({ state: "awaiting_approval", stage: "awaiting_approval", attempts: judged }))).toBe("1 of 2 criteria met, awaiting your acceptance");
    expect(verifySummary(run({ state: "judging", stage: "judging" }))).toBe("Checking results");
    expect(verifySummary(run({ state: "completed", state_reason: "judge_met" }))).toBe("Completed, every criterion met");
    expect(verifySummary(run({ state: "completed", state_reason: "approved" }))).toBe("Accepted by you");
    expect(verifySummary(run({ plan: plan([], { criteria: [] }) }))).toBe("No success criteria");
    const ci = (state: NonNullable<RunDetail["ci"]>["state"]): RunDetail["ci"] => ({ state, current: true, attempt_id: "ci", source: "github_ci", head_sha: "abc", pr_url: null,
      run_url: null, checks: [], failed: [], log: [], checked_at: null, since: null, wait_until: null, error: null });
    expect(verifySummary(run({ state: "awaiting_ci", stage: "awaiting_ci", attempts: judged, ci: ci("pending") }))).toBe("1 of 2 criteria met, waiting for GitHub CI");
    expect(verifySummary(run({ attempts: judged, ci: ci("failed") }))).toBe("1 of 2 criteria met, CI failing");
    expect(verifySummary(run({ state: "awaiting_approval", stage: "awaiting_approval", attempts: judged, ci: ci("passed") }))).toBe("1 of 2 criteria met, CI passing, awaiting your acceptance");
  });
});

describe("open sections", () => {
  it("opens only the current stage by default", () => {
    expect(openSections(run(), null)).toEqual({ plan: false, run: true, verify: false });
    expect(openSections(run({ state: "awaiting_approval", stage: "awaiting_approval" }), null)).toEqual({ plan: false, run: false, verify: true });
  });
  it("keeps a manual toggle while the stage and the waiting cards stay the same, and resets on either change", () => {
    const executing = run();
    const view = toggleSections(executing, null, "plan");
    expect(openSections(executing, view)).toEqual({ plan: true, run: true, verify: false });
    const closed = toggleSections(executing, view, "run");
    expect(openSections(executing, closed)).toEqual({ plan: true, run: false, verify: false });
    expect(openSections(run({ state: "awaiting_approval", stage: "awaiting_approval" }), closed)).toEqual({ plan: false, run: false, verify: true });
    const asked = run({ decisions: [{ id: "d", run_id: "run", kind: "control", revision: 1, status: "pending", actions: ["confirm"], payload: {}, generation: 1, plan_revision: 2 }] });
    expect(attentionCount(asked)).toBe(1);
    expect(sectionEpoch(asked)).not.toBe(sectionEpoch(executing));
    expect(openSections(asked, closed)).toEqual({ plan: false, run: true, verify: false });
  });
});
