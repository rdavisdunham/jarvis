import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { Criteria, InterviewForm } from "./Remotea";
import {
  attemptTitle, criteriaEvidence, interviewAnswers, latestAttempt, runControls, timeline,
  type Attempt, type Interview, type Run, type RunDetail,
} from "./remotea-model";

const run = (values: Partial<Run> = {}): Run => ({
  id: "r1", task: { id: "t1", title: "Task", status: "in_progress" }, state: "executing", state_revision: 2, state_reason: "",
  reason_text: "", state_since: "2026-10-28T10:00:00Z", generation: 1, plan_revision: 1, tool: "claude", model: "",
  branch: "davin/eri-1-task", base_branch: "main", worktree: "", effective_permission_mode: "", turn_count: 1, turn_cap: 15,
  started_at: null, finished_at: null, stop_requested_at: null, stop_mode: "", surf: "surf-1",
  host: { id: "h", name: "desk", connected: true }, structured_answers: true, session: null, created_at: "2026-10-28T09:59:00Z",
  ...values,
});
const interview: Interview = {
  id: "i1", run_id: "r1", generation: 1, status: "open", answers: {}, answered_at: null, created_at: "2026-10-28T10:00:00Z",
  questions: [
    { id: "scope", question: "Which widgets?", options: [{ label: "All" }, { label: "Charts" }], multi: false, free_text: false },
    { id: "where", question: "Platforms?", header: "Where", options: [{ label: "Web" }, { label: "iOS" }], multi: true, free_text: false },
    { id: "budget", question: "Latency target?", options: [], multi: false, free_text: true },
  ],
};
const attempt = (values: Partial<Attempt>): Attempt => ({
  id: "a", kind: "judge", status: "done", generation: 1, turn_seq: 3, plan_revision: 1, verify_id: "", initiator: "eridani",
  verdict: null, gap: null, summary: null, criteria: null, rules: null, non_verdict: null, exit_code: null, tail: null,
  on_demand: null, created_at: "2026-10-28T10:05:00Z", updated_at: "2026-10-28T10:06:00Z", ...values,
});
const detail = (values: Partial<RunDetail> = {}): RunDetail => ({
  ...run(), history: [], turns: [], commands: [], questions: [],
  plan: { revision: 1, title: "Task", prd_text: "", steps: [], verify: { target_id: "jarvis", verify_id: "tests" },
    criteria: [{ id: "c1", text: "p95 under 50 ms" }, { id: "c2", text: "tests pass", kind: "verify", verify_id: "tests" }] },
  ...values,
});

describe("interview form", () => {
  it("needs an allowed answer for every question before sending", () => {
    expect(interviewAnswers(interview, { scope: ["All"] }, {})).toBeNull();
    expect(interviewAnswers(interview, { scope: ["All", "Charts"], where: ["Web"] }, { budget: "50 ms" })).toBeNull();
    expect(interviewAnswers(interview, { scope: ["Nope"], where: ["Web"] }, { budget: "50 ms" })).toBeNull();
    expect(interviewAnswers(interview, { scope: ["Charts"], where: ["Web", "iOS"] }, { budget: " 50 ms " }))
      .toEqual({ scope: "Charts", where: ["Web", "iOS"], budget: "50 ms" });
  });
  it("renders every question as one form with its options", () => {
    const html = renderToStaticMarkup(<InterviewForm interview={interview} onDone={() => {}}/>);
    expect(html).toContain("Planning questions");
    expect(html).toContain("Which widgets?"); expect(html).toContain("Charts"); expect(html).toContain("(choose any)");
    expect(html).toContain('aria-label="Answer: Latency target?"');
    expect(html).toMatch(/<button[^>]*disabled[^>]*>Send answers/);
  });
});

describe("approval and the judge", () => {
  it("makes Approve the primary action only when the run awaits approval", () => {
    expect(runControls(run({ state: "awaiting_approval" })).approve).toMatchObject({ enabled: true, primary: true });
    expect(runControls(run()).approve.primary).toBe(false);
    expect(runControls(run({ state: "completed" })).approve.enabled).toBe(false);
    expect(runControls(run({ state: "needs_you" })).replan.enabled).toBe(true);
    expect(runControls(run({ state: "verifying" })).replan.enabled).toBe(false);
  });
  it("shows each criterion with the latest verdict's evidence", () => {
    const judged = detail({ state: "awaiting_approval", attempts: [
      attempt({ verdict: "met", criteria: [{ id: "c1", status: "met", evidence: "p95 41 ms in the bench" },
        { id: "c2", status: "met", evidence: "42 passed" }] }),
      attempt({ id: "old", verdict: "not_met", criteria: [{ id: "c1", status: "not_met", evidence: "old" }] }),
    ] });
    expect(criteriaEvidence(judged).map(c => [c.id, c.status, c.evidence])).toEqual([
      ["c1", "met", "p95 41 ms in the bench"], ["c2", "met", "42 passed"]]);
    expect(criteriaEvidence(detail()).every(c => c.status === "pending")).toBe(true);
    const html = renderToStaticMarkup(<Criteria run={judged}/>);
    expect(html).toContain("2/2 met"); expect(html).toContain("p95 41 ms in the bench"); expect(html).toContain("needs tests to pass");
    expect(renderToStaticMarkup(<Criteria run={detail({ plan: { ...detail().plan!, criteria: [] } })}/>)).toContain("you approve");
  });
  it("puts verifications and verdicts on the timeline", () => {
    const failed = attempt({ kind: "verify", status: "failed", verify_id: "tests", exit_code: 1, tail: "1 failed" });
    const gap = attempt({ verdict: "not_met", gap: "Add the docs page", updated_at: "2026-10-28T10:07:00Z" });
    expect(attemptTitle(failed)).toBe("Verification tests · failed");
    expect(attemptTitle(attempt({ status: "failed", non_verdict: "refusal" }))).toMatch(/no verdict \(refusal\)/);
    expect(latestAttempt(detail({ attempts: [gap, failed] }), "verify")).toBe(failed);
    const entries = timeline(detail({ attempts: [gap, failed], interviews: [interview] }));
    expect(entries.map(e => e.title)).toEqual(["Interview · waiting for you", "Verification tests · failed", "Judge · not met yet"]);
    expect(entries[2].detail).toBe("Add the docs page");
  });
});
