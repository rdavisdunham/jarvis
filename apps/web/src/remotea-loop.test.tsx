import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { Criteria, InterviewForm, OpenInterviews, PlanReview } from "./Remotea";
import {
  answerableInterviews, attemptTitle, attemptTone, commandTitle, criteriaEvidence, interviewAnswers, latestAttempt, orderOptions, planReviewBody, runControls, timeline,
  type Attempt, type Interview, type Proposals, type Run, type RunDetail,
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
  });  it("offers the form while interviewing and while parked in needs_you with the interview open", () => {
    const parked = detail({ state: "needs_you", state_reason: "wall_clock", plan_revision: 0, interviews: [interview] });
    expect(renderToStaticMarkup(<OpenInterviews run={parked} onDone={() => {}}/>)).toContain("Which widgets?");
    expect(answerableInterviews(detail({ state: "interviewing", interviews: [interview] }))).toEqual([interview]);
    expect(answerableInterviews({ ...parked, generation: 2 })).toEqual([]);  // Asked by an older generation.
    expect(answerableInterviews({ ...parked, interviews: [{ ...interview, status: "answered" }] })).toEqual([]);
    expect(renderToStaticMarkup(<OpenInterviews run={detail({ interviews: [interview] })} onDone={() => {}}/>)).toBe("");
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

describe("why a verification did not pass", () => {
  it("labels an infra attempt with its action in a setup tone, apart from failing tests", () => {
    const infra = attempt({ kind: "verify", status: "error", verify_id: "tests", exit_code: null, category: "infra_missing_binary",
      action: "Install `uv` or set PATH in the verify entry (eridani.json)." });
    const failing = attempt({ kind: "verify", status: "failed", verify_id: "tests", exit_code: 1, category: "tests_failed",
      action: "Fix the failing tests." });
    expect(attemptTitle(infra)).toBe("Verification could not run · Install `uv` or set PATH in the verify entry (eridani.json).");
    expect(attemptTitle(failing)).toBe("Verification tests · failed");
    expect(attemptTone(infra)).toBe("setup");
    expect(attemptTone(failing)).toBe("warn");
    expect(attemptTitle(attempt({ kind: "verify", status: "timed_out", verify_id: "tests", category: "infra_timeout", action: "" })))
      .toBe("Verification could not run · timed out");
    const entry = timeline(detail({ attempts: [infra] })).find(e => e.kind === "attempt");
    expect(entry?.tone).toBe("setup");
  });
  it("names refused, expired and uncertain commands plainly", () => {
    const command = (state: string) => ({ id: "c", op_id: "o", kind: "verify", status: state, generation: 1, mode: null, text: null,
      outcome: { state }, created_at: "2026-10-28T10:00:00Z", updated_at: "2026-10-28T10:00:00Z" }) as Parameters<typeof commandTitle>[0];
    expect(commandTitle(command("rejected"))).toBe("Verify · refused by remotea");
    expect(commandTitle(command("expired"))).toBe("Verify · expired before remotea took it");
    expect(commandTitle(command("uncertain"))).toBe("Verify · delivery uncertain (remotea could not tell if it ran)");
  });
});

describe("text sent to the agent (stage F)", () => {
  it("puts every sent text on the timeline with who sent it", () => {
    const entries = timeline(detail({ history: [
      { at: "2026-10-28T10:01:00Z", event: "sent", text: "Change course.", actor: { kind: "human", reason: "confirmed:p1" } },
      { at: "2026-10-28T10:02:00Z", event: "sent", text: "Also docs.", actor: { kind: "human" } },
    ] }));
    expect(entries.map(e => [e.title, e.detail])).toEqual([
      ["Sent by Eri, confirmed by you", "Change course."], ["Sent by you", "Also docs."]]);
  });
});

describe("chat timeline", () => {
  it("shows a sent text once, says who sent it, and tells the agent's turns from Eridani's events", () => {
    const entries = timeline(detail({
      history: [{ at: "2026-10-28T10:02:00Z", event: "sent", text: "Also docs.", actor: { kind: "human" } },
        { at: "2026-10-28T10:04:00Z", event: "sent", text: "The operator answered…", actor: { kind: "run_judge" } }],
      commands: [
        { id: "i1", op_id: "o1", kind: "input", status: "executed", generation: 1, mode: "queue", text: "Also docs.", outcome: { state: "executed" }, created_at: "2026-10-28T10:02:01Z", updated_at: "2026-10-28T10:02:05Z" },
        { id: "i2", op_id: "o2", kind: "input", status: "failed", generation: 1, mode: "queue", text: "The operator answered…", outcome: { state: "failed", detail: "surface gone" }, created_at: "2026-10-28T10:04:01Z", updated_at: "2026-10-28T10:04:05Z" },
      ],
      turns: [{ id: "t", seq: 1, text: "Done with step one.", clipped: false, truncated: false, generation: 1, at: "2026-10-28T10:03:00Z", prompt: "" }],
    }));
    expect(entries.map(e => [e.title, e.who])).toEqual([
      ["Sent by you", "you"], ["Turn 1", "agent"], ["Sent by Eridani", "eridani"], ["Message · failed", "eridani"]]);
    expect(entries[3].detail).toBe("surface gone");  // The failed delivery shows its outcome, not the text again.
  });
  it("puts plan submission and your review on the timeline", () => {
    const entries = timeline(detail({ history: [
      { at: "2026-10-28T10:05:00Z", event: "plan_submitted", plan_revision: 1, proposals: "Proposed changes to 1 criterion" },
      { at: "2026-10-28T10:06:00Z", event: "plan_reviewed", plan_revision: 2, accepted: ["c1"], kept: ["c3"], actor: { kind: "human" } },
    ] }));
    expect(entries.map(e => [e.title, e.detail, e.tone])).toEqual([
      ["Plan submitted · revision 1", "Proposed changes to 1 criterion — waiting for your review.", "attention"],
      ["Plan reviewed by you · revision 2", "accepted c1 · kept c3", "good"]]);
  });
});

describe("plan review", () => {
  const proposals: Proposals = {
    criteria: [{ id: "c1", change: "reword", text: "compares only cmd", proposed: "compares cmd and url", because: "You answered: compare cmd and url" },
      { id: "c3", change: "remove", text: "docs updated", because: "No docs here" }],
    completion_policy: { current: "pr_merged", proposed: "pr_open" },
  };
  const waiting = detail({ state: "needs_you", state_reason: "plan_review", reason_text: "The agent proposes changes…",
    plan: { ...detail().plan!, proposals } });
  it("renders each proposal with yours, the agent's, its reason, and Accept first in green", () => {
    const html = renderToStaticMarkup(<PlanReview run={waiting} onDone={() => {}}/>);
    expect(html).toContain("Plan review"); expect(html).toContain("compares only cmd"); expect(html).toContain("compares cmd and url");
    expect(html).toContain("You answered: compare cmd and url"); expect(html).toContain("remove this criterion");
    expect(html).toContain("Done when the pull request is merged");
    expect(html).toMatch(/aria-pressed="true"[^>]*btn-primary[^>]*>Accept</); expect(html).toContain(">Keep mine<");
    expect(html).toContain("Freeze the plan and start");
    expect(renderToStaticMarkup(<PlanReview run={detail()} onDone={() => {}}/>)).toBe("");
  });
  it("sends one decision per change and your wording only when you changed it", () => {
    expect(planReviewBody(proposals, { c1: { accept: true, text: " compares cmd and url " }, c3: { accept: false, text: "" } }, "accept", "keep", 7)).toEqual({
      decisions: [{ id: "c1", accept: true }, { id: "c3", accept: false }], verify: "keep", completion_policy: "keep", expected_revision: 7 });
    expect(planReviewBody(proposals, { c1: { accept: true, text: "compares cmd and url only" } }, "keep", "accept", 7).decisions[0])
      .toEqual({ id: "c1", accept: true, text: "compares cmd and url only" });
  });
});

describe("answer options", () => {
  it("puts the recommended option first and lets you type an answer to any question", () => {
    expect(orderOptions([{ label: "A" }, { label: "B (Recommended)" }, { label: "C", description: "recommended path" }]).map(o => o.label))
      .toEqual(["B (Recommended)", "C", "A"]);
    expect(interviewAnswers(interview, { where: ["Web"] }, { scope: "Only the charts", budget: "50 ms" }))
      .toEqual({ scope: "Only the charts", where: ["Web"], budget: "50 ms" });
    const html = renderToStaticMarkup(<InterviewForm interview={{ ...interview, questions: [{ ...interview.questions[0],
      options: [{ label: "All" }, { label: "Charts (Recommended)" }] }] }} onDone={() => {}}/>);
    expect(html.indexOf("Charts (Recommended)")).toBeLessThan(html.indexOf(">All<"));
    expect(html).toMatch(/btn-recommended[^>]*>Charts \(Recommended\)/);
    expect(html).toContain('aria-label="Answer: Which widgets?"');
  });
});
