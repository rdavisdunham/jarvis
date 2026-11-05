/* The run drawer's sectioned body: which stage a run is in (Plan, Run, Verify), how its steps group and number
   (1a, 1b, 2a …), how far along it is, and which sections open. Pure: no React, no DOM (RunStages.tsx renders it). */
import { answerableInterviews, ciChip, criteriaEvidence, isSettled, PLAN_STOPS, questionMode, stopCard, type RunDetail } from "./remotea-model";

export type RunStage = "plan" | "run" | "verify";
export const RUN_STAGES: RunStage[] = ["plan", "run", "verify"];
export const STAGE_TITLES: Record<RunStage, string> = { plan: "Plan", run: "Run", verify: "Verify" };

export type PlanStep = NonNullable<RunDetail["plan"]>["steps"][number];
export type LabeledStep = { label: string; step: PlanStep; prevDone: boolean };
export type StepGroup = { index: number; label: string; steps: LabeledStep[] };

const PLAN_KEYS = new Set(["starting", "queued", "spawning", "interviewing", "planning", "replanning", "plan_reviewing", "reviewing_plan"]);
// awaiting_ci: the criteria are met on the quick local check; GitHub CI on the pull request is the final word (run_ci.py).
const VERIFY_KEYS = new Set(["verifying", "judging", "checking_results", "awaiting_ci", "awaiting_approval", "ready_for_review", "awaiting_integration"]);
/** needs_you stops that wait on the plan itself: nothing has been worked yet. */
const PLAN_REASONS = new Set(["no_plan", "review_failed", ...PLAN_STOPS]);
/** needs_you stops whose next human action is about the criteria or the result, not the steps (ci_timeout,
 * ci_failed_again and ci_unavailable included: GitHub CI is the final verification). */
const VERIFY_REASONS = /^(judge_|verify_|ci_)|^(not_met|no_progress_gap|pr_unknown|awaiting_merge|merged|integrated)$/;

type StageInput = Pick<RunDetail, "state" | "stage" | "state_reason" | "plan" | "plan_pending">;

/** The stage a run is in. Settled runs and anything that checks results are Verify; anything before an approved plan
 * is Plan; everything else is Run. */
export function runStage(run: StageInput): RunStage {
  if (isSettled(run)) return "verify";
  const key = run.stage || run.state;
  if (VERIFY_KEYS.has(key) || VERIFY_KEYS.has(run.state)) return "verify";
  if (run.state === "stopping") return ["approved", "judge_met"].includes(run.state_reason) ? "verify" : "run";
  if (!run.plan || run.plan_pending || PLAN_KEYS.has(key) || PLAN_KEYS.has(run.state)) return "plan";
  if (run.state === "needs_you" || run.state === "waiting_answer") {
    if (PLAN_REASONS.has(run.state_reason)) return "plan";
    if (VERIFY_REASONS.test(run.state_reason)) return "verify";
  }
  return "run";
}

export const finished = (step: Pick<PlanStep, "status">) => step.status === "done" || step.status === "skipped";
const phaseOf = (step: PlanStep) => (step.phase ?? "").trim();

/** a … z, aa, ab …: a phase with more than 26 steps still labels every step. */
export function letters(index: number): string {
  let n = index + 1, out = "";
  while (n > 0) { n -= 1; out = String.fromCharCode(97 + (n % 26)) + out; n = Math.floor(n / 26); }
  return out;
}

/** Consecutive steps with the same phase form one group, numbered 1..K; steps inside are lettered (1a, 1b). Consecutive
 * steps without a phase form an unlabeled group that still gets a number. No phase on any step: flat 1..N. */
export function groupSteps(steps: PlanStep[]): { flat: boolean; groups: StepGroup[] } {
  const flat = !steps.some(phaseOf);
  const prevDone = (i: number) => i > 0 && finished(steps[i - 1]);
  if (flat) return { flat, groups: steps.length ? [{ index: 1, label: "", steps: steps.map((step, i) => ({ label: String(i + 1), step, prevDone: prevDone(i) })) }] : [] };
  const groups: StepGroup[] = [];
  steps.forEach((step, i) => {
    let group = groups.at(-1);
    if (!group || phaseOf(step) !== phaseOf(steps[i - 1])) groups.push(group = { index: groups.length + 1, label: phaseOf(step), steps: [] });
    group.steps.push({ label: group.index + letters(group.steps.length), step, prevDone: prevDone(i) });
  });
  return { flat, groups };
}

export function stepLabels(steps: PlanStep[]): Map<string, string> {
  return new Map(groupSteps(steps).groups.flatMap(g => g.steps.map(s => [s.step.id, s.label] as const)));
}

/**
 * Skipped steps count toward completion: they are settled, just not worked. A completed run settles every step
 * (the work was accepted, by the judge or by the person), so its ring is full even when the agent left a step open.
 */
export function stepProgress(steps: PlanStep[], run?: Pick<RunDetail, "state">): { done: number; total: number; percent: number } {
  const total = steps.length, done = run?.state === "completed" ? total : steps.filter(finished).length;
  return { done, total, percent: total ? Math.round(done / total * 100) : 0 };
}

/** The step the agent reports working on (the most recently reported, if several), else the first unfinished one. */
export function activeStep(steps: PlanStep[]): { step: PlanStep; label: string; kind: "doing" | "next" } | null {
  const labels = stepLabels(steps);
  const at = (step: PlanStep) => Date.parse(step.updated_at ?? "") || 0;
  const doing = steps.filter(step => step.status === "doing").sort((a, b) => at(b) - at(a))[0];
  if (doing) return { step: doing, label: labels.get(doing.id) ?? "", kind: "doing" };
  const next = steps.find(step => !finished(step));
  return next ? { step: next, label: labels.get(next.id) ?? "", kind: "next" } : null;
}

export function stepWord(status: string): string {
  return ({ done: "Done", doing: "Doing", skipped: "Skipped", disputed: "Disputed" } as Record<string, string>)[status] ?? "To do";
}

// ------------------------------------------------------------------ the cards that need you, per stage

const openDecision = (run: RunDetail, kind: string) => (run.decisions ?? []).some(d => d.kind === kind && ["open", "pending"].includes(d.status));
/** The planning recovery card (a planner that stopped without saving questions or a plan). */
export function recoveryShown(run: RunDetail): boolean {
  return !isSettled(run) && (run.state_reason === "no_plan" || !!run.coordination_failure)
    && !(run.interviews ?? []).some(i => i.status === "open") && !run.plan;
}
export function planReviewShown(run: RunDetail): boolean {
  return !!run.plan?.proposals && run.state === "needs_you" && run.state_reason === "plan_review";
}
export function approvalAwaiting(run: RunDetail): boolean {
  return openDecision(run, "plan_approval") || (run.state === "needs_you" && run.state_reason === "plan_approval" && run.plan?.approval_status !== "approved");
}
/** Agent questions, control requests and the stop card of a parked run: they belong to whatever stage the run is in. */
export function liveCount(run: RunDetail): number {
  const questions = run.questions.filter(q => !["closed", "pending"].includes(questionMode(q, run))).length;
  return questions + Number(!!stopCard(run)) + (run.decisions ?? []).filter(d => d.kind === "control" && ["open", "pending"].includes(d.status)).length;
}
export function sectionAttention(run: RunDetail): Record<RunStage, boolean> {
  const stage = runStage(run), live = liveCount(run) > 0;
  return {
    plan: recoveryShown(run) || planReviewShown(run) || approvalAwaiting(run) || answerableInterviews(run).length > 0 || (stage === "plan" && live),
    run: stage === "run" && live,
    verify: run.state === "awaiting_approval" || (stage === "verify" && live),
  };
}
export function attentionCount(run: RunDetail): number {
  return liveCount(run) + answerableInterviews(run).length + Number(recoveryShown(run)) + Number(planReviewShown(run))
    + Number(approvalAwaiting(run)) + Number(run.state === "awaiting_approval");
}

// ------------------------------------------------------------------ one-line section summaries

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

export function planSummary(run: RunDetail): string {
  const plan = run.plan, key = run.stage || run.state;
  if (!plan) {
    if (answerableInterviews(run).length) return "The planner has questions for you";
    if (recoveryShown(run)) return "Planning needs recovery";
    if (isSettled(run)) return "No plan was saved";
    return key === "interviewing" ? "The planner is reading the code" : "The planner is writing the plan";
  }
  if (planReviewShown(run)) return "Proposed changes need your answer";
  if (approvalAwaiting(run)) return `Revision ${plan.revision} needs your approval`;
  if (["plan_reviewing", "reviewing_plan"].includes(key)) return `A reviewer is reading revision ${plan.revision}`;
  if (run.plan_pending || ["planning", "replanning"].includes(key)) return `Revising, revision ${plan.revision}`;
  const { flat, groups } = groupSteps(plan.steps);
  const reviews = run.reviews?.length ?? 0;
  return [`${plan.approval_status === "approved" || !plan.approval_status ? "Approved" : "Submitted"} revision ${plan.revision}`,
    flat ? plural(plan.steps.length, "step") : plural(groups.length, "phase"),
    reviews ? `reviewed ${plural(reviews, "time")}` : "",
    run.paused && run.state_reason === "plan_approval" ? "paused" : ""].filter(Boolean).join(", ");
}

export function runSummary(run: RunDetail): string {
  const steps = run.plan?.steps ?? [];
  if (!run.plan || runStage(run) === "plan" && run.plan.approval_status !== "approved") return "Starts once the plan is approved";
  if (!steps.length) return "No steps in this plan";
  const { done, total } = stepProgress(steps), active = activeStep(steps);
  if (run.state === "completed" && active) return `${done} of ${total} steps done, the rest ${run.state_reason === "approved" ? "accepted with your approval" : "settled when the run completed"}`;
  if (isSettled(run)) return `${done} of ${total} steps done` + (active ? "" : ", every step finished");
  const where = !active ? "every step finished" : `${active.kind === "doing" ? "working on" : "next is"} step ${active.label}`;
  return `${done} of ${total} steps done, ${where}`;
}

export function verifySummary(run: RunDetail): string {
  if (run.state === "completed") return run.state_reason === "judge_met" ? "Completed, every criterion met" : run.state_reason === "approved" ? "Accepted by you" : "Completed";
  if (run.state === "stopped") return "Stopped";
  if (run.state === "capped") return "Turn cap reached";
  if (run.state === "failed") return "Failed";
  const rows = criteriaEvidence(run), judged = (run.attempts ?? []).some(a => a.kind === "judge" && a.status === "done");
  const met = rows.filter(r => r.status === "met").length;
  const criteria = !rows.length ? "No success criteria" : judged ? `${met} of ${rows.length} criteria met` : `${plural(rows.length, "criterion").replace("criterions", "criteria")}, not judged yet`;
  const key = run.stage || run.state;
  if (run.state === "awaiting_ci") return criteria + ", waiting for GitHub CI";
  // GitHub CI's state on the pull request head (run_ci.py), once there is one: "CI passing", "CI failing" …
  const ci = ciChip(run.ci)?.label;
  if (run.state === "awaiting_approval") return [criteria, ci, "awaiting your acceptance"].filter(Boolean).join(", ");
  if (["verifying", "judging", "checking_results"].includes(key)) return "Checking results";
  if (run.state === "awaiting_integration") return criteria + ", waiting for sub-task merges";
  return ci ? criteria + ", " + ci : criteria;
}

// ------------------------------------------------------------------ which sections are open

/** Manual open/closed choices and the moment they were made in (stage plus the number of cards waiting for you). */
export type SectionsView = { epoch: string; open: Partial<Record<RunStage, boolean>> } | null;

/** A change of stage, or a new card that needs you, returns every section to its default. */
export function sectionEpoch(run: RunDetail): string {
  return runStage(run) + ":" + attentionCount(run);
}
export function openSections(run: RunDetail, view: SectionsView): Record<RunStage, boolean> {
  const stage = runStage(run);
  const auto = { plan: stage === "plan", run: stage === "run", verify: stage === "verify" };
  if (!view || view.epoch !== sectionEpoch(run)) return auto;
  return { plan: view.open.plan ?? auto.plan, run: view.open.run ?? auto.run, verify: view.open.verify ?? auto.verify };
}
export function toggleSections(run: RunDetail, view: SectionsView, name: RunStage): SectionsView {
  const epoch = sectionEpoch(run), current = openSections(run, view);
  return { epoch, open: { ...(view?.epoch === epoch ? view.open : {}), [name]: !current[name] } };
}
