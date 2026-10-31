import { afterEach, describe, expect, it, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { RemoteaDataContext, type RemoteaData } from "./remotea-context";
import { Bubble, PlanApproval, RunDrawer } from "./Remotea";
import { RunButton } from "./RunButton";
import { RunRow } from "./RunsPanel";
import { RoleControls, RunTaskDialog } from "./RunTaskDialog";
import {
  actorChip, attentionKey, DEFAULT_DRAFT, DRAFT_STORAGE_KEY, effectiveDraft, HARNESSES, hostBusyText, modelLabel, planApprovalBody, preflightUrl,
  railDot, restoreDraft, reviewPacket, ROLES_UNAVAILABLE, runChip, runControls, runsBadge, SENT_REASON, startBody, storedDraft, timeline,
  type Host, type PlanReviewRow, type Run, type RunDetail, type RunDraft, type RunRoles,
} from "./remotea-model";

const ID = "11111111-1111-4111-8111-111111111111";
const roles: RunRoles = { planner: { harness: "claude", model: "fable" }, reviewer: { harness: "codex", model: "gpt-6-astra" },
  worker: { harness: "claude", model: "sonnet" }, rounds: 2, approve_final: true };
const run = (over: Partial<Run> = {}): Run => ({
  id: ID, task: { id: "t1", title: "Ship it", status: "open" }, state: "planning", state_revision: 5, state_reason: "review_done",
  reason_text: "", state_since: "2026-10-31T10:00:00Z", generation: 1, plan_revision: 2, tool: "claude", model: "fable", branch: "task/t1",
  base_branch: "main", worktree: "", effective_permission_mode: "", turn_count: 0, turn_cap: 15, started_at: null, finished_at: null,
  stop_requested_at: null, stop_mode: "", surf: "surf-1", host: { id: "h", name: "desk", connected: true }, structured_answers: true,
  session: null, created_at: "2026-10-31T09:00:00Z", roles, plan_round: 1, reviews: [],
  actor: { role: "planner", harness: "claude", model: "fable" }, handoff: null, ...over,
});
const review = (over: Partial<PlanReviewRow> = {}): PlanReviewRow => ({
  id: "r1", round: 1, plan_revision: 1, harness: "codex", model: "gpt-6-astra", attempt: 2, status: "done", verdict: "revise", via: "tool",
  findings: [
    { id: "F1", severity: "blocking", area: "migrations", issue: "The migration drops a column still read.", suggestion: "Keep it one release." },
    { id: "F2", severity: "minor", area: "docs", issue: "No doc update.", suggestion: "" },
  ],
  questions: ["Ship behind a flag?"], summary: "Two problems.", resolutions: [{ finding_id: "F1", decision: "accepted", note: "Kept the column." }],
  counts: { blocking: 2, major: 0, minor: 3 }, started_at: "2026-10-31T10:02:00Z", finished_at: "2026-10-31T10:05:00Z",
  created_at: "2026-10-31T10:01:00Z", ...over,
});
const detail = (over: Partial<RunDetail> = {}): RunDetail => ({
  ...run(), history: [], turns: [], commands: [], questions: [],
  plan: { revision: 2, title: "Ship it safely", prd_text: "Do the thing in two steps.", steps: [{ id: "s1", title: "Add the flag", status: "todo" }],
    criteria: [{ id: "c1", text: "Tests pass" }] },
  ...over,
});
const host = (): Host => ({ id: "h", name: "desk", machine: "wsl", connected: true, last_seen_at: null, conn_epoch: 1, caps: ["spawn"],
  harnesses: { claude: { spawn: true }, codex: { spawn: true } }, proto: 2 });
const data = (over: Partial<RemoteaData> = {}): RemoteaData => ({ enabled: true, status: { enabled: true, key: null, hosts: [host()], link_url: "", connect_command: "" },
  activeRuns: [], runsByTask: new Map(), pending: [], cards: {}, error: "", panel: { run: null, collapsed: false }, start: null, ...over });

afterEach(() => vi.unstubAllGlobals());

describe("the Run dialog's model choices", () => {
  it("offers Fable and GPT-6 Astra, keeping the harness defaults first", () => {
    expect(HARNESSES.map(h => [h.id, h.models.map(m => m.id)])).toEqual([
      ["claude", ["opus", "sonnet", "fable"]], ["codex", ["gpt-5.6-sol", "gpt-6-astra"]]]);
    expect(modelLabel("claude", "fable")).toBe("Fable");
    expect(modelLabel("codex", "gpt-6-astra")).toBe("GPT-6 Astra");
    expect(modelLabel("claude", "haiku-9")).toBe("haiku-9");  // An id the dialog does not list shows as itself.
  });
});

describe("startBody", () => {
  it("leaves a classic run's body as it was: no role fields", () => {
    expect(startBody("t1", DEFAULT_DRAFT)).toEqual({ task_id: "t1", harness: "claude" });
    // A worker that is the planner itself (same harness and model) is "Same as planner".
    expect(startBody("t1", { ...DEFAULT_DRAFT, worker: { harness: "claude", model: "opus" } })).toEqual({ task_id: "t1", harness: "claude" });
    // Rounds and approval alone do not make a role run.
    expect(startBody("t1", { ...DEFAULT_DRAFT, rounds: 5, approveFinal: false })).toEqual({ task_id: "t1", harness: "claude" });
  });
  it("sends a separate worker with approve_final", () => {
    expect(startBody("t1", { ...DEFAULT_DRAFT, worker: { harness: "claude", model: "sonnet" } }))
      .toEqual({ task_id: "t1", harness: "claude", worker: { harness: "claude", model: "sonnet" }, approve_final: true });
  });
  it("sends hyperplan's reviewer and rounds, default models left out", () => {
    const draft: RunDraft = { ...DEFAULT_DRAFT, model: "fable", hyperplan: true, reviewer: { harness: "codex", model: "gpt-6-astra" },
      worker: { harness: "claude", model: "sonnet" }, rounds: 3, approveFinal: false };
    expect(startBody("t1", draft)).toEqual({ task_id: "t1", harness: "claude", model: "fable",
      reviewer: { harness: "codex", model: "gpt-6-astra" }, rounds: 3, worker: { harness: "claude", model: "sonnet" }, approve_final: false });
    expect(startBody("t1", { ...DEFAULT_DRAFT, hyperplan: true }))
      .toEqual({ task_id: "t1", harness: "claude", reviewer: { harness: "codex" }, rounds: 2, approve_final: true });
  });
  it("starts classic when the host cannot spawn roles, or in freestyle", () => {
    const draft: RunDraft = { ...DEFAULT_DRAFT, hyperplan: true, worker: { harness: "codex", model: "gpt-5.6-sol" } };
    expect(startBody("t1", effectiveDraft(draft, false))).toEqual({ task_id: "t1", harness: "claude" });
    expect(startBody("t1", { ...draft, freestyle: true })).toEqual({ task_id: "t1", harness: "claude", freestyle: true });
    expect(effectiveDraft(draft, null)).toBe(draft);  // No target resolved yet: nothing is known against it.
  });
  it("asks the preflight about the roles' harnesses", () => {
    expect(preflightUrl("t1", "claude", null, { reviewer: "codex", worker: "claude" }))
      .toBe("/remotea/runs/preflight?task_id=t1&harness=claude&reviewer_harness=codex&worker_harness=claude");
    expect(preflightUrl("t1", "claude")).toBe("/remotea/runs/preflight?task_id=t1&harness=claude");
  });
});

describe("the remembered choices", () => {
  it("restore what is still offered and fall back to the defaults", () => {
    expect(restoreDraft(null)).toEqual(DEFAULT_DRAFT);
    expect(restoreDraft("{not json")).toEqual(DEFAULT_DRAFT);
    expect(restoreDraft("[1]")).toEqual(DEFAULT_DRAFT);
    const saved: RunDraft = { ...DEFAULT_DRAFT, harness: "codex", model: "gpt-6-astra", hyperplan: true, worker: { harness: "claude", model: "sonnet" },
      reviewer: { harness: "claude", model: "fable" }, rounds: 4, approveFinal: false, target: "x", freestyle: true, interview: "skip" };
    // The target, freestyle and the interview choice belong to one start and are never remembered.
    expect(restoreDraft(storedDraft(saved))).toEqual({ ...saved, target: null, freestyle: undefined, interview: "auto" });
    const odd = restoreDraft(JSON.stringify({ harness: "opencode", model: "x", worker: { harness: "codex", model: "gone" }, rounds: 99, reviewer: 3 }));
    expect(odd).toMatchObject({ harness: "claude", model: "opus", worker: { harness: "codex", model: "gpt-5.6-sol" }, rounds: 10, reviewer: DEFAULT_DRAFT.reviewer });
    expect(restoreDraft(JSON.stringify({ rounds: 0 })).rounds).toBe(1);
  });
});

describe("the Run dialog", () => {
  const choiceOk = () => ({ ok: true });
  const controls = (draft: RunDraft, values: Partial<Parameters<typeof RoleControls>[0]> = {}) => renderToStaticMarkup(
    <RoleControls draft={draft} setDraft={() => {}} choice={choiceOk} roles={true} problems={[]} idPrefix="d" {...values}/>);

  it("shows the planner, the worker and the Hyperplan toggle; storage that is missing or throws is no error", () => {
    // Node has no localStorage at all: readDraft falls back to the defaults.
    const html = renderToStaticMarkup(<RemoteaDataContext.Provider value={data()}>
      <RunTaskDialog taskId="t1" onClose={() => {}} onStarted={() => {}}/></RemoteaDataContext.Provider>);
    for (const text of ["Planner", "Planner model", "Worker", "Same as planner", "Hyperplan"]) expect(html).toContain(text);
    expect(html).not.toContain("Reviewer");
    vi.stubGlobal("localStorage", { getItem: () => { throw new Error("denied"); } });
    expect(renderToStaticMarkup(<RemoteaDataContext.Provider value={data()}>
      <RunTaskDialog taskId="t1" onClose={() => {}} onStarted={() => {}}/></RemoteaDataContext.Provider>)).toContain("Hyperplan");
  });

  it("opens with the remembered choices", () => {
    vi.stubGlobal("localStorage", { getItem: (key: string) => key === DRAFT_STORAGE_KEY
      ? storedDraft({ ...DEFAULT_DRAFT, hyperplan: true, rounds: 3, reviewer: { harness: "codex", model: "gpt-6-astra" } }) : null });
    const html = renderToStaticMarkup(<RemoteaDataContext.Provider value={data()}>
      <RunTaskDialog taskId="t1" onClose={() => {}} onStarted={() => {}}/></RemoteaDataContext.Provider>);
    expect(html).toContain("Reviewer");
    expect(html).toMatch(/<option value="codex:gpt-6-astra" selected="">Codex · GPT-6 Astra<\/option>/);
    expect(html).toContain("<output aria-live=\"polite\">3</output>");
  });

  it("reveals the reviewer, rounds and final approval under Hyperplan", () => {
    const html = controls({ ...DEFAULT_DRAFT, hyperplan: true });
    for (const text of ["Reviewer", "Rounds", "Fewer rounds", "More rounds", "Approve final plan before running"]) expect(html).toContain(text);
    expect(html).toMatch(/<input type="checkbox" checked=""\/><span>Approve final plan before running/);
    // A separate worker alone also asks whether to approve the plan first.
    expect(controls({ ...DEFAULT_DRAFT, worker: { harness: "claude", model: "sonnet" } })).toContain("Approve final plan before running");
    expect(controls(DEFAULT_DRAFT)).not.toContain("Approve final plan");
  });

  it("greys out a harness the target cannot start, and shows a role's problem under it", () => {
    const html = controls({ ...DEFAULT_DRAFT, hyperplan: true }, {
      choice: id => id === "codex" ? { ok: false, reason: "no" } : { ok: true },
      problems: [{ code: "HARNESS_UNAVAILABLE", role: "reviewer", blocking: true, message: "Reviewer: Codex is not enabled on jarvis." }] });
    expect(html).toContain('<option value="codex:gpt-5.6-sol" disabled=""');
    expect(html).toContain("Codex (cannot start here)");
    expect(html).toContain("Reviewer: Codex is not enabled on jarvis.");
  });

  it("turns Hyperplan and the separate worker off when the host cannot spawn roles", () => {
    const html = controls({ ...DEFAULT_DRAFT, hyperplan: true, worker: { harness: "claude", model: "sonnet" } }, { roles: false });
    expect(html).toContain(ROLES_UNAVAILABLE);
    expect(html).toMatch(/<select aria-label="Worker" disabled=""/);
    expect(html).toMatch(/<input type="checkbox" disabled=""\/><span>Hyperplan/);
    expect(html).not.toContain("Reviewer");
    // A HOST_NO_ROLES refusal at start says the same with the server's words.
    expect(controls({ ...DEFAULT_DRAFT, hyperplan: true }, { rolesReason: "Update remotea." })).toContain("Update remotea.");
  });
});

describe("the hyperplan timeline", () => {
  const history = [
    { at: "2026-10-31T10:00:00Z", event: "plan_submitted", plan_revision: 1 },
    { at: "2026-10-31T10:06:00Z", event: "sent", kind: "plan_critique", text: "Review 1 is in.", actor: { kind: "eridani" } },
    { at: "2026-10-31T10:10:00Z", event: "plan_submitted", plan_revision: 2, review_round: 1 },
    { at: "2026-10-31T10:12:00Z", event: "review_blocked", round: 2, detail: "target_busy" },
    { at: "2026-10-31T10:13:00Z", event: "review_blocked", round: 2, detail: "target_busy" },
    { at: "2026-10-31T10:20:00Z", event: "plan_approved", plan_revision: 2 },
    { at: "2026-10-31T10:20:01Z", event: "handoff", attempt: 4, harness: "claude", model: "sonnet" },
    { at: "2026-10-31T10:20:02Z", event: "sent", kind: "handoff", text: "Follow the plan.", actor: { kind: "eridani" } },
    { at: "2026-10-31T11:00:00Z", event: "plan_submitted", plan_revision: 3 },
  ];
  const reviews = [review(), review({ id: "r2", round: 2, verdict: "approve", findings: [], resolutions: [], counts: { blocking: 0, major: 0, minor: 0 },
    started_at: "2026-10-31T10:14:00Z", finished_at: "2026-10-31T10:18:00Z" })];

  it("numbers the planner's drafts, one entry per review round, and the handoff", () => {
    const titles = timeline(detail({ history, reviews })).map(e => e.title);
    expect(titles).toEqual([
      "Draft 1 · Fable", "Review 1 · GPT-6 Astra: revise — 2 blocking, 3 minor", "Sent by Eridani · review sent to the planner",
      "Draft 2 · Fable", "Review 2 waiting: the host is at its run limit; retrying", "Review 2 · GPT-6 Astra: approved",
      "Final plan approved by you", "Handed to Sonnet", "Sent by Eridani · plan handed to the worker",
      // A re-plan after the handoff is the worker's, not a planner draft.
      "Plan submitted · revision 3",
    ]);
    expect(SENT_REASON.plan_critique).toBe("review sent to the planner");
    expect(SENT_REASON.handoff).toBe("plan handed to the worker");
  });

  it("says when a review timed out, failed or is still running", () => {
    const titles = (r: Partial<PlanReviewRow>) => timeline(detail({ reviews: [review(r)] }))[0].title;
    expect(titles({ status: "failed", via: "timeout", verdict: null })).toBe("Review 1 · GPT-6 Astra: timed out");
    expect(titles({ status: "failed", via: "failed", verdict: null })).toBe("Review 1 · GPT-6 Astra: failed");
    expect(titles({ status: "reviewing", verdict: null, finished_at: null })).toBe("Review 1 · GPT-6 Astra: reviewing");
    expect(titles({ verdict: "approve", counts: { blocking: 0, major: 0, minor: 2 } })).toBe("Review 1 · GPT-6 Astra: approved — 2 minor");
  });

  it("leaves a classic run's plan entries as they were", () => {
    const titles = timeline(detail({ roles: null, reviews: [], history: [history[0]] })).map(e => e.title);
    expect(titles).toEqual(["Plan submitted · revision 1"]);
  });

  it("unfolds a review's findings with the planner's resolution and the reviewer's questions", () => {
    const entry = timeline(detail({ reviews: [review()] }))[0];
    const html = renderToStaticMarkup(<Bubble entry={entry} runId={ID}/>);
    expect(html).toContain("<summary>2 findings · 1 question for you</summary>");
    expect(html).toContain('class="severity-blocking"');
    expect(html).toContain("The migration drops a column still read.");
    expect(html).toContain("Suggestion: Keep it one release.");
    expect(html).toContain("Planner accepted: Kept the column.");
    expect(html).toContain("Ship behind a flag?");
    expect(html).toContain("Two problems.");
  });
});

describe("the final-plan approval", () => {
  const waiting = detail({ state: "needs_you", state_reason: "plan_approval", reviews: [review(), review({ id: "r2", round: 2, verdict: "approve",
    counts: { blocking: 0, major: 0, minor: 0 } })] });

  it("shows the frozen plan, the review history, Approve and Revise with note", () => {
    const html = renderToStaticMarkup(<PlanApproval run={waiting} onDone={() => {}}/>);
    for (const text of ["Final plan", "Ship it safely", "Do the thing in two steps.", "Add the flag", "Tests pass",
      "Review 1 · GPT-6 Astra: revise — 2 blocking, 3 minor", "Review 2 · GPT-6 Astra: approved", "Reviewed 2 times.", "with Sonnet in a fresh session"])
      expect(html).toContain(text);
    expect(html).toMatch(/<button class="btn btn-sm" disabled="">Revise with note<\/button>/);  // Needs a note first.
    expect(html).toMatch(/<button class="btn btn-sm btn-primary">.*Approve<\/button>/);
    expect(renderToStaticMarkup(<PlanApproval run={{ ...waiting, state_reason: "plan_review" }} onDone={() => {}}/>)).toBe("");
    expect(renderToStaticMarkup(<PlanApproval run={{ ...waiting, state: "planning" }} onDone={() => {}}/>)).toBe("");
  });

  it("posts approve, or revise with the note, at the revision the reader saw", () => {
    expect(planApprovalBody("approve", "ignored", 9)).toEqual({ action: "approve", expected_revision: 9 });
    expect(planApprovalBody("revise", "  Split step 2.  ", 9)).toEqual({ action: "revise", note: "Split step 2.", expected_revision: 9 });
    expect(planApprovalBody("revise", "  ", 9)).toEqual({ action: "revise", expected_revision: 9 });
  });

  it("asks for you everywhere a run is listed", () => {
    const r = run({ state: "needs_you", state_reason: "plan_approval" });
    expect(runChip(r, [])).toMatchObject({ kind: "open", label: "Approve plan", tone: "attention" });
    expect(railDot(r)).toEqual({ tone: "attention", label: "Approve plan" });
    expect(attentionKey(r)).toBe("review_plan");
    expect(runsBadge([r], [])).toMatchObject({ count: 1, attention: true });
    // The review packet, Edit criteria and Judge again stay away from a plan stop; the card decides it.
    expect(reviewPacket(detail({ state: "needs_you", state_reason: "plan_approval" }))).toBeNull();
    // The footer's Approve completes the task unworked, so at the plan stop it says that.
    expect(runControls(r).approve.label).toBe("Complete task");
    expect(runControls(run({ state: "awaiting_approval" })).approve.label).toBe("Approve");
  });

  it("renders in the drawer with the actor chip and the footer relabelled", () => {
    const html = renderToStaticMarkup(<RunDrawer id={ID} preloaded={waiting} onClose={() => {}}/>);
    expect(html).toContain('aria-label="Final plan approval"');
    expect(html).toContain('<span class="run-actor" title="Who acts on this run now">Planning · Fable</span>');
    expect(html).toContain("Complete task");
    expect(html).not.toContain('aria-label="Review packet"');
  });
});

describe("questions between rounds and a busy host", () => {
  it("shows the planner's interview while a role run is interviewing", () => {
    const html = renderToStaticMarkup(<RunDrawer id={ID} onClose={() => {}} preloaded={detail({ state: "interviewing", interviews: [{
      id: "i1", run_id: ID, generation: 1, status: "open", answers: {}, answered_at: null, created_at: "2026-10-31T10:07:00Z",
      questions: [{ id: "q1", question: "Ship behind a flag?", options: [{ label: "Yes" }, { label: "No" }], multi: false, free_text: true }] }] })}/>);
    expect(html).toContain('aria-label="Planning questions"');
    expect(html).toContain("Ship behind a flag?");
  });

  it("says which session waits for a slot on the host, and Eridani keeps retrying", () => {
    const busy = run({ state: "needs_you", state_reason: "host_busy", actor: { role: "reviewer", harness: "codex", model: "gpt-6-astra", round: 2, rounds: 2 } });
    expect(hostBusyText(busy)).toMatch(/^The reviewer for round 2 \(GPT-6 Astra\) cannot start: desk already runs as many agents as it allows\. Eridani keeps retrying/);
    const worker = run({ state: "needs_you", state_reason: "host_busy", actor: { role: "worker", harness: "claude", model: "sonnet" },
      handoff: { attempt: 4, at: "", planner: roles.planner, worker: roles.worker! } });
    expect(hostBusyText(worker)).toMatch(/^The worker \(Sonnet\) cannot start/);
    expect(hostBusyText(run())).toBeNull();
    expect(attentionKey(busy)).toBe("repair");
    const html = renderToStaticMarkup(<RunDrawer id={ID} onClose={() => {}} preloaded={detail({ ...busy, reason_text: "The host is at its run limit." })}/>);
    expect(html).toContain("The reviewer for round 2 (GPT-6 Astra) cannot start");
  });
});

describe("who acts on the run", () => {
  it("names the role and model from actor and state", () => {
    expect(actorChip(run())).toBe("Planning · Fable");
    expect(actorChip(run({ state: "plan_reviewing", actor: { role: "reviewer", harness: "codex", model: "gpt-6-astra", round: 1, rounds: 2 } })))
      .toBe("Reviewing 1/2 · GPT-6 Astra");
    expect(actorChip(run({ state: "handing_off", actor: { role: "worker", harness: "claude", model: "sonnet" } }))).toBe("Handing off → Sonnet");
    expect(actorChip(run({ state: "executing", actor: { role: "worker", harness: "claude", model: "sonnet" } }))).toBe("Working · Sonnet");
  });
  it("stays away from classic and settled runs", () => {
    expect(actorChip(run({ roles: null, actor: { role: "agent", harness: "claude", model: "opus" } }))).toBeNull();
    expect(actorChip(run({ roles: undefined, actor: undefined }))).toBeNull();
    expect(actorChip(run({ state: "completed" }))).toBeNull();
  });
  it("shows on run rows and in the task row chip's tooltip", () => {
    const row = renderToStaticMarkup(<RunRow run={run({ state: "plan_reviewing", actor: { role: "reviewer", harness: "codex", model: "gpt-6-astra", round: 1, rounds: 2 } })} onOpen={() => {}}/>);
    expect(row).toContain('<span class="run-actor">Reviewing 1/2 · GPT-6 Astra</span>');
    expect(renderToStaticMarkup(<RunRow run={run({ roles: null, actor: null })} onOpen={() => {}}/>)).not.toContain("run-actor");
    const chip = renderToStaticMarkup(<RemoteaDataContext.Provider value={data({ runsByTask: new Map([["t1", run({ summary: "Drafting" })]]) })}>
      <RunButton taskId="t1" title="Ship it"/></RemoteaDataContext.Provider>);
    expect(chip).toContain('title="Planning · Fable — Drafting"');
  });
});
