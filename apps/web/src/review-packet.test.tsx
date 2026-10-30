import { afterEach, describe, expect, it, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { RunDrawer, RunList } from "./Remotea";
import { AttentionList } from "./RunsPanel";
import {
  attentionGroups, attentionKey, criteriaBody, evidenceAge, prLabel, reviewPacket, runAction, runsBadge,
  type Attempt, type Host, type PendingControl, type Run, type RunDetail, type UnmergedWork,
} from "./remotea-model";

const NOW = new Date("2026-10-30T10:10:00Z").getTime();
const ID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa";
const run = (over: Partial<Run> = {}): Run => ({
  id: ID, task: { id: "t1", title: "Ship it", status: "in_progress" }, state: "executing", state_revision: 4,
  state_reason: "turn", reason_text: "", state_since: "2026-10-30T10:00:00Z", generation: 1, plan_revision: 1, tool: "claude", model: "",
  branch: "task/ship", base_branch: "main", worktree: "/wt/ship", effective_permission_mode: "", turn_count: 2, turn_cap: 15,
  started_at: null, finished_at: null, stop_requested_at: null, stop_mode: "", surf: "surf-1",
  host: { id: "h", name: "desk", connected: true }, structured_answers: true, session: null, created_at: "2026-10-30T09:00:00Z", ...over,
});
const attempt = (over: Partial<Attempt>): Attempt => ({
  id: "a", kind: "verify", status: "passed", generation: 1, turn_seq: 3, plan_revision: 1, verify_id: "tests", initiator: "eridani",
  verdict: null, gap: null, summary: null, criteria: null, rules: null, non_verdict: null, exit_code: 0, tail: null, on_demand: null,
  created_at: "2026-10-30T10:05:00Z", updated_at: "2026-10-30T10:06:00Z", ...over,
});
const judge = (over: Partial<Attempt> = {}) => attempt({
  id: "j", kind: "judge", status: "done", verdict: "met", summary: "Two of the three are shown by the passing tests.",
  gap: "The docs criterion has no evidence.", exit_code: null,
  criteria: [{ id: "c1", status: "met", evidence: "tests pass" }, { id: "c2", status: "met", evidence: "endpoint" },
    { id: "c3", status: "unclear", evidence: "" }], ...over,
});
const detail = (over: Partial<RunDetail> = {}): RunDetail => ({
  ...run({ state: "awaiting_approval", state_reason: "judge_met", turn_seq: 3 }), history: [], turns: [], commands: [], questions: [],
  plan: { revision: 1, title: "Ship it", prd_text: "", steps: [], verify: { target_id: "jarvis", verify_id: "tests" },
    criteria: [{ id: "c1", text: "tests pass", kind: "verify", verify_id: "tests" }, { id: "c2", text: "endpoint returns 200" },
      { id: "c3", text: "docs updated" }] },
  attempts: [judge(), attempt({ isolated: true, isolation: "bwrap", workspace_complete: true, workspace_reasons: [] })],
  changes: { branch: "task/ship", worktree: "/wt/ship", pr_urls: ["https://github.com/rdavisdunham/jarvis/pull/12"], pr_as_of: null,
    merged: false, reported_at: null },
  ...over,
});
const question = { id: "q", question_id: "q-1", run_id: ID, generation: 1, status: "open" as const, answers: null, answered_via: "",
  remote_answerable: true, hold_until: null, detail: "", created_at: "", resolved_at: null,
  items: [{ id: "i", text: "Which database?", choices: [] }] };

afterEach(() => vi.unstubAllGlobals());

describe("reviewPacket", () => {
  it("reads criteria, tests, PR and questions into the sentence Eri says", () => {
    const packet = reviewPacket(detail({ questions: [question] }), NOW)!;
    expect(packet.text).toBe("2 of 3 criteria met; tests passed 4 min ago in a sandbox; PR #12 open; 1 open question");
    expect(packet.criteria.map(c => c.status)).toEqual(["met", "met", "unknown"]);
    expect(packet.tests).toMatchObject({ verifyId: "tests", exitCode: 0, isolated: true, markers: [] });
    expect(packet.prs).toEqual([{ url: "https://github.com/rdavisdunham/jarvis/pull/12", label: "PR #12" }]);
    expect(packet.reason).toBe("Two of the three are shown by the passing tests.\n\nThe docs criterion has no evidence.");
    expect(packet.questions).toEqual(["Which database?"]);
    expect(packet.readOnly).toBe(false);
    expect(packet.decision?.kind).toBe("approval");
  });
  it("marks stale, incomplete, unsandboxed and earlier-turn evidence in words", () => {
    const packet = reviewPacket(detail({ attempts: [judge(), attempt({ status: "stale", isolated: false, isolation_reason: "bwrap is not installed",
      workspace_complete: false, workspace_reasons: ["git status failed"], turn_seq: 2 })] }), NOW)!;
    expect(packet.tests!.markers).toEqual(["stale: the code changed during the run", "workspace incomplete: git status failed",
      "not sandboxed (bwrap is not installed)", "for an earlier turn or plan"]);
    expect(packet.testsLine).toBe("tests stale (the code changed during the run) 4 min ago outside a sandbox (workspace incomplete: git status failed) (for an earlier turn or plan)");
  });
  it("says what is missing: no verify command, no PR, failing tests, an older host", () => {
    const none = reviewPacket(detail({ attempts: [judge()], changes: { branch: "task/ship", worktree: "", pr_urls: [], pr_as_of: null,
      unpushed: 2, uncommitted: 1, reported_at: null }, plan: { revision: 1, title: "", prd_text: "", steps: [], criteria: [] } }), NOW)!;
    expect(none.text).toBe("no success criteria; no verify command, so no test result is recorded; no PR recorded (2 unpushed, 1 uncommitted on task/ship); no open questions");
    const failed = reviewPacket(detail({ attempts: [attempt({ status: "failed", exit_code: 1 })] }), NOW)!;
    expect(failed.testsLine).toBe("tests failed (exit 1) 4 min ago");  // An older host said nothing about isolation.
    expect(failed.tests!.isolated).toBeNull();
    const infra = reviewPacket(detail({ attempts: [attempt({ status: "failed", exit_code: 127, category: "infra_missing_binary",
      action: "Install uv or set PATH in the verify entry (eridani.json)" })] }), NOW)!;
    expect(infra.testsLine).toBe("verification could not run (Install uv or set PATH in the verify entry (eridani.json)) 4 min ago");
  });
  it("flags a verdict reached against an earlier plan (criteria edited since)", () => {
    const packet = reviewPacket(detail({ plan_revision: 2 }), NOW)!;
    expect(packet.judgedEarlierPlan).toBe(true);
    expect(packet.text.startsWith("2 of 3 criteria met (judged against an earlier plan)")).toBe(true);
  });
  it("is read-only in needs_you and settled runs, absent while working or in a plan review", () => {
    expect(reviewPacket(detail({ state: "needs_you", state_reason: "judge_needs_you" }), NOW)?.readOnly).toBe(true);
    expect(reviewPacket(detail({ state: "executing" }), NOW)).toBeNull();
    expect(reviewPacket(detail({ state: "needs_you", state_reason: "plan_review" }), NOW)).toBeNull();
  });
  it("keeps a human approval distinct from completion established by verification", () => {
    expect(reviewPacket(detail({ state: "completed", state_reason: "approved" }), NOW)!.decision)
      .toEqual({ kind: "approved", text: "Approved by you: a human decision, not established by verification." });
    expect(reviewPacket(detail({ state: "completed", state_reason: "judge_met" }), NOW)!.decision)
      .toEqual({ kind: "verified", text: "Completed by verification: the judge found every criterion met; no one approved it." });
    expect(reviewPacket(detail({ state: "stopped", state_reason: "stopped_by_user" }), NOW)!.decision).toBeNull();
  });
  it("labels PRs and ages", () => {
    expect(prLabel("https://gitlab.com/a/b/-/merge_requests/7")).toBe("PR #7");
    expect(prLabel("https://example.com/x")).toBe("example.com/x");
    expect(evidenceAge("2026-10-30T10:09:30Z", NOW)).toBe("just now");
    expect(evidenceAge("2026-10-30T07:10:00Z", NOW)).toBe("3 h ago");
    expect(evidenceAge(null, NOW)).toBe("");
  });
});

describe("the review packet in the run drawer", () => {
  const button = (html: string, label: string) => html.match(new RegExp(`<button[^>]*>(?:(?!</button>).)*${label}</button>`))?.[0] ?? "";
  it("shows the packet beside Approve and says approving is your decision", () => {
    const html = renderToStaticMarkup(<RunDrawer id={ID} preloaded={detail()} onClose={() => {}}/>);
    expect(html).toContain('aria-label="Review packet"');
    expect(html).toContain("Review · 2 of 3 criteria met; tests passed");
    expect(html).toContain('class="review-beside-approve"');
    expect(html).toContain("Approving is your decision");
    expect(html).toContain('href="https://github.com/rdavisdunham/jarvis/pull/12"');
    expect(html).toContain("The docs criterion has no evidence.");
    expect(button(html, "Approve")).not.toContain("disabled");
    expect(button(html, "Edit criteria")).not.toContain("disabled");
    expect(button(html, "Judge again")).not.toContain("disabled");
  });
  it("shows the judge's full reason and Edit criteria in needs_you; no edit in a plan review", () => {
    const gap = "The turn says the endpoint works, but nothing shows it. ".repeat(10).trim();
    const html = renderToStaticMarkup(<RunDrawer id={ID} preloaded={detail({ state: "needs_you", state_reason: "judge_needs_you",
      reason_text: "The judge says this needs you.", reason_detail: gap, attempts: [judge({ verdict: "needs_you", summary: "", gap })] })} onClose={() => {}}/>);
    expect(html).toContain(gap);
    expect(html).toContain("Edit criteria");
    expect(html).not.toContain('class="review-beside-approve"');
    const review = renderToStaticMarkup(<RunDrawer id={ID} preloaded={detail({ state: "needs_you", state_reason: "plan_review" })} onClose={() => {}}/>);
    expect(review).not.toContain("Edit criteria");
  });
  it("shows a settled run's packet read-only with how it was completed", () => {
    const html = renderToStaticMarkup(<RunDrawer id={ID} preloaded={detail({ state: "completed", state_reason: "judge_met" })} onClose={() => {}}/>);
    expect(html).toContain("Completed by verification");
    expect(html).not.toContain("Approving is your decision");
    expect(html).not.toContain("Edit criteria");
  });
  it("binds criteria and judge actions to the loaded run's revision", () => {
    expect(runAction(run(), ID, "judge")).toEqual({ url: `/remotea/runs/${ID}/judge`, body: { expected_revision: 4 } });
    expect(criteriaBody([{ id: "c1", text: " tests pass ", kind: "verify" }, { text: "  " }, { text: "docs" }], "tests", "tests"))
      .toEqual({ criteria: [{ id: "c1", text: "tests pass", kind: "verify" }, { text: "docs", kind: "text" }] });
    expect(criteriaBody([{ text: "x" }], "all", "tests")).toEqual({ criteria: [{ text: "x", kind: "text" }], verify_id: "all" });
    expect(criteriaBody([{ text: " " }], null, null)).toBeNull();
  });
});

describe("attentionGroups", () => {
  const host = (over: Partial<Host> = {}): Host => ({ id: "h", name: "desk", machine: "wsl", connected: true, last_seen_at: null, conn_epoch: 1,
    caps: [], harnesses: {}, proto: 2, ...over });
  const pending: PendingControl[] = [{ id: "p", tool: "control.start", summary: "Start Claude Code", run_id: null, expires_at: "" }];
  const runs = [
    run({ id: "work", state: "executing" }),
    run({ id: "approve", state: "awaiting_approval", state_reason: "judge_met" }),
    run({ id: "ask", state: "waiting_answer" }),
    run({ id: "plan", state: "needs_you", state_reason: "plan_review" }),
    run({ id: "infra", state: "needs_you", state_reason: "verify_infra" }),
    run({ id: "judge", state: "needs_you", state_reason: "judge_needs_you" }),
    run({ id: "merge", state: "awaiting_integration" }),
    run({ id: "done", state: "completed", branch: "task/done" }),
  ];
  it("orders groups by the next human action and leaves empty ones out", () => {
    const groups = attentionGroups(runs, pending, [host()]);
    expect(groups.map(g => [g.verb, g.count])).toEqual([
      ["Confirm Eri's proposal", 1], ["Answer a question", 1], ["Review the plan", 1], ["Repair the host or setup", 1],
      ["Inspect the failed check", 1], ["Approve", 1], ["Resolve unmerged work", 1], ["Working", 1], ["Done", 1],
    ]);
    expect(groups.map(g => g.runs.map(r => r.id)).flat()).toEqual(["ask", "plan", "infra", "judge", "approve", "merge", "work", "done"]);
    expect(attentionGroups([run({ id: "work" })], [], [host()]).map(g => g.key)).toEqual(["working"]);
  });
  it("sends runs on an offline host to repair, and settled runs with unmerged work to resolve", () => {
    expect(attentionKey(run({ state: "executing" }), [host({ connected: false })])).toBe("repair");
    expect(attentionKey(run({ state: "needs_you", state_reason: "no_progress" }), [host({ connected: false })])).toBe("repair");
    expect(attentionKey(run({ state: "awaiting_approval" }), [host({ connected: false })])).toBe("approve");  // Approving needs no host.
    const unmerged: UnmergedWork[] = [{ host_id: "h", host_name: "desk", target: "jarvis", path: "/wt/done", branch: "task/done", base: "main",
      ahead: 2, behind: 0, unpushed: 2, uncommitted: 0, untracked: 0, merged: false, active: false, reasons: null }];
    expect(attentionKey(run({ state: "completed", branch: "task/done" }), [host()], unmerged)).toBe("unmerged");
    expect(attentionKey(run({ state: "completed", branch: "task/other" }), [host()], unmerged)).toBe("done");
    expect(attentionKey(run({ state: "needs_you", state_reason: "awaiting_merge" }), [host()])).toBe("unmerged");
  });
  it("counts only actionable items on the badge", () => {
    expect(runsBadge(runs.filter(r => r.state !== "completed"), pending, [host()])).toMatchObject({ count: 7, attention: true });
  });
  it("renders a heading with the verb and count per group", () => {
    const html = renderToStaticMarkup(<AttentionList groups={attentionGroups(runs.slice(0, 3), [], [host()])} onOpen={() => {}}/>);
    expect(html).toContain('aria-label="Answer a question: 1"');
    expect(html.indexOf("Answer a question")).toBeLessThan(html.indexOf("Approve"));
    expect(html.indexOf(">Approve <")).toBeLessThan(html.indexOf("Working"));
  });
  it("groups the remotea page's list the same way", () => {
    const html = renderToStaticMarkup(<RunList onOpen={() => {}} hosts={[host()]} unmerged={[]} preloaded={runs.slice(0, 2)}/>);
    expect(html.indexOf("Approve")).toBeLessThan(html.indexOf("Working"));
  });
});
