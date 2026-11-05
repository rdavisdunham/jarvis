import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { RunCiCard, RunCiChip } from "./RunCiCard";
import { RunRow } from "./RunsPanel";
import { attemptTitle, attemptTone, attentionKey, ciCard, ciChip, runLabel, stopCard, timeline, workflowStage, type Attempt, type CiCheck, type Run, type RunCi, type RunDetail } from "./remotea-model";

const HEAD = "1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b";
const RUN = "https://github.com/davis/jarvis/actions/runs/777";
const NOW = Date.parse("2026-11-04T11:00:00Z");
const check = (name: string, bucket: CiCheck["bucket"]): CiCheck => ({ name, bucket, link: `${RUN}/job/1` });

const ci = (values: Partial<RunCi> = {}): RunCi => ({
  state: "pending", current: true, attempt_id: "a1", source: "github_ci", head_sha: HEAD, pr_url: "https://github.com/davis/jarvis/pull/42",
  run_url: RUN, checks: [check("Backend and migrations", "pass"), check("Frontend and browser", "pending")], failed: [], log: [],
  checked_at: "2026-11-04T10:58:00Z", since: "2026-11-04T10:48:00Z", wait_until: "2026-11-04T12:18:00Z", error: null, ...values,
});

const run = (values: Partial<Run> = {}): Run => ({
  id: "run-1", task: { id: "t1", title: "CI as final verification", status: "in_progress" }, state: "awaiting_ci", state_revision: 7,
  state_reason: "ci_pending", reason_text: "The criteria are met on the quick check; waiting for GitHub CI on the pull request.",
  state_since: "2026-11-04T10:48:00Z", stage: "awaiting_ci", generation: 1, plan_revision: 1, tool: "claude", model: "", branch: "task/ci",
  base_branch: "main", worktree: "", effective_permission_mode: "", turn_count: 3, turn_cap: 15, started_at: null, finished_at: null,
  stop_requested_at: null, stop_mode: "", surf: "surf-1", host: { id: "h", name: "desk", connected: true }, structured_answers: true,
  session: null, created_at: "2026-11-04T09:00:00Z", ci: ci(), ...values,
} as Run);

describe("GitHub CI as the final verification", () => {
  it("labels the awaiting-CI state as working, not as needing you", () => {
    expect(runLabel("awaiting_ci")).toBe("Awaiting CI");
    expect(workflowStage(run())).toBe("Awaiting CI");
    expect(attentionKey(run())).toBe("working");
    expect(attentionKey(run({ state: "needs_you", state_reason: "ci_unavailable", stage: "executing" }))).toBe("repair");
    const timedOut = { ...run({ state: "needs_you", state_reason: "ci_timeout" }), plan: null,
      attention_detail: { reason: "ci_timeout", headline: "CI on 1a2b3c4 was still running after 1 h 30 min.", action: "Check it.", detail: "", last_output: [] } } as unknown as RunDetail;
    expect(stopCard(timedOut)?.approve.confirm).toBe("GitHub CI has not passed on the pull request; approve anyway?");
  });

  it("chips pending, passing and failing CI with a link to the checks", () => {
    expect(ciChip(ci())).toMatchObject({ label: "CI pending", tone: "plain", href: RUN, title: "Running on 1a2b3c4: Frontend and browser" });
    expect(ciChip(ci({ state: "passed" }))).toMatchObject({ label: "CI passing", tone: "ok" });
    expect(ciChip(ci({ state: "failed", failed: [{ name: "Frontend and browser", link: null }] })))
      .toMatchObject({ label: "CI failing", tone: "warn", title: "Failed on 1a2b3c4: Frontend and browser" });
    expect(ciChip(ci({ state: "no_pr", run_url: null, pr_url: null }))).toMatchObject({ label: "CI: no pull request", href: null });
    expect(ciChip(null)).toBeNull();
    const html = renderToStaticMarkup(<RunCiChip ci={ci({ state: "passed" })}/>);
    expect(html).toContain(`href="${RUN}"`);
    expect(html).toContain("CI passing");
  });

  it("shows the awaiting-CI card with the wait so far and when it gives up", () => {
    const card = ciCard(run(), NOW)!;
    expect(card.title).toBe("Awaiting CI");
    expect(card.line).toBe("Waiting 12 min for GitHub CI on 1a2b3c4; needs you in 78 min if it has not finished.");
    const html = renderToStaticMarkup(<RunCiCard run={run()} now={NOW}/>);
    expect(html).toContain('aria-label="Awaiting CI"');
    expect(html).toContain("Waiting 12 min for GitHub CI on 1a2b3c4");
    expect(html).toContain(`href="${RUN}"`);
    const broken = renderToStaticMarkup(<RunCiCard run={run({ ci: ci({ error: "gh: not logged in" }) })} now={NOW}/>);
    expect(broken).toContain("The host could not read CI: gh: not logged in");
  });

  it("shows the failed jobs and the log the agent was sent once CI fails", () => {
    const failed = run({ state: "executing", state_reason: "ci_failed", stage: "executing", ci: ci({ state: "failed",
      failed: [{ name: "Frontend and browser", link: `${RUN}/job/2` }], log: ["== Frontend and browser", "FAIL src/x.test.ts", "Expected 1, got 2"] }) });
    const html = renderToStaticMarkup(<RunCiCard run={failed} now={NOW}/>);
    expect(html).toContain('aria-label="CI failed"');
    expect(html).toContain("GitHub CI failed on 1a2b3c4; the agent was sent the failed jobs and the log.");
    expect(html).toContain(`<a href="${RUN}/job/2" target="_blank" rel="noreferrer">Frontend and browser</a>`);
    expect(html).toContain("FAIL src/x.test.ts\nExpected 1, got 2");
  });

  it("keeps a passing run to the chip and hides CI on runs without it", () => {
    const passed = renderToStaticMarkup(<RunCiCard run={run({ state: "awaiting_approval", ci: ci({ state: "passed" }) })} now={NOW}/>);
    expect(passed).toContain("CI passing");
    expect(passed).not.toContain("run-ci-card");
    expect(renderToStaticMarkup(<RunCiCard run={run({ state: "executing", ci: null })} now={NOW}/>)).toBe("");
  });

  it("puts the CI chip and the wait on the run row, with a link to the checks beside it", () => {
    const html = renderToStaticMarkup(<RunRow run={run()} onOpen={() => {}} now={NOW}/>);
    expect(html).toContain("CI pending · 12m");
    expect(html).toContain(`<a class="run-row-ci-link" href="${RUN}"`);
    expect(html).toContain("View checks on GitHub");
    expect(html.indexOf("</button>")).toBeLessThan(html.indexOf("run-row-ci-link"));  // Never a link inside the button.
    const settled = renderToStaticMarkup(<RunRow run={run({ state: "completed", ci: ci({ state: "passed" }) })} onOpen={() => {}} now={NOW}/>);
    expect(settled).not.toContain("run-ci-chip");
  });
  it("titles CI attempts as GitHub CI in the timeline, never as a judge result", () => {
    const attempt = (status: string, state: RunCi["state"], failed: { name: string; link: string | null }[] = []): Attempt => ({
      id: "ci-" + status, kind: "ci", status, generation: 1, turn_seq: 3, plan_revision: 1, verify_id: "github_ci", initiator: "host",
      verdict: null, gap: null, summary: null, criteria: null, rules: null, non_verdict: null, exit_code: null, tail: null, on_demand: null,
      created_at: "2026-11-04T10:50:00Z", updated_at: "2026-11-04T10:58:00Z", source: "github_ci", state, head_sha: HEAD, failed });
    expect(attemptTitle(attempt("passed", "passed"))).toBe("GitHub CI on 1a2b3c4 · passing");
    expect(attemptTitle(attempt("pending", "no_pr"))).toBe("GitHub CI on 1a2b3c4 · no pull request");
    expect([attemptTone(attempt("passed", "passed")), attemptTone(attempt("failed", "failed")), attemptTone(attempt("pending", "pending"))]).toEqual(["good", "warn", "plain"]);
    const entry = timeline({ ...run(), history: [], turns: [], commands: [], questions: [],
      attempts: [attempt("failed", "failed", [{ name: "Backend and migrations", link: null }])] } as unknown as RunDetail).find(e => e.kind === "attempt")!;
    expect(entry.title).toBe("GitHub CI on 1a2b3c4 · failing");
    expect(entry.detail).toBe("Failed: Backend and migrations");
  });
});
