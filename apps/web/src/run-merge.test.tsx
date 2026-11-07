import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { RunMergePanel } from "./RunMergePanel";
import { mergePanel, timeline, type Run, type RunMerge } from "./remotea-model";

const HEAD = "1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b";
const merge = (values: Partial<RunMerge> = {}): RunMerge => ({
  status: "ready", detail: "Merge PR #42 into main.", can_merge: true, can_update: false,
  pr: { number: 42, url: "https://github.com/davis/jarvis/pull/42", base: "main", head_sha: HEAD, state: "OPEN" },
  ci: "passed", checked_at: "2026-11-07T10:00:00Z", op: null, ...values,
});
const noop = () => undefined;
const html = (m: RunMerge | null, state: Run["state"] = "awaiting_approval") =>
  renderToStaticMarkup(<RunMergePanel run={{ merge: m, state }} disabled={false} onMerge={noop} onUpdate={noop} onSendBack={noop}/>);

describe("Merge from Eridani", () => {
  it("offers Merge into the base once CI is green", () => {
    expect(mergePanel(merge())).toMatchObject({ action: "merge", label: "Merge into main", enabled: true, tone: "plain", poll: false });
    const markup = html(merge());
    expect(markup).toContain("Merge into main");
    expect(markup).toContain("btn-primary");
    expect(markup).toContain("PR #42");
  });

  it("grays Merge out while CI runs, and keeps asking", () => {
    const panel = mergePanel(merge({ status: "waiting_ci", detail: "Waiting on CI.", can_merge: false }));
    expect(panel).toMatchObject({ action: "merge", enabled: false, line: "Waiting on CI.", poll: true });
    expect(html(merge({ status: "waiting_ci", detail: "Waiting on CI.", can_merge: false }))).toMatch(/<button[^>]*disabled=""[^>]*>.*Merge into main/);
  });

  it("offers Update branch when the base moved on", () => {
    const behind = merge({ status: "behind", detail: "The branch is behind main; update it, then CI runs again.", can_merge: false, can_update: true });
    expect(mergePanel(behind)).toMatchObject({ action: "update", label: "Update branch", enabled: true });
    // Failing CI on a branch that is also behind: Update branch is the button, and Send back stays.
    const failedBehind = merge({ status: "ci_failed", detail: "CI failed on this pull request. It is also behind main.", can_merge: false, can_update: true });
    expect(mergePanel(failedBehind)).toMatchObject({ action: "update", enabled: true, tone: "warn", sendBack: expect.stringContaining("failing checks") });
  });

  it("sends conflicts and failing CI back to the agent, unless the run settled", () => {
    const conflicts = merge({ status: "conflicts", detail: "The branch conflicts with main.", can_merge: false });
    expect(mergePanel(conflicts)?.sendBack).toContain("Merge main into the branch");
    expect(html(conflicts)).toContain("Send back to fix conflicts");
    expect(html(conflicts, "completed")).not.toContain("Send back");
    expect(mergePanel(merge({ status: "ci_failed", can_merge: false }))).toMatchObject({ tone: "warn", sendBack: expect.stringContaining("failing checks") });
  });

  it("shows a merged pull request without a button, and GitHub's refusal otherwise", () => {
    const merged = merge({ status: "merged", detail: "Merged PR #42 into main.", can_merge: false });
    expect(mergePanel(merged)).toMatchObject({ action: "none", tone: "ok" });
    expect(html(merged)).not.toContain("<button");
    const refused = merge({ op: { kind: "pr.merge", state: "refused", reason: "blocked", message: "the base branch policy prohibits the merge" } });
    expect(mergePanel(refused)?.refused).toBe("GitHub refused the last merge: the base branch policy prohibits the merge");
    expect(mergePanel(merge({ status: "merged", op: { kind: "pr.merge", state: "refused" } }))?.refused).toBeNull();
  });

  it("renders nothing for an older Eridani without merge facts", () => {
    expect(mergePanel(null)).toBeNull();
    expect(html(null)).toBe("");
  });
});

describe("Merge events on the timeline", () => {
  it("shows the merge, the update and a refusal", () => {
    const history = [
      { at: "2026-11-07T10:00:00Z", event: "branch_updated", new_head_sha: "9".repeat(40) },
      { at: "2026-11-07T10:01:00Z", event: "merge_refused", reason: "blocked" },
      { at: "2026-11-07T10:02:00Z", event: "merged_pr", number: 42, base: "main" },
    ];
    const entries = timeline({ history, turns: [], commands: [], attempts: [], questions: [] } as never);
    expect(entries.map(e => e.title)).toEqual(["Branch updated with its base; CI runs again", "GitHub refused the merge", "Merged PR #42 into main"]);
    expect(entries[2].tone).toBe("good");
  });
});
