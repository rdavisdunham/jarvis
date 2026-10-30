import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { RemovalReview } from "./WorktreeMaintenance";
import { removalPlan, worktreeClass, worktreeCounts, worktreeKey, worktreeLoss, type WorktreeRow } from "./remotea-model";

const row = (over: Partial<WorktreeRow> = {}): WorktreeRow => ({
  host_id: "h1", host_name: "desk", repo: "/home/davin/jarvis", target: "jarvis", path: "/wt/done", branch: "task/done",
  base: "main", base_ref: "origin/main", head: "a".repeat(40), active: false, ahead: 2, behind: 0, unpushed: 0,
  upstream: "origin/task/done", merged: true, uncommitted: 0, untracked: 0, ignored: 0, candidate: true, reasons: [],
  writer: null, run: null,
  cleanup: ["git -C '/home/davin/jarvis' update-ref refs/remotea/recovery/task--done " + "a".repeat(40),
    "git -C '/home/davin/jarvis' worktree remove '/wt/done'"],
  ...over,
});

describe("worktreeClass", () => {
  it("reclaims only a clean candidate with no writer", () => {
    expect(worktreeClass(row())).toBe("reclaimable");
    expect(worktreeClass(row({ merged: null, merged_reason: "no commits of its own yet", ahead: 0 }))).toBe("reclaimable");
  });

  it("puts a live writer or an unsettled run first", () => {
    expect(worktreeClass(row({ writer: { surf: "s1", run_id: "r1" } }))).toBe("active");
    expect(worktreeClass(row({ active: true }))).toBe("active");
    expect(worktreeClass(row({ run: { id: "r1", state: "working", settled: false, task_id: "t", task_title: "T" } }))).toBe("active");
    expect(worktreeClass(row({ run: { id: "r1", state: "completed", settled: true, task_id: "t", task_title: "T" } }))).toBe("reclaimable");
  });

  it("separates dirty, unpushed, unmerged and merged-but-kept rows", () => {
    expect(worktreeClass(row({ uncommitted: 2, candidate: false, unpushed: 3 }))).toBe("dirty");
    expect(worktreeClass(row({ untracked: 1, candidate: false }))).toBe("dirty");
    expect(worktreeClass(row({ unpushed: 1, merged: false, candidate: false }))).toBe("unpushed");
    expect(worktreeClass(row({ upstream: null, merged: false, candidate: false }))).toBe("unpushed");
    expect(worktreeClass(row({ merged: false, candidate: false }))).toBe("unmerged");
    expect(worktreeClass(row({ ignored: 4, candidate: false, reasons: ["4 ignored file(s) (e.g. .env) would be deleted"] }))).toBe("merged");
  });

  it("treats a gone folder as reclaimable and an older host's rows as unknown", () => {
    expect(worktreeClass(row({ reasons: ["folder missing (git worktree prune)"], candidate: false, merged: null }))).toBe("reclaimable");
    const { writer: _omit, ...older } = row();
    expect(worktreeClass(older as WorktreeRow)).toBe("unknown");
    expect(worktreeClass(row({ uncommitted: null }))).toBe("unknown");
    expect(worktreeClass(row({ candidate: false, merged: null, reasons: ["detached HEAD"] }))).toBe("unknown");
  });

  it("counts every class", () => {
    const counts = worktreeCounts([row(), row({ uncommitted: 1, candidate: false }), row({ active: true })]);
    expect(counts).toMatchObject({ reclaimable: 1, dirty: 1, active: 1, unpushed: 0, unknown: 0 });
  });
});

describe("removal review", () => {
  const dirty = row({ path: "/wt/dirty", branch: "task/dirty", uncommitted: 3, untracked: 1, ignored: 2, unpushed: 1,
    merged: false, candidate: false, cleanup: ["git -C x worktree remove '/wt/dirty'"] });
  const live = row({ path: "/wt/live", branch: "task/live", writer: { surf: "surf-9", run_id: "r9" } });

  it("says what a refused row would lose", () => {
    expect(worktreeLoss(dirty)).toEqual(["3 uncommitted changes", "1 untracked file", "2 ignored files (e.g. .env), deleted by a removal",
      "1 commit not on origin"]);
    expect(worktreeLoss(live)).toEqual(["a live session writes here (surf-9)"]);
  });

  it("produces commands only for selected reclaimable rows, never for refused ones", () => {
    const rows = [row(), dirty, live, row({ host_id: "h2", host_name: "laptop", path: "/wt/b", branch: "task/b",
      cleanup: ["git -C '/r' worktree prune"], reasons: ["folder missing (git worktree prune)"], candidate: false, merged: null })];
    const everything = new Set(rows.map(worktreeKey));  // Even a selection that names refused rows.
    const plan = removalPlan(rows, everything);
    expect(plan.reclaimable.map(r => r.path)).toEqual(["/wt/done", "/wt/b"]);
    expect(plan.refused.map(r => r.row.path)).toEqual(["/wt/dirty", "/wt/live"]);
    expect(plan.commands).toEqual([
      "# On desk (in WSL): 1 worktree", "# jarvis task/done", ...row().cleanup!,
      "# On laptop (in WSL): 1 worktree", "# jarvis task/b", "git -C '/r' worktree prune",
    ]);
    expect(plan.commands.join("\n")).not.toContain("/wt/dirty");
    expect(removalPlan(rows, new Set([worktreeKey(row())])).commands).toHaveLength(4);
  });

  it("a reclaimable-looking row without commands is not offered", () => {
    expect(removalPlan([row({ cleanup: null })], new Set([worktreeKey(row())])).reclaimable).toEqual([]);
  });

  it("renders checkboxes for reclaimable rows only, the refusals and the commands", () => {
    const html = renderToStaticMarkup(<RemovalReview rows={[row(), dirty, live]} onClose={() => {}} initialProduced/>);
    expect(html.match(/type="checkbox"/g)).toHaveLength(1);
    expect(html).toContain("Not offered: 2 worktrees would lose work or are in use");
    expect(html).toContain("Would lose: 3 uncommitted changes; 1 untracked file");
    expect(html).toContain("Would lose: a live session writes here (surf-9)");
    expect(html).toContain("Produce commands (1)");
    expect(html).toContain("worktree remove &#x27;/wt/done&#x27;");
    expect(html).not.toContain("worktree remove &#x27;/wt/dirty&#x27;");
    expect(html).toContain("remotea has no removal command");
  });

  it("offers nothing when no row is reclaimable", () => {
    const html = renderToStaticMarkup(<RemovalReview rows={[dirty]} onClose={() => {}} initialProduced/>);
    expect(html).toContain("Nothing is reclaimable right now.");
    expect(html).not.toContain("type=\"checkbox\"");
    expect(html).not.toContain("textarea");
  });
});
