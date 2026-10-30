import { describe, expect, it } from "vitest";
import {
  ago, baseLine, cutoverSummary, hostSummary, isLive, linearOutcomes, liveChip, orderSessions, paritySummary, place, stateLabel, timeline, TYPED_TITLE,
  type Host, type RemoteaSession, type Run, type RunDetail, versionLine,
} from "./remotea-model";

const now = Date.parse("2026-10-28T12:00:00Z");
const host = (values: Partial<Host>): Host => ({
  id: "h", name: "desk", machine: "WSL", connected: false, last_seen_at: null, conn_epoch: 1, caps: [], harnesses: {}, proto: 1, ...values,
});
const session = (id: string, state: string, since: string): RemoteaSession => ({
  id, tool_label: "Claude Code", title: id, repo: "", branch: "", cwd: "", host: null, last_turn: null,
  filed: { record_title: null }, live_state: { state, since, host_id: "h", reason: "" },
});

describe("remotea model", () => {
  it("labels host-reported states and the Sessions chip", () => {
    expect(stateLabel("needs_you")).toBe("Needs you");
    expect(isLive("idle")).toBe(true);
    expect(isLive("exited")).toBe(false);
    expect(liveChip({ state: "working", since: null, host_id: null, reason: "" })).toEqual({ label: "Live", tone: "live" });
    expect(liveChip({ state: "lost", since: null, host_id: null, reason: "" })?.label).toBe("Restoring");
    expect(liveChip({ state: "exited", since: null, host_id: null, reason: "" })?.label).toBe("Ended");
    expect(liveChip(null)).toBeNull();
  });

  it("describes host connection state", () => {
    expect(hostSummary([], now).tone).toBe("never");
    expect(hostSummary([host({ connected: true })], now)).toEqual({ tone: "connected", text: "desk connected" });
    expect(hostSummary([host({ last_seen_at: "2026-10-28T11:55:00Z" })], now).text).toBe("Disconnected · last seen 5m ago");
    expect(ago(null)).toBe("never");
    expect(ago("2026-10-28T11:59:58Z", now)).toBe("just now");
    expect(ago("2026-10-26T12:00:00Z", now)).toBe("2d ago");
  });

  it("puts sessions that need attention first", () => {
    const ordered = orderSessions([
      session("idle", "idle", "2026-10-28T11:59:00Z"), session("old-work", "working", "2026-10-28T10:00:00Z"),
      session("ask", "needs_you", "2026-10-28T09:00:00Z"), session("done", "exited", "2026-10-28T11:59:30Z"),
      session("new-work", "working", "2026-10-28T11:00:00Z"),
    ]);
    expect(ordered.map(s => s.id)).toEqual(["ask", "new-work", "old-work", "idle", "done"]);
  });

  it("summarizes collector parity", () => {
    expect(paritySummary(null)).toMatch(/No parity reports/);
    const base = { first_report_at: null, last_report_at: null, recent: [] };
    expect(paritySummary({ ...base, total: 3, matched: 3, missing: 0, mismatched: 0, clean: true, fields: {} })).toBe("All 3 reported sessions match the current collector.");
    expect(paritySummary({ ...base, total: 4, matched: 2, missing: 1, mismatched: 1, clean: false, fields: { branch: 1 } }))
      .toBe("2 of 4 match, 1 differ, 1 unknown to the current collector. Most different: branch (1).");
    expect(place({ repo: "", branch: "main", cwd: "/home/davin/jarvis" })).toBe("jarvis · main");
  });

  it("shows when the collector is ready for cutover", () => {
    const base = { total: 1, matched: 1, missing: 0, mismatched: 0, clean: true, fields: {}, first_report_at: null, last_report_at: null, recent: [] };
    const day = (clean: boolean, reports = 3) => ({ day: "2026-10-28", reports, clean });
    expect(cutoverSummary(null)).toBeNull();
    expect(cutoverSummary(base)).toBeNull();
    const ready = cutoverSummary({ ...base, cutover: { ready: true, streak: 7, needed: 7, days: Array(7).fill(day(true)) } });
    expect(ready).toEqual({ ready: true, text: expect.stringContaining("Ready for cutover: 7 clean parity days") });
    expect(ready!.text).toContain("Retirement");
    expect(cutoverSummary({ ...base, cutover: { ready: false, streak: 3, needed: 7, days: [day(true), ...Array(6).fill(day(true))] } })!.text)
      .toBe("Not ready for cutover: 3 of 7 clean days in a row.");
    expect(cutoverSummary({ ...base, cutover: { ready: false, streak: 0, needed: 7, days: [day(false, 0), day(false, 0)] } })!.text)
      .toContain("no parity reports since yesterday");
  });

  it("reports Linear outcomes apart from the run", () => {
    expect(linearOutcomes({})).toEqual([]);
    const out = linearOutcomes({ linear: {
      in_progress: { status: "skipped" }, done: { status: "blocked", code: "LINEAR_CONFLICT", message: "Review the Linear difference" },
      comment: { status: "posted", url: "https://linear.app/c" } } });
    expect(out.map(o => o.text)).toEqual(["Done: not sent (Review the Linear difference)", "Completion comment: posted"]);
    expect(out.map(o => o.tone)).toEqual(["warn", "ok"]);
    expect(out[1].url).toBe("https://linear.app/c");
    expect(linearOutcomes({ linear: { done: { status: "deferred" } } })[0].text).toBe("Done: waits for the start write");
  });
});

describe("run freshness", () => {
  it("shows the base a run started from, behind live from the worktrees report with its time", () => {
    const base = { sha: "1a2b3c4d5e6f", ref: "origin/main", branch: "main", fetch: { ok: true } };
    expect(baseLine({ ...base, behind: 3, as_of: "2026-10-28T11:58:00Z", live: true }, now)).toBe("based on main @ 1a2b3c4 · 3 behind · as of 2m ago");
    // Before the first report: the spawn-time count.
    expect(baseLine({ ...base, behind: 0, as_of: "2026-10-28T11:59:55Z", live: false }, now)).toBe("based on main @ 1a2b3c4 · up to date · as of just now");
    expect(baseLine({ ...base, ref: "main", fetch: { ok: false, error: "offline" }, behind: null }, now)).toBe("based on local main (fetch failed) @ 1a2b3c4");
  });

  it("shows nothing for an older host that does not send the base", () => {
    expect(baseLine(null)).toBeNull();
    expect(baseLine(undefined)).toBeNull();
  });

  it("pairs the remotea version with the latest release", () => {
    expect(versionLine(host({ version: "0.3.0", update: null }))).toEqual({ text: "remotea 0.3.0", update: null, dirty: false });
    expect(versionLine(host({ version: "0.1.0", update: { latest: "0.3.0", url: "https://github.com/x/releases/tag/v0.3.0" } }))?.text)
      .toBe("remotea 0.1.0 · update available");
    expect(versionLine(host({}))).toBeNull();
  });

  it("shows the build beside the version, with a dirty badge", () => {
    const sha = "76c5f62" + "a".repeat(33);
    expect(versionLine(host({ version: "0.1.0", build_sha: sha, build: "76c5f62", build_dirty: true }))).toEqual(
      { text: "remotea 0.1.0 · build 76c5f62", update: null, dirty: true });
    expect(versionLine(host({ version: "0.1.0", build_sha: sha }))?.text).toBe("remotea 0.1.0 · build 76c5f62");
    expect(versionLine(host({ version: "0.1.0", build_sha: "unknown" }))?.text).toBe("remotea 0.1.0 · build unknown");
  });

  it("warns while a host's journal is degraded", () => {
    const degraded = host({ connected: true, degraded_reason: "journal_unavailable (write failed: ENOSPC)" });
    expect(hostSummary([degraded], now)).toEqual({ tone: "connected", text: "desk connected",
      warning: "Host journal degraded: journal_unavailable (write failed: ENOSPC)" });
    expect(hostSummary([degraded, host({ name: "laptop" })], now).warning).toBe("desk: Host journal degraded: journal_unavailable (write failed: ENOSPC)");
    expect(hostSummary([host({ connected: true, degraded_reason: null })], now).warning).toBeUndefined();
  });
});

describe("typed prompts in the timeline", () => {
  const run: Run = {
    id: "r1", task: { id: "t1", title: "Task", status: "in_progress" }, state: "executing", state_revision: 2, state_reason: "",
    reason_text: "", state_since: "2026-10-28T10:00:00Z", generation: 1, plan_revision: 1, tool: "claude", model: "",
    branch: "davin/eri-1-task", base_branch: "main", worktree: "", effective_permission_mode: "", turn_count: 1, turn_cap: 40,
    started_at: null, finished_at: null, stop_requested_at: null, stop_mode: "", surf: "surf-1",
    host: { id: "h", name: "desk", connected: true }, structured_answers: true, session: null, created_at: "2026-10-28T09:59:00Z",
  };
  const turn = (values: Partial<RunDetail["turns"][number]>): RunDetail["turns"][number] => ({
    id: "t", seq: 1, text: "", clipped: false, truncated: false, generation: 1, at: "2026-10-28T10:02:00Z", prompt: "", ...values,
  });
  const detail = (turns: RunDetail["turns"], history: RunDetail["history"] = []): RunDetail =>
    ({ ...run, plan: null, history, turns, commands: [], questions: [] });

  it("shows a message typed in the terminal as yours, before the agent's answer", () => {
    // A host that sends `prompt` frames: the typed message is its own row the moment it is typed.
    const entries = timeline(detail([
      turn({ id: "u1", role: "user", text: "Fix the flaky test", at: "2026-10-28T10:01:00Z" }),
      turn({ id: "a1", role: "assistant", text: "Fixed it.", at: "2026-10-28T10:02:00Z" }),
    ]));
    expect(entries.map(e => [e.title, e.who, e.detail])).toEqual([
      [TYPED_TITLE, "you", "Fix the flaky test"], ["Turn 1", "agent", "Fixed it."]]);
    expect(entries[0]).toMatchObject({ kind: "command", tone: "plain" });
  });

  it("shows the typed message from an older host's finished turn, once", () => {
    // Older hosts send the typed message only with the turn's Stop (`prompt`), so it appears at the turn's time.
    const entries = timeline(detail([turn({ text: "Fixed it.", prompt: "Fix the flaky test" })]));
    expect(entries.map(e => [e.title, e.who, e.at])).toEqual([
      [TYPED_TITLE, "you", "2026-10-28T10:02:00Z"], ["Turn 1", "agent", "2026-10-28T10:02:00Z"]]);
    expect(entries[0].detail).toBe("Fix the flaky test");
    // Nothing typed: no entry.
    expect(timeline(detail([turn({ text: "Fixed it." })])).map(e => e.title)).toEqual(["Turn 1"]);
  });

  it("does not repeat a text Eridani itself sent as something you typed", () => {
    const history = [{ at: "2026-10-28T10:00:30Z", event: "sent", text: "Use pg", actor: { kind: "human" } }];
    const entries = timeline(detail([turn({ text: "Switched to pg.", prompt: "Use pg" })], history));
    expect(entries.map(e => e.title)).toEqual(["Sent by you", "Turn 1"]);
  });
});
