import { describe, expect, it } from "vitest";
import { ago, hostSummary, isLive, liveChip, orderSessions, paritySummary, place, stateLabel, type Host, type RemoteaSession } from "./remotea-model";

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
});
