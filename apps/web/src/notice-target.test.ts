import { describe, expect, it } from "vitest";
import { noticeDestination, noticeStatus, type NoticeDestination } from "./notice-target";
import cases from "./notice-targets.json";
import type { Notice } from "./types";

describe("notification destinations", () => {
  // tests/test_drift_guards.py checks worker.notification_url against the same rows' `url`.
  it.each(cases.map(c => [c.name, c] as const))("%s opens where its push does", (_, c) => {
    expect(noticeDestination(c.notification as Pick<Notice, "task_id" | "target">)).toEqual(c.open as NoticeDestination);
  });
  it("opens the run, not the task card, for an agent notification", () => {
    expect(noticeDestination({ task_id: "t", target: { view: "remotea", run_id: "r" } })).toEqual({ open: "run", run: "r" });
  });
});

describe("notification status line", () => {
  const when = (iso: string) => "at " + iso;
  it("keeps snoozed, delivered and read apart", () => {
    expect(noticeStatus({ state: "snoozed", snoozed_until: "T9", importance: "normal" }, when)).toEqual({ priority: null, state: "Snoozed until at T9" });
    expect(noticeStatus({ state: "delivered", importance: "high" }, when)).toEqual({ priority: "High priority", state: "Delivered" });
    expect(noticeStatus({ state: "read", importance: "urgent" }, when)).toEqual({ priority: "Urgent", state: null });
    expect(noticeStatus({}, when)).toEqual({ priority: null, state: null });
  });
});
