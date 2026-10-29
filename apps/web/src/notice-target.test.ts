import { describe, expect, it } from "vitest";
import { noticeDestination, type NoticeDestination } from "./notice-target";
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
