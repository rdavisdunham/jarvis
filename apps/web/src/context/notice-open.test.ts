import { describe, expect, it, vi } from "vitest";
import cases from "../notice-targets.json";
import { noticeDestination, openNotice, type NoticeOpeners } from "../notice-target";
import type { Notice } from "../types";

const openers = (): NoticeOpeners & Record<string, ReturnType<typeof vi.fn>> =>
  ({ run: vi.fn(), meetings: vi.fn(), task: vi.fn(), activity: vi.fn(), today: vi.fn() });

describe("the Notifications page's Open", () => {
  it("opens a meeting review, or the meetings list without a source", () => {
    const ready = openers();
    openNotice({ open: "meetings", source: "src-1" }, ready);
    expect(ready.meetings).toHaveBeenCalledWith("src-1");
    const digest = openers();
    openNotice({ open: "meetings", source: null }, digest);
    expect(digest.meetings).toHaveBeenCalledWith(null);
    expect(digest.today).not.toHaveBeenCalled();
  });

  it("calls the opener named by each shared notification case", () => {
    for (const c of cases as { name: string; notification: Pick<Notice, "task_id" | "target">; open: { open: string } & Record<string, unknown> }[]) {
      const open = openers();
      openNotice(noticeDestination(c.notification), open);
      const called = Object.keys(open).filter(k => open[k].mock.calls.length);
      expect(called, c.name).toEqual([c.open.open === "notifications" ? "today" : c.open.open]);
      const [arg] = Object.entries(c.open).filter(([k]) => k !== "open").map(([, v]) => v);
      if (arg !== undefined) expect(open[called[0]], c.name).toHaveBeenCalledWith(arg);
    }
  });
});
