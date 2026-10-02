import { describe, expect, it } from "vitest";
import { clockLabel, dueBucket, shortDate } from "./work-views";

describe("work date presentation", () => {
  it("buckets due dates relative to today", () => {
    expect(dueBucket("2026-10-01", "2026-10-02")).toBe("overdue");
    expect(dueBucket("2026-10-02", "2026-10-02")).toBe("today");
    expect(dueBucket("2026-10-09", "2026-10-02")).toBe("upcoming");
    expect(dueBucket(null, "2026-10-02")).toBe("none");
  });
  it("labels nearby and distant days", () => {
    expect(shortDate("2026-10-02", "2026-10-02")).toBe("Today");
    expect(shortDate("2026-10-03", "2026-10-02")).toBe("Tomorrow");
    expect(shortDate("2026-10-01", "2026-10-02")).toBe("Yesterday");
    expect(shortDate("2026-10-12", "2026-10-02")).toBe("Oct 12");
    expect(shortDate("2027-01-01", "2026-10-02")).toBe("Jan 1, 2027");
  });
  it("formats clock times in lower-case twelve-hour form", () => {
    expect(clockLabel("15:00")).toBe("3:00 pm");
    expect(clockLabel("00:05:00")).toBe("12:05 am");
    expect(clockLabel("09:30")).toBe("9:30 am");
  });
});
