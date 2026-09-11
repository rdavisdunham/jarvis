import { describe, it, expect } from "vitest";
import {
  monthDays,
  shiftMonth,
  localDateTime,
  matchesStatus,
} from "./workspace";
describe("calendar boundaries", () => {
  it("lays out a leap month and moves across years", () => {
    const days = monthDays("2028-02-29");
    expect(days).toHaveLength(42);
    expect(days).toContain("2028-02-29");
    expect(new Date(days[0] + "T12:00:00Z").getUTCDay()).toBe(0);
    expect(shiftMonth("2026-12-31", 1)).toBe("2027-01-01");
    expect(shiftMonth("2026-01-31", -1)).toBe("2025-12-01");
  });
  it("uses the record timezone when editing times", () => {
    expect(localDateTime("2026-11-01T06:30:00Z", "America/Chicago")).toBe(
      "2026-11-01T01:30",
    );
    expect(localDateTime("2026-11-01T08:30:00Z", "America/Chicago")).toBe(
      "2026-11-01T02:30",
    );
  });
  it("open includes pending work but excludes completion and cancellation", () => {
    expect(matchesStatus("in_progress", "open")).toBe(true);
    expect(matchesStatus("finished", "open")).toBe(true);
    expect(matchesStatus("completed", "open")).toBe(false);
    expect(matchesStatus("cancelled", "open")).toBe(false);
  });
});
