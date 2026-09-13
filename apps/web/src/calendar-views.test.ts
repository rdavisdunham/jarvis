import { describe, expect, it } from "vitest";
import { calendarRange, shiftDate, weekDays } from "./workspace";
describe("calendar date ranges", () => {
  it("uses one exclusive day, including leap days", () => {
    expect(calendarRange("2028-02-29", "day")).toEqual({
      days: ["2028-02-29"],
      from: "2028-02-29",
      end: "2028-03-01",
    });
    expect(shiftDate("2026-12-31", 1)).toBe("2027-01-01");
  });
  it("includes all seven days across month and year boundaries", () => {
    expect(weekDays("2027-01-01")).toEqual([
      "2026-12-27",
      "2026-12-28",
      "2026-12-29",
      "2026-12-30",
      "2026-12-31",
      "2027-01-01",
      "2027-01-02",
    ]);
    expect(calendarRange("2027-01-01", "week").end).toBe("2027-01-03");
  });
  it("keeps the month fetch range stable when only its selected day changes", () => {
    expect(calendarRange("2026-09-01", "month")).toEqual(
      calendarRange("2026-09-30", "month"),
    );
    expect(calendarRange("2026-09-01", "month").days).toHaveLength(42);
  });
});
