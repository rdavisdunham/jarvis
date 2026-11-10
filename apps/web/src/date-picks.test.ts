import { describe, expect, it } from "vitest";
import { doLabel, monthTitle, nextMonday, quickPicks, shiftMonthKeepDay, weekendFrom } from "./date-picks";
import { monthDays } from "./workspace";

// 2026-11-09 is a Monday.
describe("date quick picks", () => {
  it("offers today, tomorrow, the coming Saturday and next Monday", () => {
    expect(quickPicks("2026-11-11").map(p => [p.label, p.date])).toEqual([
      ["Today", "2026-11-11"], ["Tomorrow", "2026-11-12"], ["This weekend", "2026-11-14"], ["Next week", "2026-11-16"],
    ]);
  });
  it("keeps the weekend on Saturday and Sunday", () => {
    expect(weekendFrom("2026-11-14")).toBe("2026-11-14");
    expect(weekendFrom("2026-11-15")).toBe("2026-11-15");
    expect(weekendFrom("2026-11-13")).toBe("2026-11-14");
  });
  it("next week is the following Monday, across a month end", () => {
    expect(nextMonday("2026-11-09")).toBe("2026-11-16");
    expect(nextMonday("2026-11-15")).toBe("2026-11-16");
    expect(nextMonday("2026-11-28")).toBe("2026-11-30");
    expect(nextMonday("2026-12-29")).toBe("2027-01-04");
  });
});

describe("month grid", () => {
  it("starts on Sunday and covers the whole month", () => {
    const days = monthDays("2026-11-20");
    expect(days[0]).toBe("2026-11-01"); // November 2026 begins on a Sunday
    const dec = monthDays("2026-12-03");
    expect(dec[0]).toBe("2026-11-29");
    expect(dec).toContain("2026-12-31");
    expect(monthTitle("2026-12-03")).toBe("December 2026");
  });
  it("Page Up/Down keeps the day, clamped to the month's end", () => {
    expect(shiftMonthKeepDay("2027-01-31", 1)).toBe("2027-02-28");
    expect(shiftMonthKeepDay("2026-12-15", 1)).toBe("2027-01-15");
    expect(shiftMonthKeepDay("2026-03-31", -1)).toBe("2026-02-28");
  });
});

describe("do date chips", () => {
  it("reads as Do today, a weekday this week, or a date", () => {
    expect(doLabel("2026-11-09", "2026-11-09")).toBe("Do today");
    expect(doLabel("2026-11-10", "2026-11-09")).toBe("Do tomorrow");
    expect(doLabel("2026-11-08", "2026-11-09")).toBe("Do yesterday");
    expect(doLabel("2026-11-12", "2026-11-09")).toBe("Do Thu");
    expect(doLabel("2026-11-20", "2026-11-09")).toBe("Do Nov 20");
    expect(doLabel("2027-01-04", "2026-11-09")).toBe("Do Jan 4, 2027");
  });
});
