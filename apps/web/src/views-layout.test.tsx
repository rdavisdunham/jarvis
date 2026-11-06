import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { HomePathChip, homePathText } from "./HomePath";
import { timelinePlacement } from "./nested-timeline";
import { boardColumns, boardMeaning, statusForColumn } from "./board-status";

describe("home path chips", () => {
  it("keep the whole path in the title and mark the text for start-side truncation", () => {
    const path = homePathText([{ title: "Business" }, { title: "Brainforge" }, { title: "ABC Home & Commercial" }, { title: "Andi" }]);
    expect(path).toBe("Business / Brainforge / ABC Home & Commercial / Andi");
    const html = renderToStaticMarkup(<HomePathChip path={path} className="detail-path"/>);
    expect(html).toContain('class="chip chip-home detail-path" title="Business / Brainforge / ABC Home &amp; Commercial / Andi"');
    expect(html).toContain('<span class="chip-text home-path"><bdi>Business / Brainforge / ABC Home &amp; Commercial / Andi</bdi></span>');
  });
});

describe("timeline placement", () => {
  const window = { rangeStart: "2026-11-05", today: "2026-11-05", days: 30, finished: false };
  it("draws an overdue task as a bar that runs on to today, clipped at the range start", () => {
    expect(timelinePlacement("2026-08-10", "2026-08-10", window)).toEqual({ kind: "bar", from: 0, to: 1, overdue: true, clipped: true });
    // Looking back a month: the bar starts at its date and still ends at today (past the window's end, so clamped).
    expect(timelinePlacement("2026-10-20", "2026-10-20", { ...window, rangeStart: "2026-10-06" }))
      .toEqual({ kind: "bar", from: 14, to: 30, overdue: true, clipped: false });
    // A window wholly after today has nothing to draw.
    expect(timelinePlacement("2026-08-10", "2026-08-10", { ...window, rangeStart: "2026-12-05" })).toEqual({ kind: "before", overdue: true });
  });
  it("leaves finished, current and future records on their own dates", () => {
    expect(timelinePlacement("2026-08-10", "2026-08-10", { ...window, finished: true })).toEqual({ kind: "before", overdue: false });
    expect(timelinePlacement("2026-11-07", "2026-11-10", window)).toEqual({ kind: "bar", from: 2, to: 6, overdue: false, clipped: false });
    expect(timelinePlacement("2026-11-01", "2026-11-08", window)).toEqual({ kind: "bar", from: 0, to: 4, overdue: false, clipped: true });
    expect(timelinePlacement("2026-11-05", "2026-11-05", window)).toEqual({ kind: "bar", from: 0, to: 1, overdue: false, clipped: false });
    expect(timelinePlacement("2027-01-01", "2027-01-01", window)).toEqual({ kind: "after", overdue: false });
    expect(timelinePlacement("", "", window)).toEqual({ kind: "unscheduled", overdue: false });
  });
});

describe("board status columns", () => {
  const statuses = [
    { id: "s-backlog", name: "Backlog", meaning: "backlog" },
    { id: "s-open", name: "Open", meaning: "open" },
    { id: "s-doing", name: "In progress", meaning: "in_progress" },
    { id: "s-done", name: "Done", meaning: "completed" },
  ];
  it("puts backlog cards in the Open column", () => {
    expect(boardMeaning("backlog")).toBe("open");
    expect(boardMeaning("open")).toBe("open");
    expect(boardMeaning("in_progress")).toBe("in_progress");
    expect(boardMeaning(undefined)).toBe("");
    const { columns, column } = boardColumns(statuses);
    expect(columns.map(s => s.name)).toEqual(["Open", "In progress", "Done"]);
    expect(column("s-backlog")).toBe("s-open");
    expect(column("s-doing")).toBe("s-doing");
    expect(column(null)).toBe("");
  });
  it("keeps a Backlog column for a workflow without an Open status", () => {
    const only = statuses.filter(s => s.meaning !== "open");
    expect(boardColumns(only).columns.map(s => s.name)).toEqual(["Backlog", "In progress", "Done"]);
    expect(boardColumns(only).column("s-backlog")).toBe("s-backlog");
  });
  it("writes Open (not Backlog) for a card dropped on Open from another column", () => {
    expect(statusForColumn(statuses, "open")).toBe("s-open");
    expect(statusForColumn(statuses, "in_progress")).toBe("s-doing");
    expect(statusForColumn(statuses.filter(s => s.meaning !== "open"), "open")).toBe("s-backlog");
    expect(statusForColumn(statuses, "waiting")).toBeUndefined();
  });
});
