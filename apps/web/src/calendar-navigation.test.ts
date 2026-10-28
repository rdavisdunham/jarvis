import { describe, expect, it } from "vitest";
import { calendarDetailEntry } from "./calendar-navigation";
import type { CalendarEntry } from "./types";

const morning = { id: "series:09", entity_id: "series", date: "2026-10-27", occurrence_start: "2026-10-27T09:00:00Z" } as CalendarEntry;
const evening = { ...morning, id: "series:17", occurrence_start: "2026-10-27T17:00:00Z" };

describe("calendar detail navigation", () => {
  it("opens the chosen occurrence when several share an entity and day", () => {
    expect(calendarDetailEntry([morning, evening], "series", evening.id)).toBe(evening);
  });
  it("requires disambiguation when an old command only specifies a shared entity", () => {
    expect(() => calendarDetailEntry([morning, evening], "series")).toThrow("exact occurrence");
  });
  it("does not fall back when the chosen occurrence disappears or moves", () => {
    expect(() => calendarDetailEntry([morning], "series", evening.id)).toThrow("no longer");
  });
  it("requires the entry and entity identities to agree", () => {
    expect(() => calendarDetailEntry([morning], "another-record", morning.id)).toThrow("no longer");
  });
  it("preserves single-entry commands without an exact projection id", () => {
    expect(calendarDetailEntry([morning], "series")).toBe(morning);
  });
});
