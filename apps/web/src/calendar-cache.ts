import type { GoogleStatus } from "./GoogleSettings";
import type { CalendarEntry } from "./types";

/** One fetched /calendar range. `end` is exclusive, like the API. */
export type CalendarRangeData = {
  from: string;
  end: string;
  zone: string;
  items: CalendarEntry[];
  truncated: boolean;
  google?: GoogleStatus;
  warnings?: { calendar: string; reason: string }[];
};

const LIMIT = 12;

/** Newest first, one entry per range; a refetch of the same range replaces it. */
export function rememberRange(cache: CalendarRangeData[], data: CalendarRangeData) {
  const same = (c: CalendarRangeData) =>
    c.from === data.from && c.end === data.end && c.zone === data.zone;
  return [data, ...cache.filter((c) => !same(c))].slice(0, LIMIT);
}

/**
 * The rows to draw for a range right away: its own fetch, or the part of a wider range already
 * loaded (the month around a week or day), whichever is newer. Switching from
 * month to week or day then draws immediately while the exact range loads in the background.
 * Returns null only when nothing loaded covers the range.
 */
export function rangeView(
  cache: CalendarRangeData[],
  from: string,
  end: string,
  zone: string,
): (CalendarRangeData & { provisional: boolean }) | null {
  const isExact = (c: CalendarRangeData) => c.from === from && c.end === end && c.zone === zone;
  const exact = cache.find(isExact);
  // The newest load wins, so a month refreshed after an edit beats an older fetch of this week,
  // unless that month was cut short and may be missing rows the week has.
  const newest = cache.find((c) => c.zone === zone && c.from <= from && c.end >= end);
  const wider = newest && !isExact(newest) && !(newest.truncated && exact) ? newest : null;
  if (!wider) return exact ? { ...exact, provisional: false } : null;
  return {
    ...wider,
    from,
    end,
    items: wider.items.filter((e) => e.date >= from && e.date < end),
    // A wider range being too large says nothing about this smaller one; source warnings still apply.
    truncated: wider.truncated && !!wider.warnings?.length,
    provisional: true,
  };
}
