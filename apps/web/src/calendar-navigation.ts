import type { CalendarEntry } from "./types";

/** Resolve only current day entries; a stale exact selection must never fall back. */
export function calendarDetailEntry(
  items: CalendarEntry[],
  entityId: string,
  entryId?: string,
): CalendarEntry {
  const matches = items.filter((item) =>
    item.entity_id === entityId && (!entryId || item.id === entryId),
  );
  if (!matches.length)
    throw new Error("That calendar entry is no longer on this day. Refresh and choose it again.");
  if (matches.length > 1)
    throw new Error("Several calendar entries match this item. Choose the exact occurrence before opening details.");
  return matches[0];
}
