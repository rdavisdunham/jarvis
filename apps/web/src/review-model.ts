/** Review cadence: pure helpers for the type setting, due labels and the Atlas lens.
 * The server owns scheduling; these only read its fields and build schema drafts. */
import type { SchemaType } from "./structure-types";
import { shortDate } from "./work-views";

export type ReviewEvery = "1w" | "2w" | "1m" | "3m" | "6m" | "1y";
export type ReviewSetting = { enabled: boolean; every: ReviewEvery };
export const REVIEW_INTERVALS: [ReviewEvery, string][] = [
  ["1w", "Every week"], ["2w", "Every 2 weeks"], ["1m", "Every month"],
  ["3m", "Every 3 months"], ["6m", "Every 6 months"], ["1y", "Every year"],
];
export const REVIEW_GLYPH: [string, string] = ["↻", "Review cadence"];
export const DEFAULT_REVIEW: ReviewSetting = { enabled: false, every: "1m" };

export const reviewOf = (t: Pick<SchemaType, "review"> | undefined | null): ReviewSetting => ({ ...DEFAULT_REVIEW, ...(t?.review ?? {}) });
export const reviews = (t: Pick<SchemaType, "review" | "archived"> | undefined | null) => !!t && !t.archived && reviewOf(t).enabled;
export const intervalLabel = (every: ReviewEvery) => REVIEW_INTERVALS.find(([id]) => id === every)?.[1] ?? "Every month";

/** The draft patch for the type editors: toggling keeps the chosen interval; picking an interval keeps the toggle. */
export function reviewPatch(t: Pick<SchemaType, "review">, change: Partial<ReviewSetting>): Pick<SchemaType, "review"> {
  return { review: { ...reviewOf(t), ...change } };
}

/** The local calendar day of an instant, as YYYY-MM-DD. */
export function localDay(iso: string, zone?: string): string {
  const parts = Object.fromEntries(new Intl.DateTimeFormat("en-CA", { timeZone: zone, year: "numeric", month: "2-digit", day: "2-digit" })
    .formatToParts(new Date(iso)).map(p => [p.type, p.value]));
  return `${parts.year}-${parts.month}-${parts.day}`;
}

export type ReviewFields = { last_reviewed_at?: string | null; next_review_at?: string | null; review_paused?: boolean; review_due?: boolean; review_every?: ReviewEvery | null };
/** "Last reviewed …" and "Next review …" as two separate phrases (never a joined meta string). */
export function reviewLabels(r: ReviewFields, today: string, zone?: string): { last: string; next: string; tone: "due" | "paused" | "scheduled" } {
  const last = r.last_reviewed_at ? shortDate(localDay(r.last_reviewed_at, zone), today) : "Not yet";
  if (r.review_paused) return { last, next: "Paused for this record", tone: "paused" };
  if (!r.next_review_at) return { last, next: "Not scheduled", tone: "scheduled" };
  const day = localDay(r.next_review_at, zone);
  if (r.review_due) return { last, next: day < today ? "Due since " + shortDate(day, today).replace(/^Yesterday$/, "yesterday") : "Due now", tone: "due" };
  return { last, next: shortDate(day, today), tone: "scheduled" };
}

/** Atlas Reviews lens: due records and the regions that contain them stay lit; everything else dims. */
export const reviewLensDim = (due: boolean | undefined, inside: number) => !due && inside === 0;
