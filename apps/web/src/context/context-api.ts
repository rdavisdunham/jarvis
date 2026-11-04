/** The only way the Meetings view and meeting review reach the server: the owner's session routes under
 * /api/v1/context (sources and drill-down from the context service, reviews from the review routes). */
import { api, ApiError, commandId, post } from "../api";
import type { Destination, Evidence, Review, ReviewSummary, SourceList, SourcePage, SourceStatus } from "./context-model";

const BASE = "/context";
/** The server refuses these when meeting memory does not apply here: it is off, or this is a shared workspace. */
export const UNAVAILABLE = new Set(["CONTEXT_DISABLED", "PERSONAL_WORKSPACE"]);
export const isUnavailable = (e: unknown) => e instanceof ApiError && UNAVAILABLE.has(e.code);
/** Someone (another device, a retry, reprocessing) changed what the reader approved against. */
export const isStale = (e: unknown) => e instanceof ApiError && ["REVISION_CONFLICT", "STALE_REVIEW", "CONFLICT"].includes(e.code);

const id = (value: string) => encodeURIComponent(value);

export function listMeetings({ status, offset = 0, limit = 30 }: { status?: SourceStatus | ""; offset?: number; limit?: number } = {}) {
  const query = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  if (status) query.set("status", status);
  return api<SourceList>(`${BASE}/sources?${query}`);
}
/** Per-meeting review counts. Optional for the list: it still renders without them. */
export async function reviewSummaries(): Promise<Map<string, ReviewSummary>> {
  const page = await api<{ items: ReviewSummary[] }>(`${BASE}/reviews?limit=100`);
  return new Map(page.items.map(item => [item.source_id, item]));
}
/** Whether meeting memory answers for this account; it decides whether Meetings appears in the navigation. */
export async function meetingsAvailable() {
  try { await api<SourceList>(`${BASE}/sources?limit=1`); return true; }
  catch (e) { if (isUnavailable(e)) return false; throw e; }
}

export const getReview = (sourceId: string) => api<Review>(`${BASE}/reviews/${id(sourceId)}`);
export function transcript(sourceId: string, start = 0, end?: number) {
  const query = new URLSearchParams({ segment_start: String(start) });
  if (end !== undefined) query.set("segment_end", String(end));
  return api<SourcePage>(`${BASE}/sources/${id(sourceId)}/segments?${query}`);
}
export const evidence = (assertionId: string) => api<Evidence>(`${BASE}/evidence/${id(assertionId)}`);

// ---------------------------------------------------------------- writes

type Selection = { proposal_id: string; revision: number }[];
export type WriteResult = { review?: Review };
/** A retry after a lost connection reuses the command ID of the identical request, so the server can tell. */
const inFlight = new Map<string, string>();
async function write(path: string, body: Record<string, unknown>): Promise<WriteResult> {
  const key = path + JSON.stringify(body);
  const command_id = inFlight.get(key) ?? commandId();
  inFlight.set(key, command_id);
  try {
    const result = await post<WriteResult>(path, { ...body, command_id });
    inFlight.delete(key);
    return result ?? {};
  } catch (e) {
    if (!(e instanceof ApiError) || e.code !== "NETWORK") inFlight.delete(key);
    throw e;
  }
}
const review = (sourceId: string, action: string) => `${BASE}/reviews/${id(sourceId)}/${action}`;

export const approve = (sourceId: string, destination: Destination, items: Selection) => write(review(sourceId, "approve"), { destination, items });
export const applyUpdates = (sourceId: string, items: Selection) => write(review(sourceId, "apply-updates"), { items });
export const dismiss = (sourceId: string, items: Selection) => write(review(sourceId, "dismiss"), { items });
export const later = (sourceId: string, items: Selection = []) => write(review(sourceId, "later"), { items });
export const done = (sourceId: string, reviewRevision: number, items: Selection) =>
  write(review(sourceId, "done"), { review_revision: reviewRevision, items });
export const wrong = (sourceId: string, assertionId: string, expectedRevision: number, reason = "") =>
  write(review(sourceId, "wrong"), { assertion_id: assertionId, expected_revision: expectedRevision, reason });
export const changed = (sourceId: string, assertionId: string, expectedRevision: number, statement: string, effective: string) =>
  write(review(sourceId, "changed"), { assertion_id: assertionId, expected_revision: expectedRevision, statement, effective_at: effective });
export const speaker = (sourceId: string, linkId: string, expectedRevision: number, action: "confirm" | "reject" | "undo", decisionId?: string | null) =>
  write(review(sourceId, "speakers"), { link_id: linkId, expected_revision: expectedRevision, action, ...(decisionId ? { decision_id: decisionId } : {}) });
