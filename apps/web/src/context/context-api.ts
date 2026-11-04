/** The only way the Meetings view and meeting review reach the server: the owner's session routes under
 * /api/v1/context (sources and drill-down from the context service, reviews from the review routes). */
import { useEffect, useRef } from "react";
import { api, ApiError, commandId, post } from "../api";
import type { Destination, Evidence, Review, ReviewSummary, SourceList, SourcePage, SourceStatus } from "./context-model";

const BASE = "/context";
/** `events.ts` dispatches this when the change feed reports `context.changed` (processing, reviews, publication). */
export const CONTEXT_CHANGED = "eri-context-changed";

/** Calls `handler` once a burst of context changes settles (`delay` ms after the last one). */
export function useContextChanged(handler: () => void, delay = 1000) {
  const latest = useRef(handler);
  latest.current = handler;
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    const changed = () => { clearTimeout(timer); timer = setTimeout(() => latest.current(), delay); };
    window.addEventListener(CONTEXT_CHANGED, changed);
    return () => { window.removeEventListener(CONTEXT_CHANGED, changed); clearTimeout(timer); };
  }, [delay]);
}
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
const publicationRetry = (publicationId: string) => `${BASE}/reviews/publications/${id(publicationId)}/retry`;

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
/** Read the meeting again with the current prompt and model (human only; refused while it is still processing). */
export const reprocess = (sourceId: string) => write(review(sourceId, "reprocess"), {});
/** Send a failed, conflicting or unconfirmed publication again under its original identity (human only). */
export const retry = (publicationId: string) => write(publicationRetry(publicationId), {});
export const speaker = (sourceId: string, linkId: string, expectedRevision: number, action: "confirm" | "reject" | "undo", decisionId?: string | null) =>
  write(review(sourceId, "speakers"), { link_id: linkId, expected_revision: expectedRevision, action, ...(decisionId ? { decision_id: decisionId } : {}) });

// ---------------------------------------------------------------- person pages (D4)

import type { Brief } from "./context-model";

export type PersonCitation = { source_id: string; source_title: string | null; occurred_at: string | null; segment: number; speaker: string | null; quote: string };
export type PersonIdentity = { namespace: string; value: string; status: string };
export type PersonTask = { id: string; revision: number; title: string; status: string; due_date: string | null; completed_at: string | null };
export type PersonView = {
  actor: { id: string; name: string; kind: string; archived: boolean; revision: number; is_me: boolean };
  identities: (PersonIdentity & { id: string; source_scope: string | null })[];
  context: { assertion_id: string; revision: number; kind: string; statement: string; lifecycle: string; inferred: boolean;
    modality: string; valid_from: string | null; citation: PersonCitation }[];
  projects: { project_id: string; name: string; context_items: number; tasks: number }[];
  tasks: PersonTask[];
  meetings: { source_id: string; title: string | null; occurred_at: string | null }[];
  links_to_review: { link_id: string; revision: number; mention: string | null; status: string; role: string | null;
    assertion_id: string | null; score: number | null; statement: string | null }[];
  merges: { merge_id: string; winner_actor_id: string; loser_actor_id: string; winner_name: string | null; loser_name: string | null;
    undone_at: string | null; created_at: string | null }[];
  merge_candidates: { actor_id: string; name: string | null; reason: string; identities: PersonIdentity[] }[];
  gaps: string[];
};
export type LoopParty = { actor_id: string | null; name: string | null; is_me: boolean; resolved: boolean; collective: boolean };
export type OpenLoop = {
  assertion_id: string; revision: number; statement: string; bucket: "own" | "waiting_for" | "between_others" | "collective";
  responsible: LoopParty[]; recipient: { actor_id: string; name: string | null; is_me: boolean } | null; due_at: string | null;
  basis: "stated" | "inferred"; task_state: string; tasks?: PersonTask[]; citation: PersonCitation;
};
export type PersonBrief = Brief & { stale: boolean; built_at: string | null; notice?: string | null };
export type PersonChoice = { id: string; name: string; is_me: boolean; identities: PersonIdentity[] };
export type PersonResult = { person: PersonView };

const people = (actorId: string, action = "") => `${BASE}/people/${id(actorId)}${action}`;
export const person = (actorId: string) => api<PersonView>(people(actorId));
export const personBrief = (actorId: string) =>
  post<PersonBrief>(`${BASE}/brief`, { entity: { entity_type: "actor", entity_id: actorId } });
export const personLoops = (actorId: string) =>
  post<{ items: OpenLoop[]; gaps: string[] }>(`${BASE}/open-loops`, { entity: { entity_type: "actor", entity_id: actorId } });
export const findPeople = (q: string) => api<{ items: PersonChoice[] }>(`${BASE}/people?${new URLSearchParams({ q, limit: "10" })}`);

/** Identity corrections: human decisions, each with the client's command ID (reused on a retry). */
const personWrite = async (path: string, body: Record<string, unknown>) => (await write(path, body)) as PersonResult;
export const mergePeople = (actorId: string, winnerId: string, loserId: string) =>
  personWrite(people(actorId, "/merge"), { winner_actor_id: winnerId, loser_actor_id: loserId });
export const notSamePerson = (actorId: string, otherId: string) => personWrite(people(actorId, "/not-same"), { other_actor_id: otherId });
export const unmergePeople = (actorId: string, mergeId: string) => personWrite(people(actorId, "/unmerge"), { merge_id: mergeId });
export const settleLink = (actorId: string, linkId: string, expectedRevision: number, action: "confirm" | "reject") =>
  personWrite(people(actorId, `/links/${id(linkId)}`), { action, expected_revision: expectedRevision });
