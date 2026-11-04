/** Meeting memory types and pure helpers for the Meetings view and the meeting review (no fetching here). */

export type SourceStatus = "received" | "queued" | "processing" | "partial" | "ready" | "failed" | "awaiting_input" | "deleted";
export type Coverage = "extracted" | "no_content" | "skipped" | "failed";
export type MeetingSource = {
  id: string; connector: string; label: string; external_id: string; title: string; kind: string; status: SourceStatus;
  availability: string; revision: number; occurred_at: string | null; calendar_event_id: string | null; segment_count: number;
  processing: { stage: string; status: string; coverage: Record<string, Coverage>; attempts: number; updated_at: string } | null;
  last_error: { stage: string; status: string; message: string | null; retryable: boolean; at: string } | null;
  created_at: string; updated_at: string;
};
export type SourceList = { items: MeetingSource[]; total: number; limit: number; offset: number };

export type ReviewStatus = "pending" | "later" | "done";
export type ReviewSummary = { source_id: string; status: ReviewStatus; pending: number; later: number; published?: number };

export type EntityRef = { entity_type: string; entity_id?: string; mention?: string };
export type Citation = { source_id: string; segment: number; [key: string]: unknown };
/** One extracted statement, in the context service's assertion item shape. */
export type ContextItem = {
  id: string; revision: number; kind: string; statement: string;
  status: "current" | "contested" | "historical" | "retracted" | "future" | string;
  basis: "stated" | "inferred"; modality?: string; valid_from: string | null; valid_to: string | null;
  due_at?: string | null; responsible?: unknown[]; citation: Citation | null; entities: EntityRef[];
};
export type BriefClaim = { text: string; assertion_ids: string[]; citations: { assertion_id: string; source_id: string; segment: number }[] };
export type BriefSection = { name: string; claims: BriefClaim[]; gaps?: string[] };
export type Brief = { sections: BriefSection[]; gaps: string[] };
export type LinkStatus = "exact" | "inferred" | "confirmed" | "rejected" | "unresolved";
export type Speaker = { label: string; actor_id: string | null; actor_name: string | null; status: LinkStatus;
  link_id: string | null; link_revision: number | null; decision_id: string | null };
export type ProposalKind = "create" | "update" | "link" | "suggestion";
export type Disposition = "pending" | "selected" | "approved" | "dismissed" | "later" | "published" | "superseded";
export type PublicationStatus = "queued" | "in_progress" | "published" | "retrying" | "failed" | "conflict" | "unconfirmed";
export type Destination = "eridani" | "linear";
export type Proposal = {
  id: string; revision: number; kind: ProposalKind; title: string; description?: string;
  assignee: { id: string; name: string } | null; project: { id: string; name: string } | null; due_at: string | null;
  reason: string; preselected: boolean; disposition: Disposition; destination: Destination;
  evidence: { quote: string; segment: number | null; speaker?: string | null }[];
  target: { id: string; title: string; entity_type?: string } | null;
  before: Record<string, unknown> | null; after: Record<string, unknown> | null;
  publication: { id: string; status: PublicationStatus; task_id: string | null; error?: string | null } | null;
};
export type Review = {
  source_id: string; source?: MeetingSource; revision: number; status: ReviewStatus; destination: Destination; destinations: Destination[];
  brief: Brief | null; items: ContextItem[]; speakers: Speaker[]; proposals: Proposal[]; names?: Record<string, string>;
  /** A processing job for this meeting is waiting or running (for example after Reprocess). */
  busy?: boolean;
};
export type TranscriptSegment = { seq: number; kind: string; speaker: string | null; attribution: string | null;
  speaker_actor_id: string | null; start_ms: number | null; end_ms: number | null; text: string;
  supports: { assertion_id: string; kind: string; status: string; superseded_by?: string | null }[] };
export type SourcePage = {
  source: { id: string; title: string; occurred_at: string | null; revision: number; current_revision: number; segment_count: number;
    participants: { name?: string; role?: string; speaker_label?: string }[] };
  segments: TranscriptSegment[]; range: { start: number; end: number }; next_segment: number | null; gaps: string[];
};
export type EvidenceSpan = { evidence_id: string; role: "support" | "contradiction"; origin: string; source_id: string;
  source_title: string; occurred_at: string | null; segment: number; speaker: string | null; quote: string; current_revision: boolean };
export type Evidence = ContextItem & { evidence: EvidenceSpan[]; gaps: string[] };

// ---------------------------------------------------------------- the list

export type Tone = "neutral" | "busy" | "ok" | "warn" | "error";
export type StatusView = { label: string; tone: Tone; detail: string };

const STAGES: Record<string, string> = { ingest: "reading the transcript", extract: "finding decisions and work",
  link: "matching people and projects", reconcile: "checking against what you already know", embed: "indexing for search",
  brief: "writing briefs", lint: "checking consistency", proposals: "preparing task proposals" };
export const stageLabel = (stage: string) => STAGES[stage] ?? stage.replaceAll("_", " ");

export function coverageCounts(coverage: Record<string, Coverage> | undefined) {
  const counts = { extracted: 0, no_content: 0, skipped: 0, failed: 0, total: 0 };
  for (const value of Object.values(coverage ?? {})) { if (value in counts) counts[value]++; counts.total++; }
  return counts;
}

/** What the meeting's processing state means for the reader, including partial coverage and the failure kind. */
export function meetingStatus(source: MeetingSource): StatusView {
  const error = source.last_error;
  switch (source.status) {
    case "received": case "queued": return { label: "Queued", tone: "neutral", detail: "Waiting to be read." };
    case "processing": {
      const stage = source.processing?.stage;
      const retry = error?.retryable ? " Retrying after a problem while " + stageLabel(error.stage) + "." : "";
      return { label: "Processing", tone: "busy", detail: (stage ? "Now " + stageLabel(stage) + "." : "Being read.") + retry };
    }
    case "ready": return { label: "Ready", tone: "ok", detail: "Read in full." };
    case "partial": {
      const c = coverageCounts(source.processing?.coverage);
      const missed = [c.skipped && `${c.skipped} skipped`, c.failed && `${c.failed} failed`].filter(Boolean).join(", ");
      const detail = c.total ? `${c.extracted + c.no_content} of ${c.total} passages read; ${missed}.` : "Some passages could not be read.";
      return { label: "Partly read", tone: "warn", detail };
    }
    case "awaiting_input": return { label: "Paused", tone: "warn", detail: "Stopped at this meeting's processing budget. What was read so far is available." };
    case "failed": return { label: failureKind(error), tone: "error",
      detail: (error?.message ? error.message + " " : "") + (error?.retryable ? "Eridani will retry." : "Eridani stopped trying.") };
    case "deleted": return { label: "Deleted", tone: "neutral", detail: "Removed from meeting memory." };
    default: return { label: String(source.status), tone: "neutral", detail: "" };
  }
}
export function failureKind(error: MeetingSource["last_error"]) {
  if (!error) return "Failed";
  return ({ ingest: "Could not read", extract: "Extraction failed", link: "Matching failed", reconcile: "Reconciling failed",
    embed: "Indexing failed", brief: "Brief failed", proposals: "Proposals failed" } as Record<string, string>)[error.stage] ?? "Failed";
}

export function meetingDate(value: string | null, zone?: string) {
  if (!value) return "Undated";
  try { return new Date(value).toLocaleString([], { weekday: "short", month: "short", day: "numeric", hour: "numeric", minute: "2-digit", ...(zone ? { timeZone: zone } : {}) }); }
  catch { return new Date(value).toLocaleString(); }
}

/** `?source=` and `?actor=` deep-link IDs; anything that is not an ID opens the list instead. */
export function linkedId(search: string, name: "source" | "actor") {
  const value = new URLSearchParams(search).get(name);
  return value && /^[A-Za-z0-9_-]{1,100}$/.test(value) ? value : null;
}

// ---------------------------------------------------------------- the review

export const KIND_LABELS: Record<string, string> = { commitment: "Commitments", decision: "Decisions", rationale: "Reasons",
  fact: "Facts", risk: "Risks", open_question: "Open questions", preference: "Preferences", procedure: "How things are done",
  relationship: "Relationships" };
const KIND_ORDER = Object.keys(KIND_LABELS);
export const SECTION_LABELS: Record<string, string> = { current_state_and_why: "What was decided and why", open_loops: "Open work",
  next_meeting_prep: "Before the next meeting", who_is_who: "Who is who", recent_changes: "What changed", unknowns: "Still unknown" };

export function groupItems(items: ContextItem[]) {
  const groups = new Map<string, ContextItem[]>();
  for (const item of items) groups.set(item.kind, [...(groups.get(item.kind) ?? []), item]);
  const rank = (kind: string) => { const i = KIND_ORDER.indexOf(kind); return i < 0 ? KIND_ORDER.length : i; };
  return [...groups.entries()].sort(([a], [b]) => rank(a) - rank(b) || a.localeCompare(b))
    .map(([kind, rows]) => ({ kind, label: KIND_LABELS[kind] ?? kind.replaceAll("_", " "), items: rows }));
}

export type ProposalGroup = "tasks" | "waiting" | "updates" | "suggestions";
/** Someone else's work from the meeting: the server's reason reads "Waiting on <name>". */
export const waitingOn = (p: Proposal) => /^waiting on\b/i.test(p.reason ?? "");
export const proposalGroup = (p: Proposal): ProposalGroup => p.kind === "update" ? "updates" : p.kind === "suggestion" ? "suggestions"
  : waitingOn(p) ? "waiting" : "tasks";
/** A proposal can still be chosen: undecided, deferred, or selected, and not yet sent anywhere. */
export const selectable = (p: Proposal) => ["pending", "selected", "later"].includes(p.disposition) && !p.publication;

/** The server's preselection (never for work others owe), keeping the reader's choices for proposals whose revision did not change. */
export function nextSelection(previous: Set<string>, before: Proposal[] | null, after: Proposal[]) {
  const old = new Map((before ?? []).map(p => [p.id, p.revision]));
  return new Set(after.filter(p => selectable(p)
    && (old.get(p.id) === p.revision ? previous.has(p.id) : p.preselected && proposalGroup(p) !== "waiting")).map(p => p.id));
}
export function chosen(proposals: Proposal[], selected: Set<string>, group: ProposalGroup | "approvable") {
  return proposals.filter(p => selected.has(p.id) && selectable(p)
    && (group === "approvable" ? proposalGroup(p) !== "updates" : proposalGroup(p) === group))
    .map(p => ({ proposal_id: p.id, revision: p.revision }));
}
export const pendingCount = (review: Review) => review.proposals.filter(selectable).length;

export const PUBLICATION_LABELS: Record<PublicationStatus, string> = { queued: "Queued", in_progress: "Sending", published: "Added",
  retrying: "Retrying", failed: "Failed", conflict: "Needs a look", unconfirmed: "Unconfirmed" };
/** Publications the reader can send again (conflict answers with what to check on the task). */
export const RETRYABLE: PublicationStatus[] = ["failed", "conflict", "unconfirmed"];
export const DESTINATION_LABELS: Record<Destination, string> = { eridani: "Eridani", linear: "Linear" };

/** Segment numbers that support each statement, from the transcript's own annotations. */
export function supportedBy(segments: TranscriptSegment[], assertionId: string) {
  return segments.filter(s => s.supports.some(x => x.assertion_id === assertionId)).map(s => s.seq);
}
/** A date input's value (YYYY-MM-DD, the reader's day) as the start of that day in their zone. */
export function effectiveAt(day: string) {
  const value = new Date(day + "T00:00:00");
  return Number.isNaN(value.getTime()) ? null : value.toISOString();
}
export const timecode = (ms: number | null) => {
  if (ms === null || ms < 0) return "";
  const s = Math.floor(ms / 1000), h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = String(s % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${r}` : `${m}:${r}`;
};
