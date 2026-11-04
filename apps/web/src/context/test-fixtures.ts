import type { MeetingSource, Proposal, Review, SourcePage } from "./context-model";

export const source = (over: Partial<MeetingSource> = {}): MeetingSource => ({
  id: "src-1", connector: "upload", label: "upload", external_id: "x", title: "Pricing sync", kind: "meeting", status: "ready",
  availability: "available", revision: 1, occurred_at: "2026-11-03T15:00:00Z", calendar_event_id: null, segment_count: 3,
  processing: { stage: "proposals", status: "succeeded", coverage: { "0": "extracted", "1": "extracted", "2": "no_content" }, attempts: 1, updated_at: "2026-11-03T16:00:00Z" },
  last_error: null, created_at: "2026-11-03T15:30:00Z", updated_at: "2026-11-03T16:00:00Z", ...over,
});
export const proposal = (over: Partial<Proposal> = {}): Proposal => ({
  id: "p-own", revision: 1, kind: "create", title: "Send the pricing deck", assignee: { id: "me", name: "Me" }, project: null,
  due_at: "2026-11-07T00:00:00Z", reason: "You said you would send it.", preselected: true, disposition: "pending", destination: "eridani",
  evidence: [{ quote: "I'll send the deck Friday.", segment: 1, speaker: "Davin" }], target: null, before: null, after: null, publication: null, ...over,
});
export const review = (over: Partial<Review> = {}): Review => ({
  source_id: "src-1", source: source(), revision: 4, status: "pending", destination: "eridani", destinations: ["eridani", "linear"],
  brief: { sections: [{ name: "current_state_and_why", claims: [{ text: "Pricing moves to annual plans.", assertion_ids: ["a-dec"],
    citations: [{ assertion_id: "a-dec", source_id: "src-1", segment: 0 }] }] }], gaps: [] },
  items: [
    { id: "a-dec", revision: 2, kind: "decision", statement: "Pricing moves to annual plans.", status: "current", basis: "stated",
      valid_from: null, valid_to: null, citation: { source_id: "src-1", segment: 0 }, entities: [{ entity_type: "actor", entity_id: "act-sam" }] },
    { id: "a-risk", revision: 1, kind: "risk", statement: "Churn may rise.", status: "contested", basis: "inferred",
      valid_from: null, valid_to: null, citation: { source_id: "src-1", segment: 2 }, entities: [] },
  ],
  speakers: [
    { label: "Speaker A", actor_id: "act-sam", actor_name: "Sam", status: "inferred", link_id: "l-1", link_revision: 3, decision_id: null },
    { label: "Speaker B", actor_id: "act-kim", actor_name: "Kim", status: "confirmed", link_id: "l-2", link_revision: 2, decision_id: "d-9" },
  ],
  proposals: [
    proposal(),
    proposal({ id: "p-other", title: "Kim drafts the FAQ", assignee: { id: "act-kim", name: "Kim" }, preselected: false, reason: "Kim's work." }),
    proposal({ id: "p-upd", kind: "update", title: "Move launch to Dec 1", preselected: false, target: { id: "t-1", title: "Launch" },
      before: { due_date: "2026-11-20" }, after: { due_date: "2026-12-01" } }),
  ],
  names: { "actor:act-sam": "Sam" }, ...over,
});
export const page = (): SourcePage => ({
  source: { id: "src-1", title: "Pricing sync", occurred_at: "2026-11-03T15:00:00Z", revision: 1, current_revision: 1, segment_count: 3, participants: [] },
  segments: [
    { seq: 0, kind: "transcript", speaker: "Speaker A", attribution: "other", speaker_actor_id: "act-sam", start_ms: 0, end_ms: 4000,
      text: "We move pricing to annual plans.", supports: [{ assertion_id: "a-dec", kind: "decision", status: "current" }] },
    { seq: 1, kind: "transcript", speaker: "Davin", attribution: "owner", speaker_actor_id: null, start_ms: 4000, end_ms: 8000,
      text: "I'll send the deck Friday.", supports: [] },
  ],
  range: { start: 0, end: 49 }, next_segment: null, gaps: [],
});
