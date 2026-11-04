import { useCallback, useEffect, useState } from "react";
import { ChevronRight, RefreshCw, UserRound } from "lucide-react";
import { isUnavailable, listMeetings, reviewSummaries, useContextChanged } from "./context-api";
import { meetingDate, meetingStatus, type MeetingSource, type ReviewSummary, type SourceStatus } from "./context-model";
import { MeetingReview } from "./MeetingReview";
import "./meetings.css";

const FILTERS: { value: SourceStatus | ""; label: string }[] = [
  { value: "", label: "All meetings" }, { value: "ready", label: "Ready" }, { value: "processing", label: "Processing" },
  { value: "partial", label: "Partly read" }, { value: "awaiting_input", label: "Paused" }, { value: "failed", label: "Failed" },
];
const PAGE = 30;

/** The Meetings page: the list of meetings in meeting memory, or one meeting's review when `sourceId` is set. */
export function Meetings({ sourceId, zone, onOpen, onPerson }: {
  sourceId: string | null; zone?: string; onOpen: (sourceId: string | null) => void; onPerson: (actorId: string) => void;
}) {
  if (sourceId) return <MeetingReview key={sourceId} sourceId={sourceId} zone={zone} onBack={() => onOpen(null)} onPerson={onPerson}/>;
  return <MeetingList zone={zone} onOpen={onOpen}/>;
}

function MeetingList({ zone, onOpen }: { zone?: string; onOpen: (sourceId: string) => void }) {
  const [items, setItems] = useState<MeetingSource[] | null>(null), [total, setTotal] = useState(0);
  const [reviews, setReviews] = useState<Map<string, ReviewSummary>>(new Map());
  const [status, setStatus] = useState<SourceStatus | "">(""), [pages, setPages] = useState(1);
  const [error, setError] = useState(""), [off, setOff] = useState(false), [loading, setLoading] = useState(false);
  const load = useCallback(async () => {
    setLoading(true);
    try {
      const page = await listMeetings({ status, limit: PAGE * pages });
      setItems(page.items); setTotal(page.total); setError("");
      // Counts come from the review routes; the list stays useful without them.
      reviewSummaries().then(setReviews).catch(() => setReviews(new Map()));
    } catch (e) {
      if (isUnavailable(e)) setOff(true); else setError((e as Error).message);
    } finally { setLoading(false); }
  }, [status, pages]);
  useEffect(() => { void load(); }, [load]);
  useContextChanged(() => void load());
  const busy = !!items?.some(s => ["received", "queued", "processing"].includes(s.status));
  useEffect(() => {
    if (!busy) return;
    const timer = setInterval(() => void load(), 15000);
    return () => clearInterval(timer);
  }, [busy, load]);

  if (off) return <section className="meetings-page" aria-label="Meetings"><p className="empty">Meeting memory is not on for this workspace. It works in your personal workspace once it is turned on.</p></section>;
  return <section className="meetings-page" aria-label="Meetings">
    <div className="meetings-toolbar">
      <p>Meetings Eridani has read, newest first. Open one to review what it found and choose which work to add.</p>
      <label>Show<select aria-label="Meeting status" value={status} onChange={e => { setStatus(e.target.value as SourceStatus | ""); setPages(1); }}>
        {FILTERS.map(f => <option key={f.value} value={f.value}>{f.label}</option>)}</select></label>
      <button className="btn btn-sm" disabled={loading} onClick={() => void load()}><RefreshCw size={14}/>Refresh</button>
    </div>
    {error && <p role="alert" className="error-banner">{error}</p>}
    {!items ? !error && <p role="status" className="meetings-hint">Loading meetings…</p>
      : !items.length ? <p className="empty">{status ? "No meetings with this status." : "No meetings yet. Import a transcript in Settings → Meeting sources, or send one from a connected tool."}</p>
      : <ul className="meeting-list" aria-label="Meetings">
        {items.map(source => <li key={source.id}><MeetingRow source={source} review={reviews.get(source.id)} zone={zone} onOpen={() => onOpen(source.id)}/></li>)}
      </ul>}
    {items && items.length < total && <button className="btn meetings-more" disabled={loading} onClick={() => setPages(n => n + 1)}>More meetings</button>}
  </section>;
}

function MeetingRow({ source, review, zone, onOpen }: { source: MeetingSource; review?: ReviewSummary; zone?: string; onOpen: () => void }) {
  const state = meetingStatus(source);
  const counts = review && (review.pending ? `${review.pending} to review` : review.later ? `${review.later} for later` : review.status === "done" ? "Reviewed" : "");
  return <button type="button" className="meeting-row" onClick={onOpen}>
    <span className="meeting-row-main">
      <strong>{source.title || "Untitled meeting"}</strong>
      <span className="meeting-row-meta">
        <time dateTime={source.occurred_at ?? undefined}>{meetingDate(source.occurred_at, zone)}</time>
        <span className={"chip meeting-status tone-" + state.tone} title={state.detail}>{state.label}</span>
        {counts && <span className={"chip" + (review?.pending ? " meeting-pending" : "")}>{counts}</span>}
      </span>
      {state.tone !== "ok" && state.detail && <span className="meeting-row-detail">{state.detail}</span>}
    </span>
    <ChevronRight size={18} aria-hidden/>
  </button>;
}

/** The person page until it ships (D4 replaces this one import in App.tsx). */
export function PersonPlaceholder({ actorId, onMeetings }: {
  actorId: string | null; onMeetings: () => void; onMeeting?: (sourceId: string) => void;
}) {
  return <section className="meetings-page" aria-label="Person">
    <p className="empty"><UserRound size={20} aria-hidden/>{actorId ? "Person pages are coming soon. Until then, this person's context appears in each meeting's review." : "Choose a person from a meeting to see what you know about them."}
      <button className="btn btn-sm" onClick={onMeetings}>Open Meetings</button></p>
  </section>;
}
