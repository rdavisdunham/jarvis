import { useCallback, useEffect, useState } from "react";
import { RotateCw } from "lucide-react";
import { api } from "./api";
import { Prop } from "./RecordCard";
import { useStructureActions } from "./structure-actions";
import type { CustomRecord, SchemaType } from "./structure-types";
import { REVIEW_GLYPH, REVIEW_INTERVALS, localDay, reviewLabels, reviewOf, reviewPatch, type ReviewEvery, type ReviewFields } from "./review-model";
import "./reviews.css";

export type ReviewItem = { id: string; title: string; type_id: string; type_name: string; every: ReviewEvery; every_label: string; last_reviewed_at: string | null; next_review_at: string | null; due: boolean; revision: number; home: { id: string; title: string }[] };
export type ReviewPage = { items: ReviewItem[]; total: number; due_count: number; next_offset: number | null };

const browserToday = () => localDay(new Date().toISOString());
/** Opens a record's detail card from anywhere (Organization handles the event). */
export const openRecord = (id: string) => window.dispatchEvent(new CustomEvent("eri-open-custom-record", { detail: { id } }));

/** The "Review cadence" behavior: a switch plus an interval select. Both edit the schema draft only. */
export function ReviewCadenceControl({ type, disabled, compact = false, onChange }: { type: Pick<SchemaType, "review" | "name" | "plural">; disabled?: boolean; compact?: boolean; onChange: (patch: Pick<SchemaType, "review">) => void }) {
  const review = reviewOf(type);
  return <div className={"review-cadence" + (compact ? " is-compact" : "")}>
    <label className="review-cadence-toggle">
      <input type="checkbox" className="switch" role="switch" aria-label="Review cadence" checked={review.enabled} disabled={disabled} onChange={e => onChange(reviewPatch(type, { enabled: e.target.checked }))}/>
      <span><span aria-hidden="true" className="review-glyph">{REVIEW_GLYPH[0]}</span>Resurface {(type.plural || type.name).toLowerCase()} for review</span>
    </label>
    <select aria-label="Review interval" value={review.every} disabled={disabled || !review.enabled} onChange={e => onChange(reviewPatch(type, { every: e.target.value as ReviewEvery }))}>
      {REVIEW_INTERVALS.map(([id, label]) => <option key={id} value={id}>{label}</option>)}
    </select>
  </div>;
}

/** Detail-card rows: Last reviewed and Next review as separate values, with Mark reviewed when due. */
export function ReviewProps({ row, canEdit, onRecord }: { row: ReviewFields & { id: string }; canEdit: boolean; onRecord: (r: CustomRecord) => void }) {
  const { run, busy, error } = useStructureActions();
  if (!row.review_every) return null;
  const labels = reviewLabels(row, browserToday());
  const act = (tool: string, args: Record<string, unknown>) => void run<CustomRecord>(tool, { record_id: row.id, ...args }).then(onRecord).catch(() => {});
  return <>
    <div className="prop-divider"/>
    <Prop label="Last reviewed"><span className="prop-static tabular">{labels.last}</span></Prop>
    <Prop label="Next review">
      <span className={"prop-static tabular review-next is-" + labels.tone}>{labels.next}</span>
      {canEdit && <span className="review-prop-actions">
        {row.review_due && <button type="button" className="btn btn-soft btn-sm" disabled={busy} onClick={() => act("record.mark_reviewed", {})}>Mark reviewed</button>}
        {row.review_paused
          ? <button type="button" className="text-button" disabled={busy} onClick={() => act("record.review", { action: "resume" })}>Resume reviews</button>
          : <button type="button" className="text-button" disabled={busy} onClick={() => act("record.review", { action: "pause" })}>Stop reviewing this record</button>}
      </span>}
      {error && <span className="prop-note" role="alert">{error}</span>}
    </Prop>
  </>;
}

/** Today's small "Reviews due (N)" panel. Renders nothing when no reviews are due. */
export function TodayReviews({ today, zone, refresh, canEdit }: { today: string; zone: string; refresh: unknown; canEdit: boolean }) {
  const [page, setPage] = useState<ReviewPage | null>(null);
  const { run, busy } = useStructureActions();
  const load = useCallback(() => api<ReviewPage>("/structure/reviews?limit=5").then(setPage).catch(() => setPage(null)), []);
  useEffect(() => { void load(); }, [load, refresh]);
  if (!page?.total) return null;
  return <section className="panel today-reviews" aria-labelledby="today-reviews-title">
    <header className="panel-header"><h2 id="today-reviews-title">Reviews due</h2><span className="panel-count">{page.total}</span></header>
    <ul className="today-rows">{page.items.map(item => {
      const labels = reviewLabels({ ...item, review_due: item.due }, today, zone);
      return <li key={item.id} className="row today-row review-row">
        <span className="review-row-glyph" aria-hidden="true"><RotateCw size={15}/></span>
        <div className="row-main">
          <button type="button" className="row-title" onClick={() => openRecord(item.id)}>{item.title}</button>
          <div className="row-meta">
            <span className="chip">{item.type_name}</span>
            <span className="chip chip-due-today">{labels.next}</span>
          </div>
        </div>
        {canEdit && <button type="button" className="btn btn-ghost btn-sm review-row-action" aria-label={"Mark " + item.title + " reviewed"} disabled={busy}
          onClick={() => void run("record.mark_reviewed", { record_id: item.id }).then(load).catch(() => {})}>Mark reviewed</button>}
      </li>;
    })}</ul>
  </section>;
}

export type ReviewQuestionData = { key: string; revision: number; status: string; deferred_until?: string; record: ReviewItem };
/** A queued record review in Questions: Mark reviewed, snooze, open, or stop reviewing this record. */
export function ReviewQuestion({ q, onChanged }: { q: ReviewQuestionData; onChanged: () => Promise<void> }) {
  const { run, busy, error } = useStructureActions();
  const r = q.record;
  const labels = reviewLabels({ ...r, review_due: r.due }, browserToday());
  const act = (tool: string, args: Record<string, unknown>) => void run(tool, args).then(onChanged).catch(() => {});
  const open = ["pending", "deferred"].includes(q.status);
  return <article className="question-card review-question">
    <div className="question-meta"><span>Review due</span><span className="chip">{q.status}</span></div>
    <h3>{r.title}</h3>
    <div className="review-question-facts">
      <span className="chip">{r.type_name}</span>
      <span className="chip">{r.every_label.charAt(0).toUpperCase() + r.every_label.slice(1)}</span>
      {!!r.home.length && <span className="chip chip-home" title={r.home.map(h => h.title).join(" / ")}>{r.home.map(h => h.title).join(" / ")}</span>}
    </div>
    <p className="review-question-dates"><span>Last reviewed <strong className="tabular">{labels.last}</strong></span>
      <span>{q.status === "deferred" && q.deferred_until ? <>Snoozed until <strong className="tabular">{new Date(q.deferred_until).toLocaleDateString(undefined, { month: "short", day: "numeric" })}</strong></> : <strong className="tabular">{labels.next}</strong>}</span></p>
    {open && <div className="question-actions">
      <button type="button" className="btn btn-primary" disabled={busy} onClick={() => act("record.mark_reviewed", { record_id: r.id })}>Mark reviewed</button>
      <button type="button" className="btn" disabled={busy} onClick={() => act("record.review", { record_id: r.id, action: "snooze", until: "day" })}>Snooze a day</button>
      <button type="button" className="btn" disabled={busy} onClick={() => act("record.review", { record_id: r.id, action: "snooze", until: "week" })}>Snooze a week</button>
      <button type="button" className="btn btn-ghost" onClick={() => openRecord(r.id)}>Open</button>
      <button type="button" className="btn btn-ghost" disabled={busy} onClick={() => act("record.review", { record_id: r.id, action: "pause" })}>Stop reviewing this record</button>
    </div>}
    {error && <p role="alert">{error} Refresh to see the current review.</p>}
    <details className="question-provenance"><summary>Why this is here</summary>
      <p>{r.type_name} records resurface {r.every_label}. Marking it reviewed schedules the next review; Organization › Types & fields changes the cadence.</p></details>
  </article>;
}
