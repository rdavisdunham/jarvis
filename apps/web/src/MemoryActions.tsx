import { useEffect, useState } from "react";
import { FileText, Pencil, RotateCcw, Trash2, X } from "lucide-react";
import { api } from "./api";
import { Dialog } from "./ux";
import type { Memory, MemoryMaintenance } from "./types";
import "./notes.css";

type LearningStatus = {enabled:boolean;pending:number;retrying:number;deferred:number;queued?:number;active?:number;failed?:number;retry_waiting?:number};

/** Learning and weekly review status for the Memory page, as quiet chips rather than a dotted meta string. */
export function MemoryStatus({status, worker, reviews, maintenance, timezone, busy, onRetry, onReview}: {
  status: LearningStatus; worker: boolean; reviews: number; maintenance: MemoryMaintenance | null; timezone: string; busy: boolean;
  onRetry: () => void; onReview: () => void;
}) {
  const counts = [
    status.queued ? `${status.queued} waiting` : "",
    status.active ? `${status.active} processing` : "",
    status.retry_waiting ? `${status.retry_waiting} waiting to retry` : "",
    status.failed ? `${status.failed} failed` : "",
    status.deferred ? `${status.deferred} paused for budget` : "",
    reviews ? `${reviews} ${reviews === 1 ? "question" : "questions"} to review` : "",
  ].filter(Boolean);
  const review = maintenance?.enabled
    ? maintenance.status === "failed"
      ? "Deep sleep could not finish. Your memories are unchanged; retry the review."
      : maintenance.status === "retrying"
        ? "Deep sleep hit a problem and is waiting to retry."
        : maintenance.running
          ? "Deep sleep is reviewing memories…"
          : "Weekly deep sleep, next review " + new Date(maintenance.next_run_at).toLocaleString([], {weekday: "short", month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZone: timezone})
    : "Weekly deep sleep is off";
  return <section className="panel memory-status" aria-label="Memory learning">
    <div className="learning-status" role="status">
      <div className="memory-status-line">
        <span className={"memory-dot" + (status.enabled ? " on" : "")} aria-hidden/>
        <strong>{status.enabled ? "Automatic learning is on" : "Automatic learning is off"}</strong>
        {!worker && <span className="chip">Background processing is paused</span>}
        {counts.map(c => <span className={"chip" + (c.includes("failed") ? " chip-due-overdue" : "")} key={c}>{c}</span>)}
        {!counts.length && <span className="memory-status-quiet">{status.pending ? "Learning is queued" : "No learning waiting"}</span>}
      </div>
      <p className="memory-status-help">These are learning steps, not a count of messages. Correct a fact to change what Eri remembers; View source shows where it came from.</p>
    </div>
    <div className="memory-maintenance">
      <p className="memory-status-quiet tabular">{review}</p>
      {maintenance?.last_run_at && <p className="memory-status-quiet tabular">Last successful review {new Date(maintenance.last_run_at).toLocaleString([], {month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZone: timezone})}</p>}
      <div className="memory-status-actions">
        {status.retrying > 0 && <button className="btn btn-ghost btn-sm" onClick={onRetry}><RotateCcw size={14}/>Retry memory learning</button>}
        <button className="btn btn-sm" disabled={busy || !maintenance?.enabled || maintenance.running} onClick={onReview}>
          {maintenance?.status === "failed" ? "Retry review" : "Review now"}
        </button>
      </div>
    </div>
  </section>;
}

/** One remembered fact as a list row: the fact, quiet provenance, an expandable evidence line and hover actions. */
export function MemoryRow({memory, onCorrect, onForget}: {memory: Memory; onCorrect: () => void; onForget: (deleteSource: boolean) => Promise<unknown>}) {
  return <article className="memory row memory-row">
    <div className="memory-main">
      <p className="memory-content">{memory.content}</p>
      <div className="memory-meta">
        <span>{memory.attribution === "owner_statement" ? "You told Eri" : "Learned from a saved source"}</span>
        <time className="tabular" dateTime={memory.created_at}>{new Date(memory.created_at).toLocaleDateString([], {month: "short", day: "numeric", year: "numeric"})}</time>
        {memory.tags?.map(tag => <span className="chip" key={tag}>{tag}</span>)}
      </div>
      {memory.evidence && memory.evidence.trim() !== memory.content.trim() && <details className="memory-evidence">
        <summary>Evidence</summary>
        <blockquote>{memory.evidence}</blockquote>
      </details>}
    </div>
    <div className="row-actions memory-actions">
      <MemoryActions memory={memory} onCorrect={onCorrect} onForget={onForget}/>
    </div>
  </article>;
}

export function MemoryActions({memory, onCorrect, onForget}: {memory:Memory; onCorrect:()=>void; onForget:(deleteSource:boolean)=>Promise<unknown>}) {
  const [mode, setMode] = useState<"source"|"forget"|null>(null);
  return <><button className="btn btn-ghost btn-sm" onClick={() => setMode("source")}><FileText size={14} aria-hidden/>View source</button>
    <button className="btn btn-ghost btn-sm" onClick={onCorrect}><Pencil size={14} aria-hidden/>Correct</button>
    <button className="btn btn-danger btn-sm" onClick={() => setMode("forget")}><Trash2 size={14} aria-hidden/>Forget</button>
    {mode && <MemoryActionDialog memory={memory} mode={mode} onClose={() => setMode(null)} onForget={onForget}/>}</>;
}
function MemoryActionDialog({memory, mode, onClose, onForget}: {memory:Memory; mode:"source"|"forget"; onClose:()=>void; onForget:(deleteSource:boolean)=>Promise<unknown>}) {
  const [source, setSource] = useState<{content:string; created_at:string}|null>(null), [error,setError] = useState("");
  const [deleteSource,setDeleteSource] = useState(false), [busy,setBusy] = useState(false);
  useEffect(() => {let live = true; if (mode === "source") void api<{content:string;created_at:string}>("/sources/" + memory.source_id).then(v => {if(live)setSource(v);}).catch(e => {if(live)setError(e.message);}); return () => {live = false;};}, [mode, memory.source_id]);
  useEffect(() => {const escape = (event:KeyboardEvent) => {if(event.key === "Escape") {event.stopImmediatePropagation();if(!busy)onClose();}};document.addEventListener("keydown",escape,true);return () => document.removeEventListener("keydown",escape,true);}, [busy,onClose]);
  return <Dialog aria-label={mode === "source" ? "Memory source" : "Forget memory"} className="dialog memory-source">
    <header className="dialog-heading"><h2>{mode === "source" ? "Memory source" : "Forget this memory?"}</h2><button autoFocus className="btn-icon" aria-label="Close memory action" disabled={busy} onClick={onClose}><X size={20}/></button></header>
    {error && <p className="error-banner" role="alert">{error}</p>}
    {mode === "source" ? <>
      {source ? <>
        <p className="memory-source-date tabular">Saved {new Date(source.created_at).toLocaleString([], {month: "short", day: "numeric", year: "numeric", hour: "numeric", minute: "2-digit"})}</p>
        <blockquote className="source-content">{source.content}</blockquote>
      </> : !error && <p role="status" className="memory-status-quiet">Loading source…</p>}
      <p className="field-hint">Correct updates the remembered fact. It does not rewrite what you originally said.</p>
      <div className="dialog-actions"><button className="btn" onClick={onClose}>Done</button></div>
    </> : <>
      <blockquote className="source-content">{memory.content}</blockquote>
      <p className="memory-forget-copy">Eri will stop using this fact. Your tasks and notes stay as they are.</p>
      <label className="inline-check"><input type="checkbox" checked={deleteSource} onChange={e => setDeleteSource(e.target.checked)}/>Also delete the stored source and every memory learned from it</label>
      <p className="field-hint">This cannot be undone. Leaving the box unchecked keeps the source and other learned facts.</p>
      <div className="dialog-actions">
        <button className="btn" disabled={busy} onClick={onClose}>Keep memory</button>
        <button className="btn btn-primary memory-forget-confirm" disabled={busy} onClick={async () => {setBusy(true);try {const result = await onForget(deleteSource);if(result)onClose();else setError("Could not forget this memory. Try again.");} catch(e) {setError((e as Error).message);} finally {setBusy(false);}}}>{busy ? "Forgetting…" : "Forget memory"}</button>
      </div>
    </>}
  </Dialog>;
}
