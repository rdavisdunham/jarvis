import { useEffect, useState } from "react";
import { FileText, Pencil, Pin, PinOff, RotateCcw, ThumbsDown, Trash2, X } from "lucide-react";
import { api } from "./api";
import { Dialog } from "./ux";
import type { AboutYou, Memory, MemoryMaintenance } from "./types";
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
      <p className="memory-status-help">These are learning steps, not a count of messages. Correct a fact to change what Dani remembers; View source shows where it came from.</p>
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

const shortDate = (iso: string) => new Date(iso).toLocaleDateString([], {month: "short", day: "numeric", year: "numeric"});

/** What Dani is told about the owner at the start of each voice call, read-only. Written in the background. */
export function AboutYouSummary({about, onUnpin}: {about: AboutYou; onUnpin?: (id: string) => Promise<unknown>}) {
  const note = !about.enabled
    ? "Off while automatic learning is off."
    : !about.summary
      ? "No summary yet. Dani writes one in the background from your saved memories, about once a day."
      : !about.in_voice
        ? "Not sent to voice while Memory during voice is off."
        : "Dani is told this at the start of each voice call. It is rewritten in the background, about once a day, when your memories change.";
  return <section className="panel memory-about" aria-labelledby="memory-about">
    <div className="memory-section-head">
      <h2 id="memory-about">About you</h2>
      {about.updated_at && <span className="memory-section-sub tabular">Updated <time dateTime={about.updated_at}>{shortDate(about.updated_at)}</time></span>}
    </div>
    {about.summary && <p className="memory-about-text">{about.summary}</p>}
    <p className="memory-status-help">{note}</p>
    {!!about.core?.length && <>
      <h3 className="memory-about-core-head">Always kept in mind</h3>
      <ul className="memory-about-core">
        {about.core.map(fact => <li key={fact.id}>
          <span>{fact.content}</span>
          {onUnpin && <button type="button" className="btn btn-ghost btn-sm" aria-label={"Unpin: " + fact.content}
            onClick={() => void onUnpin(fact.id).catch(() => {})}><PinOff size={14} aria-hidden/>Unpin</button>}
        </li>)}
      </ul>
    </>}
  </section>;
}

/** Where a fact came from. A fact Dani learned on its own names the conversation it came from (a link that opens
 * it) so a wrong fact can be traced; facts you stated or asked Dani to save say so. */
export function MemorySource({memory, onOpenConversation}: {memory: Memory; onOpenConversation?: (id: string) => void}) {
  if (memory.attribution === "owner_statement") return <span>You told Dani</span>;
  if (memory.attribution === "agent_capture") return <span>Saved by Dani at your request</span>;
  const source = memory.source, conversation = source?.conversation_id;
  if (!source || !conversation) return <span>Learned from a saved source</span>;
  const label = (source.origin === "voice" ? "a voice conversation" : "a conversation") + " on " + shortDate(source.created_at);
  return <span>Learned from {onOpenConversation
    ? <button type="button" className="memory-source-link" onClick={() => onOpenConversation(conversation)}>{label}</button>
    : label}</span>;
}

/** One remembered fact as a list row: the fact, quiet provenance, an expandable evidence line and hover actions. */
export function MemoryRow({memory, onCorrect, onForget, onWrong, onOpenConversation, onPin}: {
  memory: Memory; onCorrect: () => void; onForget: (deleteSource: boolean) => Promise<unknown>; onWrong: () => Promise<unknown>;
  onOpenConversation?: (id: string) => void;
  /** Pin or unpin the fact for About you. */
  onPin?: (core: boolean) => Promise<unknown>;
}) {
  return <article className="memory row memory-row">
    <div className="memory-main">
      <p className="memory-content">{memory.content}</p>
      <div className="memory-meta">
        <MemorySource memory={memory} onOpenConversation={onOpenConversation}/>
        <time className="tabular" dateTime={memory.created_at}>{new Date(memory.created_at).toLocaleDateString([], {month: "short", day: "numeric", year: "numeric"})}</time>
        {memory.core && <span className="chip">About you</span>}
        {memory.tags?.map(tag => <span className="chip" key={tag}>{tag}</span>)}
      </div>
      {memory.evidence && memory.evidence.trim() !== memory.content.trim() && <details className="memory-evidence">
        <summary>Evidence</summary>
        <blockquote>{memory.evidence}</blockquote>
      </details>}
    </div>
    <div className="row-actions memory-actions">
      {onPin && <button className="btn btn-ghost btn-sm" onClick={() => void onPin(!memory.core).catch(() => {})}
        title={memory.core ? "Stop always keeping this in mind" : "Dani always keeps pinned facts in mind (About you)"}>
        {memory.core ? <><PinOff size={14} aria-hidden/>Unpin</> : <><Pin size={14} aria-hidden/>Pin to About you</>}</button>}
      <MemoryActions memory={memory} onCorrect={onCorrect} onForget={onForget} onWrong={onWrong}/>
    </div>
  </article>;
}

export function MemoryActions({memory, onCorrect, onForget, onWrong}: {memory:Memory; onCorrect:()=>void; onForget:(deleteSource:boolean)=>Promise<unknown>; onWrong:()=>Promise<unknown>}) {
  const [mode, setMode] = useState<"source"|"forget"|null>(null), [rejecting, setRejecting] = useState(false);
  return <><button className="btn btn-ghost btn-sm" onClick={() => setMode("source")}><FileText size={14} aria-hidden/>View source</button>
    <button className="btn btn-ghost btn-sm" onClick={onCorrect}><Pencil size={14} aria-hidden/>Correct</button>
    {/* One click: removes the fact and records the rejection so automatic learning never saves it again. */}
    <button className="btn btn-ghost btn-sm memory-wrong" disabled={rejecting} title="Remove this fact and stop Dani from learning it again"
      onClick={async () => {setRejecting(true);try {await onWrong();} finally {setRejecting(false);}}}><ThumbsDown size={14} aria-hidden/>This is wrong</button>
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
      <p className="memory-forget-copy">Dani will stop using this fact. Your tasks and notes stay as they are.</p>
      <label className="inline-check"><input type="checkbox" checked={deleteSource} onChange={e => setDeleteSource(e.target.checked)}/>Also delete the stored source and every memory learned from it</label>
      <p className="field-hint">This cannot be undone. Leaving the box unchecked keeps the source and other learned facts.</p>
      <div className="dialog-actions">
        <button className="btn" disabled={busy} onClick={onClose}>Keep memory</button>
        <button className="btn btn-primary memory-forget-confirm" disabled={busy} onClick={async () => {setBusy(true);try {const result = await onForget(deleteSource);if(result)onClose();else setError("Could not forget this memory. Try again.");} catch(e) {setError((e as Error).message);} finally {setBusy(false);}}}>{busy ? "Forgetting…" : "Forget memory"}</button>
      </div>
    </>}
  </Dialog>;
}
