import { useEffect, useRef, useState, type ReactNode } from "react";
import { X } from "lucide-react";
import { api } from "./api";
import { Dialog } from "./ux";
import { actorChip, ago, attentionGroups, ciChip, isSettled, rowFreshness, stopLine, workflowStage, workflowAttention, type AttentionGroup, type Fact, type Run, type UnmergedWork } from "./remotea-model";
import { useRemotea, useRemoteaActions } from "./remotea-context";
import { PendingControls } from "./PendingControls";
import "./runs.css";

/** Facts as small labelled chips, each with its own tone (freshness in the run header and on run rows, a
 * verification's facts). `inline` renders spans, for chips inside a button (a run row). */
export function Facts({ facts, label, className = "", inline = false }: { facts: Fact[]; label: string; className?: string; inline?: boolean }) {
  if (!facts.length) return null;
  const chips = facts.map(f => inline ? <span key={f.key} role="listitem" className={"tone-" + f.tone} title={f.title}>{f.label}</span>
    : <li key={f.key} className={"tone-" + f.tone} title={f.title}>{f.label}</li>);
  return inline ? <span className={"remotea-fresh " + className} role="list" aria-label={label}>{chips}</span>
    : <ul className={"remotea-fresh " + className} aria-label={label}>{chips}</ul>;
}

/** A clock for relative times ("host seen 12 s ago"): re-renders on its own, never fetches anything. */
export function useNow(ms = 15_000) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => { const id = window.setInterval(() => setNow(Date.now()), ms); return () => window.clearInterval(id); }, [ms]);
  return now;
}

/** One run as a row: state, task, and why it is in that state; under them, what the agent is doing now (the run
 * summary), with a quiet "updating…" while a newer one is due, and compact freshness chips (rowFreshness: the host's
 * lease, a stale or incomplete check, uncertain delivery). `now` comes from the list's clock. Used by the Runs panel
 * and the remotea page. */
export function RunRow({ run, onOpen, now = Date.now() }: { run: Run; onOpen: (id: string) => void; now?: number }) {
  // GitHub CI (run_ci.py): a chip in the row (with the wait so far while awaiting CI) and, beside the row, a link to the
  // checks (a link cannot sit inside the row's button).
  const ci = isSettled(run) ? null : ciChip(run.ci);
  const waiting = run.state === "awaiting_ci" && run.ci?.since ? " · " + ago(run.ci.since, now).replace(" ago", "") : "";
  return <>{row(run, onOpen, now, ci && <span className={"run-ci-chip tone-" + ci.tone} title={ci.title}>{ci.label}{waiting}</span>)}
    {ci?.href && (run.state === "awaiting_ci" || run.ci?.state === "failed") &&
      <a className="run-row-ci-link" href={ci.href} target="_blank" rel="noreferrer">{run.ci?.state === "failed" ? "Failed checks" : "View checks"} on GitHub</a>}</>;
}

function row(run: Run, onOpen: (id: string) => void, now: number, ci: ReactNode) {
  return <button key={run.id} className={"remotea-run-row state-" + run.state} onClick={() => onOpen(run.id)}>
    <span className={"remotea-state run-state-" + run.state}>{workflowStage(run)}</span>
    {workflowAttention(run) && <span className="run-attention">· {workflowAttention(run)}</span>}
    {actorChip(run) && <span className="run-actor">{actorChip(run)}</span>}
    <span className="remotea-run-title">{run.task?.title ?? "Task"}</span>
    <small className={run.attention_detail ? "run-stop-line" : undefined}>{stopLine(run) || ago(run.state_since, now)}</small>
    {ci}
    {!!run.summary && <span className="run-summary" title={run.summary_at ? "As of " + ago(run.summary_at, now) : undefined}>
      {run.summary}{run.summary_stale && <span className="run-summary-updating"> updating…</span>}</span>}
    <Facts facts={rowFreshness(run, null, now)} label="How current this is" className="run-row-fresh" inline/>
  </button>;
}

/** Runs grouped by the next human action (attentionGroups): a heading with the verb and count, then the rows. The
 * confirm group shows Dani's proposals (PendingControls) instead of rows. One clock re-renders every row's ages. */
export function AttentionList({ groups, onOpen }: { groups: AttentionGroup[]; onOpen: (id: string) => void }) {
  const now = useNow(30_000);
  return <>{groups.map(g => <section key={g.key} className={"run-group group-" + g.key + (g.action ? " needs-action" : "")}
    aria-label={`${g.verb}: ${g.count}`}>
    <h3>{g.verb} <span className="run-group-count">{g.count}</span></h3>
    {g.key === "confirm" ? <PendingControls heading={false}/> : g.runs.map(run => <RunRow key={run.id} run={run} onOpen={onOpen} now={now}/>)}
  </section>)}</>;
}

/** The topbar Runs panel: everything that has not settled, grouped by what you need to do next (answer, review the
 * plan, repair the host, inspect a failed check, approve, resolve unmerged work), then what is just working. Picking
 * one opens it in the side panel. */
export function RunsPanel({ onClose, onOpenPage }: { onClose: () => void; onOpenPage: () => void }) {
  const { activeRuns, pending, error, status } = useRemotea();
  const { openRun } = useRemoteaActions();
  const close = useRef<HTMLButtonElement>(null);
  const [unmerged, setUnmerged] = useState<UnmergedWork[]>([]);
  useEffect(() => { close.current?.focus(); }, []);
  useEffect(() => { api<{ items: UnmergedWork[] }>("/remotea/unmerged").then(r => setUnmerged(r.items)).catch(() => setUnmerged([])); }, []);
  const groups = attentionGroups(activeRuns, pending, status?.hosts ?? [], unmerged);
  return <Dialog onBackdrop={onClose} onEscape={onClose} backdropClassName="activity-backdrop" className="activity-panel runs-panel" aria-label="Agent runs">
    <header><div><h2>Runs</h2><p>Agents working on your tasks, grouped by what each one needs from you.</p></div>
      <button ref={close} className="btn-icon" aria-label="Close runs" onClick={onClose}><X size={20}/></button></header>
    {error && <p role="alert" className="work-error">{error}</p>}
    <div className="activity-items">
      {!groups.length && <p className="activity-empty">No agent runs are active. Press Run on a task to start one.</p>}
      <AttentionList groups={groups} onOpen={id => { openRun(id); onClose(); }}/>
    </div>
    <footer><button className="btn btn-ghost btn-sm" onClick={() => { onOpenPage(); onClose(); }}>Open the remotea page</button></footer>
  </Dialog>;
}
