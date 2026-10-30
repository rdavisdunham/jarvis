import { useEffect, useRef } from "react";
import { X } from "lucide-react";
import { Dialog } from "./ux";
import { ago, orderRuns, runLabel, type Run } from "./remotea-model";
import { useRemotea, useRemoteaActions } from "./remotea-context";
import { PendingControls } from "./PendingControls";
import "./runs.css";

/** One run as a row: state, task, and why it is in that state; under them, what the agent is doing now (the run
 * summary), with a quiet "updating…" while a newer one is due. Used by the Runs panel and the remotea page. */
export function RunRow({ run, onOpen }: { run: Run; onOpen: (id: string) => void }) {
  return <button key={run.id} className={"remotea-run-row state-" + run.state} onClick={() => onOpen(run.id)}>
    <span className={"remotea-state run-state-" + run.state}>{runLabel(run.state)}</span>
    <span className="remotea-run-title">{run.task?.title ?? "Task"}</span>
    <small>{run.reason_text || ago(run.state_since)}</small>
    {!!run.summary && <span className="run-summary" title={run.summary_at ? "As of " + ago(run.summary_at) : undefined}>
      {run.summary}{run.summary_stale && <span className="run-summary-updating"> updating…</span>}</span>}
  </button>;
}

/** The topbar Runs panel: Eri's proposals waiting for a tap, then every run that has not settled (the ones that
 * need you first). Picking one opens it in the side panel. */
export function RunsPanel({ onClose, onOpenPage }: { onClose: () => void; onOpenPage: () => void }) {
  const { activeRuns, pending, error } = useRemotea();
  const { openRun } = useRemoteaActions();
  const close = useRef<HTMLButtonElement>(null);
  useEffect(() => { close.current?.focus(); }, []);
  const runs = orderRuns(activeRuns);
  return <Dialog onBackdrop={onClose} onEscape={onClose} backdropClassName="activity-backdrop" className="activity-panel runs-panel" aria-label="Agent runs">
    <header><div><h2>Runs</h2><p>Agents working on your tasks, and the ones waiting for you.</p></div>
      <button ref={close} className="btn-icon" aria-label="Close runs" onClick={onClose}><X size={20}/></button></header>
    {error && <p role="alert" className="work-error">{error}</p>}
    <div className="activity-items">
      <PendingControls heading={false}/>
      {!runs.length && !pending.length && <p className="activity-empty">No agent runs are active. Press Run on a task to start one.</p>}
      {!!runs.length && <h3>Active runs</h3>}
      {runs.map(run => <RunRow key={run.id} run={run} onOpen={id => { openRun(id); onClose(); }}/>)}
    </div>
    <footer><button className="btn btn-ghost btn-sm" onClick={() => { onOpenPage(); onClose(); }}>Open the remotea page</button></footer>
  </Dialog>;
}
