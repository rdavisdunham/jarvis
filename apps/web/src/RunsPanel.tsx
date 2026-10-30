import { useEffect, useRef, useState } from "react";
import { X } from "lucide-react";
import { api } from "./api";
import { Dialog } from "./ux";
import { ago, attentionGroups, runLabel, type AttentionGroup, type Run, type UnmergedWork } from "./remotea-model";
import { useRemotea, useRemoteaActions } from "./remotea-context";
import { PendingControls } from "./PendingControls";
import "./runs.css";

/** One run as a row: state, task, and why it is in that state. Used by the Runs panel and the remotea page. */
export function RunRow({ run, onOpen }: { run: Run; onOpen: (id: string) => void }) {
  return <button key={run.id} className={"remotea-run-row state-" + run.state} onClick={() => onOpen(run.id)}>
    <span className={"remotea-state run-state-" + run.state}>{runLabel(run.state)}</span>
    <span className="remotea-run-title">{run.task?.title ?? "Task"}</span>
    <small>{run.reason_text || ago(run.state_since)}</small>
  </button>;
}

/** Runs grouped by the next human action (attentionGroups): a heading with the verb and count, then the rows. The
 * confirm group shows Eri's proposals (PendingControls) instead of rows. */
export function AttentionList({ groups, onOpen }: { groups: AttentionGroup[]; onOpen: (id: string) => void }) {
  return <>{groups.map(g => <section key={g.key} className={"run-group group-" + g.key + (g.action ? " needs-action" : "")}
    aria-label={`${g.verb}: ${g.count}`}>
    <h3>{g.verb} <span className="run-group-count">{g.count}</span></h3>
    {g.key === "confirm" ? <PendingControls heading={false}/> : g.runs.map(run => <RunRow key={run.id} run={run} onOpen={onOpen}/>)}
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
