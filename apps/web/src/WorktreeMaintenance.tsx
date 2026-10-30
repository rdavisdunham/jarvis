// Worktree maintenance (S-Q4) on the remotea page: every managed worktree each host reports, classified by
// worktreeClass, with Rescan (the `worktrees.scan` command) and an explicit review step that produces remotea's own
// cleanup commands for the reclaimable rows. Eridani never removes a worktree: remotea advertises no removal command.
import { useCallback, useEffect, useState } from "react";
import { Check, Copy, RefreshCw, Trash2 } from "lucide-react";
import { api, post } from "./api";
import { useAgentRefresh } from "./remotea-refresh";
import { useRemoteaActions } from "./remotea-context";
import {
  ago, removalPlan, WORKTREE_CLASSES, worktreeClass, worktreeCounts, worktreeKey, worktreeLoss,
  type WorktreeClass, type WorktreeHost, type WorktreeRow,
} from "./remotea-model";

type Report = { hosts: WorktreeHost[]; items: WorktreeRow[] };

const label = (c: WorktreeClass) => WORKTREE_CLASSES.find(x => x.id === c)?.label ?? c;

export function WorktreeMaintenance() {
  const { openRun } = useRemoteaActions();
  const [report, setReport] = useState<Report | null>(null), [error, setError] = useState("");
  const [note, setNote] = useState(""), [busy, setBusy] = useState("");
  const [filter, setFilter] = useState<WorktreeClass | "all">("all"), [reviewing, setReviewing] = useState(false);
  const refresh = useCallback(() => {
    api<Report>("/remotea/worktrees").then(r => { setReport(r); setError(""); }).catch(e => setError((e as Error).message));
  }, []);
  useEffect(() => { refresh(); }, [refresh]);
  useAgentRefresh(refresh);
  if (!report?.hosts.length && !error) return null;
  const rows = report?.items ?? [];
  const counts = worktreeCounts(rows);
  const shown = filter === "all" ? rows : rows.filter(r => worktreeClass(r) === filter);
  const rescan = async (host: WorktreeHost) => {
    setBusy(host.id); setError(""); setNote("");
    try {
      const result = await post<{ queued: boolean; connected: boolean }>(`/remotea/hosts/${host.id}/rescan`, {});
      setNote(!result.connected ? `${host.name} is offline: the rescan goes out when it reconnects.`
        : result.queued ? `Rescan requested from ${host.name}; the list updates when its report arrives.`
          : `A rescan of ${host.name} is already waiting.`);
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(""); }
  };
  return <details className="remotea-maintenance">
    <summary>Worktree maintenance ({rows.length}){counts.reclaimable ? ` · ${counts.reclaimable} reclaimable` : ""}</summary>
    {error && <p role="alert" className="work-error">{error}</p>}
    <ul className="remotea-maintenance-hosts">
      {report?.hosts.map(h => <li key={h.id}>
        <span>{h.name}{!h.connected && " (offline)"} · {h.count} {h.count === 1 ? "worktree" : "worktrees"} · reported {ago(h.report_at)}</span>
        <button className="btn btn-ghost btn-sm" disabled={!h.rescan || busy === h.id} onClick={() => void rescan(h)}
          title={h.rescan ? "Ask the host for a fresh report now" : "This remotea cannot rescan on request; update it"}>
          <RefreshCw size={14}/> {busy === h.id ? "Asking…" : "Rescan"}</button>
      </li>)}
    </ul>
    {note && <p className="footnote" role="status">{note}</p>}
    <div className="remotea-maintenance-counts" role="group" aria-label="Filter worktrees">
      <button className={"chip" + (filter === "all" ? " active" : "")} aria-pressed={filter === "all"} onClick={() => setFilter("all")}>All {rows.length}</button>
      {WORKTREE_CLASSES.map(c => <button key={c.id} title={c.hint} aria-pressed={filter === c.id}
        className={"chip wt-" + c.id + (filter === c.id ? " active" : "")} onClick={() => setFilter(c.id)}>{c.label} {counts[c.id]}</button>)}
    </div>
    {!!shown.length && <div className="remotea-maintenance-table"><table>
      <thead><tr><th>State</th><th>Branch</th><th>Why</th><th>Task / run</th></tr></thead>
      <tbody>{shown.map(r => {
        const c = worktreeClass(r);
        return <tr key={worktreeKey(r)}>
          <td><span className={"chip wt-" + c}>{label(c)}</span></td>
          <td><code>{r.branch ?? "(detached)"}</code><small>{[r.host_name, r.target, r.path].filter(Boolean).join(" · ")}</small></td>
          <td>{c === "reclaimable" ? "Nothing would be lost" : worktreeLoss(r).join("; ")}</td>
          <td>{r.run ? <button className="btn btn-ghost btn-sm" onClick={() => openRun(r.run!.id)}>
            {r.run.task_title ?? "Run"} · {r.run.state.replace(/_/g, " ")}</button> : r.writer ? `run ${r.writer.run_id.slice(0, 8)}` : "—"}</td>
        </tr>;
      })}</tbody>
    </table></div>}
    {!rows.length && <p className="footnote">No managed worktrees reported.</p>}
    {!reviewing && <div className="settings-form-actions">
      <button className="btn btn-ghost btn-sm" disabled={!counts.reclaimable} onClick={() => setReviewing(true)}>
        <Trash2 size={14}/> Review removal</button>
    </div>}
    {reviewing && <RemovalReview rows={rows} onClose={() => setReviewing(false)}/>}
  </details>;
}

/** The review step: only reclaimable rows can be picked; every other row is listed as refused with what it would
 * lose. The result is commands to copy and run on the host; nothing is removed from here. */
export function RemovalReview({ rows, onClose, initialProduced = false }: { rows: WorktreeRow[]; onClose: () => void; initialProduced?: boolean }) {
  const [selected, setSelected] = useState<Set<string>>(() =>
    new Set(rows.filter(r => worktreeClass(r) === "reclaimable" && r.cleanup?.length).map(worktreeKey)));
  const [produced, setProduced] = useState(initialProduced), [copied, setCopied] = useState(false), [error, setError] = useState("");
  const plan = removalPlan(rows, selected);
  const text = plan.commands.join("\n");
  const toggle = (key: string) => {
    setProduced(false); setCopied(false);
    setSelected(s => { const next = new Set(s); if (next.has(key)) next.delete(key); else next.add(key); return next; });
  };
  const copy = () => navigator.clipboard.writeText(text).then(() => setCopied(true))
    .catch(() => setError("Copy is unavailable here. Select the commands to copy them."));
  return <section className="remotea-removal" aria-label="Review worktree removal">
    <h3>Review removal</h3>
    <p className="footnote">Eridani does not remove worktrees: remotea has no removal command. Pick what to reclaim, then run
      the commands in WSL on that host. Each records a recovery ref (<code>refs/remotea/recovery/…</code>) before
      <code> git worktree remove</code>, which refuses a worktree that changed since the report.</p>
    {plan.reclaimable.length ? <ul className="remotea-removal-list">
      {plan.reclaimable.map(r => <li key={worktreeKey(r)}><label>
        <input type="checkbox" checked={selected.has(worktreeKey(r))} onChange={() => toggle(worktreeKey(r))}/>
        <code>{r.branch ?? r.path}</code> <small>{[r.host_name, r.target, r.path].filter(Boolean).join(" · ")}</small>
      </label></li>)}
    </ul> : <p className="footnote">Nothing is reclaimable right now.</p>}
    {!!plan.refused.length && <details className="remotea-removal-refused">
      <summary>Not offered: {plan.refused.length} {plan.refused.length === 1 ? "worktree" : "worktrees"} would lose work or are in use</summary>
      <ul>{plan.refused.map(({ row, lost }) => <li key={worktreeKey(row)}>
        <span className={"chip wt-" + worktreeClass(row)}>{label(worktreeClass(row))}</span> <code>{row.branch ?? row.path}</code>
        <small>Would lose: {lost.join("; ")}</small>
      </li>)}</ul>
    </details>}
    <div className="settings-form-actions">
      <button className="btn btn-primary btn-sm" disabled={!plan.chosen.length} onClick={() => setProduced(true)}>
        Produce commands ({plan.chosen.length})</button>
      <button className="btn btn-ghost btn-sm" onClick={onClose}>Close</button>
    </div>
    {produced && !!plan.chosen.length && <div className="remotea-removal-commands">
      <label className="sr-only" htmlFor="remotea-removal-commands">Cleanup commands</label>
      <textarea id="remotea-removal-commands" readOnly rows={Math.min(12, plan.commands.length)} value={text} onFocus={e => e.target.select()}/>
      <button className="btn btn-ghost btn-sm" onClick={() => void copy()}>{copied ? <Check size={14}/> : <Copy size={14}/>} {copied ? "Copied" : "Copy commands"}</button>
      {error && <p role="alert" className="work-error">{error}</p>}
    </div>}
  </section>;
}
