import { useCallback, useEffect, useId, useState } from "react";
import { ArrowUp, Layers, Play, X } from "lucide-react";
import { api, ApiError, post } from "./api";
import { Dialog } from "./ux";
import { useAgentRefresh } from "./remotea-refresh";
import { HARNESSES } from "./remotea-model";
import {
  batchLine, costLine, editBody, integrateLines, itemLabel, itemTone, moveTargets, startBody, startState,
  type Swarm, type SwarmBatch, type SwarmItem, type SwarmOp,
} from "./swarm-model";
import "./swarm.css";

const message = (e: unknown) => (e as Error).message || "Something went wrong.";

/** One swarm plan: batches with their tasks, the edits allowed before a batch starts, the integrate step after,
 * and Start. Every change posts to the server and shows what it returns. */
export function SwarmPlan({ swarm, onChange, onOpenRun }: { swarm: Swarm; onChange: (next: Swarm) => void; onOpenRun: (id: string) => void }) {
  const [busy, setBusy] = useState(false), [error, setError] = useState("");
  const open = swarm.state === "draft" || swarm.state === "running";
  const send = async (path: string, body: unknown) => {
    setBusy(true); setError("");
    try { onChange(await post<Swarm>(`/remotea/swarms/${swarm.id}/${path}`, body)); }
    catch (e) {
      if (e instanceof ApiError && e.code === "REVISION_CONFLICT") {
        setError("This plan changed; reloaded.");
        onChange(await api<Swarm>(`/remotea/swarms/${swarm.id}`));
      } else setError(message(e));
    } finally { setBusy(false); }
  };
  const edit = (...ops: SwarmOp[]) => void send("edit", editBody(swarm, ...ops));
  const next = startState(swarm);
  return <section className={"swarm-plan state-" + swarm.state} aria-label={"Swarm plan for " + swarm.title}>
    <header className="swarm-head">
      <h3><Layers size={16} aria-hidden="true"/>{swarm.title}</h3>
      <p className="footnote">{swarm.planner.source === "model"
        ? `Grouped by the planning model (${swarm.planner.profile ?? "model"}).`
        : swarm.planner.note || "Grouped by repository and Blocks links."}</p>
    </header>
    {open && <div className="swarm-settings">
      <label>Runs per host at once
        <select value={swarm.cap} disabled={busy} onChange={e => edit({ op: "cap", value: Number(e.target.value) })}>
          {Array.from({ length: 10 }, (_, i) => i + 1).map(n => <option key={n} value={n}>{n}</option>)}
        </select>
      </label>
      <div className="segmented" role="group" aria-label="Agent">
        {HARNESSES.map(h => <button key={h.id} type="button" aria-pressed={swarm.harness === h.id} disabled={busy}
          onClick={() => edit({ op: "harness", value: h.id })}>{h.label}</button>)}
      </div>
      <label className="swarm-auto"><input type="checkbox" checked={swarm.auto_continue} disabled={busy}
        onChange={e => edit({ op: "auto_continue", value: e.target.checked })}/>Continue automatically
        <small>the next batch starts when this one settles, unless the integrate step finds overlapping changes</small></label>
    </div>}
    {!!swarm.warnings.length && <ul className="swarm-warnings" aria-label="Plan warnings">
      {swarm.warnings.map((w, i) => <li key={i} role="note">{w.message}</li>)}</ul>}
    <ol className="swarm-batches">
      {swarm.batches.map((batch, index) => <BatchCard key={batch.id} swarm={swarm} batch={batch} busy={busy || !open}
        previous={index > 0 ? swarm.batches[index - 1] : null} edit={edit} onOpenRun={onOpenRun}/>)}
    </ol>
    {!!swarm.excluded.length && <details className="swarm-excluded">
      <summary>Left out ({swarm.excluded.length})</summary>
      <ul>{swarm.excluded.map(e => <li key={e.task_id}><span>{e.title}</span> <small>{e.reason}</small>
        {e.by_you && open && <button className="btn btn-ghost btn-sm" disabled={busy} onClick={() => edit({ op: "include", task_id: e.task_id })}>Bring back</button>}
      </li>)}</ul>
    </details>}
    {error && <p role="alert" className="work-error">{error}</p>}
    {open && <div className="settings-form-actions swarm-actions">
      <button className="btn btn-primary btn-sm" disabled={busy || !next.batch} title={next.reason || undefined}
        onClick={() => void send("start", { ...startBody(swarm), batch_id: next.batch!.id })}>
        <Play size={14} aria-hidden="true"/>{next.batch ? `Start batch ${next.batch.number}` : "Start"}</button>
      <button className="btn btn-ghost btn-sm" disabled={busy || !next.batch} title="Start this batch now and each next one when the previous settles"
        onClick={() => void send("start", { ...startBody(swarm, true), batch_id: next.batch!.id })}>Start all</button>
      <button className="btn btn-ghost btn-sm" disabled={busy} onClick={() => void send("cancel", { expected_revision: swarm.revision })}>
        <X size={14} aria-hidden="true"/>Cancel plan</button>
      {next.reason && <p className="footnote">{next.reason}</p>}
    </div>}
  </section>;
}

function BatchCard({ swarm, batch, previous, busy, edit, onOpenRun }: {
  swarm: Swarm; batch: SwarmBatch; previous: SwarmBatch | null; busy: boolean; edit: (...ops: SwarmOp[]) => void; onOpenRun: (id: string) => void;
}) {
  const pending = batch.state === "pending";
  const integrate = integrateLines(swarm, batch.integrate);
  return <li className={"swarm-batch batch-" + batch.state}>
    <div className="swarm-batch-head">
      <strong>{batchLine(batch)}</strong>
      {pending && previous?.state === "pending" && <>
        <button className="btn-icon" aria-label={`Move batch ${batch.number} up`} disabled={busy}
          onClick={() => edit({ op: "order", batch_id: batch.id, index: batch.number - 2 })}><ArrowUp size={15}/></button>
        <button className="btn btn-ghost btn-sm" disabled={busy} onClick={() => edit({ op: "merge", batch_id: batch.id, into: previous.id })}>
          Merge into batch {previous.number}</button>
      </>}
    </div>
    <p className="swarm-reason">{batch.reason}</p>
    {batch.state !== "settled" && <p className="footnote swarm-cost" title={batch.cost.note}>{costLine(batch.cost)}</p>}
    <ul className="swarm-items">
      {batch.items.map(item => <ItemRow key={item.task_id} swarm={swarm} batch={batch} item={item} busy={busy} edit={edit} onOpenRun={onOpenRun}/>)}
    </ul>
    {!!integrate.length && <div className={"swarm-integrate" + (batch.integrate?.blocking ? " blocking" : "")} role="note">
      <strong>Integrate{batch.integrate?.blocking ? " before the next batch" : ""}</strong>
      <ul>{integrate.map(line => <li key={line}>{line}</li>)}</ul>
    </div>}
  </li>;
}

function ItemRow({ swarm, batch, item, busy, edit, onOpenRun }: {
  swarm: Swarm; batch: SwarmBatch; item: SwarmItem; busy: boolean; edit: (...ops: SwarmOp[]) => void; onOpenRun: (id: string) => void;
}) {
  const pending = batch.state === "pending";
  const targets = moveTargets(swarm, item.task_id);
  return <li className={"swarm-item tone-" + itemTone(item)}>
    <div className="swarm-item-main">
      <span className="swarm-item-title">{item.title}</span>
      <span className="swarm-item-state">{itemLabel(item)}</span>
      <small>{item.target?.name ?? "no target chosen"}{item.repo ? " · " + item.repo : ""}</small>
      {item.run?.summary && <small className="swarm-item-summary">{item.run.summary}</small>}
      {item.error && <small className="swarm-item-error">{item.error.message}</small>}
      {(item.why || item.areas.length > 0) && <small>{[item.why, item.areas.length ? "likely changes " + item.areas.join(", ") : ""].filter(Boolean).join(" · ")}</small>}
      {!!item.run?.pr_urls.length && <small>{item.run.pr_urls.map(u => <a key={u} href={u} target="_blank" rel="noreferrer">{u.replace(/^https?:\/\/(www\.)?/, "")}</a>)}</small>}
    </div>
    <div className="swarm-item-actions">
      {item.run && <button className="btn btn-ghost btn-sm" onClick={() => onOpenRun(item.run!.id)}>Open run</button>}
      {item.error && batch.state === "running" && <button className="btn btn-ghost btn-sm" disabled={busy}
        onClick={() => edit({ op: "retry", task_id: item.task_id })}>Retry</button>}
      {pending && <>
        <select aria-label={"Move " + item.title} value="" disabled={busy}
          onChange={e => { if (e.target.value) edit({ op: "move", task_id: item.task_id, batch_id: e.target.value }); }}>
          <option value="">Move to…</option>
          {targets.map(b => <option key={b.id} value={b.id}>Batch {b.number}</option>)}
          <option value="new">A new batch at the end</option>
        </select>
        {batch.items.length > 1 && <button className="btn btn-ghost btn-sm" disabled={busy}
          onClick={() => edit({ op: "split", batch_id: batch.id, task_ids: [item.task_id] })}>Split off</button>}
        <button className="btn btn-ghost btn-sm" disabled={busy} onClick={() => edit({ op: "exclude", task_id: item.task_id })}>Leave out</button>
      </>}
    </div>
  </li>;
}

/** Play on a project: plan it (or reopen its open plan) and show the plan. Nothing starts until Start. */
export function SwarmDialog({ taskId, onClose, onOpenRun, onRunProject }: {
  taskId: string; onClose: () => void; onOpenRun: (id: string) => void; onRunProject: () => void;
}) {
  const heading = useId();
  const [swarm, setSwarm] = useState<Swarm | null>(null);
  const [error, setError] = useState(""), [empty, setEmpty] = useState(false);
  useEffect(() => {
    let live = true;
    post<Swarm>("/remotea/swarms", { task_id: taskId })
      .then(s => { if (live) setSwarm(s); })
      .catch(e => { if (!live) return; if (e instanceof ApiError && e.code === "NO_TASKS") setEmpty(true); setError(message(e)); });
    return () => { live = false; };
  }, [taskId]);
  return <Dialog className="dialog swarm-dialog" aria-labelledby={heading} onEscape={onClose} onBackdrop={onClose}>
    <h2 id={heading}>Plan a swarm</h2>
    {!swarm && !error && <p role="status" className="footnote">Planning which tasks can run side by side…</p>}
    {error && <p role={empty ? "status" : "alert"} className={empty ? "footnote" : "work-error"}>{error}</p>}
    {swarm && <SwarmPlan swarm={swarm} onChange={setSwarm} onOpenRun={id => { onClose(); onOpenRun(id); }}/>}
    <div className="settings-form-actions">
      {empty && <button className="btn btn-primary btn-sm" onClick={onRunProject}><Play size={14} aria-hidden="true"/>Run the project itself</button>}
      <button className="btn btn-ghost btn-sm" onClick={onClose}>Close</button>
    </div>
  </Dialog>;
}

/** The remotea page's swarm view: every open plan with its batches and each run's state. */
export function SwarmList({ onOpenRun }: { onOpenRun: (id: string) => void }) {
  const [items, setItems] = useState<Swarm[]>([]), [error, setError] = useState("");
  const refresh = useCallback(() => {
    api<{ items: Swarm[] }>("/remotea/swarms?active=true").then(r => { setItems(r.items); setError(""); }).catch(e => setError(message(e)));
  }, []);
  useEffect(() => { refresh(); }, [refresh]);
  useAgentRefresh(refresh);
  if (!items.length && !error) return null;
  return <section className="remotea-swarms" aria-label="Swarms">
    <h2>Swarms</h2>
    {error && <p role="alert" className="work-error">{error}</p>}
    {items.map(s => <SwarmPlan key={s.id} swarm={s} onOpenRun={onOpenRun}
      onChange={next => setItems(list => next.state === "canceled" ? list.filter(x => x.id !== next.id) : list.map(x => x.id === next.id ? next : x))}/>)}
  </section>;
}
