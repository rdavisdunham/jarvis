import { Check, X } from "lucide-react";
import type { Task } from "./types";
import { finishedBySync, finishedToday } from "./task-finished";
import "./today.css";

export function FinishedToday({ tasks, today, zone, busy, canEdit = true, onTask, onReopen }: {
  tasks: Task[]; today: string; zone: string; busy: boolean; canEdit?: boolean;
  onTask: (task: Task) => void; onReopen: (task: Task) => void;
}) {
  const rows = finishedToday(tasks, today, zone);
  return <div className="today-finished">{(["completed", "cancelled"] as const).map(status => {
    const all = rows.filter(task => task.status === status);
    // The count is what you finished in Eridani (it matches the day ring); Linear closures that synced in sit apart.
    const items = all.filter(task => !finishedBySync(task)), synced = all.filter(finishedBySync);
    const label = status === "completed" ? "Completed today" : "Canceled today";
    const row = (task: Task) => <li className="row today-finished-row" key={task.id}>
      {status === "completed" ? <button className="check done" aria-label={"Reopen " + task.title} aria-pressed="true" disabled={busy || !canEdit} onClick={() => onReopen(task)}><Check aria-hidden="true" /></button> : <X size={18} aria-label="Canceled" />}
      <button className="row-title" onClick={() => onTask(task)}>{task.title}</button>
      {status === "cancelled" && canEdit && <button className="btn btn-ghost btn-sm" disabled={busy} onClick={() => onReopen(task)}>Reopen</button>}
    </li>;
    return <section className="panel" aria-label={label} key={status}>
      <header className="panel-header"><h2>{label}</h2><span className="panel-count">{items.length}</span></header>
      {items.length ? <ul className="today-rows">{items.map(row)}</ul> : <p className="today-loading">No {status === "completed" ? "completed" : "canceled"} tasks today.</p>}
      {!!synced.length && <details className="today-synced">
        <summary>{synced.length} {status === "completed" ? "closed" : "canceled"} in Linear</summary>
        <ul className="today-rows">{synced.map(row)}</ul>
      </details>}
    </section>;
  })}</div>;
}
