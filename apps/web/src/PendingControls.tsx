import { lazy, Suspense } from "react";
import type { NeedsTarget, PendingControl } from "./remotea-model";
import { useRemotea, useRemoteaActions, type PendingCardState } from "./remotea-context";

// The picker lives in the remotea chunk; it loads only when a start has no target yet.
const TargetPicker = lazy(() => import("./Remotea").then(m => ({ default: m.TargetPicker })));

/** One of Eri's proposals (control_start / control_message) waiting for the owner's tap; nothing is sent before.
 * A start with no target shows the task's target picker; picking one saves it and confirms. */
export function PendingControlCard({ item, card, onDecide, onCancelPicker }: {
  item: PendingControl; card?: PendingCardState;
  onDecide: (decision: "confirm" | "decline", target?: string) => void; onCancelPicker: () => void;
}) {
  const need: NeedsTarget | null = card && card.need !== undefined ? card.need : (item.needs_target ? item.need ?? null : null);
  const busy = !!card?.busy;
  return <div className="remotea-pending-card" data-pending-id={item.id}>
    <p>{item.summary}</p>
    {(item.problems ?? []).map(problem => <p key={problem.code} className="footnote">{problem.message}</p>)}
    {need ? <Suspense fallback={<p className="footnote">Loading repositories…</p>}>
        <TargetPicker key={need.task_id} need={need} onSaved={target => onDecide("confirm", target)} onCancel={onCancelPicker}/>
      </Suspense> :
      <div className="settings-form-actions">
        <button className="btn btn-primary btn-sm" disabled={busy} onClick={() => onDecide("confirm")}>Confirm</button>
        <button className="btn btn-ghost btn-sm" disabled={busy} onClick={() => onDecide("decline")}>Decline</button>
      </div>}
    {card?.error && <p role="alert" className="work-error">{card.error}</p>}
  </div>;
}

/** Eri's open proposals. `compact` (the chat drawer) shows the newest few and points at the remotea page for the rest. */
export function PendingControls({ compact = false, limit = 2, onMore, heading = true }: {
  compact?: boolean; limit?: number; onMore?: () => void; heading?: boolean;
}) {
  const { pending, cards } = useRemotea();
  const { decide, cancelPicker, openRun } = useRemoteaActions();
  if (!pending.length) return null;
  const shown = compact ? pending.slice(0, limit) : pending;
  const act = (id: string, decision: "confirm" | "decline", target?: string) =>
    void decide(id, decision, target).then(run => { if (run) openRun(run.id); });
  return <section className={compact ? "chat-pending" : "remotea-pending"} aria-label="Waiting for your confirmation" aria-live="polite">
    {heading && !compact && <h2>Eri is waiting for you</h2>}
    {compact && <p className="remotea-card-kicker">Eri is waiting for you</p>}
    {shown.map(p => <PendingControlCard key={p.id} item={p} card={cards[p.id]} onDecide={(d, t) => act(p.id, d, t)} onCancelPicker={() => cancelPicker(p.id)}/>)}
    {compact && pending.length > shown.length && <p className="footnote">
      {pending.length - shown.length} more {onMore ? <button className="text-button" onClick={onMore}>on the remotea page</button> : "on the remotea page"}.</p>}
  </section>;
}
