import { useEffect, useRef, useState, useSyncExternalStore, type ReactNode } from "react";
import { actionReceipts, api, durableScope, postAction, subscribeActions, type ActionReceipt, type SubmittedDraft } from "./api";
import { readDraft, storeDraft, subscribeDrafts, writeDraft } from "./run-drafts";
import "./runs.css";
import { isSettled, workflowStage, workflowAttention, type Run, type RunDetail } from "./remotea-model";

export type RunDecision = { id: string; kind: "question" | "interview" | "control" | "plan_approval" | "plan_review" | "final_acceptance"; run_id: string | null; revision: number; status: string; payload: Record<string, unknown>; actions: string[]; generation: number; plan_revision: number };
export function respondDecision<T>(decision: RunDecision, action: string, values: Record<string, unknown> = {}, options?: { draft?: SubmittedDraft }) {
  return postAction<T>(`/remotea/decisions/${encodeURIComponent(decision.id)}/respond`, { expected_revision: decision.revision, action, ...values }, options);
}
export function decisionFor(run: RunDetail, kind: RunDecision["kind"], id?: string) {
  return run.decisions?.find(d => d.kind === kind && (!id || d.id === id) && ["open", "pending"].includes(d.status));
}
const EMPTY_RECEIPTS: ActionReceipt[] = [];
export function useReceipts() { return useSyncExternalStore(subscribeActions, actionReceipts, () => EMPTY_RECEIPTS); }
export function DeliveryReceipt({ receipt }: { receipt?: ActionReceipt }) {
  if (!receipt) return null;
  const labels = { queued: "Queued on this device · waiting for connection", saved: "Saved by Eridani · waiting for agent acknowledgement", delivered: "Delivered to the agent", applied: "Applied by Eridani", uncertain: "Delivery unconfirmed · your response is retained for retry", rejected: receipt.error || "Response was not accepted" };
  const stages = { saved: "Saved by Eridani · waiting for worker runtime", handed_to_harness: "Handed to worker runtime · provider acknowledgement pending", provider_acknowledged: "Provider acknowledged the message", uncertain: "Delivery unconfirmed · review before retrying" };
  const label = receipt.delivery_stage && receipt.status !== "queued" && receipt.status !== "rejected" ? stages[receipt.delivery_stage] : labels[receipt.status];
  return <p className={"run-delivery delivery-" + receipt.status} role="status">{label}{receipt.retention === "tab" ? " · available in this tab only" : ""}{["queued", "uncertain"].includes(receipt.status) && <> · <button type="button" className="text-button" onClick={() => { void postAction(receipt.url, receipt.body).catch(() => {}); }}>Retry delivery</button></>}</p>;
}
/** Shared visual and delivery language for every request, with controls supplied by the request's type. */
export function DecisionCard({ title, task, agent, reason, children, receipt, className = "", label }: {
  title: string; task?: string; agent?: string | null; reason?: string; children: ReactNode; receipt?: ActionReceipt; className?: string; label?: string;
}) {
  return <section className={"remotea-question run-decision " + className} aria-label={label ?? title}>
    <p className="remotea-card-kicker">{title}</p>
    {(task || agent) && <p className="run-decision-context">{[task, agent].filter(Boolean).join(" · ")}</p>}
    {reason && <p className="footnote">{reason}</p>}
    {children}
    <DeliveryReceipt receipt={receipt}/>
  </section>;
}

export { flushDrafts, submittedDraft } from "./run-drafts";
export function useRunDraft<T>(key: string, initial: T): [T, (value: T | ((current: T) => T)) => void] {
  const [draft, setDraft] = useState<T>(() => readDraft(key, initial).value);
  const currentKey = useRef(key), edited = useRef(false);
  useEffect(() => {
    currentKey.current = key; edited.current = false;
    const local = readDraft(key, initial); setDraft(local.value);
    const scope = durableScope();
    let live = true;
    const unsubscribe = subscribeDrafts((changedKey, changedScope) => {
      if (live && changedKey === key && changedScope === scope) setDraft(readDraft(key, initial).value);
    });
    void api<{ value: T | null; revision: number }>(`/remotea/drafts/${encodeURIComponent(key)}`).then(result => {
      if (!live || durableScope() !== scope || edited.current || readDraft(key, initial).dirty) return;
      const current = readDraft(key, initial), value = result.value ?? initial;
      storeDraft(key, { value, dirty: false, revision: result.revision,
        edit_id: JSON.stringify(value) === JSON.stringify(current.value) ? current.edit_id : crypto.randomUUID() });
    }).catch(() => {});
    return () => { live = false; unsubscribe(); };
  }, [key]);
  const update = (value: T | ((current: T) => T)) => {
    edited.current = true;
    const current = readDraft(key, initial).value;
    const next = typeof value === "function" ? (value as (current: T) => T)(current) : value;
    writeDraft(currentKey.current, next);
  };
  return [draft, update];
}

export { workflowStage, workflowAttention } from "./remotea-model";
export function activityStatus(run: Run, now = Date.now()): { label: string; animate: boolean } {
  if (isSettled(run)) return { label: workflowStage(run), animate: false };
  if (run.connectivity === "offline" || run.host?.connected === false) return { label: "Reconnecting…", animate: false };
  const attention = workflowAttention(run);
  if (attention) return { label: ["Paused", "Terminal has control", "Terminal session ended", "Permission needs review", "Delivery unconfirmed"].includes(attention) ? attention : "Waiting for you", animate: false };
  const activity = run.activity;
  if (!activity || !activity.at || !Number.isFinite(Date.parse(activity.at))) return { label: "Activity unconfirmed", animate: false };
  const labels: Record<string, string> = { thinking: "Thinking…", deliberating: "Deliberating…", reading: "Reading files", searching: "Reading files", editing: "Editing", testing: "Running tests", tool: "Using tools" };
  if (now - Date.parse(activity.at) > 90_000) {
    const previous = activity.label || (labels[activity.kind] ?? activity.kind.replaceAll("_", " ")).replace(/…$/, "").toLowerCase();
    return { label: `No recent output · last reported ${previous}`, animate: false };
  }
  return { label: activity.label || labels[activity.kind] || "Working", animate: ["thinking", "deliberating", "reading", "searching", "editing", "testing", "tool"].includes(activity.kind) };
}
export function RunActivity({ run, now }: { run: Run; now?: number }) {
  const activity = activityStatus(run, now);
  return <div className={"run-activity" + (activity.animate ? " is-active" : "")} role="status" aria-live="polite">
    <span className="run-activity-mark" aria-hidden="true"><i/><i/><i/></span><span>{activity.label}</span>
  </div>;
}

/** A short approval only resolves one open request on the selected run. It cannot invent interview answers. */
export function approvalShortcut(run: RunDetail, text: string): { decision: RunDecision; action: string; values: Record<string, unknown> } | null {
  if (run.controller === "terminal" || !/^approved[.!]?$/i.test(text.trim())) return null;
  const open = (run.decisions ?? []).filter(d => ["open", "pending"].includes(d.status));
  if (open.length !== 1) return null;
  const decision = open[0];
  if (["plan_approval", "final_acceptance"].includes(decision.kind) && decision.actions.includes("approve")) return { decision, action: "approve", values: {} };
  if (decision.kind === "control" && decision.actions.includes("confirm")) return { decision, action: "confirm", values: {} };
  if (decision.kind === "question" && decision.payload.kind === "permission" && decision.actions.includes("answer")) {
    const items = decision.payload.items as { id: string; choices?: { id: string }[] }[] | undefined;
    if (items?.length === 1 && items[0].choices?.some(c => c.id === "allow_once")) return { decision, action: "answer", values: { answers: [{ item_id: items[0].id, choice_ids: ["allow_once"] }] } };
  }
  return null;
}
