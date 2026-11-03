import { useEffect, useRef, useState, useSyncExternalStore, type ReactNode } from "react";
import { actionReceipts, api, postAction, subscribeActions, type ActionReceipt } from "./api";
import { isSettled, type Run, type RunDetail } from "./remotea-model";

export type RunDecision = { id: string; kind: "question" | "interview" | "control" | "plan_approval" | "plan_review" | "final_acceptance"; run_id: string | null; revision: number; status: string; payload: Record<string, unknown>; actions: string[]; generation: number; plan_revision: number };
export function respondDecision<T>(decision: RunDecision, action: string, values: Record<string, unknown> = {}) {
  return postAction<T>(`/remotea/decisions/${encodeURIComponent(decision.id)}/respond`, { expected_revision: decision.revision, action, ...values });
}
export function decisionFor(run: RunDetail, kind: RunDecision["kind"], id?: string) {
  return run.decisions?.find(d => d.kind === kind && (!id || d.id === id) && ["open", "pending"].includes(d.status));
}
const EMPTY_RECEIPTS: ActionReceipt[] = [];
export function useReceipts() { return useSyncExternalStore(subscribeActions, actionReceipts, () => EMPTY_RECEIPTS); }
export function DeliveryReceipt({ receipt }: { receipt?: ActionReceipt }) {
  if (!receipt) return null;
  const labels = { queued: "Queued on this device · waiting for connection", saved: "Saved by Eridani · waiting for agent acknowledgement", delivered: "Delivered to the agent", uncertain: "Delivery unconfirmed · your response is retained for retry", rejected: receipt.error || "Response was not accepted" };
  return <p className={"run-delivery delivery-" + receipt.status} role="status">{labels[receipt.status]}{receipt.retention === "tab" ? " · available in this tab only" : ""}{["queued", "uncertain"].includes(receipt.status) && <> · <button type="button" className="text-button" onClick={() => { void postAction(receipt.url, receipt.body).catch(() => {}); }}>Retry delivery</button></>}</p>;
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

type DraftRecord<T> = { value: T; dirty: boolean; revision: number };
const DRAFT_PREFIX = "eri-draft:";
const draftTimers = new Map<string, ReturnType<typeof setTimeout>>();
const memoryDrafts = new Map<string, DraftRecord<unknown>>();
const draftSaving = new Set<string>();
function readDraft<T>(key: string, initial: T): DraftRecord<T> {
  try {
    const stored = localStorage.getItem(DRAFT_PREFIX + key);
    if (stored === null) return { value: initial, dirty: false, revision: 0 };
    const parsed = JSON.parse(stored);
    return parsed && typeof parsed === "object" && "dirty" in parsed && "value" in parsed ? parsed : { value: parsed, dirty: true, revision: 0 };
  } catch { return (memoryDrafts.get(key) as DraftRecord<T> | undefined) ?? { value: initial, dirty: false, revision: 0 }; }
}
function storeDraft<T>(key: string, record: DraftRecord<T>) {
  memoryDrafts.set(key, record);
  try { localStorage.setItem(DRAFT_PREFIX + key, JSON.stringify(record)); } catch { /* Editing remains available in memory. */ }
}
async function saveDraftKey(key: string) {
  const record = readDraft<unknown>(key, null);
  if (!record.dirty || draftSaving.has(key)) return;
  draftSaving.add(key);
  try {
    const saved = await api<{ revision: number }>(`/remotea/drafts/${encodeURIComponent(key)}`, { method: "PUT", body: JSON.stringify({ value: record.value }) });
    const latest = readDraft<unknown>(key, null);
    if (JSON.stringify(latest.value) === JSON.stringify(record.value)) storeDraft(key, { ...latest, dirty: false, revision: saved.revision });
  } catch { /* Dirty local copies are retried by the existing run provider after reconnection. */ }
  finally { draftSaving.delete(key); }
}
export async function flushDrafts() {
  try { await Promise.all([...new Set([...Object.keys(localStorage).filter(key => key.startsWith(DRAFT_PREFIX)).map(key => key.slice(DRAFT_PREFIX.length)), ...memoryDrafts.keys()])].map(saveDraftKey)); } catch { await Promise.all([...memoryDrafts.keys()].map(saveDraftKey)); }
}
export function useRunDraft<T>(key: string, initial: T): [T, (value: T | ((current: T) => T)) => void] {
  const [draft, setDraft] = useState<T>(() => readDraft(key, initial).value);
  const currentKey = useRef(key), edited = useRef(false);
  useEffect(() => {
    currentKey.current = key; edited.current = false;
    const local = readDraft(key, initial); setDraft(local.value);
    let live = true;
    void api<{ value: T | null; revision: number }>(`/remotea/drafts/${encodeURIComponent(key)}`).then(result => {
      if (!live || edited.current || readDraft(key, initial).dirty) return;
      const value = result.value ?? initial;
      storeDraft(key, { value, dirty: false, revision: result.revision }); setDraft(value);
    }).catch(() => {});
    return () => { live = false; };
  }, [key]);
  const update = (value: T | ((current: T) => T)) => {
    edited.current = true;
    setDraft(current => {
      const next = typeof value === "function" ? (value as (current: T) => T)(current) : value;
      const draftKey = currentKey.current;
      storeDraft(draftKey, { ...readDraft(draftKey, initial), value: next, dirty: true });
      clearTimeout(draftTimers.get(draftKey));
      // The timer belongs to the draft, so navigation cannot cancel its server save.
      draftTimers.set(draftKey, setTimeout(() => { draftTimers.delete(draftKey); void saveDraftKey(draftKey); }, 600));
      return next;
    });
  };
  return [draft, update];
}

const STAGES: Record<string, string> = {
  starting: "Starting", queued: "Starting", spawning: "Starting", interviewing: "Interviewing", planning: "Planning", replanning: "Planning",
  plan_reviewing: "Reviewing plan", reviewing_plan: "Reviewing plan", handing_off: "Starting", executing: "Working", working: "Working",
  verifying: "Checking results", judging: "Checking results", checking_results: "Checking results", awaiting_approval: "Ready for review", ready_for_review: "Ready for review",
  completed: "Complete", complete: "Complete", stopped: "Stopped", failed: "Failed", capped: "Needs attention", stopping: "Stopping",
};
export function workflowStage(run: Run): string {
  if (run.stage) return STAGES[run.stage] ?? run.stage.replaceAll("_", " ");
  if (run.state === "needs_you" && ["plan_approval", "plan_review"].includes(run.state_reason)) return "Reviewing plan";
  if (run.state === "waiting_answer") return "Interviewing";
  if (run.state === "needs_you") return run.plan_revision ? "Working" : "Planning";
  return STAGES[run.state] ?? "Working";
}
export function workflowAttention(run: Run): string | null {
  if (run.paused || run.attention === "paused") return "Paused";
  if (run.attention === "error") return "Needs attention";
  if (run.attention === "decision" || ["needs_you", "waiting_answer", "awaiting_approval"].includes(run.state)) return run.state_reason === "plan_approval" || run.state === "awaiting_approval" ? "Needs your approval" : "Waiting for you";
  return null;
}
export function activityStatus(run: Run, now = Date.now()): { label: string; animate: boolean } {
  if (isSettled(run)) return { label: workflowStage(run), animate: false };
  if (run.connectivity === "offline" || run.host?.connected === false) return { label: "Reconnecting…", animate: false };
  const attention = workflowAttention(run);
  if (attention) return { label: attention === "Paused" ? "Paused" : "Waiting for you", animate: false };
  const activity = run.activity;
  if (!activity || !activity.at || now - Date.parse(activity.at) > 90_000 || !Number.isFinite(Date.parse(activity.at))) return { label: "Activity unconfirmed", animate: false };
  const labels: Record<string, string> = { thinking: "Thinking…", deliberating: "Deliberating…", reading: "Reading files", searching: "Reading files", editing: "Editing", testing: "Running tests", tool: "Using tools" };
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
  if (!/^approved[.!]?$/i.test(text.trim())) return null;
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
