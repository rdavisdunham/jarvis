import { useEffect, useState } from "react";
import { durableStorageKey } from "./api";
import { modelLabel, timeline, workflowAttention, workflowStage, type RunDetail, type TimelineEntry } from "./remotea-model";
import type { SectionsView } from "./run-stages";

/** `sections`: the Plan / Run / Verify sections opened or closed by hand, valid only for the stage they were set in. */
type LocalView = { chat: boolean | null; resumeRevision: number | null; sections: SectionsView };
const initialView: LocalView = { chat: null, resumeRevision: null, sections: null };
function readView(id: string): LocalView {
  try { return { ...initialView, ...JSON.parse(localStorage.getItem(durableStorageKey("eri-run-view:" + id) ?? "") ?? "{}") }; }
  catch { return initialView; }
}
/** View preferences stay on this device and never mutate or pause the run. */
export function useRunView(id: string) {
  const [view, setView] = useState(() => ({ id, ...readView(id) }));
  useEffect(() => { setView({ id, ...readView(id) }); }, [id]);
  const update = (values: Partial<LocalView>) => setView(current => {
    const next = { ...(current.id === id ? current : { id, ...readView(id) }), ...values };
    try { const key = durableStorageKey("eri-run-view:" + id); if (key) localStorage.setItem(key, JSON.stringify(next)); } catch { /* Keep this tab's view. */ }
    return next;
  });
  return [view.id === id ? view : { id, ...readView(id) }, update] as const;
}
export function chatByDefault(run: RunDetail): boolean {
  return !run.plan || ["queued", "spawning", "interviewing", "planning", "replanning", "plan_reviewing"].includes(run.stage ?? run.state)
    || !!run.plan_pending || ["plan_review", "plan_approval"].includes(run.state_reason) && run.plan.approval_status !== "approved";
}
export function drawerStatus(run: RunDetail, pendingHold = false, connection = ""): string {
  const actor = run.actor;
  const model = modelLabel(actor?.harness ?? run.tool, actor?.model ?? run.model);
  const attention = workflowAttention(run);
  return [workflowStage(run), model, run.automation?.recovery === "worker_starting" ? "Restoring worker · Automation held" : run.automation?.releasing ? "Reconciling automation" : run.automation?.held ? "Automation held" : pendingHold ? "Hold pending" : "",
    connection || (run.connectivity === "offline" || run.host?.connected === false ? "Offline" : ""), attention].filter(Boolean).join(" · ");
}
export function conversationEntries(run: RunDetail): TimelineEntry[] {
  const entries = timeline(run).filter(entry => entry.kind === "turn" || entry.who === "you" && entry.kind === "command" && !!entry.detail);
  const complete = new Map(run.turns.filter(turn => turn.role !== "user" && turn.provider_turn_id && turn.provider_session_id)
    .map(turn => [turn.provider_session_id + ":" + turn.provider_turn_id, turn]));
  for (const message of run.live_messages ?? []) {
    const final = message.turn_id ? complete.get(message.session_id + ":" + message.turn_id) : undefined;
    if (!message.text || final && (final.text.trim() === message.text.trim() || final.clipped && message.text.startsWith(final.text))) continue;
    entries.push({ id: "live:" + message.session_id + ":" + message.message_id, at: message.at, kind: "turn", who: "agent", tone: "plain",
      title: message.kind === "commentary" ? "Assistant commentary" : "Assistant response", detail: message.text,
      ...(message.truncated ? { facts: [{ key: "published-excerpt", label: "Published excerpt · truncated", tone: "plain" }] } : {}) });
  }
  return entries.sort((a, b) => Date.parse(a.at) - Date.parse(b.at) || a.id.localeCompare(b.id));
}
