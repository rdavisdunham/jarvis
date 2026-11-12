import { useCallback, useEffect, useState } from "react";
import { api, command, post } from "./api";

/** How much of the owner a run on this task needs (readiness.py). Null until the readiness pass or the owner decides. */
export type Route = "auto" | "direction" | "plan_review" | "human";
export type TaskReadiness = {
  task_id: string; revision: number; route: Route | null; route_by: "owner" | "readiness" | null; reasons: string[];
  state: "ready" | "needs_target" | "unavailable" | "skipped" | "stale" | null;
  target: { id: string; name: string; by: "resolved" | "readiness" } | null; filled: string[]; risks: string[]; pending: boolean;
};

export const ROUTE_OPTIONS: { value: Route; label: string }[] = [
  { value: "auto", label: "Nothing: an agent can finish it" },
  { value: "direction", label: "Your direction first" },
  { value: "plan_review", label: "Your review of the full plan" },
  { value: "human", label: "You do it" },
];
const FILLED: Record<string, string> = { criteria: "draft success criteria", verify: "a verify command", target: "a repository" };

/** The line under the route: why it was chosen and what the pass filled in, in plain words. */
export function routeNote(r: TaskReadiness): string {
  if (r.pending || r.state === "stale") return "Looking at this task…";
  const parts: string[] = [];
  if (r.route_by === "owner") parts.push("You chose this.");
  else if (r.route && r.reasons.length) parts.push(r.reasons.join(" "));
  else if (r.state === "unavailable") parts.push("Could not work this out yet; it will be tried again.");
  else if (!r.route) parts.push("Not looked at yet.");
  const filled = r.filled.map(name => FILLED[name]).filter(Boolean);
  if (filled.length) parts.push("Filled in for you: " + filled.join(", ") + (r.target?.by === "readiness" ? ` (${r.target.name})` : "") + ".");
  return parts.join(" ");
}

/** The route on a task's Run control: what the task needs from the owner, changeable in one tap. */
export function TaskRoute({ taskId, canEdit }: { taskId: string; canEdit: boolean }) {
  const [state, setState] = useState<TaskReadiness | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const load = useCallback(() => api<TaskReadiness>("/remotea/tasks/" + taskId + "/readiness").then(setState).catch(() => {}), [taskId]);
  useEffect(() => { setState(null); void load(); }, [load]);
  const waiting = !!state && (state.pending || state.state === "stale");
  useEffect(() => {
    if (!waiting) return;
    const timer = setInterval(() => void load(), 4000);
    return () => clearInterval(timer);
  }, [waiting, load]);
  if (!state) return null;
  const act = async (work: () => Promise<unknown>) => {
    setBusy(true); setError("");
    try { await work(); } catch (e) { setError(e instanceof Error ? e.message : "That did not save."); }
    await load(); setBusy(false);
  };
  const choose = (value: string) => act(() => command("task.update",
    { task_id: taskId, expected_revision: state.revision, run_route: value || null }).send());
  return <div className="task-route">
    <label>What it needs from you
      <select value={state.route ?? ""} disabled={!canEdit || busy} onChange={e => void choose(e.target.value)}>
        <option value="">{state.route ? "Let Eridani decide" : "Not decided yet"}</option>
        {ROUTE_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </label>
    {canEdit && <button type="button" className="btn btn-sm" disabled={busy || waiting}
      onClick={() => void act(() => post("/remotea/tasks/" + taskId + "/readiness"))}>Check again</button>}
    <small aria-live="polite">{routeNote(state)}</small>
    {error && <p role="alert" className="work-error">{error}</p>}
  </div>;
}
