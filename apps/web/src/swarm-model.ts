/** Swarm mode (backend swarm.py, docs/AGENT_CONTROL.md "Swarm mode"): the plan's shape and the pure helpers the
 * Swarm view uses. Nothing here fetches. */
import { stateLabel } from "./remotea-model";

export type SwarmItemStatus = "planned" | "queued" | "not_started" | string;
export type SwarmItem = {
  task_id: string; title: string; repo: string; target: { id: string; name: string; repo: string } | null;
  blocked_by: string[]; areas: string[]; why: string; error: { code: string; message: string } | null; retries: number;
  status: SwarmItemStatus;
  run: { id: string; state: string; state_reason: string; branch: string; summary: string; pr_urls: string[] } | null;
};
export type SwarmConflict = { tasks: [string, string]; shared: string[]; basis: "files" | "planner areas"; unmerged: boolean };
export type SwarmIntegrate = {
  blocking: boolean;
  repos: { repo: string; merge_order: string[]; conflicts: SwarmConflict[];
    runs: { task_id: string; title: string; run_id: string; state: string; branch: string; pr_urls: string[]; merged: boolean | null;
      ahead: number | null; files: string[] | null; areas: string[] }[] }[];
};
export type SwarmCost = { runs: number; basis: string; eridani_cost: number | null; turns: number | null; minutes: number | null;
  judge_model: string; note: string };
export type SwarmBatch = { id: string; number: number; reason: string; state: "pending" | "running" | "settled";
  started_at: string | null; settled_at: string | null; items: SwarmItem[]; cost: SwarmCost; integrate: SwarmIntegrate | null };
export type Swarm = {
  id: string; record_id: string; title: string; state: "draft" | "running" | "done" | "canceled"; revision: number; cap: number;
  auto_continue: boolean; harness: string; planner: { source: "model" | "fallback"; note?: string; profile?: string; cost?: number | null };
  excluded: { task_id: string; title: string; reason: string; by_you?: boolean }[];
  warnings: { code: string; message: string; task_id?: string; batch?: number }[];
  batches: SwarmBatch[]; next_batch: string | null; created_at: string; updated_at: string;
};
export type SwarmOp =
  | { op: "exclude" | "retry"; task_id: string }
  | { op: "include"; task_id: string; batch_id?: string }
  | { op: "move"; task_id: string; batch_id: string }
  | { op: "split"; batch_id: string; task_ids: string[] }
  | { op: "merge"; batch_id: string; into: string }
  | { op: "order"; batch_id: string; index: number }
  | { op: "cap"; value: number }
  | { op: "auto_continue"; value: boolean }
  | { op: "harness"; value: string };

export const editBody = (swarm: Pick<Swarm, "revision">, ...ops: SwarmOp[]) => ({ expected_revision: swarm.revision, ops });
export const startBody = (swarm: Pick<Swarm, "revision">, all = false) => ({ expected_revision: swarm.revision, all });

/** One item's state as words: the run's state once it has one. */
export function itemLabel(item: Pick<SwarmItem, "status" | "error" | "run">) {
  if (item.run) return stateLabel(item.run.state);
  if (item.status === "not_started") return "Could not start";
  if (item.status === "queued") return "Waiting for a free slot";
  return "Planned";
}
export function itemTone(item: Pick<SwarmItem, "status" | "run">) {
  if (item.status === "not_started") return "bad";
  const state = item.run?.state;
  if (!state) return "idle";
  if (["needs_you", "waiting_answer", "interviewing", "awaiting_approval", "awaiting_integration"].includes(state)) return "attention";
  if (["failed", "capped", "stopped"].includes(state)) return "bad";
  if (state === "completed") return "done";
  return "working";
}

/** "Batch 2 · running · 3 tasks" */
export function batchLine(batch: Pick<SwarmBatch, "number" | "state" | "items">) {
  const word = { pending: "not started", running: "running", settled: "settled" }[batch.state] ?? batch.state;
  return `Batch ${batch.number} · ${word} · ${batch.items.length} task${batch.items.length === 1 ? "" : "s"}`;
}

/** The estimate shown before Start: Eridani's own judge and summary spend, and what history says about turns. */
export function costLine(cost: SwarmCost) {
  if (cost.eridani_cost == null) return `${cost.runs} run${cost.runs === 1 ? "" : "s"}; no finished runs yet to estimate from`;
  const money = cost.eridani_cost < 0.01 ? "under $0.01" : `about $${cost.eridani_cost.toFixed(2)}`;
  const turns = cost.turns != null ? `, about ${Math.round(cost.turns)} agent turns` : "";
  const time = cost.minutes != null ? `, about ${Math.round(cost.minutes)} min each` : "";
  return `${cost.runs} run${cost.runs === 1 ? "" : "s"}: ${money} in Eridani checks${turns}${time} (${cost.basis})`;
}

/** What the plan's Start buttons do now: the next batch, or why nothing can start. */
export function startState(swarm: Pick<Swarm, "state" | "batches" | "next_batch">) {
  if (swarm.state === "done") return { batch: null, reason: "Every batch has settled." };
  if (swarm.state === "canceled") return { batch: null, reason: "This plan was canceled." };
  const running = swarm.batches.find(b => b.state === "running");
  if (running) return { batch: null, reason: `Batch ${running.number} is running; the next one waits until its runs finish or wait for a merge.` };
  const next = swarm.batches.find(b => b.id === swarm.next_batch) ?? null;
  return next ? { batch: next, reason: "" } : { batch: null, reason: "Nothing left to start." };
}

/** Pending batches a task may move to (its own excluded). */
export function moveTargets(swarm: Pick<Swarm, "batches">, taskId: string) {
  return swarm.batches.filter(b => b.state === "pending" && !b.items.some(i => i.task_id === taskId));
}

export function titleOf(swarm: Pick<Swarm, "batches" | "excluded">, taskId: string) {
  for (const b of swarm.batches) { const hit = b.items.find(i => i.task_id === taskId); if (hit) return hit.title; }
  return swarm.excluded.find(e => e.task_id === taskId)?.title ?? taskId.slice(0, 8);
}

/** The integrate step in one sentence per repository. */
export function integrateLines(swarm: Pick<Swarm, "batches" | "excluded">, integrate: SwarmIntegrate | null) {
  if (!integrate) return [];
  return integrate.repos.map(repo => {
    const order = repo.merge_order.map(id => titleOf(swarm, id)).join(" → ");
    const clash = repo.conflicts.length
      ? repo.conflicts.map(c => `${titleOf(swarm, c.tasks[0])} and ${titleOf(swarm, c.tasks[1])} both change ${c.shared.slice(0, 3).join(", ")}${c.shared.length > 3 ? "…" : ""} (${c.basis})`).join("; ")
      : "no overlapping changes found";
    return `${repo.repo || "repository"}: merge ${order}; ${clash}.`;
  });
}
