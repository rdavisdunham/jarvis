// Pure helpers for the remotea page: live agent sessions reported over the agent link.

export type LiveState = { state: string; since: string | null; host_id: string | null; reason: string };
export type Host = {
  id: string; name: string; machine: string; connected: boolean; last_seen_at: string | null;
  conn_epoch: number; caps: string[]; harnesses: Record<string, unknown>; proto: number;
};
export type Turn = {
  id: string; seq: number; text: string; length: number; clipped: boolean; truncated: boolean;
  generation: number; source: string; at: string;
};
export type RemoteaSession = {
  id: string; tool_label: string; title: string; repo: string; branch: string; cwd: string;
  live_state: LiveState | null; host: { id: string; name: string; connected: boolean } | null;
  last_turn: Turn | null; filed: { record_title: string | null };
};
export type RemoteaStatus = {
  enabled: boolean; key: { id: string; prefix: string; expires_at: string; last_used_at: string | null } | null;
  hosts: Host[]; link_url: string; connect_command: string; token?: string | null;
};
export type Parity = {
  total: number; matched: number; missing: number; mismatched: number; clean: boolean;
  fields: Record<string, number>; first_report_at: string | null; last_report_at: string | null;
  recent: { tool: string; session_id: string; missing: boolean; diff: Record<string, { stored: string; reported: string }>; updated_at: string }[];
};

/** The page refreshes on the change feed; this is the fallback when the feed is quiet or offline. */
export const REMOTEA_POLL_MS = 10_000;

const LABELS: Record<string, string> = {
  working: "Working", needs_you: "Needs you", idle: "Idle", restoring: "Restoring", lost: "Restoring", exited: "Ended",
};
export function stateLabel(state: string) {
  return LABELS[state] ?? state;
}
export function isLive(state: string | undefined | null) {
  return state === "working" || state === "needs_you" || state === "idle";
}
/** The Sessions drawer's two-word summary: host-reported state wins over the collector snapshot. */
export function liveChip(live: LiveState | null | undefined): { label: string; tone: "live" | "restoring" | "ended" } | null {
  if (!live?.state) return null;
  if (isLive(live.state)) return { label: "Live", tone: "live" };
  if (live.state === "restoring" || live.state === "lost") return { label: "Restoring", tone: "restoring" };
  return { label: "Ended", tone: "ended" };
}

export function ago(value: string | null, now = Date.now()) {
  if (!value) return "never";
  const seconds = Math.max(0, Math.round((now - new Date(value).getTime()) / 1000));
  if (seconds < 60) return seconds < 10 ? "just now" : seconds + "s ago";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return minutes + "m ago";
  const hours = Math.round(minutes / 60);
  if (hours < 48) return hours + "h ago";
  return Math.round(hours / 24) + "d ago";
}

export function hostSummary(hosts: Host[], now = Date.now()) {
  if (!hosts.length) return { tone: "never" as const, text: "No remotea server has connected yet." };
  const online = hosts.filter(h => h.connected);
  if (online.length) return { tone: "connected" as const, text: online.map(h => h.name || h.machine || "remotea").join(", ") + " connected" };
  const last = hosts.map(h => h.last_seen_at).filter(Boolean).sort().at(-1) ?? null;
  return { tone: "offline" as const, text: "Disconnected · last seen " + ago(last, now) };
}

/** Needs-you first, then working, idle, restoring and ended; newest state first within each. */
export function orderSessions(items: RemoteaSession[]) {
  const rank = (s: RemoteaSession) => ["needs_you", "working", "idle", "restoring", "lost", "exited"].indexOf(s.live_state?.state ?? "exited");
  const time = (s: RemoteaSession) => (s.live_state?.since ? new Date(s.live_state.since).getTime() : 0);
  return [...items].sort((a, b) => rank(a) - rank(b) || time(b) - time(a));
}

export function paritySummary(parity: Parity | null) {
  if (!parity || !parity.total) return "No parity reports yet. Run the new collector in parity mode to compare it.";
  if (parity.clean) return `All ${parity.total} reported sessions match the current collector.`;
  const parts = [`${parity.matched} of ${parity.total} match`];
  if (parity.mismatched) parts.push(`${parity.mismatched} differ`);
  if (parity.missing) parts.push(`${parity.missing} unknown to the current collector`);
  const fields = Object.entries(parity.fields).slice(0, 4).map(([field, count]) => `${field} (${count})`);
  return parts.join(", ") + "." + (fields.length ? " Most different: " + fields.join(", ") + "." : "");
}

export function place(item: Pick<RemoteaSession, "repo" | "branch" | "cwd">) {
  return [item.repo || item.cwd.split("/").filter(Boolean).at(-1), item.branch].filter(Boolean).join(" · ");
}

// ---------------------------------------------------------------- runs (stage C)

export type RunState =
  | "queued" | "spawning" | "interviewing" | "planning" | "executing" | "waiting_answer" | "verifying" | "judging"
  | "needs_you" | "awaiting_integration" | "awaiting_approval" | "stopping" | "replanning"
  | "completed" | "stopped" | "capped" | "failed";
export type Run = {
  id: string; task: { id: string; title: string; status: string } | null; state: RunState; state_revision: number;
  state_reason: string; reason_text: string; state_since: string; generation: number; plan_revision: number;
  tool: string; model: string; branch: string; base_branch: string; worktree: string; effective_permission_mode: string;
  turn_count: number; turn_cap: number; started_at: string | null; finished_at: string | null;
  stop_requested_at: string | null; stop_mode: string; surf: string;
  host: { id: string; name: string; connected: boolean } | null; structured_answers: boolean;
  session: { id: string; state: string; since: string | null } | null; created_at: string;
};
export type QuestionItem = { id: string; text: string; header?: string; choices?: { id: string; label: string; description?: string }[]; multi?: boolean };
export type RunQuestion = {
  id: string; question_id: string; run_id: string | null; generation: number; items: QuestionItem[];
  status: "open" | "submitted" | "accepted" | "timed_out" | "failed" | "canceled" | "answered_locally" | "expired";
  answers: unknown; answered_via: string; remote_answerable: boolean; hold_until: string | null; detail: string;
  created_at: string; resolved_at: string | null;
};
export type RunCommand = {
  id: string; op_id: string; kind: string; status: string; generation: number; mode: string | null; text: string | null;
  outcome: { state?: string; detail?: string } | null; created_at: string; updated_at: string;
};
export type HistoryEntry = { at: string; from?: string; to?: string; reason?: string; detail?: string; event?: string;
  kind?: string; state?: string; actor?: { kind: string } };
export type RunDetail = Run & {
  plan: { revision: number; title: string; prd_text: string; steps: { id: string; title: string; status: string }[]; criteria: { id: string; text: string }[] } | null;
  history: HistoryEntry[];
  turns: { id: string; seq: number; text: string; clipped: boolean; truncated: boolean; generation: number; at: string; prompt: string }[];
  commands: RunCommand[]; questions: RunQuestion[];
};
export type TimelineEntry = { at: string; kind: "turn" | "command" | "question" | "state"; title: string; detail: string; tone: "plain" | "good" | "warn" | "attention" };

export const TERMINAL_RUN_STATES: RunState[] = ["completed", "stopped", "capped", "failed"];
const RUN_LABELS: Partial<Record<RunState, string>> = {
  queued: "Queued", spawning: "Starting", executing: "Working", waiting_answer: "Asked you", needs_you: "Needs you",
  awaiting_approval: "Awaiting approval", stopping: "Stopping…", completed: "Completed", stopped: "Stopped",
  capped: "Turn cap", failed: "Failed", interviewing: "Interviewing", planning: "Planning", verifying: "Verifying",
  judging: "Judging", awaiting_integration: "Awaiting integration", replanning: "Replanning",
};
export function runLabel(state: RunState) {
  return RUN_LABELS[state] ?? state;
}
export function isSettled(run: Pick<Run, "state">) {
  return TERMINAL_RUN_STATES.includes(run.state);
}

/** Which controls the drawer offers, and why one is off. Stop stays available while stopping (to force a kill). */
export function runControls(run: Run) {
  const settled = isSettled(run), stopping = run.state === "stopping";
  const offline = !run.host?.connected;
  const started = !!run.surf;
  const why = settled ? "This run has finished." : stopping ? "The run is stopping." : offline ? "The remotea server is offline; it will receive this when it reconnects."
    : !started ? "The agent has not started yet." : "";
  return {
    message: { enabled: !settled && !stopping && started, reason: why },
    steer: { enabled: !settled && !stopping && started, reason: why },
    stop: { enabled: !settled, label: stopping ? (run.stop_mode === "kill" ? "Stopping…" : "Force stop") : "Stop", mode: stopping ? "kill" as const : "interrupt" as const },
    approve: { enabled: !settled && !stopping, reason: settled ? "This run has finished." : stopping ? "The run is stopping." : "" },
    offline,
  };
}

/** The task detail's Run button: hidden unless remotea is on; disabled with a reason when no host can take it. */
export function runButton(enabled: boolean, hosts: Host[], active: Run | null) {
  if (!enabled) return { visible: false, disabled: true, label: "Run", reason: "" };
  if (active) return { visible: true, disabled: false, label: "Open run", reason: runLabel(active.state), run: active.id };
  if (!hosts.length) return { visible: true, disabled: true, label: "Run", reason: "Connect your remotea server first (Settings → Connections)." };
  if (!hosts.some(h => h.connected)) return { visible: true, disabled: true, label: "Run", reason: "The remotea server is offline." };
  return { visible: true, disabled: false, label: "Run", reason: "" };
}

/** How the questions card treats a question: answer here, waiting for the agent, terminal only, or done. */
export function questionMode(q: RunQuestion, run: Pick<Run, "structured_answers" | "generation" | "state">) {
  if (["accepted", "answered_locally", "canceled", "expired"].includes(q.status)) return "closed" as const;
  if (q.status === "submitted") return "pending" as const;
  if (!run.structured_answers || !q.remote_answerable || q.status === "timed_out" || q.status === "failed") return "terminal" as const;
  if (q.generation !== run.generation || isSettled(run as Run) || run.state === "stopping") return "closed" as const;
  return "answer" as const;
}

/** Answers for one group, from chosen option ids and free text per item. Returns null until every item has one. */
export function buildAnswers(q: RunQuestion, picked: Record<string, string[]>, typed: Record<string, string>) {
  const answers = q.items.map(item => {
    const choice_ids = (picked[item.id] ?? []).filter(id => item.choices?.some(c => c.id === id));
    const text = (typed[item.id] ?? "").trim();
    return { item_id: item.id, ...(choice_ids.length ? { choice_ids } : {}), ...(text ? { text } : {}) };
  });
  return answers.every(a => "choice_ids" in a || "text" in a) ? answers : null;
}

function commandTitle(c: RunCommand) {
  const what = c.kind === "input" ? (c.mode === "steer" ? "Steer" : "Message") : c.kind === "spawn" ? "Start" : c.kind === "stop" ? "Stop" : c.kind === "answer" ? "Answer" : c.kind;
  const state = c.outcome?.state ?? c.status;
  const shown: Record<string, string> = { queued: "queued", sent: "sent", acked: "accepted by remotea", accepted: "accepted by remotea",
    executed: "done", failed: "failed", rejected: "refused", uncertain: "delivery uncertain", expired: "expired", canceled: "canceled" };
  return `${what} · ${shown[state] ?? state}`;
}
function commandTone(c: RunCommand): TimelineEntry["tone"] {
  const state = c.outcome?.state ?? c.status;
  return state === "executed" ? "good" : ["failed", "rejected", "expired"].includes(state) ? "warn" : state === "uncertain" ? "attention" : "plain";
}
const QUESTION_TEXT: Record<RunQuestion["status"], string> = {
  open: "waiting for you", submitted: "answer sent, waiting for the agent", accepted: "answer accepted",
  timed_out: "timed out here: answer in the terminal", failed: "could not be delivered", canceled: "canceled",
  answered_locally: "answered in the terminal", expired: "expired",
};

/** One time-ordered timeline: final responses per turn, commands with their outcomes, questions and state changes. */
export function timeline(run: RunDetail): TimelineEntry[] {
  const entries: TimelineEntry[] = [];
  for (const t of run.turns) entries.push({ at: t.at, kind: "turn", title: `Turn ${t.seq}`, detail: t.text + (t.truncated ? " [truncated]" : ""), tone: "plain" });
  for (const c of run.commands) entries.push({ at: c.created_at, kind: "command", title: commandTitle(c), detail: c.text ?? c.outcome?.detail ?? "", tone: commandTone(c) });
  for (const q of run.questions) entries.push({ at: q.created_at, kind: "question", title: "Question · " + QUESTION_TEXT[q.status],
    detail: q.items.map(i => i.text).join(" / "), tone: q.status === "open" || q.status === "timed_out" ? "attention" : q.status === "accepted" ? "good" : "plain" });
  for (const h of run.history) {
    if (!h.to) continue;
    entries.push({ at: h.at, kind: "state", title: runLabel(h.to as RunState), detail: [h.actor?.kind === "human" ? "by you" : "", h.detail ?? ""].filter(Boolean).join(" · "),
      tone: h.to === "needs_you" ? "attention" : h.to === "completed" ? "good" : ["failed", "stopped"].includes(h.to) ? "warn" : "plain" });
  }
  return entries.sort((a, b) => new Date(a.at).getTime() - new Date(b.at).getTime());
}

/** Needs-you and questions first, then working runs, then the rest, newest first. */
export function orderRuns(runs: Run[]) {
  const rank = (r: Run) => (["needs_you", "waiting_answer", "awaiting_approval"].includes(r.state) ? 0 : isSettled(r) ? 2 : 1);
  return [...runs].sort((a, b) => rank(a) - rank(b) || new Date(b.state_since).getTime() - new Date(a.state_since).getTime());
}

/** ?view=remotea&run=<id> from a push notification opens that run's drawer. */
export function linkedRun(search: string) {
  const id = new URLSearchParams(search).get("run");
  return id && /^[0-9a-f-]{36}$/i.test(id) ? id : null;
}

export type SavedTarget = { id: string; host_id: string; host_target: string; name: string; repo: string; worktree_root: string; record_id: string | null };
export type AdvertisedTarget = { host_id: string; host_name: string; host_target: string; repo: string; worktree_root: string };
export type NeedsTarget = { task_id: string; candidates: SavedTarget[]; advertised: AdvertisedTarget[]; records: { id: string; title: string }[] };
export type UnmergedWork = { host_id: string; host_name: string; target: string | null; path: string | null; branch: string | null;
  base: string | null; ahead: number | null; behind: number | null; unpushed: number | null; uncommitted: number | null;
  untracked: number | null; merged: boolean | null; active: boolean | null; reasons: string[] | null };

/** Picker options when a start needs a target: saved targets first, then advertised ones not yet saved. */
export function targetOptions(need: NeedsTarget) {
  const saved = need.candidates.map(t => ({ key: `saved:${t.id}`, label: t.name || t.host_target, detail: t.repo, saved: t }));
  const known = new Set(need.candidates.map(t => `${t.host_id}:${t.host_target}`));
  const offered = need.advertised.filter(t => !known.has(`${t.host_id}:${t.host_target}`))
    .map(t => ({ key: `host:${t.host_id}:${t.host_target}`, label: t.host_target, detail: t.repo || t.host_name, advertised: t }));
  return [...saved, ...offered];
}

/** A short "why it is listed" line for an unmerged worktree. */
export function unmergedSummary(w: UnmergedWork) {
  const parts = [w.ahead ? `${w.ahead} ahead of ${w.base || "base"}` : "", w.unpushed ? `${w.unpushed} unpushed` : "",
    w.uncommitted ? `${w.uncommitted} uncommitted` : "", w.untracked ? `${w.untracked} untracked` : ""].filter(Boolean);
  return parts.length ? parts.join(" · ") : (w.reasons ?? []).join(", ") || "unmerged";
}
