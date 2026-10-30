// Pure helpers for the remotea page: live agent sessions reported over the agent link.

export type LiveState = { state: string; since: string | null; host_id: string | null; reason: string };
export type Host = {
  id: string; name: string; machine: string; connected: boolean; last_seen_at: string | null;
  conn_epoch: number; caps: string[]; harnesses: Record<string, unknown>; proto: number; status?: "ok" | "degraded";
  /** remotea's own version from hello; `update` when it trails the latest release by more than one (older Eridani: absent). */
  version?: string | null; update?: { latest: string; url: string | null } | null;
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
  hosts: Host[]; link_url: string; connect_command: string; token?: string | null; auto_complete?: boolean; quiet_bypass?: boolean;
};
export type Parity = {
  total: number; matched: number; missing: number; mismatched: number; clean: boolean;
  fields: Record<string, number>; first_report_at: string | null; last_report_at: string | null;
  recent: { tool: string; session_id: string; missing: boolean; diff: Record<string, { stored: string; reported: string }>; updated_at: string }[];
  /** Stage E: consecutive clean parity days (no session differed) ending today or yesterday. */
  cutover?: { ready: boolean; streak: number; needed: number;
    days: { day: string; reports: number; matched?: number; mismatched?: number; missing?: number; clean: boolean }[] };
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

/** "remotea 0.1.0 · update available" for Settings; null when the host never said its version. */
export function versionLine(host: Host): { text: string; update: { latest: string; url: string | null } | null } | null {
  if (!host.version) return null;
  return { text: "remotea " + host.version + (host.update ? " · update available" : ""), update: host.update ?? null };
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

/** The "ready for cutover" indicator: seven clean parity days in a row retire the Python collector. */
export function cutoverSummary(parity: Parity | null): { ready: boolean; text: string } | null {
  const c = parity?.cutover;
  if (!c) return null;
  if (c.ready) return { ready: true, text: `Ready for cutover: ${c.needed} clean parity days in a row. See scripts/session_collector/README.md, "Retirement".` };
  const silent = c.days.slice(0, 2).every(d => !d.reports);
  return { ready: false, text: `Not ready for cutover: ${c.streak} of ${c.needed} clean days in a row` +
    (silent ? "; no parity reports since yesterday." : ".") };
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
  plan_pending?: boolean; turn_seq?: number; integration?: { task_id: string; reason: string; branch?: string }[];
  /** The full text behind the current state (the judge's notes, the proposal summary); reason_text is the one-liner. */
  reason_detail?: string;
  completion_policy?: "code_ready" | "pr_open" | "pr_merged" | string;
  /** What the run's worktree is based on; behind is live from the host's worktrees report when `live`. Null: older host. */
  base?: RunBase | null;
  linear?: Partial<Record<"in_progress" | "done" | "comment", LinearOutcome>>;
  session: { id: string; state: string; since: string | null } | null; created_at: string;
};
export type MigrationTarget = {
  host_id: string; host_name: string; host_target: string; dir: string; head: number | null; override: number | null;
  last_reserved: number | null; next: number; reported_at: string | null;
};

export type RunBase = {
  sha: string; ref: string; branch: string; fetch?: { ok: boolean; error?: string } | null;
  behind?: number | null; behind_ref?: string | null; as_of?: string | null; live?: boolean;
};

/** The run header's base line: "based on main @ 1a2b3c4 · 3 behind · as of 2m ago"; null when the host did not say. */
export function baseLine(base: RunBase | null | undefined, now = Date.now()): string | null {
  if (!base?.sha) return null;
  const failed = base.fetch && !base.fetch.ok;
  const name = failed ? `local ${base.branch || base.ref} (fetch failed)` : (base.branch || base.ref.replace(/^origin\//, ""));
  const parts = [`based on ${name} @ ${base.sha.slice(0, 7)}`];
  if (typeof base.behind === "number") parts.push(base.behind ? `${base.behind} behind` : "up to date");
  if (base.as_of) parts.push("as of " + ago(base.as_of, now));
  return parts.join(" · ");
}

export type LinearOutcome = { status: string; code?: string | null; message?: string | null; url?: string | null; reconciled?: boolean | null };

const LINEAR_WHAT = { in_progress: "In Progress", done: "Done", comment: "Completion comment" } as const;
const LINEAR_STATUS: Record<string, string> = {
  queued: "queued", retrying: "retrying", posted: "posted", deferred: "waits for the start write", skipped: "not linked",
  blocked: "not sent", failed: "failed", unconfirmed: "unconfirmed: check Linear",
};
/** Linear sync outcomes are reported apart from the run: a failed write never undoes local work. */
export function linearOutcomes(run: Pick<Run, "linear">) {
  return (Object.keys(LINEAR_WHAT) as (keyof typeof LINEAR_WHAT)[]).flatMap(key => {
    const o = run.linear?.[key];
    if (!o || o.status === "skipped") return [];
    const detail = o.status === "blocked" || o.status === "failed" ? o.message || o.code || "" : "";
    return [{ key, text: `${LINEAR_WHAT[key]}: ${LINEAR_STATUS[o.status] ?? o.status}${detail ? " (" + detail + ")" : ""}`,
      tone: ["failed", "blocked", "unconfirmed"].includes(o.status) ? "warn" as const : "ok" as const, url: o.url ?? null }];
  });
}
export const POLICY_TEXT: Record<string, string> = { code_ready: "Done when the code is ready", pr_open: "Done when a pull request is open", pr_merged: "Done when the pull request is merged" };

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
  kind?: string; state?: string; actor?: { kind: string; reason?: string }; text?: string;
  plan_revision?: number; proposals?: string; accepted?: string[]; kept?: string[] };
export type InterviewQuestion = { id: string; question: string; header?: string; options: { label: string; description?: string }[];
  multi: boolean; free_text: boolean };
export type Interview = { id: string; run_id: string; generation: number; questions: InterviewQuestion[];
  answers: Record<string, string | string[]>; status: "open" | "answered" | "superseded"; answered_at: string | null; created_at: string };
export type Criterion = { id: string; text: string; kind?: "text" | "verify"; verify_id?: string };
export type CriterionVerdict = { id: string; status: "met" | "not_met" | "unclear"; evidence: string };
export type Attempt = {
  id: string; kind: "verify" | "judge"; status: string; generation: number; turn_seq: number; plan_revision: number;
  verify_id: string; initiator: string; verdict: "met" | "not_met" | "needs_you" | null; gap: string | null; summary: string | null;
  criteria: CriterionVerdict[] | null; rules: string[] | null; non_verdict: string | null; exit_code: number | null;
  tail: string | null; on_demand: boolean | null; created_at: string; updated_at: string;
};
/** What the agent proposes to change about what you set (agent_plan.plan_proposals); you accept or keep each. */
export type Proposals = {
  criteria?: { id: string; change: "reword" | "remove"; text: string; proposed?: string; because: string }[];
  verify?: { current: { target_id: string; verify_id: string }; proposed: { target_id: string; verify_id: string } };
  completion_policy?: { current: string; proposed: string };
};
export type RunDetail = Run & {
  plan: { revision: number; title: string; prd_text: string; steps: { id: string; title: string; status: string; task_id?: string }[];
    criteria: Criterion[]; verify?: { target_id: string; verify_id: string } | null; required_children?: string[];
    proposals?: Proposals | null } | null;
  history: HistoryEntry[];
  turns: { id: string; seq: number; text: string; clipped: boolean; truncated: boolean; generation: number; at: string; prompt: string }[];
  commands: RunCommand[]; questions: RunQuestion[]; attempts?: Attempt[]; interviews?: Interview[];
};
export type TimelineEntry = { at: string; kind: "turn" | "command" | "question" | "state" | "attempt" | "plan"; title: string; detail: string;
  tone: "plain" | "good" | "warn" | "attention"; who: "agent" | "you" | "eridani" };

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
    approve: { enabled: !settled && !stopping, primary: run.state === "awaiting_approval",
      reason: settled ? "This run has finished." : stopping ? "The run is stopping." : "" },
    replan: { enabled: started && ["executing", "needs_you", "awaiting_approval", "awaiting_integration", "planning"].includes(run.state),
      reason: !started ? "The agent has not started yet." : "Re-plan works while the run is working, waiting for you or awaiting approval." },
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

/** Interview answers keyed by question id; null until every question has an allowed answer. */
export function interviewAnswers(interview: Interview, picked: Record<string, string[]>, typed: Record<string, string>) {
  const answers: Record<string, string | string[]> = {};
  for (const q of interview.questions) {
    const labels = new Set(q.options.map(o => o.label));
    const chosen = (picked[q.id] ?? []).filter(l => labels.has(l));
    const text = (typed[q.id] ?? "").trim();
    const values = [...chosen, ...(text ? [text] : [])];  // Typing an answer is always allowed.
    if (!values.length || (!q.multi && values.length > 1)) return null;
    answers[q.id] = q.multi ? values : values[0];
  }
  return answers;
}

/** An option the agent marked as its recommendation ("(Recommended)" in the label or description). */
export function recommended(option: { label: string; description?: string }) {
  return /recommended/i.test(option.label) || /recommended/i.test(option.description ?? "");
}

/** Recommended options first, the rest in the agent's order. */
export function orderOptions<T extends { label: string; description?: string }>(options: T[]) {
  return [...options.filter(recommended), ...options.filter(o => !recommended(o))];
}

export type PlanDecision = { accept: boolean; text: string };
/** The plan-review request: one decision per proposed criterion change, plus verify and policy. */
export function planReviewBody(proposals: Proposals, decisions: Record<string, PlanDecision>, verify: "accept" | "keep",
  completion_policy: "accept" | "keep", expected_revision: number) {
  return {
    decisions: (proposals.criteria ?? []).map(c => {
      const d = decisions[c.id] ?? { accept: true, text: "" };
      const text = d.text.trim();
      return { id: c.id, accept: d.accept, ...(d.accept && c.change === "reword" && text && text !== c.proposed ? { text } : {}) };
    }),
    verify: proposals.verify ? verify : "keep", completion_policy: proposals.completion_policy ? completion_policy : "keep",
    expected_revision,
  };
}

/** The newest attempt of a kind (the API lists newest first). */
export function latestAttempt(run: Pick<RunDetail, "attempts">, kind: Attempt["kind"]) {
  return (run.attempts ?? []).find(a => a.kind === kind) ?? null;
}

/** Each frozen criterion with the latest judge's status and evidence (pending until a verdict exists). */
export function criteriaEvidence(run: Pick<RunDetail, "plan" | "attempts">) {
  const judged = (run.attempts ?? []).find(a => a.kind === "judge" && a.criteria);
  const byId = new Map((judged?.criteria ?? []).map(c => [c.id, c]));
  return (run.plan?.criteria ?? []).map(c => {
    const found = byId.get(c.id);
    return { ...c, status: found?.status ?? "pending", evidence: found?.evidence ?? "", turn: judged?.turn_seq ?? null };
  });
}

const VERIFY_TEXT: Record<string, string> = { running: "running", passed: "passed", failed: "failed", timed_out: "timed out",
  refused: "not run (refused)", uncertain: "result uncertain", stale: "stale: code changed during the run", canceled: "canceled",
  error: "error", superseded: "superseded" };
const VERDICT_TEXT: Record<string, string> = { met: "criteria met", not_met: "not met yet", needs_you: "needs you" };
export function attemptTitle(a: Attempt) {
  if (a.kind === "verify") return `Verification ${a.verify_id} · ${VERIFY_TEXT[a.status] ?? a.status}${a.on_demand ? " (on demand)" : ""}`;
  if (a.status === "done" && a.verdict) return "Judge · " + VERDICT_TEXT[a.verdict];
  if (a.status === "failed") return "Judge · no verdict (" + (a.non_verdict ?? "error") + ")";
  return "Judge · " + a.status;
}
function attemptTone(a: Attempt): TimelineEntry["tone"] {
  if (a.kind === "verify") return a.status === "passed" ? "good" : a.status === "running" ? "plain" : "warn";
  return a.verdict === "met" ? "good" : a.verdict === "needs_you" || a.status === "failed" ? "attention" : "plain";
}

function commandTitle(c: RunCommand) {
  const what = c.kind === "input" ? (c.mode === "steer" ? "Steer" : "Message") : c.kind === "spawn" ? "Start" : c.kind === "stop" ? "Stop"
    : c.kind === "answer" ? "Answer" : c.kind === "verify" ? "Verify" : c.kind;
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
  // A text that left for the agent appears once: the "sent" history entry says who sent it, so the matching
  // input command (same text) is skipped unless it failed, where its outcome is the news.
  const sent = new Set(run.history.filter(h => h.event === "sent" && h.text).map(h => h.text as string));
  for (const t of run.turns) entries.push({ at: t.at, kind: "turn", title: `Turn ${t.seq}`, detail: t.text + (t.truncated ? " [truncated]" : ""), tone: "plain", who: "agent" });
  for (const c of run.commands) {
    const state = c.outcome?.state ?? c.status;
    if (c.kind === "input" && c.text && sent.has(c.text) && !["failed", "rejected", "expired", "uncertain"].includes(state)) continue;
    entries.push({ at: c.created_at, kind: "command", title: commandTitle(c), detail: c.kind === "input" && sent.has(c.text ?? "") ? c.outcome?.detail ?? "" : c.text ?? c.outcome?.detail ?? "",
      tone: commandTone(c), who: "eridani" });
  }
  for (const q of run.questions) entries.push({ at: q.created_at, kind: "question", title: "Question · " + QUESTION_TEXT[q.status],
    detail: q.items.map(i => i.text).join(" / "), tone: q.status === "open" || q.status === "timed_out" ? "attention" : q.status === "accepted" ? "good" : "plain", who: "agent" });
  for (const a of run.attempts ?? []) entries.push({ at: a.updated_at, kind: "attempt", title: attemptTitle(a),
    detail: a.kind === "judge" ? (a.verdict === "met" ? a.summary ?? "" : a.gap ?? "") : (a.status === "failed" ? a.tail ?? "" : ""), tone: attemptTone(a), who: "eridani" });
  for (const i of run.interviews ?? []) entries.push({ at: i.created_at, kind: "question", title: "Interview · " + (i.status === "open" ? "waiting for you" : i.status),
    detail: i.questions.map(q => q.question).join(" / "), tone: i.status === "open" ? "attention" : "plain", who: "agent" });
  for (const h of run.history) {
    if (h.event === "sent" && h.text) {
      // Every text sent to the agent, with who sent it (stage F): you on this page, Eri after your confirmation,
      // or Eridani itself (interview answers as the planning prompt, the judge's gap).
      const confirmed = !!h.actor?.reason?.startsWith("confirmed:");
      const human = h.actor?.kind === "human";
      const who = confirmed ? (human ? "Eri, confirmed by you" : "Eri, after your yes") : human ? "you" : h.actor?.kind === "run_judge" ? "Eridani" : h.actor?.kind ?? "";
      entries.push({ at: h.at, kind: "command", title: "Sent by " + who, detail: h.text, tone: "plain", who: human && !confirmed ? "you" : "eridani" });
      continue;
    }
    if (h.event === "plan_submitted") {
      entries.push({ at: h.at, kind: "plan", title: `Plan submitted · revision ${h.plan_revision ?? ""}`.trim(),
        detail: h.proposals ? h.proposals + " — waiting for your review." : "", tone: h.proposals ? "attention" : "plain", who: "agent" });
      continue;
    }
    if (h.event === "plan_reviewed") {
      const accepted = (h.accepted ?? []).join(", "), kept = (h.kept ?? []).join(", ");
      entries.push({ at: h.at, kind: "plan", title: `Plan reviewed by you · revision ${h.plan_revision ?? ""}`.trim(),
        detail: [accepted ? "accepted " + accepted : "", kept ? "kept " + kept : ""].filter(Boolean).join(" · "), tone: "good", who: "you" });
      continue;
    }
    if (!h.to) continue;
    entries.push({ at: h.at, kind: "state", title: runLabel(h.to as RunState), detail: [h.actor?.kind === "human" ? "by you" : "", h.detail ?? ""].filter(Boolean).join(" · "),
      tone: h.to === "needs_you" ? "attention" : h.to === "completed" ? "good" : ["failed", "stopped"].includes(h.to) ? "warn" : "plain", who: "eridani" });
  }
  return entries.sort((a, b) => new Date(a.at).getTime() - new Date(b.at).getTime());
}

/** Needs-you and questions first, then working runs, then the rest, newest first. */
export function orderRuns(runs: Run[]) {
  const rank = (r: Run) => (["needs_you", "waiting_answer", "awaiting_approval", "awaiting_integration"].includes(r.state) ? 0 : isSettled(r) ? 2 : 1);
  return [...runs].sort((a, b) => rank(a) - rank(b) || new Date(b.state_since).getTime() - new Date(a.state_since).getTime());
}

/** ?view=remotea&run=<id> from a push notification opens that run's drawer. */
export function linkedRun(search: string) {
  const id = new URLSearchParams(search).get("run");
  return id && /^[0-9a-f-]{36}$/i.test(id) ? id : null;
}

export type SavedTarget = { id: string; host_id: string; host_target: string; name: string; repo: string; worktree_root: string; record_id: string | null };
export type AdvertisedTarget = { host_id: string; host_name: string; host_target: string; repo: string; worktree_root: string };
export type NeedsTarget = { task_id: string; candidates: SavedTarget[]; advertised: AdvertisedTarget[];
  records: { id: string; title: string; type_id?: string }[] };
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

/** The picker's default "Remember for": the nearest record that is not a task (usually the project), so the choice
 * also covers the task's siblings; else the task's own record; else "task" (only this task). */
export function defaultScope(need: NeedsTarget) {
  return (need.records.find(r => r.type_id && r.type_id !== "task") ?? need.records[0])?.id ?? "task";
}

/** A NEEDS_TARGET error's data, when it carries what the picker needs (a target without a server does not). */
export function needsTargetData(data: unknown): NeedsTarget | null {
  const need = data as Partial<NeedsTarget> | null;
  return need && typeof need.task_id === "string" && Array.isArray(need.candidates) && Array.isArray(need.advertised)
    && Array.isArray(need.records) ? need as NeedsTarget : null;
}

/** A short "why it is listed" line for an unmerged worktree. */
export function unmergedSummary(w: UnmergedWork) {
  const parts = [w.ahead ? `${w.ahead} ahead of ${w.base || "base"}` : "", w.unpushed ? `${w.unpushed} unpushed` : "",
    w.uncommitted ? `${w.uncommitted} uncommitted` : "", w.untracked ? `${w.untracked} untracked` : ""].filter(Boolean);
  return parts.length ? parts.join(" · ") : (w.reasons ?? []).join(", ") || "unmerged";
}
