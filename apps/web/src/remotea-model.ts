// Pure helpers for the remotea page: live agent sessions reported over the agent link.

export type LiveState = { state: string; since: string | null; host_id: string | null; reason: string };
export type Host = {
  id: string; name: string; machine: string; connected: boolean; last_seen_at: string | null;
  conn_epoch: number; caps: string[]; harnesses: Record<string, unknown>; proto: number; status?: "ok" | "degraded";
  /** remotea's own version from hello; `update` when it trails the latest release by more than one (older Eridani: absent). */
  version?: string | null; update?: { latest: string; url: string | null } | null;
  /** Build identity from hello (remotea PR #18): the commit (40 hex or "unknown"), its short form, uncommitted
   * changes, the caps shape and the hook helper's hash. Null when the host did not say. */
  build_sha?: string | null; build?: string | null; build_dirty?: boolean | null; caps_version?: number | null; hook_version?: string | null;
  /** Set only while the host's journal or outbox is degraded (remotea PR #15): it refuses effectful commands. */
  degraded_reason?: string | null;
  /** The host's own self-check (`remotea doctor`, S-Q1); null when it sent none (older remotea), absent from older Eridani. */
  doctor?: DoctorSummary | null;
};
/** `remotea doctor`'s summary as Eridani keeps it: when the host computed it (`at`), when Eridani received it, the
 * counts over every check, and the warn/fail lines (fails first, at most 40; `truncated` when some were left out). */
export type DoctorItem = { id: string; level: "warn" | "fail"; summary: string; fix?: string };
export type DoctorSummary = { at: string; received_at?: string | null; ok: number; warn: number; fail: number; items: DoctorItem[]; truncated?: boolean };
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
  /** The MCP key the last setup code carried (prefix and expiry only); `setup_code` is in a minting response only. */
  mcp_key?: { id: string; prefix: string; expires_at: string; last_used_at: string | null } | null;
  setup_command?: string; setup_code?: string | null;
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
/** "remotea 0.1.0 · build 76c5f62" (plus the "dirty" badge when the build had uncommitted changes); null when
 * the host said neither its version nor its build. */
export function versionLine(host: Host): { text: string; update: { latest: string; url: string | null } | null; dirty: boolean } | null {
  const build = host.build ?? (host.build_sha && host.build_sha !== "unknown" ? host.build_sha.slice(0, 7) : host.build_sha) ?? null;
  if (!host.version && !build) return null;
  const parts = [host.version ? "remotea " + host.version : "remotea", build ? "build " + build : "", host.update ? "update available" : ""];
  return { text: parts.filter(Boolean).join(" · "), update: host.update ?? null, dirty: !!host.build_dirty };
}

/** The warning a degraded host shows: its journal or outbox cannot persist, so it refuses starts and messages. */
export function hostWarning(host: Pick<Host, "degraded_reason">) {
  return host.degraded_reason ? "Host journal degraded: " + host.degraded_reason : null;
}

/** The page header's host line. `warning` is set while a host's journal is degraded (one line per host). */
export function hostSummary(hosts: Host[], now = Date.now()): { tone: "never" | "connected" | "offline"; text: string; warning?: string } {
  if (!hosts.length) return { tone: "never" as const, text: "No remotea server has connected yet." };
  const warnings = hosts.map(h => hostWarning(h) && (hosts.length > 1 ? `${h.name || h.machine || "remotea"}: ` : "") + hostWarning(h)).filter(Boolean);
  const warning = warnings.length ? { warning: warnings.join(" ") } : {};
  const online = hosts.filter(h => h.connected);
  if (online.length) return { tone: "connected" as const, text: online.map(h => h.name || h.machine || "remotea").join(", ") + " connected", ...warning };
  const last = hosts.map(h => h.last_seen_at).filter(Boolean).sort().at(-1) ?? null;
  return { tone: "offline" as const, text: "Disconnected · last seen " + ago(last, now), ...warning };
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
  host: { id: string; name: string; connected: boolean; degraded_reason?: string | null;
    /** The host's lease, renewed every 30 s while connected (older Eridani: absent). */
    last_seen_at?: string | null } | null; structured_answers: boolean;
  /** Whether the agent was spawned with Eridani's MCP tools (spawn outcome; null when the host did not say). */
  mcp?: TargetMcp | null;
  /** Freshness facts kept apart (E-S6; older Eridani: absent): when the last turn landed, the latest verification
   * and how far it proves the current turn, and controls the host has not taken yet or whose delivery is uncertain. */
  last_turn_at?: string | null; verification?: VerificationFact | null; pending_controls?: PendingCommand[];
  plan_pending?: boolean; turn_seq?: number; integration?: { task_id: string; reason: string; branch?: string }[];
  /** Freestyle: you talk to the agent directly; turns are recorded but not verified, judged or capped. */
  freestyle?: boolean;
  /** The full text behind the current state (the judge's notes, the proposal summary); reason_text is the one-liner. */
  reason_detail?: string;
  completion_policy?: "code_ready" | "pr_open" | "pr_merged" | string;
  /** What the run's worktree is based on; behind is live from the host's worktrees report when `live`. Null: older host. */
  base?: RunBase | null;
  linear?: Partial<Record<"in_progress" | "done" | "comment", LinearOutcome>>;
  /** One plain-language line on what the agent is doing now, written by a small model (run_summaries.py); "" until
   * the first one. summary_stale: something newer landed (a turn, a question, a verdict) and a fresh line is due. */
  summary?: string; summary_at?: string | null; summary_stale?: boolean;
  session: { id: string; state: string; since: string | null } | null; created_at: string;
};
/** The run's latest verification as a decision fact (agent_control.verification_fact). `evidence`: `fresh` (this
 * turn, an unchanged workspace), `stale` (an older turn, or the checkout changed around it), `incomplete` (remotea's
 * fingerprint missed part of the workspace), `pending` (still running). */
export type VerificationFact = {
  verify_id: string; status: string; at: string; evidence: "fresh" | "stale" | "incomplete" | "pending"; turn_seq?: number;
  isolated?: boolean | null; isolation_reason?: string | null; workspace_complete?: boolean | null; workspace_reasons?: string[];
  deadline_at?: string | null;
};
/** A command whose effect is not known yet: `pending` (the host has not taken it) or `uncertain` (it could not tell). */
export type PendingCommand = { kind: string; status: "pending" | "uncertain"; mode?: string | null; at: string };
export type MigrationTarget = {
  host_id: string; host_name: string; host_target: string; configured: boolean; source: "eridani" | "host" | null;
  dir: string | null; pattern: string | null; host_dir: string | null; head: number | null; override: number | null;
  last_reserved: number | null; next: number | null; reported_at: string | null;
};
export type MigrationDraft = { dir: string; pattern: string; override: string };

/** The draft a Settings row starts from: Eridani's own values (blank when the host's config applies). */
export function migrationDraft(m: MigrationTarget): MigrationDraft {
  return { dir: m.source === "eridani" ? m.dir ?? "" : "", pattern: m.source === "eridani" ? m.pattern ?? "" : "",
    override: m.override == null ? "" : String(m.override) };
}

/** The POST body for a row, or the reason it cannot be saved (the server checks the same and more). */
export function migrationBody(m: MigrationTarget, d: MigrationDraft):
  { host_id: string; host_target: string; dir: string | null; pattern: string | null; override: number | null } | string {
  const dir = d.dir.trim(), pattern = d.pattern.trim(), override = d.override.trim();
  if (dir && (dir.startsWith("/") || dir.startsWith("~") || dir.split("/").includes(".."))) return "The folder is relative to the repository, such as migrations/versions.";
  if (pattern && !dir) return "Set the folder together with its file pattern.";
  if (pattern) {
    try { if (!/\((?!\?)/.test(pattern)) return "The file pattern needs a group around the number, such as ^(\\d{4})_."; new RegExp(pattern); }
    catch { return "The file pattern is not a valid regular expression."; }
  }
  if (override && !/^\d{1,5}$/.test(override)) return "The chain head override is a migration number, such as 33.";
  return { host_id: m.host_id, host_target: m.host_target, dir: dir || null, pattern: pattern || null, override: override ? Number(override) : null };
}

/** "migrations/versions (from the host) · head 0032 · next 0033"; for a target with no folder, how to turn it on. */
export function migrationSummary(m: MigrationTarget, now = Date.now()): string {
  const pad = (n: number | null) => n == null ? "none" : String(n).padStart(4, "0");
  if (!m.configured) return "Off: set the folder to have Eridani reserve migration numbers for this repository.";
  const parts = [`${m.dir}${m.source === "host" ? " (from the host)" : ""}`,
    m.head != null ? `head ${pad(m.head)} on the host${m.reported_at ? " (" + ago(m.reported_at, now) + ")" : ""}`
      : m.host_dir && m.host_dir !== m.dir ? `the host reports ${m.host_dir}, not this folder` : "no head reported"];
  if (m.override != null) parts.push(`override ${pad(m.override)}`);
  if (m.last_reserved != null) parts.push(`last reserved ${pad(m.last_reserved)}`);
  parts.push(`next ${pad(m.next)}`);
  return parts.join(" · ");
}

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
  /** Verify only (remotea PR #24; null from older hosts): sandboxed or not, apart from the exit code; whether the
   * fingerprint covered the whole workspace; the entry's advertised timeout and the deadline it set. */
  isolated?: boolean | null; isolation?: "bwrap" | "none" | null; isolation_reason?: string | null;
  workspace_complete?: boolean | null; workspace_reasons?: string[] | null; timeout_s?: number | null;
  stale_workspace?: boolean | null; deadline_at?: string | null;
  /** Why a verification did not pass (run_coordinator.classify_verify) and the one-line fix; null when it passed. */
  category?: VerifyCategory | null; action?: string | null;
  /** How long the verify command ran (null from older Eridani). */
  duration_ms?: number | null;
};
/** What the run changed (GET /runs/{id} `changes`): the session's reported PRs and the host's worktree row. */
export type RunChanges = {
  branch: string; worktree: string; pr_urls: string[]; pr_as_of: string | null; ahead?: number | null; behind?: number | null;
  unpushed?: number | null; uncommitted?: number | null; untracked?: number | null; merged?: boolean | null; reported_at: string | null;
};
export type VerifyCategory = "tests_failed" | "infra_missing_binary" | "infra_missing_dependency" | "infra_database"
  | "infra_timeout" | "infra_refused" | "infra_uncertain";
/** The verification could not run on the host (setup, not the code): the run needs you, the agent gets no gap. */
export const infraAttempt = (a: Pick<Attempt, "kind" | "category">) => a.kind === "verify" && !!a.category?.startsWith("infra_");
/** What the agent proposes to change about what you set (agent_plan.plan_proposals); you accept or keep each. */
export type Proposals = {
  criteria?: { id: string; change: "reword" | "remove"; text: string; proposed?: string; because: string }[];
  /** `current` is null when Eridani itself picked a verify command because a criterion asks for test evidence. */
  verify?: { current: { target_id: string; verify_id: string } | null; proposed: { target_id: string; verify_id: string };
    options?: string[]; because?: string };
  completion_policy?: { current: string; proposed: string };
};
export type RunDetail = Run & {
  plan: { revision: number; title: string; prd_text: string; steps: { id: string; title: string; status: string; task_id?: string }[];
    criteria: Criterion[]; verify?: { target_id: string; verify_id: string } | null; required_children?: string[];
    proposals?: Proposals | null } | null;
  history: HistoryEntry[];
  /** `role` "user" is the message typed in the terminal (host `prompt` frame); absent or "assistant" is the agent's
   *  final response, whose `prompt` carries the typed message when the host only sent it with the Stop (older hosts).
   *  `clipped`: the text is cut at 4,000 characters for display and the turn route returns it whole; `truncated`: the
   *  host cut the final response (or the daily quota dropped it), so the rest was never stored. */
  turns: { id: string; seq: number; text: string; clipped: boolean; truncated: boolean; generation: number; at: string; prompt: string;
    role?: "assistant" | "user" }[];
  commands: RunCommand[]; questions: RunQuestion[]; attempts?: Attempt[]; interviews?: Interview[];
  /** Verify ids the run's execution target offers (Set verify). */
  verify_options?: string[];
  /** Each verify id's readiness on the host (caps_version 2); empty or absent for an older host. */
  verify_readiness?: Record<string, { ready: boolean; reason?: string }>;
  /** Turns and commands come newest first, one page at a time; this says whether older ones remain (`remaining`
   *  counts them; older servers leave it out). After `mergeDetail` it describes the oldest page loaded. */
  more?: DetailPage;
  /** A settled run's turns and commands are deleted 90 days after it ended; `expires_at` is null while it runs. */
  retention?: { expires_at: string | null; expired: boolean };
  changes?: RunChanges;
};
export type DetailPage = { turns: boolean; commands: boolean; offset: number; limit: number; remaining?: { turns: number; commands: number } };
/** One turn's whole text (`GET /remotea/runs/{id}/turns/{turn_id}`), for a `clipped` turn's "Full response". */
export type FullTurn = { id: string; seq: number; text: string; truncated: boolean };

/** The header's verify control: which command runs after each turn, or that none does and what could. A command
 * the host reports not ready says so (the run would record a failed verification, not a test result). */
export function verifyLine(run: Pick<RunDetail, "plan" | "verify_options" | "verify_readiness">) {
  const current = run.plan?.verify?.verify_id ?? null;
  const options = run.verify_options ?? [];
  const notReady = (id: string) => run.verify_readiness?.[id]?.ready === false;
  const why = (id: string) => `${id} is not ready on the host: ${run.verify_readiness?.[id]?.reason || "it cannot run"}`;
  if (current) return { current, options, canSet: options.length > 1,
    text: `Verified by ${current} after each turn` + (notReady(current) ? ` (${why(current)})` : "") };
  if (!run.plan?.criteria?.length) return { current, options, text: "", canSet: false };
  const unready = options.filter(notReady);
  return { current, options, canSet: options.length > 0,
    text: (options.length ? "No verify command: test results cannot be recorded" : "This target offers no verify command")
      + (unready.length ? `; ${unready.map(why).join("; ")}` : "") };
}
/** `id` is stable across polls and older pages (the drawer keys bubbles and anchors its scroll by it); `turn` is set on
 *  an agent response so its bubble can show Clipped / Truncated by host and fetch the whole text. */
export type TimelineEntry = { id: string; at: string; kind: "turn" | "command" | "question" | "state" | "attempt" | "plan"; title: string; detail: string;
  /** `setup`: a host setup problem (a verification that could not run), distinct from `warn` (failing tests, refusals). */
  tone: "plain" | "good" | "warn" | "setup" | "attention"; who: "agent" | "you" | "eridani";
  turn?: { id: string; clipped: boolean; truncated: boolean };
  /** Separate facts shown as labels on the card (a verification's exit code, sandbox and workspace coverage). */
  facts?: Fact[] };
/** One labelled fact with its tone: `ok` reassures, `warn` weakens a decision, `plain` is context. */
export type Fact = { key: string; label: string; tone: "ok" | "warn" | "plain"; title?: string };

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

// ---------------------------------------------------------------- the review packet beside Approve (E-Q3)

export type PacketCriterion = { id: string; text: string; status: "met" | "not met" | "unknown"; evidence: string };
export type PacketTests = { verifyId: string; status: string; exitCode: number | null; age: string; isolated: boolean | null;
  /** Stale or incomplete evidence, in words: "stale: the code changed during the run", "workspace incomplete: …". */
  markers: string[]; line: string };
export type ReviewPacket = {
  criteria: PacketCriterion[]; met: number; total: number;
  /** The latest verdict was reached against an earlier plan revision (criteria edited since): Judge again. */
  judgedEarlierPlan: boolean;
  tests: PacketTests | null; testsLine: string;
  prs: { url: string; label: string }[]; prState: "open" | "merged" | null; changesLine: string; branch: string; worktree: string;
  questions: string[]; reason: string;
  /** Approval is a human decision; "verified" is completion the judge established on its own (auto_complete). */
  decision: { kind: "approval" | "approved" | "verified"; text: string } | null;
  /** Only an awaiting_approval run is decided here; needs_you and settled runs show the packet read-only. */
  readOnly: boolean;
  text: string;
};

const PR_NUMBER = /\/(?:pull|merge_requests)\/(\d+)/;
/** "PR #12" for a GitHub/GitLab link, else the link's path. */
export function prLabel(url: string) {
  const number = PR_NUMBER.exec(url)?.[1];
  return number ? "PR #" + number : url.replace(/^https?:\/\//, "");
}
/** "just now", "4 min ago", "2 h ago": how old the evidence is, spoken the way control_status says it. */
export function evidenceAge(value: string | null | undefined, now = Date.now()) {
  if (!value) return "";
  const seconds = Math.max(0, (now - new Date(value).getTime()) / 1000);
  if (seconds < 90) return "just now";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  return hours < 36 ? `${hours} h ago` : `${Math.round(hours / 24)} days ago`;
}
const PACKET_TESTS: Record<string, string> = { passed: "tests passed", stale: "tests stale (the code changed during the run)",
  timed_out: "tests timed out", running: "tests running" };
const OPEN_QUESTION = new Set(["open", "submitted", "timed_out"]);

/** What a reviewer needs beside Approve: the frozen criteria with the latest verdict per criterion, the changes (PR
 * links, branch state), the latest verification with stale/incomplete markers and its age, open questions, and the
 * judge's full reason. Null while the run is working (nothing to review) or in a plan review (PlanReview shows it).
 * agent_control.review_packet builds the same sentence for control_status. */
export function reviewPacket(run: RunDetail, now = Date.now()): ReviewPacket | null {
  const parked = run.state === "awaiting_approval" || (run.state === "needs_you" && run.state_reason !== "plan_review");
  if (!parked && !isSettled(run) && run.state !== "stopping") return null;
  const word = { met: "met", not_met: "not met", unclear: "unknown", pending: "unknown" } as const;
  const criteria = criteriaEvidence(run).map(c => ({ id: c.id, text: c.text, status: word[c.status as keyof typeof word] ?? "unknown",
    evidence: c.evidence }));
  const met = criteria.filter(c => c.status === "met").length;
  const judge = (run.attempts ?? []).find(a => a.kind === "judge" && a.status === "done") ?? null;
  const judgedEarlierPlan = !!judge && judge.plan_revision !== run.plan_revision;
  const verify = (run.attempts ?? []).find(a => a.kind === "verify" && a.initiator !== "host") ?? null;
  let tests: PacketTests | null = null, testsLine: string;
  if (!verify) testsLine = run.plan?.verify?.verify_id ? "tests not run yet" : "no verify command, so no test result is recorded";
  else {
    // A verification that could not run (an infra category) says nothing about the code: name the host fix instead.
    const status = infraAttempt(verify) ? `verification could not run (${verify.action || verify.status.replace("_", " ")})`
      : verify.status === "failed" ? "tests failed" + (verify.exit_code != null ? ` (exit ${verify.exit_code})` : "")
      : PACKET_TESTS[verify.status] ?? "tests " + verify.status.replace("_", " ");
    const age = evidenceAge(verify.updated_at || verify.created_at, now);
    const isolated = verify.isolated ?? null;
    const markers = [
      verify.status === "stale" ? "stale: the code changed during the run" : "",
      verify.workspace_complete === false ? "workspace incomplete: " + ((verify.workspace_reasons ?? []).join("; ") || "not fully read") : "",
      isolated === false ? "not sandboxed" + (verify.isolation_reason ? ` (${verify.isolation_reason})` : "") : "",
      verify.turn_seq !== run.turn_seq || verify.plan_revision !== run.plan_revision ? "for an earlier turn or plan" : "",
    ].filter(Boolean);
    const sandbox = isolated === true ? " in a sandbox" : isolated === false ? " outside a sandbox" : "";
    testsLine = `${status}${age ? " " + age : ""}${sandbox}` + (verify.workspace_complete === false ? ` (${markers.find(m => m.startsWith("workspace"))})` : "")
      + (markers.includes("for an earlier turn or plan") ? " (for an earlier turn or plan)" : "");
    tests = { verifyId: verify.verify_id, status: verify.status, exitCode: verify.exit_code, age, isolated, markers, line: testsLine };
  }
  const changes = run.changes;
  const prs = (changes?.pr_urls ?? []).map(url => ({ url, label: prLabel(url) }));
  let changesLine: string;
  if (prs.length) changesLine = `${prs.map(p => p.label).join(", ")} ${changes?.merged ? "merged" : "open"}`;
  else {
    const pending = (["unpushed", "uncommitted"] as const).filter(k => changes?.[k]).map(k => `${changes![k]} ${k}`);
    changesLine = "no PR recorded" + (pending.length ? ` (${pending.join(", ")} on ${run.branch})` : "");
  }
  const questions = [
    ...run.questions.filter(q => OPEN_QUESTION.has(q.status)).flatMap(q => q.items.map(i => i.text)),
    ...(run.interviews ?? []).filter(i => i.status === "open").flatMap(i => i.questions.map(q => q.question)),
  ];
  const reason = [judge?.summary, judge?.gap].filter(Boolean).join("\n\n") || run.reason_detail || "";
  const done = run.state === "completed" || run.state === "stopping";
  const decision = run.state === "awaiting_approval"
    ? { kind: "approval" as const, text: "Approving is your decision: it completes the task as you. It is not completion established by verification." }
    : done && run.state_reason === "approved" ? { kind: "approved" as const, text: "Approved by you: a human decision, not established by verification." }
    : done && run.state_reason === "judge_met" ? { kind: "verified" as const, text: "Completed by verification: the judge found every criterion met; no one approved it." }
    : null;
  const parts = [
    criteria.length ? `${met} of ${criteria.length} criteria met` + (judgedEarlierPlan ? " (judged against an earlier plan)" : "") : "no success criteria",
    testsLine, changesLine,
    questions.length ? `${questions.length} open question${questions.length === 1 ? "" : "s"}` : "no open questions",
  ];
  return { criteria, met, total: criteria.length, judgedEarlierPlan, tests, testsLine, prs,
    prState: prs.length ? (changes?.merged ? "merged" : "open") : null, changesLine, branch: run.branch,
    worktree: changes?.worktree || run.worktree, questions, reason, decision, readOnly: run.state !== "awaiting_approval",
    text: parts.join("; ") };
}

/** Edit criteria's body: the rows as edited (blank ones dropped), the verify choice when it changed, and the
 * revision the editor saw (POST /runs/{id}/criteria refuses a stale one). Null when nothing is left to save. */
export function criteriaBody(rows: { id?: string; text: string; kind?: "text" | "verify" }[], verifyId: string | null,
  current: string | null) {
  const criteria = rows.map(r => ({ ...(r.id ? { id: r.id } : {}), text: r.text.trim(), kind: r.kind ?? "text" })).filter(r => r.text);
  if (!criteria.length) return null;
  return { criteria, ...(verifyId && verifyId !== current ? { verify_id: verifyId } : {}) };
}

const VERIFY_TEXT: Record<string, string> = { running: "running", passed: "passed", failed: "failed", timed_out: "timed out",
  refused: "not run (refused)", uncertain: "result uncertain", stale: "stale: code changed during the run", canceled: "canceled",
  error: "error", superseded: "superseded" };
const VERDICT_TEXT: Record<string, string> = { met: "criteria met", not_met: "not met yet", needs_you: "needs you" };
export function attemptTitle(a: Attempt) {
  if (infraAttempt(a)) return `Verification could not run · ${a.action || VERIFY_TEXT[a.status] || a.status}${a.on_demand ? " (on demand)" : ""}`;
  if (a.kind === "verify") return `Verification ${a.verify_id} · ${VERIFY_TEXT[a.status] ?? a.status}${a.on_demand ? " (on demand)" : ""}`;
  if (a.status === "done" && a.verdict) return "Judge · " + VERDICT_TEXT[a.verdict];
  if (a.status === "failed") return "Judge · no verdict (" + (a.non_verdict ?? "error") + ")";
  return "Judge · " + a.status;
}
export function attemptTone(a: Attempt): TimelineEntry["tone"] {
  if (infraAttempt(a)) return "setup";
  if (a.kind === "verify") return a.status === "passed" ? "good" : a.status === "running" ? "plain" : "warn";
  return a.verdict === "met" ? "good" : a.verdict === "needs_you" || a.status === "failed" ? "attention" : "plain";
}

export function commandTitle(c: RunCommand) {
  const what = c.kind === "input" ? (c.mode === "steer" ? "Steer" : "Message") : c.kind === "spawn" ? "Start" : c.kind === "stop" ? "Stop"
    : c.kind === "answer" ? "Answer" : c.kind === "verify" ? "Verify" : c.kind;
  const state = c.outcome?.state ?? c.status;
  const shown: Record<string, string> = { queued: "queued", sent: "sent", acked: "accepted by remotea", accepted: "accepted by remotea",
    executed: "done", failed: "failed", rejected: "refused by remotea",
    uncertain: "delivery uncertain (remotea could not tell if it ran)", expired: "expired before remotea took it", canceled: "canceled" };
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

/** A settled verification's facts, each on its own: the exit code, whether it ran sandboxed (remotea PR #24) and
 * whether its fingerprint covered the whole workspace. "Tests passed" and "the check was isolated" are different
 * facts; an older host says nothing about the last two, so nothing is shown for them. */
export function attemptFacts(a: Attempt): Fact[] {
  if (a.kind !== "verify" || a.status === "running") return [];
  const facts: Fact[] = [];
  if (a.exit_code != null) facts.push({ key: "exit", label: `exit ${a.exit_code}`, tone: a.exit_code === 0 ? "ok" : "warn" });
  if (a.isolated === true) facts.push({ key: "sandbox", label: "isolated", tone: "ok" });
  else if (a.isolated === false) facts.push({ key: "sandbox", label: `not sandboxed (${a.isolation_reason || "the host did not say why"})`, tone: "warn" });
  if (a.workspace_complete === false) facts.push({ key: "workspace", tone: "warn",
    label: `workspace incomplete (${(a.workspace_reasons ?? []).join("; ") || "the host did not say what it missed"})` });
  else if (a.workspace_complete === true) facts.push({ key: "workspace", label: "complete workspace", tone: "ok" });
  return facts;
}

/** How long since `value`, compactly ("12s", "4m", "3h", "2d"). */
export function elapsed(value: string, now = Date.now()) {
  const seconds = Math.max(0, Math.round((now - new Date(value).getTime()) / 1000));
  if (seconds < 60) return seconds + "s";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return minutes + "m";
  const hours = Math.floor(minutes / 60);
  return hours < 48 ? hours + "h" : Math.floor(hours / 24) + "d";
}

const EVIDENCE_TEXT: Record<VerificationFact["evidence"], string> = {
  fresh: "fresh", stale: "stale: an older turn, or the checkout changed", incomplete: "incomplete workspace", pending: "running",
};
// Past this, a command the host has not taken is worth a warning (the link drains commands at least every 30 s).
const SLOW_COMMAND_MS = 60_000;

/** The facts a decision rests on, each with its own age (E-S6): when the host was last seen (its lease), when the
 * last turn landed, how long the run has been in its state, the latest verification and how far it proves the
 * current turn, its workspace fingerprint and sandbox, and controls sent but not yet taken by the host or whose
 * delivery is uncertain. `host` is the full host row when the page has it (else the run's own `host`). Facts the
 * server did not send are left out. Pure: the drawer re-renders it on a clock. */
export function freshnessFacts(run: Run, host?: Pick<Host, "connected" | "last_seen_at"> | null, now = Date.now()): Fact[] {
  const facts: Fact[] = [];
  const h = host ?? run.host;
  if (h) {
    const seen = h.last_seen_at;
    if (h.connected) facts.push({ key: "host", label: seen ? `host seen ${ago(seen, now)}` : "host connected",
      tone: seen && now - new Date(seen).getTime() > 90_000 ? "warn" : "ok" });
    else facts.push({ key: "host", label: seen ? `host silent for ${elapsed(seen, now)}` : "host never seen", tone: "warn" });
  }
  if (run.last_turn_at) facts.push({ key: "turn", label: `last turn ${ago(run.last_turn_at, now)}`, tone: "plain" });
  else if (run.last_turn_at === null && run.started_at) facts.push({ key: "turn", label: "no turn yet", tone: "plain" });
  if (run.state_since) facts.push({ key: "state", label: `${runLabel(run.state).toLowerCase()} for ${elapsed(run.state_since, now)}`, tone: "plain" });
  const v = run.verification;
  if (v) {
    if (v.evidence === "pending") {
      const late = !!v.deadline_at && new Date(v.deadline_at).getTime() < now;
      facts.push({ key: "verify", label: `verify ${v.verify_id} running for ${elapsed(v.at, now)}${late ? " (past its deadline)" : ""}`, tone: late ? "warn" : "plain" });
    } else {
      facts.push({ key: "verify", label: `verify ${v.verify_id} ${VERIFY_TEXT[v.status] ?? v.status} ${ago(v.at, now)} · ${EVIDENCE_TEXT[v.evidence]}`,
        tone: v.evidence === "fresh" && v.status === "passed" ? "ok" : "warn" });
      if (v.workspace_complete != null) facts.push({ key: "workspace", tone: v.workspace_complete ? "plain" : "warn",
        label: `workspace fingerprint ${ago(v.at, now)}` + (v.workspace_complete ? "" : ` · incomplete (${(v.workspace_reasons ?? []).join("; ") || "the host did not say what it missed"})`) });
      if (v.isolated === false) facts.push({ key: "sandbox", label: `not sandboxed (${v.isolation_reason || "the host did not say why"})`, tone: "warn" });
      else if (v.isolated === true) facts.push({ key: "sandbox", label: "sandboxed", tone: "ok" });
    }
  }
  for (const [n, c] of (run.pending_controls ?? []).entries()) {
    const what = c.kind === "input" ? (c.mode === "steer" ? "steer" : "message") : c.kind;
    facts.push(c.status === "uncertain"
      ? { key: "control:" + n, label: `${what} ${ago(c.at, now)}: delivery uncertain`, tone: "warn" }
      : { key: "control:" + n, label: `${what} sent ${ago(c.at, now)}, not taken by remotea yet`,
        tone: now - new Date(c.at).getTime() > SLOW_COMMAND_MS ? "warn" : "plain" });
  }
  return facts;
}
/** The few freshness facts that fit a run row (RunsPanel RunRow), as compact chips: the host's lease ("host seen 2m
 * ago", "host silent for 5m"), then only what changes how far the row can be trusted: a verification that is stale,
 * ran on an incomplete workspace or is past its deadline, and controls whose delivery is uncertain. Each chip's title
 * carries the full fact. Settled runs get none. */
export function rowFreshness(run: Run, host?: Pick<Host, "connected" | "last_seen_at"> | null, now = Date.now()): Fact[] {
  if (isSettled(run)) return [];
  const facts = freshnessFacts(run, host, now);
  const out = facts.filter(f => f.key === "host");
  const v = run.verification, verify = facts.find(f => f.key === "verify");
  const short = v?.evidence === "stale" ? "stale" : v?.evidence === "incomplete" ? "workspace incomplete"
    : v?.evidence === "pending" && verify?.tone === "warn" ? "verify late" : "";
  if (verify && short) out.push({ key: "verify", label: short, tone: "warn", title: verify.label });
  const uncertain = facts.filter(f => f.key.startsWith("control:") && f.label.endsWith("delivery uncertain"));
  if (uncertain.length) out.push({ key: "control", label: "delivery uncertain", tone: "warn", title: uncertain.map(f => f.label).join("; ") });
  return out;
}

/** The facts an approval rests on: the verification, its workspace and sandbox, the host, and anything uncertain. */
export const APPROVAL_FACTS = (f: Fact) => ["verify", "workspace", "sandbox", "host"].includes(f.key) || f.key.startsWith("control:");

/** Why Eridani sent a text it wrote itself: the `kind` on a "sent" history entry (docs/AGENT_CONTROL.md, "Messages
 * Eridani sends to the agent"). Kinds not listed here (an answer to the agent's question) keep the plain title. */
export const SENT_REASON: Record<string, string> = {
  execute: "plan frozen, start work", gap: "judge gap", replan: "re-plan", verify_set: "verify command set",
  interview_answers: "your interview answers", merged: "pull request merged", integrated: "sub-task pull requests merged",
};

/** Title of a timeline entry for a message typed in the terminal (not sent from this page). */
export const TYPED_TITLE = "You, in the terminal";

// A clipped copy counts as the same text only once this many characters agree (short texts must match whole).
const PREFIX_MATCH = 200;
const normal = (text: string) => text.replace(/[…�]+\s*$/u, "").replace(/\s+/gu, " ").trim();
/** Whether two copies of a text sent to the agent are the same message: equal after whitespace normalisation, or one
 * a long prefix of the other. Each side clips on its own (the host's prompt at 4,000 bytes, the "sent" entry at
 * 2,000 characters, a turn's `prompt` at 500), so a long Eridani text never matched exactly and showed twice. */
export function sameText(a: string, b: string) {
  const x = normal(a), y = normal(b);
  if (!x || !y) return false;
  if (x === y) return true;
  const [short, long] = x.length <= y.length ? [x, y] : [y, x];
  return short.length >= PREFIX_MATCH && long.startsWith(short);
}

/** One time-ordered timeline: final responses per turn, commands with their outcomes, questions and state changes. */
export function timeline(run: RunDetail): TimelineEntry[] {
  const entries: TimelineEntry[] = [];
  // A text that left for the agent appears once: the "sent" history entry says who sent it, so the matching
  // input command (same text) is skipped unless it failed, where its outcome is the news.
  const sentTexts = run.history.filter(h => h.event === "sent" && h.text).map(h => h.text as string);
  const sent = { has: (text: string) => sentTexts.some(s => sameText(s, text)) };
  for (const t of run.turns) {
    // The message typed in the terminal: its own row when the host sent it as the turn started, otherwise the
    // assistant turn's `prompt` (older hosts, so only once the turn ended). A text Eridani itself sent is already
    // a "Sent by" entry and is not repeated, even when either copy was clipped (sameText).
    if (t.role === "user") {
      if (!sent.has(t.text)) entries.push({ id: "turn:" + t.id, at: t.at, kind: "command", title: TYPED_TITLE, detail: t.text, tone: "plain", who: "you" });
      continue;
    }
    if (t.prompt && !sent.has(t.prompt)) entries.push({ id: "prompt:" + t.id, at: t.at, kind: "command", title: TYPED_TITLE, detail: t.prompt, tone: "plain", who: "you" });
    entries.push({ id: "turn:" + t.id, at: t.at, kind: "turn", title: `Turn ${t.seq}`, detail: t.text, tone: "plain", who: "agent",
      turn: { id: t.id, clipped: t.clipped, truncated: t.truncated } });
  }
  for (const c of run.commands) {
    const state = c.outcome?.state ?? c.status;
    if (c.kind === "input" && c.text && sent.has(c.text) && !["failed", "rejected", "expired", "uncertain"].includes(state)) continue;
    entries.push({ id: "command:" + c.id, at: c.created_at, kind: "command", title: commandTitle(c), detail: c.kind === "input" && sent.has(c.text ?? "") ? c.outcome?.detail ?? "" : c.text ?? c.outcome?.detail ?? "",
      tone: commandTone(c), who: "eridani" });
  }
  for (const q of run.questions) entries.push({ id: "question:" + q.id, at: q.created_at, kind: "question", title: "Question · " + QUESTION_TEXT[q.status],
    detail: q.items.map(i => i.text).join(" / "), tone: q.status === "open" || q.status === "timed_out" ? "attention" : q.status === "accepted" ? "good" : "plain", who: "agent" });
  for (const a of run.attempts ?? []) entries.push({ id: "attempt:" + a.id, at: a.updated_at, kind: "attempt", title: attemptTitle(a),
    detail: a.kind === "judge" ? (a.verdict === "met" ? a.summary ?? "" : a.gap ?? "") : (a.status === "failed" ? a.tail ?? "" : ""), tone: attemptTone(a), who: "eridani",
    ...(attemptFacts(a).length ? { facts: attemptFacts(a) } : {}) });
  for (const i of run.interviews ?? []) entries.push({ id: "interview:" + i.id, at: i.created_at, kind: "question", title: "Interview · " + (i.status === "open" ? "waiting for you" : i.status),
    detail: i.questions.map(q => q.question).join(" / "), tone: i.status === "open" ? "attention" : "plain", who: "agent" });
  // History is never paged (and only grows), so its index is a stable id.
  for (const [n, h] of run.history.entries()) {
    const id = "history:" + n;
    if (h.event === "sent" && h.text) {
      // Every text sent to the agent, with who sent it (stage F): you on this page, Eri after your confirmation,
      // or Eridani itself (interview answers as the planning prompt, the judge's gap).
      // A text Eridani wrote carries the `kind` that triggered it (run_coordinator.say), even when your action caused it.
      const reason = h.kind ? SENT_REASON[h.kind] : undefined;
      if (reason) {
        entries.push({ id, at: h.at, kind: "command", title: "Sent by Eridani · " + reason, detail: h.text, tone: "plain", who: "eridani" });
        continue;
      }
      const confirmed = !!h.actor?.reason?.startsWith("confirmed:");
      const human = h.actor?.kind === "human";
      const who = confirmed ? (human ? "Eri, confirmed by you" : "Eri, after your yes") : human ? "you" : ["run_judge", "eridani"].includes(h.actor?.kind ?? "") ? "Eridani" : h.actor?.kind ?? "";
      entries.push({ id, at: h.at, kind: "command", title: "Sent by " + who, detail: h.text, tone: "plain", who: human && !confirmed ? "you" : "eridani" });
      continue;
    }
    if (h.event === "plan_submitted") {
      entries.push({ id, at: h.at, kind: "plan", title: `Plan submitted · revision ${h.plan_revision ?? ""}`.trim(),
        detail: h.proposals ? h.proposals + " — waiting for your review." : "", tone: h.proposals ? "attention" : "plain", who: "agent" });
      continue;
    }
    if (h.event === "plan_reviewed") {
      const accepted = (h.accepted ?? []).join(", "), kept = (h.kept ?? []).join(", ");
      entries.push({ id, at: h.at, kind: "plan", title: `Plan reviewed by you · revision ${h.plan_revision ?? ""}`.trim(),
        detail: [accepted ? "accepted " + accepted : "", kept ? "kept " + kept : ""].filter(Boolean).join(" · "), tone: "good", who: "you" });
      continue;
    }
    if (!h.to) continue;
    entries.push({ id, at: h.at, kind: "state", title: runLabel(h.to as RunState), detail: [h.actor?.kind === "human" ? "by you" : "", h.detail ?? ""].filter(Boolean).join(" · "),
      tone: h.to === "needs_you" ? "attention" : h.to === "completed" ? "good" : ["failed", "stopped"].includes(h.to) ? "warn" : "plain", who: "eridani" });
  }
  return entries.sort((a, b) => new Date(a.at).getTime() - new Date(b.at).getTime());
}

// ---------------------------------------------------------------- older pages of the timeline (E-N5)

const time = (value: string) => new Date(value).getTime();
function union<T extends { id: string }>(newer: T[], older: T[], order: (a: T, b: T) => number) {
  const byId = new Map<string, T>();
  for (const item of older) byId.set(item.id, item);
  for (const item of newer) byId.set(item.id, item);  // A row in both pages: the newer copy wins (its outcome may have moved on).
  return [...byId.values()].sort(order);
}

/** The run with an older page of turns and commands folded in: one row per id, ordered by time, and everything else
 * (state, plan, history, questions) from `current`. `more` becomes the oldest page's, so the next Load earlier asks
 * for the page after it. A page for another run changes nothing. Offsets count from the newest row, so turns that
 * arrived since the first load make the older page overlap what is shown; the overlap is deduplicated, never lost. */
export function mergeDetail(current: RunDetail, older: RunDetail): RunDetail {
  if (older.id !== current.id) return current;
  const turns = union(current.turns, older.turns, (a, b) => time(a.at) - time(b.at) || a.seq - b.seq);
  const commands = union(current.commands, older.commands, (a, b) => time(a.created_at) - time(b.created_at));
  const more = !older.more ? current.more : !current.more || older.more.offset >= current.more.offset ? older.more : current.more;
  return { ...current, turns, commands, more };
}

/** The offset of the next older page, or null when nothing older remains. */
export function nextOffset(run: Pick<RunDetail, "more">): number | null {
  const more = run.more;
  return more && (more.turns || more.commands) ? more.offset + more.limit : null;
}

/** The Paged out line beside Load earlier: how many older rows remain when the server counted them. */
export function earlierLine(more: DetailPage | undefined): string | null {
  if (!more || !(more.turns || more.commands)) return null;
  const left = more.remaining;
  if (!left || !(left.turns || left.commands)) return "Older turns and commands are not shown yet";
  const part = (n: number, one: string) => `${n} older ${one}${n === 1 ? "" : "s"}`;
  return [left.turns ? part(left.turns, "turn") : "", left.commands ? part(left.commands, "command") : ""].filter(Boolean).join(" and ")
    + " not shown yet";
}

/** Where the reader is before a page is merged: the first entry whose bottom is below the top of the log, and how far
 * its top sits from that edge (negative when it is partly scrolled off). Rows are in timeline order. */
export function scrollAnchor(rows: { id: string; top: number; bottom: number }[], viewTop: number): { id: string; offset: number } | null {
  const row = rows.find(r => r.bottom > viewTop);
  return row ? { id: row.id, offset: row.top - viewTop } : null;
}

/** The scrollTop that puts the anchored entry back where it was, given where its top sits now relative to the log's top. */
export function anchoredScrollTop(scrollTop: number, anchor: { offset: number }, nowOffset: number) {
  return scrollTop + nowOffset - anchor.offset;
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

/** What a caps_version 2 host says about one target (remotea PRs #18/#20); each is null/absent when unsaid.
 * `enabled`: harnesses the target's launch permits (installed or not); `launch`: each one's launcher probe;
 * `verify_status`: each verify id's readiness; `mcp`: whether spawned agents get Eridani's MCP tools. */
export type TargetLaunch = { ready: boolean; version: string | null; error?: string };
export type VerifyStatus = { id: string; ready: boolean; reason?: string; timeout?: number;
  isolated?: boolean; isolation?: "bwrap" | "none"; isolation_reason?: string };
export type TargetMcp = { configured: boolean; reason?: string };
type TargetFacts = { enabled?: string[] | null; launch?: Partial<Record<string, TargetLaunch>> | null;
  verify_status?: VerifyStatus[] | null; mcp?: TargetMcp | null };
/** A saved target. `harnesses`/`verify` (and the TargetFacts) are what its host advertised for it, present only when known. */
export type SavedTarget = { id: string; host_id: string; host_target: string; name: string; repo: string; worktree_root: string; record_id: string | null;
  harnesses?: string[]; verify?: string[] } & TargetFacts;
/** A target a host listed in hello. `harnesses` is what the target can start; null when the host did not say (older
 * remotea: its host-wide flags decide). `verify` is its verify ids (absent from older Eridani). */
export type AdvertisedTarget = { host_id: string; host_name: string; host_target: string; repo: string; worktree_root: string;
  harnesses?: string[] | null; verify?: string[] } & TargetFacts;
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

// ---------------------------------------------------------------- worktree maintenance (S-Q4)

/** The live session writing a worktree (remotea S-S4). */
export type WorktreeWriter = { surf: string; run_id: string };
/** One managed worktree from GET /remotea/worktrees: the host's worktrees.report row (candidates included), the run
 * bound to it, and `cleanup`, remotea's own commands for a row that loses nothing (null otherwise). `writer` is
 * absent from an older host's rows (it cannot rule out a live writer). */
export type WorktreeRow = {
  host_id: string; host_name: string; repo: string | null; target: string | null; path: string | null; branch: string | null;
  base: string | null; base_ref?: string | null; head: string | null; active: boolean | null; ahead: number | null;
  behind: number | null; unpushed: number | null; upstream: string | null; merged: boolean | null; merged_reason?: string | null;
  uncommitted: number | null; untracked: number | null; ignored: number | null; candidate: boolean | null; reasons: string[] | null;
  writer?: WorktreeWriter | null;
  run: { id: string; state: string; settled: boolean; task_id: string | null; task_title: string | null } | null;
  cleanup: string[] | null;
};
export type WorktreeHost = { id: string; name: string; connected: boolean; report_at: string | null; rescan: boolean; count: number };
export type WorktreeClass = "active" | "dirty" | "unpushed" | "unmerged" | "merged" | "reclaimable" | "unknown";

export const WORKTREE_CLASSES: { id: WorktreeClass; label: string; hint: string }[] = [
  { id: "active", label: "Active", hint: "A live session writes here, or its run has not settled." },
  { id: "dirty", label: "Dirty", hint: "Uncommitted changes or untracked files." },
  { id: "unpushed", label: "Unpushed", hint: "Commits that are not on origin." },
  { id: "unmerged", label: "Unmerged", hint: "Pushed but not merged into its base yet (an open pull request)." },
  { id: "merged", label: "Merged", hint: "Merged into its base, but something else keeps it (ignored files, a lock, detached HEAD)." },
  { id: "reclaimable", label: "Reclaimable", hint: "Merged or empty, clean, no writer: removing it loses nothing." },
  { id: "unknown", label: "Unknown", hint: "The host did not report enough to judge (older remotea), or a check failed." },
];

const gone = (row: Pick<WorktreeRow, "reasons">) => (row.reasons ?? []).some(r => r.startsWith("folder missing"));

/** Which bucket a worktree falls in, worst first: active (a live writer or an unsettled run), dirty, unpushed,
 * unmerged, reclaimable (the host says nothing is lost and names no writer, or the folder is already gone), merged
 * (merged, but kept by something else), unknown (an older host's row without the fields, or anything else). Only
 * `reclaimable` is ever offered for removal. */
export function worktreeClass(row: WorktreeRow): WorktreeClass {
  if (typeof row.uncommitted !== "number" || typeof row.candidate !== "boolean" || !Array.isArray(row.reasons)) return "unknown";
  if (row.active || row.writer || (row.run && !row.run.settled)) return "active";
  const knowsWriter = "writer" in row && row.writer === null;
  if (gone(row)) return knowsWriter ? "reclaimable" : "unknown";
  if (row.uncommitted > 0 || (row.untracked ?? 0) > 0) return "dirty";
  if ((row.unpushed ?? 0) > 0 || (row.merged === false && !row.upstream)) return "unpushed";
  if (row.merged === false) return "unmerged";
  if (row.candidate) return knowsWriter ? "reclaimable" : "unknown";
  if (row.merged === true) return "merged";
  return "unknown";
}

/** What removing a worktree would lose (or why it must not be removed now), one plain line each. */
export function worktreeLoss(row: WorktreeRow): string[] {
  const out: string[] = [];
  if (row.writer) out.push(`a live session writes here (${row.writer.surf})`);
  else if (row.active) out.push("a live session works here");
  if (row.run && !row.run.settled) out.push(`its run is still ${row.run.state.replace(/_/g, " ")}`);
  if (row.uncommitted) out.push(`${row.uncommitted} uncommitted change${row.uncommitted === 1 ? "" : "s"}`);
  if (row.untracked) out.push(`${row.untracked} untracked file${row.untracked === 1 ? "" : "s"}`);
  if (row.ignored) out.push(`${row.ignored} ignored file${row.ignored === 1 ? "" : "s"} (e.g. .env), deleted by a removal`);
  if (row.unpushed) out.push(`${row.unpushed} commit${row.unpushed === 1 ? "" : "s"} not on origin`);
  else if (row.merged === false && !row.upstream) out.push(`${row.ahead ? row.ahead + " commit(s)" : "commits"} that exist only here`);
  else if (row.merged === false) out.push(`not merged into ${row.base_ref || row.base || "its base"}`);
  if (!out.length && worktreeClass(row) === "unknown") out.push((row.reasons ?? []).join("; ") || "the host did not report enough to tell (update remotea)");
  if (!out.length && worktreeClass(row) !== "reclaimable") out.push((row.reasons ?? []).join("; ") || "kept by the host");
  return out;
}

export function worktreeCounts(rows: WorktreeRow[]) {
  const counts = Object.fromEntries(WORKTREE_CLASSES.map(c => [c.id, 0])) as Record<WorktreeClass, number>;
  for (const row of rows) counts[worktreeClass(row)] += 1;
  return counts;
}

export const worktreeKey = (row: Pick<WorktreeRow, "host_id" | "path" | "branch">) => `${row.host_id}:${row.path ?? row.branch ?? ""}`;

/** The review step: the reclaimable rows (each with its commands) and every other row refused with what it would
 * lose. `commands` holds only the selected reclaimable rows' commands, grouped per host, for the owner to run there:
 * Eridani has no removal command (remotea advertises none), and a row without commands is never included. */
export function removalPlan(rows: WorktreeRow[], selected: ReadonlySet<string>) {
  const reclaimable = rows.filter(r => worktreeClass(r) === "reclaimable" && r.cleanup?.length);
  const refused = rows.filter(r => !reclaimable.includes(r)).map(row => ({ row, lost: worktreeLoss(row) }));
  const chosen = reclaimable.filter(r => selected.has(worktreeKey(r)));
  const hosts = [...new Set(chosen.map(r => r.host_id))];
  const commands = hosts.flatMap(h => {
    const mine = chosen.filter(r => r.host_id === h);
    return [`# On ${mine[0].host_name} (in WSL): ${mine.length} worktree${mine.length === 1 ? "" : "s"}`,
      ...mine.flatMap(r => [`# ${r.target ?? ""} ${r.branch ?? r.path ?? ""}`.trimEnd(), ...(r.cleanup ?? [])])];
  });
  return { reclaimable, refused, chosen, commands };
}

// ---------------------------------------------------------------- run panel, row buttons, Run modal (run from anywhere)

/** States where the run waits on the human; the panel rail, row chips and the Runs badge highlight them. */
export const ATTENTION_STATES: RunState[] = ["needs_you", "waiting_answer", "awaiting_approval", "awaiting_integration", "interviewing"];

/** A proposal from Eri (control_start / control_message) waiting for the owner; a start also carries what still
 * stands in its way (no target chosen, server offline). */
export type PendingControl = { id: string; tool: string; summary: string; run_id: string | null; expires_at: string;
  task_id?: string | null; needs_target?: boolean; need?: NeedsTarget | null; problems?: { code: string; message: string }[] };

export type PanelState = { run: string | null; collapsed: boolean };
export const PANEL_STORAGE_KEY = "eri-run-panel";

/** The panel on load: the run from `?run=` (push deep links, reloads) and whether it was left collapsed. */
export function restorePanel(search: string, stored: string | null): PanelState {
  return { run: linkedRun(search), collapsed: stored === "rail" };
}
export function storedPanel(panel: PanelState): string | null {
  return panel.collapsed ? "rail" : null;
}

export function runsByTask(runs: Run[]): Map<string, Run> {
  const map = new Map<string, Run>();
  for (const run of orderRuns(runs)) if (run.task && !map.has(run.task.id)) map.set(run.task.id, run);
  return map;
}

/** The collapsed rail's dot: what the open run needs right now. */
export function railDot(run: Run | undefined): { tone: "attention" | "working" | "settled" | "unknown"; label: string } {
  if (!run) return { tone: "unknown", label: "Run" };
  if (ATTENTION_STATES.includes(run.state)) return { tone: "attention", label: runLabel(run.state) };
  if (isSettled(run)) return { tone: "settled", label: runLabel(run.state) };
  return { tone: "working", label: runLabel(run.state) };
}

/** A task row's Run control: start a run, or open the one that exists (labelled by what it needs). */
export function runChip(run: Run | undefined, hosts: Host[]):
  { kind: "start"; label: "Run"; disabled: boolean; reason: string } |
  { kind: "open"; label: "Running" | "Needs you" | "Approve" | "Starting"; tone: "attention" | "working"; run: string } {
  if (run && !isSettled(run)) {
    if (run.state === "awaiting_approval") return { kind: "open", label: "Approve", tone: "attention", run: run.id };
    if (ATTENTION_STATES.includes(run.state)) return { kind: "open", label: "Needs you", tone: "attention", run: run.id };
    if (["queued", "spawning"].includes(run.state)) return { kind: "open", label: "Starting", tone: "working", run: run.id };
    return { kind: "open", label: "Running", tone: "working", run: run.id };
  }
  const button = runButton(true, hosts, null);
  return { kind: "start", label: "Run", disabled: button.disabled, reason: button.reason };
}

// ---------------------------------------------------------------- attention-first grouping (E-Q5)

export type AttentionKey = "confirm" | "answer" | "review_plan" | "repair" | "inspect" | "approve" | "unmerged" | "working" | "done";
/** Groups in the order a human acts on them; `action` groups count toward the Runs badge. */
export const ATTENTION_GROUPS: { key: AttentionKey; verb: string; action: boolean }[] = [
  { key: "confirm", verb: "Confirm Eri's proposal", action: true },
  { key: "answer", verb: "Answer a question", action: true },
  { key: "review_plan", verb: "Review the plan", action: true },
  { key: "repair", verb: "Repair the host or setup", action: true },
  { key: "inspect", verb: "Inspect the failed check", action: true },
  { key: "approve", verb: "Approve", action: true },
  { key: "unmerged", verb: "Resolve unmerged work", action: true },
  { key: "working", verb: "Working", action: false },
  { key: "done", verb: "Done", action: false },
];
// needs_you reasons the host or its setup must fix (verify_infra: a verification that could not run).
const REPAIR_REASONS = new Set(["host_offline", "surface_lost", "exited", "spawn_expired", "delivery_uncertain", "verify_infra"]);
const ANSWER_REASONS = new Set(["question_timed_out"]);
const UNMERGED_REASONS = new Set(["awaiting_merge"]);

function hostOffline(run: Run, hosts: Host[]) {
  if (!run.host) return false;
  const known = hosts.find(h => h.id === run.host!.id);
  return known ? !known.connected : !run.host.connected;
}
function unmergedFor(run: Run, unmerged: UnmergedWork[]) {
  return !!run.branch && unmerged.some(w => w.branch === run.branch && w.host_id === run.host?.id && !w.active);
}

/** The next human action a run waits for. A run whose host is offline (or whose setup broke) needs a repair before
 * anything the agent could do; settled runs with unmerged work on their branch still need resolving. */
export function attentionKey(run: Run, hosts: Host[] = [], unmerged: UnmergedWork[] = []): AttentionKey {
  if (isSettled(run)) return unmergedFor(run, unmerged) ? "unmerged" : "done";
  if (run.state === "stopping") return "working";
  if (run.state === "waiting_answer" || run.state === "interviewing") return "answer";
  if (run.state === "awaiting_approval") return "approve";
  if (run.state === "awaiting_integration") return "unmerged";
  if (run.state === "needs_you") {
    if (run.state_reason === "plan_review") return "review_plan";
    if (REPAIR_REASONS.has(run.state_reason) || hostOffline(run, hosts)) return "repair";
    if (ANSWER_REASONS.has(run.state_reason)) return "answer";
    if (UNMERGED_REASONS.has(run.state_reason)) return "unmerged";
    return "inspect";
  }
  return hostOffline(run, hosts) ? "repair" : "working";
}

export type AttentionGroup = { key: AttentionKey; verb: string; action: boolean; count: number; runs: Run[]; pending: PendingControl[] };
/** Runs (and Eri's proposals) grouped by the next human action, in the order to act on them; empty groups left out.
 * Within a group the run that changed most recently comes first. */
export function attentionGroups(runs: Run[], pending: PendingControl[], hosts: Host[] = [], unmerged: UnmergedWork[] = []): AttentionGroup[] {
  const sorted = [...runs].sort((a, b) => new Date(b.state_since).getTime() - new Date(a.state_since).getTime());
  return ATTENTION_GROUPS.map(g => {
    const inGroup = sorted.filter(r => attentionKey(r, hosts, unmerged) === g.key);
    const waiting = g.key === "confirm" ? pending : [];
    return { ...g, runs: inGroup, pending: waiting, count: inGroup.length + waiting.length };
  }).filter(g => g.count > 0);
}

/** The topbar Runs button: the badge counts what needs an action from you (proposals to confirm, questions, plan
 * reviews, repairs, failed checks, approvals, unmerged work), not runs that are just working. */
export function runsBadge(runs: Run[], pending: PendingControl[], hosts: Host[] = []) {
  const active = runs.filter(r => !isSettled(r));
  const groups = attentionGroups(active, pending, hosts);
  const count = groups.filter(g => g.action).reduce((n, g) => n + g.count, 0);
  const working = active.length - groups.filter(g => g.action).reduce((n, g) => n + g.runs.length, 0);
  const parts = [count ? `${count} need${count === 1 ? "s" : ""} you` : "", working ? `${working} working` : ""].filter(Boolean);
  return { count, attention: count > 0, aria: "Runs" + (parts.length ? ", " + parts.join(", ") : "") };
}

// The Run modal's choices. Mirrors agent_control.HARNESSES (harness → default model); a drift guard reads both.
export const HARNESSES = [
  { id: "claude", label: "Claude Code", models: [{ id: "opus", label: "Opus" }, { id: "sonnet", label: "Sonnet" }] },
  { id: "codex", label: "Codex", models: [{ id: "gpt-5.6-sol", label: "GPT-5.6 Sol" }] },
] as const;
export type HarnessId = typeof HARNESSES[number]["id"];
export type RunDraft = { harness: HarnessId; model: string; interview: "auto" | "skip"; target: string | null;
  /** Start in freestyle: no interview, no judge; the only way to start where agents have no Eridani tools. */
  freestyle?: boolean };
export const DEFAULT_DRAFT: RunDraft = { harness: "claude", model: "opus", interview: "auto", target: null };

/** Can any connected host start this harness? (`caps.harnesses.<id>.spawn` from hello.) */
export function harnessAvailable(hosts: Host[], id: HarnessId) {
  return hosts.some(h => h.connected && !!(h.harnesses?.[id] as { spawn?: boolean } | undefined)?.spawn);
}

/** The body for POST /remotea/runs. "Auto" leaves `interview` to the backend (interview unless the task is planned). */
export function startBody(taskId: string, draft: RunDraft) {
  const harness = HARNESSES.find(h => h.id === draft.harness) ?? HARNESSES[0];
  const model = draft.model && draft.model !== harness.models[0].id ? draft.model : undefined;
  return { task_id: taskId, harness: draft.harness, ...(model ? { model } : {}),
    ...(draft.target ? { execution_target_id: draft.target } : {}), ...(draft.interview === "skip" ? { interview: false } : {}),
    ...(draft.freestyle ? { freestyle: true } : {}) };
}

/** GET /remotea/runs/preflight: what a start would do, checked without acting. `harnesses` is what the task's
 * target can start (the union over the offered targets while none is chosen, with `harness_note`);
 * `harness_reasons` says why one cannot. */
export type Preflight = {
  target: { id: string; name: string; repo: string } | null; needs_target: boolean; need: NeedsTarget | null;
  problems: PreflightProblem[]; active_run: { id: string; state: RunState } | null;
  interview: boolean; harnesses: Record<string, boolean>;
  harness_reasons?: Record<string, string>; harness_note?: string | null;
  /** The resolved target's readiness (caps_version 2 / remotea PR #20; null when the host did not say):
   * `verify_ready` for the verify ids a run would use, `verify_reasons` why not, `mcp` the agents' Eridani tools. */
  verify_status?: VerifyStatus[] | null; verify_ready?: boolean | null; verify_reasons?: Record<string, string>;
  mcp?: TargetMcp | null;
};
/** What stands in a start's way. `blocking` false is a warning (VERIFY_NOT_READY: the run starts and records the
 * failure); `freestyle_ok` lifts a block for a freestyle start (MCP_UNCONFIGURED). Older servers send neither:
 * every problem then blocks, as start_run refuses it. */
export type PreflightProblem = { code: string; message: string; blocking?: boolean; freestyle_ok?: boolean };

/** The problem that keeps the dialog's Start disabled for this draft, if any; whether a freestyle start lifts it
 * (`freestyle`: offer the toggle). */
export function startBlock(p: Preflight | null | undefined, draft: Pick<RunDraft, "freestyle">):
  { problem: PreflightProblem | null; freestyle: boolean } {
  const blocking = (p?.active_run ? [] : p?.problems ?? []).filter(x => x.blocking !== false);
  const problem = blocking.find(x => !(x.freestyle_ok && draft.freestyle)) ?? null;
  return { problem, freestyle: blocking.length > 0 && blocking.every(x => x.freestyle_ok) };
}

/** The preflight URL; `target` is the execution target the picker just saved, so the re-read resolves to it. */
export function preflightUrl(taskId: string, harness: string, target?: string | null) {
  return `/remotea/runs/preflight?task_id=${encodeURIComponent(taskId)}&harness=${encodeURIComponent(harness)}`
    + (target ? `&execution_target_id=${encodeURIComponent(target)}` : "");
}

/** After the dialog's picker saves a target: remember it in the draft and close the picker, so the preflight is
 * read again for that target (its harnesses, verify and MCP notes) instead of starting at once. */
export function afterTargetSaved(draft: RunDraft, target: string): { draft: RunDraft; need: null } {
  return { draft: { ...draft, target }, need: null };
}

/** The dialog's notes under the agent choice: the target's verify command not ready, agents without Eridani tools. */
export function preflightNotes(p: Preflight | null | undefined): string[] {
  if (!p || p.active_run) return [];
  const notes: string[] = [];
  if (p.verify_ready === false) {
    const reasons = Object.entries(p.verify_reasons ?? {}).map(([id, why]) => `${id}: ${why}`);
    notes.push("This target's verify command is not ready" + (reasons.length ? ` (${reasons.join("; ")})` : "") + ".");
  }
  if (p.mcp && !p.mcp.configured) notes.push(`Agents on this target have no Eridani tools: ${p.mcp.reason || "the host did not say why"}.`);
  return notes;
}

/** Whether the Run dialog may pick a harness, and why not. The preflight decides for the task's target; only
 * when there is no preflight (it failed, or is still loading) does any connected host count (`harnessAvailable`). */
export function harnessChoice(pre: Preflight | null | undefined, hosts: Host[], id: HarnessId): { ok: boolean; reason?: string } {
  const label = HARNESSES.find(h => h.id === id)?.label ?? id;
  const said = pre?.harnesses?.[id];
  if (said === undefined) {
    return harnessAvailable(hosts, id) ? { ok: true } : { ok: false, reason: `Your remotea server cannot start ${label} yet.` };
  }
  if (said) return { ok: true };
  return { ok: false, reason: pre?.harness_reasons?.[id] || `This task's target cannot start ${label}.` };
}

export function preflightLine(p: Preflight | null): string {
  if (!p) return "Where it runs is chosen when the run starts.";
  if (p.active_run) return "An agent run is already working on this task.";
  if (p.needs_target) return "Choose where this task's agent should run.";
  if (p.target) return `Runs in ${p.target.name}${p.target.repo ? " · " + p.target.repo : ""}.`;
  return p.problems[0]?.message ?? "No remotea server can take this run yet.";
}

/** The run composer's keys: Enter queues, Ctrl/Cmd+Enter steers, Shift+Enter is a new line. */
export function composerAction(e: { key: string; shiftKey: boolean; ctrlKey: boolean; metaKey: boolean; altKey?: boolean }): "queue" | "steer" | null {
  if (e.key !== "Enter" || e.shiftKey || e.altKey) return null;
  return e.ctrlKey || e.metaKey ? "steer" : "queue";
}

// ---------------------------------------------------------------- the run drawer's requests and controls

/** A latest-only request guard: `begin()` aborts the previous request, and only the newest one may install its
 * result (`current()`). `cancel()` retires whatever is in flight (unmount, or the panel switched runs). */
export type LatestOnly = { begin(): { signal: AbortSignal; current(): boolean }; cancel(): void };
export function latestOnly(): LatestOnly {
  let seq = 0, controller: AbortController | null = null;
  return {
    begin() {
      controller?.abort();
      const mine = ++seq, own = new AbortController();
      controller = own;
      return { signal: own.signal, current: () => mine === seq };
    },
    cancel() { controller?.abort(); controller = null; seq++; },
  };
}

/** Load one run for the drawer through the guard. Null means "ignore this response": a newer request superseded
 * it, or the server answered for a run other than the one asked for. */
export async function latestRun<T extends { id: string }>(guard: LatestOnly, id: string, load: (signal: AbortSignal) => Promise<T>):
  Promise<{ run: T } | { error: unknown } | null> {
  const request = guard.begin();
  try {
    const run = await load(request.signal);
    return request.current() && run.id === id ? { run } : null;
  } catch (error) {
    return request.current() ? { error } : null;
  }
}

export type RunAction = "message" | "steer" | "replan" | "verify" | "verify-config" | "freestyle" | "approve" | "stop"
  | "criteria" | "judge";
const ACTION_PATHS: Record<RunAction, string> = {
  message: "message", steer: "message", replan: "replan", verify: "verify", "verify-config": "verify-config",
  freestyle: "freestyle", approve: "approve", stop: "stop", criteria: "criteria", judge: "judge",
};

/** The URL and body of a drawer control, bound to the run on screen (never the panel's id prop) and carrying the
 * state_revision the reader saw, so the server refuses it with REVISION_CONFLICT if the run moved on. Null while
 * the loaded run is not the one the panel asked for: every control is disabled then. */
export function runAction(run: Pick<Run, "id" | "state_revision"> | null, id: string, action: RunAction, values: Record<string, unknown> = {}) {
  if (!run || run.id !== id) return null;
  // Stop stays unconditional: it is the emergency path and must work from a stale view, so it sends no revision.
  const body = action === "stop" ? { ...values } : { ...values, expected_revision: run.state_revision };
  return { url: `/remotea/runs/${run.id}/${ACTION_PATHS[action]}`, body };
}

/** Shown (instead of an error) when a control was refused because the run changed; the drawer reloads it. */
export const RUN_CHANGED = "This run changed; reloaded";
export function isRevisionConflict(error: unknown) {
  return (error as { code?: unknown } | null)?.code === "REVISION_CONFLICT";
}

/** POST /remotea/runs/{id}/focus: whether a desktop app was told to bring the run's terminal tab forward. */
export type FocusResult = { sent: true; desktops?: number | null } | { sent: false; reason: string; detail?: string };

/** Why the tab could not be brought forward, in words (the reasons the route returns). */
export const FOCUS_TEXT: Record<string, string> = {
  no_desktop: "No remotea desktop app is connected to this host right now.",
  host_offline: "The remotea server is offline.",
  no_surface: "This run has no terminal open on the host.",
  host_too_old: "This host's remotea server is too old to open terminals from here; update it.",
  no_answer: "The remotea server did not answer in time.",
  failed: "The remotea server could not open the terminal.",
};

/** A path as one POSIX shell word: single-quoted unless it is plainly safe. */
export function shellWord(value: string) {
  return /^[A-Za-z0-9_./~:@%+=,-]+$/.test(value) ? value : "'" + value.replace(/'/g, "'\\''") + "'";
}

/** Where to find the terminal by hand when the tab could not be brought forward: the host, the worktree and branch,
 * and commands to copy (remotea has no interactive attach: `cd` into the worktree, `remotea screen` to look). */
export function terminalFallback(run: Pick<Run, "host" | "worktree" | "branch" | "surf">, result: FocusResult | null) {
  const reason = result && !result.sent ? result.reason : null;
  const commands: { label: string; command: string }[] = [];
  if (run.worktree) commands.push({ label: "Open a shell in the worktree", command: "cd " + shellWord(run.worktree) });
  if (run.surf && /^surf-[A-Za-z0-9_-]+$/.test(run.surf))
    commands.push({ label: "See the terminal's screen", command: "remotea screen " + run.surf });
  return {
    message: reason ? (FOCUS_TEXT[reason] ?? FOCUS_TEXT.failed) + (result && !result.sent && result.detail ? " (" + result.detail + ")" : "") : "",
    host: run.host?.name || "the remotea host",
    worktree: run.worktree || null,
    branch: run.branch || null,
    commands,
  };
}

/** The status line after a successful focus. */
export function focusSentText(run: Pick<Run, "host">, result: FocusResult) {
  if (!result.sent) return "";
  const where = run.host?.name ? " on " + run.host.name : "";
  return "Brought the terminal forward in remotea" + where + ".";
}
