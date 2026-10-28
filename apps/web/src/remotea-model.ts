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
