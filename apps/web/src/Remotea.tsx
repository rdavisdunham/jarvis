import { useCallback, useEffect, useState } from "react";
import { Check, ChevronDown, ChevronUp, Copy, MonitorSmartphone, RefreshCw } from "lucide-react";
import { api, post } from "./api";
import { SettingRow, SettingsGroup } from "./SettingsLayout";
import {
  ago, hostSummary, orderSessions, paritySummary, place, REMOTEA_POLL_MS, stateLabel,
  type Parity, type RemoteaSession, type RemoteaStatus, type Turn,
} from "./remotea-model";
import "./remotea.css";

async function copyText(value: string) {
  await navigator.clipboard.writeText(value);
}

/** Refresh on agent/session change-feed signals (coalesced), with a slower poll as the fallback. */
function useAgentRefresh(refresh: () => void) {
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | null = null;
    const soon = () => { if (!timer) timer = setTimeout(() => { timer = null; refresh(); }, 300); };
    const poll = window.setInterval(() => { if (document.visibilityState === "visible") refresh(); }, REMOTEA_POLL_MS);
    window.addEventListener("eri-agents-changed", soon);
    document.addEventListener("visibilitychange", soon);
    return () => {
      if (timer) clearTimeout(timer);
      window.clearInterval(poll);
      window.removeEventListener("eri-agents-changed", soon);
      document.removeEventListener("visibilitychange", soon);
    };
  }, [refresh]);
}

function TurnText({ session, turn }: { session: string; turn: Turn }) {
  const [open, setOpen] = useState(false), [full, setFull] = useState<string | null>(null), [error, setError] = useState("");
  const expand = async () => {
    setOpen(!open);
    if (open || full !== null || !turn.clipped) return;
    try {
      const turns = await api<{ items: Turn[] }>(`/remotea/sessions/${session}/turns?limit=1`);
      setFull(turns.items[0]?.text ?? turn.text);
    } catch (e) { setError((e as Error).message); }
  };
  const text = open ? (full ?? turn.text) : turn.text.slice(0, 280);
  const more = turn.text.length > 280 || turn.clipped;
  return <div className="remotea-turn">
    <p className={open ? "open" : ""}>{text}{!open && more && "…"}</p>
    <div className="remotea-turn-meta">
      <small>Turn {turn.seq} · {ago(turn.at)}{turn.truncated && " · final response truncated"}</small>
      {more && <button className="btn btn-ghost btn-sm" aria-expanded={open} onClick={() => void expand()}>
        {open ? <ChevronUp size={14}/> : <ChevronDown size={14}/>}{open ? "Less" : "Full response"}</button>}
    </div>
    {error && <p role="alert" className="work-error">{error}</p>}
  </div>;
}

function SessionRow({ item }: { item: RemoteaSession }) {
  const state = item.live_state?.state ?? "";
  return <article className={"remotea-session state-" + state} data-session-id={item.id}>
    <header>
      <span className={"remotea-state state-" + state}>{stateLabel(state)}</span>
      <span className="remotea-since">since {ago(item.live_state?.since ?? null)}</span>
      <span className="remotea-host">{item.host ? item.host.name || "remotea" : ""}{item.host && !item.host.connected && " (offline)"}</span>
    </header>
    <h3>{item.title}</h3>
    <p className="remotea-place">{item.tool_label}{place(item) && " · " + place(item)}{item.filed.record_title && " · " + item.filed.record_title}</p>
    {item.live_state?.reason && <p className="remotea-reason">{item.live_state.reason}</p>}
    {item.last_turn ? <TurnText session={item.id} turn={item.last_turn}/> : <p className="footnote">No final response captured yet.</p>}
  </article>;
}

export function Remotea() {
  const [status, setStatus] = useState<RemoteaStatus | null>(null);
  const [items, setItems] = useState<RemoteaSession[] | null>(null);
  const [parity, setParity] = useState<Parity | null>(null);
  const [error, setError] = useState(""), [loading, setLoading] = useState(false);
  const refresh = useCallback(() => {
    setLoading(true);
    Promise.all([
      api<RemoteaStatus>("/remotea/status"),
      api<{ items: RemoteaSession[] }>("/remotea/sessions"),
      api<Parity>("/remotea/parity"),
    ]).then(([s, list, p]) => { setStatus(s); setItems(orderSessions(list.items)); setParity(p); setError(""); })
      .catch(e => setError((e as Error).message))
      .finally(() => setLoading(false));
  }, []);
  useEffect(() => { refresh(); }, [refresh]);
  useAgentRefresh(refresh);
  const hosts = hostSummary(status?.hosts ?? []);
  return <div className="remotea-page">
    <section className={"remotea-hosts tone-" + hosts.tone} aria-label="remotea server">
      <MonitorSmartphone size={18} aria-hidden="true"/><span>{hosts.text}</span>
      <button className="btn-icon" aria-label="Refresh remotea" disabled={loading} onClick={refresh}><RefreshCw size={16}/></button>
    </section>
    {error && <p className="error-banner" role="alert">{error}</p>}
    <section className="remotea-sessions" aria-label="Agent sessions" aria-busy={loading}>
      {items && !items.length && <p className="activity-empty">No agent sessions are running in remotea right now.</p>}
      {items?.map(item => <SessionRow key={item.id} item={item}/>)}
      {!items && !error && <p role="status" className="settings-empty">Loading sessions…</p>}
    </section>
    <details className="remotea-parity">
      <summary>Collector parity</summary>
      <p>{paritySummary(parity)}</p>
      {!!parity?.recent.length && <ul>{parity.recent.slice(0, 8).map(r =>
        <li key={r.tool + r.session_id}><code>{r.session_id.slice(0, 12)}</code> {r.missing ? "not reported by the current collector" : Object.keys(r.diff).join(", ")}</li>)}</ul>}
    </details>
  </div>;
}

export function RemoteaSettings({ readOnly, onChanged }: { readOnly: boolean; onChanged: () => void }) {
  const [status, setStatus] = useState<RemoteaStatus | null>(null), [token, setToken] = useState("");
  const [busy, setBusy] = useState(false), [error, setError] = useState(""), [copied, setCopied] = useState("");
  useEffect(() => { void api<RemoteaStatus>("/remotea/status").then(setStatus).catch(e => setError(e.message)); }, []);
  const change = async (path: string, body?: unknown) => {
    setBusy(true); setError("");
    try {
      const next = await post<RemoteaStatus>(path, body);
      setStatus(next); setToken(next.token ?? ""); onChanged();
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  };
  const copy = (value: string, label: string) => copyText(value).then(() => setCopied(label))
    .catch(() => setError("Copy is unavailable here. Select the text to copy it."));
  const hosts = hostSummary(status?.hosts ?? []);
  return <SettingsGroup className="remotea-settings" icon={<MonitorSmartphone size={18}/>} title="remotea"
    description="Follow Claude Code and Codex sessions running in your remotea server from this app.">
    {error && <p className="error-banner" role="alert">{error}</p>}
    <SettingRow as="label" label="Show remotea" hint="Adds the remotea page and links your remotea server. Off by default.">
      <input type="checkbox" checked={!!status?.enabled} disabled={!status || busy || readOnly}
        onChange={e => void change(e.target.checked ? "/remotea/enable" : "/remotea/disable", e.target.checked ? {} : undefined)}/>
    </SettingRow>
    {status?.enabled && <>
      <SettingRow label="Server" hint={status.key ? `Link key ${status.key.prefix}… · expires ${new Date(status.key.expires_at).toLocaleDateString()}` : "No active link key."}>
        <span className={"remotea-host-chip tone-" + hosts.tone}>{hosts.tone === "connected" ? "Connected" : hosts.tone === "never" ? "Never connected" : ago(status.hosts.map(h => h.last_seen_at).filter(Boolean).sort().at(-1) ?? null)}</span>
      </SettingRow>
      {token && <div className="bot-secret" role="region" aria-label="remotea link key">
        <strong>Copy this key now</strong><p>This is the only time it will be shown. On your desktop, run the command below and paste the key when asked.</p>
        <label className="sr-only" htmlFor="remotea-secret">remotea link key</label>
        <input id="remotea-secret" readOnly type="password" value={token} autoComplete="off" onFocus={e => e.target.select()}/>
        <div className="bot-key-actions">
          <button className="btn btn-primary btn-sm" onClick={() => void copy(token, "key")}>{copied === "key" ? <Check size={14}/> : <Copy size={14}/>} {copied === "key" ? "Copied" : "Copy key"}</button>
          <button className="btn btn-ghost btn-sm" onClick={() => setToken("")}>Done</button>
        </div>
      </div>}
      <div className="copy-field">
        <label className="field"><span className="field-label-text">Connect command</span>
          <input readOnly aria-label="Connect command" value={status.connect_command} onFocus={e => e.target.select()}/></label>
        <button className="btn-icon" aria-label="Copy connect command" onClick={() => void copy(status.connect_command, "command")}>{copied === "command" ? <Check size={15}/> : <Copy size={15}/>}</button>
      </div>
      {!readOnly && <div className="settings-form-actions">
        <button className="btn btn-ghost btn-sm" disabled={busy} onClick={() => void change("/remotea/enable", { rotate: true })}>New link key</button>
      </div>}
    </>}
  </SettingsGroup>;
}
