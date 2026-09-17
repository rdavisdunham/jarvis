import { useEffect, useState } from "react";
import { Check, Copy, KeyRound, Plus, X } from "lucide-react";
import { api, post } from "./api";

export type BotKey = { id: string; name: string; prefix: string; scopes: string[];
  expires_at: string; created_at: string; revoked_at: string | null; last_used_at: string | null };
type KeyList = { items: BotKey[]; api_url: string; mcp_url: string };
export function permissionScopes(levels: Record<string, string>, queued: boolean) {
  return [...Object.entries(levels).filter(([,level]) => level !== "none").map(([group,level]) => `${group}:${level}`),
    ...(queued ? ["work:run"] : [])];
}
export function BotSettings({ workspace, readOnly = false, canDesign = true }: { workspace: string; readOnly?: boolean; canDesign?: boolean }) {
  const [data, setData] = useState<KeyList | null>(null), [error, setError] = useState("");
  const [adding, setAdding] = useState(false), [busy, setBusy] = useState(false), [name, setName] = useState("");
  const [days, setDays] = useState(90), [queued, setQueued] = useState(false);
  const [levels, setLevels] = useState<Record<string,string>>({ tasks: readOnly ? "read" : "write", organization: "read", notes: "none", records: "none", schema: "read" });
  const [secret, setSecret] = useState(""), [visible, setVisible] = useState(false), [copied, setCopied] = useState("");
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    let active = true;
    api<KeyList>("/bot-keys").then(value => { if (active) setData(value); })
      .catch(e => { if (active) setError(e.message); });
    return () => { active = false; };
  }, [revision]);
  async function copy(value: string, label: string) {
    try { await navigator.clipboard.writeText(value); setCopied(label); }
    catch { setError("Copy is unavailable here. Select the value to copy it manually."); }
  }
  async function create(event: React.FormEvent) {
    event.preventDefault(); setBusy(true); setError(""); setSecret(""); setCopied("");
    try {
      const result = await post<{ credential: BotKey; token: string }>("/bot-keys", {
        name, scopes: permissionScopes(levels, queued), expires_in_days: days,
      });
      setSecret(result.token); setVisible(false); setAdding(false); setName(""); setRevision(n => n + 1);
    } catch(e) { setError((e as Error).message); setRevision(n => n + 1); }
    finally { setBusy(false); }
  }
  async function revoke(id: string) {
    setBusy(true); setError("");
    try { await post(`/bot-keys/${id}/revoke`); setSecret(""); setRevision(n => n + 1); }
    catch(e) { setError((e as Error).message); }
    finally { setBusy(false); }
  }
  return <section className="bot-settings google-settings">
    <header className="bot-heading"><div><h2><KeyRound size={18}/> Connected agents</h2>
      <p>Let other assistants work in {workspace}. Their changes appear in Activity.</p></div>
      <button className="secondary compact" disabled={busy || !data} onClick={() => { setAdding(!adding); setSecret(""); }}>
        {adding ? <X size={15}/> : <Plus size={15}/>} {adding ? "Cancel" : "Add agent"}</button></header>
    {error && <p className="error-banner" role="alert">{error}</p>}
    {!data && !error && <p role="status">Loading connected agents…</p>}
    {adding && <form className="bot-key-form" onSubmit={create}>
      <label>Agent name<input autoFocus value={name} maxLength={80} required placeholder="e.g. Codex" onChange={e => setName(e.target.value)}/></label>
      <div className="bot-permissions">{[["records","Custom records (includes note content)"],["schema","Structure definitions"],["tasks","Task scheduling"],["organization","Legacy organization"],["notes","Notes"]].map(([group,label]) =>
        <label key={group}>{label}<select value={levels[group]} onChange={e => setLevels({...levels, [group]:e.target.value})}>
          <option value="none">No access</option><option value="read">Read</option>{!readOnly && (group!=="schema"||canDesign) && <option value="write">Read & edit</option>}
        </select></label>)}</div>
      <label>Key expires after<select value={days} onChange={e => setDays(Number(e.target.value))}>
        <option value={30}>30 days</option><option value={90}>90 days</option><option value={365}>1 year</option></select></label>
      {!readOnly && <label className="bot-queued"><input type="checkbox" checked={queued} onChange={e => setQueued(e.target.checked)}/>
        Allow requests to Eri’s background agent</label>}
      <p className="footnote">Access applies to this workspace. Personal memory, conversations and connected accounts stay private. Revoking a key stops future actions, including queued work.</p>
      <button className="primary compact" disabled={busy || !name.trim() || Object.values(levels).every(v => v === "none")}>{busy ? "Creating…" : "Create key"}</button>
    </form>}
    {secret && <div className="bot-secret" role="region" aria-label="New agent key">
      <strong>Copy this key now</strong><p>This is the only time it will be shown.</p>
      <label className="sr-only" htmlFor="bot-secret">New agent key</label>
      <input id="bot-secret" readOnly type={visible ? "text" : "password"} value={secret} autoComplete="off" onFocus={e => e.target.select()}/>
      <div className="bot-key-actions"><button className="secondary compact" onClick={() => void copy(secret, "key")}>{copied === "key" ? <Check size={14}/> : <Copy size={14}/>} {copied === "key" ? "Copied" : "Copy key"}</button>
        <button className="text-button" onClick={() => setVisible(!visible)}>{visible ? "Hide" : "Reveal"}</button>
        <button className="text-button" onClick={() => setSecret("")}>Done</button></div>
    </div>}
    {!!data && <div className="bot-key-list">{!data.items.length && <p className="footnote">No agents connected yet.</p>}
      {data.items.map(item => {
        const expired = new Date(item.expires_at).getTime() <= Date.now();
        return <article className="bot-key-row" key={item.id}><div><strong>{item.name}</strong>
          <small>{item.revoked_at ? "Revoked" : expired ? "Expired" : `Expires ${new Date(item.expires_at).toLocaleDateString()}`} · {item.prefix}…</small>
          <small>{item.scopes.filter(scope => !scope.endsWith(":read") || !item.scopes.includes(scope.replace(":read",":write"))).map(scope =>
            scope === "work:run" ? "Eri requests" : scope.replace("organization", "Goals & projects").replace(":write", " · edit").replace(":read", " · read")).join(" / ")}</small>
          {item.last_used_at && <small>Last used {new Date(item.last_used_at).toLocaleString()}</small>}</div>
          {!item.revoked_at && !expired && <button className="text-button" disabled={busy} onClick={() => void revoke(item.id)}>Revoke</button>}
        </article>;
      })}</div>}
    {!!data && <details className="bot-connection"><summary>Connection details</summary>
      <p>Use your agent’s key as a Bearer token.</p>
      {[["MCP", data.mcp_url], ["API", data.api_url]].map(([label,value]) => <div key={label}>
        <label>{label}<input readOnly aria-label={`${label} URL`} value={value} onFocus={e => e.target.select()}/></label>
        <button className="icon-button" aria-label={`Copy ${label} URL`} onClick={() => void copy(value,label)}>{copied === label ? <Check size={15}/> : <Copy size={15}/>}</button>
      </div>)}
      <a href="https://github.com/rdavisdunham/jarvis/blob/main/docs/EXTERNAL_AGENTS.md" target="_blank" rel="noreferrer">Connection guide</a>
    </details>}
  </section>;
}
