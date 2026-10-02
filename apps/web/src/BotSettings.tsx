import { useEffect, useState } from "react";
import { Check, Copy, KeyRound, Plus, X } from "lucide-react";
import { api, post } from "./api";
import { SettingsGroup } from "./SettingsLayout";

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
  const scopeLabel = (scope: string) => scope === "work:run" ? "Eri requests"
    : scope.replace("organization", "Goals & projects").replace(":write", ", edit").replace(":read", ", read").replace(/^./, c => c.toUpperCase());
  return <SettingsGroup className="bot-settings" icon={<KeyRound size={18}/>} title="Connected agents"
    description={`Let other assistants work in ${workspace}. Their changes appear in Activity.`}
    action={<button className={adding ? "btn btn-ghost" : "btn btn-soft"} disabled={busy || !data} onClick={() => { setAdding(!adding); setSecret(""); }}>
      {adding ? <X size={15}/> : <Plus size={15}/>} {adding ? "Cancel" : "Add agent"}</button>}>
    {error && <p className="error-banner" role="alert">{error}</p>}
    {!data && !error && <p role="status" className="settings-empty">Loading connected agents…</p>}
    {adding && <form className="settings-form bot-key-form" onSubmit={create}>
      <label className="field wide"><span className="field-label-text">Agent name</span><input autoFocus value={name} maxLength={80} required placeholder="For example, Codex" onChange={e => setName(e.target.value)}/></label>
      {[["records","Custom records (includes note content)"],["schema","Structure definitions"],["tasks","Task scheduling"],["organization","Legacy organization"],["notes","Notes"]].map(([group,label]) =>
        <label className="field" key={group}><span className="field-label-text">{label}</span><select value={levels[group]} onChange={e => setLevels({...levels, [group]:e.target.value})}>
          <option value="none">No access</option><option value="read">Read</option>{!readOnly && (group!=="schema"||canDesign) && <option value="write">Read & edit</option>}
        </select></label>)}
      <label className="field"><span className="field-label-text">Key expires after</span><select value={days} onChange={e => setDays(Number(e.target.value))}>
        <option value={30}>30 days</option><option value={90}>90 days</option><option value={365}>1 year</option></select></label>
      {!readOnly && <label className="inline-check wide"><input type="checkbox" checked={queued} onChange={e => setQueued(e.target.checked)}/>
        Allow requests to Eri’s background agent</label>}
      <p className="footnote">Access applies to this workspace. Personal memory, conversations and connected accounts stay private. Revoking a key stops future actions, including queued work.</p>
      <div className="settings-form-actions"><button className="btn btn-primary" disabled={busy || !name.trim() || Object.values(levels).every(v => v === "none")}>{busy ? "Creating…" : "Create key"}</button></div>
    </form>}
    {secret && <div className="bot-secret" role="region" aria-label="New agent key">
      <strong>Copy this key now</strong><p>This is the only time it will be shown.</p>
      <label className="sr-only" htmlFor="bot-secret">New agent key</label>
      <input id="bot-secret" readOnly type={visible ? "text" : "password"} value={secret} autoComplete="off" onFocus={e => e.target.select()}/>
      <div className="bot-key-actions"><button className="btn btn-primary btn-sm" onClick={() => void copy(secret, "key")}>{copied === "key" ? <Check size={14}/> : <Copy size={14}/>} {copied === "key" ? "Copied" : "Copy key"}</button>
        <button className="btn btn-ghost btn-sm" onClick={() => setVisible(!visible)}>{visible ? "Hide" : "Reveal"}</button>
        <button className="btn btn-ghost btn-sm" onClick={() => setSecret("")}>Done</button></div>
    </div>}
    {!!data && <div className="settings-list bot-key-list">{!data.items.length && <p className="settings-empty">No agents connected yet.</p>}
      {data.items.map(item => {
        const expired = new Date(item.expires_at).getTime() <= Date.now();
        return <article className="settings-item bot-key-row" key={item.id}><div className="settings-item-main"><strong className="settings-item-title">{item.name}</strong>
          <span className="settings-item-meta"><span className={"chip" + (item.revoked_at || expired ? "" : " chip-done")}>{item.revoked_at ? "Revoked" : expired ? "Expired" : `Expires ${new Date(item.expires_at).toLocaleDateString()}`}</span><code>{item.prefix}…</code>
            {item.last_used_at && <span>Last used {new Date(item.last_used_at).toLocaleString()}</span>}</span>
          <span className="chip-row">{item.scopes.filter(scope => !scope.endsWith(":read") || !item.scopes.includes(scope.replace(":read",":write"))).map(scope =>
            <span className="chip" key={scope}>{scopeLabel(scope)}</span>)}</span></div>
          {!item.revoked_at && !expired && <div className="settings-item-actions"><button className="btn btn-danger btn-sm" disabled={busy} onClick={() => void revoke(item.id)}>Revoke</button></div>}
        </article>;
      })}</div>}
    {!!data && <details className="settings-block bot-connection"><summary>Connection details</summary>
      <p className="footnote">Use your agent’s key as a Bearer token.</p>
      {[["MCP", data.mcp_url], ["API", data.api_url]].map(([label,value]) => <div className="copy-field" key={label}>
        <label className="field"><span className="field-label-text">{label}</span><input readOnly aria-label={`${label} URL`} value={value} onFocus={e => e.target.select()}/></label>
        <button className="btn-icon" aria-label={`Copy ${label} URL`} onClick={() => void copy(value,label)}>{copied === label ? <Check size={15}/> : <Copy size={15}/>}</button>
      </div>)}
      <div className="settings-links"><a href="https://github.com/rdavisdunham/jarvis/blob/main/docs/EXTERNAL_AGENTS.md" target="_blank" rel="noreferrer">Connection guide</a></div>
    </details>}
  </SettingsGroup>;
}
