import { useEffect, useState } from "react";
import { AudioLines, Check, Copy, FileUp, RefreshCw, Trash2 } from "lucide-react";
import { api, ApiError, post } from "../api";
import { SettingRow, SettingsGroup } from "../SettingsLayout";
import { MeetingMemorySettings, type MeetingPreferences, type PreferenceLimits, type ProcessingStatus } from "./MeetingMemorySettings";

export type SourceDelivery = { connector: string; label: string; sources: number; last_delivery_at: string | null };
export type GranolaConnection = { id: string; connector: "granola"; account: string; has_key: boolean; enabled: boolean;
  health: string; last_error: string | null; last_success_at: string | null; created_at: string | null };
export type SourcesOverview = {
  adapters: { key: string; description: string; live: boolean }[];
  deliveries: SourceDelivery[];
  submission: { method: "PUT"; url: string; scope: string; labels: string[]; bridge: string };
  limits: { max_source_chars: number; max_segments: number; max_body_bytes: number; formats: string[]; extensions: string[] };
  granola: { enabled: boolean; connections: GranolaConnection[] };
  preferences: MeetingPreferences;
  preference_limits: PreferenceLimits;
};
export type ImportResult = { id: string; external_id: string; title: string; format: string; revision: number;
  ingestion_status: "created" | "content" | "metadata" | "none" | "deleted"; processing_status: string };

const BASE = "/context/settings";
/** The server refuses these when meeting context does not apply here; the panel then stays out of the way. */
const HIDDEN = new Set(["CONTEXT_DISABLED", "PERSONAL_WORKSPACE"]);
const LABELS: Record<string, string> = { upload: "File import", recorder: "Windows recorder", zoom: "Zoom", granola: "Granola", api: "Submission endpoint", other: "Other tool" };

export const sourceLabel = (label: string) => LABELS[label] ?? label.replace(/^./, c => c.toUpperCase());
export function when(value: string | null) { return value ? new Date(value).toLocaleString() : "Never"; }
export function importOutcome(result: ImportResult) {
  return { created: `Imported “${result.title}”.`, content: `Updated “${result.title}” with the new transcript.`,
    metadata: `“${result.title}” was already imported; its details were refreshed.`,
    none: `“${result.title}” was already imported. Nothing changed.`,
    deleted: `“${result.title}” was deleted earlier and stays deleted.` }[result.ingestion_status];
}
/** Query string for an import: the meeting time as an ISO instant plus the browser's zone. */
export function importParams(file: { name: string }, title: string, local: string, zone: string) {
  const params = new URLSearchParams({ filename: file.name });
  if (title.trim()) params.set("title", title.trim());
  if (local) params.set("occurred_at", new Date(local).toISOString());
  if (zone) params.set("timezone", zone);
  return params.toString();
}
function localNow() {
  const d = new Date(); d.setSeconds(0, 0);
  return new Date(d.getTime() - d.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
}
const zone = () => { try { return Intl.DateTimeFormat().resolvedOptions().timeZone; } catch { return ""; } };
const HEALTH: Record<string, string> = { ok: "Working", auth_failed: "Key rejected", degraded: "Retrying", not_entitled: "Not entitled",
  unconfigured: "No key", rejected: "Refused", unknown: "Not checked" };

export function SourcesSettings({ initial = null, initialStatus = null }: { initial?: SourcesOverview | null; initialStatus?: ProcessingStatus | null }) {
  const [data, setData] = useState<SourcesOverview | null>(initial), [hidden, setHidden] = useState(false);
  const [error, setError] = useState(""), [refresh, setRefresh] = useState(0);
  useEffect(() => {
    if (initial && refresh === 0) return;
    let live = true;
    api<SourcesOverview>(BASE).then(value => { if (live) { setData(value); setError(""); } })
      .catch(e => { if (!live) return; if (e instanceof ApiError && HIDDEN.has(e.code)) setHidden(true); else setError(e.message); });
    return () => { live = false; };
  }, [refresh, initial]);
  if (hidden) return null;
  const reload = () => setRefresh(n => n + 1);
  return <><SettingsGroup className="sources-settings" icon={<AudioLines size={18}/>} title="Meeting sources"
    description="Transcripts reach meeting memory through any of these. Eridani keeps its own copy of every meeting it receives.">
    {error && <p role="alert" className="error-banner">{error}</p>}
    {!data ? !error && <p role="status" className="settings-empty">Loading meeting sources…</p> : <>
      <Deliveries items={data.deliveries}/>
      <FileImport limits={data.limits} onImported={reload}/>
      <Submission submission={data.submission}/>
      {data.granola.enabled && <GranolaSection connections={data.granola.connections} onChanged={reload}/>}
    </>}
  </SettingsGroup>
  {data && <MeetingMemorySettings preferences={data.preferences} limits={data.preference_limits} initialStatus={initialStatus}/>}
  </>;
}

function Deliveries({ items }: { items: SourceDelivery[] }) {
  return <div className="settings-list source-deliveries" aria-label="Delivered meetings">
    {!items.length && <p className="settings-empty">No meetings yet. Import a file or send one from another tool.</p>}
    {items.map(item => <article className="settings-item" key={item.connector + ":" + item.label}>
      <div className="settings-item-main"><strong className="settings-item-title">{sourceLabel(item.label)}</strong>
        <span className="settings-item-meta"><span className="chip">{item.sources} {item.sources === 1 ? "meeting" : "meetings"}</span>
          <span>Last delivery {when(item.last_delivery_at)}</span></span></div>
    </article>)}
  </div>;
}

function FileImport({ limits, onImported }: { limits: SourcesOverview["limits"]; onImported: () => void }) {
  const [file, setFile] = useState<File | null>(null), [title, setTitle] = useState(""), [at, setAt] = useState(localNow);
  const [busy, setBusy] = useState(false), [message, setMessage] = useState(""), [error, setError] = useState("");
  const isJson = !!file && file.name.toLowerCase().endsWith(".json");
  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = event.currentTarget;
    if (!file) return;
    setBusy(true); setError(""); setMessage("");
    try {
      const result = await api<ImportResult>(`${BASE}/import?${importParams(file, title, isJson ? "" : at, zone())}`,
        { method: "POST", body: file, headers: { "Content-Type": file.type || "application/octet-stream" } });
      setMessage(importOutcome(result)); setFile(null); setTitle(""); form.reset(); onImported();
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  }
  return <form className="settings-form settings-block source-import" onSubmit={e => void submit(e)}>
    <SettingRow label="Import a transcript" hint={`TXT, Markdown, WebVTT, SRT or versioned JSON, up to ${limits.max_source_chars.toLocaleString()} characters. Importing the same file again changes nothing.`}/>
    <label className="field wide"><span className="field-label-text">File</span>
      <input type="file" aria-label="Transcript file" accept={limits.extensions.join(",")} required
        onChange={e => { setFile(e.target.files?.[0] ?? null); setMessage(""); setError(""); }}/></label>
    <label className="field"><span className="field-label-text">Title</span>
      <input value={title} maxLength={500} placeholder="From the file" onChange={e => setTitle(e.target.value)}/></label>
    {!isJson && <label className="field"><span className="field-label-text">When it happened</span>
      <input type="datetime-local" required value={at} onChange={e => setAt(e.target.value)}/></label>}
    {error && <p role="alert" className="error-banner">{error}</p>}
    {message && <p role="status" className="footnote">{message}</p>}
    <div className="settings-form-actions"><button className="btn btn-primary" disabled={busy || !file}>
      <FileUp size={15}/> {busy ? "Importing…" : "Import"}</button></div>
  </form>;
}

function Submission({ submission }: { submission: SourcesOverview["submission"] }) {
  const [copied, setCopied] = useState(""), [error, setError] = useState("");
  async function copy(value: string, label: string) {
    try { await navigator.clipboard.writeText(value); setCopied(label); }
    catch { setError("Copy is unavailable here. Select the value to copy it manually."); }
  }
  const fields: [string, string][] = [["Endpoint", `${submission.method} ${submission.url}`], ["Bridge script", submission.bridge]];
  return <div className="settings-block source-submission">
    <SettingRow label="Send from another tool" hint={<>The Windows recorder, Zoom exports and Granola exports post transcripts here.
      Create a key under Connected agents with “Submit meeting transcripts and notes” turned on ({submission.scope}), then use
      it as a Bearer token. Labels: {submission.labels.join(", ")}.</>}/>
    {error && <p role="alert" className="error-banner">{error}</p>}
    {fields.map(([label, value]) => <div className="copy-field" key={label}>
      <label className="field"><span className="field-label-text">{label}</span>
        <input readOnly aria-label={label} value={value} onFocus={e => e.target.select()}/></label>
      <button type="button" className="btn-icon" aria-label={`Copy ${label.toLowerCase()}`} onClick={() => void copy(value, label)}>
        {copied === label ? <Check size={15}/> : <Copy size={15}/>}</button>
    </div>)}
  </div>;
}

function GranolaSection({ connections, onChanged }: { connections: GranolaConnection[]; onChanged: () => void }) {
  const [account, setAccount] = useState(""), [key, setKey] = useState(""), [days, setDays] = useState(90);
  const [busy, setBusy] = useState(false), [error, setError] = useState(""), [message, setMessage] = useState("");
  async function act(fn: () => Promise<string>) {
    setBusy(true); setError(""); setMessage("");
    try { setMessage(await fn()); } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); onChanged(); }
  }
  const connection = connections[0];
  return <div className="settings-block source-granola">
    <SettingRow label="Granola API (optional)" hint="For Granola Business or Enterprise workspaces: Eridani checks for your new meetings every few minutes. The key is stored encrypted."/>
    {error && <p role="alert" className="error-banner">{error}</p>}
    {message && <p role="status" className="footnote">{message}</p>}
    {!connection ? <form className="settings-form" onSubmit={e => { e.preventDefault(); void act(async () => {
        await post(`${BASE}/granola`, { account, api_key: key }); setKey(""); return "Granola connected.";
      }); }}>
      <label className="field"><span className="field-label-text">Granola account email</span>
        <input type="email" required autoComplete="off" value={account} onChange={e => setAccount(e.target.value)}/>
        <span className="field-hint">Only meetings this account owns are imported.</span></label>
      <label className="field wide"><span className="field-label-text">API key</span>
        <input type="password" aria-label="Granola API key" required autoComplete="off" value={key} onChange={e => setKey(e.target.value)}/></label>
      <div className="settings-form-actions"><button className="btn btn-primary" disabled={busy || !key || !account}>Connect Granola</button></div>
    </form> : <>
      <div className="integration-status">
        <span className={"status-dot" + (connection.health === "ok" ? " on" : "")} aria-hidden="true"/>
        <strong>{connection.account}</strong>
        <span className="chip">{HEALTH[connection.health] ?? connection.health}</span>
        <span className="footnote">Last successful sync: {connection.last_success_at ? when(connection.last_success_at) : "Not yet synced"}</span>
      </div>
      {connection.last_error && <p role="alert" className="error-banner">{connection.last_error}</p>}
      <div className="setting-actions">
        <button type="button" className="btn" disabled={busy} onClick={() => void act(async () => {
          const r = await post<GranolaConnection & { ok: boolean }>(`${BASE}/granola/${connection.id}/test`);
          return r.ok ? "The key works." : "The check failed.";
        })}>Test key</button>
        <button type="button" className="btn" disabled={busy} onClick={() => void act(async () => {
          const r = await post<{ already_queued: boolean }>(`${BASE}/granola/${connection.id}/sync`);
          return r.already_queued ? "A sync is already queued." : "Sync queued.";
        })}><RefreshCw size={15}/> Sync now</button>
        <label className="field"><span className="field-label-text">Backfill days</span>
          <input type="number" min={1} max={365} value={days} onChange={e => setDays(Number(e.target.value))}/></label>
        <button type="button" className="btn" disabled={busy || days < 1 || days > 365} onClick={() => void act(async () => {
          await post(`${BASE}/granola/${connection.id}/backfill`, { days });
          return `Backfill of the last ${days} days queued.`;
        })}>Backfill</button>
        <button type="button" className="btn btn-danger" disabled={busy} onClick={() => void act(async () => {
          await api(`${BASE}/granola/${connection.id}`, { method: "DELETE" });
          return "Granola removed. Meetings already imported stay in Eridani.";
        })}><Trash2 size={15}/> Remove</button>
      </div>
    </>}
  </div>;
}
