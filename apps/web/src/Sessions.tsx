import { useEffect, useRef, useState, type ReactNode } from "react";
import { Copy, ExternalLink, FolderInput, Trash2, X } from "lucide-react";
import { api, post } from "./api";
import { Dialog, Popover } from "./ux";

export type AgentSession = {
  id: string; tool: string; tool_label: string; session_id: string; title: string; summary: string;
  first_prompt: string; last_prompt: string; machine: string; environment: string; cwd: string; repo: string; branch: string;
  turn_count: number; started_at: string | null; last_active_at: string | null; pr_urls: string[];
  issue_refs: { identifier: string; task_id: string; url: string | null }[];
  resume: { cmd?: string; url?: string; hint?: string; live?: { state: "live" | "lost"; workspace: string; terminal: string; host: string } };
  filed: { source: string | null; record_id: string | null; task_id: string | null; record_title: string | null };
};
type Listing = { items: AgentSession[]; total: number; next_offset: number | null };
type Rule = { id: string; path_prefix: string; record_id: string; record_title: string | null };
type Choice = { id: string; title: string; type_name?: string };
type View = "all" | "live" | "unfiled" | "rules";

const filedBy: Record<string, string> = {
  branch: "from the branch", mention: "from the prompts", path_rule: "by folder rule", manual: "by you", explicit: "by the agent",
};

export function since(value: string | null, now = Date.now()) {
  if (!value) return "";
  const minutes = Math.round((now - new Date(value).getTime()) / 60000);
  if (minutes < 60) return Math.max(minutes, 1) + "m ago";
  if (minutes < 60 * 24) return Math.round(minutes / 60) + "h ago";
  if (minutes < 60 * 24 * 30) return Math.round(minutes / 1440) + "d ago";
  return new Date(value).toLocaleDateString();
}

function RecordChooser({ label, button, onChoose }: { label: string; button: ReactNode; onChoose: (choice: { record_id: string } | { identifier: string }) => Promise<void> }) {
  const [query, setQuery] = useState(""), [items, setItems] = useState<Choice[]>([]), [error, setError] = useState("");
  const issue = /^[A-Za-z][A-Za-z0-9]{0,9}-\d{1,6}$/.test(query.trim());
  useEffect(() => {
    let live = true;
    if (!query.trim() || issue) { setItems([]); return; }
    void api<{ items: Choice[] }>("/structure/records?" + new URLSearchParams({ query, limit: "12" }))
      .then(r => live && setItems(r.items)).catch(e => live && setError(e.message));
    return () => { live = false; };
  }, [query, issue]);
  const choose = (close: () => void, choice: { record_id: string } | { identifier: string }) =>
    onChoose(choice).then(close).catch(e => setError(e.message));
  return <Popover label={label} button={button} panelClassName="home-picker">{close => <>
    <p className="footnote">A Linear issue key such as ENG-123, or a client, project or task.</p>
    <input type="search" aria-label="Find an issue or record" placeholder="ENG-123 or search…" value={query} onChange={e => { setQuery(e.target.value); setError(""); }}/>
    {issue && <button className="btn btn-soft" onClick={() => void choose(close, { identifier: query.trim().toUpperCase() })}>File under {query.trim().toUpperCase()}</button>}
    {items.map(r => <div className="home-option" key={r.id}><span>{r.title}<small>{r.type_name}</small></span>
      <button className="btn btn-sm" aria-label={"File under " + r.title} onClick={() => void choose(close, { record_id: r.id })}>Use</button></div>)}
    {error && <p role="alert">{error}</p>}
  </>}</Popover>;
}

export function SessionCard({ item, onChanged }: { item: AgentSession; onChanged: () => void }) {
  const [copied, setCopied] = useState(false), [error, setError] = useState("");
  const link = (choice: { record_id: string } | { identifier: string } | { clear: true }) =>
    post<AgentSession>(`/sessions/${item.id}/link`, choice).then(onChanged);
  const place = [item.repo || item.cwd.split("/").filter(Boolean).at(-1), item.branch].filter(Boolean).join(" · ");
  const live = item.resume.live;
  return <article className="work-card session-card" data-session-id={item.id}>
    <header><span className="session-tool">{item.tool_label}</span><time dateTime={item.last_active_at ?? undefined}>{since(item.last_active_at)}</time></header>
    <h3>{item.title}</h3>
    {(item.summary || item.first_prompt) && <p className="work-result">{item.summary || item.first_prompt}</p>}
    <p className="session-place" title={item.cwd}>{place}{item.environment && <small> · {item.environment}</small>}{item.turn_count > 0 && <small> · {item.turn_count} {item.turn_count === 1 ? "prompt" : "prompts"}</small>}</p>
    <div className="session-filed">
      {live && <span className={"chip " + (live.state === "live" ? "chip-done" : "chip-due-today")} title={[live.host, live.terminal].filter(Boolean).join(" · ")}>
        {live.state === "live" ? "Live" : "Restoring"} in remotea · {live.workspace || "Inbox"}</span>}
      {item.filed.record_id
        ? <span>Filed under <strong>{item.filed.record_title ?? "a record"}</strong> {filedBy[item.filed.source ?? ""] ?? ""}</span>
        : <span className="chip chip-due-today">Unfiled</span>}
      {item.issue_refs.map(r => r.url
        ? <a key={r.identifier} className="chip" href={r.url} target="_blank" rel="noreferrer">{r.identifier}</a>
        : <span key={r.identifier} className="chip">{r.identifier}</span>)}
    </div>
    <div className="work-card-actions">
      {item.resume.cmd && <button className="btn btn-ghost btn-sm" onClick={() => void navigator.clipboard.writeText(item.resume.cmd!).then(() => setCopied(true))}>
        <Copy size={14}/>{copied ? "Copied" : "Copy resume command"}</button>}
      {item.resume.url && <a className="btn btn-ghost btn-sm" href={item.resume.url} target="_blank" rel="noreferrer"><ExternalLink size={14}/>Open</a>}
      {item.pr_urls.map((url, i) => <a key={url} className="btn btn-ghost btn-sm" href={url} target="_blank" rel="noreferrer"><ExternalLink size={14}/>PR{item.pr_urls.length > 1 ? " " + (i + 1) : ""}</a>)}
      <RecordChooser label={"File " + item.title} button={<><FolderInput size={14}/>{item.filed.record_id ? "Refile" : "File"}</>} onChoose={choice => link(choice)}/>
      {(item.filed.source === "manual" || item.filed.source === "explicit") &&
        <button className="btn btn-ghost btn-sm" onClick={() => void link({ clear: true }).catch(e => setError(e.message))}>Use automatic filing</button>}
      <button className="btn btn-ghost btn-sm" onClick={() => void post(`/sessions/${item.id}/archive`, { archived: true }).then(onChanged).catch(e => setError(e.message))}>Hide</button>
    </div>
    {!item.resume.cmd && !live && item.resume.hint && <small className="work-revert-note">{item.resume.hint}</small>}
    {error && <p className="work-error" role="alert">{error}</p>}
  </article>;
}

function Rules({ onChanged }: { onChanged: () => void }) {
  const [rules, setRules] = useState<Rule[]>([]), [path, setPath] = useState(""), [error, setError] = useState("");
  const load = () => api<{ items: Rule[] }>("/sessions/rules").then(r => setRules(r.items)).catch(e => setError(e.message));
  useEffect(() => { void load(); }, []);
  const saved = (r: { items: Rule[] }) => { setRules(r.items); setError(""); onChanged(); };
  return <section className="session-rules" aria-label="Folder rules">
    <p className="footnote">Sessions in a folder file under a client or project unless a branch or prompt names an issue.</p>
    {rules.map(r => <div className="home-option" key={r.id}><span><code>{r.path_prefix}</code><small>{r.record_title ?? "Unavailable record"}</small></span>
      <button className="btn-icon" aria-label={"Delete rule for " + r.path_prefix} onClick={() => void api<{ items: Rule[] }>(`/sessions/rules/${r.id}`, { method: "DELETE" }).then(saved).catch(e => setError(e.message))}><Trash2 size={15}/></button></div>)}
    <div className="session-rule-form">
      <input aria-label="Folder" placeholder="/home/davin/clients/acme" value={path} onChange={e => setPath(e.target.value)}/>
      <RecordChooser label="Choose the record for this folder" button={<><FolderInput size={14}/>Add rule</>} onChoose={async choice => {
        if (!("record_id" in choice)) throw new Error("Choose a client or project record for a folder rule.");
        if (path.trim().length < 2) throw new Error("Enter a folder first.");
        saved(await post<{ items: Rule[] }>("/sessions/rules", { path_prefix: path.trim(), record_id: choice.record_id }));
        setPath("");
      }}/>
    </div>
    {error && <p role="alert">{error}</p>}
  </section>;
}

const empty: Record<View, string> = {
  all: "No sessions yet. Run the session collector on your work machine to report them.",
  live: "Nothing is running in a remotea terminal right now.",
  unfiled: "Every session is filed.",
  rules: "",
};

export function SessionsPanel({ onClose }: { onClose: () => void }) {
  const [query, setQuery] = useState(""), [view, setView] = useState<View>("all");
  const [data, setData] = useState<Listing | null>(null), [error, setError] = useState(""), [tick, setTick] = useState(0);
  const close = useRef<HTMLButtonElement>(null);
  useEffect(() => { close.current?.focus(); }, []);
  useEffect(() => {
    let live = true;
    const timer = setTimeout(() => {
      if (view === "rules") return;
      const params = new URLSearchParams({ limit: "50", ...(query.trim() ? { query: query.trim() } : {}), ...(view === "live" || view === "unfiled" ? { [view]: "true" } : {}) });
      void api<Listing>("/sessions?" + params).then(r => { if (live) { setData(r); setError(""); } }).catch(e => live && setError(e.message));
    }, query ? 250 : 0);
    return () => { live = false; clearTimeout(timer); };
  }, [query, view, tick]);
  const refresh = () => setTick(t => t + 1);
  return <Dialog onBackdrop={() => onClose()} backdropClassName="activity-backdrop" className="activity-panel sessions-panel" aria-label="Agent sessions"
    onKeyDown={event => { if (event.key === "Escape") onClose(); }}>
    <header><div><h2>Sessions</h2><p>Where work happened in Claude Code, Codex, ChatGPT and Claude Desktop.</p></div>
      <button ref={close} className="btn-icon" aria-label="Close sessions" onClick={onClose}><X size={20}/></button></header>
    <div className="activity-toolbar session-toolbar">
      <input type="search" aria-label="Search sessions" placeholder="Topic, issue key, repo or folder…" value={query} onChange={e => setQuery(e.target.value)}/>
      <div className="segmented" role="group" aria-label="Show">
        {([["all", "All"], ["live", "Live"], ["unfiled", "Unfiled"], ["rules", "Rules"]] as const).map(([value, label]) =>
          <button key={value} aria-pressed={view === value} onClick={() => setView(value)}>{label}</button>)}
      </div>
    </div>
    {error && <p role="alert">{error}</p>}
    {view === "rules" ? <div className="activity-items"><Rules onChanged={refresh}/></div> : <>
      {data && !data.items.length && <p className="activity-empty">{query ? "No sessions match." : empty[view]}</p>}
      <div className="activity-items">{data?.items.map(item => <SessionCard key={item.id} item={item} onChanged={refresh}/>)}</div>
    </>}
  </Dialog>;
}
