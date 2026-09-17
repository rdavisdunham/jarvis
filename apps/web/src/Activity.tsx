import { useEffect, useRef, useState } from "react";
import { Check, ChevronDown, Clock3, Pencil, RotateCcw, X } from "lucide-react";
import { api, post } from "./api";
import { useDialogFocus } from "./components";

export type ActionChange = {
  id: string; command_id: string; kind: string; entity_id: string | null;
  title: string; operation: string; summary?: string; request_id?: string; revert_command_id?: string | null; fields: Record<string, { before: unknown; after: unknown }>;
  can_revert: boolean; revert_reason: string; reverted: boolean; remote_status?: string | null;
};
export type WorkItem = {
  voice_session_id?: string | null; response_native_id?: string; finished_at?: string | null; navigation_only?: boolean;
  clarification_history?: { question: string; answer: string }[];
  actor?: { type: "bot"; id: string; name: string } | null;
  id: string; parent_id: string | null; conversation_id: string; request: string;
  status: string; revision: number; message: string; actions: ActionChange[]; children: WorkItem[];
  cancel_requested: boolean; seen: boolean; waiting?: boolean; related_request_id?: string | null; created_at: string; updated_at: string; can_continue: boolean; can_revise?: boolean;
};
export const workActive = (item: WorkItem) => ["queued", "dispatched", "running", "waiting_sync"].includes(item.status);
export const workAttention = (item: WorkItem) => ["needs_input", "failed", "partial", "expired"].includes(item.status);
const labels: Record<string, string> = {
  queued: "Queued", dispatched: "Queued", running: "Working", needs_input: "Waiting for you",
  succeeded: "Completed", partial: "Partly completed", failed: "Couldn't finish",
  cancelled: "Cancelled", expired: "Expired", waiting_sync: "Syncing",
};
export function useWork(enabled: boolean, scope?: string, conversation?: string | null) {
  const [items, setItems] = useState<WorkItem[]>([]), [error, setError] = useState("");
  const [chatItems, setChatItems] = useState<WorkItem[]>([]);
  const current = useRef(0);
  async function refresh() {
    const generation = current.current;
    try {
      const conversationItems = async () => {
        const all = new Map<string, WorkItem>();
        let offset: number | null = 0;
        while (conversation && offset !== null && generation === current.current) {
          const page: {items: WorkItem[]; next_offset?: number | null} = await api("/work?conversation_id=" + encodeURIComponent(conversation) + "&offset=" + offset);
          page.items.forEach(item => all.set(item.id, item));
          offset = page.next_offset ?? null;
        }
        return {items: [...all.values()]};
      };
      const [result, chat] = await Promise.all([
        api<{ items: WorkItem[] }>("/work"),
        conversationItems(),
      ]);
      if (generation === current.current) { setItems(result.items); setChatItems(chat.items); setError(""); }
    } catch (e) {
      if (generation === current.current) setError((e as Error).message);
    }
  }
  useEffect(() => {
    current.current++;
    setItems([]); setChatItems([]);
    if (!enabled) return;
    void refresh();
    const update = () => void refresh();
    const timer = setInterval(update, 3000);
    window.addEventListener("eri-work-changed", update);
    return () => { current.current++; clearInterval(timer); window.removeEventListener("eri-work-changed", update); };
  }, [enabled, scope, conversation]);
  return { items, chatItems, error, refresh };
}
type Props = { item: WorkItem; onRefresh: () => Promise<void>; onOpen: (action: ActionChange) => Promise<void>; nested?: boolean; compact?: boolean };
function text(value: unknown): string {
  if (value === null || value === undefined || value === "") return "None";
  if (Array.isArray(value)) return value.join(", ") || "None";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "object") return Object.entries(value as Record<string,unknown>).map(([key,v])=>`${key}: ${text(v)}`).join(", ") || "None";
  return String(value).replaceAll("_", " ");
}
function changePreview(action: ActionChange, compact = false) {
  const important = compact && action.operation === "created" ? ["due_date", "due_time", "project"] : ["due_date", "due_time", "status", "project", "space", "assignee", "priority"];
  return important.filter(key => key in action.fields).slice(0, 3).map(key =>
    `${key.replaceAll("_", " ")}: ${text(action.fields[key].after)}`).join(" · ");
}
export function WorkCard({ item, onRefresh, onOpen, nested, compact = false }: Props) {
  const [busy, setBusy] = useState(false), [error, setError] = useState(""),
    [editing, setEditing] = useState(false), [correction, setCorrection] = useState("");
  const [expanded, setExpanded] = useState(false);
  const terminal = !workActive(item) && item.status !== "needs_input";
  useEffect(() => { if (terminal) setExpanded(false); }, [terminal]);
  const details = !compact || expanded;
  const revertIds = useRef(new Map<string, string>());
  async function act(fn: () => Promise<unknown>) {
    setBusy(true); setError("");
    try { await fn(); await onRefresh(); }
    catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  }
  const disclosure = compact ? <button type="button" className="text-button work-expand" aria-expanded={expanded} onClick={() => setExpanded(!expanded)}>{expanded ? "Less" : "Details"}<ChevronDown size={12}/></button> : null;
  const editable = new Set(["record", "task", "note", "project", "goal", "space", "area", "actor", "schedule", "planning", "google_event"]);
  return <article data-work-id={item.id} className={"work-card " + (nested ? "nested " : "") + (compact ? "chat-work-card " : "") + (workAttention(item) ? "attention" : "")}>
    <header><span className={"work-status " + item.status}>
      {workActive(item) ? <Clock3 size={13}/> : item.status === "succeeded" ? <Check size={13}/> : null}
      {item.cancel_requested && workActive(item) ? "Stopping unfinished work" : item.waiting ? "Waiting for related work" : labels[item.status] ?? item.status}
    </span><time dateTime={item.created_at}>{new Date(item.created_at).toLocaleTimeString([], {hour:"numeric",minute:"2-digit"})}</time></header>
    {details && item.actor && <p className="work-actor">{item.actor.name} <span>· connected agent</span></p>}
    {details && item.related_request_id && <p className="work-related">Follow-up to earlier work</p>}
    {!item.actions.length && !item.children.length && <p className="work-result">{
      item.message || (workActive(item) ? "Working on your request…" : "No changes were saved.")
    }</p>}
    {!!item.actions.length && workAttention(item) && !!item.message && <p className="work-result">{item.message}</p>}
    {item.status === "needs_input" && !!item.children.length && <p className="work-result">{item.message}</p>}
    {item.children.map(child => <WorkCard key={child.id} item={child} onRefresh={onRefresh} onOpen={onOpen} nested compact={compact}/>)}
    {item.actions.map(action => <div className="work-change" key={action.id}>
      <div><strong>{action.summary ?? `${action.operation.charAt(0).toUpperCase() + action.operation.slice(1)}: ${action.title}`}</strong>
        {action.remote_status && <span>{labels[action.remote_status] ?? action.remote_status}</span>}</div>
      {!!changePreview(action, compact) && <p className="work-change-preview">{changePreview(action, compact)}</p>}
      {action.reverted && <small className="work-reverted"><Check size={12}/>Change reverted</small>}
      {details && !!Object.keys(action.fields).length && <details><summary>Changes <ChevronDown size={12}/></summary>
        <dl>{Object.entries(action.fields).map(([field, values]) => <div key={field}><dt>{field.replaceAll("_", " ")}</dt>
          <dd>{action.operation !== "created" && <><del>{text(values.before)}</del><span aria-hidden="true"> → </span></>}
            <span>{text(values.after)}</span></dd></div>)}</dl></details>}
      <div className="work-card-actions">
        {action.entity_id && editable.has(action.kind) && <button className="text-button" disabled={busy}
          onClick={() => void act(() => onOpen(action))}><Pencil size={13}/>Edit</button>}
        <button className="text-button" disabled={busy || !action.can_revert} title={action.revert_reason}
          onClick={() => void act(() => {
            const command_id = revertIds.current.get(action.id) ?? crypto.randomUUID();
            revertIds.current.set(action.id, command_id);
            return post("/work/actions/"+action.id+"/revert", {command_id});
          })}><RotateCcw size={13}/>{action.reverted ? "Reverted" : "Revert"}</button>
        {action.id === item.actions.at(-1)?.id && disclosure}
      </div>
      {details && !action.can_revert && !action.reverted && <small className="work-revert-note">{action.revert_reason}</small>}
    </div>)}
    <div className="work-card-actions">
      {(workActive(item) || item.status === "needs_input") && <button className="text-button" disabled={busy || item.cancel_requested}
        title="Stop unfinished work. Saved changes stay in place."
        onClick={() => void act(() => post("/work/"+item.id+"/cancel", {}))}>Cancel work</button>}
      {item.can_revise !== false && (!item.children.length || item.status === "needs_input") && (item.can_continue || workActive(item)) && <button className="text-button" disabled={busy}
        onClick={() => setEditing(!editing)}>Revise</button>}
      {item.can_continue && <button className="text-button" disabled={busy} onClick={() => void act(() =>
        post("/work/"+item.id+"/revise", {message:"", expected_revision:item.revision, continue_work:true}))}>{item.status === "failed" ? "Retry" : "Continue"}</button>}
      {!nested && workAttention(item) && item.status !== "needs_input" && !item.seen && <button className="text-button" disabled={busy}
        onClick={() => void act(() => post("/work/"+item.id+"/seen", {}))}>Dismiss notification</button>}
      {!item.actions.length && disclosure}
    </div>
    {details && !!item.clarification_history?.length && <details className="work-original"><summary>Clarification history</summary>
      {item.clarification_history.map((turn, index) => <div key={index}><p><strong>Eri:</strong> {turn.question}</p><p><strong>You:</strong> {turn.answer}</p></div>)}
    </details>}
    {details && !!item.request && item.request !== "Request" && <details className="work-original"><summary>Original request</summary>
      <p>{item.request}</p></details>}
    {editing && <form className="work-revision" onSubmit={event => {
      event.preventDefault(); void act(async () => {
        await post("/work/"+item.id+"/revise", {message:correction, expected_revision:item.revision, continue_work:true});
        setEditing(false); setCorrection("");
      });
    }}><label>Clarification or correction<textarea autoFocus value={correction} maxLength={12000}
      onChange={event => setCorrection(event.target.value)}/></label>
      <button className="primary compact" disabled={busy || !correction.trim()}>Send correction</button></form>}
    {error && <p className="work-error" role="alert">{error}</p>}
  </article>;
}
export function ActivityPanel({ items, error, onClose, onRefresh, onOpen }: {
  items: WorkItem[]; error: string; onClose: () => void; onRefresh: () => Promise<void>; onOpen: Props["onOpen"];
}) {
  useDialogFocus();
  const [clearing, setClearing] = useState(false), [clearError, setClearError] = useState("");
  const close = useRef<HTMLButtonElement>(null);
  useEffect(() => { close.current?.focus(); }, []);
  return <div className="modal-backdrop activity-backdrop" onClick={event => {if(event.target===event.currentTarget) onClose();}}>
    <section className="activity-panel" role="dialog" aria-modal="true" aria-label="Eri activity"
      onKeyDown={event => {if(event.key === "Escape") onClose();}}>
      <header><div><h2>Activity</h2><p>Saved changes and work in progress.</p></div>
        <button ref={close} className="icon-button" aria-label="Close activity" onClick={onClose}><X size={20}/></button></header>
      {!!items.length && <button className="text-button" disabled={clearing} onClick={async () => {
        if (!window.confirm("Clear your activity history in this workspace and stop unfinished requests? Saved tasks, notes, and other changes will stay.")) return;
        setClearing(true); setClearError("");
        try { await post("/work/clear", {}); await onRefresh(); }
        catch (e) { setClearError((e as Error).message); }
        finally { setClearing(false); }
      }}>{clearing ? "Clearing…" : "Clear activity history"}</button>}
      {(error || clearError) && <p role="alert">{error || clearError}</p>}
      {!items.length && <p className="activity-empty">Ask Eri to do something. Its progress and saved changes will appear here.</p>}
      <div className="activity-items">{items.map(item => <WorkCard key={item.id} item={item} onRefresh={onRefresh} onOpen={onOpen}/>)}</div>
    </section>
  </div>;
}
