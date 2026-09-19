import { useEffect, useRef, useState } from "react";
import { Plus, Settings2, Sparkles, X } from "lucide-react";
import { z } from "zod";
import { useEditor } from "./editor-control";
import { api, post } from "./api";
import type { Schema, CustomRecord, SchemaField } from "./structure-types";
import type { NoteRecord } from "./Notes";
import "./note-lists.css";

export type NoteList = {
  id: string; revision: number; name: string; description: string;
  filters: {tags: string[]; type_id: string | null; values: Record<string, unknown>};
  automatic: boolean; extract_entries: boolean; archived: boolean; count: number; error: string | null;
};
export async function noteListCommand<T>(tool: string, args: unknown, commandId: string = crypto.randomUUID()): Promise<T> {
  const result = await post<{data: T}>("/commands", {command_id: commandId, tool, arguments: args});
  return result.data;
}
const editable = (v: NoteList | undefined) => v ? JSON.stringify([v.name,v.description,v.filters,v.automatic,v.extract_entries,v.archived]) : "";
const blank = (): NoteList => ({id: "", revision: 0, name: "", description: "",
  filters: {tags: [], type_id: null, values: {}}, automatic: true, extract_entries: true, archived: false, count: 0, error: null});

export function NoteLists({selected, onSelect, onOpen, revision, canEdit = true}: {
  selected: string; onSelect: (id: string) => void; onOpen?: (id: string) => void; revision: number; canEdit?: boolean;
}) {
  const [items, setItems] = useState<NoteList[]>([]), [draft, setDraft] = useState<NoteList | null>(null);
  const [schema, setSchema] = useState<Schema | null>(null), [records, setRecords] = useState<CustomRecord[]>([]);
  const [discarding, setDiscarding] = useState(false);
  const [busy, setBusy] = useState(false), [error, setError] = useState(""), [refresh, setRefresh] = useState(0);
  const [possible, setPossible] = useState<NoteRecord[] | null>(null), [finding, setFinding] = useState(false);
  const selection = useRef(selected); selection.current = selected;
  useEffect(() => {setPossible(null);setFinding(false);}, [selected, revision]);
  async function findPossible() {
    const id = selected; setFinding(true); setError("");
    try {const result = await api<{items:NoteRecord[]}>("/note-lists/" + id + "/suggestions"); if (selection.current === id) setPossible(result.items);}
    catch (e) {if (selection.current === id) setError((e as Error).message);}
    finally {if (selection.current === id) setFinding(false);}
  }
  const retry = useRef<{key: string; id: string} | null>(null);
  useEffect(() => {
    let live = true;
    api<{items: NoteList[]}>("/note-lists").then(r => {if (live) {setItems(r.items);setError("");}})
      .catch(e => {if (live) setError(e.message);});
    return () => {live = false;};
  }, [revision, refresh]);
  useEffect(() => {
    if (!draft) return;
    let live = true;
    api<Schema>("/structure").then(s => {if (live) setSchema(s);}).catch(e => {if (live) setError(e.message);});
    return () => {live = false;};
  }, [!!draft]);
  const type = schema?.types.find(t => t.id === draft?.filters.type_id);
  const fields = (type?.fields ?? []).filter(f => !f.archived && !f.binding && !["date", "datetime", "long_text"].includes(f.kind));
  useEffect(() => {
    if (!draft?.filters.type_id || !fields.some(f => f.kind === "relation")) return;
    let live = true;
    (async () => {
      let offset: number | null = 0;
      const rows: CustomRecord[] = [];
      while (offset !== null && live) {
        const result: {items: CustomRecord[]; next_offset: number | null} = await api("/structure/records?limit=200&offset=" + offset);
        rows.push(...result.items); offset = result.next_offset;
      }
      if (live) setRecords(rows);
    })().catch(e => {if (live) setError(e.message);});
    return () => {live = false;};
  }, [draft?.filters.type_id, schema?.revision]);
  async function run(tool: string, args: unknown) {
    const key = JSON.stringify({tool, args});
    if (retry.current?.key !== key) retry.current = {key, id: crypto.randomUUID()};
    setBusy(true); setError("");
    try {
      const result = await noteListCommand<NoteList>(tool, args, retry.current.id);
      retry.current = null; setRefresh(r => r + 1); setDraft(null); setDiscarding(false);
      if (tool === "notelist.save") onSelect(result.archived ? "" : result.id);
      return result;
    } catch (e) {setError((e as Error).message); return null;}
    finally {setBusy(false);}
  }
  function value(field: string, v: unknown) {
    if (!draft) return;
    const values = {...draft.filters.values};
    if (v === undefined) delete values[field]; else values[field] = v;
    setDraft({...draft, filters: {...draft.filters, values}});
  }
  function fieldInput(f: SchemaField) {
    const v = draft!.filters.values[f.id], multi = f.kind === "multiselect" || (f.kind === "relation" && f.multiple);
    const options = f.kind === "relation" ? records.filter(r => f.target_types.includes(r.type_id)).map(r => ({id:r.id, name:r.title})) : f.options;
    if (f.kind === "boolean") return <select value={v === undefined ? "" : String(v)} onChange={e => value(f.id, e.target.value === "" ? undefined : e.target.value === "true")}><option value="">Any value</option><option value="true">Yes</option><option value="false">No</option></select>;
    if (["select", "multiselect", "relation"].includes(f.kind)) return <select multiple={multi} value={multi ? (Array.isArray(v) ? v as string[] : []) : String(v ?? "")}
      onChange={e => {const selected = multi ? Array.from(e.target.selectedOptions, o => o.value).filter(Boolean) : e.target.value; value(f.id, selected.length ? selected : undefined);}}>
      {!multi && <option value="">Any value</option>}{options.map(o => <option key={o.id} value={o.id}>{o.name}</option>)}</select>;
    return <input type={f.kind === "number" ? "number" : "text"} value={String(v ?? "")} onChange={e => value(f.id, e.target.value === "" ? undefined : f.kind === "number" ? Number(e.target.value) : e.target.value)}/>;
  }
  const current = items.find(i => i.id === selected);
  const original = draft?.id ? items.find(i => i.id === draft.id) : blank();
  const dirty = !!draft && editable(draft) !== editable(original);
  const saveDraft = async () => {
    if (!draft) return null;
    return run("notelist.save", {id:draft.id || null,expected_revision:draft.revision,name:draft.name,description:draft.description,
      filters:{...draft.filters,tags:draft.filters.tags.filter(Boolean)},automatic:draft.automatic,extract_entries:draft.extract_entries,archived:draft.archived});
  };
  return <div className="note-lists">
    <div className="note-list-toolbar">
      <label>List<select aria-label="Note list" disabled={!!draft} value={selected} onChange={e => onSelect(e.target.value)}>
        <option value="">All notes</option><option value="uncategorized">Uncategorized</option>
        {selected && selected !== "uncategorized" && !current && <option value={selected}>Unavailable list</option>}
        {items.map(i => <option key={i.id} value={i.id}>{i.name} · {i.count}</option>)}
      </select></label>
      {canEdit && <button type="button" className="secondary compact" disabled={!!draft} onClick={() => {setError("");setDiscarding(false);setDraft(blank());}}><Plus size={15}/>New list</button>}
      {canEdit && current && <button type="button" className="icon-button" aria-label="Edit list" disabled={!!draft} onClick={() => {setError("");setDraft(structuredClone(current));}}><Settings2 size={18}/></button>}
      {canEdit && !items.length && <button type="button" className="secondary compact" disabled={busy} onClick={() => void run("notelist.setup", {})}>Set up suggested lists</button>}
    </div>
    {current && <p className="footnote">{current.description}</p>}
    {current && !current.error && <button type="button" className="text-button" disabled={finding} onClick={() => void findPossible()}>{finding ? "Finding possible matches…" : "Find possible missing items"}</button>}
    {possible && <aside className="note-possible"><div className="dialog-heading"><strong>Possible matches · not filed here</strong><button type="button" className="text-button" onClick={() => setPossible(null)}>Hide</button></div><p className="footnote">These may be uncategorized or filed elsewhere. Open an item to review its lists.</p>{possible.length ? possible.map(n => <button type="button" className="text-button" key={n.id} onClick={() => onOpen?.(n.id)}>{n.title}</button>) : <p className="footnote">No additional matches found.</p>}</aside>}
    {current?.error && <p className="error-banner" role="alert">{current.error}</p>}
    {error && <p className="error-banner" role="alert">{error}</p>}
    {draft && <ListDraftBridge draft={draft} dirty={dirty} busy={busy} patch={setDraft} close={() => setDraft(null)} save={saveDraft}/>}
    {draft && <form className="note-list-editor" onSubmit={e => {e.preventDefault();void saveDraft();}}>
      <div className="dialog-heading"><h3>{draft.id ? "Edit list" : "New list"}</h3><button type="button" className="icon-button" aria-label="Close list settings" disabled={busy} onClick={() => dirty ? setDiscarding(true) : setDraft(null)}><X size={18}/></button></div>
      {discarding && <div role="alert"><p>Discard unsaved list changes?</p><button type="button" onClick={() => setDiscarding(false)}>Keep editing</button><button type="button" onClick={() => {setDiscarding(false);setDraft(null);}}>Discard changes</button></div>}
      <div className="form-grid">
        <label>Name<input autoFocus required maxLength={80} value={draft.name} onChange={e => setDraft({...draft, name: e.target.value})}/></label>
        <label>Matching tags<input maxLength={800} placeholder="movies, date-night" value={draft.filters.tags.join(", ")} onChange={e => setDraft({...draft, filters: {...draft.filters, tags: e.target.value.split(",").map(t => t.trim())}})}/></label>
      </div>
      <label>Description<textarea required rows={3} maxLength={5000} placeholder="What belongs here? Eri uses this to organize your notes." value={draft.description} onChange={e => setDraft({...draft, description: e.target.value})}/></label>
      <details><summary>Custom field filters</summary>
        <label>Content collection<select value={draft.filters.type_id ?? ""} onChange={e => setDraft({...draft, filters: {...draft.filters, type_id: e.target.value || null, values: {}}})}>
          <option value="">Any content collection</option>{schema?.types.filter(t => t.capabilities.includes("content") && !t.archived).map(t => <option value={t.id} key={t.id}>{t.plural}</option>)}
        </select></label>
        <div className="form-grid">{fields.map(f => <label key={f.id}>{f.name}{fieldInput(f)}<small>{f.description}</small></label>)}</div>
      </details>
      <p className="footnote">Items must match every selected filter. Removing a list keeps its notes.</p>
      <label className="check-label"><input type="checkbox" checked={draft.automatic} onChange={e => setDraft({...draft, automatic: e.target.checked})}/>Let Eri organize new and edited notes into this list</label>
      <label className="check-label"><input type="checkbox" checked={draft.extract_entries} onChange={e => setDraft({...draft, extract_entries: e.target.checked})}/>Extract individual saved items from longer notes</label>
      <div className="note-list-actions"><button className="primary compact" disabled={busy}>{busy ? "Saving…" : "Save list"}</button>
        {draft.id && <button type="button" className="text-button danger" disabled={busy} onClick={() => void run("notelist.save", {
          id: draft.id, expected_revision: draft.revision, name: draft.name, description: draft.description, filters: draft.filters,
          automatic: draft.automatic, extract_entries: draft.extract_entries, archived: true,
        })}>Remove list</button>}
      </div>
    </form>}
  </div>;
}

export function NoteFiling({note, disabled, readOnly = false, onSaved, onOpen}: {
  note: NoteRecord; disabled: boolean; readOnly?: boolean; onSaved: (note: NoteRecord) => void; onOpen?: (id: string) => void;
}) {
  const [lists, setLists] = useState<NoteList[]>([]), [busy, setBusy] = useState(false), [error, setError] = useState("");
  const retry = useRef<{key: string; id: string} | null>(null);
  useEffect(() => {
    let live = true; api<{items: NoteList[]}>("/note-lists").then(r => {if (live) setLists(r.items);}).catch(e => {if (live) setError(e.message);});
    return () => {live = false;};
  }, [note.id]);
  useEffect(() => {
    if (disabled || !["queued", "running", "retry_waiting"].includes(note.organization?.status ?? "")) return;
    let live = true;
    const timer = window.setTimeout(() => {api<NoteRecord>("/notes/" + note.id).then(n => {if (live) onSaved(n);}).catch(e => {if(live) setError(e.message);});}, 2500);
    return () => {live = false;window.clearTimeout(timer);};
  }, [note, disabled, onSaved]);
  async function act(tool: string, args: unknown) {
    setBusy(true);setError("");
    const key = JSON.stringify({tool,args});
    if (retry.current?.key !== key) retry.current = {key,id:crypto.randomUUID()};
    try {await noteListCommand(tool, args, retry.current.id);retry.current=null;onSaved(await api<NoteRecord>("/notes/" + note.id));}
    catch(e) {setError((e as Error).message);}
    finally {setBusy(false);}
  }
  return <section className="note-filing">
    <div className="note-list-toolbar">
      <label>File in list<select aria-label="File note in list" value="" disabled={disabled || busy || readOnly} onChange={e => {if(e.target.value) void act("note.file", {note_id:note.id, expected_revision:note.revision, list_id:e.target.value});}}>
        <option value="">Choose a list…</option>{lists.filter(l => !l.error).map(l => <option value={l.id} key={l.id}>{l.name}</option>)}
      </select></label>
      {!note.organization?.generated && <button type="button" className="secondary compact" disabled={disabled || busy || readOnly || !lists.some(l => l.automatic)} onClick={() => void act("note.organize", {note_id:note.id,expected_revision:note.revision})}><Sparkles size={15}/>Organize now</button>}
    </div>
    {note.organization?.tags_locked && <small>Your tag choices take priority over automatic filing.</small>}
    {note.organization && ["queued", "running", "retry_waiting", "stale", "failed"].includes(note.organization.status) && <p className="footnote" role="status">{({queued:"Organization queued",running:"Organizing saved items…",retry_waiting:"Organization will retry; your note is saved",stale:"The note or list changed. Organize again when ready.",failed:"Organization could not finish. Your note is saved; try Organize now."} as Record<string,string>)[note.organization.status]}</p>}
    {!!note.organization?.uncertain_count && <p className="footnote">Some possible entries were left unchanged because their identity or category was uncertain.</p>}
    {error && <p className="error-banner" role="alert">{error}</p>}
    {(note.sources ?? []).map(s => <div className="note-source" key={s.id}><button type="button" className="text-button" disabled={disabled || busy} onClick={() => onOpen?.(s.id)}>From {s.title}</button><blockquote>{s.evidence}</blockquote>{s.source_changed && <small>The source has changed since this was saved.</small>}</div>)}
    {!!note.saved_entries?.length && <div className="note-saved-entries"><strong>Saved from this note</strong>{note.saved_entries.map(s => <button type="button" className="text-button" key={s.id} disabled={disabled || busy} onClick={() => onOpen?.(s.id)}>{s.title}</button>)}</div>}
  </section>;
}

function ListDraftBridge({draft,dirty,busy,patch,close,save}:{draft:NoteList;dirty:boolean;busy:boolean;patch:(value:NoteList)=>void;close:()=>void;save:()=>Promise<NoteList|null>}) {
  useEditor({kind:"note_list",record_id:draft.id || null,dirty,busy,
    schema:z.object({name:z.string().min(1).max(80),description:z.string().min(1).max(5000),tags:z.array(z.string().min(1).max(40)).max(20),automatic:z.boolean(),extract_entries:z.boolean()}),
    values:{name:draft.name,description:draft.description,tags:draft.filters.tags,automatic:draft.automatic,extract_entries:draft.extract_entries},
    patch: values => {const {tags,...rest}=values;patch({...draft,...rest,filters:{...draft.filters,...(tags ? {tags:tags as string[]} : {})}} as NoteList);},
    save:async()=>{const result=await save();if(!result)throw Error("The list could not be saved. Check its fields and try again.");return result;},close,discard:close});
  return null;
}
