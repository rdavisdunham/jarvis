import { useCallback, useEffect, useState } from "react";
import { IndentDecrease, IndentIncrease, LayoutTemplate, Plus, Trash2, X } from "lucide-react";
import { api, command, post } from "./api";
import { Dialog } from "./ux";
import { useStructureActions } from "./structure-actions";
import type { CustomRecord, Schema, SchemaField, SchemaType } from "./structure-types";
import {
  build, childTypes, createdLabel, defaultableFields, flatten, onboardingExample, parentTypeOf, previewLines, removeRow, rowIssues, shift,
  type InstantiatePreview, type InstantiateResult, type RecordTemplate, type Row, type TemplatePayload,
} from "./templates";
import "./templates.css";

// ---- Data -------------------------------------------------------------------------------
/** Active templates for the workspace (bounded server-side), refreshed on demand. */
export function useTemplates(refresh = 0) {
  const [items, setItems] = useState<RecordTemplate[]>([]);
  const [version, setVersion] = useState(0);
  useEffect(() => {
    let live = true;
    void api<{ items: RecordTemplate[] }>("/structure/templates").then(r => live && setItems(r.items)).catch(() => {});
    return () => { live = false; };
  }, [refresh, version]);
  const reload = useCallback(() => setVersion(v => v + 1), []);
  return { templates: items, reload };
}

/** One command, one receipt: the created subtree can be undone as a group. */
export async function instantiate(preview: InstantiatePreview, title?: string) {
  const run = command<InstantiateResult>("record.instantiate", {
    template_id: preview.template_id, parent_id: preview.parent_id, schema_revision: preview.schema_revision,
    ...(title ? { title } : {}), preview_hash: preview.preview_hash,
  });
  const result = (await run.send()).data;
  return { result, undo: () => command("record.restore_contents", { source_command_id: run.id }).send() };
}

// ---- Undo toast -------------------------------------------------------------------------
export type UndoToastState = { message: string; undo?: () => Promise<unknown> } | null;
export function useUndoToast(onUndone?: () => Promise<void> | void) {
  const [toast, setToast] = useState<UndoToastState>(null), [busy, setBusy] = useState(false);
  useEffect(() => { if (!toast) return; const timer = setTimeout(() => setToast(null), 6000); return () => clearTimeout(timer); }, [toast]);
  const element = toast && <div className="toast" role="status"><span>{toast.message}</span>
    {toast.undo && <button type="button" disabled={busy} onClick={() => {
      const undo = toast.undo!; setToast(null); setBusy(true);
      void undo().then(async () => { await onUndone?.(); setToast({ message: "Undone. The new records were archived." }); })
        .catch(e => setToast({ message: (e as Error).message })).finally(() => setBusy(false));
    }}>Undo</button>}</div>;
  return { element, show: setToast };
}

// ---- Stamps and preview -----------------------------------------------------------------
/** Dashed stamp cards: "Start from template: Client onboarding — A project with 4 tasks: …". */
export function TemplateStamps({ templates, onPick, disabled }: { templates: RecordTemplate[]; onPick: (t: RecordTemplate) => void; disabled?: boolean }) {
  if (!templates.length) return null;
  return <div className="template-stamps">{templates.map(t =>
    <button key={t.id} type="button" className="template-stamp" disabled={disabled} onClick={() => onPick(t)}>
      <strong>Start from template: {t.name}</strong><small>{t.summary}</small>
    </button>)}</div>;
}

export function PreviewTree({ preview }: { preview: InstantiatePreview }) {
  return <ul className="template-tree" aria-label="Records to create">{previewLines(preview.tree).map((l, i) =>
    <li key={i} style={{ paddingLeft: l.depth * 18 }} className={l.depth ? "" : "is-root"}><span className="chip">{l.type_name}</span><span>{l.title}</span></li>)}</ul>;
}

/** Preview what a template creates in this home, then create it. Used inline (Atlas) and in a dialog. */
export function TemplateStart({ template, parentId, schemaRevision, homeTitle, onCreate, onCancel }: {
  template: RecordTemplate; parentId: string | null; schemaRevision: number; homeTitle?: string;
  onCreate: (preview: InstantiatePreview, title: string) => Promise<void>; onCancel: () => void;
}) {
  const [title, setTitle] = useState(template.name), [preview, setPreview] = useState<InstantiatePreview | null>(null);
  const [error, setError] = useState(""), [busy, setBusy] = useState(false);
  useEffect(() => {
    let live = true; setError("");
    const timer = setTimeout(() => {
      void post<InstantiatePreview>("/structure/templates/instantiate/preview", { template_id: template.id, parent_id: parentId, schema_revision: schemaRevision, ...(title.trim() ? { title: title.trim() } : {}) })
        .then(p => live && setPreview(p)).catch(e => { if (live) { setPreview(null); setError((e as Error).message); } });
    }, 200);
    return () => { live = false; clearTimeout(timer); };
  }, [template.id, template.revision, parentId, schemaRevision, title]);
  const create = async () => {
    if (!preview) return;
    setBusy(true);
    try { await onCreate(preview, title.trim()); } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  };
  return <section className="template-start" aria-label={"Start from template " + template.name}>
    <label className="field">Title<input value={title} maxLength={500} onChange={e => setTitle(e.target.value)} aria-label="New record title"/></label>
    <p className="template-start-where">{homeTitle ? <>Inside <strong>{homeTitle}</strong></> : "Unfiled"}</p>
    {preview ? <><PreviewTree preview={preview}/><p className="footnote">{createdLabel(preview)} {preview.source_effect}</p></>
      : !error && <p className="footnote" role="status">Preparing a preview…</p>}
    {error && <p role="alert" className="template-error">{error}</p>}
    <div className="template-start-actions">
      <button type="button" className="btn btn-ghost btn-sm" onClick={onCancel}>Cancel</button>
      <button type="button" className="btn btn-primary btn-sm" disabled={busy || !preview || !title.trim()} onClick={() => void create()}>Create {preview?.record_count ?? ""} {preview?.record_count === 1 ? "record" : "records"}</button>
    </div>
  </section>;
}

/** Browse "Add here" and quick-add: choose a template for the chosen type(s), preview, create. */
export function StartFromTemplateDialog({ templates, parentId, homeTitle, schemaRevision, onClose, onCreated }: {
  templates: RecordTemplate[]; parentId: string | null; homeTitle?: string; schemaRevision: number; onClose: () => void;
  onCreated: (result: InstantiateResult, undo: () => Promise<unknown>) => Promise<void> | void;
}) {
  const [chosen, setChosen] = useState<RecordTemplate | null>(templates.length === 1 ? templates[0] : null);
  return <Dialog className="dialog template-dialog" aria-label="Start from template" onEscape={onClose} onBackdrop={onClose}>
    <div className="dialog-heading"><h2>Start from template</h2><button type="button" className="btn-icon" aria-label="Close" onClick={onClose}><X size={18}/></button></div>
    {chosen ? <>
      {templates.length > 1 && <button type="button" className="text-button template-back" onClick={() => setChosen(null)}>All templates</button>}
      <h3 className="template-chosen">{chosen.name}</h3>
      {chosen.description && <p className="footnote">{chosen.description}</p>}
      <TemplateStart template={chosen} parentId={parentId} homeTitle={homeTitle} schemaRevision={schemaRevision} onCancel={onClose}
        onCreate={async (preview, title) => { const made = await instantiate(preview, title); await onCreated(made.result, made.undo); onClose(); }}/>
    </> : <TemplateStamps templates={templates} onPick={setChosen}/>}
  </Dialog>;
}

// ---- Template editor --------------------------------------------------------------------
function ValueInput({ field, value, onChange }: { field: SchemaField; value: unknown; onChange: (v: unknown) => void }) {
  const label = { "aria-label": "Default " + field.name };
  if (field.kind === "boolean") return <label className="check-label"><input type="checkbox" {...label} checked={Boolean(value)} onChange={e => onChange(e.target.checked || null)}/>{field.name}</label>;
  if (field.kind === "select" || field.kind === "multiselect") {
    const multi = field.kind === "multiselect";
    return <select {...label} multiple={multi} value={multi ? (Array.isArray(value) ? value as string[] : []) : String(value ?? "")}
      onChange={e => onChange(multi ? (Array.from(e.target.selectedOptions, o => o.value).length ? Array.from(e.target.selectedOptions, o => o.value) : null) : e.target.value || null)}>
      {!multi && <option value="">No default</option>}{field.options.map(o => <option key={o.id} value={o.id}>{o.name}</option>)}</select>;
  }
  if (field.kind === "long_text") return <textarea {...label} rows={2} value={String(value ?? "")} onChange={e => onChange(e.target.value || null)}/>;
  return <input {...label} type={field.kind === "number" ? "number" : "text"} value={value === null || value === undefined ? "" : String(value)}
    onChange={e => onChange(e.target.value === "" ? null : field.kind === "number" ? Number(e.target.value) : e.target.value)}/>;
}

const clean = (values: Record<string, unknown>) => Object.fromEntries(Object.entries(values).filter(([, v]) => v !== null && v !== undefined && v !== ""));

export function TemplateEditor({ schema, typeId, template, initial, onClose, onSaved }: {
  schema: Schema; typeId: string; template?: RecordTemplate; initial?: { name: string; description: string; payload: TemplatePayload };
  onClose: () => void; onSaved: (t: RecordTemplate) => void;
}) {
  const type = schema.types.find(t => t.id === typeId);
  const source = template ?? initial;
  const [name, setName] = useState(source?.name ?? ""), [description, setDescription] = useState(source?.description ?? "");
  const [body, setBody] = useState(source?.payload.body ?? ""), [values, setValues] = useState<Record<string, unknown>>(source?.payload.values ?? {});
  const [rows, setRows] = useState<Row[]>(() => flatten(source?.payload.children ?? []));
  const { run, busy, error } = useStructureActions();
  if (!type) return null;
  const issues = rowIssues(rows, type.id, schema);
  const fields = defaultableFields(type);
  const update = (i: number, patch: Partial<Row>) => setRows(rows.map((r, n) => (n === i ? { ...r, ...patch } : r)));
  const add = () => {
    const options = childTypes(schema, type.id);
    const kind = options.find(t => t.id === "task") ?? options[0];
    if (kind) setRows([...rows, { key: crypto.randomUUID(), depth: 1, type_id: kind.id, title: "", body: "", values: {} }]);
  };
  const save = async () => {
    const payload = { values: clean(values), body, children: build(rows) };
    const saved = template
      ? await run<RecordTemplate>("template.update", { template_id: template.id, expected_revision: template.revision, name: name.trim(), description, payload })
      : await run<RecordTemplate>("template.create", { type_id: type.id, name: name.trim(), description, payload });
    onSaved(saved);
  };
  return <Dialog className="dialog template-editor" aria-label="Template editor" onEscape={busy ? undefined : onClose}>
    <div className="dialog-heading"><div><h2>{template ? "Edit template" : "New template"}</h2>
      <p className="template-sub">For {type.plural.toLowerCase()}. Saves right away; records created earlier never change.</p></div>
      <button type="button" className="btn-icon" aria-label="Close template editor" disabled={busy} onClick={onClose}><X size={18}/></button></div>
    <label>Name<input value={name} maxLength={120} autoFocus onChange={e => setName(e.target.value)} placeholder="Client onboarding"/></label>
    <label>Description <span className="template-optional">Optional</span><input value={description} maxLength={2000} onChange={e => setDescription(e.target.value)} placeholder="When to use this template"/></label>
    <label>Description outline<textarea rows={4} value={body} maxLength={30000} onChange={e => setBody(e.target.value)} placeholder={"## Goals\n- "}/>
      <span className="template-hint">Copied into each new {type.name.toLowerCase()}’s details.</span></label>
    {!!fields.length && <fieldset className="template-defaults"><legend>Default values</legend>
      <p className="template-hint">Dates, assignees and status are left for each new record.</p>
      {fields.map(f => <div key={f.id} className="template-default-row"><span>{f.name}</span><ValueInput field={f} value={values[f.id]} onChange={v => setValues({ ...values, [f.id]: v })}/></div>)}
    </fieldset>}
    <section className="template-rows" aria-label="Records inside">
      <div className="template-rows-head"><h3>Records inside</h3><span className="detail-count">{rows.length}</span></div>
      {!rows.length && <p className="template-hint">Add the tasks or notes every new {type.name.toLowerCase()} should start with.</p>}
      <ol>{rows.map((r, i) => {
        const options = childTypes(schema, parentTypeOf(rows, i, type.id));
        const label = r.title.trim() || "row " + (i + 1);
        return <li key={r.key} className="template-row" style={{ paddingLeft: (r.depth - 1) * 22 }}>
          <select aria-label={"Type of " + label} value={r.type_id} onChange={e => update(i, { type_id: e.target.value, values: {} })}>
            {!options.some(t => t.id === r.type_id) && <option value={r.type_id}>{schema.types.find(t => t.id === r.type_id)?.name ?? "Unavailable type"}</option>}
            {options.map(t => <option key={t.id} value={t.id}>{t.name}</option>)}</select>
          <input aria-label={"Title of " + label} value={r.title} maxLength={500} placeholder="Title" onChange={e => update(i, { title: e.target.value })}/>
          <button type="button" className="btn-icon" aria-label={"Move " + label + " out a level"} disabled={r.depth === 1} onClick={() => setRows(shift(rows, i, -1))}><IndentDecrease size={16}/></button>
          <button type="button" className="btn-icon" aria-label={"Nest " + label + " inside the record above"} disabled={!i || rows[i - 1].depth < r.depth || r.depth >= 4} onClick={() => setRows(shift(rows, i, 1))}><IndentIncrease size={16}/></button>
          <button type="button" className="btn-icon" aria-label={"Remove " + label} onClick={() => setRows(removeRow(rows, i))}><Trash2 size={16}/></button>
        </li>;
      })}</ol>
      {!!childTypes(schema, type.id).length && <button type="button" className="btn btn-soft btn-sm template-add-row" onClick={add}><Plus size={15}/>Add a record inside</button>}
    </section>
    {(!!issues.length || error) && <div role="alert" className="template-error">{[...issues, error].filter(Boolean).slice(0, 4).map((m, i) => <p key={i}>{m}</p>)}</div>}
    <div className="dialog-actions">
      <button type="button" className="btn" disabled={busy} onClick={onClose}>Cancel</button>
      <button type="button" className="btn btn-primary" disabled={busy || !name.trim() || !!issues.length} onClick={() => void save().catch(() => {})}>Save template</button>
    </div>
  </Dialog>;
}

// ---- Save as template (record menu) -----------------------------------------------------
export function SaveAsTemplateDialog({ row, type, hasContents, onClose, onSaved }: {
  row: CustomRecord; type: SchemaType; hasContents: boolean; onClose: () => void; onSaved: (t: RecordTemplate) => void;
}) {
  const [name, setName] = useState(row.title), [contents, setContents] = useState(hasContents);
  const { run, busy, error } = useStructureActions();
  return <Dialog className="dialog template-dialog" aria-label="Save as template" onEscape={busy ? undefined : onClose}>
    <div className="dialog-heading"><h2>Save as template</h2><button type="button" className="btn-icon" aria-label="Close" disabled={busy} onClick={onClose}><X size={18}/></button></div>
    <p className="template-sub">New {type.plural.toLowerCase()} can start from a copy of {row.title}.</p>
    <label>Template name<input value={name} maxLength={120} autoFocus onChange={e => setName(e.target.value)}/></label>
    {hasContents && <label className="check-label"><input type="checkbox" checked={contents} onChange={e => setContents(e.target.checked)}/>Include the records inside</label>}
    <p className="footnote">Keeps the title, details and plain field values. Dates, assignees, status and links to other records are left out, so templates never carry deadlines.</p>
    {error && <p role="alert" className="template-error">{error}</p>}
    <div className="dialog-actions">
      <button type="button" className="btn" disabled={busy} onClick={onClose}>Cancel</button>
      <button type="button" className="btn btn-primary" disabled={busy || !name.trim()} onClick={() => void run<RecordTemplate>("template.capture", { record_id: row.id, name: name.trim(), include_children: contents }).then(onSaved).catch(() => {})}>Save template</button>
    </div>
  </Dialog>;
}

// ---- Types & fields: Templates section ---------------------------------------------------
/** Templates live outside the schema: edits here save immediately, with no structure preview. */
export function TemplatesSection({ type, saved }: { type: SchemaType; saved: Schema }) {
  const { templates, reload } = useTemplates();
  const [editing, setEditing] = useState<{ template?: RecordTemplate; initial?: ReturnType<typeof onboardingExample> } | null>(null);
  const { run, busy, error } = useStructureActions();
  const applied = saved.types.some(t => t.id === type.id && !t.archived);
  const mine = templates.filter(t => t.type_id === type.id);
  const example = onboardingExample(saved, type.id);
  return <section className="schema-block template-section" aria-label="Templates">
    <div className="schema-block-head"><h3>Templates</h3>{!!mine.length && <span className="detail-count">{mine.length}</span>}
      {applied && <button type="button" className="btn btn-soft btn-sm" onClick={() => setEditing({})}><Plus size={15}/>New template</button>}</div>
    <p className="schema-block-hint">Starting structures for new {type.plural.toLowerCase()}. They save right away and never change records already made.</p>
    {!applied ? <p className="detail-empty">Apply this type first, then add templates.</p>
      : !mine.length ? <div className="empty template-empty"><span>No templates yet. Save a record as a template from its menu, or start from an example.</span>
        {example && <button type="button" className="btn btn-soft btn-sm" onClick={() => setEditing({ initial: example })}><LayoutTemplate size={15}/>Try the Client onboarding example</button>}</div>
      : <ul className="template-list">{mine.map(t => <li key={t.id} className="template-list-row">
        <div><strong>{t.name}</strong><small>{t.summary}</small></div>
        <button type="button" className="btn btn-ghost btn-sm" onClick={() => setEditing({ template: t })}>Edit</button>
        <button type="button" className="btn btn-ghost btn-sm" disabled={busy} aria-label={"Archive template " + t.name}
          onClick={() => void run("template.archive", { template_id: t.id, expected_revision: t.revision }).then(reload).catch(() => {})}>Archive</button>
      </li>)}</ul>}
    {error && <p role="alert" className="template-error">{error}</p>}
    {editing && <TemplateEditor schema={saved} typeId={type.id} template={editing.template} initial={editing.initial ?? undefined}
      onClose={() => setEditing(null)} onSaved={() => { setEditing(null); reload(); }}/>}
  </section>;
}
