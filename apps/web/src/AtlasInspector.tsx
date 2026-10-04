import { useEffect, useState } from "react";
import { FolderOpen, Info, Plus, X } from "lucide-react";
import { api } from "./api";
import { HomePicker } from "./HomePicker";
import { OPENS_AS } from "./TypeEditor";
import { capabilityPatch, updateType } from "./field-library";
import { ancestors, dueState, isRegion, itemState, moveChoices, stateLabel, type AtlasIndex, type AtlasRecord, type MoveOption } from "./atlas-model";
import {
  CAPABILITY_GLYPHS, addAllowedHome, diagramParent, fieldRole, fieldRoleLabel, recordsLivingIn, removeAllowedHome, typeMoveChoice,
} from "./blueprint-model";
import type { CustomRecord, Schema } from "./structure-types";
import { TemplateStamps, TemplateStart } from "./Templates";
import { templatesFor, type InstantiatePreview, type RecordTemplate } from "./template-model";

/** Template stamps in Add here: preview, then create through record.instantiate. */
export type TemplateHooks = { templates: RecordTemplate[]; onTemplate: (preview: InstantiatePreview, title: string) => Promise<void> };
import { shortDate } from "./work-views";

const sentence = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

/** Two contents choices, used after a drop (as a popover) and after the keyboard Move to… picker. */
export function MoveChoiceList({ title, options, busy, onChoose, onCancel }: { title: string; options: MoveOption[]; busy: boolean; onChoose: (o: MoveOption) => void; onCancel: () => void }) {
  return <div className="atlas-choice" role="group" aria-label={title}>
    <p className="atlas-choice-title">{title}</p>
    {options.map(o => <button key={o.mode} type="button" className="atlas-choice-option" disabled={busy} onClick={() => onChoose(o)}><strong>{o.label}</strong><small>{o.note}</small></button>)}
    <button type="button" className="atlas-choice-option is-cancel" onClick={onCancel}><strong>Cancel</strong></button>
  </div>;
}

export function RecordInspector({ record, index, schema, today, canEdit, focused, busy, onFly, onSelect, onOpen, onBrowse, onType, onMove, onCreate, templates }: {
  record: AtlasRecord; index: AtlasIndex; schema: Schema; today: string; canEdit: boolean; focused: boolean; busy: boolean; templates?: TemplateHooks;
  onFly: (id: string) => void; onSelect: (id: string) => void; onOpen: (id: string) => void; onBrowse: (id: string) => void; onType: (id: string) => void;
  onMove: (record: AtlasRecord, target: string | null, mode: MoveOption["mode"]) => Promise<void>; onCreate: (type: string, title: string, parent: string | null) => Promise<void>;
}) {
  const type = index.types.get(record.type_id);
  const [full, setFull] = useState<CustomRecord | null>(null);
  const [moving, setMoving] = useState<{ id: string | null; title: string } | null>(null);
  useEffect(() => { let live = true; setFull(null); setMoving(null); void api<CustomRecord>("/structure/records/" + record.id).then(r => live && setFull(r)).catch(() => {}); return () => { live = false; }; }, [record.id, record.revision]);
  const state = itemState(record, today), due = dueState(record, today);
  const kids = index.children.get(record.id) ?? [], p = index.progress.get(record.id);
  const path = ancestors(index, record.id);
  return <>
    <div className="atlas-kicker">
      <span className="chip chip-type">{type?.name ?? "Record"}</span>
      {state && <span className={"chip" + (state === "done" ? " chip-done" : state === "overdue" ? " chip-due-overdue" : "")}>{sentence(stateLabel[state])}</span>}
      {record.due_date && state !== "done" && <span className={"chip" + (due === "overdue" ? " chip-due-overdue" : due === "today" ? " chip-due-today" : "")}>Due {shortDate(record.due_date, today)}</span>}
    </div>
    <h2 className="atlas-title">{record.title}</h2>
    <p className="atlas-path">{path.length ? path.map((a, i) => <span key={a.id}>{i > 0 && <span aria-hidden="true" className="atlas-path-sep">/</span>}<button type="button" className="text-button" onClick={() => onFly(a.id)}>{a.title}</button></span>) : <span>Unfiled</span>}</p>
    <dl className="atlas-facts">
      <div><dt>Opens</dt><dd>{isRegion(record) ? "As its contents" : "As a detail card"}</dd></div>
      {!!kids.length && <div><dt>Inside</dt><dd>{kids.length} {kids.length === 1 ? "record" : "records"}</dd></div>}
      {!!p?.total && <div><dt>Work done</dt><dd className="tabular">{p.done} of {p.total}</dd></div>}
      <RelationRows record={record} index={index} onSelect={onSelect}/>
    </dl>
    <div className="atlas-actions">
      {isRegion(record) && !focused && <button type="button" className="btn btn-soft btn-sm" onClick={() => onFly(record.id)}>Fly in</button>}
      {isRegion(record) && <button type="button" className="btn btn-sm" onClick={() => onBrowse(record.id)}><FolderOpen size={15} aria-hidden="true"/>Open in Browse</button>}
      <button type="button" className="btn btn-sm" onClick={() => onOpen(record.id)}><Info size={15} aria-hidden="true"/>Details</button>
      <button type="button" className="btn btn-ghost btn-sm" onClick={() => onType(record.type_id)}>Type: {type?.name}</button>
    </div>
    {canEdit && <section className="atlas-section" aria-label="Move">
      <h3>Home</h3>
      {full ? <HomePicker row={full} schema={schema} disabled={busy} onChoose={id => setMoving({ id, title: id ? (index.byId.get(id)?.title ?? "the chosen home") : "Unfiled" })}/> : <span className="atlas-muted">Loading…</span>}
      {moving && <MoveChoiceList title={`Move ${record.title} to ${moving.title}`} options={moveChoices(record, index)} busy={busy}
        onCancel={() => setMoving(null)} onChoose={o => void onMove(record, moving.id, o.mode).then(() => setMoving(null))}/>}
    </section>}
    {canEdit && isRegion(record) && <AddHere parent={record} schema={schema} busy={busy} onCreate={onCreate} templates={templates}/>}
  </>;
}

function RelationRows({ record, index, onSelect }: { record: AtlasRecord; index: AtlasIndex; onSelect: (id: string) => void }) {
  return <>{index.links.filter(l => l.source_id === record.id || l.target_id === record.id).map(l => {
    const outgoing = l.source_id === record.id, other = index.byId.get(outgoing ? l.target_id : l.source_id);
    if (!other) return null;
    const label = l.behavior === "blocks" ? (outgoing ? "Blocks" : "Blocked by") : outgoing ? l.label : l.label + " (from)";
    return <div key={l.id}><dt>{label}</dt><dd><button type="button" className="text-button" onClick={() => onSelect(other.id)}>{other.title}</button></dd></div>;
  })}</>;
}

export function AddHere({ parent, schema, busy, onCreate, templates }: { parent: AtlasRecord | null; schema: Schema; busy: boolean; onCreate: (type: string, title: string, parent: string | null) => Promise<void>; templates?: TemplateHooks }) {
  const [kind, setKind] = useState(""), [title, setTitle] = useState("");
  const [stamp, setStamp] = useState<RecordTemplate | null>(null);
  useEffect(() => { setStamp(null); setKind(""); }, [parent?.id]);
  const types = schema.types.filter(t => !t.archived && (parent ? t.parent_types.includes(parent.type_id) : (t.opens_as ?? "auto") === "container"));
  const chosen = types.find(t => t.id === kind);
  if (!types.length) return null;
  return <section className="atlas-section" aria-label="Add here">
    <h3>Add here</h3>
    <div className="atlas-add-types">{types.map(t => <button key={t.id} type="button" className={"btn btn-sm" + (kind === t.id ? " btn-soft" : "")} aria-pressed={kind === t.id} onClick={() => setKind(kind === t.id ? "" : t.id)}><Plus size={14} aria-hidden="true"/>{t.name}</button>)}</div>
    {chosen && <form className="atlas-add" onSubmit={e => { e.preventDefault(); if (!title.trim()) return; void onCreate(chosen.id, title.trim(), parent?.id ?? null).then(() => { setTitle(""); setKind(""); }); }}>
      <input aria-label={"New " + chosen.name.toLowerCase() + " title"} placeholder={"New " + chosen.name.toLowerCase()} value={title} maxLength={500} autoFocus onChange={e => setTitle(e.target.value)}/>
      <button className="btn btn-primary btn-sm" disabled={busy || !title.trim()}>Add</button>
    </form>}
    {templates && (stamp
      ? <TemplateStart template={stamp} parentId={parent?.id ?? null} homeTitle={parent?.title} schemaRevision={schema.revision}
          onCancel={() => setStamp(null)} onCreate={async (preview, name) => { await templates.onTemplate(preview, name); setStamp(null); }}/>
      : <TemplateStamps disabled={busy} templates={templatesFor(templates.templates, kind ? [kind] : types.map(t => t.id))} onPick={setStamp}/>)}
  </section>;
}

export function TypeInspector({ typeId, draft, saved, records, index, canDesign, onDraft, onFly, onEditTypes, onPlace }: {
  typeId: string; draft: Schema; saved: Schema; records: AtlasRecord[]; index: AtlasIndex; canDesign: boolean;
  onDraft: (s: Schema) => void; onFly: (id: string) => void; onEditTypes: (type: string) => void; onPlace: (type: string, target: string) => void;
}) {
  const t = draft.types.find(x => x.id === typeId);
  const [blocked, setBlocked] = useState<{ home: string; records: AtlasRecord[] } | null>(null);
  useEffect(() => setBlocked(null), [typeId]);
  if (!t) return <p className="atlas-muted">This type is no longer in the draft.</p>;
  const name = (id: string) => draft.types.find(x => x.id === id)?.name ?? id;
  const plural = (id: string) => (draft.types.find(x => x.id === id)?.plural ?? id).toLowerCase();
  const count = records.filter(r => r.type_id === t.id).length;
  const remove = (home: string) => {
    const living = recordsLivingIn(records, t.id, home);
    if (living.length) { setBlocked({ home, records: living }); return; }
    setBlocked(null); onDraft(removeAllowedHome(draft, t.id, home));
  };
  const parent = diagramParent(draft, t.id);
  const fields = t.fields.filter(f => !f.archived);
  return <>
    <div className="atlas-kicker"><span className="chip chip-type">Type</span><span className="chip">{count} {count === 1 ? "record" : "records"}</span></div>
    <h2 className="atlas-title">{t.name}</h2>
    {t.description && <p className="atlas-muted">{t.description}</p>}
    <section className="atlas-section" aria-label="Behaviors">
      <h3>Behaviors</h3>
      <div className="atlas-toggles">{CAPABILITY_GLYPHS.map(([c, g, label]) => { const on = t.capabilities.includes(c);
        return <button key={c} type="button" className={"btn btn-sm" + (on ? " btn-soft" : "")} aria-pressed={on} disabled={!canDesign}
          onClick={() => onDraft(updateType(draft, t.id, capabilityPatch(t, draft, saved, c, !on)))}><span aria-hidden="true" className="atlas-glyph-text">{g}</span>{label}</button>; })}</div>
      <div className="atlas-subhead">Organizing container</div>
      <div className="segmented atlas-opens" role="group" aria-label="Organizing container">{OPENS_AS.map(([id, label]) =>
        <button key={id} type="button" disabled={!canDesign} aria-pressed={(t.opens_as ?? "auto") === id} onClick={() => onDraft(updateType(draft, t.id, { opens_as: id }))}>{label}</button>)}</div>
      <p className="atlas-hint">Behaviors switch features on. Plain information, like industry, stays a field.</p>
    </section>
    <section className="atlas-section" aria-label="Allowed homes">
      <h3>Allowed homes</h3>
      <div className="chip-row">{t.parent_types.map(h => <span key={h} className={"chip atlas-home-chip" + (h === parent ? " is-diagram" : "")}>
        {name(h)}{h === parent && <span className="atlas-chip-note">diagram</span>}
        {canDesign && <button type="button" aria-label={"Remove allowed home " + name(h)} onClick={() => remove(h)}><X size={12} aria-hidden="true"/></button>}
      </span>)}{!t.parent_types.length && <span className="atlas-muted">Top level only</span>}</div>
      {blocked && <div className="atlas-impact" role="alert">
        <strong>{blocked.records.length} {blocked.records.length === 1 ? t.name.toLowerCase() + " lives" : (t.plural || t.name).toLowerCase() + " live"} in {plural(blocked.home)} today</strong>
        <ul>{blocked.records.slice(0, 12).map(r => <li key={r.id}><button type="button" className="text-button" onClick={() => onFly(r.id)}>{r.title}</button>
          <span className="atlas-muted"> in {index.byId.get(r.parent_id!)?.title}</span></li>)}</ul>
        {blocked.records.length > 12 && <span>and {blocked.records.length - 12} more</span>}
        <span>Move {blocked.records.length === 1 ? "it" : "them"} first, or keep this home. Nothing changes until you apply.</span>
      </div>}
      {canDesign && <div className="atlas-row-controls">
        <label className="atlas-inline-field">Allow inside<select aria-label={"Allow " + (t.plural || t.name).toLowerCase() + " inside"} value="" onChange={e => { if (e.target.value) onDraft(addAllowedHome(draft, t.id, e.target.value)); }}>
          <option value="">Choose a type…</option>{draft.types.filter(x => !x.archived && !t.parent_types.includes(x.id)).map(x => <option key={x.id} value={x.id}>{x.name}</option>)}</select></label>
        <label className="atlas-inline-field">Diagram parent<select aria-label="Place in diagram under" value={parent ?? ""} onChange={e => { if (e.target.value) onPlace(t.id, e.target.value); }}>
          <option value="">Top level</option>{draft.types.filter(x => !x.archived && x.id !== t.id).map(x => <option key={x.id} value={x.id} disabled={typeMoveChoice(draft, t.id, x.id).cycle}>{x.name}</option>)}</select></label>
      </div>}
    </section>
    <details className="atlas-section atlas-fields" open>
      <summary><h3>Fields</h3><span className="detail-count">{fields.length}</span></summary>
      <ul>{fields.map(f => <li key={f.id}><span>{f.name || "Untitled field"}</span><small className={"atlas-field-role is-" + fieldRole(f)}>{fieldRoleLabel[fieldRole(f)]}</small></li>)}</ul>
      {!fields.length && <p className="atlas-muted">Every record has a title and details.</p>}
    </details>
    {canDesign && <button type="button" className="btn btn-sm" onClick={() => onEditTypes(t.id)}>Edit all settings in Types & fields</button>}
  </>;
}
