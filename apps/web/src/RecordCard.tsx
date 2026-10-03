import { useEffect, useRef, useState, type ReactNode } from "react";
import { z } from "zod";
import { SourceDetails } from "./SourceDetails";
import { HomePicker } from "./HomePicker";
import { ContentsDialog } from "./ContentsDialog";
import type {ContentsSummary} from "./structure-types";
import { RecordTools } from "./record-links";
import { useEditor } from "./editor-control";
import { Link2, X } from "lucide-react";
import { api } from "./api";
import { Dialog, priorityLabels } from "./ux";
import { useStructureActions } from "./structure-actions";
import type { Schema, SchemaField, CustomRecord } from "./structure-types";
import "./details.css";

/** One compact label / value row in a detail card's properties list. */
export function Prop({label, hint, children}: {label: string; hint?: string; children: ReactNode}) {
  return <div className="prop-row"><span className="prop-label" title={hint}>{label}</span><div className="prop-value">{children}</div></div>;
}

/** Small segmented flag control for 0–3 priority. */
export function PriorityControl({value, disabled, onChange}: {value: number; disabled: boolean; onChange: (value: number) => void}) {
  return <div className="segmented priority-control" role="group" aria-label="Priority">
    {priorityLabels.map((name, level) => <button key={name} type="button" data-level={level} aria-pressed={value === level}
      disabled={disabled} onClick={() => { if (value !== level) onChange(level); }}>
      {name}</button>)}
  </div>;
}

function FieldValue({field,value,choices,disabled,onSave,className}:{field:SchemaField;value:unknown;choices:CustomRecord[];disabled:boolean;onSave:(v:unknown)=>void;className?:string}){
  const [draft,setDraft]=useState(value===null||value===undefined?"":String(value));const focused=useRef(false);
  useEffect(()=>{if(!focused.current)setDraft(value===null||value===undefined?"":String(value));},[value]);
  const base={disabled,"aria-label":field.name,className};
  if(field.kind==="boolean")return <label className="prop-switch"><input {...base} className="switch" type="checkbox" role="switch" checked={Boolean(value)} onChange={e=>onSave(e.target.checked)}/>{value?"Yes":"No"}</label>;
  const options=field.kind==="relation"?choices.filter(r=>field.target_types.includes(r.type_id)).map(r=>({id:r.id,name:r.title})):field.options;
  if(["select","multiselect","relation"].includes(field.kind)){const multi=field.kind==="multiselect"||field.multiple;return <select {...base} multiple={multi} value={multi?(Array.isArray(value)?value as string[]:[]):String(value??"")} onChange={e=>onSave(multi?Array.from(e.target.selectedOptions,o=>o.value):e.target.value||null)}>{!multi&&<option value="">Empty</option>}{options.map(o=><option value={o.id} key={o.id}>{o.name}</option>)}</select>;}
  const save=()=>{focused.current=false;const next=draft===""?null:field.kind==="number"?Number(draft):draft;if(next!==value)onSave(next);};
  const events={onFocus:()=>{focused.current=true;},onBlur:save};
  if(field.kind==="long_text")return <textarea {...base} {...events} value={draft} rows={2} placeholder="Empty" onChange={e=>setDraft(e.target.value)}/>;
  const type=field.binding==="due_time"?"time":field.kind==="date"?"date":field.kind==="datetime"?"datetime-local":field.kind==="number"?"number":"text";
  const input=<input {...base} {...events} type={type} value={draft} data-empty={draft===""} placeholder={field.binding==="due_timezone"?"Time zone":"Empty"}
    min={field.binding==="estimate_minutes"?1:undefined} onChange={e=>setDraft(e.target.value)}
    onKeyDown={e=>{if(e.key==="Enter")e.currentTarget.blur();if(e.key==="Escape"){e.preventDefault();e.stopPropagation();setDraft(value===null||value===undefined?"":String(value));focused.current=false;}}}/>;
  return field.binding==="estimate_minutes"?<span className="prop-suffix">{input}<span aria-hidden="true">min</span></span>:input;
}

const DUE = ["due_date","due_time","due_timezone"];

export function RecordCard({schema,initial,choices:initialChoices,canEdit,onClose,onChanged,onOpen}:{schema:Schema;initial:CustomRecord;choices:CustomRecord[];canEdit:boolean;onClose:()=>void;onChanged:()=>Promise<void>;onOpen?:(r:CustomRecord)=>void}){
  const [choices,setChoices]=useState(initialChoices),[relatedQuery,setRelatedQuery]=useState("");
  const [contents,setContents]=useState<ContentsSummary|undefined>(initial.contents);
  const [children,setChildren]=useState<CustomRecord[]>([]),[childOffset,setChildOffset]=useState(0),[childNext,setChildNext]=useState<number|null>(null);
  const [contentsAction,setContentsAction]=useState<{operation:"move"|"archive";parentId?:string|null}|null>(null);
  const [row,setRow]=useState(initial);const [body,setBody]=useState(initial.body);const [title,setTitle]=useState(initial.title);const [relation,setRelation]=useState("");const [target,setTarget]=useState("");const {run,busy,error,setError}=useStructureActions();
  const [localNotes,setLocalNotes]=useState(initial.local_notes??"");
  const [failed,setFailed]=useState<Record<string,unknown>|null>(null);
  const [comparison,setComparison]=useState<CustomRecord|null>(null);
  const current=useRef(row);current.current=row;const flight=useRef<Promise<unknown>|null>(null);
  const [alert,setAlert]=useState<{revision:number;deadline_alert:string;alert_urgent:boolean}|null>(null);
  useEffect(()=>{if(row.task_id)void api<{revision:number;deadline_alert:string;alert_urgent:boolean}>("/tasks/"+row.task_id).then(setAlert);},[row.task_id,row.revision]);
  const alertSave=async(values:Record<string,unknown>)=>{if(!alert||!row.task_id)return;const task=await run<{revision:number;deadline_alert:string;alert_urgent:boolean}>("task.update",{task_id:row.task_id,expected_revision:alert.revision,...values});setAlert(task);setRow(await api<CustomRecord>("/structure/records/"+row.id));await onChanged();};
  useEffect(()=>{let live=true;void api<CustomRecord>("/structure/records/"+row.id).then(r=>live&&setContents(r.contents)).catch(()=>{});
    void api<{items:CustomRecord[];next_offset:number|null}>("/structure/browse?parent_id="+row.id+"&limit=30&offset="+childOffset).then(r=>{if(live){setChildren(r.items);setChildNext(r.next_offset);}}).catch(()=>{});
    return()=>{live=false;};
  },[row.id,row.revision,childOffset]);
  useEffect(()=>{let live=true;const timer=setTimeout(()=>{void (async()=>{
    const page=await api<{items:CustomRecord[]}>("/structure/records?limit=50&query="+encodeURIComponent(relatedQuery));
    const ids=new Set(row.links.flatMap(l=>[l.source_id,l.target_id]));
    for(const f of schema.types.find(t=>t.id===row.type_id)?.fields??[])if(f.kind==="relation"){const v=row.values[f.id];for(const id of Array.isArray(v)?v:[v])if(typeof id==="string"&&id)ids.add(id);}
    const linked=await Promise.all([...ids].filter(id=>id!==row.id&&!page.items.some(r=>r.id===id)).map(id=>api<CustomRecord>("/structure/records/"+id).catch(()=>null)));
    if(live)setChoices([...page.items,...linked.filter((r):r is CustomRecord=>r!==null)]);
  })().catch(e=>live&&setError(e.message));},200);return()=>{live=false;clearTimeout(timer);};},[row.id,row.revision,relatedQuery,schema,setError]);
  const t=schema.types.find(t=>t.id===row.type_id)!;
  const save=async(changes:Record<string,unknown>)=>{
    if(changes.status_id&&t.statuses.find(s=>s.id===changes.status_id)?.meaning==="completed"&&contents?.work_open){
      if(!window.confirm("Complete only this record? "+contents.work_open+" unfinished descendants will stay open."))return current.current;
    }
    if(flight.current)await flight.current;
    const op=run<CustomRecord>("record.update",{record_id:current.current.id,expected_revision:current.current.revision,schema_revision:schema.revision,...changes});flight.current=op;
    try{const result=await op;current.current=result;setRow(result);setTitle(result.title);setBody(result.body);setLocalNotes(result.local_notes??"");setFailed(null);setComparison(null);await onChanged();return result;}catch(e){setFailed(changes);throw e;}finally{flight.current=null;}
  };
  const finish=async()=>{if(document.activeElement instanceof HTMLElement)document.activeElement.blur();await new Promise(r=>setTimeout(r,0));if(flight.current)await flight.current;if(error)throw new Error(error);};
  const close=()=>void finish().then(onClose).catch(()=>{});
  useEditor({kind:"record",record_id:row.id,mode:"detail",auto_save:true,dirty:!!failed||title!==row.title||body!==row.body||localNotes!==(row.local_notes??""),busy,
    schema:z.object({title:z.string().min(1).max(500),body:z.string().max(30000),local_notes:z.string().max(30000),parent_id:z.string().nullable(),status_id:z.string().nullable(),values:z.record(z.string(),z.unknown()),archived:z.boolean()}).partial(),
    values:{title:row.title,body:row.body,local_notes:row.local_notes??"",parent_id:row.parent_id,status_id:row.status_id,values:row.values,archived:row.archived},beforeLeave:finish,patch:async v=>{if(!canEdit)throw new Error("Read-only workspace");await finish();return save(v);},close:onClose});
  const link=async(remove=false,relationship_id=relation,target_id=target)=>{await run("record.link",{source_id:row.id,target_id,relationship_id,expected_revision:row.revision,schema_revision:schema.revision,remove});setRow(await api<CustomRecord>("/structure/records/"+row.id));await onChanged();setTarget("");};
  useEffect(()=>{const key=(e:KeyboardEvent)=>{if(e.key==="Escape"&&!busy&&!e.defaultPrevented)close();};document.addEventListener("keydown",key);return()=>document.removeEventListener("keydown",key);},[busy,onClose]);
  const locked=!canEdit||busy||!!failed;
  const fields=t.fields.filter(f=>f.visible&&!f.archived);
  const byBinding=(binding:string)=>fields.find(f=>f.binding===binding);
  const field=(f:SchemaField,className?:string)=><FieldValue key={f.id} className={className} field={f} value={row.values[f.id]} choices={choices} disabled={locked} onSave={v=>void save({values:{[f.id]:v}}).catch(()=>{})}/>;
  const inherited=(f:SchemaField)=><>{f.inherit&&!row.inherited[f.id]&&canEdit&&<span className="prop-note"><button className="text-button" onClick={()=>void save({reset_fields:[f.id]}).catch(()=>{})}>Use inherited value</button></span>}
    {row.inherited[f.id]&&<span className="prop-note">From {choices.find(r=>r.id===row.inherited[f.id])?.title??"main home"}</span>}</>;
  const due=DUE.map(byBinding).filter((f):f is SchemaField=>!!f);
  const priority=byBinding("priority");
  const rest=fields.filter(f=>!DUE.includes(f.binding??"")&&f!==priority);
  const home=row.home.map(h=>h.title).join(" / ");
  const links=row.links.map(l=>({l,other:l.source_id===row.id?l.target_id:l.source_id}));
  return <Dialog className="detail-card" aria-label={t.name+" details"} onBackdrop={()=>{if(!busy)close();}}>
    <header className="detail-head">
      <span className="detail-kind">{t.name}</span>
      {home&&<span className="chip chip-home detail-path" title={home}>{home}</span>}
      <span className={"detail-save-state"+(busy?" busy":error?" attention":"")} aria-live="polite">{busy?"Saving…":error?"Changes need attention":canEdit?"Changes save automatically":"View only"}</span>
      <button className="btn-icon" aria-label="Close record" disabled={busy} onClick={close}><X size={20}/></button>
    </header>
    {error&&<div role="alert" className="detail-alert"><p>{error} Your draft is kept here.</p>
      <button className="btn btn-sm" onClick={()=>void api<CustomRecord>("/structure/records/"+row.id).then(setComparison).catch(e=>setError(e.message))}>Compare saved version</button>
      {comparison&&<><details open><summary>Saved version</summary><strong>{comparison.title}</strong><p>{comparison.body}</p><pre>{JSON.stringify({values:comparison.values,local_notes:comparison.local_notes,parent_id:comparison.parent_id,status_id:comparison.status_id},null,2)}</pre></details>
        <button className="btn btn-sm" onClick={()=>{setRow(comparison);current.current=comparison;setTitle(comparison.title);setBody(comparison.body);setLocalNotes(comparison.local_notes??"");setError("");setFailed(null);setComparison(null);}}>Use saved version</button>
        {failed&&canEdit&&comparison.schema_revision===schema.revision&&<button className="btn btn-sm" onClick={()=>{current.current=comparison;void save(failed).catch(()=>{});}}>Reapply my change</button>}</>}
    </div>}
    <div className="detail-grid">
      <div className="detail-main">
        <textarea rows={1} className="detail-title" aria-label="Record title" value={title} disabled={locked} onChange={e=>setTitle(e.target.value)} onBlur={()=>{if(title.trim()&&title!==row.title)void save({title}).catch(()=>{});}}/>
        <textarea aria-label="Record content" className="detail-body" value={body} disabled={locked} rows={4} placeholder="Add details" onChange={e=>setBody(e.target.value)} onBlur={()=>{if(body!==row.body)void save({body}).catch(()=>{});}}/>
        <details className="detail-section"><summary>Eridani-only notes</summary><p className="footnote">Visible in this workspace. Never sent to connected services.</p><textarea aria-label="Eridani-only notes" value={localNotes} disabled={locked} rows={3} onChange={e=>setLocalNotes(e.target.value)} onBlur={()=>{if(localNotes!==(row.local_notes??""))void save({local_notes:localNotes}).catch(()=>{});}}/></details>
        <SourceDetails source={row.source}/>
        <RecordTools kind="record" id={row.id}/>
        {contents&&contents.work_total>0&&<p className="detail-progress">{contents.work_done}/{contents.work_total} descendants completed{contents.ready_to_complete?" · Ready for your final check":""}. Completing this record leaves child statuses unchanged.</p>}
        {!!row.blockers?.length&&<p className="detail-alert">Unfinished prerequisites: {row.blockers.map(b=>b.title).join(", ")}. You can still record progress.</p>}
        {(children.length>0||childOffset>0)&&<section className="detail-section" aria-label="Work in this home"><h3>Direct contents</h3>{children.map(r=><div className="detail-link-row" key={r.id}><button className="text-button" onClick={()=>void finish().then(()=>onOpen?.(r)).catch(()=>{})}>{r.title}</button><span className="chip">{r.type_name}</span><span>{r.status_meaning?.replaceAll("_"," ")}</span></div>)}
          <div className="org-pagination">{childOffset>0&&<button className="btn btn-sm" onClick={()=>setChildOffset(Math.max(0,childOffset-30))}>Previous</button>}{childNext!==null&&<button className="btn btn-sm" onClick={()=>setChildOffset(childNext)}>More contents</button>}</div></section>}
        <details className="detail-section" aria-label="Related records"><summary>Extra relationships{links.length?" · "+links.length:""}</summary>

          <label className="field">Find related records<input aria-label="Find related records" value={relatedQuery} onChange={e=>setRelatedQuery(e.target.value)} placeholder="Search names (first 50 matches)"/></label>
          <div className="detail-section-head"><h3>Related records</h3>{!!links.length&&<span className="detail-count">{links.length}</span>}</div>
          {!links.length&&<p className="detail-empty">No related records yet.</p>}
          {links.map(({l,other})=><div className="detail-link-row" key={l.id}><Link2 size={15}/>
            <button className="text-button detail-link-title" onClick={()=>{const next=choices.find(r=>r.id===other);if(next)void finish().then(()=>onOpen?.(next)).catch(()=>{});}}>{choices.find(r=>r.id===other)?.title??"Linked record"}</button>
            <span className="chip">{(schema.relationships.find(r=>r.id===l.relationship_id)?.behavior==="blocks"&&l.target_id===row.id?"Blocked by":schema.relationships.find(r=>r.id===l.relationship_id)?.name)}</span>
            {canEdit&&l.source_id===row.id&&<span className="row-actions"><button className="btn-icon" aria-label="Remove link" onClick={()=>void link(true,l.relationship_id,other).catch(()=>{})}><X size={14}/></button></span>}
          </div>)}
          {canEdit&&<div className="detail-add-link"><select aria-label="Relationship" value={relation} onChange={e=>{setRelation(e.target.value);setTarget("");}}><option value="">Add a relationship</option>{schema.relationships.filter(r=>!r.archived&&r.source_types.includes(row.type_id)).map(r=><option key={r.id} value={r.id}>{r.name}</option>)}</select>
            {relation&&<><select aria-label="Related record" value={target} onChange={e=>setTarget(e.target.value)}><option value="">Choose a record</option>{choices.filter(r=>schema.relationships.find(l=>l.id===relation)?.target_types.includes(r.type_id)).map(r=><option key={r.id} value={r.id}>{r.title}</option>)}</select><button className="btn btn-soft" disabled={!target||busy} onClick={()=>void link().catch(()=>{})}>Link</button></>}</div>}
        </details>
      </div>
      <aside className="detail-props" aria-label={t.name+" properties"}>
        <div className="prop-list">
          <Prop label="Main home"><HomePicker row={row} schema={schema} disabled={locked} onChoose={id=>{if(id!==row.parent_id)setContentsAction({operation:"move",parentId:id});}}/></Prop>
          {!!t.statuses.length&&<Prop label="Status"><select aria-label="Status" value={row.status_id??""} disabled={locked} onChange={e=>void save({status_id:e.target.value||null}).catch(()=>{})}>{!t.capabilities.includes("work")&&<option value="">Empty</option>}{t.statuses.map(s=><option key={s.id} value={s.id}>{s.name}</option>)}</select></Prop>}
          {!!due.length&&<Prop label="Due" hint={due.map(f=>f.description).join(" ")}><div className="prop-inline">
            {due.map(f=>field(f,f.binding==="due_date"?"prop-date":f.binding==="due_time"?"prop-time":"prop-zone"))}</div>{due.map(f=><span key={f.id}>{inherited(f)}</span>)}</Prop>}
          {priority&&<Prop label={priority.name} hint={priority.description}><PriorityControl value={Number(row.values[priority.id]??0)} disabled={locked} onChange={v=>void save({values:{[priority.id]:v}}).catch(()=>{})}/>{inherited(priority)}</Prop>}
          {rest.map(f=><Prop key={f.id} label={f.name} hint={f.description}>{field(f)}{inherited(f)}</Prop>)}
          {alert&&<><div className="prop-divider"/>
            <Prop label="Deadline alert"><select aria-label="Deadline alert" value={alert.deadline_alert} disabled={locked} onChange={e=>void alertSave({deadline_alert:e.target.value}).catch(()=>{})}><option value="default">Use my setting</option><option value="on">On</option><option value="off">Off</option></select></Prop>
            <Prop label="Urgent"><label className="prop-switch"><input type="checkbox" className="switch" role="switch" aria-label="Urgent alert" checked={alert.alert_urgent} disabled={locked} onChange={e=>void alertSave({alert_urgent:e.target.checked}).catch(()=>{})}/><small>Bypasses quiet hours</small></label></Prop></>}
        </div>
        {canEdit&&<div className="detail-props-foot"><button className="btn btn-ghost btn-sm" disabled={busy} onClick={()=>{if(row.archived)void save({archived:false}).then(onClose).catch(()=>{});else setContentsAction({operation:"archive"});}}>{row.archived?"Restore":"Archive"}</button></div>}
      </aside>
    </div>
    {contentsAction&&<ContentsDialog row={row} schema={schema} {...contentsAction} onClose={()=>setContentsAction(null)} onDone={async()=>{await onChanged();if(contentsAction.operation==="archive")onClose();else{const next=await api<CustomRecord>("/structure/records/"+row.id);setRow(next);current.current=next;}}}/>}
  </Dialog>;
}
