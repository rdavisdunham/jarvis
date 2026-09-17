import { useEffect, useRef, useState } from "react";
import { z } from "zod";
import { RecordTools } from "./record-links";
import { useEditor } from "./editor-control";
import { useDialogFocus } from "./components";
import { X } from "lucide-react";
import { api } from "./api";
import { useBodyLock } from "./ux";
import { useStructureActions } from "./structure-actions";
import type { Schema, SchemaField, CustomRecord } from "./structure-types";

function FieldValue({field,value,choices,disabled,onSave}:{field:SchemaField;value:unknown;choices:CustomRecord[];disabled:boolean;onSave:(v:unknown)=>void}){
  const [draft,setDraft]=useState(value===null||value===undefined?"":String(value));const focused=useRef(false);
  useEffect(()=>{if(!focused.current)setDraft(value===null||value===undefined?"":String(value));},[value]);
  const base={disabled,"aria-label":field.name};
  if(field.kind==="boolean")return <input {...base} type="checkbox" checked={Boolean(value)} onChange={e=>onSave(e.target.checked)}/>;
  const options=field.kind==="relation"?choices.filter(r=>field.target_types.includes(r.type_id)).map(r=>({id:r.id,name:r.title})):field.options;
  if(["select","multiselect","relation"].includes(field.kind)){const multi=field.kind==="multiselect"||field.multiple;return <select {...base} multiple={multi} value={multi?(Array.isArray(value)?value as string[]:[]):String(value??"")} onChange={e=>onSave(multi?Array.from(e.target.selectedOptions,o=>o.value):e.target.value||null)}>{!multi&&<option value="">Unassigned</option>}{options.map(o=><option value={o.id} key={o.id}>{o.name}</option>)}</select>;}
  const save=()=>{focused.current=false;const next=draft===""?null:field.kind==="number"?Number(draft):draft;if(next!==value)onSave(next);};
  if(field.kind==="long_text")return <textarea {...base} value={draft} rows={3} onFocus={()=>{focused.current=true;}} onChange={e=>setDraft(e.target.value)} onBlur={save}/>;
  return <input {...base} type={field.kind==="date"?"date":field.kind==="datetime"?"datetime-local":field.kind==="number"?"number":"text"} value={draft} onFocus={()=>{focused.current=true;}} onChange={e=>setDraft(e.target.value)} onBlur={save} onKeyDown={e=>{if(e.key==="Enter")e.currentTarget.blur();if(e.key==="Escape"){e.preventDefault();e.stopPropagation();setDraft(value===null||value===undefined?"":String(value));focused.current=false;}}}/>;
}

export function RecordCard({schema,initial,choices,canEdit,onClose,onChanged}:{schema:Schema;initial:CustomRecord;choices:CustomRecord[];canEdit:boolean;onClose:()=>void;onChanged:()=>Promise<void>}){
  useBodyLock(true);useDialogFocus();const [row,setRow]=useState(initial);const [body,setBody]=useState(initial.body);const [title,setTitle]=useState(initial.title);const [relation,setRelation]=useState("");const [target,setTarget]=useState("");const {run,busy,error,setError}=useStructureActions();
  const current=useRef(row);current.current=row;const flight=useRef<Promise<unknown>|null>(null);
  const [alert,setAlert]=useState<{revision:number;deadline_alert:string;alert_urgent:boolean}|null>(null);
  useEffect(()=>{if(row.task_id)void api<{revision:number;deadline_alert:string;alert_urgent:boolean}>("/tasks/"+row.task_id).then(setAlert);},[row.task_id,row.revision]);
  const alertSave=async(values:Record<string,unknown>)=>{if(!alert||!row.task_id)return;const task=await run<{revision:number;deadline_alert:string;alert_urgent:boolean}>("task.update",{task_id:row.task_id,expected_revision:alert.revision,...values});setAlert(task);setRow(await api<CustomRecord>("/structure/records/"+row.id));await onChanged();};
  const t=schema.types.find(t=>t.id===row.type_id)!;
  const save=async(changes:Record<string,unknown>)=>{
    if(flight.current)await flight.current;
    const op=run<CustomRecord>("record.update",{record_id:current.current.id,expected_revision:current.current.revision,schema_revision:schema.revision,...changes});flight.current=op;
    try{const result=await op;current.current=result;setRow(result);setTitle(result.title);setBody(result.body);await onChanged();return result;}finally{flight.current=null;}
  };
  const finish=async()=>{if(document.activeElement instanceof HTMLElement)document.activeElement.blur();await new Promise(r=>setTimeout(r,0));if(flight.current)await flight.current;if(error)throw new Error(error);};
  const close=()=>void finish().then(onClose).catch(()=>{});
  useEditor({kind:"record",record_id:row.id,mode:"detail",auto_save:true,dirty:title!==row.title||body!==row.body,busy,
    schema:z.object({title:z.string().min(1).max(500),body:z.string().max(30000),parent_id:z.string().nullable(),status_id:z.string().nullable(),values:z.record(z.string(),z.unknown()),archived:z.boolean()}).partial(),
    values:{title:row.title,body:row.body,parent_id:row.parent_id,status_id:row.status_id,values:row.values,archived:row.archived},beforeLeave:finish,patch:async v=>{if(!canEdit)throw new Error("Read-only workspace");await finish();return save(v);},close:onClose});
  const link=async(remove=false,relationship_id=relation,target_id=target)=>{await run("record.link",{source_id:row.id,target_id,relationship_id,expected_revision:row.revision,schema_revision:schema.revision,remove});setRow(await api<CustomRecord>("/structure/records/"+row.id));await onChanged();setTarget("");};
  useEffect(()=>{const key=(e:KeyboardEvent)=>{if(e.key==="Escape"&&!busy&&!e.defaultPrevented)close();};document.addEventListener("keydown",key);return()=>document.removeEventListener("keydown",key);},[busy,onClose]);
  return <div className="modal-backdrop" onClick={e=>{if(e.target===e.currentTarget&&!busy)close();}}><section className="record-detail" role="dialog" aria-modal="true" aria-label={t.name+" details"}>
    <header><span className="subtle">{t.name}</span><button aria-label="Close record" disabled={busy} onClick={close}><X size={20}/></button></header>
    {error&&<div role="alert" className="error"><p>{error}</p><button onClick={()=>void api<CustomRecord>("/structure/records/"+row.id).then(r=>{setRow(r);current.current=r;setTitle(r.title);setBody(r.body);setError("");})}>Reload saved record</button></div>}
    <div className="record-detail-grid"><main><textarea rows={2} className="record-heading" aria-label="Record title" value={title} disabled={!canEdit||busy} onChange={e=>setTitle(e.target.value)} onBlur={()=>{if(title.trim()&&title!==row.title)void save({title}).catch(()=>{});}}/>
      <textarea aria-label="Record content" value={body} disabled={!canEdit||busy} rows={4} placeholder="Details…" onChange={e=>setBody(e.target.value)} onBlur={()=>{if(body!==row.body)void save({body}).catch(()=>{});}}/>
      <RecordTools kind="record" id={row.id}/>
      <h3>Related records</h3>{row.links.map(l=>{const other=l.source_id===row.id?l.target_id:l.source_id;return <div className="record-link" key={l.id}><span>{schema.relationships.find(r=>r.id===l.relationship_id)?.name}: {choices.find(r=>r.id===other)?.title??"Linked record"}</span>{canEdit&&l.source_id===row.id&&<button aria-label="Remove link" onClick={()=>void link(true,l.relationship_id,other).catch(()=>{})}><X size={14}/></button>}</div>;})}
      {canEdit&&<div className="record-link-create"><select aria-label="Relationship" value={relation} onChange={e=>{setRelation(e.target.value);setTarget("");}}><option value="">Add relationship…</option>{schema.relationships.filter(r=>!r.archived&&r.source_types.includes(row.type_id)).map(r=><option key={r.id} value={r.id}>{r.name}</option>)}</select>{relation&&<><select aria-label="Related record" value={target} onChange={e=>setTarget(e.target.value)}><option value="">Choose record…</option>{choices.filter(r=>schema.relationships.find(l=>l.id===relation)?.target_types.includes(r.type_id)).map(r=><option key={r.id} value={r.id}>{r.title}</option>)}</select><button disabled={!target||busy} onClick={()=>void link().catch(()=>{})}>Link</button></>}</div>}
    </main><aside>
      <label>Main home<select aria-label="Main home" value={row.parent_id??""} disabled={!canEdit||busy} onChange={e=>void save({parent_id:e.target.value||null}).catch(()=>{})}><option value="">Unfiled</option>{choices.filter(r=>r.id!==row.id&&t.parent_types.includes(r.type_id)).map(r=><option key={r.id} value={r.id}>{r.title} · {r.type_name}</option>)}</select></label>
      {!!t.statuses.length&&<label>Status<select value={row.status_id??""} disabled={!canEdit||busy} onChange={e=>void save({status_id:e.target.value||null}).catch(()=>{})}>{!t.capabilities.includes("work")&&<option value="">Unassigned</option>}{t.statuses.map(s=><option key={s.id} value={s.id}>{s.name}</option>)}</select></label>}
      {t.fields.filter(f=>f.visible&&!f.archived).map(f=><label key={f.id} title={f.description}>{f.name}<FieldValue field={f} value={row.values[f.id]} choices={choices} disabled={!canEdit||busy} onSave={v=>void save({values:{[f.id]:v}}).catch(()=>{})}/>{f.inherit&&!row.inherited[f.id]&&canEdit&&<button className="text-button" onClick={()=>void save({reset_fields:[f.id]}).catch(()=>{})}>Use inherited value</button>}{row.inherited[f.id]&&<small>From {choices.find(r=>r.id===row.inherited[f.id])?.title??"main home"}</small>}</label>)}
      {alert&&<><label>Deadline alert<select value={alert.deadline_alert} disabled={!canEdit||busy} onChange={e=>void alertSave({deadline_alert:e.target.value}).catch(()=>{})}><option value="default">Use my setting</option><option value="on">On</option><option value="off">Off</option></select></label><label className="check-label"><input type="checkbox" checked={alert.alert_urgent} disabled={!canEdit||busy} onChange={e=>void alertSave({alert_urgent:e.target.checked}).catch(()=>{})}/>Urgent alert · bypass quiet hours</label></>}
      {canEdit&&<button className="text-button" disabled={busy} onClick={()=>void save({archived:!row.archived}).then(onClose).catch(()=>{})}>{row.archived?"Restore":"Archive"}</button>}
      <small aria-live="polite">{busy?"Saving…":error?"Changes need attention":"Changes save as you leave each field"}</small>
    </aside></div>
  </section></div>;
}
