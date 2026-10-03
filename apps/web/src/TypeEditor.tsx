import { canAttach,attachField } from "./field-library";
import { ArrowUp, Plus, X } from "lucide-react";
import type { OpensAs, Schema, SchemaField, SchemaType } from "./structure-types";
import { meanings, fieldKinds, describe, emptyField } from "./structure-types";

const OPENS_AS: [OpensAs, string, string][] = [
  ["container", "Container", "Opens to its contents and progress, like a folder."],
  ["item", "Item", "Opens its details, even when other records live inside it."],
  ["auto", "Let Eridani decide", "Opens as contents when it holds records or has no work or notes; otherwise opens its details."],
];
const sentence = (value: string) => { const text = describe(value); return text.charAt(0).toUpperCase() + text.slice(1); };

export function MultiSelect({label,values,options,onChange}:{label:string;values:string[];options:{id:string;name:string}[];onChange:(v:string[])=>void}){
  const toggle=(id:string,on:boolean)=>{const next=new Set(values);if(on)next.add(id);else next.delete(id);onChange(options.filter(o=>next.has(o.id)).map(o=>o.id));};
  return <fieldset className="schema-checks"><legend>{label}</legend>{options.map(o=><label className="schema-toggle" key={o.id}><input type="checkbox" checked={values.includes(o.id)} onChange={e=>toggle(o.id,e.target.checked)}/>{o.name}</label>)}</fieldset>;
}
export const Required = () => <span className="schema-required">Required</span>;
export function TypeEditor({type,schema,original,update,mappings,onMappings,compact=false,focusedField=""}:{compact?:boolean;focusedField?:string;type:SchemaType;schema:Schema;original:Schema;update:(p:Partial<SchemaType>)=>void;mappings:Record<string,string>;onMappings:(m:Record<string,string>)=>void}){
  const field=(id:string,patch:Partial<SchemaField>)=>update({fields:type.fields.map(f=>f.id===id?{...f,...patch}:f)});
  const capability=(name:string,enabled:boolean)=>{
    const fields=type.fields.map(f=>({...f}));
    if(!enabled){for(const f of fields){const belongs=f.binding&&(name==="work"?["due_date","due_time","due_timezone","planned_date","priority","estimate_minutes","assignee"].includes(f.binding):name==="timeline"?["start_date","target_date"].includes(f.binding):name==="metric"&&f.binding.startsWith("metric_"));if(belongs){f.binding=null;f.archived=true;}}}
    if(enabled){const source=schema.field_library??original.types.flatMap(t=>t.fields);for(const f of source){const match=f.binding&&(name==="work"?["due_date","due_time","due_timezone","planned_date","priority","estimate_minutes","assignee"].includes(f.binding):name==="metric"?f.binding.startsWith("metric_"):name==="timeline"&&["start_date","target_date"].includes(f.binding));if(match&&!fields.some(x=>x.binding===f.binding)){const added=schema.field_library?attachField(f):structuredClone(f);if(fields.some(x=>x.id===added.id))added.id=crypto.randomUUID();fields.push(added);}}}
    update({capabilities:enabled?[...type.capabilities,name]:type.capabilities.filter(c=>c!==name),fields,statuses:name==="work"&&enabled&&!type.statuses.length?meanings.map(s=>({id:s,name:describe(s),meaning:s})):type.statuses});
  };
  const retired=original.types.find(t=>t.id===type.id)?.statuses.filter(s=>!type.statuses.some(x=>x.id===s.id))??[];
  return <div className="schema-main">
    {!focusedField&&<><div className="schema-pair"><label className="field"><span className="field-label-text">Singular name</span><input value={type.name} onChange={e=>update({name:e.target.value})}/></label><label className="field"><span className="field-label-text">Plural name</span><input value={type.plural} onChange={e=>update({plural:e.target.value})}/></label></div>
    <label className="field"><span className="field-label-text">Description <Required/></span><textarea required value={type.description} onChange={e=>update({description:e.target.value})} placeholder="What does this type mean? When should Eri use it?" rows={3}/></label>
    <fieldset className="schema-checks"><legend>Behaviors</legend>{[["work","Actionable work"],["content","Authored content"],["timeline","Timeline dates"],["metric","Outcome metric"]].map(([id,label])=><label className="schema-toggle" key={id}><input type="checkbox" checked={type.capabilities.includes(id)} onChange={e=>capability(id,e.target.checked)}/>{label}</label>)}</fieldset>
    <fieldset className="schema-opens-as"><legend>Organizing container</legend>
      <div className="segmented" role="group" aria-label="Organizing container">{OPENS_AS.map(([id,label])=><button type="button" key={id} aria-pressed={(type.opens_as??"auto")===id} onClick={()=>update({opens_as:id})}>{label}</button>)}</div>
      <p className="schema-block-hint">{OPENS_AS.find(([id])=>id===(type.opens_as??"auto"))![2]}</p>
    </fieldset>
    <MultiSelect label="Allowed main-home types" values={type.parent_types} options={schema.types.map(t=>({id:t.id,name:t.name}))} onChange={v=>update({parent_types:v})}/></>}
    <section className="schema-block" aria-label="Fields">
      <div className="schema-block-head"><h3>Fields</h3><span className="detail-count">{type.fields.length}</span><button className="btn btn-soft btn-sm" onClick={()=>update({fields:[...type.fields,emptyField()]})}><Plus size={15}/>Add field</button></div>
      {!type.fields.length&&<p className="detail-empty">Every record has a title and details. Add fields for anything else you track.</p>}
      <div className="field-reuse"><label>Reuse a field <select aria-label="Add existing field" value="" onChange={e=>{const f=schema.field_library?.find(f=>f.id===e.target.value);if(f)update({fields:[...type.fields,attachField(f)]});}}><option value="">Choose from library…</option>{schema.field_library?.filter(f=>canAttach(f,type)).map(f=><option key={f.id} value={f.id}>{f.name}</option>)}</select></label></div>
      {(compact?type.fields.filter(f=>f.id===focusedField):type.fields).map(f=><fieldset key={f.id} className={"schema-card"+(f.archived?" archived":"")}>
        {f.library_id&&<p className="shared-field-notice">Shared definition used by {schema.types.filter(t=>t.fields.some(v=>v.library_id===f.library_id)).map(t=>t.name).join(", ")}. Name, meaning and value type changes apply to all of them. Visibility, inheritance and archive apply here.</p>}
        <div className="schema-card-head"><span className="schema-card-title">{f.name||"Untitled field"}</span>{f.binding&&<span className="chip">Used by {describe(f.binding)}</span>}{f.archived&&<span className="chip">Archived</span>}
          <button className="btn-icon" aria-label="Move field up" title="Move field up" disabled={type.fields[0].id===f.id} onClick={()=>{const fields=[...type.fields];const i=fields.findIndex(x=>x.id===f.id);[fields[i-1],fields[i]]=[fields[i],fields[i-1]];update({fields});}}><ArrowUp size={16}/></button></div>
        <div className="schema-pair"><label className="field"><span className="field-label-text">Field name</span><input value={f.name} onChange={e=>field(f.id,{name:e.target.value})}/></label><label className="field"><span className="field-label-text">Value type</span><select value={f.kind} disabled={!!f.binding} onChange={e=>field(f.id,{kind:e.target.value})}>{fieldKinds.map(k=><option value={k} key={k}>{sentence(k)}</option>)}</select></label></div>
        <label className="field"><span className="field-label-text">Description <Required/></span><textarea rows={2} required value={f.description} onChange={e=>field(f.id,{description:e.target.value})}/></label>
        {f.kind==="relation"&&<><MultiSelect label="Links to types" values={f.target_types} options={schema.types.map(t=>({id:t.id,name:t.name}))} onChange={v=>field(f.id,{target_types:v})}/><label className="schema-check"><input type="checkbox" checked={f.multiple} onChange={e=>field(f.id,{multiple:e.target.checked})}/>Allow multiple values</label></>}
        {["select","multiselect"].includes(f.kind)&&<div className="schema-options"><span className="field-label-text">Options</span>{f.options.map(o=><div className="schema-option" key={o.id}><input aria-label="Option name" value={o.name} onChange={e=>field(f.id,{options:f.options.map(x=>x.id===o.id?{...x,name:e.target.value}:x)})}/><button className="btn-icon" aria-label={"Remove option "+o.name} onClick={()=>field(f.id,{options:f.options.filter(x=>x.id!==o.id)})}><X size={14}/></button></div>)}<button className="btn btn-ghost btn-sm" style={{alignSelf:"flex-start"}} onClick={()=>field(f.id,{options:[...f.options,{id:crypto.randomUUID(),name:"New option"}]})}><Plus size={14}/>Add option</button></div>}
        <div className="schema-card-foot"><label className="schema-check"><input type="checkbox" checked={f.inherit} disabled={!!f.binding} onChange={e=>field(f.id,{inherit:e.target.checked})}/>Inherit through main home</label><label className="schema-check"><input type="checkbox" checked={f.visible} onChange={e=>field(f.id,{visible:e.target.checked})}/>Visible</label><label className="schema-check"><input type="checkbox" checked={f.archived} onChange={e=>field(f.id,{archived:e.target.checked})}/>Archived</label></div>
      </fieldset>)}
    </section>
    {!focusedField&&<><section className="schema-block" aria-label="Workflow statuses">
      <div className="schema-block-head"><h3>Workflow statuses</h3>{!!type.statuses.length&&<span className="detail-count">{type.statuses.length}</span>}<button className="btn btn-soft btn-sm" onClick={()=>update({statuses:[...type.statuses,{id:crypto.randomUUID(),name:"New status",meaning:"open"}]})}><Plus size={15}/>Add status</button></div>
      <p className="schema-block-hint">Each status maps to a meaning Eri understands, such as open or completed.</p>
      {type.statuses.map((s,index)=><div className="schema-status-row" key={s.id}><label className="field"><span className="field-label-text">Status name</span><input value={s.name} onChange={e=>update({statuses:type.statuses.map(x=>x.id===s.id?{...x,name:e.target.value}:x)})}/></label><label className="field"><span className="field-label-text">Meaning</span><select value={s.meaning} onChange={e=>update({statuses:type.statuses.map(x=>x.id===s.id?{...x,meaning:e.target.value}:x)})}>{meanings.map(m=><option key={m} value={m}>{sentence(m)}</option>)}</select></label><button className="btn-icon" aria-label={"Move "+s.name+" earlier"} disabled={!index} onClick={()=>{const statuses=[...type.statuses];[statuses[index-1],statuses[index]]=[statuses[index],statuses[index-1]];update({statuses});}}><ArrowUp size={16}/></button><button className="btn-icon" aria-label={"Retire "+s.name} onClick={()=>update({statuses:type.statuses.filter(x=>x.id!==s.id)})}><X size={15}/></button></div>)}
      {retired.map(s=><label className="field" key={s.id}><span className="field-label-text">Move records from {s.name}</span><select value={mappings[s.id]??""} onChange={e=>onMappings({...mappings,[s.id]:e.target.value})}><option value="">Choose a replacement</option>{type.statuses.map(x=><option key={x.id} value={x.id}>{x.name}</option>)}</select></label>)}
    </section>
    <div className="schema-block"><label className="schema-check"><input type="checkbox" checked={type.archived} onChange={e=>update({archived:e.target.checked})}/>Archive this type</label></div></>}
  </div>;
}
