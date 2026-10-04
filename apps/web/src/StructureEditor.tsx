import { TypeMap } from "./TypeMap";
import { updateType } from "./field-library";
import { api } from "./api";
import { useState, useEffect } from "react";
import { Check, History, Link2, Plus, X } from "lucide-react";
import { Dialog } from "./ux";
import { useStructureActions } from "./structure-actions";
import { MultiSelect, Required, TypeEditor } from "./TypeEditor";
import type { Schema, SchemaType, SchemaRelation, Proposal, ProposalIssue } from "./structure-types";
import { describe } from "./structure-types";
import "./details.css";

const sentence = (value: string) => { const text = describe(value); return text.charAt(0).toUpperCase() + text.slice(1); };

/** Records still living in a home their type would no longer allow; Apply stays blocked until they move. */
export function PlacementImpact({issues,types}:{issues:ProposalIssue[];types:Pick<SchemaType,"id"|"name"|"plural">[]}){
  const placed=issues.filter(i=>i.kind==="placement");
  if(!placed.length)return null;
  const name=(id?:string)=>types.find(t=>t.id===id);
  return <section className="placement-impact" role="alert" aria-label="Records in homes that would no longer be allowed">
    <h4>{placed.length===1?"1 record lives":placed.length+" records live"} in a home this change disallows</h4>
    <p>Move {placed.length===1?"it":"them"} first, or keep the allowed home. Nothing changes until you apply.</p>
    <ul>{placed.map(i=><li key={i.record_id}>
      <a href={"?view=organize&layout=browse&home="+encodeURIComponent(i.home?.id??"")} target="_blank" rel="noreferrer">{i.title}</a>
      <span className="placement-meta"><span>{name(i.type_id)?.name??"Record"}</span>{i.home&&<span>in {i.home.title} ({name(i.home.type_id)?.name??"home"})</span>}{i.archived&&<span className="chip">Archived</span>}</span>
    </li>)}</ul>
  </section>;
}

export function StructureEditor({schema,onClose,onApplied,initialProposal,initialDraft,initialType,onDirtyChange}:{onDirtyChange?:(dirty:boolean)=>void;initialProposal?:Proposal;/** A draft started elsewhere (the Atlas Blueprint) continues here. */initialDraft?:Schema;initialType?:string;schema:Schema;onClose:()=>void;onApplied:()=>Promise<void>}){
  const [draft,setDraft]=useState(()=>structuredClone(initialDraft??schema));const [selected,setSelected]=useState(initialType??schema.types[0]?.id??"");const [proposal,setProposal]=useState<Proposal|null>(initialProposal??null);const {run,busy,error,setError}=useStructureActions();
  const [visual,setVisual]=useState(true),[fieldId,setFieldId]=useState("");
  const [pane,setPane]=useState<"type"|"relationships"|"history">("type");
  const [history,setHistory]=useState<Proposal[]>([]);
  useEffect(()=>{void api<{items:Proposal[]}>("/structure/history/applied").then(r=>setHistory(r.items));},[]);
  const type=draft.types.find(t=>t.id===selected);const [statusMappings,setStatusMappings]=useState<Record<string,Record<string,string>>>({});
  useEffect(()=>{onDirtyChange?.(busy||JSON.stringify(draft)!==JSON.stringify(schema)||Object.keys(statusMappings).length>0);},[draft,schema,busy,statusMappings,onDirtyChange]);
  const update=(patch:Partial<SchemaType>)=>{const added=patch.fields?.find(f=>!type?.fields.some(old=>old.id===f.id));setDraft(updateType(draft,selected,patch));if(added)setFieldId(added.id);setProposal(null);};
  const addType=(parent?:string)=>{const t:SchemaType={id:crypto.randomUUID(),name:"New type",plural:"New types",description:"",capabilities:[],parent_types:draft.types.map(t=>t.id),fields:[],statuses:[],archived:false};setDraft({...draft,types:[...draft.types,t],type_layout:[...(draft.type_layout??[]),{type_id:t.id,parent_type_id:parent??null}]});setFieldId("");setSelected(t.id);setPane("type");setProposal(null);};
  const preview=async()=>setProposal(await run<Proposal>("structure.preview",{expected_revision:schema.revision,definition:{types:draft.types,relationships:draft.relationships,field_library:draft.field_library,type_layout:draft.type_layout},status_mappings:statusMappings}));
  const apply=async()=>{await run("structure.apply",{proposal_id:proposal!.id,expected_revision:proposal!.schema_revision});await onApplied();};
  const relation=(id:string,patch:Partial<SchemaRelation>)=>{setDraft({...draft,relationships:draft.relationships.map(r=>r.id===id?{...r,...patch}:r)});setProposal(null);};
  const typeOptions=draft.types.map(t=>({id:t.id,name:t.name}));
  return <Dialog className="schema-dialog" aria-label="Workspace structure">
    <header className="schema-dialog-head"><div><h2>Workspace structure</h2><p>Define your record types and how they relate. Review the impact before applying changes.</p></div>
      <button className="btn-icon" aria-label="Close structure" disabled={busy} onClick={onClose}><X size={20}/></button></header>
    {error&&<p role="alert" className="detail-alert schema-alert">{error}</p>}
    {proposal?<>
      <div className="schema-dialog-body single"><section className="schema-review">
        <div><h3>Review your changes</h3><p className="schema-review-summary">{proposal.impact.affected_count} existing {proposal.impact.affected_count===1?"record uses":"records use"} changed definitions. Their original data is retained.</p></div>
        {!!proposal.impact.affected_records.length&&<ul className="schema-review-records">{proposal.impact.affected_records.map(r=><li key={r.id} className="chip">{r.title}</li>)}</ul>}
        {!!proposal.impact.review_changes?.length&&<ul className="schema-review-cadence" aria-label="Review cadence changes">{proposal.impact.review_changes.map(c=><li key={c.type_id}>{c.every?<>{c.name} records will resurface for review {c.every_label}. The first reviews of its {c.records} {c.records===1?"record":"records"} are spread over the first interval.</>:<>{c.name} records stop resurfacing for review. Review dates are kept.</>}</li>)}</ul>}
        <PlacementImpact issues={proposal.impact.issues} types={proposal.definition.types}/>
        {proposal.impact.issues.filter(i=>i.kind!=="placement").map((i,n)=><p role="alert" className="schema-issue" key={n}>{i.message}</p>)}
        <div className="schema-tree-grid">{proposal.definition.types.filter(t=>!t.archived).map(t=><div key={t.id}><strong>{t.plural}</strong>{t.description&&<p>{t.description}</p>}<div className="chip-row">{t.fields.filter(f=>!f.archived).map(f=><span className="chip" key={f.id}>{f.name}</span>)}{!t.fields.some(f=>!f.archived)&&<span className="chip">Title and details</span>}</div></div>)}</div>
      </section></div>
      <footer className="schema-dialog-foot"><button className="btn" onClick={()=>setProposal(null)}>Keep editing</button><button className="btn btn-primary" disabled={busy||!!proposal.impact.blocking_count} onClick={()=>void apply().catch(()=>{})}><Check size={17}/>Apply structure</button></footer>
    </>:<>
      <div className="schema-view-switch"><button className="btn btn-sm" aria-pressed={visual} onClick={()=>setVisual(true)}>Visual tree</button><button className="btn btn-sm" aria-pressed={!visual} onClick={()=>setVisual(false)}>Advanced</button><button className="btn btn-sm" onClick={()=>setPane("relationships")}>Extra relationships</button>{!!history.length&&<button className="btn btn-sm" onClick={()=>setPane("history")}>Versions</button>}</div>
      <div className={"schema-dialog-body"+(visual?" visual":"")}>
        {visual?<TypeMap schema={draft} selected={selected} fieldId={fieldId} onSelect={(id,field="")=>{setSelected(id);setFieldId(field);setPane("type");}} onChange={s=>{setDraft(s);setProposal(null);}} onAdd={addType} onError={setError}/>:
        <nav className="schema-rail" aria-label="Record types"><span className="schema-rail-label">Record types</span>
          {draft.types.map(t=><button key={t.id} aria-current={pane==="type"&&selected===t.id} onClick={()=>{setSelected(t.id);setPane("type");}}>{t.plural}{t.archived&&<span className="chip">Archived</span>}</button>)}
          <button className="schema-rail-add" onClick={()=>addType()}><Plus size={15}/>New type</button>
          <div className="schema-rail-extra">
            <button aria-current={pane==="relationships"} onClick={()=>setPane("relationships")}><Link2 size={15}/>Relationships</button>
            {!!history.length&&<button aria-current={pane==="history"} onClick={()=>setPane("history")}><History size={15}/>Previous versions</button>}
          </div>
        </nav>}
        {pane==="type"&&type&&<TypeEditor compact={visual} focusedField={visual?fieldId:""} type={type} schema={draft} original={schema} update={update} mappings={statusMappings[type.id]??{}} onMappings={m=>{setStatusMappings({...statusMappings,[type.id]:m});setProposal(null);}}/>}
        {pane==="relationships"&&<div className="schema-main">
          <div className="schema-block-head"><h3>Additional relationships</h3><span className="detail-count">{draft.relationships.length}</span><button className="btn btn-soft btn-sm" onClick={()=>{setDraft({...draft,relationships:[...draft.relationships,{id:crypto.randomUUID(),name:"",description:"",source_types:[],target_types:[],cardinality:"many_to_many",archived:false}]});setProposal(null);}}><Plus size={15}/>Add relationship</button></div>
          <p className="schema-block-hint">Links between records beyond their main home, such as a project that serves a goal.</p>
          {!draft.relationships.length&&<p className="detail-empty">No additional relationships yet.</p>}
          {draft.relationships.map(r=><fieldset key={r.id} className={"schema-card"+(r.archived?" archived":"")}>
            <div className="schema-card-head"><span className="schema-card-title">{r.name||"Untitled relationship"}</span>{r.archived&&<span className="chip">Archived</span>}</div>
            <label className="field"><span className="field-label-text">Name</span><input value={r.name} onChange={e=>relation(r.id,{name:e.target.value})}/></label>
            <label className="field"><span className="field-label-text">Description <Required/></span><textarea rows={2} required value={r.description} onChange={e=>relation(r.id,{description:e.target.value})}/></label>
            <div className="schema-pair"><MultiSelect label="From types" values={r.source_types} options={typeOptions} onChange={v=>relation(r.id,{source_types:v})}/><MultiSelect label="To types" values={r.target_types} options={typeOptions} onChange={v=>relation(r.id,{target_types:v})}/></div>
            <label className="field"><span className="field-label-text">Behavior</span><select aria-label="Relationship behavior" value={r.behavior??"related"} onChange={e=>relation(r.id,{behavior:e.target.value as "related"|"blocks"})}><option value="related">Extra link</option><option value="blocks">Advisory blocker</option></select></label>
            <label className="field"><span className="field-label-text">Connections</span><select value={r.cardinality} onChange={e=>relation(r.id,{cardinality:e.target.value})}>{["one_to_one","one_to_many","many_to_one","many_to_many"].map(v=><option key={v} value={v}>{sentence(v)}</option>)}</select></label>
            <div className="schema-card-foot"><label className="schema-check"><input type="checkbox" checked={r.archived} onChange={e=>relation(r.id,{archived:e.target.checked})}/>Archived</label></div>
          </fieldset>)}
        </div>}
        {pane==="history"&&<div className="schema-main">
          <div className="schema-block-head"><h3>Previous structure changes</h3></div>
          <p className="schema-block-hint">Preview undoing an applied change. Nothing changes until you apply it.</p>
          <div className="schema-history">{history.map(p=><button key={p.id} className="btn btn-ghost" style={{justifyContent:"flex-start"}} disabled={busy} onClick={()=>void run<Proposal>("structure.restore",{proposal_id:p.id,expected_revision:schema.revision}).then(setProposal).catch(()=>{})}>Preview undo of version {p.schema_revision+1}</button>)}</div>
        </div>}
      </div>
      <footer className="schema-dialog-foot"><span className="schema-foot-note">Changes apply after you review them.</span><button className="btn" onClick={onClose}>Close</button><button className="btn btn-primary" disabled={busy} onClick={()=>void preview().catch(()=>{})}>Preview changes</button></footer>
    </>}
  </Dialog>;
}
