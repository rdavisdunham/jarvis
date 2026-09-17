import { useDialogFocus } from "./components";
import { api } from "./api";
import { useState, useEffect } from "react";
import { ArrowLeft, Check, Plus, X } from "lucide-react";
import { useBodyLock } from "./ux";
import { useStructureActions } from "./structure-actions";
import { MultiSelect, TypeEditor } from "./TypeEditor";
import type { Schema, SchemaType, SchemaRelation, Proposal } from "./structure-types";
import { describe } from "./structure-types";

export function StructureEditor({schema,onClose,onApplied,initialProposal,onDirtyChange}:{onDirtyChange?:(dirty:boolean)=>void;initialProposal?:Proposal;schema:Schema;onClose:()=>void;onApplied:()=>Promise<void>}){
  useBodyLock(true);useDialogFocus();const [draft,setDraft]=useState(()=>structuredClone(schema));const [selected,setSelected]=useState(schema.types[0]?.id??"");const [proposal,setProposal]=useState<Proposal|null>(initialProposal??null);const {run,busy,error}=useStructureActions();
  const [history,setHistory]=useState<Proposal[]>([]);
  useEffect(()=>{void api<{items:Proposal[]}>("/structure/history/applied").then(r=>setHistory(r.items));},[]);
  const type=draft.types.find(t=>t.id===selected);const [statusMappings,setStatusMappings]=useState<Record<string,Record<string,string>>>({});
  useEffect(()=>{onDirtyChange?.(busy||JSON.stringify(draft)!==JSON.stringify(schema)||Object.keys(statusMappings).length>0);},[draft,schema,busy,statusMappings,onDirtyChange]);
  const update=(patch:Partial<SchemaType>)=>{setDraft({...draft,types:draft.types.map(t=>t.id===selected?{...t,...patch}:t)});setProposal(null);};
  const addType=()=>{const t:SchemaType={id:crypto.randomUUID(),name:"New type",plural:"New types",description:"",capabilities:[],parent_types:draft.types.map(t=>t.id),fields:[],statuses:[],archived:false};setDraft({...draft,types:[...draft.types,t]});setSelected(t.id);setProposal(null);};
  const preview=async()=>setProposal(await run<Proposal>("structure.preview",{expected_revision:schema.revision,definition:{types:draft.types,relationships:draft.relationships},status_mappings:statusMappings}));
  const apply=async()=>{await run("structure.apply",{proposal_id:proposal!.id,expected_revision:proposal!.schema_revision});await onApplied();};
  const relation=(id:string,patch:Partial<SchemaRelation>)=>{setDraft({...draft,relationships:draft.relationships.map(r=>r.id===id?{...r,...patch}:r)});setProposal(null);};
  return <div className="modal-backdrop"><section role="dialog" aria-modal="true" aria-label="Workspace structure" className="structure-editor">
    <header><h2>Workspace structure</h2><button aria-label="Close structure" disabled={busy} onClick={onClose}><X size={20}/></button></header>
    {!!history.length&&<details><summary>Previous structure changes</summary>{history.map(p=><button key={p.id} disabled={busy} onClick={()=>void run<Proposal>("structure.restore",{proposal_id:p.id,expected_revision:schema.revision}).then(setProposal).catch(()=>{})}>Preview undo of version {p.schema_revision+1}</button>)}</details>}
    <p className="subtle">Define your types and relationships. Review the impact before applying changes.</p>{error&&<p role="alert" className="error">{error}</p>}
    {proposal?<section className="schema-preview"><button className="text-button" onClick={()=>setProposal(null)}><ArrowLeft size={15}/>Keep editing</button><h3>Review your changes</h3><p>{proposal.impact.affected_count} existing records use changed definitions. Their original data is retained.</p><ul>{proposal.impact.affected_records.map(r=><li key={r.id}>{r.title}</li>)}</ul>{proposal.impact.issues.map((i,n)=><p role="alert" key={n}>{i.message}</p>)}<div className="schema-tree">{proposal.definition.types.filter(t=>!t.archived).map(t=><div key={t.id}><strong>{t.plural}</strong><p>{t.description}</p><small>{t.fields.filter(f=>!f.archived).map(f=>f.name).join(" · ")||"Title and details"}</small></div>)}</div><button className="primary" disabled={busy||!!proposal.impact.blocking_count} onClick={()=>void apply().catch(()=>{})}><Check size={17}/> Apply structure</button></section>:
      <><div className="schema-editor-grid"><nav aria-label="Record types">{draft.types.map(t=><button key={t.id} className={selected===t.id?"active":""} onClick={()=>setSelected(t.id)}>{t.plural}{t.archived?" (archived)":""}</button>)}<button onClick={addType}><Plus size={15}/> New type</button></nav>
      {type&&<TypeEditor type={type} schema={draft} original={schema} update={update} mappings={statusMappings[type.id]??{}} onMappings={m=>{setStatusMappings({...statusMappings,[type.id]:m});setProposal(null);}}/>}</div>
      <details className="relationship-editor"><summary>Additional relationships</summary>{draft.relationships.map(r=><fieldset key={r.id}><label>Name<input value={r.name} onChange={e=>relation(r.id,{name:e.target.value})}/></label><label>Description <span className="required">Required</span><textarea required value={r.description} onChange={e=>relation(r.id,{description:e.target.value})}/></label><div className="schema-inline"><MultiSelect label="From types" values={r.source_types} options={draft.types.map(t=>({id:t.id,name:t.name}))} onChange={v=>relation(r.id,{source_types:v})}/><MultiSelect label="To types" values={r.target_types} options={draft.types.map(t=>({id:t.id,name:t.name}))} onChange={v=>relation(r.id,{target_types:v})}/><label>Connections<select value={r.cardinality} onChange={e=>relation(r.id,{cardinality:e.target.value})}>{["one_to_one","one_to_many","many_to_one","many_to_many"].map(v=><option key={v} value={v}>{describe(v)}</option>)}</select></label></div><label className="check-label"><input type="checkbox" checked={r.archived} onChange={e=>relation(r.id,{archived:e.target.checked})}/>Archived</label></fieldset>)}<button onClick={()=>{setDraft({...draft,relationships:[...draft.relationships,{id:crypto.randomUUID(),name:"",description:"",source_types:[],target_types:[],cardinality:"many_to_many",archived:false}]});setProposal(null);}}><Plus size={15}/> Add relationship</button></details>
      <footer><button onClick={onClose}>Close</button><button className="primary" disabled={busy} onClick={()=>void preview().catch(()=>{})}>Preview changes</button></footer></>}
  </section></div>;
}
