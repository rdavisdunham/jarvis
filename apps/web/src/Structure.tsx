import { useCallback, useEffect, useState } from "react";
import { GripVertical, Plus, Settings2 } from "lucide-react";
import { api, post } from "./api";
import { useStructureActions } from "./structure-actions";
import { RecordCard } from "./RecordCard";
import { StructureEditor } from "./StructureEditor";
import type { Schema, CustomRecord, Proposal } from "./structure-types";
import "./structure.css";
import { meanings, describe } from "./structure-types";

export function StructureWorkspace({selecting=false,selectedIds=[],onSelecting,onSelection,onBulk,query="",canEdit=true,canDesign=true,refresh=0,onChanged,capability="",tab="all",today=new Date().toISOString().slice(0,10),onTask,onVisible,control,onQuery,onTab,statusFilter,homeFilter,layoutFilter,groupFilter,onContext}:{selecting?:boolean;selectedIds?:string[];onSelecting?:(v:boolean)=>void;onSelection?:(ids:string[])=>void;onBulk?:()=>void;statusFilter?:string;homeFilter?:string;layoutFilter?:string;groupFilter?:string;onContext?:(state:Record<string,string|number|null>)=>void;query?:string;canEdit?:boolean;canDesign?:boolean;refresh?:number;onChanged?:()=>void;capability?:string;tab?:string;today?:string;onTask?:(id:string)=>void;onQuery?:(value:string)=>void;onTab?:(value:"today"|"inbox"|"week"|"all")=>void;onVisible?:(ids:string[])=>void;control?:{nonce:string;type_id?:string;parent_id?:string;layout?:string;group?:string;record_id?:string;proposal_id?:string;field?:string;value?:string}}){
  const [schema,setSchema]=useState<Schema|null>(null);const [items,setItems]=useState<CustomRecord[]>([]);
  const [typeId,setTypeId]=useState("");const [layout,setLayout]=useState("list");const [parent,setParent]=useState("");
  const [selected,setSelected]=useState<CustomRecord|null>(null);const [design,setDesign]=useState(false);
  const [title,setTitle]=useState("");const [archived,setArchived]=useState(false);const [group,setGroup]=useState("status");
  const [proposal,setProposal]=useState<Proposal>();
  const [status,setStatus]=useState("active"),[filterField,setFilterField]=useState(""),[filterValue,setFilterValue]=useState("");
  const [saved,setSaved]=useState<{id:string;name:string;revision:number;state:Record<string,unknown>}[]>([]),[viewName,setViewName]=useState("");
  const [timelineStart,setTimelineStart]=useState(today);
  const {run,error,busy,setError}=useStructureActions();
  const load=useCallback(async()=>{
    const s=await api<Schema>("/structure");const all:CustomRecord[]=[];let offset:number|null=0;
    while(offset!==null){const page: {items:CustomRecord[];next_offset:number|null}=await api("/structure/records?limit=200&offset="+offset+"&archived="+archived);all.push(...page.items);offset=page.next_offset;}
    setSchema(s);setItems(all);
    const views=await api<{items:typeof saved}>("/task-views");setSaved(views.items.filter(v=>v.state.collection_view));
  },[archived]);
  useEffect(()=>{let live=true;load().catch(e=>{if(live)setError(String(e.message??e));});return()=>{live=false;};},[load,refresh,setError]);
  useEffect(()=>{const handler=(event:Event)=>{const id=(event as CustomEvent).detail?.id;if(id)api<CustomRecord>("/structure/records/"+id).then(setSelected).catch(e=>setError(e.message));};window.addEventListener("eri-open-record",handler);return()=>window.removeEventListener("eri-open-record",handler);},[setError]);
  useEffect(()=>{if(!control)return;if(control.type_id!==undefined)setTypeId(control.type_id);if(control.parent_id!==undefined)setParent(control.parent_id);if(control.layout)setLayout(control.layout);if(control.group)setGroup(control.group);if(control.field!==undefined)setFilterField(control.field);if(control.value!==undefined)setFilterValue(control.value);if(control.record_id)void api<CustomRecord>("/structure/records/"+control.record_id).then(setSelected).catch(e=>setError(e.message));if(control.proposal_id)void api<Proposal>("/structure/proposals/"+control.proposal_id).then(p=>{setProposal(p);setDesign(true);}).catch(e=>setError(e.message));},[control,setError]);

  useEffect(()=>{if(statusFilter!==undefined)setStatus(statusFilter);},[statusFilter]);
  useEffect(()=>{if(homeFilter!==undefined)setParent(homeFilter);},[homeFilter]);
  useEffect(()=>{if(layoutFilter)setLayout(layoutFilter);},[layoutFilter]);
  useEffect(()=>{if(groupFilter)setGroup(groupFilter==="project"?"parent":groupFilter);},[groupFilter]);
  useEffect(()=>{onContext?.({type_id:typeId,parent_id:parent,layout,group,status,field:filterField,value:filterValue,record_id:selected?.id??null,schema_revision:schema?.revision??0});},[typeId,parent,layout,group,status,filterField,filterValue,selected?.id,schema?.revision,onContext]);
  const types=(schema?.types??[]).filter(t=>!t.archived&&(!capability||t.capabilities.includes(capability)));const type=types.find(t=>t.id===typeId);
  const captureType=type??types.find(t=>t.id==="task")??types[0];
  const weekEnd=new Date(today+"T12:00:00Z");weekEnd.setUTCDate(weekEnd.getUTCDate()+6);const end=weekEnd.toISOString().slice(0,10);
  const visible=items.filter(r=>{
    if(!schema)return false;
    if(capability&&!r.capabilities.includes(capability))return false;
    if(typeId&&r.type_id!==typeId)return false;
    if(parent&&r.parent_id!==parent&&!r.home.some(p=>p.id===parent))return false;
    if(query&&![r.title,r.body,r.type_name,JSON.stringify(r.values)].join(" ").toLowerCase().includes(query.toLowerCase()))return false;
    if(status==="active"&&["completed","cancelled"].includes(r.status_meaning??""))return false;
    if(status!=="active"&&status!=="all"&&r.status_meaning!==status)return false;
    if(filterField&&filterValue&&!String(r.values[filterField]??"").toLowerCase().includes(filterValue.toLowerCase()))return false;
    const rt=schema.types.find(t=>t.id===r.type_id)!;const dateFor=(binding:string)=>String(r.values[rt.fields.find(f=>f.binding===binding)?.id??""]??"");
    if(tab==="inbox"&&r.parent_id)return false;
    if(tab==="today"||tab==="week"){const dates=[dateFor("planned_date"),dateFor("due_date")].filter(Boolean);if(!dates.some(d=>d<=(tab==="today"?today:end)))return false;}
    return true;
  });
  const visibleIds = JSON.stringify(visible.map(r=>r.task_id??r.id));
  useEffect(()=>{onVisible?.(JSON.parse(visibleIds));},[visibleIds,onVisible]);
  if(!schema)return <p role="status">{error||"Loading your structure…"}</p>;
  const refreshAll=async()=>{await load();onChanged?.();};
  const edit=async(row:CustomRecord,changes:Record<string,unknown>)=>{await run("record.update",{record_id:row.id,expected_revision:row.revision,schema_revision:schema.revision,...changes});await refreshAll();};
  const create=async()=>{if(!title.trim()||!captureType)return;await run<CustomRecord>("record.create",{type_id:captureType.id,title:title.trim(),schema_revision:schema.revision,...(parent?{parent_id:parent}:{})});setTitle("");await refreshAll();};
  const gf=type?.fields.find(f=>f.id===group);
  const groups=group==="status"?(type?.statuses??meanings.filter(m=>visible.some(r=>r.status_meaning===m)).map(m=>({id:m,name:describe(m),meaning:m}))):group==="parent"?items.filter(r=>visible.some(v=>v.parent_id===r.id)).map(r=>({id:r.id,name:r.title,meaning:""})):gf?.kind==="relation"?items.filter(r=>gf.target_types.includes(r.type_id)).map(r=>({id:r.id,name:r.title,meaning:""})):(gf?.options.length?gf.options.map(o=>({...o,meaning:""})):Array.from(new Set(visible.map(r=>String(r.values[group]??"")).filter(Boolean))).map(v=>({id:v,name:v,meaning:""})));
  const groupValue=(r:CustomRecord)=>group==="status"?(type?r.status_id:r.status_meaning):group==="parent"?r.parent_id:String(r.values[group]??"");
  const columns=[...groups,{id:"",name:"Unassigned",meaning:""}];
  const move=async(id:string,column:string)=>{const row=items.find(r=>r.id===id);if(!row||!canEdit||busy)return;const rt=schema.types.find(t=>t.id===row.type_id)!;const statusId=type?column:rt.statuses.find(s=>s.meaning===column)?.id;if(group==="status"&&!statusId)throw new Error("Choose a workflow status for this record.");const changes=group==="status"?{status_id:statusId}:group==="parent"?{parent_id:column||null}:{values:{[group]:column||null}};await edit(row,changes);};
  const saveView=async()=>{if(!viewName.trim())return;await post("/task-views",{id:crypto.randomUUID(),name:viewName.trim(),expected_revision:0,state:{collection_view:true,collection_type:typeId,collection_parent:parent,collection_group:group,collection_field:filterField,collection_value:filterValue,tab:capability?tab:"all",query,status,layout}});setViewName("");await load();};
  const grip=(event:React.PointerEvent,id:string)=>{if(event.pointerType==="mouse")return;event.preventDefault();const target=event.currentTarget;target.setPointerCapture(event.pointerId);const up=(e:Event)=>{const point=e as PointerEvent;const col=document.elementFromPoint(point.clientX,point.clientY)?.closest<HTMLElement>("[data-record-column]");if(col)void move(id,col.dataset.recordColumn??"").catch(()=>{});target.removeEventListener("pointerup",up);};target.addEventListener("pointerup",up);};
  const card=(r:CustomRecord)=><article key={r.id} className="custom-record" draggable={canEdit&&layout==="board"} onDragStart={e=>e.dataTransfer.setData("application/eri-record",r.id)}>
    {selecting&&r.task_id&&<input aria-label={"Select "+r.title} type="checkbox" checked={selectedIds.includes(r.task_id)} onChange={e=>onSelection?.(e.target.checked?[...selectedIds,r.task_id!].slice(0,100):selectedIds.filter(id=>id!==r.task_id))}/>}
    {!selecting&&r.capabilities.includes("work")&&<button className="record-complete" aria-label={(r.status_meaning==="completed"?"Reopen ":"Complete ")+r.title} disabled={!canEdit||busy} onClick={()=>{const rt=schema.types.find(t=>t.id===r.type_id)!;const next=rt.statuses.find(s=>s.meaning===(r.status_meaning==="completed"?"open":"completed"))??(r.status_meaning==="completed"?rt.statuses.find(s=>s.meaning==="backlog"):undefined);if(next)void edit(r,{status_id:next.id}).catch(()=>{});}}>{r.status_meaning==="completed"?"✓":"○"}</button>}
    <button className="record-title" onClick={()=>setSelected(r)}>{r.title}</button><span className="subtle">{r.type_name}{r.home.length?" · "+r.home.map(p=>p.title).join(" / "):""}</span>
    {layout==="board"&&canEdit&&<button aria-label={"Move "+r.title} className="record-grip" onPointerDown={e=>grip(e,r.id)}><GripVertical size={16}/></button>}
    {r.task_id&&onTask&&<button className="text-button" onClick={()=>onTask(r.task_id!)}>Schedule & linked notes</button>}
    {r.status_id&&<span className="record-state">{schema.types.find(t=>t.id===r.type_id)?.statuses.find(s=>s.id===r.status_id)?.name}</span>}
  </article>;
  return <section className="structure-workspace">
    <header className="section-head"><div></div>{canDesign&&<button onClick={()=>{setProposal(undefined);setDesign(true);}}><Settings2 size={16}/> Structure</button>}</header>
    {onSelecting&&canEdit&&<div className="structure-controls"><button onClick={()=>onSelecting(!selecting)}>{selecting?"Done selecting":"Select tasks"}</button>{selecting&&<><label className="check-label"><input type="checkbox" aria-label="Select visible tasks" checked={visible.some(r=>r.task_id)&&visible.filter(r=>r.task_id).slice(0,100).every(r=>selectedIds.includes(r.task_id!))} onChange={e=>onSelection?.(e.target.checked?visible.filter(r=>r.task_id).slice(0,100).map(r=>r.task_id!):[])}/>Select visible tasks</label><span>{selectedIds.length} selected</span><button disabled={!selectedIds.length} onClick={onBulk}>Edit selected tasks</button></>}</div>}
    {error&&<p role="alert" className="error">{error}</p>}
    <details><summary>Saved views</summary><div className="structure-controls"><select aria-label="Saved collection view" defaultValue="" onChange={e=>{const v=saved.find(v=>v.id===e.target.value)?.state;if(!v)return;setTypeId(String(v.collection_type??""));setParent(String(v.collection_parent??""));setLayout(String(v.layout));setGroup(String(v.collection_group));setFilterField(String(v.collection_field??""));setFilterValue(String(v.collection_value??""));setStatus(String(v.status??"active"));onQuery?.(String(v.query??""));if(capability)onTab?.((["today","week","inbox"].includes(String(v.tab))?v.tab:"all") as "today"|"inbox"|"week"|"all");}}><option value="">Choose a view…</option>{saved.map(v=><option key={v.id} value={v.id}>{v.name}</option>)}</select><input aria-label="View name" value={viewName} onChange={e=>setViewName(e.target.value)} placeholder="Name this view"/><button disabled={!viewName.trim()} onClick={()=>void saveView().catch(e=>setError(e.message))}>Save view</button></div></details>
    <div className="structure-controls"><label>Collection<select value={typeId} onChange={e=>{setTypeId(e.target.value);setGroup("status");}}><option value="">{capability?"All actionable work":"All records"}</option>{types.map(t=><option key={t.id} value={t.id}>{t.plural}</option>)}</select></label>
      <label>Main home<select value={parent} onChange={e=>setParent(e.target.value)}><option value="">All homes</option>{items.map(r=><option key={r.id} value={r.id}>{r.title} · {r.type_name}</option>)}</select></label>
      <label>View<select value={layout} onChange={e=>setLayout(e.target.value)}><option value="list">List</option><option value="board">Board</option><option value="timeline">Timeline</option></select></label>
      {layout==="board"&&<label>Group<select value={group} onChange={e=>setGroup(e.target.value)}><option value="status">Status</option><option value="parent">Main home</option>{type?.fields.filter(f=>(f.kind==="select"||(f.kind==="relation"&&!f.multiple))&&!f.archived).map(f=><option key={f.id} value={f.id}>{f.name}</option>)}</select></label>}
      <label>Status<select value={status} onChange={e=>setStatus(e.target.value)}><option value="active">Active</option><option value="all">All statuses</option>{meanings.map(m=><option key={m} value={m}>{describe(m)}</option>)}</select></label>
      {type&&<><label>Field filter<select value={filterField} onChange={e=>{setFilterField(e.target.value);setFilterValue("");}}><option value="">Any field</option>{type.fields.filter(f=>!f.archived).map(f=><option value={f.id} key={f.id}>{f.name}</option>)}</select></label>{filterField&&<label>Value{["select","multiselect","relation"].includes(type.fields.find(f=>f.id===filterField)?.kind??"")?<select aria-label="Field filter value" value={filterValue} onChange={e=>setFilterValue(e.target.value)}><option value="">Any value</option>{(type.fields.find(f=>f.id===filterField)?.kind==="relation"?items.filter(r=>type.fields.find(f=>f.id===filterField)?.target_types.includes(r.type_id)).map(r=>({id:r.id,name:r.title})):type.fields.find(f=>f.id===filterField)?.options??[]).map(o=><option key={o.id} value={o.id}>{o.name}</option>)}</select>:<input aria-label="Field filter value" value={filterValue} onChange={e=>setFilterValue(e.target.value)}/>}</label>}</>}
      {layout==="timeline"&&<label>Timeline start<input type="date" value={timelineStart} onChange={e=>setTimelineStart(e.target.value)}/></label>}
      <label className="check-label"><input type="checkbox" checked={archived} onChange={e=>setArchived(e.target.checked)}/>Archived</label>
    </div>
    {captureType&&canEdit&&!archived&&<form className="record-capture" onSubmit={e=>{e.preventDefault();void create().catch(()=>{});}}><input aria-label={"New "+captureType.name} value={title} onChange={e=>setTitle(e.target.value)} placeholder={"Add "+captureType.name.toLowerCase()+"…"} maxLength={500}/><button disabled={busy||!title.trim()}><Plus size={17}/> Add</button></form>}
    {layout==="board"?<div className="custom-board">{columns.filter(c=>c.id||visible.some(r=>!groupValue(r))).map(c=><section key={c.id} data-record-column={c.id} onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();void move(e.dataTransfer.getData("application/eri-record"),c.id).catch(()=>{});}}><h3>{c.name}<span>{visible.filter(r=>groupValue(r)===c.id||(!c.id&&!groupValue(r))).length}</span></h3>{visible.filter(r=>groupValue(r)===c.id||(!c.id&&!groupValue(r))).map(card)}</section>)}</div>:
      layout==="timeline"?<div className="custom-timeline"><p>30 days from {timelineStart} · task dates are markers, not reserved time</p>{visible.map(r=>{const t=schema.types.find(t=>t.id===r.type_id)!;const date=(b:string)=>String(r.values[t.fields.find(f=>f.binding===b)?.id??""]??"");const start=date("start_date")||date("planned_date")||date("due_date");const finish=date("target_date")||start;const offset=(value:string)=>(Date.parse(value+"T12:00:00Z")-Date.parse(timelineStart+"T12:00:00Z"))/86400000;const a=offset(start),b=offset(finish);return <div key={r.id}>{card(r)}<section>{start?<><div className="timeline-track">{b>=0&&a<30&&<span className="timeline-bar" title={start+(finish!==start?" – "+finish:"")} style={{left:Math.max(0,a)/30*100+"%",width:Math.max(.4,(Math.min(30,b+1)-Math.max(0,a))/30*100)+"%"}}/>}</div><small>{start}{finish!==start?" – "+finish:""}</small></>:<span className="subtle">Unscheduled</span>}</section></div>;})}</div>:<div className="custom-record-list">{visible.map(card)}</div>}
    {!visible.length&&<p className="empty-state">{query?"No matching records.":"Choose a collection to add a record, or shape your workspace in Structure."}</p>}
    {selected&&<RecordCard schema={schema} initial={selected} choices={items} canEdit={canEdit} onClose={()=>setSelected(null)} onChanged={refreshAll}/>}
    {design&&<StructureEditor initialProposal={proposal} schema={schema} onClose={()=>setDesign(false)} onApplied={async()=>{setDesign(false);await refreshAll();}}/>}
  </section>;
}
