import { useState } from "react";
import { ArrowDown, ArrowUp, FolderInput } from "lucide-react";
import type { CustomRecord, Schema } from "./structure-types";
import { Popover } from "./ux";

export function validHomes(row: CustomRecord, items: CustomRecord[], schema: Schema) {
  const allowed = schema.types.find(t=>t.id===row.type_id)?.parent_types ?? [];
  return items.filter(r=>!r.archived && r.id!==row.id && !r.home.some(h=>h.id===row.id) && allowed.includes(r.type_id));
}
export function ordered(items: CustomRecord[]) {
  return [...items].sort((a,b)=>(a.sort_order??0)-(b.sort_order??0)||a.id.localeCompare(b.id));
}
/** Native disclosure/buttons keep every move available to keyboard and touch users. */
export function OrganizationTree({items,visible,schema,disabled,onOpen,onBrowse,onMove}: {
  items:CustomRecord[];visible:CustomRecord[];schema:Schema;disabled:boolean;
  onOpen:(r:CustomRecord)=>void;onBrowse:(r:CustomRecord)=>void;
  onMove:(r:CustomRecord,changes:Record<string,unknown>)=>Promise<void>;
}) {
  const [moving,setMoving]=useState<string|null>(null);
  const [closed,setClosed]=useState<Set<string>>(new Set());
  const included=new Set(visible.flatMap(r=>[r.id,...r.home.map(h=>h.id)]));
  const rows=ordered(items.filter(r=>included.has(r.id)));
  const move=(r:CustomRecord,parent:string|null,before:string|null)=>{
    setMoving(r.id);void onMove(r,{parent_id:parent,move_before_id:before}).catch(()=>{}).finally(()=>setMoving(null));
  };
  const branch=(parent:string|null,depth=0): React.ReactNode=>{
    if(depth>100)return null;
    const children=rows.filter(r=>r.parent_id===parent || (parent===null&&!items.some(p=>p.id===r.parent_id)));
    return <ul className="organization-branch">{children.map(row=>{
      const allSiblings=ordered(items.filter(r=>r.parent_id===row.parent_id));
      const index=allSiblings.findIndex(r=>r.id===row.id);
      const descendants=items.filter(r=>r.home.some(h=>h.id===row.id));
      const homes=validHomes(row,items,schema);
      const locked=disabled||moving!==null;
      const content=<div className="organization-node" draggable={!locked}
        onDragStart={e=>{e.stopPropagation();e.dataTransfer.setData("application/eri-record",row.id);}}
        onDragOver={e=>{if(!locked)e.preventDefault();}} onDrop={e=>{e.preventDefault();e.stopPropagation();
          const source=items.find(r=>r.id===e.dataTransfer.getData("application/eri-record"));
          if(source&&!locked&&source.id!==row.id){
            const top=e.clientY-e.currentTarget.getBoundingClientRect().top<e.currentTarget.clientHeight*.3;
            const home=top?row.parent_id:row.id;
            if(home===null||validHomes(source,items,schema).some(h=>h.id===home))move(source,home,top?row.id:null);
          }
        }}>
        <button className="row-title" onClick={()=>onOpen(row)}>{row.title}</button><span className="chip">{row.type_name}</span>
        {!!descendants.length&&<button className="text-button" onClick={()=>onBrowse(row)}>View {descendants.length} inside</button>}
        {!disabled&&<Popover label={"Move "+row.title} button={<FolderInput size={16}/>} panelClassName="filter-popover">{close=><>
          <label className="field">Main home<select aria-label={"Move "+row.title+" to"} disabled={locked} value={row.parent_id??""}
            onChange={e=>{move(row,e.target.value||null,null);close();}}><option value="">Unfiled</option>{homes.map(h=><option key={h.id} value={h.id}>{[...h.home.map(p=>p.title),h.title].join(" / ")}</option>)}</select></label>
          <div className="settings-form-actions"><button className="btn btn-sm" disabled={locked||index<1} onClick={()=>{move(row,row.parent_id,allSiblings[index-1].id);close();}}><ArrowUp size={15}/>Move up</button>
          <button className="btn btn-sm" disabled={locked||index>=allSiblings.length-1} onClick={()=>{move(row,row.parent_id,allSiblings[index+2]?.id??null);close();}}><ArrowDown size={15}/>Move down</button></div>
        </>}</Popover>}
      </div>;
      return <li key={row.id}>{descendants.length?<details open={!closed.has(row.id)} onToggle={e=>{const open=e.currentTarget.open;setClosed(old=>{if(old.has(row.id)===!open)return old;const next=new Set(old);if(open)next.delete(row.id);else next.add(row.id);return next;});}}><summary>{content}</summary>{branch(row.id,depth+1)}</details>:content}</li>;
    })}</ul>;
  };
  return <section className="organization-tree" aria-label="Organization tree"><p className="footnote">Each record has one main home. Drop above a record to reorder, onto its center to nest, or use Move. Additional links connect related records without moving them.</p>{branch(null)}</section>;
}
