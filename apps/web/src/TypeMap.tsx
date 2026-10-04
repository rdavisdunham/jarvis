import {useState,type ReactNode} from "react";
import {ArrowDown,ArrowUp,Plus} from "lucide-react";
import type {Schema} from "./structure-types";
import {placeType,reorderType,presentation} from "./field-library";
import {allowAndPlace,alsoAllowedIn,typeMoveChoice,type TypeMoveChoice} from "./blueprint-model";

/** A drop onto a type that isn't an allowed home asks instead of failing: rearrange only, or also allow. */
export function TypeMoveChoices({choice,onRearrange,onAllow,onCancel}:{choice:TypeMoveChoice;onRearrange:()=>void;onAllow:()=>void;onCancel:()=>void}){
 return <div className="type-move-choice" role="group" aria-label={choice.title}>
  <p className="type-move-title">{choice.title}</p>
  <button type="button" className="type-move-option" disabled={!choice.rearrange.enabled} onClick={onRearrange}><strong>Rearrange the diagram only</strong><small>{choice.rearrange.note}</small></button>
  <button type="button" className="type-move-option" disabled={!choice.allow.enabled} onClick={onAllow}><strong>{choice.allow.label}</strong><small>{choice.allow.note}</small></button>
  <button type="button" className="btn btn-ghost btn-sm" onClick={onCancel}>Cancel</button>
 </div>;
}

export function TypeMap({schema,selected,fieldId,onSelect,onChange,onAdd,onError}:{schema:Schema;selected:string;fieldId:string;onSelect:(type:string,field?:string)=>void;onChange:(s:Schema)=>void;onAdd:(parent?:string)=>void;onError:(s:string)=>void}){
 const [pending,setPending]=useState<{type:string;target:string}|null>(null);
 const placements=new Map(presentation(schema).map(p=>[p.type_id,p.parent_type_id]));
 const move=(id:string,parent:string|null)=>{
  if(parent&&parent!==id){const choice=typeMoveChoice(schema,id,parent);if(!choice.allowed&&!choice.cycle){setPending({type:id,target:parent});onError("");return;}}
  try{onChange(placeType(schema,id,parent));onError("");setPending(null);}catch(e){onError((e as Error).message);}
 };
 const type=schema.types.find(t=>t.id===selected);
 const branch=(parent:string|null,seen=new Set<string>()):ReactNode=><ul>{schema.types.filter(t=>(placements.get(t.id)??null)===parent&&!seen.has(t.id)).map(t=>{const also=alsoAllowedIn(schema,t.id);return <li key={t.id}>
  <div className="type-node" aria-current={selected===t.id&&!fieldId} draggable onDragStart={e=>{e.stopPropagation();e.dataTransfer.setData("application/eri-type",t.id);}} onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();e.stopPropagation();const id=e.dataTransfer.getData("application/eri-type");if(id)move(id,t.id);}}>
   <button onClick={()=>onSelect(t.id)}>{t.plural||"New type"}</button><small>{t.archived?"Archived":t.parent_types.includes(t.id)?"Can nest itself":""}</small>
  </div>
  {!!also.length&&<p className="type-also">Also allowed in {also.map(h=>h.name).join(", ")}</p>}
  {!!t.fields.length&&<details className="type-fields" open={selected===t.id}><summary><span className="type-fields-label">Fields</span><span className="detail-count">{t.fields.length}</span></summary><ul>{t.fields.map(f=><li key={f.id}><button className="field-node" aria-current={selected===t.id&&fieldId===f.id}
    draggable onDragStart={e=>{e.stopPropagation();e.dataTransfer.setData("application/eri-field",JSON.stringify({type:t.id,id:f.id}));}}
    onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();e.stopPropagation();try{const source=JSON.parse(e.dataTransfer.getData("application/eri-field"));if(source.id===f.id&&source.type===t.id)return;if(source.type!==t.id)throw Error("Reuse a field through Add existing field; dragging only reorders this type.");const fields=t.fields.filter(v=>v.id!==source.id);const item=t.fields.find(v=>v.id===source.id);if(item){fields.splice(fields.findIndex(v=>v.id===f.id),0,item);onChange({...schema,types:schema.types.map(v=>v.id===t.id?{...v,fields}:v)});}}catch(err){onError((err as Error).message);}}}
    onClick={()=>onSelect(t.id,f.id)}>{f.name||"New field"}{f.archived?" (archived)":""}</button></li>)}</ul></details>}
  {branch(t.id,new Set([...seen,t.id]))}
 </li>;})}</ul>;
 return <nav className="type-map" aria-label="Type and field tree">
  <p className="footnote">This diagram arranges definitions. Moving a type here never moves your records. Each type appears once; its other allowed homes are listed under it.</p>
  {pending&&<TypeMoveChoices choice={typeMoveChoice(schema,pending.type,pending.target)} onCancel={()=>setPending(null)}
   onRearrange={()=>move(pending.type,pending.target)} onAllow={()=>{try{onChange(allowAndPlace(schema,pending.type,pending.target));setPending(null);}catch(e){onError((e as Error).message);}}}/>}
  <div onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();const id=e.dataTransfer.getData("application/eri-type");if(id)move(id,null);}}>{branch(null)}</div>
  <div className="type-map-controls"><button className="btn btn-sm" onClick={()=>onAdd()}><Plus size={14}/>New type</button>{type&&<button className="btn btn-sm" onClick={()=>onAdd(type.id)}>Add type inside {type.name}</button>}</div>
  {type&&<div className="type-presentation-controls"><label>Diagram parent<select aria-label="Diagram parent" value={placements.get(type.id)??""} onChange={e=>move(type.id,e.target.value||null)}><option value="">Top level</option>{schema.types.filter(t=>t.id!==type.id).map(t=><option key={t.id} value={t.id}>{t.name}{type.parent_types.includes(t.id)?"":" (not allowed yet)"}</option>)}</select></label>
   <button className="btn-icon" aria-label="Move type earlier" onClick={()=>onChange(reorderType(schema,type.id,-1))}><ArrowUp size={14}/></button><button className="btn-icon" aria-label="Move type later" onClick={()=>onChange(reorderType(schema,type.id,1))}><ArrowDown size={14}/></button>
  </div>}
 </nav>;
}
