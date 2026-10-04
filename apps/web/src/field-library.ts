import type {Schema,SchemaField,SchemaType} from "./structure-types";
import {meanings,describe} from "./structure-types";
const semantic=["name","description","kind","options","target_types","multiple","binding"] as const;
const meaning=(field:SchemaField)=>Object.fromEntries(semantic.map(k=>[k,field[k]]));
export function updateType(schema:Schema,id:string,patch:Partial<SchemaType>):Schema{
 const draft=structuredClone(schema),type=draft.types.find(t=>t.id===id)!;
 const library=draft.field_library??=[];
 if(patch.fields){
  patch={...patch,fields:structuredClone(patch.fields)};
  for(const field of patch.fields!){
   const old=type.fields.find(f=>f.id===field.id);
   // Disabling a behavior retires this attachment without changing the shared field.
   if(patch.capabilities&&old?.binding!==field.binding){field.library_id=null;}
   if(!field.library_id){field.library_id=crypto.randomUUID();library.push({...structuredClone(field),id:field.library_id,library_id:null,inherit:false,archived:false,visible:true});}
   else if(old&&JSON.stringify(meaning(old))!==JSON.stringify(meaning(field))){
    const shared=library.find(f=>f.id===field.library_id);
    if(shared)Object.assign(shared,meaning(field));
   }
  }
 }
 Object.assign(type,patch);
 for(const t of draft.types)for(const f of t.fields){
  const shared=library.find(l=>l.id===f.library_id);if(shared)Object.assign(f,meaning(shared));
 }
 draft.field_library=library;
 draft.type_layout=draft.type_layout?.map(p=>p.type_id===id&&p.parent_type_id&&!type.parent_types.includes(p.parent_type_id)?{...p,parent_type_id:null}:p);
 return draft;
}
export function canAttach(field:SchemaField,type:SchemaType){
 if(type.fields.some(f=>f.library_id===field.id||field.binding&&f.binding===field.binding))return false;
 if(!field.binding)return !field.archived;
 const capability=["start_date","target_date"].includes(field.binding)?"timeline":field.binding.startsWith("metric_")?"metric":"work";
 return type.capabilities.includes(capability)&&!field.archived;
}
export function attachField(field:SchemaField):SchemaField{
 return {...structuredClone(field),id:field.binding??crypto.randomUUID(),library_id:field.id,inherit:false,archived:false,visible:true};
}
export function placeType(schema:Schema,id:string,parent:string|null):Schema{
 if(parent===id)throw Error("A type cannot appear inside itself in the diagram. Same-type records are still allowed.");
 const type=schema.types.find(t=>t.id===id)!;
 if(parent&&!type.parent_types.includes(parent))throw Error("Choose an allowed home type, or edit allowed homes first.");
 const placements=new Map(presentation(schema).map(p=>[p.type_id,p.parent_type_id]));
 placements.set(id,parent);const seen=new Set([id]);let next=parent;
 while(next){if(seen.has(next))throw Error("This would create a loop in the diagram.");seen.add(next);next=placements.get(next)??null;}
 return {...schema,type_layout:[...placements].map(([type_id,parent_type_id])=>({type_id,parent_type_id}))};
}
export function reorderType(schema:Schema,id:string,direction:number):Schema{
 const types=[...schema.types],index=types.findIndex(t=>t.id===id),other=index+direction;
 if(other<0||other>=types.length)return schema;
 [types[index],types[other]]=[types[other],types[index]];return {...schema,types};
}

/** Initial illustration only; a saved arrangement always takes precedence. */
export function presentation(schema:Schema){
 if(schema.type_layout?.length)return schema.type_layout;
 const defaults:Record<string,string>={area:"space",client:"space",project:"client",task:"project",note:"project",goal:"space"};
 return schema.types.map(t=>({type_id:t.id,parent_type_id:schema.types.some(p=>p.id===defaults[t.id])&&t.parent_types.includes(defaults[t.id])?defaults[t.id]:null}));
}

const WORK_BINDINGS=["due_date","due_time","due_timezone","planned_date","priority","estimate_minutes","assignee"];
const owns=(name:string,binding:string|null)=>!!binding&&(name==="work"?WORK_BINDINGS.includes(binding):name==="timeline"?["start_date","target_date"].includes(binding):name==="metric"&&binding.startsWith("metric_"));
/** Turning a behavior on attaches its shared fields (and default statuses for work); off retires them. */
export function capabilityPatch(type:SchemaType,schema:Schema,original:Schema,name:string,enabled:boolean):Partial<SchemaType>{
 const fields=type.fields.map(f=>({...f}));
 if(!enabled){for(const f of fields)if(owns(name,f.binding)){f.binding=null;f.archived=true;}}
 if(enabled){const source=schema.field_library??original.types.flatMap(t=>t.fields);for(const f of source){if(owns(name,f.binding)&&!fields.some(x=>x.binding===f.binding)){const added=schema.field_library?attachField(f):structuredClone(f);if(fields.some(x=>x.id===added.id))added.id=crypto.randomUUID();fields.push(added);}}}
 return {capabilities:enabled?[...type.capabilities,name]:type.capabilities.filter(c=>c!==name),fields,statuses:name==="work"&&enabled&&!type.statuses.length?meanings.map(s=>({id:s,name:describe(s),meaning:s})):type.statuses};
}
