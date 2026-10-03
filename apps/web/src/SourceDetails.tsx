import { useCallback, useEffect, useRef, useState, type CSSProperties } from "react";
import { ExternalLink } from "lucide-react";
import { api } from "./api";
import { useStructureActions } from "./structure-actions";
import { SettingRow, SettingsGroup } from "./SettingsLayout";

export type SourceInfo={provider:string;label:string;context?:string;url?:string|null;sync_state?:string;identifier?:string;read_only_reason?:string;editable_fields?:string[];read_only_fields?:string[];local_only_fields?:string[];details?:Record<string,unknown>};
const defaults:Record<string,string>={eridani:"#9edac8",linear:"#e4a261",google:"#e995bd"};
export function SourceBadge({source}:{source?:SourceInfo}){
 const s=source??{provider:"eridani",label:"Eridani"};
 const state=s.sync_state&&!['synced','local'].includes(s.sync_state)?s.sync_state.replaceAll('_',' '):null;
 return <span className="chip source-badge" title={[s.label,s.context,s.sync_state].filter(Boolean).join(" · ")}
   style={{"--source-color":`var(--source-${Object.hasOwn(defaults,s.provider)?s.provider:"eridani"}, ${defaults[s.provider]??defaults.eridani})`} as CSSProperties}>
   <span aria-hidden="true" className="source-dot"/>{s.label}{state&&<span className="source-state"> · {state}</span>}</span>;
}
const fieldLabels:Record<string,string>={updatedAt:"Last updated",archivedAt:"Archived",dueDate:"Due date",state:"Status",parent:"Parent issue",team:"Team",project:"Remote project",assignee:"Assignee",identifier:"Issue",priority:"Priority",labels:"Labels"};
function providerValue(value:unknown):string {
 if(value===null||value===undefined)return "Empty";
 if(Array.isArray(value))return value.map(providerValue).join(", ")||"None";
 if(typeof value==="object"){
  const v=value as Record<string,unknown>;
  if(v.nodes)return providerValue(v.nodes);
  return String(v.name??v.title??v.identifier??"Linked record — open the source for details");
 }
 return String(value);
}
export function SourceDetails({source}:{source?:SourceInfo}){
 if(!source)return null;
 const names=(values?:string[])=>values?.map(v=>v.replaceAll("_"," ")).join(", ");
 return <section className="detail-section source-details" aria-label="Source"><div className="detail-section-head"><SourceBadge source={source}/>{source.identifier&&<strong>{source.identifier}</strong>}
 {source.url&&/^https?:\/\//i.test(source.url)&&<a href={source.url} target="_blank" rel="noreferrer">Open original <ExternalLink size={13}/></a>}</div>
 {source.context&&<p className="footnote">{source.context}</p>}{source.sync_state&&source.sync_state!=="local"&&<p role="status">Sync: {source.sync_state.replaceAll("_"," ")}</p>}
 {source.read_only_reason&&<p>{source.read_only_reason}</p>}
 {source.provider!=="eridani"&&<details><summary>Source fields & sync rules</summary><p>Edits to supported source fields update the original item. Pending changes are not yet confirmed by the source.</p><p><strong>Synced:</strong> {names(source.editable_fields)||"Read only"}</p><p><strong>Read only here:</strong> {names(source.read_only_fields)||"None"}</p><p><strong>Eridani only:</strong> {names(source.local_only_fields)}</p>{source.details&&<dl>{Object.entries(source.details).map(([key,value])=><div key={key}><dt>{fieldLabels[key]??key}</dt><dd>{providerValue(value)}</dd></div>)}</dl>}</details>}
 </section>;
}
type Guard={dirty:boolean;busy:boolean;flush:()=>Promise<unknown>};
export function useInlineGuard(){
 const current=useRef<Guard|null>(null);const [state,setState]=useState({dirty:false,busy:false});
 const update=useCallback((guard:Guard|null)=>{current.current=guard;setState(old=>old.dirty===!!guard?.dirty&&old.busy===!!guard?.busy?old:{dirty:!!guard?.dirty,busy:!!guard?.busy});},[]);
 const flush=useCallback(async()=>{await current.current?.flush();},[]);
 return {...state,update,flush};
}
/** Native task notes share their custom-record annotation, never the synced description. */
export function SourceNotes({kind,id,canEdit=true,onGuard}:{onGuard?:(guard:Guard|null)=>void;kind:"task"|"planning"|"google";id:string;canEdit?:boolean}){
 const [data,setData]=useState<Record<string,unknown>|null>(null),[draft,setDraft]=useState(""),[saved,setSaved]=useState("");
 const [comparison,setComparison]=useState<Record<string,unknown>|null>(null);const {run,busy,error,setError}=useStructureActions();
 const path=kind==="task"?"/structure/by-core/task/"+id:kind==="google"?"/calendar/events/"+id:"/planning/"+id;
 const fetch=()=>api<Record<string,unknown>>(path);
 useEffect(()=>{let active=true;void fetch().then(d=>{if(active){setData(d);setDraft(String(d.local_notes??""));setSaved(String(d.local_notes??""));}}).catch(e=>{if(active)setError(e.message);});return()=>{active=false;};},[path,setError]);
 const flight=useRef<Promise<unknown>|null>(null);
 const [saving,setSaving]=useState(false);
 const latest=useRef({data,draft,saved});latest.current={data,draft,saved};
 const save=(base=latest.current.data):Promise<unknown>=>{
  if(flight.current)return flight.current;
  if(!base)return Promise.resolve();
  const text=latest.current.draft;setSaving(true);setError("");
  const operation=(async()=>{
   if(base===latest.current.data){const fresh=await fetch();if(String(fresh.local_notes??"")===latest.current.saved)base=fresh;}
   const args=kind==="task"?{record_id:base!.id,expected_revision:base!.revision,schema_revision:base!.schema_revision}:kind==="google"?{event_id:id,expected_revision:base!.annotation_revision}:{entry_id:id,expected_revision:base!.revision};
   const result=await run<Record<string,unknown>>(kind==="task"?"record.update":kind==="google"?"calendar.annotate":"planning.annotate",{...args,local_notes:text});
   const next={...base,...result};latest.current={...latest.current,data:next,saved:text};
   setData(next);setSaved(text);setComparison(null);
  })().catch(e=>{setError(e.message??"Could not save local notes.");throw e;}).finally(()=>{flight.current=null;setSaving(false);});
  flight.current=operation;return operation;
 };
 useEffect(()=>{onGuard?.({dirty:draft!==saved,busy:busy||saving,flush:async()=>{if(flight.current)await flight.current;if(latest.current.draft===latest.current.saved)return;if(error)throw new Error(error);await save();}});},[draft,saved,busy,saving,error,data,onGuard]);
 useEffect(()=>()=>onGuard?.(null),[onGuard]);
 return <details className="detail-section"><summary>Eridani-only notes{draft!==saved?" · Unsaved":""}</summary><p className="footnote">Visible in this workspace; never sent to the source.{kind==="google"?" Notes on a recurring event belong to the cached event or series, not each projected occurrence.":""}</p>
 {error&&<div role="alert"><p>{error} Your draft is kept.</p><button className="btn btn-sm" onClick={()=>void fetch().then(setComparison).catch(e=>setError(e.message))}>Compare saved notes</button>{comparison&&<><p>{String(comparison.local_notes??"Empty")}</p><button className="btn btn-sm" disabled={busy||!canEdit} onClick={()=>void save(comparison).catch(()=>{})}>Reapply my notes</button><button className="btn btn-sm" onClick={()=>{setData(comparison);setDraft(String(comparison.local_notes??""));setSaved(String(comparison.local_notes??""));setComparison(null);setError("");}}>Use saved notes</button></>}</div>}
 <textarea rows={3} aria-label="Eridani-only notes" maxLength={30000} value={draft} disabled={!canEdit||busy||saving||!data} onChange={e=>setDraft(e.target.value)} onBlur={()=>{if(draft!==saved)void save().catch(()=>{});}}/>
 </details>;
}
export function SourceColorSettings(){
 const [colors,setColors]=useState(defaults);const {run,busy,error}=useStructureActions();
 useEffect(()=>{void api<{preferences:{source_colors?:Record<string,string>}}>("/bootstrap").then(d=>setColors({...defaults,...d.preferences.source_colors}));},[]);
 return <SettingsGroup title="Source colors" description="Colors identify where an item lives; labels remain visible.">{error&&<p role="alert">{error}</p>}{Object.entries(defaults).map(([key])=><SettingRow key={key} label={key==="google"?"Google Calendar":key==="linear"?"Linear":"Eridani"}><input type="color" aria-label={key+" source color"} value={colors[key]} disabled={busy} onChange={e=>{const value=e.target.value;void run<{source_colors:Record<string,string>}>("settings.update",{source_colors:{[key]:value}}).then(p=>{setColors(p.source_colors);document.documentElement.style.setProperty("--source-"+key,value);}).catch(()=>{});}}/></SettingRow>)}</SettingsGroup>;
}
