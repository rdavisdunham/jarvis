import {useEffect,useState} from "react";
import {ChevronRight,FolderOpen,Info,Plus} from "lucide-react";
import {api} from "./api";
import {SourceBadge} from "./SourceDetails";
import {openTarget,type CustomRecord,type Schema} from "./structure-types";

type Page={parent:CustomRecord|null;items:CustomRecord[];total:number;next_offset:number|null};
export function OrganizationBrowse({schema,parent,section,onSection,query,archived,status,refresh,canEdit,onBrowse,onOpen,onCreate,onVisible}:{
 schema:Schema;parent:string;section:string;onSection:(s:string)=>void;query:string;archived:boolean;status:string;refresh:number;canEdit:boolean;
 onBrowse:(id:string)=>void;onOpen:(r:CustomRecord)=>void;onCreate:(type:string,title:string)=>Promise<void>;onVisible?:(ids:string[])=>void;
}){
 const [page,setPage]=useState<Page|null>(null),[error,setError]=useState(""),[offset,setOffset]=useState(0),[busy,setBusy]=useState(false);
 const [title,setTitle]=useState(""),[kind,setKind]=useState(""),[adding,setAdding]=useState(false);
 useEffect(()=>setOffset(0),[parent,section,query,archived,status]);
 useEffect(()=>{let live=true;setBusy(true);setError("");
  const p=new URLSearchParams({section:section==="related"?"all":section,scope:section==="related"?"related":"children",archived:String(archived),status,query,limit:"40",offset:String(offset)});
  if(parent)p.set("parent_id",parent);
  void api<Page>("/structure/browse?"+p).then(v=>{if(live){setPage(v);onVisible?.(v.items.map(r=>r.task_id??r.id));}}).catch(e=>live&&setError(e.message)).finally(()=>live&&setBusy(false));
  return()=>{live=false;};
 },[parent,section,query,archived,status,offset,refresh]);
 const types=schema.types.filter(t=>!t.archived&&(!page?.parent||t.parent_types.includes(page.parent.type_id)));
 const selected=types.find(t=>t.id===kind)??types.find(t=>t.id==="task")??types[0];
 const children=page?.items??[];
 return <section className="organization-browse" aria-label="Browse organization">
  <nav className="org-breadcrumb" aria-label="Organization path"><button onClick={()=>onBrowse("")}>Organization</button>
   {page?.parent?.home.map(h=><span key={h.id}><ChevronRight size={13}/><button onClick={()=>onBrowse(h.id)}>{h.title}</button></span>)}
   {page?.parent&&<span><ChevronRight size={13}/><strong>{page.parent.title}</strong></span>}
  </nav>
  <div className="org-container-head"><div><h2>{page?.parent?.title??(section==="all"?"Unfiled":"Your organization")}</h2>
   <p>{page?.parent?.body||(!parent?"Open a group to find the work and notes that live there.":"Direct contents of this home. Extra links live under Related.")}</p>
   {!!page?.parent?.contents?.work_total&&<div className="org-progress"><span className="chip">{page.parent.contents.work_done}/{page.parent.contents.work_total} complete</span>{page.parent.contents.ready_to_complete&&<span className="chip chip-done">Ready for your final check</span>}</div>}</div>
   {page?.parent&&<button className="btn btn-sm" onClick={()=>onOpen(page.parent!)}><Info size={15}/>Details</button>}
   {canEdit&&!archived&&<button className="btn btn-soft btn-sm" onClick={()=>setAdding(v=>!v)}><Plus size={15}/>Add here</button>}
  </div>
  <div className="org-tabs" role="group" aria-label="Container contents">{(parent?[["groups","Groups"],["work","Tasks"],["content","Notes"],["related","Related"],["all","All contents"]]:[["groups","Groups"],["all","Unfiled"]]).map(([id,label])=>
   <button key={id} aria-pressed={section===id} onClick={()=>onSection(id)}>{label}</button>)}</div>
  {!parent&&section==="all"&&<p className="footnote">Records without a home, including top-level groups.</p>}
  {adding&&selected&&<form className="org-add" onSubmit={e=>{e.preventDefault();setBusy(true);void onCreate(selected.id,title).then(()=>{setTitle("");setAdding(false);}).catch(e=>setError(e.message)).finally(()=>setBusy(false));}}>
   <select aria-label="Type to add" value={selected.id} onChange={e=>setKind(e.target.value)}>{types.map(t=><option key={t.id} value={t.id}>{t.name}</option>)}</select>
   <input aria-label="New record title" placeholder={"New "+selected.name.toLowerCase()} value={title} maxLength={500} onChange={e=>setTitle(e.target.value)}/>
   <button className="btn btn-primary" disabled={busy||!title.trim()}>Add</button>
  </form>}
  {error&&<p role="alert">{error}</p>}
  <p className="org-count" aria-live="polite">{busy?"Loading…":page?.total+" "+(section==="related"?"linked records":"records in this view")}</p>
  <div className="org-tiles">{children.map(r=>{
   const container=openTarget(r,schema)==="contents";
   return <article key={r.id} className="org-tile" data-record-id={r.id}>
    <div className="org-tile-kind"><span>{r.type_name}</span><SourceBadge source={r.source}/></div>
    <button className="org-tile-title" onClick={()=>container?onBrowse(r.id):onOpen(r)}>{container&&<FolderOpen size={17}/>}<strong>{r.title}</strong>{container&&<ChevronRight size={16}/>}</button>
    {r.body&&<p>{r.body}</p>}
    <div className="org-tile-foot">{r.contents?.work_total?<span>{r.contents.work_done}/{r.contents.work_total} complete</span>:<span>{r.contents?.direct??0} directly inside</span>}
     {r.status_meaning&&<span className="chip">{r.status_meaning.replaceAll("_"," ")}</span>}
     <button className="btn-icon" aria-label={"Details for "+r.title} onClick={()=>onOpen(r)}><Info size={16}/></button></div>
   </article>;
  })}</div>
  {!busy&&!error&&!children.length&&<p className="empty">Nothing here yet. Choose another view or add a record here.</p>}
  {page&&(offset>0||page.next_offset!==null)&&<div className="org-pagination"><button className="btn" disabled={busy||!offset} onClick={()=>setOffset(Math.max(0,offset-40))}>Previous</button><span>{offset+1}–{Math.min(offset+40,page.total)} of {page.total}</span><button className="btn" disabled={busy||page.next_offset===null} onClick={()=>setOffset(page.next_offset!)}>Next</button></div>}
 </section>;
}
