import {useEffect,useState} from "react";
import {ChevronRight,FolderInput} from "lucide-react";
import {api} from "./api";
import {Popover} from "./ux";
import type {CustomRecord,Schema} from "./structure-types";
export function HomePicker({row,schema,disabled,onChoose}:{row:CustomRecord;schema:Schema;disabled:boolean;onChoose:(id:string|null)=>void}){
 const [branch,setBranch]=useState(""),[query,setQuery]=useState(""),[items,setItems]=useState<CustomRecord[]>([]),[home,setHome]=useState<CustomRecord|null>(null),[error,setError]=useState(""),[offset,setOffset]=useState(0),[next,setNext]=useState<number|null>(null);
 const allowed=schema.types.find(t=>t.id===row.type_id)?.parent_types??[];
 useEffect(()=>setOffset(0),[branch,query]);
 useEffect(()=>{let live=true;const p=new URLSearchParams({limit:"40",offset:String(offset)});
  if(query)p.set("query",query);else if(branch)p.set("parent_id",branch);
  setError("");
  void api<{items:CustomRecord[];parent?:CustomRecord|null;next_offset:number|null}>((query?"/structure/records?":"/structure/browse?")+p).then(r=>{if(live){setItems(r.items.filter(r=>r.id!==row.id&&!r.home.some(h=>h.id===row.id)));setHome(r.parent??null);setNext(r.next_offset);}}).catch(e=>live&&setError(e.message));
  return()=>{live=false;};
 },[branch,query,offset,row.id]);
 return <Popover label="Choose main home" button={<><FolderInput size={15}/>{row.home.at(-1)?.title??"Unfiled"}</>} panelClassName="home-picker">{close=><>
  <p className="footnote">Choose one home. Its ancestors provide the rest of the path.</p>
  <input type="search" aria-label="Find a home" placeholder="Search or browse…" value={query} onChange={e=>setQuery(e.target.value)}/>
  <nav className="org-breadcrumb"><button onClick={()=>{setQuery("");setBranch("");}}>Top level</button>{home&&<><ChevronRight size={12}/><button onClick={()=>{setQuery("");setBranch(home.parent_id??"");}}>Up</button><strong>{home.title}</strong></>}</nav>
  {!branch&&!query&&<button className="menu-item" disabled={disabled} onClick={()=>{onChoose(null);close();}}>Leave unfiled</button>}
  {home&&allowed.includes(home.type_id)&&<button className="btn btn-soft" disabled={disabled} onClick={()=>{onChoose(home.id);close();}}>Use {home.title}</button>}
  {items.map(r=><div className="home-option" key={r.id}><button className="menu-item" onClick={()=>{setQuery("");setBranch(r.id);}}>{r.title}<small>{r.type_name}</small><ChevronRight size={14}/></button>
   <button className="btn btn-sm" aria-label={"Use "+r.title+" as home"} disabled={disabled||!allowed.includes(r.type_id)} onClick={()=>{onChoose(r.id);close();}}>Use</button></div>)}
  {error&&<p role="alert">{error}</p>}{!items.length&&<p className="footnote">No records here.</p>}
  <div className="org-pagination">{offset>0&&<button className="btn btn-sm" onClick={()=>setOffset(Math.max(0,offset-40))}>Previous</button>}{next!==null&&<button className="btn btn-sm" onClick={()=>setOffset(next)}>More</button>}</div>
 </>}</Popover>;
}
