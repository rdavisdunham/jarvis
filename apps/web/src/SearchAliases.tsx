import {useEffect,useState} from "react";
import {api,post} from "./api";
type Alias={id:string;phrase:string;label:string;target_key:string;status:string;revision:number;positive_count:number;available:boolean;sources:{search_id:string;query:string;signal:string;outcome:string}[]};
type State={learning:boolean;enabled:boolean;items:Alias[];targets:{key:string;label:string;kind:string}[];index:{status:string;documents:number}};
export function SearchAliases(){
 const [state,setState]=useState<State|null>(null),[error,setError]=useState(""),[busy,setBusy]=useState(false),[editing,setEditing]=useState<string|null>(null),[target,setTarget]=useState("");
 const load=()=>api<State>("/search/aliases").then(setState);
 useEffect(()=>{void load().catch(e=>setError(e.message));},[]);
 async function act(fn:()=>Promise<unknown>){setBusy(true);setError("");try{await fn();await load();}catch(e){setError((e as Error).message);}finally{setBusy(false);}}
 return <section className="search-alias-settings"><h3>Search aliases</h3><p>The names and phrases you use to find things. These help search; organization rules still need review.</p>{error&&<p role="alert">{error}</p>}
 {state&&<><label className="check-label"><input type="checkbox" checked={state.learning} disabled={busy} onChange={e=>void act(()=>post("/search/preferences",{learning:e.target.checked}))}/>Learn from searches I use or continue from</label>
 <p className="subtle">{state.enabled?(state.index.status==="ready"?"Search index is up to date.":"Semantic search is catching up; text search remains available."):"Semantic search is being prepared."}</p>
 {!state.items.length&&<p>No aliases learned yet.</p>}
 {state.items.map(a=><article className="routing-rule" key={a.id}><strong>“{a.phrase}” → {a.label}</strong><span>{a.available?a.status:"Needs review"} · {a.positive_count} positive interactions</span><div className="structure-controls">
 <button disabled={busy} onClick={()=>{setEditing(editing===a.id?null:a.id);setTarget(a.target_key);}}>Correct</button>
 <button disabled={busy||!a.available} onClick={()=>void act(()=>post("/search/aliases/"+a.id,{expected_revision:a.revision,action:"confirm"}))}>Confirm</button>
 <button disabled={busy} onClick={()=>void act(()=>post("/search/aliases/"+a.id,{expected_revision:a.revision,action:"pause"}))}>Pause</button>
 <button disabled={busy} onClick={()=>void act(()=>post("/search/aliases/"+a.id,{expected_revision:a.revision,action:"forget"}))}>Forget</button></div>
 {editing===a.id&&<div className="structure-controls"><label>Meaning<select aria-label="Alias meaning" value={target} onChange={e=>setTarget(e.target.value)}>{state.targets.map(t=><option key={t.key} value={t.key}>{t.label} · {t.kind}</option>)}</select></label><button disabled={busy||!target} onClick={()=>void act(()=>post("/search/aliases/"+a.id,{expected_revision:a.revision,action:"correct",target_key:target})).then(()=>setEditing(null))}>Save correction</button></div>}
 <details><summary>Sources</summary>{a.sources.map(s=><p key={s.search_id}>{s.query}<small> · {s.outcome}{s.signal?.startsWith("continued")?" after continuing the conversation":""}</small></p>)}</details></article>)}</>}
 </section>;
}
