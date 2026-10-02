import {useEffect,useState} from "react";
import {api,post} from "./api";
import {SettingRow} from "./SettingsLayout";
type Alias={id:string;phrase:string;label:string;target_key:string;status:string;revision:number;positive_count:number;available:boolean;sources:{search_id:string;query:string;signal:string;outcome:string}[]};
type State={learning:boolean;enabled:boolean;items:Alias[];targets:{key:string;label:string;kind:string}[];index:{status:string;documents:number}};
export function SearchAliases(){
 const [state,setState]=useState<State|null>(null),[error,setError]=useState(""),[busy,setBusy]=useState(false),[editing,setEditing]=useState<string|null>(null),[target,setTarget]=useState("");
 const load=()=>api<State>("/search/aliases").then(setState);
 useEffect(()=>{void load().catch(e=>setError(e.message));},[]);
 async function act(fn:()=>Promise<unknown>){setBusy(true);setError("");try{await fn();await load();}catch(e){setError((e as Error).message);}finally{setBusy(false);}}
 return <section className="settings-panel search-alias-settings"><header className="settings-panel-head"><div><h3 className="panel-title">Search aliases</h3><p>The names and phrases you use to find things. These help search; organization rules still need review.</p></div></header>{error&&<p role="alert" className="error-banner">{error}</p>}
 {state&&<><SettingRow as="label" className="switch-row" label="Learn from searches" hint={state.enabled?(state.index.status==="ready"?"Search index is up to date.":"Semantic search is catching up; text search remains available."):"Semantic search is being prepared."}><input type="checkbox" className="switch" role="switch" aria-label="Learn from searches I use or continue from" checked={state.learning} disabled={busy} onChange={e=>void act(()=>post("/search/preferences",{learning:e.target.checked}))}/></SettingRow>
 {!state.items.length&&<p className="settings-empty">No aliases learned yet.</p>}
 <div className="settings-list">{state.items.map(a=><article className="settings-item" key={a.id}><div className="settings-item-main"><strong className="settings-item-title">“{a.phrase}” finds {a.label}</strong><span className="settings-item-meta"><span className={"chip"+(a.available&&a.status==="confirmed"?" chip-done":"")}>{a.available?a.status:"Needs review"}</span><span>{a.positive_count} positive interactions</span></span></div><div className="settings-item-actions">
 <button className="btn btn-ghost btn-sm" disabled={busy} onClick={()=>{setEditing(editing===a.id?null:a.id);setTarget(a.target_key);}}>Correct</button>
 <button className="btn btn-sm" disabled={busy||!a.available} onClick={()=>void act(()=>post("/search/aliases/"+a.id,{expected_revision:a.revision,action:"confirm"}))}>Confirm</button>
 <button className="btn btn-ghost btn-sm" disabled={busy} onClick={()=>void act(()=>post("/search/aliases/"+a.id,{expected_revision:a.revision,action:"pause"}))}>Pause</button>
 <button className="btn btn-danger btn-sm" disabled={busy} onClick={()=>void act(()=>post("/search/aliases/"+a.id,{expected_revision:a.revision,action:"forget"}))}>Forget</button></div>
 <div className="settings-item-extra">{editing===a.id&&<div className="settings-form" style={{borderTop:0,paddingTop:4}}><label className="field"><span className="field-label-text">Meaning</span><select aria-label="Alias meaning" value={target} onChange={e=>setTarget(e.target.value)}>{state.targets.map(t=><option key={t.key} value={t.key}>{t.label} ({t.kind})</option>)}</select></label><div className="settings-form-actions"><button className="btn btn-primary" disabled={busy||!target} onClick={()=>void act(()=>post("/search/aliases/"+a.id,{expected_revision:a.revision,action:"correct",target_key:target})).then(()=>setEditing(null))}>Save correction</button></div></div>}
 {!!a.sources.length&&<details className="alias-sources"><summary>Sources</summary>{a.sources.map(s=><p key={s.search_id} className="setting-hint">{s.query} <span className="chip">{s.outcome}{s.signal?.startsWith("continued")?" after continuing the conversation":""}</span></p>)}</details>}</div></article>)}</div></>}
 </section>;
}
