import {useState} from "react";
export function NoticeSnooze({onSnooze}:{onSnooze:(args:{minutes?:number;until?:string})=>Promise<unknown>}){
 const [until,setUntil]=useState(""),[busy,setBusy]=useState(false),[error,setError]=useState("");
 const save=async(args:{minutes?:number;until?:string})=>{setBusy(true);setError("");try{await onSnooze(args);}catch(e){setError((e as Error).message);}finally{setBusy(false);}};
 return <details className="notice-snooze"><summary>Snooze…</summary><div className="structure-controls">{[[10,"10 min"],[60,"1 hour"],[180,"3 hours"]].map(([n,label])=><button key={String(n)} disabled={busy} onClick={()=>void save({minutes:Number(n)})}>{label}</button>)}<label>Local date and time<input type="datetime-local" aria-label="Snooze until" value={until} onChange={e=>setUntil(e.target.value)}/></label><button disabled={!until||busy} onClick={()=>void save({until})}>Snooze until then</button></div>{error&&<p role="alert">{error}</p>}</details>;
}
