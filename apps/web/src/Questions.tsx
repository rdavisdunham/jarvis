import {useCallback,useEffect,useRef,useState} from "react";
import {api,post} from "./api";
import {useStructureActions} from "./structure-actions";
import "./questions.css";
type Question={key:string;kind:string;revision:number;status:string;question:string;source_id:string;definition_id?:string;deferred_until?:string;candidates?:{id:string;content:string}[];evidence?:unknown;answer_tool:string;answer_args:Record<string,unknown>;delivery?:{channel:string;state:string;updated_at:string}|null};
type Page={items:Question[];total:number;next_offset:number|null;counts:Record<string,number>;learning:{runs:{kind:string;id:string|null;status:string;created_at:string|null;finished_at:string|null;result:unknown}[]}};
export function Questions({organization=false,compact=false,refresh=0}:{organization?:boolean;compact?:boolean;refresh?:number}){
 const [page,setPage]=useState<Page|null>(null),[status,setStatus]=useState("open"),[offset,setOffset]=useState(0),[error,setError]=useState("");
 const load=useCallback(async()=>{try{setPage(await api<Page>("/questions?category="+(organization?"organization":"all")+"&status="+status+"&offset="+offset));setError("");}catch(e){setError((e as Error).message);}},[organization,status,offset]);
 useEffect(()=>{void load();const timer=setInterval(()=>void load(),30000);return()=>clearInterval(timer);},[load,refresh]);
 const content=<><div className="question-toolbar"><p>Optional questions help Eri learn. They never hold up your tasks.</p><label>Show<select aria-label="Question status" value={status} onChange={e=>{setStatus(e.target.value);setOffset(0);}}>{["open","pending","deferred","resolved","stale","all"].map(s=><option key={s}>{s}</option>)}</select></label><button className="btn btn-sm" onClick={()=>void load()}>Refresh</button></div>
 {error&&<p role="alert">{error}</p>}
 {page?.items.map(q=><QuestionCard key={q.key} q={q} onChanged={load}/>)}
 {page&&!page.items.length&&<p className="empty">No questions in this view.</p>}
 {page&&(offset>0||page.next_offset!==null)&&<div className="question-pagination"><button className="btn" disabled={!offset} onClick={()=>setOffset(Math.max(0,offset-40))}>Previous questions</button><span>{page.total} questions</span><button className="btn" disabled={page.next_offset===null} onClick={()=>setOffset(page.next_offset!)}>More questions</button></div>}
 {!compact&&page&&<details className="question-learning"><summary>What has the dream sequence done?</summary><p>These are the latest recorded runs for your account. “Not run” means no run has been recorded.</p>{page.learning.runs.map(run=><article key={run.kind}><strong>{{extract_memory:"Memory extraction",embed_memory:"Memory search index",review_memory:"Memory review",review_routing:"Organization patterns",assess_field:"Field understanding"}[run.kind]??run.kind}</strong><span className="chip">{run.status.replaceAll("_"," ")}</span>{run.created_at&&<time>{new Date(run.created_at).toLocaleString()}</time>}{run.id&&<details><summary>Result and provenance</summary><small>Job {run.id}{run.finished_at?" · finished "+new Date(run.finished_at).toLocaleString():""}</small><pre>{JSON.stringify(run.result,null,2)}</pre></details>}</article>)}</details>}</>;
 return compact?<details className="organization-questions"><summary>Organization questions{page?" · "+page.total:""}</summary>{content}</details>:<section className="questions-page" aria-label="Questions">{content}</section>;
}
function QuestionCard({q,onChanged}:{q:Question;onChanged:()=>Promise<void>}){
 const [answer,setAnswer]=useState(""),{run,busy,error}=useStructureActions();
 const apply=async(tool:string,args:unknown)=>{await run(tool,args);setAnswer("");await onChanged();};
 const canAnswer=["pending","deferred"].includes(q.status);
 return <article className="question-card"><div className="question-meta"><span>{q.kind==="field"?"Field clarification":q.kind==="routing"?"Organization pattern":"Memory detail"}</span><span className="chip">{q.status}</span></div><h3>{q.question}</h3>
 {q.deferred_until&&q.status==="deferred"&&<p className="footnote">Snoozed until {new Date(q.deferred_until).toLocaleString()}. You can still answer now.</p>}
 {!!q.candidates?.length&&<ul>{q.candidates.map(c=><li key={c.id}>{c.content}</li>)}</ul>}
 {canAnswer&&<>{q.kind!=="routing"&&<label className="field">{q.kind==="memory"?"Complete corrected fact":"Your clarification"}<textarea aria-label={q.kind==="memory"?"Complete corrected fact":"Your clarification"} rows={2} value={answer} onChange={e=>setAnswer(e.target.value)}/></label>}
 <div className="question-actions">{q.kind==="memory"?<><button className="btn btn-primary" disabled={busy||!answer.trim()} onClick={()=>void apply(q.answer_tool,{...q.answer_args,action:"merge",content:answer}).catch(()=>{})}>Save corrected memory</button><button className="btn" disabled={busy} onClick={()=>void apply(q.answer_tool,{...q.answer_args,action:"distinct"}).catch(()=>{})}>These are different</button></>:q.kind==="field"?<button className="btn btn-primary" disabled={busy||!answer.trim()} onClick={()=>void apply(q.answer_tool,{...q.answer_args,answer}).catch(()=>{})}>Answer</button>:<><button className="btn btn-primary" disabled={busy} onClick={()=>void apply(q.answer_tool,{...q.answer_args,action:"accept"}).catch(()=>{})}>Use this rule</button><button className="btn" disabled={busy} onClick={()=>void apply(q.answer_tool,{...q.answer_args,action:"dismiss"}).catch(()=>{})}>Dismiss rule</button></>}
 <button className="btn btn-ghost" disabled={busy} onClick={()=>void apply("review.defer",{question_key:q.key,expected_revision:q.revision,until:"week"}).catch(()=>{})}>Ask next week</button></div></>}
 {error&&<p role="alert">{error} Your answer is kept. Refresh to see the current question.</p>}
 <details className="question-provenance"><summary>Why Eri is asking</summary><p>{q.kind==="memory"?"Two saved facts may refer to the same thing. Only your answer can merge them.":q.kind==="field"?"A field description needs clarification before Eri relies on it.":"Repeated organization choices produced a candidate rule. It is not active until confirmed or qualified by its separate quality gate."}</p>
 <small>Source: {q.source_id} · revision {q.revision}</small>{q.evidence!==undefined&&<pre>{JSON.stringify(q.evidence,null,2)}</pre>}{q.delivery&&<p>Last invitation: {q.delivery.channel} · {q.delivery.state} · {new Date(q.delivery.updated_at).toLocaleString()}. Voice forwarding does not confirm it was heard.</p>}</details></article>;
}
type Invitation={id:string;message:string;expires_at:string};
export function ReviewInvitation({enabled,conversation,onOpen}:{enabled:boolean;conversation:string|null;onOpen:()=>void}){
 const [item,setItem]=useState<Invitation|null>(null),root=useRef<HTMLDivElement>(null);
 useEffect(()=>{setItem(null);if(!enabled||!conversation)return;let live=true;let reserved:Invitation|null=null;
  const timer=setTimeout(()=>{if(document.visibilityState!=="visible")return;void post<{invitation:Invitation|null}>("/questions/reserve",{conversation_id:conversation}).then(r=>{reserved=r.invitation;if(live)setItem(reserved);else if(reserved)void post("/questions/delivery/"+reserved.id,{event:"interrupted"}).catch(()=>{});}).catch(()=>{});},8000);
  return()=>{live=false;clearTimeout(timer);if(reserved)void post("/questions/delivery/"+reserved.id,{event:"interrupted"}).catch(()=>{});};
 },[enabled,conversation]);
 useEffect(()=>{if(!item||!root.current)return;let sent=false;const observer=new IntersectionObserver(entries=>{if(sent||document.visibilityState!=="visible")return;
  for(const e of entries){if(!e.isIntersecting)continue;const r=e.intersectionRect,top=document.elementFromPoint(r.x+r.width/2,r.y+r.height/2);if(!top||!e.target.contains(top))continue;
   sent=true;void post("/questions/delivery/"+item.id,{event:"presented"}).then((r)=>{if((r as {state:string}).state!=="presented")setItem(null);}).catch(()=>setItem(null));}
 },{threshold:.75});observer.observe(root.current);const timer=setTimeout(()=>setItem(null),Math.max(0,Date.parse(item.expires_at)-Date.now()));return()=>{observer.disconnect();clearTimeout(timer);};},[item]);
 return enabled&&item?<div ref={root} className="review-invitation" role="status"><p>{item.message}</p><button className="text-button" onClick={onOpen}>Open Questions</button><button className="text-button" onClick={()=>setItem(null)}>Not now</button></div>:null;
}
