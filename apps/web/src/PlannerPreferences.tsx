import {SearchAliases} from "./SearchAliases";
import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import { useStructureActions } from "./structure-actions";
import { readTheme, saveTheme, type ThemePreference } from "./theme";
import { SettingRow, SettingsGroup } from "./SettingsLayout";
type Rule={id:string;revision:number;status:string;origin:string;reason:string;condition:{phrase:string;type_id:string};evidence_ids:string[]};
type Review={id:string;revision:number;status:string;deferred_until:string|null;summary:{message?:string;activated?:string[]};questions:{id:string;question:string;evidence_count:number;held_out:number}[]};
type Understanding={definition_id:string;revision:number;status:string;questions:string[];understanding:{meaning?:string}};
type Prefs={routing_mode:string;routing_learning:boolean;routing_review_enabled:boolean;routing_review_day:number;routing_review_hour:number;work_windows:{days:number[];start:string;end:string}[];timezone:string};
type State={patterns:Rule[];reviews:Review[];understandings:Understanding[];preferences:Prefs;quality_gate:string};
export function RoutingReviewPanel(){
 const [state,setState]=useState<State|null>(null),[open,setOpen]=useState(false),[answer,setAnswer]=useState(""),[count,setCount]=useState(0),[more,setMore]=useState(false);const {run,busy,error,setError}=useStructureActions();
 const load=useCallback(()=>api<State>("/structure/routing/state").then(setState),[]);
 useEffect(()=>{void load().catch(e=>setError(e.message));},[load,setError]);
 const pending=Boolean(state?.reviews.some(r=>["queued","running"].includes(r.status))||state?.understandings.some(s=>s.status==="assessing"));
 useEffect(()=>{if(!pending)return;const timer=setInterval(()=>void load().catch(e=>setError(e.message)),5000);return()=>clearInterval(timer);},[pending,load,setError]);
 const action=async(tool:string,args:unknown)=>{await run(tool,args);await load();};
 if(!state)return <p role="status">{error||"Loading organization learning…"}</p>;
 const prefs=state.preferences;
 const change=(p:Record<string,unknown>)=>void action("settings.update",p).catch(()=>{});
 const review=state.reviews.find(r=>r.status==="pending"&&r.questions.length&&(!r.deferred_until||new Date(r.deferred_until)<=new Date()));const question=review?.questions[0];
 const understanding=state.understandings.find(s=>s.status==="needs_input");
 const days=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"];
 const candidates=state.patterns.filter(r=>r.status==="candidate").length;
 return <><SettingsGroup className="routing-preferences" title="Organization learning" description="Rules learn how you file work. Personal memories stay separate."
   action={<button className="btn btn-soft" onClick={()=>{setOpen(!open);setCount(0);setMore(false);void load();}}>{open?"Hide review":candidates?"Review "+candidates:"Review"}</button>}>
 {error&&<p role="alert" className="error-banner">{error}</p>}
 {pending&&<p role="status" className="settings-callout">Eri is reviewing your organization in the background.</p>}
 {open&&<div className="routing-review" aria-live="polite">{count>=3&&!more?<><p>Three answered. Keep going or leave the rest for later?</p><div className="settings-form-actions"><button className="btn btn-primary" onClick={()=>setMore(true)}>Continue</button><button className="btn" onClick={()=>setOpen(false)}>Done for now</button></div></>:understanding?<><h3>Help Eri understand {understanding.definition_id.split(":").at(-1)}</h3><p>{understanding.questions[0]}</p><textarea aria-label="Field clarification" value={answer} onChange={e=>setAnswer(e.target.value)}/><button className="btn btn-primary" disabled={busy||!answer.trim()} onClick={()=>void action("routing.understand",{definition_id:understanding.definition_id,expected_revision:understanding.revision,answer}).then(()=>{setAnswer("");setCount(count+1);}).catch(()=>{})}>Answer</button></>:question&&review?<><h3>{question.question}</h3><div className="chip-row" style={{margin:"4px 0 12px"}}><span className="chip">{question.evidence_count} independent examples</span><span className="chip">{question.held_out} held-out matches</span></div><div className="settings-form-actions">{[["accept","Yes, use this rule"],["dismiss","Dismiss"],["defer_today","Later today"],["defer_tomorrow","Tomorrow"],["defer_week","Next week"]].map(([value,label],i)=><button key={value} className={i===0?"btn btn-primary":"btn btn-ghost"} disabled={busy} onClick={()=>void action("routing.answer",{review_id:review.id,expected_revision:review.revision,question_id:question.id,action:value}).then(()=>setCount(count+1)).catch(()=>{})}>{label}</button>)}</div></>:<p>{state.reviews[0]?.summary.message||"No questions waiting. You can still inspect your rules below."}</p>}</div>}
 <SettingRow label="New task routing" hint="What Eri does when a new task matches a learned rule."><select aria-label="New task routing" value={prefs.routing_mode} disabled={busy} onChange={e=>change({routing_mode:e.target.value})}><option value="automatic">Apply strong matches</option><option value="suggest">Suggest only</option><option value="off">Off</option></select></SettingRow>
 <SettingRow as="label" className="switch-row" label="Learn from my organization" hint="Notice how you file tasks and propose rules."><input type="checkbox" className="switch" role="switch" checked={prefs.routing_learning} onChange={e=>change({routing_learning:e.target.checked})}/></SettingRow>
 <SettingRow as="label" className="switch-row" label="Weekly review" hint="A short set of questions about new rules."><input type="checkbox" className="switch" role="switch" checked={prefs.routing_review_enabled} onChange={e=>change({routing_review_enabled:e.target.checked})}/></SettingRow>
 <SettingRow label="Review time" hint={"Day and hour (0 to 23) in "+prefs.timezone+"."}><select aria-label="Review day" value={prefs.routing_review_day} onChange={e=>change({routing_review_day:Number(e.target.value)})}>{days.map((d,i)=><option key={d} value={i}>{d}</option>)}</select><input aria-label={"Hour ("+prefs.timezone+")"} type="number" min={0} max={23} defaultValue={prefs.routing_review_hour} onBlur={e=>change({routing_review_hour:Number(e.target.value)})}/></SettingRow>
 <SettingRow label="Run a review now" hint="Looks for new patterns in your recent filing."><button className="btn" disabled={busy||pending||!prefs.routing_learning} onClick={()=>void action("routing.run",{}).catch(()=>{})}>Run now</button></SettingRow>
 <details className="settings-block"><summary>Work context for filing</summary><p className="footnote">A weak hint for filing tasks, never a scheduling rule. End before start means an overnight shift.</p>{prefs.work_windows.map((w,i)=><div className="work-window" key={i}><label className="field"><span className="field-label-text">Days</span><select multiple value={w.days.map(String)} onChange={e=>change({work_windows:prefs.work_windows.map((x,n)=>n===i?{...x,days:Array.from(e.target.selectedOptions,o=>Number(o.value))}:x)})}>{["Mon","Tue","Wed","Thu","Fri","Sat","Sun"].map((d,n)=><option key={d} value={n}>{d}</option>)}</select></label>{(["start","end"] as const).map(key=><label className="field" key={key}><span className="field-label-text">{key==="start"?"Starts":"Ends"}</span><input type="time" defaultValue={w[key]} onBlur={e=>change({work_windows:prefs.work_windows.map((x,n)=>n===i?{...x,[key]:e.target.value}:x)})}/></label>)}<button className="btn btn-ghost" onClick={()=>change({work_windows:prefs.work_windows.filter((_,n)=>n!==i)})}>Remove</button></div>)}<button className="btn btn-soft btn-sm" style={{margin:"4px 0 8px"}} onClick={()=>change({work_windows:[...prefs.work_windows,{days:[0,1,2,3,4],start:"08:00",end:"17:00"}]})}>Add hours</button></details>
 <details className="settings-block"><summary>Rules <span className="panel-count">{state.patterns.length}</span></summary>{state.quality_gate&&<p className="footnote">{state.quality_gate}</p>}{!state.patterns.length&&<p className="settings-empty">No rules yet. Eri proposes them as it learns how you file work.</p>}<div className="settings-list">{state.patterns.map(r=><article className="settings-item" key={r.id}><div className="settings-item-main"><span className="settings-item-title">“{r.condition.phrase}”</span><span className="settings-item-meta"><span className={"chip"+(r.status==="active"?" chip-done":"")}>{r.status}</span><span className="chip">{r.origin}</span><span>{r.evidence_ids.length} examples</span></span>{r.reason&&<span className="setting-hint">{r.reason}</span>}</div><div className="settings-item-actions">{[[r.status==="active"?"pause":"activate",r.status==="active"?"Pause":"Confirm & enable"],["forget","Forget"]].map(([value,label])=><button key={value} className={value==="forget"?"btn btn-danger btn-sm":"btn btn-sm"} disabled={busy} onClick={()=>void action("routing.change",{pattern_id:r.id,expected_revision:r.revision,action:value}).catch(()=>{})}>{label}</button>)}</div></article>)}</div></details>
 </SettingsGroup>
 <SearchAliases/></>;
}
export function NotificationPreferences(){
 const [p,setP]=useState<Record<string,unknown>|null>(null);const {run,busy,error}=useStructureActions();
 useEffect(()=>{api<{preferences:Record<string,unknown>}>("/bootstrap").then(r=>setP(r.preferences)).catch(()=>{});},[]);
 const save=async(change:Record<string,unknown>)=>{const result=await run<Record<string,unknown>>("settings.update",change);setP(result);};
 if(!p)return null;
 const toggle=(key:string)=><input type="checkbox" className="switch" role="switch" disabled={busy} checked={Boolean(p[key])} onChange={e=>void save({[key]:e.target.checked}).catch(()=>{})}/>;
 return <SettingsGroup title="Alerts and quiet hours" description={"Times use "+String(p.timezone)+"."}>{error&&<p role="alert" className="error-banner">{error}</p>}
  <SettingRow as="label" className="switch-row" label="Alert me at timed task deadlines" hint="Date-only deadlines appear in your summary instead.">{toggle("deadline_alerts")}</SettingRow>
  <SettingRow as="label" className="switch-row" label="Quiet hours" hint="Only alerts explicitly marked urgent bypass them. Task priority does not.">{toggle("quiet_enabled")}</SettingRow>
  <SettingRow label="Quiet between" hint="Alerts wait until quiet time ends."><input aria-label="Quiet starts" type="time" defaultValue={String(p.quiet_start)} onBlur={e=>void save({quiet_start:e.target.value}).catch(()=>{})}/><span className="range-sep">to</span><input aria-label="Quiet ends" type="time" defaultValue={String(p.quiet_end)} onBlur={e=>void save({quiet_end:e.target.value}).catch(()=>{})}/></SettingRow>
  <SettingRow as="label" className="switch-row" label="Daily summary" hint="One message with the day's deadlines and plans.">{toggle("morning_summary")}</SettingRow>
  <SettingRow label="Summary hour" hint="Hour of the day, 0 to 23."><input aria-label="Summary hour" type="number" min={0} max={23} defaultValue={Number(p.morning_hour)} onBlur={e=>void save({morning_hour:Number(e.target.value)}).catch(()=>{})}/></SettingRow>
 </SettingsGroup>;
}

/** Light / Dark / System appearance, stored on this device only. */
export function ThemeSetting(){
 const [theme,setTheme]=useState<ThemePreference>(readTheme);
 const options:{id:ThemePreference;label:string}[]=[{id:"light",label:"Light"},{id:"dark",label:"Dark"},{id:"system",label:"System"}];
 return <div className="setting-row theme-setting"><span><strong id="theme-setting-label">Appearance</strong><small>Applies to this device. System follows your device setting.</small></span>
  <div className="segmented" role="radiogroup" aria-labelledby="theme-setting-label">
   {options.map(o=><button key={o.id} type="button" role="radio" aria-checked={theme===o.id} tabIndex={theme===o.id?0:-1}
    onClick={()=>{saveTheme(o.id);setTheme(o.id);}}
    onKeyDown={e=>{const i=options.findIndex(x=>x.id===theme);const n=e.key==="ArrowRight"||e.key==="ArrowDown"?(i+1)%3:e.key==="ArrowLeft"||e.key==="ArrowUp"?(i+2)%3:-1;if(n<0)return;e.preventDefault();saveTheme(options[n].id);setTheme(options[n].id);(e.currentTarget.parentElement?.children[n] as HTMLElement|undefined)?.focus();}}>{o.label}</button>)}
  </div></div>;
}

export function SchedulingPreferences(){
 type Window={days:number[];start:string;end:string};
 const [prefs,setPrefs]=useState<{timezone:string;scheduling_windows:Record<string,Window[]>}|null>(null);
 const {run,busy,error}=useStructureActions();
 useEffect(()=>{void api<{preferences:typeof prefs}>("/bootstrap").then(d=>setPrefs(d.preferences));},[]);
 const change=async(p:Record<string,unknown>)=>setPrefs(await run("settings.update",p));
 if(!prefs)return null;
 const days=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"];
 return <SettingsGroup title="Scheduling availability" description="Eri's planner respects these hours. Personal time is separate from work; filing hints do not restrict either. An empty group is unrestricted.">{error&&<p role="alert">{error}</p>}
 <SettingRow label="Scheduling time zone"><input aria-label="Scheduling time zone" defaultValue={prefs.timezone} onBlur={e=>void change({timezone:e.target.value}).catch(()=>{})}/></SettingRow>
 {(["work","personal"] as const).map(kind=><details className="settings-block" key={kind} open><summary>{kind==="work"?"Work":"Personal"} hours</summary>{prefs.scheduling_windows[kind].map((w,i)=>{
 const update=(patch:Partial<Window>)=>void change({scheduling_windows:{[kind]:prefs.scheduling_windows[kind].map((v,n)=>n===i?{...v,...patch}:v)}}).catch(()=>{});
 return <fieldset className="scheduling-window" key={i} disabled={busy}><legend className="sr-only">{kind} window {i+1}</legend><div className="day-choices">{days.map((day,n)=><label key={day}><input type="checkbox" checked={w.days.includes(n)} onChange={e=>update({days:e.target.checked?[...w.days,n]:w.days.filter(d=>d!==n)})}/>{day}</label>)}</div>
 {(["start","end"] as const).map(key=><label className="field" key={key}>{key==="start"?"Starts":"Ends"}<input type="time" aria-label={kind+" "+key+" "+(i+1)} defaultValue={w[key]} onBlur={e=>{if(e.target.value!==w[key])update({[key]:e.target.value});}}/></label>)}
 <button className="btn btn-ghost" onClick={()=>void change({scheduling_windows:{[kind]:prefs.scheduling_windows[kind].filter((_,n)=>n!==i)}}).catch(()=>{})}>Remove hours</button></fieldset>;})}
 <button className="btn btn-sm" disabled={busy} onClick={()=>void change({scheduling_windows:{[kind]:[...prefs.scheduling_windows[kind],{days:[0,1,2,3,4],start:"08:00",end:"17:00"}]}}).catch(()=>{})}>Add {kind} hours</button></details>)}
 <p className="footnote">End before start means overnight. Eri can make an explicit exception when you request one; it is recorded with the plan.</p></SettingsGroup>;
}
