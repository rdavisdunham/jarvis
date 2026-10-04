import { lazy, Suspense, useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import {
  Bookmark, Building2, CalendarClock, ChartNoAxesGantt, Check, ChevronLeft, ChevronRight, CircleCheck, Columns3, Compass,
  Flag, FolderKanban, GripVertical, Layers, Link, List, ListFilter, NotebookPen, Orbit, Plus, Search, Settings2, Shapes, Target,
  Trash2, User, X, type LucideIcon,
} from "lucide-react";
import { api, post } from "./api";
import { useStructureActions } from "./structure-actions";
import { OrganizationBrowse } from "./OrganizationBrowse";
import { ContentsDialog } from "./ContentsDialog";
import "./organization.css";
import { nestedRows, childSpan } from "./nested-timeline";
import { OrganizationTree } from "./OrganizationTree";
import { SourceBadge } from "./SourceDetails";
import {QuickListDetail} from "./QuickLists";
import { RecordCard } from "./RecordCard";
import { StructureEditor } from "./StructureEditor";
import type { Schema, CustomRecord, Proposal } from "./structure-types";
import "./tasks.css";
import { useSemanticSearch } from "./semantic-search";
import { meanings, describe, openTarget } from "./structure-types";
import { Popover, priorityLabels, useMaxWidth } from "./ux";
import { clockLabel, dueBucket, dueBuckets, monthDay, shortDate } from "./work-views";
import { shiftDate } from "./workspace";

type Control = {nonce:string;type_id?:string;parent_id?:string;layout?:string;group?:string;record_id?:string;record_ids?:string[];search_id?:string;proposal_id?:string;field?:string;value?:string;status?:string;archived?:boolean;design?:boolean;section?:string};
type SavedEntry = {id:string;name:string;revision:number;state:Record<string,unknown>};
type Sort = "default" | "due" | "planned" | "priority" | "title";
const sortOptions: [Sort, string][] = [["due", "Due date"], ["planned", "Planned day"], ["priority", "Priority"], ["title", "Title"], ["default", "Recently changed"]];
const typeIcons: Record<string, LucideIcon> = {task: CircleCheck, project: FolderKanban, client: Building2, space: Layers, area: Compass, goal: Target, note: NotebookPen, person: User, actor: User};
export const typeIcon = (typeId: string) => typeIcons[typeId] ?? Shapes;
const TIMELINE_DAYS = 30;
/** The Atlas is a desktop and tablet canvas; phones keep Browse. Loaded on demand to keep the main bundle small. */
const AtlasView = lazy(() => import("./AtlasView"));
export const ATLAS_MIN_WIDTH = 720;


export function StructureWorkspace({selecting=false,selectedIds=[],onSelecting,onSelection,onBulk,query="",canEdit=true,canDesign=true,refresh=0,onChanged,capability="",tab="all",today=new Date().toISOString().slice(0,10),onTask,onVisible,control,onQuery,onTab,statusFilter,homeFilter,layoutFilter,groupFilter,sortFilter,onContext,tabs,searchLabel,viewLink,onApplyTaskView}:{selecting?:boolean;selectedIds?:string[];onSelecting?:(v:boolean)=>void;onSelection?:(ids:string[])=>void;onBulk?:()=>void;statusFilter?:string;homeFilter?:string;layoutFilter?:string;groupFilter?:string;sortFilter?:string;onContext?:(state:Record<string,string|number|null>)=>void;query?:string;canEdit?:boolean;canDesign?:boolean;refresh?:number;onChanged?:()=>void;capability?:string;tab?:string;today?:string;onTask?:(id:string)=>void;onQuery?:(value:string)=>void;onTab?:(value:"today"|"inbox"|"week"|"all")=>void;onVisible?:(ids:string[])=>void;control?:Control;
  /** Leading toolbar content, such as the task tabs. */ tabs?:ReactNode; searchLabel?:string;
  /** Returns a shareable link for the current view. */ viewLink?:()=>string;
  /** Applies a saved task view that was not saved from a collection. */ onApplyTaskView?:(state:Record<string,unknown>)=>void}){
  const [schema,setSchema]=useState<Schema|null>(null);const [items,setItems]=useState<CustomRecord[]>([]);
  const [resultIds,setResultIds]=useState<string[]|null>(null);
  const [resultSearch,setResultSearch]=useState<string|null>(null);
  const [typeId,setTypeId]=useState("");const [layout,setLayout]=useState(capability?"list":"browse");const [parent,setParent]=useState("");
  const [browseSection,setBrowseSection]=useState("groups");
  const [browseRefresh,setBrowseRefresh]=useState(0);
  const [moving,setMoving]=useState<{row:CustomRecord;parentId:string|null}|null>(null);
  const [atlasTarget,setAtlasTarget]=useState<{nonce:string;record_id?:string}|null>(null);const [designDraft,setDesignDraft]=useState<{draft:Schema|null;type?:string}|null>(null);
  const narrow=useMaxWidth(ATLAS_MIN_WIDTH-1);
  const [selected,setSelected]=useState<CustomRecord|null>(null);const [design,setDesign]=useState(false);const [designDirty,setDesignDirty]=useState(false);
  const [title,setTitle]=useState("");const [archived,setArchived]=useState(false);const [group,setGroup]=useState("status");
  const [proposal,setProposal]=useState<Proposal>();
  const [status,setStatus]=useState(capability?"active":"all"),[filterField,setFilterField]=useState(""),[filterValue,setFilterValue]=useState("");
  const [sort,setSort]=useState<Sort>(capability?"due":"default");
  const [saved,setSaved]=useState<SavedEntry[]>([]),[viewName,setViewName]=useState(""),[viewNotice,setViewNotice]=useState("");
  const [timelineStart,setTimelineStart]=useState(today);
  const [timelineClosed,setTimelineClosed]=useState<Set<string>>(new Set());
  const captureInput=useRef<HTMLInputElement>(null);
  const {run,error,busy,setError}=useStructureActions();
  const load=useCallback(async()=>{
    const s=await api<Schema>("/structure");const all:CustomRecord[]=[];let offset:number|null=0;
    while(offset!==null && ((layout!=="browse"&&layout!=="atlas")||!!query)){const page: {items:CustomRecord[];next_offset:number|null}=await api("/structure/records?limit=200&offset="+offset+"&archived="+archived);all.push(...page.items);offset=page.next_offset;}
    setSchema(s);setItems(all);
    const views=await api<{items:SavedEntry[]}>("/task-views");setSaved(views.items.filter(v=>v.state.collection_view||onApplyTaskView));
  },[archived,!!onApplyTaskView,layout,!!query]);
  useEffect(()=>{let live=true;load().catch(e=>{if(live)setError(String(e.message??e));});return()=>{live=false;};},[load,refresh,setError]);
  useEffect(()=>{const handler=(event:Event)=>{const id=(event as CustomEvent).detail?.id;if(id)api<CustomRecord>("/structure/records/"+id).then(setSelected).catch(e=>setError(e.message));};window.addEventListener("eri-open-record",handler);return()=>window.removeEventListener("eri-open-record",handler);},[setError]);
  useEffect(()=>{if(!control)return;if(control.record_ids!==undefined){setResultIds(control.record_ids.length?control.record_ids:null);setResultSearch(control.search_id??null);}else if(control.type_id!==undefined||control.parent_id!==undefined||control.field!==undefined){setResultIds(null);setResultSearch(null);}if(control.section!==undefined)setBrowseSection(control.section);else if(control.parent_id!==undefined)setBrowseSection(control.parent_id?"all":"groups");if(!control.layout&&!capability){if(control.type_id||control.field||control.record_ids?.length)setLayout("list");else if(control.parent_id!==undefined)setLayout("browse");}if(control.design!==undefined)setDesign(control.design);if(control.status!==undefined)setStatus(control.status);if(control.archived!==undefined)setArchived(control.archived);if(control.type_id!==undefined)setTypeId(control.type_id);if(control.parent_id!==undefined)setParent(control.parent_id);if(control.layout)setLayout(control.layout);if(control.group)setGroup(control.group);if(control.field!==undefined)setFilterField(control.field);if(control.value!==undefined)setFilterValue(control.value);if(control.record_id&&control.layout==="atlas")setAtlasTarget({nonce:control.nonce,record_id:control.record_id});else if(control.record_id)void api<CustomRecord>("/structure/records/"+control.record_id).then(setSelected).catch(e=>setError(e.message));if(control.proposal_id)void api<Proposal>("/structure/proposals/"+control.proposal_id).then(p=>{setProposal(p);setDesign(true);}).catch(e=>setError(e.message));},[control,setError]);

  useEffect(()=>{if(statusFilter!==undefined)setStatus(statusFilter);},[statusFilter]);
  useEffect(()=>{if(homeFilter!==undefined)setParent(homeFilter);},[homeFilter]);
  useEffect(()=>{if(layoutFilter)setLayout(layoutFilter);},[layoutFilter]);
  useEffect(()=>{if(groupFilter)setGroup(groupFilter==="project"?"parent":groupFilter);},[groupFilter]);
  // A sort chosen elsewhere (Eri, a saved view) applies once it changes; the page keeps its own default until then.
  const firstSort=useRef(true);
  useEffect(()=>{if(firstSort.current){firstSort.current=false;return;}if(sortFilter)setSort(sortFilter==="updated"?"default":sortFilter as Sort);},[sortFilter]);
  useEffect(()=>{onContext?.({section:browseSection,type_id:typeId,parent_id:parent,layout,group,status,archived:Number(archived),design:Number(design),design_dirty:Number(design&&designDirty),field:filterField,value:filterValue,record_id:selected?.id??null,schema_revision:schema?.revision??0});},[browseSection,typeId,parent,layout,group,status,archived,design,designDirty,filterField,filterValue,selected?.id,schema?.revision,onContext]);
  const types=(schema?.types??[]).filter(t=>!t.archived&&(!capability||t.capabilities.includes(capability)));const type=types.find(t=>t.id===typeId);
  const captureType=type??types.find(t=>t.id==="task")??types[0];
  const weekEnd=new Date(today+"T12:00:00Z");weekEnd.setUTCDate(weekEnd.getUTCDate()+6);const end=weekEnd.toISOString().slice(0,10);
  const semantic=useSemanticSearch(query,capability,archived,refresh);
  useEffect(()=>{if(query){setResultIds(null);setResultSearch(null);}},[query]);
  const ranked=semantic.result?new Map(semantic.result.ids.map((id,i)=>[id,i])):null;
  const typeOf=(r:CustomRecord)=>schema?.types.find(t=>t.id===r.type_id);
  const bound=(r:CustomRecord,binding:string)=>{const id=typeOf(r)?.fields.find(f=>f.binding===binding)?.id;const value=id?r.values[id]:undefined;return value===undefined||value===null?"":String(value);};
  const effectiveDate=(r:CustomRecord)=>sort==="planned"?bound(r,"planned_date"):bound(r,"due_date")||bound(r,"planned_date");
  const visible=items.filter(r=>{
    if(!schema)return false;
    if(r.quick_list_parent_id&&!query&&!resultIds)return false;
    if(resultIds&&!resultIds.includes(r.id))return false;
    if(capability&&!r.capabilities.includes(capability))return false;
    if(typeId&&r.type_id!==typeId)return false;
    if(parent&&r.parent_id!==parent&&!r.home.some(p=>p.id===parent))return false;
    if(query&&(ranked?!ranked.has(r.id):![r.title,r.body,r.type_name,JSON.stringify(r.values)].join(" ").toLowerCase().includes(query.toLowerCase())))return false;
    if(status==="active"&&["completed","cancelled"].includes(r.status_meaning??""))return false;
    if(status!=="active"&&status!=="all"&&r.status_meaning!==status)return false;
    if(filterField&&filterValue&&!String(r.values[filterField]??"").toLowerCase().includes(filterValue.toLowerCase()))return false;
    const rt=schema.types.find(t=>t.id===r.type_id)!;const dateFor=(binding:string)=>String(r.values[rt.fields.find(f=>f.binding===binding)?.id??""]??"");
    if(tab==="inbox"&&r.parent_id)return false;
    if(tab==="today"||tab==="week"){const dates=[dateFor("planned_date"),dateFor("due_date")].filter(Boolean);if(!dates.some(d=>d<=(tab==="today"?today:end)))return false;}
    return true;
  }).sort((a,b)=>{
    if(ranked)return (ranked.get(a.id)??0)-(ranked.get(b.id)??0);
    if(sort==="title")return a.title.localeCompare(b.title);
    if(sort==="priority")return Number(bound(b,"priority")||0)-Number(bound(a,"priority")||0)||(bound(a,"due_date")||"9999").localeCompare(bound(b,"due_date")||"9999");
    if(sort==="due"||sort==="planned")return (effectiveDate(a)||"9999").localeCompare(effectiveDate(b)||"9999")||(bound(a,"due_time")||"99:99").localeCompare(bound(b,"due_time")||"99:99")||Number(bound(b,"priority")||0)-Number(bound(a,"priority")||0);
    return 0;
  });
  const visibleIds = JSON.stringify(visible.map(r=>r.task_id??r.id));
  const visibleSearchIds=JSON.stringify(visible.map(r=>r.id));
  useEffect(()=>{
    if(!resultSearch||!resultIds)return;
    let sent=false;
    const observer=new IntersectionObserver(entries=>{
      if(sent||document.visibilityState!=="visible")return;
      for(const entry of entries){
        if(!entry.isIntersecting)continue;
        const rect=entry.intersectionRect;
        const top=document.elementFromPoint(rect.x+rect.width/2,rect.y+rect.height/2);
        if(!top||!entry.target.contains(top))continue;
        sent=true;void post("/search/events",{search_id:resultSearch,kind:"presented"}).catch(()=>{});break;
      }
    },{threshold:.25});
    document.querySelectorAll(".work-page [data-record-id]").forEach(el=>observer.observe(el));
    return()=>observer.disconnect();
  },[resultSearch,resultIds,visibleSearchIds]);
  useEffect(()=>{if(layout!=="browse"&&layout!=="atlas")onVisible?.(JSON.parse(visibleIds));},[visibleIds,onVisible,layout]);
  useEffect(()=>{if((layout==="browse"||layout==="atlas")&&(query||typeId||filterField||resultIds?.length))setLayout("list");},[query,typeId,filterField,resultIds,layout]);
  if(!schema)return <p role="status" className="work-loading">{error||"Loading your structure…"}</p>;
  const refreshAll=async()=>{await load();setBrowseRefresh(r=>r+1);onChanged?.();};
  const edit=async(row:CustomRecord,changes:Record<string,unknown>)=>{
    const rt=schema.types.find(t=>t.id===row.type_id);
    if(changes.status_id&&rt?.statuses.find(s=>s.id===changes.status_id)?.meaning==="completed"){
      const fresh=await api<CustomRecord>("/structure/records/"+row.id);
      if(fresh.contents?.work_open&&!window.confirm("Complete only "+row.title+"? "+fresh.contents.work_open+" unfinished descendants will stay open."))return;
    }
    await run("record.update",{record_id:row.id,expected_revision:row.revision,schema_revision:schema.revision,...changes});await refreshAll();};
  const create=async()=>{if(!title.trim()||!captureType)return;await run<CustomRecord>("record.create",{type_id:captureType.id,title:title.trim(),schema_revision:schema.revision,...(parent?{parent_id:parent}:{})});setTitle("");await refreshAll();};
  const gf=type?.fields.find(f=>f.id===group);
  const groups=group==="status"?(type?.statuses??meanings.filter(m=>visible.some(r=>r.status_meaning===m)).map(m=>({id:m,name:describe(m),meaning:m}))):group==="parent"?items.filter(r=>visible.some(v=>v.parent_id===r.id)).map(r=>({id:r.id,name:r.title,meaning:""})):gf?.kind==="relation"?items.filter(r=>gf.target_types.includes(r.type_id)).map(r=>({id:r.id,name:r.title,meaning:""})):(gf?.options.length?gf.options.map(o=>({...o,meaning:""})):Array.from(new Set(visible.map(r=>String(r.values[group]??"")).filter(Boolean))).map(v=>({id:v,name:v,meaning:""})));
  const groupValue=(r:CustomRecord)=>group==="status"?(type?r.status_id:r.status_meaning):group==="parent"?r.parent_id:String(r.values[group]??"");
  const columns=[...groups,{id:"",name:"Unassigned",meaning:""}];
  const move=async(id:string,column:string)=>{const row=items.find(r=>r.id===id);if(!row||!canEdit||busy)return;const rt=schema.types.find(t=>t.id===row.type_id)!;const statusId=type?column:rt.statuses.find(s=>s.meaning===column)?.id;if(group==="status"&&!statusId)throw new Error("Choose a workflow status for this record.");const changes=group==="status"?{status_id:statusId}:group==="parent"?{parent_id:column||null}:{values:{[group]:column||null}};await edit(row,changes);};
  const saveView=async()=>{if(!viewName.trim())return;await post("/task-views",{id:crypto.randomUUID(),name:viewName.trim(),expected_revision:0,state:{collection_view:true,collection_type:typeId,collection_parent:parent,collection_group:group,collection_field:filterField,collection_value:filterValue,tab:capability?tab:"all",query,status,layout}});setViewName("");await load();};
  const applyView=(v:Record<string,unknown>)=>{
    if(!v.collection_view){onApplyTaskView?.(v);return;}
    setTypeId(String(v.collection_type??""));setParent(String(v.collection_parent??""));setLayout(String(v.layout));setGroup(String(v.collection_group));setFilterField(String(v.collection_field??""));setFilterValue(String(v.collection_value??""));setStatus(String(v.status??"active"));onQuery?.(String(v.query??""));if(capability)onTab?.((["today","week","inbox"].includes(String(v.tab))?v.tab:"all") as "today"|"inbox"|"week"|"all");
  };
  const removeView=async(v:SavedEntry)=>{await post("/task-views/remove",{id:v.id,expected_revision:v.revision});await load();};
  const grip=(event:React.PointerEvent,id:string)=>{if(event.pointerType==="mouse")return;event.preventDefault();const target=event.currentTarget;target.setPointerCapture(event.pointerId);const up=(e:Event)=>{const point=e as PointerEvent;const col=document.elementFromPoint(point.clientX,point.clientY)?.closest<HTMLElement>("[data-record-column]");if(col)void move(id,col.dataset.recordColumn??"").catch(()=>{});target.removeEventListener("pointerup",up);};target.addEventListener("pointerup",up);};
  const setQueryForBrowse=()=>{onQuery?.("");setTypeId("");setFilterField("");setFilterValue("");setResultIds(null);};
  const go=(r:CustomRecord)=>{if(!capability&&openTarget(r,schema??undefined)==="contents"){setSelected(null);setParent(r.id);setLayout("browse");setBrowseSection("all");setQueryForBrowse();}else open(r);};
  const open=(r:CustomRecord)=>{setSelected(r);void semantic.used(r.id);if(resultSearch)void post("/search/events",{search_id:resultSearch,kind:"used",record_id:r.id}).catch(()=>{});};
  const complete=(r:CustomRecord)=>{const rt=schema.types.find(t=>t.id===r.type_id)!;const next=rt.statuses.find(s=>s.meaning===(r.status_meaning==="completed"?"open":"completed"))??(r.status_meaning==="completed"?rt.statuses.find(s=>s.meaning==="backlog"):undefined);if(next)void edit(r,{status_id:next.id}).catch(()=>{});};

  // ---- Row pieces ------------------------------------------------------------------------
  const lead=(r:CustomRecord)=>{
    if(selecting)return r.task_id?<input className="row-select" aria-label={"Select "+r.title} type="checkbox" checked={selectedIds.includes(r.task_id)} onChange={e=>onSelection?.(e.target.checked?[...selectedIds,r.task_id!].slice(0,100):selectedIds.filter(id=>id!==r.task_id))}/>:<span className="row-glyph" aria-hidden="true"/>;
    if(r.capabilities.includes("work")){const done=r.status_meaning==="completed";return <button className={"check"+(done?" done":"")} aria-label={(done?"Reopen ":"Complete ")+r.title} disabled={!canEdit||busy} onClick={()=>complete(r)}><Check aria-hidden="true"/></button>;}
    const Icon=typeIcon(r.type_id);return <span className="row-glyph" aria-hidden="true"><Icon size={17}/></span>;
  };
  const chips=(r:CustomRecord,{home=true}:{home?:boolean}={})=>{
    const rt=typeOf(r);const out:ReactNode[]=[<SourceBadge key="source" source={r.source}/>];
    if(r.is_quick_list)out.push(<span key="quick" className="chip">Quick list · {r.quick_done}/{r.quick_total}</span>);
    const showType=!typeId&&(!capability||r.type_id!=="task");
    if(home&&(r.home.length||showType)){const Icon=typeIcon(r.home.at(-1)?.type_id??r.type_id);
      out.push(<span key="home" className="chip chip-home" title={[r.type_name,...r.home.map(p=>p.title)].join(" in ")}>
        <Icon aria-hidden="true"/><span className="chip-text">{r.home.length?r.home.map(p=>p.title).join(" / "):r.type_name}</span></span>);}
    const due=bound(r,"due_date"),time=bound(r,"due_time"),planned=bound(r,"planned_date");
    const doneish=["completed","cancelled"].includes(r.status_meaning??"");
    if(due){const bucket=doneish?"none":dueBucket(due,today);
      out.push(<span key="due" className={"chip"+(bucket==="overdue"?" chip-due-overdue":bucket==="today"?" chip-due-today":"")} title="Deadline">
        <CalendarClock aria-hidden="true"/>{shortDate(due,today)}{time?" "+clockLabel(time):""}</span>);}
    else if(planned){const label=shortDate(planned,today);out.push(<span key="planned" className={"chip"+(!doneish&&planned<today?" chip-due-overdue":"")} title="Planned day">Planned {/^(Today|Tomorrow|Yesterday)$/.test(label)?label.toLowerCase():label}</span>);}
    const start=bound(r,"start_date"),target=bound(r,"target_date");
    if(start||target)out.push(<span key="range" className="chip" title="Start to target">{start?shortDate(start,today):"No start"} – {target?shortDate(target,today):"no target"}</span>);
    const priority=Number(bound(r,"priority")||0);
    if(priority>0)out.push(<span key="priority" className={"chip chip-priority p"+priority} title="Priority"><Flag aria-hidden="true"/>{priorityLabels[priority]??"Priority "+priority}</span>);
    if(r.status_id&&r.status_meaning!=="open"){const name=rt?.statuses.find(s=>s.id===r.status_id)?.name;if(name)out.push(<span key="status" className={"chip"+(r.status_meaning==="completed"?" chip-done":"")}>{name}</span>);}
    return out;
  };
  const rowActions=(r:CustomRecord)=>r.task_id&&onTask?<div className="row-actions"><button className="btn-icon" aria-label={"Schedule and linked notes for "+r.title} title="Schedule and linked notes" onClick={()=>onTask(r.task_id!)}><CalendarClock size={17}/></button></div>:null;
  const row=(r:CustomRecord)=>{const meta=chips(r);return <article key={r.id} className={"row work-row"+(r.status_meaning==="completed"?" done":"")+(selecting&&r.task_id&&selectedIds.includes(r.task_id)?" selected":"")} data-record-id={r.id}>
    {lead(r)}
    <div className="row-main">
      <button className="row-title" onClick={()=>open(r)}>{r.title}</button>
      {!!meta.length&&<div className="row-meta">{meta}</div>}
    </div>
    {rowActions(r)}
  </article>;};
  const boardCard=(r:CustomRecord)=>{const meta=chips(r,{home:group!=="parent"});return <article key={r.id} className={"board-rec"+(r.status_meaning==="completed"?" done":"")} data-record-id={r.id} draggable={canEdit} onDragStart={e=>e.dataTransfer.setData("application/eri-record",r.id)}>
    <div className="board-rec-head">
      {lead(r)}
      <button className="board-rec-title" onClick={()=>open(r)}>{r.title}</button>
      {canEdit&&<button aria-label={"Move "+r.title} className="btn-icon board-rec-grip" onPointerDown={e=>grip(e,r.id)}><GripVertical size={16}/></button>}
    </div>
    {(!!meta.length||(r.task_id&&onTask))&&<div className="board-rec-foot">
      <div className="chip-row">{meta}</div>
      {rowActions(r)}
    </div>}
  </article>;};

  // ---- Toolbar ---------------------------------------------------------------------------
  const filterCount=[!!typeId,!!parent,status!=="active",archived,!!(filterField&&filterValue)].filter(Boolean).length;
  const resetFilters=()=>{setTypeId("");setParent("");setStatus("active");setArchived(false);setFilterField("");setFilterValue("");setGroup("status");};
  const homeGroups=Array.from(new Set(items.map(r=>r.type_name))).map(name=>({name,rows:items.filter(r=>r.type_name===name)}));
  const field=type?.fields.find(f=>f.id===filterField);
  const activeChips=[
    typeId&&{key:"type",label:type?.plural??"Collection",clear:()=>{setTypeId("");setGroup("status");}},
    parent&&{key:"home",label:"In "+(items.find(r=>r.id===parent)?.title??"selected home"),clear:()=>setParent("")},
    status!=="active"&&{key:"status",label:status==="all"?"All statuses":describe(status).replace(/^./,c=>c.toUpperCase()),clear:()=>setStatus("active")},
    filterField&&filterValue&&{key:"field",label:(field?.name??"Field")+": "+((field?.kind==="relation"?items.find(r=>r.id===filterValue)?.title:field?.options.find(o=>o.id===filterValue)?.name)??filterValue),clear:()=>{setFilterField("");setFilterValue("");}},
    archived&&{key:"archived",label:"Archived",clear:()=>setArchived(false)},
  ].filter(Boolean) as {key:string;label:string;clear:()=>void}[];
  const layoutIcons={browse:Compass,atlas:Orbit,tree:Layers,list:List,board:Columns3,timeline:ChartNoAxesGantt} as const;
  const atlas=layout==="atlas"&&!capability&&!narrow;
  const toolbar=<div className={"work-toolbar"+(tabs?" has-tabs":"")}>
    {tabs}
    <div className="work-toolbar-controls">
      {!atlas&&<label className="toolbar-search"><Search size={16} aria-hidden="true"/><input type="search" aria-label={searchLabel??(capability?"Search this list":"Search organization")} placeholder="Search…" value={query} onChange={e=>onQuery?.(e.target.value)}/></label>}
      {!atlas&&<Popover label={"Filter"+(filterCount?", "+filterCount+" active":"")} button={<><ListFilter size={16} aria-hidden="true"/><span className="toolbar-label">Filter</span>{!!filterCount&&<span className="toolbar-count">{filterCount}</span>}</>} panelClassName="filter-popover">
        {()=><div className="filter-fields">
          <label className="field">Collection<select value={typeId} onChange={e=>{setTypeId(e.target.value);setGroup("status");}}><option value="">{capability?"All actionable work":"All records"}</option>{types.map(t=><option key={t.id} value={t.id}>{t.plural}</option>)}</select></label>
          <label className="field">Main home<select value={parent} onChange={e=>setParent(e.target.value)}><option value="">All homes</option>{homeGroups.map(g=><optgroup key={g.name} label={g.name}>{g.rows.map(r=><option key={r.id} value={r.id}>{r.title}</option>)}</optgroup>)}</select></label>
          <label className="field">Status<select value={status} onChange={e=>setStatus(e.target.value)}><option value="active">Active</option><option value="all">All statuses</option>{meanings.map(m=><option key={m} value={m}>{describe(m).replace(/^./,c=>c.toUpperCase())}</option>)}</select></label>
          {layout==="list"&&<label className="field">Sort<select value={sort} onChange={e=>setSort(e.target.value as Sort)}>{sortOptions.map(([value,label])=><option key={value} value={value}>{label}</option>)}</select></label>}
          {layout==="board"&&<label className="field">Group<select value={group} onChange={e=>setGroup(e.target.value)}><option value="status">Status</option><option value="parent">Main home</option>{type?.fields.filter(f=>(f.kind==="select"||(f.kind==="relation"&&!f.multiple))&&!f.archived).map(f=><option key={f.id} value={f.id}>{f.name}</option>)}</select></label>}
          {type&&<><label className="field">Field filter<select value={filterField} onChange={e=>{setFilterField(e.target.value);setFilterValue("");}}><option value="">Any field</option>{type.fields.filter(f=>!f.archived).map(f=><option value={f.id} key={f.id}>{f.name}</option>)}</select></label>
            {filterField&&<label className="field">Value{["select","multiselect","relation"].includes(field?.kind??"")?<select aria-label="Field filter value" value={filterValue} onChange={e=>setFilterValue(e.target.value)}><option value="">Any value</option>{(field?.kind==="relation"?items.filter(r=>field?.target_types.includes(r.type_id)).map(r=>({id:r.id,name:r.title})):field?.options??[]).map(o=><option key={o.id} value={o.id}>{o.name}</option>)}</select>:<input aria-label="Field filter value" value={filterValue} onChange={e=>setFilterValue(e.target.value)}/>}</label>}</>}
          <label className="check-label filter-archived"><input type="checkbox" checked={archived} onChange={e=>setArchived(e.target.checked)}/>Archived</label>
          {!!filterCount&&<button type="button" className="btn btn-ghost btn-sm filter-reset" onClick={resetFilters}>Reset filters</button>}
        </div>}
      </Popover>}
      <Popover label="Saved views" button={<><Bookmark size={16} aria-hidden="true"/><span className="toolbar-label">Saved views</span></>} panelClassName="views-popover">
        {close=><>
          {saved.length?<ul className="view-list" aria-label="Saved collection view">{saved.map(v=><li key={v.id}>
            <button type="button" className="menu-item" onClick={()=>{applyView(v.state);close();}}><Bookmark size={15} aria-hidden="true"/><span>{v.name}</span></button>
            <button type="button" className="btn-icon" aria-label={"Delete saved view "+v.name} onClick={()=>void removeView(v).catch(e=>setError(e.message))}><Trash2 size={15}/></button>
          </li>)}</ul>:<p className="popover-note">Save the filters and layout you use often.</p>}
          <div className="menu-divider"/>
          <form className="view-save" onSubmit={e=>{e.preventDefault();void saveView().catch(err=>setError(err.message));}}>
            <input aria-label="View name" value={viewName} maxLength={80} onChange={e=>setViewName(e.target.value)} placeholder="Name this view"/>
            <button className="btn btn-soft btn-sm" disabled={!viewName.trim()}>Save view</button>
          </form>
          {viewLink&&<button type="button" className="menu-item" onClick={()=>void navigator.clipboard.writeText(viewLink()).then(()=>setViewNotice("Link copied")).catch(()=>setViewNotice("Could not copy. The address bar has this view's link."))}><Link size={15} aria-hidden="true"/>Copy view link</button>}
          {viewNotice&&<p className="popover-note" role="status">{viewNotice}</p>}
        </>}
      </Popover>
      <div className="segmented work-layout" role="group" aria-label="Layout">
        {((capability?["list","board","timeline"]:narrow?["browse","tree","list","board","timeline"]:["browse","atlas","tree","list","board","timeline"]) as (keyof typeof layoutIcons)[]).map(value=>{const Icon=layoutIcons[value];const label=value==="tree"?"Structure":value.charAt(0).toUpperCase()+value.slice(1);return <button key={value} type="button" aria-label={value==="tree"?"Tree":label} title={label} aria-pressed={layout===value} onClick={()=>setLayout(value)}><Icon size={16} aria-hidden="true"/><span className="toolbar-label">{label}</span></button>;})}
      </div>
      {canDesign&&<button type="button" className="btn btn-ghost toolbar-button" aria-label="Structure" title="Shape your workspace structure" onClick={()=>{setProposal(undefined);setDesignDraft(null);setDesign(true);}}><Settings2 size={16} aria-hidden="true"/><span className="toolbar-label">Types & fields</span></button>}
    </div>
  </div>;

  // ---- Body ------------------------------------------------------------------------------
  const timelineEnd=shiftDate(timelineStart,TIMELINE_DAYS-1);
  const offset=(value:string)=>(Date.parse(value+"T12:00:00Z")-Date.parse(timelineStart+"T12:00:00Z"))/86400000;
  const pct=(days:number)=>days/TIMELINE_DAYS*100+"%";
  const todayOffset=offset(today);
  const timeline=<section className="work-timeline" aria-label="Timeline">
    <div className="work-timeline-head">
      <div className="work-timeline-nav">
        <button type="button" className="btn-icon" aria-label="Previous 30 days" onClick={()=>setTimelineStart(shiftDate(timelineStart,-TIMELINE_DAYS))}><ChevronLeft size={17}/></button>
        <button type="button" className="btn btn-sm" onClick={()=>setTimelineStart(today)}>Today</button>
        <button type="button" className="btn-icon" aria-label="Next 30 days" onClick={()=>setTimelineStart(shiftDate(timelineStart,TIMELINE_DAYS))}><ChevronRight size={17}/></button>
        <span className="work-timeline-range tabular">{monthDay(timelineStart)} – {monthDay(timelineEnd)}</span>
      </div>
      <label className="work-timeline-start">Timeline start<input type="date" value={timelineStart} onChange={e=>e.target.value&&setTimelineStart(e.target.value)}/></label>
    </div>
    <p className="work-timeline-note">Solid bars show this record’s own dates. The thin line shows its children’s date span. Task dates are markers, not reserved time; neither shifts the other.</p>
    <div className="work-timeline-grid">
      <div className="work-timeline-axis" aria-hidden="true"><span/>
        <div className="work-timeline-scale">{[0,7,14,21,28].map(d=><span key={d} style={{left:pct(d)}}>{monthDay(shiftDate(timelineStart,d))}</span>)}</div>
      </div>
      {nestedRows(visible,timelineClosed).map(({row:r,depth,hasChildren})=>{const span=childSpan(r,items,bound);const start=bound(r,"start_date")||bound(r,"planned_date")||bound(r,"due_date");const finish=bound(r,"target_date")||start;const a=offset(start),b=offset(finish);const Icon=typeIcon(r.type_id);
        return <div key={r.id} className="work-timeline-row">
          <div className="work-timeline-name" data-record-id={r.id} style={{paddingLeft:12+Math.min(depth,8)*16}}>
            {hasChildren&&<button className="btn-icon" aria-label={(timelineClosed.has(r.id)?"Expand ":"Collapse ")+r.title} aria-expanded={!timelineClosed.has(r.id)} onClick={()=>setTimelineClosed(v=>{const n=new Set(v);if(n.has(r.id))n.delete(r.id);else n.add(r.id);return n;})}>{timelineClosed.has(r.id)?"▸":"▾"}</button>}
            <span className="row-glyph" aria-hidden="true"><Icon size={16}/></span>
            <div className="row-main">
              <button className="row-title" onClick={()=>open(r)}>{r.title}</button>
              {(r.home.length>0||start)&&<div className="chip-row">
                {r.home.length>0&&<span className="chip chip-home">{r.home.map(p=>p.title).join(" / ")}</span>}
                {start&&<span className="chip tabular">{shortDate(start,today)}{finish!==start?" – "+shortDate(finish,today):""}</span>}
              </div>}
            </div>
          </div>
          <div className="timeline-track">
            {span&&offset(span[1])>=0&&offset(span[0])<TIMELINE_DAYS&&<span className="timeline-child-span" title={"Children: "+span.join(" – ")} style={{left:pct(Math.max(0,offset(span[0]))),width:pct(Math.max(0,Math.min(TIMELINE_DAYS,offset(span[1])+1)-Math.max(0,offset(span[0]))))}}/>}
            {todayOffset>=0&&todayOffset<TIMELINE_DAYS&&<span className="work-timeline-today" style={{left:pct(todayOffset+.5)}}/>}
            {start?(b>=0&&a<TIMELINE_DAYS?<span className={"timeline-bar"+(r.status_meaning==="completed"?" done":"")} title={start+(finish!==start?" – "+finish:"")} style={{left:pct(Math.max(0,a)),width:"max(6px, "+pct(Math.min(TIMELINE_DAYS,b+1)-Math.max(0,a))+")"}}/>:<span className="work-timeline-out">{b<0?"Before this range":"After this range"}</span>):<span className="work-timeline-out">Unscheduled</span>}
          </div>
        </div>;})}
    </div>
  </section>;
  const board=<div className="work-board" aria-label="Board">{columns.filter(c=>c.id||visible.some(r=>!groupValue(r))).map(c=>{const rows=visible.filter(r=>groupValue(r)===c.id||(!c.id&&!groupValue(r)));
    return <section key={c.id} className="work-column" data-record-column={c.id} onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();void move(e.dataTransfer.getData("application/eri-record"),c.id).catch(()=>{});}}>
      <h3 className="work-column-title"><span>{c.name.replace(/^./,ch=>ch.toUpperCase())}</span><span className="work-count">{rows.length}</span></h3>
      {rows.map(boardCard)}
      {!rows.length&&<p className="work-column-empty">Nothing here</p>}
    </section>;})}</div>;
  const grouped=(sort==="due"||sort==="planned")&&!ranked&&visible.some(r=>effectiveDate(r));
  const list=grouped?dueBuckets.map(b=>{const rows=visible.filter(r=>dueBucket(effectiveDate(r),today)===b.key);return rows.length?<section key={b.key} className={"work-group work-group-"+b.key}>
      <h3 className="work-group-title"><span>{b.label}</span><span className="work-count">{rows.length}</span></h3>
      <div className="row-list">{rows.map(row)}</div>
    </section>:null;}):<div className="row-list">{visible.map(row)}</div>;
  const visibleTasks=visible.filter(r=>r.task_id).slice(0,100);
  const capture=captureType&&canEdit&&!archived?<form className="work-capture" onSubmit={e=>{e.preventDefault();void create().catch(()=>{});}}>
    <Plus size={17} aria-hidden="true"/>
    <input ref={captureInput} aria-label={"New "+captureType.name} value={title} onChange={e=>setTitle(e.target.value)} placeholder={"Add "+captureType.name.toLowerCase()+"…"} maxLength={500}/>
    <button className="btn btn-primary btn-sm" disabled={busy||!title.trim()}>Add</button>
  </form>:null;
  const empty=!visible.length&&<div className="empty work-empty">
    {query?<><span>No records match “{query}”.</span><button type="button" className="btn btn-soft" onClick={()=>onQuery?.("")}>Clear search</button></>
      :filterCount||resultIds?<><span>Nothing matches these filters.</span><button type="button" className="btn btn-soft" onClick={()=>{resetFilters();setResultIds(null);setResultSearch(null);}}>Clear filters</button></>
      :capture?<><span>{capability?"Nothing here yet. Add a task to get started.":"Nothing here yet. Add a record, or shape your workspace in Structure."}</span><button type="button" className="btn btn-soft" onClick={()=>captureInput.current?.focus()}>Add {captureType!.name.toLowerCase()}</button></>
      :<span>{capability?"No tasks in this view.":"Choose a collection to see its records."}</span>}
  </div>;
  return <section className="structure-workspace work-page">
    {toolbar}
    {!!activeChips.length&&<div className="work-filter-chips" aria-label="Active filters">{activeChips.map(c=><button key={c.key} type="button" className="chip work-filter-chip" aria-label={"Remove filter: "+c.label} onClick={c.clear}>{c.label}<X aria-hidden="true"/></button>)}</div>}
    {error&&<p role="alert" className="work-error">{error}</p>}
    {resultIds&&<div className="work-notice">{resultIds.length} search results <button type="button" className="text-button" onClick={()=>{setResultIds(null);setResultSearch(null);}}>Show all records</button></div>}
    {query&&semantic.pending&&<p role="status" className="work-notice">Searching…</p>}
    {query&&(semantic.error||semantic.result?.incomplete)&&<p className="work-notice">Showing available matches. Semantic search is still catching up.</p>}
    {layout!=="browse"&&!atlas&&!(layout==="atlas"&&narrow)&&<div className="work-listbar">
      {selecting?<div className="work-bulk" role="group" aria-label="Selection">
        <label className="check-label"><input type="checkbox" aria-label="Select visible tasks" checked={visibleTasks.length>0&&visibleTasks.every(r=>selectedIds.includes(r.task_id!))} onChange={e=>onSelection?.(e.target.checked?visibleTasks.map(r=>r.task_id!):[])}/>Select visible</label>
        <span className="work-bulk-count tabular">{selectedIds.length} selected</span>
        <button type="button" className="btn btn-primary btn-sm" disabled={!selectedIds.length} onClick={onBulk}>Edit selected tasks</button>
        <button type="button" className="btn btn-ghost btn-sm" onClick={()=>onSelecting?.(false)}>Done selecting</button>
      </div>:capture}
      {!selecting&&onSelecting&&canEdit&&layout!=="timeline"&&<button type="button" className="btn btn-ghost btn-sm work-select" onClick={()=>onSelecting(true)}>Select tasks</button>}
    </div>
    }
    <div className="work-body" id={tabs?"task-workspace":undefined} role={tabs?"tabpanel":undefined} aria-labelledby={tabs?"task-tab-"+tab:undefined}>
      {atlas?<Suspense fallback={<p role="status" className="work-loading">Loading the Atlas…</p>}><AtlasView schema={schema} canEdit={canEdit} canDesign={canDesign} focus={parent} target={atlasTarget} refresh={refresh+browseRefresh} today={today}
        onFocus={setParent} onVisible={onVisible} onOpen={id=>void api<CustomRecord>("/structure/records/"+id).then(open).catch(e=>setError(e.message))}
        onBrowse={id=>{setParent(id);setLayout("browse");setBrowseSection("all");}} onChanged={async()=>{await load();onChanged?.();}}
        onEditTypes={(draft,type)=>{setProposal(undefined);setDesignDraft({draft,type});setDesign(true);}}/></Suspense>
       :(layout==="browse"||layout==="atlas")&&!capability?<OrganizationBrowse schema={schema} parent={parent} section={browseSection} onSection={setBrowseSection} query={query} archived={archived} status={status} refresh={refresh+browseRefresh} canEdit={canEdit}
        onBrowse={id=>{setParent(id);setBrowseSection(id?"all":"groups");setQueryForBrowse();}} onOpen={open} onVisible={onVisible}
        onCreate={async(type,title)=>{await run("record.create",{type_id:type,title,schema_revision:schema.revision,...(parent?{parent_id:parent}:{})});await refreshAll();}}/>
       :!visible.length?empty:layout==="tree"?<OrganizationTree items={items} visible={visible} schema={schema} disabled={!canEdit||busy} onOpen={open} onBrowse={r=>{setParent(r.id);setLayout("browse");setBrowseSection("all");}} onMove={async(r,c)=>{if(c.parent_id!==r.parent_id){setMoving({row:r,parentId:c.parent_id as string|null});}else await edit(r,c);}}/>:layout==="board"?board:layout==="timeline"?timeline:list}
    </div>
    {selected?.is_quick_list&&selected.task_id?<QuickListDetail refresh={refresh} key={selected.id} id={selected.task_id} today={today} canEdit={canEdit} onClose={()=>setSelected(null)} onChanged={refreshAll}/>:selected&&<RecordCard key={selected.id} schema={schema} initial={selected} onOpen={go} choices={items} canEdit={canEdit} onClose={()=>setSelected(null)} onChanged={refreshAll}/>}
    {moving&&<ContentsDialog row={moving.row} schema={schema} operation="move" parentId={moving.parentId} onClose={()=>setMoving(null)} onDone={refreshAll}/>}
    {design&&<StructureEditor onDirtyChange={setDesignDirty} initialProposal={proposal} initialDraft={designDraft?.draft??undefined} initialType={designDraft?.type} schema={schema} onClose={()=>{setDesign(false);setDesignDraft(null);}} onApplied={async()=>{setDesign(false);setDesignDraft(null);await refreshAll();}}/>}
  </section>;
}
