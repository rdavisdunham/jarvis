import {useEffect,useRef,useState} from "react";
import {post} from "./api";
import type {CustomRecord} from "./structure-types";
export type SearchMatch={record:CustomRecord;match:string;evidence:string;score:number};
export type SearchTiming={snapshot_ms?:number;embedding_ms?:number;refresh_ms?:number;rank_ms?:number;total_ms?:number};
export type SearchResult={enabled:boolean;search_id:string|null;mode:string;resolved:string[];resolutions:{key:string;label:string}[];structured:{items:SearchMatch[];next_offset:number|null};possible:{items:SearchMatch[];next_offset:number|null};index:{pending:number;status:string};timing?:SearchTiming};
type SearchRecord={search:string;target?:string};
type Ranking={query:string;ids:string[];meta:Map<string,SearchRecord>;incomplete:boolean;lexical:boolean};
type Post=(path:string,body:unknown)=>Promise<SearchResult>;

/**
 * Two passes. The lexical pass (no embedding call) answers a title lookup in one quick request; the
 * hybrid pass then re-ranks with meaning and learned aliases. Callers keep their own literal
 * substring match on top, so nothing a pass returns can hide a row that contains the query.
 * Returns null when semantic search is off.
 */
export async function rankRecords(query:string,capability:string,archived:boolean,lexical:boolean,current:()=>boolean,send:Post=post as Post):Promise<Ranking|null>{
  const ids:string[]=[];const meta=new Map<string,SearchRecord>();let offset:number|null=0;let incomplete=false;
  while(offset!==null&&current()){
    const page:SearchResult=await send("/search/records",{query,capability:capability||null,archived,limit:100,offset,lexical_only:lexical});
    if(!page.enabled)return null;
    incomplete ||= page.index.pending>0||(!lexical&&page.mode!=="hybrid");
    for(const match of [...page.structured.items,...page.possible.items]){const id=match.record.id;if(!ids.includes(id))ids.push(id);if(page.search_id)meta.set(id,{search:page.search_id,target:page.resolved.length===1?page.resolved[0]:undefined});}
    offset=page.structured.next_offset??page.possible.next_offset;
  }
  return {query,ids,meta,incomplete,lexical};
}

export function useSemanticSearch(query:string,capability:string,archived:boolean,refresh:unknown){
 const [result,setResult]=useState<Ranking|null>(null);
 const [pending,setPending]=useState(false),[error,setError]=useState("");const generation=useRef(0);
 useEffect(()=>{const version=++generation.current;setResult(null);setError("");setPending(false);if(!query.trim())return;
  const current=()=>version===generation.current;
  const timer=setTimeout(()=>{setPending(true);void(async()=>{
   const fast=await rankRecords(query,capability,archived,true,current);
   if(!fast||!current())return;
   setResult(fast);
   const full=await rankRecords(query,capability,archived,false,current);
   if(full&&current())setResult(full);
  })().catch(e=>{if(current())setError(e.message);}).finally(()=>{if(current())setPending(false);});},250);
  return()=>{clearTimeout(timer);generation.current++;};
 },[query,capability,archived,refresh]);
 async function used(id:string){const entry=result?.meta.get(id);if(!entry?.target||!query.trim()||query.length>200)return;
  try{await post("/search/selection",{search_id:entry.search,target_key:entry.target,phrase:query,record_ids:[id]});await post("/search/events",{search_id:entry.search,kind:"used",record_id:id});}catch{/* Learning cannot prevent opening a record. */}
 }
 return {result:result?.query===query?result:null,pending,error,used};
}
