import {useEffect,useRef,useState} from "react";
import {post} from "./api";
import type {CustomRecord} from "./structure-types";
export type SearchMatch={record:CustomRecord;match:string;evidence:string;score:number};
export type SearchResult={enabled:boolean;search_id:string|null;mode:string;resolved:string[];resolutions:{key:string;label:string}[];structured:{items:SearchMatch[];next_offset:number|null};possible:{items:SearchMatch[];next_offset:number|null};index:{pending:number;status:string}};
type SearchRecord={search:string;target?:string};
export function useSemanticSearch(query:string,capability:string,archived:boolean,refresh:unknown){
 const [result,setResult]=useState<{query:string;ids:string[];meta:Map<string,SearchRecord>;incomplete:boolean}|null>(null);
 const [pending,setPending]=useState(false),[error,setError]=useState("");const generation=useRef(0);
 useEffect(()=>{const version=++generation.current;setResult(null);setError("");setPending(false);if(!query.trim())return;
  const timer=setTimeout(()=>{setPending(true);void(async()=>{
   const ids:string[]=[];const meta=new Map<string,SearchRecord>();let offset:number|null=0;let incomplete=false;
   while(offset!==null&&version===generation.current){
    const page:SearchResult=await post("/search/records",{query,capability:capability||null,archived,limit:100,offset});
    if(!page.enabled)return;
    incomplete ||= page.index.pending>0||page.mode!=="hybrid";
    for(const match of [...page.structured.items,...page.possible.items]){const id=match.record.id;if(!ids.includes(id))ids.push(id);if(page.search_id)meta.set(id,{search:page.search_id,target:page.resolved.length===1?page.resolved[0]:undefined});}
    offset=page.structured.next_offset??page.possible.next_offset;
   }
   if(version===generation.current)setResult({query,ids,meta,incomplete});
  })().catch(e=>{if(version===generation.current)setError(e.message);}).finally(()=>{if(version===generation.current)setPending(false);});},350);
  return()=>{clearTimeout(timer);generation.current++;};
 },[query,capability,archived,refresh]);
 async function used(id:string){const entry=result?.meta.get(id);if(!entry?.target||!query.trim()||query.length>200)return;
  try{await post("/search/selection",{search_id:entry.search,target_key:entry.target,phrase:query,record_ids:[id]});await post("/search/events",{search_id:entry.search,kind:"used",record_id:id});}catch{/* Learning cannot prevent opening a record. */}
 }
 return {result:result?.query===query?result:null,pending,error,used};
}
