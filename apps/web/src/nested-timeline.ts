import type {CustomRecord} from "./structure-types";
/** Each visible record appears once, under its nearest visible ancestor. */
export function nestedRows(rows:CustomRecord[],closed:Set<string>){
 const ids=new Set(rows.map(r=>r.id)),seen=new Set<string>();
 const parent=(r:CustomRecord)=>ids.has(r.parent_id??"")?r.parent_id:[...r.home].reverse().find(h=>ids.has(h.id))?.id??null;
 const result:{row:CustomRecord;depth:number;hasChildren:boolean}[]=[];
 const visit=(id:string|null,depth:number)=>{for(const row of rows.filter(r=>parent(r)===id)){if(seen.has(row.id))continue;seen.add(row.id);
  const hasChildren=rows.some(r=>parent(r)===row.id);result.push({row,depth,hasChildren});if(!closed.has(row.id))visit(row.id,depth+1);
 }};
 visit(null,0);return result;
}
export function childSpan(row:CustomRecord,rows:CustomRecord[],bound:(r:CustomRecord,k:string)=>string):[string,string]|null{
 const dates=rows.filter(r=>!r.archived&&r.home.some(h=>h.id===row.id)).flatMap(r=>["start_date","target_date","planned_date","due_date"].map(k=>bound(r,k))).filter(Boolean).sort();
 return dates.length?[dates[0],dates[dates.length-1]]:null;
}
