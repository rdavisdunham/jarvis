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
export type TimelinePlacement =
 | {kind:"bar";from:number;to:number;overdue:boolean;clipped:boolean}
 | {kind:"before"|"after"|"unscheduled";overdue:boolean};
/** Where a record's bar sits in the `days`-long window that opens on `rangeStart` (`from`/`to` are day offsets,
 * clamped to the window, `to` exclusive). An unfinished record whose dates all fall before today is overdue: its bar
 * runs on to today, so it shows (clipped at the window's start) instead of "Before this range". */
export function timelinePlacement(start:string,finish:string,{rangeStart,today,days,finished}:{rangeStart:string;today:string;days:number;finished:boolean}):TimelinePlacement{
 const overdue=!!start&&!finished&&(finish||start)<today;
 if(!start)return {kind:"unscheduled",overdue};
 const offset=(value:string)=>Math.round((Date.parse(value+"T12:00:00Z")-Date.parse(rangeStart+"T12:00:00Z"))/86400000);
 const a=offset(start),b=offset(overdue?today:finish||start);
 if(b<0)return {kind:"before",overdue};
 if(a>=days)return {kind:"after",overdue};
 return {kind:"bar",from:Math.max(0,a),to:Math.min(days,b+1),overdue,clipped:a<0};
}
export function childSpan(row:CustomRecord,rows:CustomRecord[],bound:(r:CustomRecord,k:string)=>string):[string,string]|null{
 const dates=rows.filter(r=>!r.archived&&r.home.some(h=>h.id===row.id)).flatMap(r=>["start_date","target_date","planned_date","due_date"].map(k=>bound(r,k))).filter(Boolean).sort();
 return dates.length?[dates[0],dates[dates.length-1]]:null;
}
