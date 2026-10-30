import {describe,expect,it} from "vitest";
import {inHome,inTab,literalMatch,matchesQuery,rankOrder} from "./record-filters";
import {rankRecords,type SearchResult} from "./semantic-search";

const PARITY="Parity: ignore time-varying parts of a session's resume when comparing";
const row=(id:string,title:string,extra:Record<string,unknown>={})=>({id,title,body:"",type_name:"Task",values:{},parent_id:null as string|null,home:[] as {id:string;title:string;type_id:string}[],project_id:null as string|null,...extra});

describe("Tasks search (ce2e0e8d)",()=>{
 it("matches any case-insensitive substring of the title or notes",()=>{
  const parity=row("p",PARITY);
  for(const q of ["ignore","IGNORE","time-var","gnore","resume when","  ignore "])expect(literalMatch(parity,q)).toBe(true);
  expect(literalMatch(row("n","Release",{body:"Nothing ignorable here"}),"ignorable")).toBe(true);
  expect(literalMatch(parity,"unrelated")).toBe(false);
 });
 it("keeps a literal match the server ranking left out",()=>{
  const parity=row("p",PARITY),other=row("o","Ship the release");
  const ranked=new Map([["o",0]]);
  expect(matchesQuery(parity,"ignore",ranked)).toBe(true);
  expect(matchesQuery(other,"ignore",ranked)).toBe(true);
  expect(matchesQuery(row("x","Plan the week"),"ignore",ranked)).toBe(false);
  expect(matchesQuery(row("x","Plan the week"),"",ranked)).toBe(true);
 });
 it("orders server-ranked rows first, then literal-only rows",()=>{
  const ranked=new Map([["b",0],["c",1]]);
  expect([row("a","a"),row("c","c"),row("b","b")].sort(rankOrder(ranked)).map(r=>r.id)).toEqual(["b","c","a"]);
 });
});

describe("Main home filter (b8d2e28e)",()=>{
 const project={id:"roadmap",title:"Eridani roadmap",type_id:"project"};
 it("includes direct children, descendants and tasks of the legacy project",()=>{
  expect(inHome(row("d","Direct",{parent_id:"roadmap",home:[project]}),"roadmap")).toBe(true);
  expect(inHome(row("c","Child step",{parent_id:"parent",home:[project,{id:"parent",title:"Parent",type_id:"task"}]}),"roadmap")).toBe(true);
  expect(inHome(row("l","Drifted",{project_id:"roadmap"}),"roadmap")).toBe(true);
  expect(inHome(row("e","Elsewhere"),"roadmap")).toBe(false);
  expect(inHome(row("e","Elsewhere"),"")).toBe(true);
 });
 it("explains why a home looked empty on Today: undated tasks are outside the tab",()=>{
  const task=row("d","Direct",{parent_id:"roadmap"});
  expect(inTab(task,"today",["",""],"2026-10-28","2026-11-03")).toBe(false);
  expect(inTab(task,"inbox",[],"2026-10-28","2026-11-03")).toBe(false);
  expect(inTab(task,"all",[],"2026-10-28","2026-11-03")).toBe(true);
  expect(inTab(task,"week",["2026-11-01"],"2026-10-28","2026-11-03")).toBe(true);
 });
});

describe("two-pass ranking (b14920b4)",()=>{
 const page=(ids:string[],extra:Partial<SearchResult>={}):SearchResult=>({enabled:true,search_id:null,mode:"lexical",resolved:[],resolutions:[],structured:{items:[],next_offset:null},possible:{items:ids.map(id=>({record:{id} as never,match:"possible",evidence:"",score:1})),next_offset:null},index:{pending:0,status:"ready"},...extra});
 it("asks for the lexical pass without the embedding call, then the hybrid pass",async()=>{
  const bodies:Record<string,unknown>[]=[];
  const send=async(_path:string,body:unknown)=>{bodies.push(body as Record<string,unknown>);return (body as {lexical_only:boolean}).lexical_only?page(["p"]):page(["p","q"],{mode:"hybrid",search_id:"s1"});};
  const fast=await rankRecords("parity","work",false,true,()=>true,send);
  expect(fast?.ids).toEqual(["p"]);expect(fast?.incomplete).toBe(false);
  const full=await rankRecords("parity","work",false,false,()=>true,send);
  expect(full?.ids).toEqual(["p","q"]);expect(full?.meta.get("q")?.search).toBe("s1");
  expect(bodies.map(b=>b.lexical_only)).toEqual([true,false]);
 });
 it("returns nothing when semantic search is off, and stops when superseded",async()=>{
  expect(await rankRecords("x","",false,true,()=>true,async()=>page([],{enabled:false}))).toBeNull();
  let calls=0;
  const ranking=await rankRecords("x","",false,false,()=>false,async()=>{calls++;return page([]);});
  expect(calls).toBe(0);expect(ranking?.ids).toEqual([]);
 });
});
