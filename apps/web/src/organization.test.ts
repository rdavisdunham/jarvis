import {describe,it,expect} from "vitest";
import {updateType,placeType,canAttach,attachField} from "./field-library";
import {nestedRows,childSpan} from "./nested-timeline";
import {emptyField,type Schema,type CustomRecord} from "./structure-types";
const field={...emptyField(),id:"shared",name:"Client code",description:"Client identifier"};
const type=(id:string)=>({id,name:id,plural:id,description:id,capabilities:[],parent_types:["a","b"],fields:[{...field,id:"attachment-"+id,library_id:field.id}],statuses:[],archived:false});
const schema=():Schema=>({revision:1,field_library:[field],types:[type("a"),type("b")],relationships:[],understandings:[]});
describe("field library and visual layout",()=>{
 it("propagates meaning across attachments but keeps local visibility",()=>{const s=schema();const n=updateType(s,"a",{fields:[{...s.types[0].fields[0],name:"Customer code",visible:false}]});expect(n.types[1].fields[0].name).toBe("Customer code");expect(n.types[1].fields[0].visible).toBe(true);expect(n.types[0].fields[0].visible).toBe(false);expect(s.types[0].fields[0].name).toBe("Client code");});
 it("allows reuse without duplicating library identity and rejects incompatible bindings",()=>{expect(attachField(field).library_id).toBe("shared");expect(canAttach(field,schema().types[0])).toBe(false);expect(canAttach({...field,binding:"due_date"},{...type("c"),fields:[]})).toBe(false);});
 it("keeps presentation changes separate from allowed homes and rejects cycles",()=>{const s=schema(),n=placeType(s,"a","b");expect(n.types).toEqual(s.types);expect(()=>placeType(n,"b","a")).toThrow("loop");expect(()=>placeType(s,"a","a")).toThrow("itself");expect(()=>placeType(s,"a","unknown")).toThrow("allowed");});
});
const row=(id:string,parent_id:string|null,home:string[],values:Record<string,string>={}):CustomRecord=>({id,parent_id,home:home.map(id=>({id,title:id,type_id:"task"})),values,archived:false} as CustomRecord);
describe("nested timelines",()=>{
 it("renders each record once under the nearest visible ancestor and supports collapse",()=>{const rows=[row("leaf","hidden",["root","hidden"]),row("root",null,[])];expect(nestedRows(rows,new Set()).map(r=>[r.row.id,r.depth])).toEqual([["root",0],["leaf",1]]);expect(nestedRows(rows,new Set(["root"])).map(r=>r.row.id)).toEqual(["root"]);});
 it("shows child dates independently of the parent and excludes archived children",()=>{const root=row("root",null,[],{start_date:"2026-01-01"}),child=row("child","root",["root"],{due_date:"2026-03-04",planned_date:"2026-03-01"});expect(childSpan(root,[root,child],(r,k)=>String(r.values[k]??""))).toEqual(["2026-03-01","2026-03-04"]);expect(childSpan(root,[root,{...child,archived:true}],(r,k)=>String(r.values[k]??""))).toBe(null);});
});
