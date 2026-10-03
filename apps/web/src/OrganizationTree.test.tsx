import {describe,it,expect} from "vitest";
import {renderToStaticMarkup} from "react-dom/server";
import {validHomes,ordered} from "./OrganizationTree";
import {SourceBadge,SourceDetails} from "./SourceDetails";
import type {CustomRecord,Schema} from "./structure-types";
const row=(id:string,parent_id:string|null=null,home:CustomRecord["home"]=[],sort_order=0)=>({id,parent_id,home,sort_order,type_id:"project",archived:false,title:id} as CustomRecord);
const schema={types:[{id:"project",parent_types:["project"]}]} as Schema;
describe("Batch B organization/source presentation",()=>{
 it("excludes self, descendants and archived homes",()=>{
  const root=row("root"),child=row("child","root",[{id:"root",title:"Root",type_id:"project"}]),other=row("other"),archived={...row("old"),archived:true};
  expect(validHomes(root,[root,child,other,archived],schema).map(r=>r.id)).toEqual(["other"]);
 });
 it("sorts sibling positions without mutating fetched records",()=>{
  const rows=[row("b",null,[],20),row("a",null,[],10)];
  expect(ordered(rows).map(r=>r.id)).toEqual(["a","b"]);expect(rows[0].id).toBe("b");
 });
 it("shows provider text and pending state without relying on color",()=>{
  const html=renderToStaticMarkup(<SourceBadge source={{provider:"linear",label:"Linear",context:"Workspace · Team",sync_state:"pending"}}/>);
  expect(html).toContain("Linear");expect(html).toContain("pending");expect(html).toContain("Workspace · Team");
 });
 it("shows literal source names and rejects executable source links",()=>{
  const html=renderToStaticMarkup(<SourceDetails source={{provider:"google",label:"Google Calendar",url:"javascript:alert(1)"}}/>);
  expect(html).toContain("Google Calendar");expect(html).not.toContain('href="javascript');
 });
 it("formats remote names without showing JSON objects",()=>{
  const html=renderToStaticMarkup(<SourceDetails source={{provider:"linear",label:"Linear",details:{team:{id:"secret-id",name:"Product"},labels:{nodes:[{id:"one",name:"Design"}]}}}}/>);
  expect(html).toContain("Product");expect(html).toContain("Design");expect(html).not.toContain("secret-id");
 });
});
