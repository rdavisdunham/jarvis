import {chromium,expect} from "@playwright/test";
import {mkdirSync} from "node:fs";
const browser=await chromium.launch();const page=await browser.newPage({viewport:{width:1440,height:1000}});const errors=[];page.on("pageerror",e=>errors.push(e.message));
const base=process.env.JARVIS_PLANNER_TEST_URL;
async function request(path,body){return page.evaluate(async({path,body})=>{const boot=await(await fetch("/api/v1/bootstrap")).json();const r=await fetch("/api/v1"+path,{method:body?"POST":"GET",headers:{"Content-Type":"application/json","X-CSRF-Token":boot.csrf},...(body?{body:JSON.stringify(body)}:{})});const data=await r.json();if(!r.ok)throw Error(JSON.stringify(data));return data;},{path,body});}
const cmd=async(tool,args)=>(await request("/commands",{command_id:crypto.randomUUID(),tool,arguments:args})).data;
const ui=async(name,args)=>{const r=await request("/__test_ui",{name,arguments:args});if(r.status!=="displayed")throw Error(JSON.stringify(r));};
try{
 await page.goto(base);await page.getByLabel("Pairing code").fill("planner-fixture");await page.locator(".login-card button.primary").click();await expect(page.getByRole("region",{name:"Today",exact:true})).toBeVisible();
 const schema=await request("/structure"),rev=schema.revision;
 const client=await cmd("record.create",{type_id:"client",title:"Organization test client",schema_revision:rev});
 const project=await cmd("record.create",{type_id:"project",title:"Organization test project",parent_id:client.id,schema_revision:rev});
 const task=await cmd("record.create",{type_id:"task",title:"Organization child",parent_id:project.id,schema_revision:rev});
 await ui("ui_records",{layout:"browse",parent_id:"",type_id:""});
 const browse=page.getByRole("region",{name:"Browse organization"});
 await expect(browse).toBeVisible();await page.evaluate(()=>window.__navigationSentinel="kept");
 await browse.getByRole("button",{name:client.title,exact:true}).click();
 await expect(browse.getByRole("heading",{name:client.title})).toBeVisible();
 await browse.getByRole("button",{name:project.title,exact:true}).click();
 await expect(browse.getByRole("heading",{name:project.title})).toBeVisible();
 await expect.poll(()=>new URL(page.url()).searchParams.get("home")).toBe(project.id);
 await page.goBack();await expect(browse.getByRole("heading",{name:client.title})).toBeVisible();
 expect(await page.evaluate(()=>window.__navigationSentinel)).toBe("kept");
 await page.goForward();await expect(browse.getByRole("heading",{name:project.title})).toBeVisible();
 await page.reload();await expect(browse.getByRole("heading",{name:project.title})).toBeVisible();
 await browse.getByRole("button",{name:"Add here"}).click();await page.getByLabel("New record title").fill("Added in home");await browse.getByRole("button",{name:"Add",exact:true}).click();
 await expect.poll(async()=>(await request("/structure/browse?parent_id="+project.id)).total).toBe(2);
 await browse.getByRole("button",{name:"Details for "+task.title,exact:true}).click();
 await page.getByLabel("New subtask").fill("Nested without a new type");await page.getByRole("button",{name:"Add subtask",exact:true}).click();
 await expect(page.getByRole("region",{name:"Work in this home"})).toContainText("Nested without a new type");
 expect((await request("/structure/browse?parent_id="+task.id)).items[0].type_id).toBe("task");
 await page.getByRole("button",{name:"Close record"}).click();
 await browse.getByRole("button",{name:"Details",exact:true}).click();await page.getByRole("button",{name:"Choose main home"}).click();
 await page.getByRole("button",{name:"Leave unfiled",exact:true}).click();
 const move=page.getByRole("dialog",{name:"Move contents",exact:true});await expect(move).toBeVisible();
 await move.getByLabel(/Move only this record/).check();await expect(move).toContainText("2 children promoted");
 await move.getByRole("button",{name:"Move",exact:true}).click();await expect(move).toHaveCount(0);
 expect((await request("/structure/records/"+task.id)).parent_id).toBe(client.id);
 await page.getByRole("button",{name:"Close record"}).click();
 // A task with a subtask moves straight away, subtask included: no contents question.
 const subtask=(await request("/structure/browse?parent_id="+task.id)).items[0];
 await ui("ui_records",{record_id:task.id,open_details:true});
 for(const home of [project,client]){
  await page.getByRole("button",{name:"Choose main home"}).click();
  await page.getByLabel("Find a home").fill(home.title);
  await page.getByRole("button",{name:"Use "+home.title+" as home"}).click();
  await expect.poll(async()=>(await request("/structure/records/"+task.id)).parent_id).toBe(home.id);
 }
 await expect(page.getByRole("dialog",{name:"Move contents",exact:true})).toHaveCount(0);
 expect((await request("/structure/records/"+subtask.id)).parent_id).toBe(task.id);
 await page.getByRole("button",{name:"Close record"}).click();
 await ui("ui_records",{layout:"browse",parent_id:"",type_id:""});
 await page.getByRole("button",{name:"Structure",exact:true}).click();
 const editor=page.getByRole("dialog",{name:"Workspace structure"});
 await expect(editor.getByRole("navigation",{name:"Type and field tree"})).toBeVisible();
 await editor.getByRole("button",{name:"Clients",exact:true}).click();await editor.getByLabel("Singular name").fill("Customer");
 await editor.getByRole("button",{name:"Advanced",exact:true}).click();await expect(editor.getByLabel("Singular name")).toHaveValue("Customer");
 await editor.getByRole("button",{name:"Visual tree",exact:true}).click();await expect(editor.getByLabel("Singular name")).toHaveValue("Customer");
 for(const width of [390,892,1440]){
  await page.setViewportSize({width,height:1000});await expect.poll(()=>page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBeTruthy();
  mkdirSync("../../artifacts/organization",{recursive:true});await page.screenshot({path:"../../artifacts/organization/types-"+width+".png",fullPage:true});
 }
 // Discard this deliberate draft, then test nested timeline collapse.
 page.once("dialog",dialog=>dialog.accept());await editor.getByRole("button",{name:"Close structure"}).click();
 await ui("ui_records",{layout:"timeline",parent_id:"",type_id:""});
 await expect(page.getByRole("button",{name:"Collapse "+client.title,exact:true})).toBeVisible();
 await page.getByRole("button",{name:"Collapse "+client.title,exact:true}).click();
 await expect(page.locator(".work-timeline-row").filter({hasText:task.title})).toHaveCount(0);
 if(errors.length)throw Error(errors.join("\n"));
 console.log("Organization acceptance passed: browse/back/deep link, Add here, item-only move, shared editor draft, responsive tree and nested timeline.");
}finally{await browser.close();}
