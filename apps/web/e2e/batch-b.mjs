import {chromium,expect} from "@playwright/test";
import {mkdirSync} from "node:fs";
const browser=await chromium.launch();const page=await browser.newPage({viewport:{width:1440,height:1000}});const errors=[];page.on("pageerror",e=>errors.push(e.message));
const base=process.env.JARVIS_PLANNER_TEST_URL;
async function request(path,body){return page.evaluate(async({path,body})=>{const boot=await(await fetch("/api/v1/bootstrap")).json();const r=await fetch("/api/v1"+path,{method:body?"POST":"GET",headers:{"Content-Type":"application/json","X-CSRF-Token":boot.csrf},...(body?{body:JSON.stringify(body)}:{})});const data=await r.json();if(!r.ok)throw Error(JSON.stringify(data));return data;},{path,body});}
const cmd=async(tool,args)=>(await request("/commands",{command_id:crypto.randomUUID(),tool,arguments:args})).data;
const ui=async(name,args)=>{const r=await request("/__test_ui",{name,arguments:args});if(r.status!=="displayed")throw Error(JSON.stringify(r));return r;};
try{
 await page.goto(base);await page.getByLabel("Pairing code").fill("planner-fixture");await page.locator(".login-card button.primary").click();await expect(page.getByRole("region",{name:"Today",exact:true})).toBeVisible();
 const schema=await request("/structure");const revision=schema.revision;
 const home=await cmd("record.create",{type_id:"client",title:"Batch B client",schema_revision:revision});
 const a=await cmd("record.create",{type_id:"project",title:"Batch B project A",parent_id:home.id,schema_revision:revision});
 const b=await cmd("record.create",{type_id:"project",title:"Batch B project B",parent_id:home.id,schema_revision:revision});
 const task=await cmd("record.create",{type_id:"task",title:"Batch B editable task",parent_id:a.id,body:"Public description",schema_revision:revision});
 await ui("ui_records",{layout:"tree",type_id:"",parent_id:""});
 await expect(page.getByRole("region",{name:"Organization tree"})).toBeVisible();
 await page.getByRole("button",{name:"Move Batch B project B",exact:true}).click();await page.getByRole("button",{name:"Move up",exact:true}).click();
 await expect.poll(async()=>(await request("/structure/records/"+b.id)).sort_order).toBeLessThan(a.sort_order);
 await page.getByRole("button",{name:"Batch B project A",exact:true}).click();await expect(page.getByRole("region",{name:"Work in this home"})).toContainText(task.title);
 await page.getByRole("button",{name:task.title,exact:true}).last().click();await expect(page.getByLabel("Record title")).toHaveValue(task.title);
 await page.getByText("Eridani-only notes",{exact:true}).click();await page.getByLabel("Eridani-only notes").fill("A local annotation");await page.getByLabel("Record content").click();
 await expect.poll(async()=>(await request("/structure/records/"+task.id)).local_notes).toBe("A local annotation");
 await expect.poll(async()=>(await request("/tasks/"+task.task_id)).notes).toBe("Public description");
 // Simulate another writer after this card was opened. Keep the draft until explicit reapply.
 let current=await request("/structure/records/"+task.id);await cmd("record.update",{record_id:task.id,expected_revision:current.revision,schema_revision:revision,body:"Other writer"});
 await page.getByLabel("Record content").fill("My retained draft");await page.getByLabel("Record title").click();await expect(page.getByRole("alert")).toContainText("Your draft is kept");
 await expect(page.getByLabel("Record content")).toHaveValue("My retained draft");await page.getByRole("button",{name:"Compare saved version"}).click();await expect(page.getByRole("alert")).toContainText("Other writer");
 await page.getByRole("button",{name:"Reapply my change"}).click();await expect.poll(async()=>(await request("/structure/records/"+task.id)).body).toBe("My retained draft");
 await ui("ui_workspace",{view:"organize",layout:"tree"});await expect(page.getByRole("dialog")).toHaveCount(0);
 // The native task detail card flushes its nested annotation editor before Eri navigates.
 await page.goto(base+"?view=tasks&tab=all&record=task:"+task.task_id+"&workspace=personal");
 await expect(page.getByRole("dialog",{name:"Task details"})).toBeVisible();
 await page.getByText("Eridani-only notes",{exact:true}).click();
 await page.getByLabel("Eridani-only notes").fill("Annotation from native task card");
 await ui("ui_workspace",{view:"organize",layout:"tree"});
 await expect(page.getByRole("dialog")).toHaveCount(0);
 await expect.poll(async()=>(await request("/structure/records/"+task.id)).local_notes).toBe("Annotation from native task card");
 for(const width of [390,892]){
  await page.setViewportSize({width,height:900});await expect.poll(()=>page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBeTruthy();
  await ui("ui_workspace",{view:"settings",settings_section:"organization"});await expect(page.getByRole("heading",{name:"Scheduling availability",exact:true})).toBeVisible();
  await expect.poll(()=>page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBeTruthy();
  mkdirSync("../../artifacts/batch-b",{recursive:true});await page.screenshot({path:`../../artifacts/batch-b/settings-${width}.png`,fullPage:true});
 }
 await ui("ui_workspace",{view:"all",layout:"list"});await expect(page.locator(".source-badge").first()).toContainText("Eridani");
 if(errors.length)throw Error(errors.join("\n"));console.log("Batch B acceptance passed: tree moves, related work, local notes, draft conflict/reapply, mobile/fold settings and sources.");
}finally{await browser.close();}
