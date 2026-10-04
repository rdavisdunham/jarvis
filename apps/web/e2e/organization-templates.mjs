import {chromium,expect} from "@playwright/test";
import {mkdirSync} from "node:fs";
const browser=await chromium.launch();const page=await browser.newPage({viewport:{width:1440,height:960}});const errors=[];page.on("pageerror",e=>errors.push(e.message));
const base=process.env.JARVIS_PLANNER_TEST_URL;
async function request(path,body){return page.evaluate(async({path,body})=>{const boot=await(await fetch("/api/v1/bootstrap")).json();const r=await fetch("/api/v1"+path,{method:body?"POST":"GET",headers:{"Content-Type":"application/json","X-CSRF-Token":boot.csrf},...(body?{body:JSON.stringify(body)}:{})});const data=await r.json();if(!r.ok)throw Error(JSON.stringify(data));return data;},{path,body});}
const cmd=async(tool,args)=>(await request("/commands",{command_id:crypto.randomUUID(),tool,arguments:args})).data;
const ui=async(name,args)=>{const r=await request("/__test_ui",{name,arguments:args});if(r.status!=="displayed")throw Error(JSON.stringify(r));};
const children=async id=>(await request("/structure/browse?parent_id="+id+"&limit=100")).items;
try{
 await page.goto(base);await page.getByLabel("Pairing code").fill("planner-fixture");await page.locator(".login-card button.primary").click();await expect(page.getByRole("region",{name:"Today",exact:true})).toBeVisible();
 const schema=await request("/structure"),rev=schema.revision;
 const T=Object.fromEntries(schema.types.map(t=>[t.id,t]));
 const mk=(type_id,title,parent,extra={})=>cmd("record.create",{type_id,title,schema_revision:rev,...(parent?{parent_id:parent.id}:{}),...extra});
 const acme=await mk("client","Template client Acme"),beacon=await mk("client","Template client Beacon");
 const source=await mk("project","Acme onboarding",acme,{body:"## Goals"});
 for(const title of ["Contract","Access","Kickoff","Discovery"])await mk("task",title,source,{values:{due_date:"2026-12-01"}});
 // Save as template from the record's detail card; deadlines stay behind.
 await ui("ui_records",{record_id:source.id,open_details:true});
 const card=page.getByRole("dialog",{name:T.project.name+" details"});
 await card.getByRole("button",{name:"Save as template"}).click();
 const save=page.getByRole("dialog",{name:"Save as template"});
 await expect(save.getByRole("checkbox",{name:"Include the records inside"})).toBeChecked();
 await save.getByLabel("Template name").fill("Client onboarding");
 await save.getByRole("button",{name:"Save template"}).click();
 await expect(card.getByRole("status").filter({hasText:"Saved template Client onboarding"})).toBeVisible();
 const saved=(await request("/structure/templates?type_id=project")).items.find(t=>t.name==="Client onboarding");
 if(saved.payload.children.map(c=>c.title).join()!=="Contract,Access,Kickoff,Discovery"||saved.payload.children.some(c=>"due_date" in c.values))throw Error("Template snapshot: "+JSON.stringify(saved.payload));
 await card.getByRole("button",{name:"Close record"}).click();
 // Atlas: a dashed stamp in Beacon's Add here previews the records, then creates them in one step.
 await ui("ui_records",{layout:"browse",parent_id:"",type_id:""});
 await page.getByRole("group",{name:"Layout"}).getByRole("button",{name:"Atlas",exact:true}).click();
 const atlas=page.getByRole("region",{name:"Atlas"});await expect(atlas).toBeVisible();
 await ui("ui_records",{layout:"atlas",parent_id:beacon.id});
 await expect(atlas.getByRole("navigation",{name:"Atlas path"}).getByRole("button",{name:"Template client Beacon"})).toHaveAttribute("aria-current","location");
 const inspector=atlas.getByRole("complementary",{name:"Inspector"});
 const stamp=inspector.getByRole("button",{name:/Start from template: Client onboarding/});
 await expect(stamp).toContainText(/with 4 \w+: Contract, Access, Kickoff, Discovery/);
 await stamp.click();
 const start=inspector.getByRole("region",{name:"Start from template Client onboarding"});
 await start.getByLabel("New record title").fill("Beacon onboarding");
 const tree=start.getByRole("list",{name:"Records to create"});
 await expect(tree).toContainText("Beacon onboarding");await expect(tree).toContainText("Discovery");
 await expect(start).toContainText("5 records will be created.");
 mkdirSync("../../artifacts/organization",{recursive:true});await page.screenshot({path:"../../artifacts/organization/template-stamp-preview-1440.png"});
 await start.getByRole("button",{name:"Create 5 records"}).click();
 const toast=page.locator(".toast").filter({hasText:"Created Beacon onboarding from the Client onboarding template."});
 await expect(toast).toBeVisible();
 const made=(await children(beacon.id)).find(r=>r.title==="Beacon onboarding");
 const tasks=await children(made.id);
 if(tasks.map(r=>r.title).join()!=="Contract,Access,Kickoff,Discovery"||tasks.some(r=>r.values.due_date))throw Error("Instance tree: "+JSON.stringify(tasks.map(r=>[r.title,r.values.due_date])));
 await expect(inspector.getByRole("heading",{name:"Beacon onboarding"})).toBeVisible();
 await toast.getByRole("button",{name:"Undo"}).click();
 await expect(page.locator(".toast").filter({hasText:"Undone. The new records were archived."})).toBeVisible();
 if(!(await request("/structure/records/"+made.id)).archived||!(await request("/structure/records/"+tasks[0].id)).archived)throw Error("Undo did not archive the created records");
 // Browse Add here: Start from template opens a preview dialog.
 await ui("ui_records",{layout:"browse",parent_id:acme.id,type_id:""});
 const browse=page.getByRole("region",{name:"Browse organization"});
 await browse.getByRole("button",{name:"Add here"}).click();
 await browse.getByLabel("Type to add").selectOption("project");
 await browse.getByRole("button",{name:"Start from template"}).click();
 const dialog=page.getByRole("dialog",{name:"Start from template"});
 await expect(dialog.getByRole("list",{name:"Records to create"})).toContainText("Kickoff");
 await dialog.getByRole("button",{name:"Create 5 records"}).click();
 await expect(page.locator(".toast").filter({hasText:"Created Client onboarding from the Client onboarding template."})).toBeVisible();
 const instance=(await children(acme.id)).find(r=>r.title==="Client onboarding");
 // Template editor in Types & fields: edits save at once and never touch existing instances.
 await page.getByRole("button",{name:"Structure",exact:true}).click();
 const structure=page.getByRole("dialog",{name:"Workspace structure"});
 await structure.getByRole("navigation",{name:"Type and field tree"}).getByRole("button",{name:T.project.plural,exact:true}).click();
 const section=structure.getByRole("region",{name:"Templates"});
 await section.getByRole("listitem").filter({hasText:"Client onboarding"}).getByRole("button",{name:"Edit"}).click();
 const editor=page.getByRole("dialog",{name:"Template editor"});
 await editor.getByRole("button",{name:"Add a record inside"}).click();
 await editor.getByLabel("Title of row 5").fill("Wrap-up");
 await editor.getByRole("button",{name:"Nest Wrap-up inside the record above"}).click();
 await page.screenshot({path:"../../artifacts/organization/template-editor-1440.png"});
 await editor.getByRole("button",{name:"Save template"}).click();
 await expect(editor).toHaveCount(0);
 await expect(section).toContainText("Client onboarding");
 const edited=(await request("/structure/templates/"+saved.id));
 if(edited.revision!==2||edited.payload.children[3].children[0].title!=="Wrap-up")throw Error("Template edit: "+JSON.stringify(edited.payload));
 if((await children(instance.id)).length!==4)throw Error("A template edit changed an existing instance");
 await structure.getByRole("button",{name:"Close structure"}).click();
 if(errors.length)throw Error(errors.join("\n"));
 console.log("Organization templates acceptance passed: save as template without deadlines, Atlas stamp preview and create, Undo, Browse start from template, editor nesting and independent instances.");
}catch(e){console.error("Page errors:",errors.join("\n"));mkdirSync("../../artifacts/organization",{recursive:true});await page.screenshot({path:"../../artifacts/organization/templates-failure.png"}).catch(()=>{});throw e;}
finally{await browser.close();}
