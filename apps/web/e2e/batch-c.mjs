import {chromium,expect} from "@playwright/test";
import {mkdirSync} from "node:fs";
const base=process.env.JARVIS_PLANNER_TEST_URL;
const browser=await chromium.launch();const page=await browser.newPage({viewport:{width:390,height:844}});const errors=[];page.on("pageerror",e=>errors.push(e.message));
const setupRequests=[];
await page.route("**/api/v1/bootstrap",async route=>{const response=await route.fetch();const json=await response.json();if(json.capabilities)json.capabilities.chat=true;await route.fulfill({response,json});});
await page.route("**/api/v1/work",async route=>{
 if(route.request().method()!=="POST")return route.continue();
 const body=route.request().postDataJSON();
 const response=await route.fetch({url:base+"/api/v1/__test_work"});const json=await response.json();
 if(!response.ok())throw Error(JSON.stringify(json));
 setupRequests.push({body,item:json});
 await route.fulfill({response,json});
});
async function request(path,body){return page.evaluate(async({path,body})=>{const boot=await(await fetch("/api/v1/bootstrap")).json();const r=await fetch("/api/v1"+path,{method:body?"POST":"GET",headers:{"Content-Type":"application/json","X-CSRF-Token":boot.csrf},...(body?{body:JSON.stringify(body)}:{})});const data=await r.json();if(!r.ok)throw Error(JSON.stringify(data));return data;},{path,body});}
const ui=async(name,args)=>{const r=await request("/__test_ui",{name,arguments:args});if(r.status!=="displayed")throw Error(JSON.stringify(r));return r;};
try{
 await page.goto(base);await page.getByLabel("Pairing code").fill("planner-fixture");await page.locator(".login-card button.primary").click();await expect(page.getByRole("region",{name:"Today",exact:true})).toBeVisible();
 await page.getByRole("button",{name:"+ Quick list",exact:true}).click();await page.getByLabel("List title",{exact:true}).fill("Houston packing acceptance");await page.getByLabel("Checklist",{exact:true}).fill("# Hayes\nPack Hayes food\n# Personal\nPack clothes\n# Moped\nGather tools\n# Work\nPack laptop");await page.getByLabel("Pin to Today").check();await page.getByRole("button",{name:"Create list",exact:true}).click();
 await expect(page.getByRole("dialog",{name:"Quick list",exact:true})).toBeVisible();
 await page.getByLabel("Complete Pack Hayes food",{exact:true}).click();await expect(page.getByRole("status").filter({hasText:"1 of 4 done"})).toBeVisible();
 await page.getByLabel("New checklist item",{exact:true}).fill("Pack charger");await page.getByRole("dialog",{name:"Quick list",exact:true}).getByRole("button",{name:"Add",exact:true}).click();await expect(page.getByRole("status").filter({hasText:"1 of 5 done"})).toBeVisible();
 const row=(await request("/quick-lists?query=Houston packing acceptance")).items[0];const ids=row.items.map(t=>t.id);
 await page.getByRole("button",{name:"Move up Pack charger",exact:true}).click();await expect.poll(async()=>(await request("/quick-lists/"+row.id)).items[3].title).toBe("Pack charger");
 const fresh=await request("/quick-lists/"+row.id);const laptop=fresh.items.find(t=>t.title==="Pack laptop");
 await request("/commands",{command_id:crypto.randomUUID(),tool:"quicklist.item",arguments:{list_id:row.id,expected_revision:fresh.revision,operation:"edit",item_id:laptop.id,expected_item_revision:laptop.revision,title:"Pack laptop and adapter"}});
 await expect.poll(async()=>page.getByLabel("Item 5",{exact:true}).inputValue()).toBe("Pack laptop and adapter");
 mkdirSync("../../artifacts/batch-c",{recursive:true});await page.getByRole("dialog",{name:"Quick list",exact:true}).evaluate(el=>{el.scrollTop=0;});await page.screenshot({path:"../../artifacts/batch-c/quick-list-phone.png"});
 await page.getByRole("button",{name:"Close Quick list",exact:true}).click();await ui("ui_workspace",{view:"all"});await expect(page.getByRole("button",{name:"Houston packing acceptance",exact:true})).toBeVisible();await expect(page.getByRole("button",{name:"Pack clothes",exact:true})).toHaveCount(0);
 await page.getByRole("button",{name:"Houston packing acceptance",exact:true}).click();await expect(page.getByLabel("Complete Pack Hayes food")).toBeChecked();await page.getByText("Use as normal tasks",{exact:true}).click();await page.getByRole("button",{name:"Promote to tasks",exact:true}).click();await expect(page.getByRole("dialog")).toHaveCount(0);
 await expect.poll(async()=>(await request("/tasks/"+row.id)).is_quick_list).toBe(false);for(const id of ids)expect((await request("/tasks/"+id)).id).toBe(id);
 await ui("ui_workspace",{view:"settings",settings_section:"organization"});await page.getByRole("button",{name:"Fill out the form",exact:true}).click();await page.getByLabel("Setup name").fill("Morgan");await page.getByLabel("Setup purpose").fill("Work projects and travel lists");await page.getByRole("button",{name:"Save progress",exact:true}).click();await expect.poll(async()=>(await request("/onboarding")).preferred_name).toBe("Morgan");
 await page.getByText("How would you like to organize it?",{exact:true}).click();await page.getByLabel("Grouping name").fill("Course");await page.getByLabel("Grouping description").fill("Classes I attend with assignments and lecture notes.");await page.getByLabel("Grouping example").fill("Biology contains my lab project and notes.");await page.getByRole("button",{name:"Preview my organization",exact:true}).click();await expect(page.getByRole("heading",{name:"Preview — nothing added yet",exact:true})).toBeVisible();
 expect((await request("/structure")).types.some(t=>t.id==="setup-group")).toBe(false);await page.getByRole("button",{name:"Confirm this setup",exact:true}).click();await expect.poll(async()=>(await request("/onboarding")).status).toBe("completed");expect((await request("/structure")).types.find(t=>t.id==="setup-group").name).toBe("Course");
 await page.reload();await expect(page.getByRole("button",{name:"Update with Dani",exact:true})).toBeVisible();await expect(page.getByLabel("Setup name")).toHaveCount(0);
 for(const width of [390,892]){await page.setViewportSize({width,height:900});await expect.poll(()=>page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBeTruthy();await page.screenshot({path:`../../artifacts/batch-c/settings-${width}.png`,fullPage:true});}

 // Set up with Dani saves the form's answers first, then asks for voice or typing before the conversation starts.
 await page.getByRole("button",{name:"Fill out the form",exact:true}).click();
 await page.getByLabel("Setup design notes").fill("A customer can have several projects. ABC is an example only.");
 await page.getByRole("button",{name:"Open Dani",exact:true}).click();
 await page.getByRole("textbox",{name:"Message Dani",exact:true}).fill("Keep this unsent draft.");
 await page.getByRole("button",{name:"Close Dani",exact:true}).click();
 await page.getByRole("button",{name:"Talk to Dani",exact:true}).click();
 const setup=page.getByRole("region",{name:"Set up with Dani",exact:true});
 await expect(setup).toContainText("Want to do this by voice?");
 await setup.getByRole("button",{name:"Type instead",exact:true}).click();
 await expect.poll(async()=>(await request("/onboarding")).design_notes).toBe("A customer can have several projects. ABC is an example only.");
 const composer=page.getByRole("textbox",{name:"Message Dani",exact:true});
 await expect(page.locator(".message.assistant").filter({hasText:"Tell me about you."})).toHaveCount(1);
 await expect(composer).toHaveAttribute("placeholder","Type your answer…");
 await expect(composer).toHaveValue("Keep this unsent draft.");
 await expect(page.getByRole("button",{name:"Switch to voice",exact:true})).toBeVisible();
 const session=await request("/setup");expect(session.status).toBe("interviewing");expect(session.mode).toBe("typed");
 await composer.fill("I teach chemistry and coach soccer.");
 await page.getByRole("button",{name:"Send message",exact:true}).click();
 await expect.poll(()=>setupRequests.length).toBe(1);
 expect(setupRequests[0].body.purpose).toBe("setup");expect(setupRequests[0].body.conversation_id).toBe(session.conversation_id);
 await page.getByRole("button",{name:"Close Dani",exact:true}).click();
 // Resume returns to the same typed conversation without asking again; a rapid second click starts nothing new.
 await setup.getByRole("button",{name:"Resume with Dani",exact:true}).dblclick();
 await expect(page.locator(".message.user").filter({hasText:"I teach chemistry"})).toHaveCount(1);
 expect((await request("/setup")).conversation_id).toBe(session.conversation_id);
 expect(setupRequests.length).toBe(1);
 await page.getByRole("button",{name:"Close Dani",exact:true}).click();
 if(errors.length)throw Error(errors.join("\n"));console.log("Batch C browser passed: compact capture, completion/reorder/reopen/promotion, setup preview/confirmation/resume, guided launch/dirty save/unknown-response retry and phone/fold layout.");
}finally{await browser.close();}
