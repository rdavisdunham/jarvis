import {chromium,expect} from "@playwright/test";
import {mkdirSync} from "node:fs";
const base=process.env.JARVIS_PLANNER_TEST_URL;
const browser=await chromium.launch();
const page=await browser.newPage({viewport:{width:820,height:1000}});
const errors=[];page.on("pageerror",error=>errors.push(error.message));
async function request(path,body){return page.evaluate(async({path,body})=>{const boot=await(await fetch("/api/v1/bootstrap")).json();const response=await fetch("/api/v1"+path,{method:body?"POST":"GET",headers:{"Content-Type":"application/json","X-CSRF-Token":boot.csrf},...(body?{body:JSON.stringify(body)}:{})});const data=await response.json();if(!response.ok)throw Error(JSON.stringify(data));return data;},{path,body});}
const cmd=async(tool,args)=>(await request("/commands",{command_id:crypto.randomUUID(),tool,arguments:args})).data;
const ui=async(name,args)=>{const r=await request("/__test_ui",{name,arguments:args});if(r.status!=="displayed")throw Error(JSON.stringify(r));return r;};
const settle=()=>page.waitForTimeout(150);
async function back(){await settle();await page.evaluate(()=>history.back());await settle();}
async function forward(){await settle();await page.evaluate(()=>history.forward());await settle();}
async function sameDocument(){expect(await page.evaluate(()=>window.__shellDocument)).toBe("still-here");}
async function noOverflow(){expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBeTruthy();}
try{
 await page.goto(base);await page.getByLabel("Pairing code").fill("planner-fixture");await page.locator(".login-card button.primary").click();await expect(page.getByRole("heading",{name:"Tasks",exact:true})).toBeVisible();await page.evaluate(()=>window.__shellDocument="still-here");
 const schema=await request("/structure");const task=await cmd("record.create",{type_id:"task",title:"Fold navigation target",body:"Neutrino documentation in the task details",schema_revision:schema.revision});
 await ui("ui_workspace",{view:"notes"});await expect(page.getByRole("heading",{name:"Notes",exact:true})).toBeVisible();await settle();
 await ui("ui_workspace",{view:"settings",settings_section:"profile"});await expect(page.getByLabel("Preferred name")).toBeVisible();await back();await expect(page.getByRole("heading",{name:"Notes",exact:true})).toBeVisible();await forward();await expect(page.getByLabel("Preferred name")).toBeVisible();await sameDocument();
 for(const section of ["profile","organization","notifications","voice","integrations","privacy","sharing","system"]){
   await ui("ui_workspace",{view:"settings",settings_section:section});await expect(page.locator(".settings-content")).toBeVisible();await noOverflow();await expect(page.getByRole("button",{name:"Open Eridani",exact:true})).toBeVisible();
 }
 await ui("ui_workspace",{view:"settings",settings_section:"organization"});await page.getByRole("heading",{name:"Organization learning",exact:true}).waitFor();
 mkdirSync("../../artifacts/custom-planner",{recursive:true});await page.screenshot({path:"../../artifacts/custom-planner/fold-settings.png",fullPage:true});
 await page.getByRole("button",{name:"Open Eridani",exact:true}).click();await expect(page.getByRole("dialog",{name:"Eridani conversation"})).toBeVisible();await page.screenshot({path:"../../artifacts/custom-planner/fold-chat.png",animations:"disabled"});await back();await expect(page.locator(".companion.visible")).toHaveCount(0);await forward();await expect(page.locator(".companion.visible")).toBeVisible();await page.getByRole("button",{name:"Close Eridani",exact:true}).click();await expect(page.locator(".companion.visible")).toHaveCount(0);await settle();await sameDocument();
 await ui("ui_workspace",{view:"notifications"});await page.getByRole("button",{name:"Search tasks",exact:true}).click();await page.getByRole("textbox",{name:"Search tasks",exact:true}).fill("Neutrino");await page.getByRole("textbox",{name:"Search tasks",exact:true}).press("Enter");await expect(page.getByRole("heading",{name:"Tasks",exact:true})).toBeVisible();await expect(page.getByRole("button",{name:"Fold navigation target",exact:true})).toBeVisible();await expect(page.getByRole("combobox",{name:"Status",exact:true})).toHaveValue("all");
 await page.getByRole("button",{name:"Fold navigation target",exact:true}).click();await expect(page.getByRole("dialog",{name:"Task details"})).toBeVisible();await back();await expect(page.getByRole("dialog")).toHaveCount(0);await expect(page.getByRole("button",{name:"Fold navigation target",exact:true})).toBeVisible();await forward();await expect(page.getByRole("dialog",{name:"Task details"})).toBeVisible();await sameDocument();await page.getByRole("button",{name:"Close record",exact:true}).click();await settle();
 await ui("ui_workspace",{view:"notes"});await ui("ui_workspace",{view:"all"});await expect(page.getByRole("dialog")).toHaveCount(0);
 await ui("ui_form",{form:"note"});await ui("ui_editor",{operation:"patch",changes:{title:"Keep my unsaved draft",content:"Back must preserve these words."}});await back();const draft=await ui("ui_editor",{operation:"read"});expect(draft.data.values.title).toBe("Keep my unsaved draft");expect(draft.data.values.content).toBe("Back must preserve these words.");await sameDocument();await ui("ui_editor",{operation:"discard"});await settle();
 await ui("ui_workspace",{view:"organize"});await page.getByRole("button",{name:"Structure",exact:true}).click();await expect(page.getByRole("dialog",{name:"Workspace structure"})).toBeVisible();await back();await expect(page.getByRole("dialog",{name:"Workspace structure"})).toHaveCount(0);await forward();await expect(page.getByRole("dialog",{name:"Workspace structure"})).toBeVisible();await page.getByLabel("Singular name",{exact:true}).fill("Draft type name");await back();await expect(page.getByLabel("Singular name",{exact:true})).toHaveValue("Draft type name");await page.getByRole("button",{name:"Close structure",exact:true}).click();await settle();await sameDocument();
 for(const width of [390,600,768,1000,1440]){
  await page.setViewportSize({width,height:1000});await ui("ui_workspace",{view:"settings",settings_section:"notifications"});await expect(page.getByText("Quiet hours",{exact:true})).toBeVisible();await noOverflow();
  await page.getByRole("button",{name:"Open Eridani",exact:true}).click();await expect(page.locator(".companion.visible")).toBeVisible();await noOverflow();const panel=await page.locator(".companion.visible").boundingBox();expect(panel.x).toBeGreaterThanOrEqual(0);expect(panel.x+panel.width).toBeLessThanOrEqual(width);await page.getByRole("button",{name:"Close Eridani",exact:true}).click();await settle();
 }
 await page.emulateMedia({reducedMotion:"reduce"});await page.getByRole("button",{name:"Open Eridani",exact:true}).click();expect(await page.locator(".companion").evaluate(el=>getComputedStyle(el).animationName)).toBe("none");await page.getByRole("button",{name:"Close Eridani",exact:true}).click();await settle();
 if(errors.length)throw Error(errors.join("\n"));console.log("Shell acceptance passed: Back/Forward without reload, card and chat history, task search from notifications, all settings sections, fold/phone/desktop and reduced motion.");
} finally {await browser.close();}
