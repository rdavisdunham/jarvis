import {chromium,expect} from "@playwright/test";
import {mkdirSync} from "node:fs";
const browser=await chromium.launch();const page=await browser.newPage({viewport:{width:1440,height:1000}});const errors=[];page.on("pageerror",e=>errors.push(e.message));
const base=process.env.JARVIS_PLANNER_TEST_URL;
async function request(path,body){return page.evaluate(async({path,body})=>{const boot=await(await fetch("/api/v1/bootstrap")).json();const r=await fetch("/api/v1"+path,{method:body?"POST":"GET",headers:{"Content-Type":"application/json","X-CSRF-Token":boot.csrf},...(body?{body:JSON.stringify(body)}:{})});const data=await r.json();if(!r.ok)throw Error(JSON.stringify(data));return data;},{path,body});}
const ui=async(name,args)=>{const r=await request("/__test_ui",{name,arguments:args});if(r.status!=="displayed")throw Error(JSON.stringify(r));};
try{
 await page.goto(base);await page.getByLabel("Pairing code").fill("planner-fixture");await page.locator(".login-card button.primary").click();await expect(page.getByRole("region",{name:"Today",exact:true})).toBeVisible();
 await page.waitForTimeout(500);await request("/__test_reviews",{});
 await ui("ui_workspace",{view:"questions"});
 const section=page.getByRole("region",{name:"Questions",exact:true});
 await expect(section.locator(".question-card")).toHaveCount(2);
 await page.evaluate(()=>window.__questionsDocument="kept");
 await page.reload();await expect(section.locator(".question-card")).toHaveCount(2);
 const field=section.locator(".question-card").filter({hasText:"Does Client mean a company or a person?"});
 await field.getByRole("button",{name:"Ask next week"}).click();await expect(field).toContainText("deferred");
 expect((await request("/questions?status=deferred")).total).toBe(1);
 await section.getByText("What has the dream sequence done?").click();await expect(section.locator(".question-learning")).toContainText("succeeded");
 for(const width of [390,892,1440]){
  await page.setViewportSize({width,height:1000});await expect.poll(()=>page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBeTruthy();
  mkdirSync("../../artifacts/questions",{recursive:true});await page.screenshot({path:"../../artifacts/questions/inbox-"+width+".png",fullPage:true});
 }
 const memory=section.locator(".question-card").filter({hasText:"Memory detail"});await memory.getByRole("button",{name:"These are different"}).click();
 await expect(section.locator(".question-card")).toHaveCount(1);
 await section.getByLabel("Question status").selectOption("resolved");await expect(section.locator(".question-card")).toHaveCount(1);
 await ui("ui_records",{layout:"browse",type_id:"",parent_id:""});
 await page.locator(".organization-questions>summary").click();await expect(page.locator(".organization-questions .question-card")).toHaveCount(1);
 await expect(page.locator(".organization-questions")).not.toContainText("Memory detail");
 await page.getByRole("button",{name:/Profile menu for/}).click();await page.getByRole("menuitem",{name:"Questions",exact:true}).click();
 await expect(section).toBeVisible();
 if(errors.length)throw Error(errors.join("\n"));
 console.log("Questions acceptance passed: personal entry/deep link, adapters, snooze, explicit resolution, provenance, Organization subset and phone/fold/desktop layouts.");
}finally{await browser.close();}
