import {chromium,expect} from "@playwright/test";
import {mkdirSync} from "node:fs";
const base=process.env.JARVIS_PLANNER_TEST_URL;
const browser=await chromium.launch();
const page=await browser.newPage({viewport:{width:390,height:844}});
const errors=[];page.on("pageerror",error=>errors.push(error.message));
const requests=[];
const timingResponses=[];
page.on("response",response=>{
 const request=response.request();
 if(request.method()==="POST" && /\/api\/v1\/work\/[^/]+\/latency$/.test(request.url()))
  timingResponses.push({id:request.url().split("/").at(-2),status:response.status(),...request.postDataJSON()});
});
await page.route("**/api/v1/bootstrap",async route=>{const response=await route.fetch();const json=await response.json();if(json.capabilities)json.capabilities.chat=true;await route.fulfill({response,json});});
await page.route("**/api/v1/work",async route=>{
 if(route.request().method()!=="POST")return route.continue();
 const response=await route.fetch({url:base+"/api/v1/__test_work"});const json=await response.json();if(!response.ok())throw Error(JSON.stringify(json));requests.push(json);await route.fulfill({response,json});
});
async function finish(item,body){return page.evaluate(async({id,body})=>{const boot=await(await fetch("/api/v1/bootstrap")).json();const r=await fetch("/api/v1/__test_work/"+id+"/finish",{method:"POST",headers:{"Content-Type":"application/json","X-CSRF-Token":boot.csrf},body:JSON.stringify(body)});const data=await r.json();if(!r.ok)throw Error(JSON.stringify(data));window.dispatchEvent(new Event("eri-work-changed"));return data;},{id:item.id,body});}
async function send(text){const count=requests.length;await page.getByRole("textbox",{name:"Message Eridani"}).fill(text);await page.getByRole("button",{name:"Send message",exact:true}).click();await expect.poll(()=>requests.length).toBe(count+1);return requests.at(-1);}
const card=id=>page.locator(`.messages [data-work-id="${id}"]`);
async function order(){return page.locator(".messages").evaluate(el=>[...el.children].filter(e=>e.matches(".message,.work-card")).map(e=>e.getAttribute("data-work-id")?"card:"+e.getAttribute("data-work-id"):e.querySelector("p")?.textContent));}
try{
 await page.goto(base);await page.getByLabel("Pairing code").fill("planner-fixture");await page.locator(".login-card button.primary").click();await expect(page.getByRole("region",{name:"Today",exact:true})).toBeVisible();await page.getByRole("button",{name:"Open Eridani",exact:true}).click();
 const navigation=await send("Show me my calendar");await expect(card(navigation.id)).toHaveCount(0);await finish(navigation,{navigation:true});await expect(page.locator(".message.assistant").filter({hasText:"Here is your calendar."})).toBeVisible();await expect(card(navigation.id)).toHaveCount(0);
 await expect.poll(()=>timingResponses.some(t=>t.id===navigation.id && t.stage==="work_reply_rendered" && t.status===200)).toBe(true);
 const first=await send("Add a task to call Alex");const second=await send("Add a task to buy milk");await finish(second,{title:"Buy milk"});await expect(card(second.id)).toBeVisible();await finish(first,{title:"Call Alex"});await expect(card(first.id)).toBeVisible();
 await expect.poll(()=>timingResponses.some(t=>t.id===second.id && t.status===200)).toBe(true);
 expect(timingResponses.every(t=>Number.isFinite(t.elapsed_ms) && t.elapsed_ms>=0 && t.elapsed_ms<=300000)).toBe(true);
 let entries=await order();expect(entries.indexOf("card:"+first.id)).toBeLessThan(entries.indexOf("Add a task to buy milk"));expect(entries.indexOf("card:"+second.id)).toBeGreaterThan(entries.indexOf("Add a task to buy milk"));
 await expect(card(first.id).getByRole("button",{name:"Edit",exact:true})).toBeVisible();await expect(card(first.id).getByRole("button",{name:"Revert",exact:true})).toBeEnabled();await expect(card(first.id).getByRole("button",{name:"Details",exact:true})).toHaveAttribute("aria-expanded","false");await expect(card(first.id).getByText("Original request",{exact:true})).toHaveCount(0);
 await card(first.id).getByRole("button",{name:"Details",exact:true}).click();await expect(card(first.id).getByText("Original request",{exact:true})).toBeVisible();await card(first.id).getByRole("button",{name:"Less",exact:true}).click();
 const third=await send("And a task to read tonight");entries=await order();expect(entries.indexOf("card:"+first.id)).toBeLessThan(entries.indexOf("And a task to read tonight"));expect(entries.indexOf("card:"+second.id)).toBeLessThan(entries.indexOf("And a task to read tonight"));
 await card(first.id).getByRole("button",{name:"Revert",exact:true}).click();await expect(card(first.id).getByRole("button",{name:"Reverted",exact:true})).toBeDisabled();
 const measuredBeforeReload=timingResponses.length;
 await page.reload();await expect(page.getByRole("region",{name:"Today",exact:true})).toBeVisible();await page.getByRole("button",{name:"Open Eridani",exact:true}).click();await expect(card(first.id)).toBeVisible();await expect(card(navigation.id)).toHaveCount(0);entries=await order();expect(entries.indexOf("card:"+first.id)).toBeLessThan(entries.indexOf("Add a task to buy milk"));expect(entries.indexOf("card:"+second.id)).toBeLessThan(entries.indexOf("And a task to read tonight"));
 expect(timingResponses.length).toBe(measuredBeforeReload);
 const notices=await page.evaluate(async()=>(await(await fetch("/api/v1/notifications")).json()).items);expect(notices.some(n=>n.category==="work_result")).toBe(false);

 // Recovery uses the real encrypted inbox, listing and disposition paths. Only model-provider
 // availability is stubbed; no provider call or external write is possible in this fixture.
 await page.route("**/api/v1/voice/drafts/*/send", async route => {
   const url=route.request().url().replace("/api/v1/voice/drafts/", "/api/v1/__test_voice_draft/");
   const response=await route.fetch({url}); if(!response.ok()) console.error("Draft send failed", response.status(), await response.text()); await route.fulfill({response});
 });
 async function makeDraft(message) {
   return page.evaluate(async ({conversation_id,message}) => {
     const boot=await(await fetch("/api/v1/bootstrap")).json();
     const response=await fetch("/api/v1/__test_voice_draft",{method:"POST",headers:{"Content-Type":"application/json","X-CSRF-Token":boot.csrf},body:JSON.stringify({conversation_id,message})});
     if(!response.ok)throw Error(await response.text());
     window.dispatchEvent(new Event("eri-voice-drafts-changed"));
     return response.json();
   },{conversation_id:first.conversation_id,message});
 }
 await makeDraft("Bring the packing checklist tomorrow");
 const recovered=page.getByRole("region",{name:"Unsent voice draft"});
 await expect(recovered).toBeVisible();
 await page.reload(); await page.getByRole("button",{name:"Open Eridani",exact:true}).click();
 await expect(recovered).toBeVisible();
 await recovered.getByRole("textbox").fill("Add a task to bring the packing checklist tomorrow");
 await recovered.getByRole("button",{name:"Send",exact:true}).click();
 await expect(recovered).toHaveCount(0);
 const recoveredWork=await page.evaluate(async()=>(await(await fetch("/api/v1/work")).json()).items);
 expect(recoveredWork.filter(w=>w.request==="Add a task to bring the packing checklist tomorrow")).toHaveLength(1);
 await makeDraft("Actually never mind, I need to think about this");
 await expect(recovered).toBeVisible();
 await recovered.getByRole("button",{name:"Discard",exact:true}).click();
 await expect(recovered).toHaveCount(0);
 await page.locator(".messages").evaluate(el=>{el.scrollTop=0;});mkdirSync((process.env.ERIDANI_EVAL_ARTIFACT_DIR || "../../artifacts/chat-activity"),{recursive:true});await page.screenshot({path:(process.env.ERIDANI_EVAL_ARTIFACT_DIR || "../../artifacts/chat-activity") + "/phone.png",animations:"disabled"});
 await page.setViewportSize({width:1440,height:1000});await page.screenshot({path:(process.env.ERIDANI_EVAL_ARTIFACT_DIR || "../../artifacts/chat-activity") + "/desktop.png",animations:"disabled"});expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
 if(errors.length)throw Error(errors.join("\n"));console.log("Chat activity acceptance passed: silent navigation, compact Edit/Revert, reverse completion order, anchored cards after another request and reload, no completion notifications.");
}finally{await browser.close();}
