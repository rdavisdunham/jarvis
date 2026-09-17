import {chromium,expect} from "@playwright/test";
import {mkdirSync} from "node:fs";
const base=process.env.JARVIS_PLANNER_TEST_URL;
const browser=await chromium.launch();
const page=await browser.newPage({viewport:{width:390,height:844}});
const errors=[];page.on("pageerror",error=>errors.push(error.message));
const requests=[];
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
 await page.goto(base);await page.getByLabel("Pairing code").fill("planner-fixture");await page.locator(".login-card button.primary").click();await expect(page.getByRole("heading",{name:"Tasks",exact:true})).toBeVisible();await page.getByRole("button",{name:"Open Eridani",exact:true}).click();
 const navigation=await send("Show me my calendar");await expect(card(navigation.id)).toHaveCount(0);await finish(navigation,{navigation:true});await expect(page.locator(".message.assistant").filter({hasText:"Here is your calendar."})).toBeVisible();await expect(card(navigation.id)).toHaveCount(0);
 const first=await send("Add a task to call Alex");const second=await send("Add a task to buy milk");await finish(second,{title:"Buy milk"});await expect(card(second.id)).toBeVisible();await finish(first,{title:"Call Alex"});await expect(card(first.id)).toBeVisible();
 let entries=await order();expect(entries.indexOf("card:"+first.id)).toBeLessThan(entries.indexOf("Add a task to buy milk"));expect(entries.indexOf("card:"+second.id)).toBeGreaterThan(entries.indexOf("Add a task to buy milk"));
 await expect(card(first.id).getByRole("button",{name:"Edit",exact:true})).toBeVisible();await expect(card(first.id).getByRole("button",{name:"Revert",exact:true})).toBeEnabled();await expect(card(first.id).getByRole("button",{name:"Details",exact:true})).toHaveAttribute("aria-expanded","false");await expect(card(first.id).getByText("Original request",{exact:true})).toHaveCount(0);
 await card(first.id).getByRole("button",{name:"Details",exact:true}).click();await expect(card(first.id).getByText("Original request",{exact:true})).toBeVisible();await card(first.id).getByRole("button",{name:"Less",exact:true}).click();
 const third=await send("And a task to read tonight");entries=await order();expect(entries.indexOf("card:"+first.id)).toBeLessThan(entries.indexOf("And a task to read tonight"));expect(entries.indexOf("card:"+second.id)).toBeLessThan(entries.indexOf("And a task to read tonight"));
 await card(first.id).getByRole("button",{name:"Revert",exact:true}).click();await expect(card(first.id).getByRole("button",{name:"Reverted",exact:true})).toBeDisabled();
 await page.reload();await expect(page.getByRole("heading",{name:"Tasks",exact:true})).toBeVisible();await page.getByRole("button",{name:"Open Eridani",exact:true}).click();await expect(card(first.id)).toBeVisible();await expect(card(navigation.id)).toHaveCount(0);entries=await order();expect(entries.indexOf("card:"+first.id)).toBeLessThan(entries.indexOf("Add a task to buy milk"));expect(entries.indexOf("card:"+second.id)).toBeLessThan(entries.indexOf("And a task to read tonight"));
 const notices=await page.evaluate(async()=>(await(await fetch("/api/v1/notifications")).json()).items);expect(notices.some(n=>n.category==="work_result")).toBe(false);
 await page.locator(".messages").evaluate(el=>{el.scrollTop=0;});mkdirSync("../../artifacts/chat-activity",{recursive:true});await page.screenshot({path:"../../artifacts/chat-activity/phone.png",animations:"disabled"});
 await page.setViewportSize({width:1440,height:1000});await page.screenshot({path:"../../artifacts/chat-activity/desktop.png",animations:"disabled"});expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
 if(errors.length)throw Error(errors.join("\n"));console.log("Chat activity acceptance passed: silent navigation, compact Edit/Revert, reverse completion order, anchored cards after another request and reload, no completion notifications.");
}finally{await browser.close();}
