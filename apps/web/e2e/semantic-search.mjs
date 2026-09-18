import {chromium,expect} from "@playwright/test";
import {mkdirSync} from "node:fs";
const base=process.env.JARVIS_PLANNER_TEST_URL;
const browser=await chromium.launch();
const page=await browser.newPage({viewport:{width:820,height:1000}});
const errors=[];page.on("pageerror",e=>errors.push(e.message));
async function request(path,body){return page.evaluate(async({path,body})=>{const boot=await(await fetch("/api/v1/bootstrap")).json();const r=await fetch("/api/v1"+path,{method:body?"POST":"GET",headers:{"Content-Type":"application/json","X-CSRF-Token":boot.csrf},...(body?{body:JSON.stringify(body)}:{})});const data=await r.json();if(!r.ok)throw Error(JSON.stringify(data));return data;},{path,body});}
const cmd=async(tool,args)=>(await request("/commands",{command_id:crypto.randomUUID(),tool,arguments:args})).data;
const ui=async(name,args)=>{const result=await request("/__test_ui",{name,arguments:args});if(result.status!=="displayed")throw Error(JSON.stringify(result));};
try{
 await page.goto(base);await page.getByLabel("Pairing code").fill("planner-fixture");await page.locator(".login-card button.primary").click();await expect(page.getByRole("heading",{name:"Tasks",exact:true})).toBeVisible();
 const schema=await request("/structure");const create=(type_id,title,extra={})=>cmd("record.create",{type_id,title,schema_revision:schema.revision,...extra});
 const abc=await create("client","Synthetic Pest ABC",{body:"Commercial pest control company"});
 const task=await create("task","Renew spraying coverage",{parent_id:abc.id});
 const loose=await create("task","Review field estimate",{body:"Termite control service contract"});
 await create("note","Termite briefing",{body:"Pest service visit notes"});
 await create("task","Ship software release");
 await request("/__test_search_index",{});
 await page.getByRole("button",{name:"Search tasks",exact:true}).click();await page.getByRole("textbox",{name:"Search tasks",exact:true}).fill("extermination");await page.getByRole("textbox",{name:"Search tasks",exact:true}).press("Enter");
 await expect(page.getByRole("button",{name:task.title,exact:true})).toBeVisible();await expect(page.getByRole("button",{name:loose.title,exact:true})).toBeVisible();await expect(page.getByRole("button",{name:"Termite briefing",exact:true})).toHaveCount(0);await expect(page.getByRole("button",{name:"Ship software release",exact:true})).toHaveCount(0);
 const found=await request("/search/records",{query:"the pest control company",resolved:["record:"+abc.id],capability:"work"});
 await request("/search/selection",{search_id:found.search_id,target_key:"record:"+abc.id,phrase:"pest control company",record_ids:[task.id,loose.id]});
 await ui("ui_records",{record_ids:[task.id,loose.id],search_id:found.search_id});
 await expect(page.getByText("2 search results",{exact:false})).toBeVisible();await page.getByRole("button",{name:task.title,exact:true}).click();await expect(page.getByRole("dialog",{name:"Task details"})).toBeVisible();await page.getByRole("button",{name:"Close record",exact:true}).click();
 await expect.poll(async()=>(await request("/search/aliases")).items.length).toBe(1);
 await ui("ui_workspace",{view:"settings",settings_section:"organization"});await expect(page.getByRole("heading",{name:"Search aliases",exact:true})).toBeVisible();const aliases=page.locator(".search-alias-settings");await expect(aliases.getByText(/pest control company.*Synthetic Pest ABC/)).toBeVisible();
 await aliases.getByRole("button",{name:"Confirm",exact:true}).click();await expect(aliases.getByText(/confirmed/)).toBeVisible();await aliases.getByRole("button",{name:"Pause",exact:true}).click();await expect(aliases.getByText(/paused/)).toBeVisible();
 for(const width of [390,820,1440]){await page.setViewportSize({width,height:1000});expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBeTruthy();}
 mkdirSync("../../artifacts/semantic-search",{recursive:true});await page.screenshot({path:"../../artifacts/semantic-search/aliases.png",fullPage:true});
 await aliases.getByRole("button",{name:"Forget",exact:true}).click();await expect(aliases.getByText("No aliases learned yet.")).toBeVisible();
 expect(errors).toEqual([]);console.log("Semantic browser acceptance passed: task-scoped meaning search, selected mixed results, click acceptance, alias controls and responsive settings.");
}finally{await browser.close();}
