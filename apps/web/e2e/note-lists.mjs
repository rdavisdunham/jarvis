import {chromium, expect} from "@playwright/test";
import {mkdirSync} from "node:fs";
const browser=await chromium.launch();
const page=await browser.newPage({viewport:{width:1440,height:1000}});
const errors=[];page.on("pageerror",e=>errors.push(e.message));
async function request(path,body){return page.evaluate(async({path,body})=>{
 const boot=await(await fetch("/api/v1/bootstrap")).json();
 const response=await fetch("/api/v1"+path,{method:body?"POST":"GET",headers:{"Content-Type":"application/json","X-CSRF-Token":boot.csrf},...(body?{body:JSON.stringify(body)}:{})});
 const data=await response.json();if(!response.ok)throw Error(JSON.stringify(data));return data;
},{path,body});}
const command=async(tool,args)=>(await request("/commands",{command_id:crypto.randomUUID(),tool,arguments:args})).data;
const ui=async(args)=>{const result=await request("/__test_ui",{name:"ui_workspace",arguments:args});expect(result.status).toBe("displayed");};
try{
 await page.goto(process.env.JARVIS_PLANNER_TEST_URL);await page.getByLabel("Pairing code").fill("planner-fixture");
 await page.locator(".login-card button.primary").click();await expect(page.getByRole("heading",{name:"Tasks",exact:true})).toBeVisible();
 await ui({view:"notes"});await page.getByRole("button",{name:"Set up suggested lists",exact:true}).click();
 await expect(page.getByLabel("Note list")).toContainText("Movies");
 const lists=(await request("/note-lists")).items, movies=lists.find(l=>l.name==="Movies");
 const source=await command("note.create",{title:"Friday recommendations",content:"Sam recommended Arrival and Dune—watch these."});
 await request("/__test_organize_note",{note_id:source.id});
 await ui({view:"notes",note_list_id:movies.id});
 await expect(page.locator(".note-card")).toHaveCount(2);
 await expect(page.locator(".note-grid")).toContainText("Arrival");
 await expect(page.locator(".note-grid")).toContainText("Dune");
 const listIndex=await page.evaluate(()=>history.state.eriNavigation.index);
 await page.locator(".note-card").filter({has:page.locator("strong",{hasText:/^Arrival$/})}).click();
 await expect(page.locator(".note-source")).toContainText("Friday recommendations");
 await expect(page.locator(".note-source")).toContainText("Sam recommended Arrival");
 await page.getByRole("button",{name:"From Friday recommendations",exact:true}).click();
 await expect(page.locator(".note-saved-entries")).toContainText("Dune");
 await page.getByRole("button",{name:"Close note",exact:true}).click();
 await expect(page.getByRole("dialog")).toHaveCount(0);
 await expect.poll(()=>page.evaluate(()=>history.state.eriNavigation.index)).toBe(listIndex);
 await page.getByLabel("Note list").selectOption("uncategorized");
 await expect(page.locator(".note-grid")).toContainText("Friday recommendations");
 await expect.poll(()=>page.evaluate(()=>new URL(location.href).searchParams.get("note_list"))).toBe("uncategorized");
 await page.evaluate(()=>history.back());
 await expect(page.getByLabel("Note list")).toHaveValue(movies.id);
 await page.getByRole("button",{name:"Edit list",exact:true}).click();
 await page.getByLabel("Name",{exact:true}).fill("Cinema");
 await page.getByRole("button",{name:"Save list",exact:true}).click();
 await expect(page.getByLabel("Note list")).toContainText("Cinema");
 await page.getByRole("button",{name:"New list",exact:true}).click();
 await page.getByLabel("Name",{exact:true}).fill("Date night");
 await page.getByLabel("Matching tags",{exact:true}).fill("movies, date-night");
 await page.getByLabel("Description",{exact:true}).fill("Movies saved for a date night.");
 await page.getByRole("button",{name:"Save list",exact:true}).click();
 await expect(page.getByLabel("Note list")).toContainText("Date night");
 const date=(await request("/note-lists")).items.find(l=>l.name==="Date night");
 expect(date.filters.tags).toEqual(["movies","date-night"]);
 await page.getByLabel("Note list").selectOption(movies.id);
 const token=await page.evaluate(()=>{window.__notesDocument=crypto.randomUUID();return window.__notesDocument;});
 for(const width of [390,600,820,1440]){
  await page.setViewportSize({width,height:950});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBeTruthy();
  await page.getByRole("button",{name:"Edit list",exact:true}).click();
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBeTruthy();
  await page.getByRole("button",{name:"Close list settings",exact:true}).click();
 }
 expect(await page.evaluate(()=>window.__notesDocument)).toBe(token);
 mkdirSync("../../artifacts/note-lists",{recursive:true});
 await page.setViewportSize({width:390,height:844});
 await page.screenshot({path:"../../artifacts/note-lists/mobile.png",fullPage:true});
 expect(errors).toEqual([]);
 console.log("Notes lists browser acceptance passed: filters, extracted sources, edit/create lists, Eri navigation, history and responsive layouts.");
}finally{await browser.close();}
