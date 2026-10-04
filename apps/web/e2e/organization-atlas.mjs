import {chromium,expect} from "@playwright/test";
import {mkdirSync} from "node:fs";
const browser=await chromium.launch();const page=await browser.newPage({viewport:{width:1440,height:960}});const errors=[];page.on("pageerror",e=>errors.push(e.message));
const base=process.env.JARVIS_PLANNER_TEST_URL;
async function request(path,body){return page.evaluate(async({path,body})=>{const boot=await(await fetch("/api/v1/bootstrap")).json();const r=await fetch("/api/v1"+path,{method:body?"POST":"GET",headers:{"Content-Type":"application/json","X-CSRF-Token":boot.csrf},...(body?{body:JSON.stringify(body)}:{})});const data=await r.json();if(!r.ok)throw Error(JSON.stringify(data));return data;},{path,body});}
const cmd=async(tool,args)=>(await request("/commands",{command_id:crypto.randomUUID(),tool,arguments:args})).data;
const ui=async(name,args)=>{const r=await request("/__test_ui",{name,arguments:args});if(r.status!=="displayed")throw Error(JSON.stringify(r));};
const parentOf=async id=>(await request("/structure/records/"+id)).parent_id;
try{
 await page.goto(base);await page.getByLabel("Pairing code").fill("planner-fixture");await page.locator(".login-card button.primary").click();await expect(page.getByRole("region",{name:"Today",exact:true})).toBeVisible();
 const schema=await request("/structure"),rev=schema.revision;
 // Earlier suites may rename types, so names come from the live schema.
 const T=Object.fromEntries(schema.types.map(t=>[t.id,t]));
 const mk=(type_id,title,parent,extra={})=>cmd("record.create",{type_id,title,schema_revision:rev,...(parent?{parent_id:parent.id}:{}),...extra});
 const abc=await mk("client","Atlas client ABC"),beacon=await mk("client","Atlas client Beacon");
 const ti=await mk("project","Atlas transcript project",abc),portal=await mk("project","Atlas portal project",abc);
 const task=await mk("task","Atlas central docs",ti),sub=await mk("task","Atlas check examples",task);await mk("note","Atlas discovery notes",ti);
 // Open the Atlas layout from the toolbar and fly into a client with the keyboard.
 await ui("ui_records",{layout:"browse",parent_id:"",type_id:""});
 await page.getByRole("group",{name:"Layout"}).getByRole("button",{name:"Atlas",exact:true}).click();
 const atlas=page.getByRole("region",{name:"Atlas"});await expect(atlas).toBeVisible();
 const crumbs=atlas.getByRole("navigation",{name:"Atlas path"});
 await atlas.getByRole("button",{name:/: Atlas client ABC/}).focus();await page.keyboard.press("Enter");
 await expect(crumbs.getByRole("button",{name:"Atlas client ABC"})).toHaveAttribute("aria-current","location");
 await expect.poll(()=>new URL(page.url()).searchParams.get("home")).toBe(abc.id);
 await page.keyboard.press("Backspace");await expect(crumbs.getByRole("button",{name:"Atlas client ABC"})).toHaveCount(0);
 // Eri can fly the map: an item opens its home region and is selected.
 await ui("ui_records",{layout:"atlas",record_id:task.id});
 await expect(crumbs.getByRole("button",{name:"Atlas transcript project"})).toHaveAttribute("aria-current","location");
 const inspector=atlas.getByRole("complementary",{name:"Inspector"});
 await expect(inspector.getByRole("heading",{name:"Atlas central docs"})).toBeVisible();
 // Keyboard Move to…: a task moves straight away and its subtask travels with it, no contents question.
 await inspector.getByRole("button",{name:"Choose main home"}).click();
 await page.getByLabel("Find a home").fill("Atlas portal");
 await page.getByRole("button",{name:"Use Atlas portal project as home"}).click();
 const toast=page.locator(".toast").filter({hasText:"Moved Atlas central docs to Atlas portal project."});
 await expect(toast).toBeVisible();
 await expect(inspector.getByRole("group",{name:/^Move Atlas central docs/})).toHaveCount(0);
 await expect.poll(()=>parentOf(task.id)).toBe(portal.id);expect(await parentOf(sub.id)).toBe(task.id);
 await toast.getByRole("button",{name:"Undo"}).click();
 await expect(page.locator(".toast").filter({hasText:"Move undone."})).toBeVisible();
 expect(await parentOf(task.id)).toBe(ti.id);expect(await parentOf(sub.id)).toBe(task.id);
 // A project with contents still asks; choose "Move only this record" from the inspector.
 await ui("ui_records",{layout:"atlas",record_id:ti.id});
 await expect(inspector.getByRole("heading",{name:"Atlas transcript project"})).toBeVisible();
 await inspector.getByRole("button",{name:"Choose main home"}).click();
 await page.getByLabel("Find a home").fill("Atlas client Beacon");
 await page.getByRole("button",{name:"Use Atlas client Beacon as home"}).click();
 const choice=inspector.getByRole("group",{name:"Move Atlas transcript project to Atlas client Beacon"});
 await expect(choice.getByRole("button",{name:/Move with contents/})).toContainText("3 items inside move along");
 await choice.getByRole("button",{name:/Move only this record/}).click();
 const itemToast=page.locator(".toast").filter({hasText:"Moved Atlas transcript project to Atlas client Beacon without its contents."});
 await expect(itemToast).toBeVisible();
 await expect.poll(()=>parentOf(ti.id)).toBe(beacon.id);expect(await parentOf(task.id)).toBe(abc.id);
 await itemToast.getByRole("button",{name:"Undo"}).click();
 await expect(page.locator(".toast").filter({hasText:"Move undone."})).toBeVisible();
 await expect.poll(()=>parentOf(task.id)).toBe(ti.id);expect(await parentOf(ti.id)).toBe(abc.id);
 // Pointer drag shows legal homes; dropping a task on the other project moves it with its subtask, no popover.
 await ui("ui_records",{layout:"atlas",record_id:beacon.id});
 await expect(crumbs.getByRole("button",{name:"Atlas client Beacon"})).toHaveAttribute("aria-current","location");
 await ui("ui_records",{layout:"atlas",parent_id:abc.id});
 await expect(crumbs.getByRole("button",{name:"Atlas client ABC"})).toHaveAttribute("aria-current","location");
 await page.waitForTimeout(1200);
 const from=await atlas.getByRole("button",{name:/: Atlas central docs/}).boundingBox(),to=await atlas.getByRole("button",{name:/: Atlas portal project/}).boundingBox();
 await page.mouse.move(from.x+from.width/2,from.y+from.height/2);await page.mouse.down();
 await page.mouse.move(from.x+from.width/2+20,from.y+from.height/2+10,{steps:4});
 await expect(atlas.getByRole("button",{name:/: Atlas portal project/})).toHaveClass(/is-legal|is-target/);
 await expect(atlas.getByRole("button",{name:/: Atlas transcript project/})).toHaveClass(/is-illegal/);
 await page.mouse.move(to.x+to.width/2,to.y+to.height*0.8,{steps:8});await page.mouse.up();
 await expect(page.locator(".toast").filter({hasText:"Moved Atlas central docs to Atlas portal project."})).toBeVisible();
 await expect(page.getByRole("dialog",{name:"Move choice"})).toHaveCount(0);
 await expect.poll(()=>parentOf(task.id)).toBe(portal.id);expect(await parentOf(sub.id)).toBe(task.id);
 await page.locator(".toast").getByRole("button",{name:"Undo"}).click();
 await expect.poll(()=>parentOf(task.id)).toBe(ti.id);
 // Both mode: hovering Task in the Blueprint lights tasks and dims the rest.
 await atlas.getByRole("group",{name:"Atlas view"}).getByRole("button",{name:"Both"}).click();
 await ui("ui_records",{layout:"atlas",parent_id:ti.id});
 await expect(crumbs.getByRole("button",{name:"Atlas transcript project"})).toHaveAttribute("aria-current","location");
 await atlas.locator('[data-type="task"]').hover();
 await expect(atlas.getByRole("button",{name:/: Atlas central docs/})).toHaveClass(/is-lit/);
 await expect(atlas.getByRole("button",{name:/: Atlas discovery notes/})).toHaveClass(/is-dim/);
 mkdirSync("../../artifacts/organization",{recursive:true});await page.screenshot({path:"../../artifacts/organization/atlas-both-1440.png"});
 // Removing an allowed home that records use shows exactly which records live there.
 await atlas.locator('[data-type="task"]').click();
 await inspector.getByRole("button",{name:"Remove allowed home "+T.task.name}).click();
 const blocked=inspector.getByRole("alert");
 await expect(blocked).toContainText("in "+T.task.plural.toLowerCase()+" today");await expect(blocked.getByRole("button").first()).toBeVisible();
 // A free home is removed in the draft only; review happens in Types & fields.
 const skeleton=await request("/structure/atlas"),typeOf=new Map(skeleton.records.map(r=>[r.id,r.type_id]));
 const used=new Set(skeleton.records.filter(r=>r.type_id==="task"&&r.parent_id).map(r=>typeOf.get(r.parent_id)));
 const free=T.task.parent_types.find(h=>!used.has(h)&&T[h]);if(!free)throw Error("No unused home type for the draft check");
 await inspector.getByRole("button",{name:"Remove allowed home "+T[free].name}).click();
 await expect(atlas.getByRole("status").filter({hasText:"Your structure draft has 1 change"})).toBeVisible();
 await atlas.getByRole("button",{name:"Review and apply"}).click();
 const editor=page.getByRole("dialog",{name:"Workspace structure"});await expect(editor).toBeVisible();
 page.once("dialog",d=>d.accept());await editor.getByRole("button",{name:"Close structure"}).click();
 await atlas.getByRole("button",{name:"Discard draft"}).click();
 // Map controls: zoom buttons, wheel zoom that never scrolls the page, drag to pan, full screen and multiselect.
 await atlas.getByRole("group",{name:"Atlas view"}).getByRole("button",{name:"Map"}).click();
 await ui("ui_records",{layout:"atlas",parent_id:abc.id});
 await expect(crumbs.getByRole("button",{name:"Atlas client ABC"})).toHaveAttribute("aria-current","location");
 await page.waitForTimeout(1200);
 const controls=atlas.getByRole("toolbar",{name:"Map controls"});
 const zoomText=atlas.locator(".atlas-zoom"),percent=async()=>Number((await zoomText.textContent()).replace("%",""));
 const start=await percent();
 await controls.getByRole("button",{name:"Zoom in"}).click();await expect.poll(percent).toBeGreaterThan(start*1.4);
 await controls.getByRole("button",{name:"Zoom out"}).click();await expect.poll(percent).toBe(start);
 const map=await atlas.locator(".atlas-map").boundingBox();
 await page.evaluate(()=>window.scrollTo(0,0));
 await page.mouse.move(map.x+map.width/2,map.y+map.height/2);await page.mouse.wheel(0,-400);
 await expect.poll(percent).toBeGreaterThan(start);
 if(await page.evaluate(()=>window.scrollY)!==0)throw Error("Wheel over the map must not scroll the page");
 await page.mouse.wheel(0,400);await expect.poll(percent).toBeLessThan(start*1.05);
 const before=await atlas.getByRole("button",{name:/: Atlas portal project/}).boundingBox();
 // Press on empty space: the first grid point whose topmost element is the map background.
 const empty=await page.evaluate(({x,y,width,height})=>{for(let gy=y+height-60;gy>y+60;gy-=20)for(let gx=x+20;gx<x+width-120;gx+=20){const el=document.elementFromPoint(gx,gy);if(el?.classList.contains("atlas-bg"))return[gx,gy];}return null;},map);
 if(!empty)throw Error("No empty map space to pan from");
 await page.mouse.move(empty[0],empty[1]);await page.mouse.down();
 await page.mouse.move(empty[0]+80,empty[1]-40,{steps:6});await page.mouse.up();
 const after=await atlas.getByRole("button",{name:/: Atlas portal project/}).boundingBox();
 if(Math.abs(after.x-before.x-80)>3||Math.abs(after.y-before.y+40)>3)throw Error("Dragging empty space must pan the map: "+JSON.stringify([before,after]));
 await controls.getByRole("button",{name:"Full screen"}).click();
 await expect(atlas).toHaveClass(/is-full/);
 const box=await atlas.boundingBox(),vp=page.viewportSize();
 if(box.x!==0||box.y!==0||Math.round(box.width)!==vp.width||Math.round(box.height)!==vp.height)throw Error("Full screen must fill the page: "+JSON.stringify(box));
 await page.keyboard.press("Escape");await expect(atlas).not.toHaveClass(/is-full/);
 // Multiselect: pick both projects, move them into another client together, then Undo.
 await controls.getByRole("button",{name:"Select several"}).click();
 await expect(inspector.getByRole("heading",{name:"Select several"})).toBeVisible();
 await atlas.getByRole("button",{name:/: Atlas transcript project/}).click();
 await atlas.getByRole("button",{name:/: Atlas portal project/}).click();
 await expect(inspector.getByRole("heading",{name:"2 selected"})).toBeVisible();
 await inspector.getByLabel("Find a home for the selection").fill("Beacon");
 await inspector.getByRole("button",{name:"Move selection to Atlas client Beacon"}).click();
 const many=page.locator(".toast").filter({hasText:"Moved 2 records to Atlas client Beacon."});await expect(many).toBeVisible();
 await expect.poll(()=>parentOf(ti.id)).toBe(beacon.id);expect(await parentOf(portal.id)).toBe(beacon.id);expect(await parentOf(task.id)).toBe(ti.id);
 await many.getByRole("button",{name:"Undo"}).click();
 await expect.poll(()=>parentOf(ti.id)).toBe(abc.id);expect(await parentOf(portal.id)).toBe(abc.id);
 await inspector.getByRole("button",{name:"Done selecting"}).click();
 await expect(controls.getByRole("button",{name:"Select several"})).toHaveAttribute("aria-pressed","false");
 // Phones keep Browse; the Atlas option is hidden.
 await page.setViewportSize({width:390,height:900});
 await expect(page.getByRole("region",{name:"Browse organization"})).toBeVisible();
 await expect(page.getByRole("group",{name:"Layout"}).getByRole("button",{name:"Atlas",exact:true})).toHaveCount(0);
 if(errors.length)throw Error(errors.join("\n"));
 console.log("Organization Atlas acceptance passed: fly in/out, Eri focus, keyboard Move to (tasks move whole, projects ask), Undo, Blueprint hover linking, blocking home list, draft review, map controls (zoom, wheel, pan, full screen, multiselect move) and phone fallback.");
}finally{await browser.close();}
