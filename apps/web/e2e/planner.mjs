import { chromium, expect } from "@playwright/test";
import { writeFileSync } from "node:fs";
const browser=await chromium.launch();
const page=await browser.newPage({viewport:{width:1440,height:1000}});
const root=new URL("../../../",import.meta.url);
const errors=[],checks=[];
page.on("pageerror",e=>errors.push(e.message));
async function request(path,body) {
  return page.evaluate(async ({path,body})=>{
    const boot=await(await fetch("/api/v1/bootstrap")).json();
    const response=await fetch("/api/v1"+path,{method:body ? "POST":"GET",headers:{"Content-Type":"application/json","X-CSRF-Token":boot.csrf},...(body ? {body:JSON.stringify(body)}:{})});
    const result=await response.json();if(!response.ok)throw new Error(JSON.stringify(result));return result;
  },{path,body});
}
const cmd=async(tool,args)=>(await request("/commands",{command_id:crypto.randomUUID(),tool,arguments:args})).data;
const ui=async(name,args={},status="displayed")=>{
  const result=await request("/__test_ui",{name,arguments:args});
  if(result.status!==status)throw new Error(name+" expected "+status+": "+JSON.stringify(result));
  console.log(name, args.operation ?? args.layout ?? args.view ?? args.form ?? args.mode ?? status);
  checks.push(name+":"+(args.operation ?? args.layout ?? args.view ?? args.form ?? args.mode ?? status));
  return result;
};
const shot=name=>page.screenshot({path:new URL(".runtime/planner-"+name+".png",root).pathname,fullPage:true});
const close=()=>ui("ui_editor",{operation:"close"});
try {
  await page.goto(process.env.JARVIS_PLANNER_TEST_URL);
  await page.getByLabel("Pairing code").fill("planner-fixture");
  await page.locator(".login-card button.primary").click();
  await expect(page.getByRole("heading",{name:"Tasks",exact:true})).toBeVisible();
  const org=await request("/organization");
  const business=org.spaces.find(s=>s.name==="Business"),personal=org.spaces.find(s=>s.name==="Personal");
  const goal=await cmd("goal.create",{name:"Launch the planner",space_id:business.id,metric_target:100,metric_current:35});
  const project=await cmd("project.create",{name:"Website release",space_id:business.id,goal_ids:[goal.id],start_date:"2030-01-02",target_date:"2030-02-20",status:"active"});
  await cmd("project.create",{name:"Home renovation",space_id:personal.id,status:"planned"});
  for(let i=0;i<9;i++)await cmd("task.create",{title:["Review mobile layout","Fix timezone issue","Write release notes","Verify calendar sync","Link research notes","Test microphone recovery","Plan launch","Review privacy settings","Undated follow-up"][i],project_id:project.id,
    planned_date:i<8?"2030-01-"+String(i+3).padStart(2,"0"):null,due_date:i<8?"2030-01-"+String(i+5).padStart(2,"0"):null,priority:i%4,tags:i%2?["launch"]:["mobile"],work_type:"product"});
  await page.reload();await expect(page.getByRole("heading",{name:"Tasks",exact:true})).toBeVisible();
  const all=(await request("/tasks?limit=200")).items;
  const task=all.find(t=>t.title==="Review mobile layout");
  await ui("ui_filter",{view:"all",project_id:project.id,status:"active"});
  await ui("ui_workspace",{view:"all",layout:"board",group_by:"status"});
  await expect(page.locator(".board-card")).toHaveCount(9);
  await page.getByLabel("Status for Review mobile layout").selectOption("in_progress");
  await expect.poll(async()=>(await request("/tasks/"+task.id)).status).toBe("in_progress");
  await shot("work-board-desktop");
  await ui("ui_workspace",{view:"all",layout:"timeline",timeline_date:"2030-01-01",timeline_span:30});
  await expect(page.getByRole("region",{name:"Task timeline dates"})).toBeVisible();
  await expect(page.locator(".timeline-other")).toContainText("Undated follow-up");
  await shot("work-timeline-desktop");
  await ui("ui_workspace",{view:"organize",organization_tab:"project",layout:"board"});
  await expect(page.getByLabel("Project board")).toBeVisible();
  await page.getByLabel("Status for Home renovation").selectOption("active");
  await ui("ui_workspace",{view:"organize",layout:"timeline",timeline_date:"2030-01-01",timeline_span:90});
  await expect(page.locator(".timeline-bar")).toHaveCount(1);
  await shot("projects-timeline-desktop");
  // Draft reads, patches, invalid patches, save, dirty guard and explicit discard use actual CopilotKit handlers.
  await ui("ui_form",{form:"note"});
  let draft=await ui("ui_editor",{operation:"read"});
  if(!draft.data?.fields?.properties?.content)throw new Error("Missing runtime draft schema");
  const exact="  Research: Hāyes / Haze?\n\nKeep the final newline.\n";
  await ui("ui_editor",{operation:"patch",changes:{title:"Research note",content:exact,goal_ids:[goal.id],project_ids:[project.id],task_ids:[task.id]}});
  await ui("ui_show",{view:"calendar"},"failed");
  await ui("ui_editor",{operation:"close"},"failed");
  await ui("ui_editor",{operation:"patch",changes:{title:"Should not apply",unsupported_field:"bad"}},"failed");
  draft=await ui("ui_editor",{operation:"read"});if(draft.data.values.title!=="Research note")throw new Error("Invalid patch partially applied");
  let saved=await ui("ui_editor",{operation:"save"});
  const note=saved.data.result;if(!note?.id)throw new Error("Note save did not return a real result");
  if((await request("/notes/"+note.id)).content!==exact)throw new Error("Authored text changed");
  await ui("ui_editor",{operation:"patch",changes:{title:"Renamed research"}});
  await ui("ui_editor",{operation:"save"});
  const persisted=await request("/notes/"+note.id);
  if(persisted.content!==exact || !persisted.goals.some(g=>g.id===goal.id) || !persisted.tasks.some(t=>t.id===task.id))throw new Error("Sparse note save lost fields");
  await close();
  await ui("ui_form",{form:"task",entity_id:task.id});
  await ui("ui_editor",{operation:"patch",changes:{title:"Discarded title"}});
  await ui("ui_editor",{operation:"discard"});
  if((await request("/tasks/"+task.id)).title!=="Review mobile layout")throw new Error("Discard saved changes");
  // Concurrent server edits must not erase the user's draft.
  await ui("ui_form",{form:"task",entity_id:task.id});
  await ui("ui_editor",{operation:"patch",changes:{notes:"Keep this unsaved context"}});
  const current=await request("/tasks/"+task.id);
  await cmd("task.update",{task_id:task.id,expected_revision:current.revision,priority:2});
  await ui("ui_editor",{operation:"save"},"failed");
  draft=await ui("ui_editor",{operation:"read"});
  if(draft.data.values.notes!=="Keep this unsaved context" || !draft.data.dirty)throw new Error("Conflict erased the draft");
  await ui("ui_editor",{operation:"discard"});
  await ui("ui_workspace",{view:"today",layout:"list"});
  await page.getByLabel("New task",{exact:true}).fill("Quick capture today");
  await page.locator(".compact-capture button").click();
  await expect(page.getByRole("button",{name:"Edit Quick capture today",exact:true})).toBeVisible();
  // Every organization editor is openable/fillable; same domain validation as manual UI.
  for(const [form,changes] of [
    ["space",{name:"Learning"}],["area",{name:"Operations",space_id:business.id}],
    ["goal",{name:"Less admin",metric_target:10,metric_current:2}],["project",{name:"Planning controls",status:"planned",goal_ids:[goal.id]}],
    ["actor",{name:"Research assistant",kind:"agent"}],
  ]){await ui("ui_form",{form});await ui("ui_editor",{operation:"read"});await ui("ui_editor",{operation:"patch",changes});await ui("ui_editor",{operation:"save"});await expect(page.getByRole("dialog")).toHaveCount(0);}
  await ui("ui_form",{form:"reminder"});
  await ui("ui_editor",{operation:"patch",changes:{title:"Launch check",when:"2030-01-05T10:00",task_id:task.id}});
  await ui("ui_editor",{operation:"save"});
  await ui("ui_form",{form:"event"});
  await ui("ui_editor",{operation:"patch",changes:{title:"Focus session",start:"2030-01-05T11:00",end:"2030-01-05T12:00",kind:"block",task_id:task.id}});
  await ui("ui_editor",{operation:"save"});
  await ui("ui_select",{task_ids:all.slice(0,2).map(t=>t.id)});
  await ui("ui_form",{form:"bulk"});
  await ui("ui_editor",{operation:"patch",changes:{priority:"3"}});
  await ui("ui_editor",{operation:"save"});
  await expect(page.getByRole("dialog")).toHaveCount(0);
  // Device controls and every Settings section.
  await ui("ui_device",{density:"comfortable"});
  await expect(page.locator(".app-shell")).toHaveClass(/density-comfortable/);
  await ui("ui_device",{density:"compact"});
  await ui("ui_device",{voice:"not-a-voice"},"failed");
  for(const section of ["profile","voice","integrations","privacy","system"]){
    await ui("ui_workspace",{view:"settings",settings_section:section});
    await expect(page.getByRole("tab",{name:section,exact:false})).toHaveAttribute("aria-selected","true");
    await shot("settings-"+section);
  }
  // Mobile navigation keeps selected query/filter state while changing layouts, clears only where documented.
  await page.setViewportSize({width:390,height:844});
  await ui("ui_chat",{mode:"open"});
  await ui("ui_filter",{view:"all",project_id:project.id,status:"completed"});
  await expect(page.locator(".companion")).toBeHidden();
  await ui("ui_search",{view:"all",query:"Review"});
  await ui("ui_filter",{view:"all",project_id:project.id,status:"active",tag:"mobile"});
  let state=await ui("ui_workspace",{view:"all",layout:"board",group_by:"project"});
  if(state.screen.project!=="Website release" || state.screen.query!=="Review" || state.screen.tag!=="mobile")throw new Error("Layout lost filters");
  await shot("work-board-mobile");
  await ui("ui_search",{view:"all",query:""});
  state=await ui("ui_workspace",{view:"all",layout:"timeline",timeline_date:"2030-01-01",timeline_span:30});
  if(state.screen.project || state.screen.tag)throw new Error("Search did not reset filters");
  await shot("work-timeline-mobile");
  await ui("ui_workspace",{view:"organize",organization_tab:"project",layout:"board"});
  await shot("projects-board-mobile");
  await ui("ui_workspace",{view:"notes"});
  await shot("notes-mobile");
  await ui("ui_calendar",{date:"2030-01-05",calendar_view:"day"});
  await shot("calendar-mobile");
  const memory=await cmd("memory.capture",{content:"Planner fixture: I like paper notebooks."});
  await ui("ui_workspace",{view:"memory"});
  await ui("ui_form",{form:"memory",entity_id:memory.id});
  await ui("ui_editor",{operation:"patch",changes:{content:"Planner fixture: I prefer digital notebooks."}});
  await ui("ui_editor",{operation:"save"});
  await shot("memory-mobile");
  await ui("ui_workspace",{view:"notifications"});
  await shot("notifications-mobile");
  for(const view of ["today","inbox","week","all","organize","notes","settings"]){
    await ui("ui_show",{view});
    if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1))throw new Error("Page overflow: "+view);
  }
  if(errors.length)throw new Error(errors.join("; "));
  writeFileSync(new URL(".runtime/planner-browser-evidence.json",root),JSON.stringify({result:"passed",checks,errors},null,2));
  console.log("Planner real-browser acceptance passed: "+checks.length+" acknowledged actions; desktop/mobile boards, timelines, drafts, settings and overflow.");
}catch(e){await shot("failure");writeFileSync(new URL(".runtime/planner-browser-evidence.json",root),JSON.stringify({result:"failed",error:e.message,checks,errors},null,2));throw e;}
finally{await browser.close();}
