import {chromium,expect} from "@playwright/test";
import {mkdirSync} from "node:fs";
const browser=await chromium.launch();const page=await browser.newPage({viewport:{width:1440,height:960}});const errors=[];page.on("pageerror",e=>errors.push(e.message));
const base=process.env.JARVIS_PLANNER_TEST_URL;
async function request(path,body){return page.evaluate(async({path,body})=>{const boot=await(await fetch("/api/v1/bootstrap")).json();const r=await fetch("/api/v1"+path,{method:body?"POST":"GET",headers:{"Content-Type":"application/json","X-CSRF-Token":boot.csrf},...(body?{body:JSON.stringify(body)}:{})});const data=await r.json();if(!r.ok)throw Error(JSON.stringify(data));return data;},{path,body});}
const cmd=async(tool,args)=>(await request("/commands",{command_id:crypto.randomUUID(),tool,arguments:args})).data;
const ui=async(name,args)=>{const r=await request("/__test_ui",{name,arguments:args});if(r.status!=="displayed")throw Error(JSON.stringify(r));};
try{
 await page.goto(base);await page.getByLabel("Pairing code").fill("planner-fixture");await page.locator(".login-card button.primary").click();await expect(page.getByRole("region",{name:"Today",exact:true})).toBeVisible();
 let schema=await request("/structure");
 const client=schema.types.find(t=>t.id==="client");
 const abc=await cmd("record.create",{type_id:"client",title:"ABC Holdings",schema_revision:schema.revision});
 if(abc.review_every!==null||abc.review_due)throw Error("Reviews must start off: "+JSON.stringify(abc));
 // Turn on a monthly review cadence for the client type in the type editor, then preview and apply.
 await ui("ui_records",{layout:"browse",parent_id:"",type_id:""});
 await page.getByRole("button",{name:"Structure",exact:true}).click();
 const structure=page.getByRole("dialog",{name:"Workspace structure"});
 await structure.getByRole("navigation",{name:"Type and field tree"}).getByRole("button",{name:client.plural,exact:true}).click();
 const cadence=structure.getByRole("group",{name:"Review cadence"});
 await expect(cadence.getByLabel("Review interval")).toBeDisabled();
 await cadence.getByRole("switch",{name:"Review cadence"}).check();
 await cadence.getByLabel("Review interval").selectOption("1m");
 await structure.getByRole("button",{name:"Preview changes"}).click();
 await expect(structure.getByRole("heading",{name:"Review your changes"})).toBeVisible();
 await expect(structure.getByRole("list",{name:"Review cadence changes"})).toContainText(client.name+" records will resurface for review every month");
 await structure.getByRole("button",{name:"Apply structure",exact:true}).click();
 await expect(structure).toHaveCount(0);
 await expect.poll(async()=>(await request("/structure")).types.find(t=>t.id==="client").review.enabled).toBe(true);
 const seeded=await request("/structure/records/"+abc.id);
 if(!seeded.next_review_at||seeded.review_due||Date.parse(seeded.next_review_at)<=Date.now())throw Error("Enabling must seed a future first review: "+seeded.next_review_at);
 // Fast-forward: ABC's review is due, and the daily queue puts it in Questions.
 const queued=await request("/__test_record_reviews",{record_ids:[abc.id]});
 if(queued.queued<1)throw Error("The queue surfaced nothing: "+JSON.stringify(queued));
 await ui("ui_workspace",{view:"questions"});
 const questions=page.getByRole("region",{name:"Questions",exact:true});
 const item=questions.locator(".question-card").filter({hasText:"ABC Holdings"});
 await expect(item).toBeVisible();await expect(item).toContainText("Review due");await expect(item).toContainText("Every month");
 await expect(item.getByRole("button",{name:"Mark reviewed"})).toBeVisible();
 // Today shows the small panel while something is due.
 await ui("ui_workspace",{view:"today"});
 const today=page.getByRole("region",{name:"Today",exact:true});
 await expect(today.getByRole("region",{name:"Reviews due"})).toContainText("ABC Holdings");
 // The Atlas Reviews lens lights ABC and dims the rest; the Blueprint shows the review glyph.
 await ui("ui_records",{layout:"browse",parent_id:"",type_id:""});
 await page.getByRole("group",{name:"Layout"}).getByRole("button",{name:"Atlas",exact:true}).click();
 const atlas=page.getByRole("region",{name:"Atlas"});await expect(atlas).toBeVisible();
 await atlas.getByRole("group",{name:"Lens"}).getByRole("button",{name:"Reviews due"}).click();
 const mark=atlas.locator(`[data-id="${abc.id}"]`);
 await expect(mark).toHaveClass(/is-review-due/);await expect(mark).not.toHaveClass(/is-faint|is-dim/);
 await expect(mark).toHaveAttribute("aria-label",/review due/);
 await expect(mark.locator(".atlas-review-dot")).toHaveCount(1);
 await expect(atlas.locator(".atlas-legend")).toContainText("Review due");
 const others=await atlas.locator(".atlas-region:not(.is-review-due)").count();
 if(others&&!await atlas.locator(".atlas-region.is-faint").count())throw Error("The Reviews lens should dim regions without due reviews");
 await mark.click();
 const inspector=atlas.getByRole("complementary",{name:"Inspector"});
 await expect(inspector.getByText("Review due",{exact:true})).toBeVisible();
 await atlas.getByRole("group",{name:"Atlas view"}).getByRole("button",{name:"Both"}).click();
 await expect(atlas.locator(`.blueprint-node[data-type="client"] .blueprint-glyph.is-review`)).toHaveCount(1);
 await atlas.getByRole("group",{name:"Atlas view"}).getByRole("button",{name:"Map"}).click();
 // The detail card shows Last reviewed and Next review as separate values.
 await ui("ui_records",{record_id:abc.id,open_details:true});
 const card=page.getByRole("dialog",{name:client.name+" details"});
 const props=card.getByRole("complementary",{name:client.name+" properties"});
 await expect(props.locator(".prop-row").filter({hasText:"Last reviewed"})).toContainText("Not yet");
 await expect(props.locator(".prop-row").filter({hasText:"Next review"})).toContainText(/Due since|Due now/);
 await card.getByRole("button",{name:"Close record"}).click();
 // Mark it reviewed from Questions: the item leaves and the next review is a month out.
 await ui("ui_workspace",{view:"questions"});
 await item.getByRole("button",{name:"Mark reviewed"}).click();
 await expect(item).toHaveCount(0);
 const reviewed=await request("/structure/records/"+abc.id);
 const days=(Date.parse(reviewed.next_review_at)-Date.parse(reviewed.last_reviewed_at))/864e5;
 if(reviewed.review_due||days<28||days>31.5)throw Error("Mark reviewed must schedule a month out: "+JSON.stringify([reviewed.last_reviewed_at,reviewed.next_review_at]));
 if((await request("/structure/reviews")).items.some(i=>i.id===abc.id))throw Error("ABC is still listed as due");
 await ui("ui_records",{record_id:abc.id,open_details:true});
 await expect(props.locator(".prop-row").filter({hasText:"Last reviewed"})).toContainText("Today");
 await expect(props.locator(".prop-row").filter({hasText:"Next review"})).not.toContainText(/Due/);
 await card.getByRole("button",{name:"Close record"}).click();
 await ui("ui_workspace",{view:"today"});
 await expect(today.getByRole("region",{name:"Reviews due"})).toHaveCount(0);
 if(errors.length)throw Error(errors.join("\n"));
 console.log("Organization reviews acceptance passed: type editor cadence preview/apply, seeded first review, queued Questions item, Today panel, Atlas Reviews lens and glyph, detail card spans, mark reviewed.");
}catch(e){console.error("Page errors:",errors.join("\n"));mkdirSync("../../artifacts/organization",{recursive:true});await page.screenshot({path:"../../artifacts/organization/reviews-failure.png"}).catch(()=>{});throw e;}
finally{await browser.close();}
