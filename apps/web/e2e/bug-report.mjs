// Report a bug in a real browser under the production CSP: the button shows on desktop and phone,
// the screenshot is captured before the dialog opens, and the report lands as a tagged task whose card
// shows the screenshot. Manual fixture: JARVIS_PLANNER_TEST_URL=<server> node e2e/bug-report.mjs
import {chromium,expect} from "@playwright/test";
const base=process.env.JARVIS_PLANNER_TEST_URL;
const browser=await chromium.launch();
const page=await browser.newPage({viewport:{width:1280,height:800}});
const errors=[];page.on("pageerror",error=>errors.push(error.message));
page.on("console",message=>{if(message.type()==="error"&&/Content Security Policy/.test(message.text()))errors.push(message.text());});
const button=page.getByRole("button",{name:"Report a bug",exact:true});
try{
 await page.goto(base);await page.getByLabel("Pairing code").fill("planner-fixture");await page.locator(".login-card button.primary").click();
 await expect(page.getByRole("region",{name:"Today",exact:true})).toBeVisible();
 await expect(button).toBeVisible();
 const desk=await button.boundingBox(),launcher=await page.locator(".chat-launcher").boundingBox();
 expect(desk.x).toBeGreaterThan(232);expect(desk.x+desk.width).toBeLessThan(launcher.x);
 expect(await button.evaluate(el=>getComputedStyle(el).backgroundColor)).toBe("rgb(217, 45, 32)");
 await page.evaluate(()=>console.error("fixture: something broke"));
 await page.evaluate(()=>fetch("/api/v1/tasks/not-a-task?secret=1"));
 await button.click();
 const dialog=page.getByRole("dialog",{name:"Report a bug"});await expect(dialog).toBeVisible({timeout:15000});
 await expect(button).toHaveCount(0);
 const shot=dialog.locator("img");await expect(shot).toBeVisible();
 const size=await shot.evaluate(img=>({w:img.naturalWidth,h:img.naturalHeight,src:img.src.slice(0,23)}));
 expect(size.src).toBe("data:image/jpeg;base64,");expect(size.w).toBeGreaterThan(600);expect(size.w).toBeLessThanOrEqual(1600);
 await dialog.getByText("Details that will be attached").click();await expect(dialog.getByText("App build")).toBeVisible();
 await dialog.getByLabel("What went wrong?").fill("Fixture bug: the page looks odd\nsecond line");
 await dialog.getByRole("button",{name:"Send",exact:true}).click();
 await expect(page.getByRole("status").filter({hasText:"Reported. Thanks"})).toBeVisible();
 const reports=await page.evaluate(async()=>(await (await fetch("/api/v1/bug-reports")).json()).items);
 expect(reports.length).toBeGreaterThan(0);expect(reports[0].title).toBe("Fixture bug: the page looks odd");expect(reports[0].screenshot.bytes).toBeGreaterThan(5000);
 const detail=await page.evaluate(async id=>(await fetch("/api/v1/bug-reports/"+id)).json(),reports[0].id);
 expect(detail.context.errors.some(e=>e.message.includes("fixture: something broke"))).toBeTruthy();
 expect(JSON.stringify(detail.context)).not.toContain("secret=1");
 await page.getByRole("button",{name:"Open task",exact:true}).click();
 await expect(page.getByRole("region",{name:"Bug report"}).locator("img")).toBeVisible();
 const card=await page.locator(".task-detail-card").boundingBox(),corner=await button.boundingBox();
 expect(corner.x+corner.width).toBeLessThan(card.x);expect(corner.y+corner.height).toBeGreaterThan(760);
 await page.screenshot({path:process.env.BUG_REPORT_ARTIFACT||"/tmp/bug-report-task.png"});
 await page.keyboard.press("Escape");
 await page.setViewportSize({width:390,height:844});
 await expect(button).toBeVisible();
 const phone=await button.boundingBox(),tabbar=await page.locator(".tabbar").boundingBox(),eri=await page.locator(".chat-launcher").boundingBox();
 expect(phone.x).toBeLessThan(40);expect(phone.y+phone.height).toBeLessThanOrEqual(tabbar.y);expect(phone.x+phone.width).toBeLessThan(eri.x);
 await page.keyboard.press("Alt+Shift+B");await expect(page.getByRole("dialog",{name:"Report a bug"})).toBeVisible({timeout:15000});
 await page.getByRole("button",{name:"Cancel",exact:true}).click();await expect(button).toBeVisible();
 if(errors.length)throw Error(errors.join("\n"));
 console.log("Report a bug passed: desktop and phone placement, CSP-safe capture, report + task + inline screenshot, shortcut.");
} finally {await browser.close();}
