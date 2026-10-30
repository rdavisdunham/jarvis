import { chromium, expect } from "@playwright/test";
const browser = await chromium.launch();
const page = await browser.newPage({viewport:{width:1440,height:1000}});
const errors=[];page.on("pageerror",e=>errors.push(e.message));
const base=process.env.JARVIS_GOOGLE_TEST_URL;
const request=(path,body={})=>page.evaluate(async({path,body})=>{
 const boot=await(await fetch("/api/v1/bootstrap")).json();
 const response=await fetch("/api/v1"+path,{method:"POST",headers:{"Content-Type":"application/json","X-CSRF-Token":boot.csrf},body:JSON.stringify(body)});
 const result=await response.json();if(!response.ok)throw Error(JSON.stringify(result));return result;
},{path,body});
try {
 await page.goto(base);
 await page.getByLabel("Pairing code").fill("google-fixture");
 await page.locator(".login-card button.primary").click();
 await expect(page.getByRole("region",{name:"Today",exact:true})).toBeVisible();
 await request("/__test_google_invitation");
 await request("/__test_ui",{name:"ui_calendar",arguments:{date:"2026-09-18"}});
 await page.locator(".agenda-row").filter({hasText:"Dentist"}).click();
 await expect(page.getByRole("group",{name:"RSVP"})).toBeVisible();
 await expect(page.getByRole("button",{name:"Edit event",exact:true})).toBeDisabled();
 for (const [width,response] of [[1440,"Yes"],[390,"Maybe"]]) {
   await page.setViewportSize({width,height:1000});
   await page.getByRole("group",{name:"RSVP"}).getByRole("button",{name:response,exact:true}).click();
   await expect(page.getByText("Your response: "+response,{exact:true})).toBeVisible();
   await expect(page.getByRole("button",{name:"Revert response",exact:true})).toBeEnabled();
   await page.getByRole("button",{name:"Revert response",exact:true}).click();
   await expect(page.getByText("Your response: Not answered",{exact:true})).toBeVisible();
   expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBeTruthy();
 }
 await page.getByRole("button",{name:"Close calendar event",exact:true}).click();
 // The same answer from the calendar row: declined stays visible but dimmed, and Revert restores it.
 const row=page.getByRole("group",{name:"Your response to Dentist",exact:true});
 await row.getByRole("button",{name:"No",exact:true}).click();
 await expect(row.getByText("Saved in Google.",{exact:true})).toBeVisible();
 await expect(page.locator(".agenda-row.is-declined").filter({hasText:"Dentist"})).toBeVisible();
 expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBeTruthy();
 await row.getByRole("button",{name:"Revert",exact:true}).click();
 await expect(page.locator(".agenda-row.is-declined")).toHaveCount(0);
 await expect(row.getByText("Not answered",{exact:true})).toBeVisible();
 if(errors.length)throw Error(errors.join("\n"));
 console.log("RSVP browser acceptance passed: self response, read-only organizer fields, confirmed feedback and Revert at desktop/phone widths, in the dialog and on the calendar row.");
} finally {await browser.close();}
