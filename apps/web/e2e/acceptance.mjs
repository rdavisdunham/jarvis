import { chromium, expect } from '@playwright/test';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
const base = process.env.JARVIS_TEST_ORIGIN || 'https://davispc.tail957c2.ts.net:9443';
const token = readFileSync('../../.runtime/pairing-code','utf8').trim();
const browser = await chromium.launch({headless:true});
const context = await browser.newContext({viewport:{width:1440,height:1000}});
const page = await context.newPage();
const errors=[];
page.on('pageerror', e=>errors.push(e.message));
const reminderTitle='Acceptance check — reminder '+Date.now();
const title='Acceptance check — task recovery '+Date.now();
await page.goto(base);
await page.getByLabel('Pairing code').fill(token);
await page.locator('.login-card button.primary').click();
await expect(page.getByRole('heading',{name:/Good .*Davin/})).toBeVisible();
await page.locator('nav button').filter({hasText:'Inbox'}).click();
await page.getByLabel('New task').fill(title);
await page.getByRole('button',{name:'Add task',exact:false}).click();
await expect(page.locator('.task-title').filter({hasText:title})).toBeVisible();
await page.getByRole('button',{name:'Edit '+title,exact:true}).click();
await page.getByLabel('Notes',{exact:true}).fill('Synthetic acceptance record. Safe to archive after validation.');
const today=new Intl.DateTimeFormat('en-CA',{timeZone:'America/Chicago',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date());
await page.getByLabel('Due date',{exact:true}).fill(today);
await page.locator('form.dialog select').first().selectOption('2');
await page.getByRole('button',{name:'Save task',exact:true}).click();
await expect(page.getByRole('dialog')).toHaveCount(0);
await page.getByRole('navigation').getByRole('button',{name:'Today',exact:true}).click();
await expect(page.locator('.task-title').filter({hasText:title})).toBeVisible();
await page.getByRole('button',{name:'Complete '+title,exact:true}).click();
await page.locator('details.completed summary').click();
await page.getByRole('button',{name:'Reopen '+title,exact:true}).click();
await expect(page.getByRole('button',{name:'Complete '+title,exact:true})).toBeVisible();
await page.reload();
await expect(page.getByRole('button',{name:'Complete '+title,exact:true})).toBeVisible();
mkdirSync('../../.runtime/screenshots',{recursive:true});
await page.screenshot({path:'../../.runtime/screenshots/desktop.png',fullPage:true,animations:"disabled"});
await page.getByRole('button',{name:'Set a reminder',exact:true}).click();
await page.getByLabel('Remind me to',{exact:true}).fill(reminderTitle);
await page.locator('input[type="datetime-local"]').fill(today+'T23:59');
await page.locator('form.dialog button.primary').click();
await expect(page.getByRole('dialog')).toHaveCount(0);
await page.getByRole('button',{name:'Reminders',exact:true}).click();
await expect(page.getByText(reminderTitle,{exact:true})).toBeVisible();
const data=await page.evaluate(async()=>{
 const boot=await (await fetch('/api/v1/bootstrap')).json();
 const tasks=await (await fetch('/api/v1/tasks')).json();
 const schedules=await (await fetch('/api/v1/schedules')).json();
 return {boot,tasks,schedules};
});
await page.setViewportSize({width:390,height:844});
await page.getByRole('button',{name:'Open navigation'}).click();
await page.getByRole('navigation').getByRole('button',{name:'Today',exact:true}).click();
await expect(page.getByLabel('New task')).toBeVisible();
expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
await page.screenshot({path:'../../.runtime/screenshots/mobile.png',fullPage:true,animations:"disabled"});
await page.getByRole('button',{name:'Edit '+title,exact:true}).click();
await expect(page.getByRole('button',{name:'Save task',exact:true})).toBeVisible();
await page.screenshot({path:'../../.runtime/screenshots/mobile-dialog.png',fullPage:true,animations:"disabled"});
await page.getByRole('button',{name:'Close',exact:true}).click();
await page.getByRole('button',{name:'Open Eridani',exact:true}).click();
await expect(page.getByRole('complementary',{name:'Eridani conversation'})).toBeVisible();
await page.screenshot({path:'../../.runtime/screenshots/mobile-chat.png',fullPage:true,animations:"disabled"});
if (process.env.JARVIS_E2E_CHAT === '1') {
  await page.getByLabel('Message Eridani').fill('Synthetic connection test: reply with exactly READY. Do not use tools or save a memory.');
  await page.getByRole('button',{name:'Send message',exact:true}).click();
  await expect(page.locator('.message.assistant').last()).toContainText('READY',{timeout:60000});
  await page.reload();
  await page.getByRole('button',{name:'Open Eridani',exact:true}).click();
  await expect(page.locator('.message.assistant').last()).toContainText('READY');
}
expect(errors).toEqual([]);
writeFileSync('../../.runtime/browser-evidence.json', JSON.stringify({
 origin:base, result:'passed', task_id:data.tasks.items.find(t=>t.title===title).id,
 schedule_id:data.schedules.items.find(s=>s.title===reminderTitle).id,
 viewports:['1440x1000','390x844'],pageErrors:errors,
 capabilities:data.boot.capabilities
},null,2));
console.log('Browser acceptance passed: pairing, task create/edit/complete/reopen, reload, reminder creation, desktop and mobile layout.');
await browser.close();
