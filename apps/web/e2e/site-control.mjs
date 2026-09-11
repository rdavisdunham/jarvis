import { chromium, expect } from "@playwright/test";
import { readFileSync, writeFileSync } from "node:fs";
const root = new URL("../../../", import.meta.url);
const token = readFileSync(new URL(".runtime/pairing-code", root), "utf8").trim();
const browser = await chromium.launch();
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
const evidence = {result: "failed", errors: [], uiActions: []};
page.on("response", async r => { if(r.url().endsWith("/ui/sync")) { try {
  const v=await r.json(); evidence.uiActions.push(...v.actions.map(a => ({kind:a.kind, view:a.view, mode:a.mode})));
} catch {} }});
page.on("pageerror", e => evidence.errors.push(e.message));
const open = async () => { if (!await page.locator(".companion").isVisible()) await page.getByRole("button", {name: "Open Eridani", exact: true}).click(); };
const say = async text => { await open(); await page.getByLabel("Message Eridani").fill(text); await page.getByRole("button", {name: "Send message", exact: true}).click(); };
try {
  await page.goto("https://davispc.tail957c2.ts.net:9443");
  await page.getByLabel("Pairing code").fill(token);
  await page.locator(".login-card button.primary").click();
  await expect(page.getByLabel("New task")).toBeVisible();
  await expect(page.locator(".companion")).toBeHidden();
  await open();
  await page.getByRole("button", {name: "Start a private session", exact: true}).click();
  await say("Close this chat panel, please.");
  await expect(page.locator(".companion")).toBeHidden({timeout: 45000});
  await expect(page.locator(".thinking")).toHaveCount(0, {timeout: 45000});
  evidence.closeChat = true;
  await say("Open All tasks and set the project filter to Eridani roadmap and the status filter to open.");
  await expect(page.getByLabel("Project filter")).toHaveValue("Eridani roadmap", {timeout: 45000});
  await expect(page.getByLabel("Task status filter")).toHaveValue("open");
  await expect(page.locator(".thinking")).toHaveCount(0, {timeout: 45000});
  evidence.filters = true;
  await say("Search my tasks for preferred name and display the results.");
  await expect(page.locator(".search input")).toHaveValue(/preferred name/i, {timeout: 45000});
  await expect(page.locator(".thinking")).toHaveCount(0, {timeout: 45000});
  evidence.search = true;
  await page.setViewportSize({width: 390, height: 844});
  await say("Show me the Memory page.");
  await expect(page.locator(".learning-status")).toBeVisible({timeout: 45000});
  await expect(page.locator(".companion")).toBeHidden();
  evidence.mobileSmartClose = true;
  await expect.poll(async () => page.locator(".memory:not(.legacy)").count(), {timeout: 60000}).toBeGreaterThan(0);
  evidence.learnedMemoriesVisible = await page.locator(".memory:not(.legacy)").count();
  evidence.mobileOverflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
  await page.screenshot({path: new URL(".runtime/memory-mobile.png", root).pathname, fullPage: true});
  await open();
  await page.getByRole("button", {name: "Voice settings", exact: true}).click();
  await expect(page.getByLabel("Preferred name")).toBeVisible();
  evidence.editableName = true;
  if(evidence.errors.length || evidence.mobileOverflow) throw new Error("UI errors or mobile overflow");
  evidence.result = "passed";
} catch (error) { evidence.browserState = await page.evaluate(() => ({
  width: innerWidth, companion: document.querySelector(".companion")?.className,
  mobile: matchMedia("(max-width: 1000px)").matches
})); await page.screenshot({path: new URL(".runtime/site-control-failure.png", root).pathname}); evidence.failure = error.message.split("\n")[0]; }
finally { await browser.close(); writeFileSync(new URL(".runtime/site-control-evidence.json", root), JSON.stringify(evidence,null,2)); console.log(JSON.stringify(evidence,null,2)); }
if(evidence.result !== "passed") process.exitCode = 1;
