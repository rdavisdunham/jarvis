// JARVIS_LINEAR_TEST_URL=http://127.0.0.1:5179 node e2e/linear-log.mjs (Vite)
import { chromium, expect } from "@playwright/test";
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 390, height: 844 }, timezoneId: "America/Chicago" });
const errors = [], calls = [];
page.on("pageerror", error => errors.push(error.message));
const longTitle = "長い issue title " + "UnbrokenTitle".repeat(60);
const change = {
  job_id: "one", status: "queued", message: "Waiting to save", created_at: "2026-11-04T15:04:05+00:00",
  finished_at: null, task_id: "outside-loaded-list", issue_identifier: "ERI-42", issue_title: longTitle,
  payload: { base: { description: "first line\nsecond line 雪" }, patch: { title: longTitle },
    nested: [null, {}, { text: "<img src=x onerror=alert(1)>" + "x".repeat(3000) }] }, result: null,
};
let recent_changes = [change, { ...change, job_id: "two", task_id: null, status: "succeeded", result: {} }];
await page.route("**/api/v1/integrations/linear", route => {
  calls.push(route.request().method());
  return route.fulfill({ json: { connected: true, workspace: "Workspace", revision: 1, team_ids: [],
    only_mine: true, status: "ready", error: "", last_sync_at: null, teams: [], states: [], users: [], projects: [], recent_changes } });
});
// Render the real component/styles with fixtures; no live Linear connection required.
await page.route("**/__linear-log-test", route => route.fulfill({ contentType: "text/html", body: `<!doctype html>
<html><head><meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="/src/theme.css"><link rel="stylesheet" href="/src/style.css">
<link rel="stylesheet" href="/src/settings.css"></head><body><main class="settings-container"><div class="settings-content" id="root"></div></main>
<script type="module">
import RefreshRuntime from '/@react-refresh';
RefreshRuntime.injectIntoGlobalHook(window);
window.$RefreshReg$ = () => {}; window.$RefreshSig$ = () => type => type;
window.__vite_plugin_react_preamble_installed__ = true;
const React = (await import('/node_modules/.vite/deps/react.js')).default;
const { createRoot } = (await import('/node_modules/.vite/deps/react-dom_client.js')).default;
const { LinearSettings } = await import('/src/Linear.tsx');
const root = createRoot(document.getElementById('root'));
window.renderLog = (revision, key = 'settings') => root.render(React.createElement(LinearSettings,
  { key, revision, mutate: async () => true, onOpenTask: id => window.openedTask = id }));
window.renderLog(1);
</script></body></html>` }));
try {
  await page.goto((process.env.JARVIS_LINEAR_TEST_URL ?? "http://127.0.0.1:5173") + "/__linear-log-test");
  const outer = page.locator(".linear-log"), entries = page.locator(".linear-log-details");
  const title = page.locator(".linear-log-entry .settings-item-title").first();
  await expect(outer).toBeVisible(); await expect(title).toBeHidden();
  await expect(page.locator(".linear-log-json").first()).toBeHidden();
  await outer.locator(":scope > summary").focus(); await page.keyboard.press("Enter");
  await expect(title).toBeVisible(); await expect(entries.first().locator("pre").first()).toBeHidden();
  await entries.first().locator("summary").focus(); await page.keyboard.press("Space");
  await expect(entries.first().locator("pre")).toBeVisible();
  await expect(entries.nth(1).locator("pre").first()).toBeHidden();
  await expect(entries.first()).toContainText("No outcome recorded yet");
  await expect(page.locator(".linear-log time").first()).toContainText("CST");
  expect(await entries.first().locator("pre code").textContent()).toBe(JSON.stringify(change.payload, null, 2));
  expect(await page.locator(".linear-log img").count()).toBe(0);
  await entries.nth(1).locator("summary").click();
  await expect(entries.nth(1).locator("pre code").last()).toHaveText("{}");
  await page.getByRole("button", { name: "Open task", exact: true }).click();
  expect(await page.evaluate(() => window.openedTask)).toBe("outside-loaded-list");
  await expect(entries.first()).toHaveAttribute("open", "");
  if (process.env.JARVIS_LINEAR_SCREENSHOT) await page.screenshot({ path: process.env.JARVIS_LINEAR_SCREENSHOT, fullPage: true });
  for (const width of [390, 320, 1440]) {
    await page.setViewportSize({ width, height: 844 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
    for (const pre of await page.locator(".linear-log-json").all()) {
      const box = await pre.boundingBox();
      expect(box.x).toBeGreaterThanOrEqual(0); expect(box.x + box.width).toBeLessThanOrEqual(width);
    }
  }
  await outer.locator(":scope > summary").click();
  await expect(title).toBeHidden(); await expect(entries.first().locator("pre")).toBeHidden();
  recent_changes = [{ ...change, job_id: "new" }, ...recent_changes];
  await page.evaluate(() => window.renderLog(2)); await expect(entries).toHaveCount(3);
  await expect(title).toBeHidden(); await outer.locator(":scope > summary").click();
  await expect(entries.nth(1)).toHaveAttribute("open", "");
  await expect(entries.first()).not.toHaveAttribute("open", "");
  await page.evaluate(() => window.renderLog(3, "remount")); await expect(title).toBeHidden();
  await expect.poll(() => calls).toEqual(["GET", "GET", "GET"]);
  await expect(outer).toBeVisible(); await expect(title).toBeHidden();
  expect(errors).toEqual([]);
  console.log("Linear logs: keyboard disclosures, stored JSON, timezone, task callback, refresh stability and 320/390/1440px containment passed.");
} catch (error) { console.error(errors); throw error; } finally { await browser.close(); }
