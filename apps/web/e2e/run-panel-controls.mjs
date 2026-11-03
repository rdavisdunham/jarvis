// Run against `npm run dev`: ERIDANI_PANEL_TEST_URL=http://127.0.0.1:5173 node e2e/run-panel-controls.mjs
// Uses the real provider, drawer and production shell styles with synthetic API responses.
import { chromium, expect } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
const base = process.env.ERIDANI_PANEL_TEST_URL;
if (!base || !/^http:\/\/(127\.0\.0\.1|localhost):\d+$/.test(base)) throw Error("Set ERIDANI_PANEL_TEST_URL to the local Vite server");
const artifacts = resolve(process.env.ERIDANI_EVAL_ARTIFACT_DIR || "../../artifacts/run-panel-controls");
mkdirSync(artifacts, { recursive: true });
const browser = await chromium.launch();
const evidence = [], errors = [], writes = [];
const run = { id: "11111111-1111-4111-8111-111111111111", task: { id: "task", title: "Implement the exceptionally long task title that must never push close or collapse controls beyond the viewport", status: "in_progress" },
  state: "executing", stage: "executing", state_revision: 3, state_reason: "", reason_text: "", generation: 1, plan_revision: 1, tool: "codex", model: "gpt-6.1-sol", surf: "terminal", turn_count: 1, turn_cap: 0, direct_supported: true,
  created_at: "2026-11-03T10:00:00Z", history: [], turns: [], commands: [], questions: [], automation: { held: false, revision: 0, held_at: null, reason: null, releasing: false },
  plan: { revision: 1, title: "Measured widget improvement", approval_status: "approved", prd_text: "The approved plan remains available.", criteria: [], steps: [{ id: "one", title: "Investigate widgets", status: "done" }, { id: "two", title: "Apply the focused change", status: "doing", note: "Preserve existing behavior" }] } };
try {
  for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 420 }, { width: 720, height: 450 }, { width: 375, height: 812 }, { width: 320, height: 568 }, { width: 320, height: 360 }]) {
    const context = await browser.newContext({ viewport });
    const page = await context.newPage();
    page.on("pageerror", error => errors.push(error.message));
    const drafts = new Map();
    await page.route("**/api/**", async route => {
      const draft = route.request().url().includes("/drafts/");
      if (!["GET", "HEAD"].includes(route.request().method()) && !draft) writes.push(route.request().url());
      if (draft && route.request().method() === "PUT") drafts.set(route.request().url(), route.request().postDataJSON().value);
      await route.fulfill({ json: draft ? { value: drafts.get(route.request().url()) ?? null, revision: 1 } : run });
    });
    await page.goto(`${base}/e2e/fixtures/run-panel.html`);
    const opener = page.getByRole("button", { name: "Open run", exact: true });
    await opener.click();
    const panel = page.locator(".run-panel"), drawer = page.locator(".run-view"), close = page.getByRole("button", { name: "Close run", exact: true });
    const collapse = page.getByRole("button", { name: "Collapse run panel", exact: true }), input = page.getByLabel("Message the agent", { exact: true });
    async function layout(mode) {
      await expect(close).toBeVisible(); await expect(collapse).toBeVisible();
      const bounds = await panel.evaluate(node => {
        const box = el => { const r = el.getBoundingClientRect(); return { x: r.x, y: r.y, right: r.right, bottom: r.bottom, width: r.width, height: r.height }; };
        const close = node.querySelector('[aria-label="Close run"]'), collapse = node.querySelector(".run-panel-collapse");
        const hit = el => { const r = el.getBoundingClientRect(); return el.contains(document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2)); };
        return { viewport: { width: innerWidth, height: visualViewport.height, top: visualViewport.offsetTop }, panel: box(node), close: box(close), collapse: box(collapse), title: box(node.querySelector("h2")), middle: box(node.querySelector(".run-drawer-middle")), footer: box(node.querySelector("footer")), hitClose: hit(close), hitCollapse: hit(collapse), scrollWidth: document.documentElement.scrollWidth };
      });
      for (const control of [bounds.close, bounds.collapse]) {
        expect(control.x).toBeGreaterThanOrEqual(0); expect(control.y).toBeGreaterThanOrEqual(bounds.viewport.top);
        expect(control.right).toBeLessThanOrEqual(viewport.width); expect(control.bottom).toBeLessThanOrEqual(bounds.viewport.height + bounds.viewport.top);
        expect(control.width).toBeGreaterThanOrEqual(44); expect(control.height).toBeGreaterThanOrEqual(44);
      }
      expect(bounds.hitClose).toBe(true); expect(bounds.hitCollapse).toBe(true);
      expect(bounds.title.right).toBeLessThanOrEqual(bounds.close.x); expect(bounds.close.right).toBeGreaterThanOrEqual(bounds.panel.right - 24);
      expect(bounds.middle.height).toBeGreaterThan(25); expect(bounds.footer.bottom).toBeLessThanOrEqual(bounds.panel.bottom + 1);
      expect(bounds.scrollWidth).toBeLessThanOrEqual(viewport.width);
      evidence.push({ viewport, mode, bounds });
      await page.screenshot({ path: resolve(artifacts, `${viewport.width}x${viewport.height}-${mode}.png`) });
    }
    await expect(drawer).toHaveAttribute("data-run-id", run.id); await layout("plan");
    await page.getByRole("button", { name: "Show chat", exact: true }).click();
    await input.fill("Keep this unsent instruction"); await page.getByText("Details", { exact: true }).click(); await layout("chat");
    if (viewport.width <= 700) {
      await collapse.focus(); await page.keyboard.press("Shift+Tab"); await expect(page.getByRole("button", { name: "Send", exact: true })).toBeFocused();
      await page.keyboard.press("Tab"); await expect(collapse).toBeFocused();
    }
    await collapse.click();
    const expand = page.getByRole("button", { name: /^Expand run panel/ });
    await expect(expand).toBeFocused(); await expect(expand).toBeVisible();
    await page.screenshot({ path: resolve(artifacts, `${viewport.width}x${viewport.height}-collapsed.png`) });
    await expand.click(); await expect(close).toBeFocused(); await expect(input).toHaveValue("Keep this unsent instruction");
    await expect(page.locator(".run-drawer-details")).toHaveAttribute("open", "");
    if (viewport.width <= 700) {
      await page.keyboard.press("Escape"); await expect(expand).toBeFocused(); await expand.click();
      await page.evaluate(() => { Object.defineProperty(visualViewport, "height", { configurable: true, value: 300 }); Object.defineProperty(visualViewport, "offsetTop", { configurable: true, value: 20 }); visualViewport.dispatchEvent(new Event("resize")); });
      await expect.poll(() => panel.evaluate(node => Math.round(node.getBoundingClientRect().bottom))).toBe(320); await layout("keyboard");
    }
    await close.click(); await expect(panel).toHaveCount(0); await expect(opener).toBeFocused();
    await opener.click(); await expect(input).toHaveValue("Keep this unsent instruction"); await expect(drawer).toHaveAttribute("data-state-revision", "3");
    await context.close();
  }
  // A delayed lazy chunk and a failed detail request must leave both controls usable.
  const context = await browser.newContext({ viewport: { width: 375, height: 812 } });
  const page = await context.newPage();
  let releaseChunk;
  const chunkGate = new Promise(resolve => { releaseChunk = resolve; });
  await page.route("**/src/Remotea.tsx*", async route => { await chunkGate; await route.continue(); });
  page.on("pageerror", error => errors.push(error.message));
  await page.route("**/api/**", route => {
    if (!["GET", "HEAD"].includes(route.request().method()) && !route.request().url().includes("/drafts/")) writes.push(route.request().url());
    return route.fulfill({ status: 503, json: { message: "Run temporarily unavailable" } });
  });
  await page.goto(`${base}/e2e/fixtures/run-panel.html`);
  const opener = page.getByRole("button", { name: "Open run", exact: true });
  await opener.click();
  const close = page.getByRole("button", { name: "Close run", exact: true });
  const collapse = page.getByRole("button", { name: "Collapse run panel", exact: true });
  await expect(page.getByRole("status")).toHaveText("Loading run…");
  await expect(close).toBeVisible(); await expect(collapse).toBeVisible();
  await expect(page.locator(".run-panel")).toBeFocused();
  await page.screenshot({ path: resolve(artifacts, "375-loading.png") });
  evidence.push({ scenario: "lazy-loading controls and retained focus", passed: true });
  await collapse.click();
  const expand = page.getByRole("button", { name: /^Expand run panel/ });
  await expect(expand).toBeFocused(); await expand.click();
  releaseChunk();
  await expect(page.getByRole("alert")).toBeVisible();
  await expect(close).toBeVisible(); await expect(collapse).toBeVisible();
  await expect(page.locator(".run-panel")).toBeFocused();
  await page.screenshot({ path: resolve(artifacts, "375-error.png") });
  evidence.push({ scenario: "API error controls and original-opener focus restoration", passed: true });
  await close.click(); await expect(opener).toBeFocused();
  await context.close();
  expect(writes).toEqual([]); expect(errors).toEqual([]);
  writeFileSync(resolve(artifacts, "result.json"), JSON.stringify({ passed: true, evidence, writes, errors }, null, 2));
  console.log(`Passed ${evidence.length} browser layout/loading/error states; no run API mutations or browser errors.`);
} catch (error) {
  writeFileSync(resolve(artifacts, "result.json"), JSON.stringify({ passed: false, evidence, writes, errors, error: String(error) }, null, 2)); throw error;
} finally { await browser.close(); }
