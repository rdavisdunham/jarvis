import { chromium, expect } from "@playwright/test";
import { mkdirSync } from "node:fs";
const base = process.env.JARVIS_PLANNER_TEST_URL;
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 820, height: 1000 } });
const errors = [];
page.on("pageerror", e => errors.push(e.message));
try {
  await page.route("**/api/v1/bootstrap", async route => {
    const response = await route.fetch();
    if (!response.ok()) return route.fulfill({ response });
    const data = await response.json();
    data.budget = { tracking_enabled: true, enforcement_enabled: false, budget_mode: "tracking_only",
      spent_usd: 1.026, uncertain_usd: 0, active_reserved_usd: 0, reserved_usd: 0,
      limit_usd: 150, remaining_usd: 148.974, projected_month_usd: null, usage_by_model: {},
      report: { as_of: "2026-09-19T12:00:00Z", tracking_since: "2026-09-19T06:00:00Z",
        partial_7_days: true, partial_30_days: true, last_7_days_usd: 1.026, last_30_days_usd: 1.026,
        features: [
          {id: "voice", label: "Voice conversations", last_7_days_usd: 1, last_30_days_usd: 1, usage_records: 3},
          {id: "assistant", label: "Assistant tasks and chat", last_7_days_usd: .025, last_30_days_usd: .025, usage_records: 4},
          {id: "note_organization", label: "Note organization", last_7_days_usd: .001, last_30_days_usd: .001, usage_records: 2}
        ] } };
    await route.fulfill({ response, json: data });
  });
  await page.goto(base);
  await page.getByLabel("Pairing code").fill("planner-fixture");
  await page.locator(".login-card button.primary").click();
  await expect(page.getByRole("heading", { name: "Tasks", exact: true })).toBeVisible();
  await page.evaluate(async () => {
    const b = await (await fetch("/api/v1/bootstrap")).json();
    const r = await fetch("/api/v1/__test_ui", {method: "POST", headers: {"Content-Type": "application/json", "X-CSRF-Token": b.csrf},
      body: JSON.stringify({name: "ui_workspace", arguments: {view: "settings", settings_section: "system"}})});
    if (!r.ok) throw Error("Settings navigation failed");
  });
  await expect(page.getByRole("heading", {name: "Model usage", exact: true})).toBeVisible();
  await expect(page.getByText("Recorded AI cost by feature", {exact: true})).toBeVisible();
  await expect(page.getByText("Partial history", {exact: true})).toHaveCount(2);
  await expect(page.getByRole("button", {name: "Save limit", exact: true})).toHaveCount(0);
  await expect(page.getByText("$0.0010", {exact: true})).toHaveCount(2);
  await expect(page.getByText("At this month's pace:", {exact: false})).toHaveCount(0);
  mkdirSync("../../artifacts/usage-report", {recursive: true});
  for (const [name, width] of [["phone", 390], ["fold", 820], ["desktop", 1440]]) {
    await page.setViewportSize({width, height: 1000});
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
    await page.locator(".usage-report").screenshot({path: "../../artifacts/usage-report/" + name + ".png"});
  }
  if (errors.length) throw Error(errors.join("\n"));
  console.log("Usage Settings passed: phone/fold/desktop, partial history, tiny charges, no false budget blocking or projection.");
} finally { await browser.close(); }
