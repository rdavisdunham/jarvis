import { chromium, expect } from "@playwright/test";
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
const errors = [];
page.on("pageerror", error => errors.push(error.message));
try {
  await page.goto(process.env.JARVIS_REVIEW_TEST_URL);
  await page.getByLabel("Pairing code").fill("memory-review-fixture");
  await page.locator(".login-card button.primary").click();
  await page.getByRole("button", { name: "All tasks", exact: true }).click();
  const task = await page.evaluate(async () => {
    const boot = await (await fetch("/api/v1/bootstrap")).json();
    const response = await fetch("/api/v1/commands", {
      method: "POST", headers: { "Content-Type": "application/json", "X-CSRF-Token": boot.csrf },
      body: JSON.stringify({ command_id: crypto.randomUUID(), tool: "task.create",
        arguments: { title: "Due time fixture" } }),
    });
    if (!response.ok) throw new Error("Fixture did not save");
    return (await response.json()).data;
  });
  await page.getByRole("button", { name: "Edit Due time fixture", exact: true }).click();
  await page.getByLabel("Due date", { exact: true }).fill("2026-10-12");
  await page.getByLabel("Due time (optional)", { exact: true }).fill("14:30");
  await expect(page.getByLabel("Due time zone", { exact: true })).toHaveValue("America/Chicago");
  await page.setViewportSize({ width: 390, height: 844 });
  if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)) throw new Error("Task editor overflows on mobile");
  await page.screenshot({ path: new URL("../../../.runtime/task-time-mobile.png", import.meta.url).pathname, fullPage: true });
  await page.getByRole("dialog").getByRole("button", { name: /save/i }).click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  const saved = await page.evaluate(async id => (await fetch("/api/v1/tasks/" + id)).json(), task.id);
  if (saved.due_time !== "14:30" || saved.due_timezone !== "America/Chicago") throw new Error("Timed deadline did not persist");
  await expect(page.getByRole("button", { name: "Oct 12 · 14:30", exact: true })).toBeVisible();
  if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)) throw new Error("Task list overflows on mobile");
  await page.screenshot({ path: new URL("../../../.runtime/task-time-list-mobile.png", import.meta.url).pathname, fullPage: true });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.getByRole("button", { name: "Settings", exact: true }).click();
  await expect(page.getByText(/At this month's pace:/)).toBeVisible();
  await expect(page.getByText(/reserved for active work/)).toBeVisible();
  if (errors.length) throw new Error(errors.join("; "));
  console.log(JSON.stringify({ result: "passed", taskDueTime: true, mobile: true, budgetStatus: true }));
} finally { await browser.close(); }
