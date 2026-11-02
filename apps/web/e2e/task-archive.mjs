import { chromium, expect } from "@playwright/test";
import { mkdirSync } from "node:fs";

const base = process.env.JARVIS_PLANNER_TEST_URL;
if (!base) throw new Error("Use the isolated planner acceptance harness");
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
const errors = [];
page.on("pageerror", error => errors.push(error.message));
async function request(path, body) {
  return page.evaluate(async ({ path, body }) => {
    const boot = await (await fetch("/api/v1/bootstrap")).json();
    const response = await fetch("/api/v1" + path, {
      method: body ? "POST" : "GET",
      headers: { "Content-Type": "application/json", "X-CSRF-Token": boot.csrf },
      ...(body ? { body: JSON.stringify(body) } : {}),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(JSON.stringify(data));
    return data;
  }, { path, body });
}
const command = async (tool, args) => (await request("/commands", {
  command_id: crypto.randomUUID(), tool, arguments: args,
})).data;
const ui = async (name, args) => {
  const result = await request("/__test_ui", { name, arguments: args });
  if (result.status !== "displayed") throw new Error(JSON.stringify(result));
};
const titleRow = title => page.locator(".row-title").filter({ hasText: title });
async function filter(name, checked) {
  await page.getByRole("button", { name: /^Filter/ }).click();
  await page.getByRole("checkbox", { name, exact: true }).setChecked(checked);
  await page.keyboard.press("Escape");
}

try {
  await page.goto(base);
  await page.getByLabel("Pairing code").fill("planner-fixture");
  await page.locator(".login-card button.primary").click();
  await expect(page.getByRole("region", { name: "Today", exact: true })).toBeVisible();
  await command("settings.update", { auto_archive_finished_tasks: true, timezone: "America/Chicago" });
  const today = await page.evaluate(() => new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/Chicago", year: "numeric", month: "2-digit", day: "2-digit",
  }).format(new Date()));
  const make = title => command("task.create", { title, due_date: today, notes: "Archive fixture retained history" });
  const active = await make("Archive fixture unfinished");
  let completed = await make("Archive fixture completed");
  let canceled = await make("Archive fixture canceled");
  completed = await command("task.complete", { task_id: completed.id, expected_revision: completed.revision });
  canceled = await command("task.update", { task_id: canceled.id, expected_revision: canceled.revision, status: "cancelled" });

  // Today remains a visible daily checklist, with no duplicates in outstanding work.
  await page.goto(base + "/?view=today");
  const done = page.getByRole("region", { name: "Completed today", exact: true });
  const cancelled = page.getByRole("region", { name: "Canceled today", exact: true });
  await expect(done.getByRole("button", { name: completed.title, exact: true })).toBeVisible();
  await expect(cancelled.getByRole("button", { name: canceled.title, exact: true })).toBeVisible();
  await expect(titleRow(completed.title)).toHaveCount(1);
  await expect(titleRow(active.title)).toBeVisible();
  const activeBox = await titleRow(active.title).boundingBox();
  const completedBox = await done.boundingBox();
  expect(completedBox.y).toBeGreaterThan(activeBox.y);

  // The Today task tab has the same checklist; other task tabs default to active.
  await page.goto(base + "/?view=tasks&tab=today");
  await expect(done.getByRole("button", { name: completed.title, exact: true })).toBeVisible();
  await expect(titleRow(completed.title)).toHaveCount(1);
  await page.goto(base + "/?view=tasks&tab=all");
  await expect(titleRow(active.title)).toBeVisible();
  await expect(titleRow(completed.title)).toHaveCount(0);
  await expect(titleRow(canceled.title)).toHaveCount(0);
  await filter("Show completed and canceled", true);
  await expect(titleRow(completed.title)).toBeVisible();
  await expect(titleRow(canceled.title)).toBeVisible();

  // Atlas honors the same visibility toggle and retains unfinished work.
  await ui("ui_records", { layout: "atlas", parent_id: "" });
  const atlas = page.getByRole("region", { name: "Atlas", exact: true });
  await expect(atlas).toBeVisible();
  await expect(atlas.getByRole("button", { name: new RegExp(": " + active.title + "(?:,|$)") })).toBeVisible();
  await expect(atlas.getByRole("button", { name: new RegExp(": " + completed.title + "(?:,|$)") })).toHaveCount(0);
  await filter("Show completed and canceled", true);
  await expect(atlas.getByRole("button", { name: new RegExp(": " + completed.title + "(?:,|$)") })).toBeVisible();

  // The real worker files aged records and preserves their outcomes/content.
  await request("/__test_task_archive", { task_ids: [completed.id, canceled.id] });
  const archived = await request("/tasks/" + completed.id);
  expect(archived.archived).toBe(true);
  expect(archived.archive_reason).toBe("automatic");
  expect(archived.status).toBe("completed");
  expect(archived.notes).toBe("Archive fixture retained history");
  expect((await request("/tasks/" + canceled.id)).archived).toBe(true);
  expect((await request("/tasks/" + active.id)).archived).toBe(false);
  await page.goto(base + "/?view=tasks&tab=all");
  await filter("Archived", true);
  await expect(titleRow(completed.title)).toBeVisible();
  await titleRow(completed.title).click();
  await page.getByRole("button", { name: /^Restore( task)?$/ }).click();
  await expect.poll(async () => (await request("/tasks/" + completed.id)).archived).toBe(false);
  const restored = await request("/tasks/" + completed.id);
  expect(restored.status).toBe("completed");
  expect(Date.parse(restored.archive_eligible_at)).toBeGreaterThan(Date.now() + 23 * 60 * 60 * 1000);

  // A canceled result can be reopened as ordinary work.
  await titleRow(canceled.title).click();
  await page.getByRole("button", { name: /^Reopen( task)?$/ }).click();
  await expect.poll(async () => (await request("/tasks/" + canceled.id)).status).toBe("open");
  const reopened = await request("/tasks/" + canceled.id);
  expect(reopened.status).toBe("open");
  expect(reopened.archived).toBe(false);
  expect(reopened.finished_at).toBeNull();

  await ui("ui_workspace", { view: "settings", settings_section: "organization" });
  const policy = page.getByRole("checkbox", { name: "Automatically archive finished tasks", exact: true });
  await expect(policy).toBeChecked();
  await policy.click();
  await expect.poll(async () => (await request("/bootstrap")).preferences.auto_archive_finished_tasks).toBe(false);
  await expect(policy).not.toBeChecked();
  await expect(page.getByText("Automatic archiving is paused.", { exact: false })).toBeVisible();
  await policy.click();
  await expect.poll(async () => (await request("/bootstrap")).preferences.auto_archive_finished_tasks).toBe(true);
  await expect(policy).toBeChecked();

  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(base + "/?view=today");
  await expect(titleRow(canceled.title)).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  mkdirSync("../../artifacts/task-archive", { recursive: true });
  await page.screenshot({ path: "../../artifacts/task-archive/today-mobile.png", fullPage: true });
  expect(errors).toEqual([]);
  console.log("PASS task archive: daily checklist, active filters, Atlas, worker, restore/reopen, policy, mobile");
} finally {
  await browser.close();
}
