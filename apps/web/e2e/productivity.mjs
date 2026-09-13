import { chromium, expect } from "@playwright/test";
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
const errors = [];
page.on("pageerror", e => errors.push(e.message));
const dialog = () => page.getByRole("dialog");
const graph = () => page.evaluate(async () => (await (await fetch("/api/v1/organization")).json()));
const shot = name => page.screenshot({ path: new URL("../../../.runtime/" + name, import.meta.url).pathname, fullPage: true });
async function create(kind, name, fill) {
  await page.getByRole("button", { name: "New " + kind, exact: true }).click();
  await dialog().getByLabel("Name", { exact: true }).fill(name);
  if (fill) await fill(dialog());
  await dialog().getByRole("button", { name: "Save " + kind, exact: true }).click();
  await expect(dialog()).toHaveCount(0);
}
const uiAction = (name, args) => page.evaluate(async ({ name, args }) => {
  const boot = await (await fetch("/api/v1/bootstrap")).json();
  const response = await fetch("/api/v1/__test_ui", { method: "POST", headers: { "Content-Type": "application/json", "X-CSRF-Token": boot.csrf }, body: JSON.stringify({ name, arguments: args }) });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}, { name, args });
async function tab(name) { await page.getByRole("tab", { name, exact: true }).click(); }
try {
  await page.goto(process.env.JARVIS_PRODUCTIVITY_TEST_URL);
  await page.getByLabel("Pairing code").fill("productivity-fixture");
  await page.locator(".login-card button.primary").click();
  await page.getByRole("button", { name: "Goals & projects", exact: true }).click();
  await tab("Areas");
  await create("area", "Operations", async d => { await d.getByLabel("Space", { exact: true }).selectOption({ label: "Business" }); });
  await tab("Goals");
  await create("goal", "Five new clients", async d => {
    await d.getByLabel("Space", { exact: true }).selectOption({ label: "Business" });
    await d.getByLabel("Area", { exact: true }).selectOption({ label: "Operations" });
    await d.getByLabel("Success criteria").fill("Five signed customers this quarter.");
    await d.locator("summary").filter({ hasText: "Optional outcome metric" }).click();
    await d.getByLabel("Current value").fill("1");
    await d.getByLabel("Target value").fill("5");
    await d.getByLabel("Unit", { exact: true }).fill("clients");
  });
  await create("goal", "Less admin");
  await tab("Projects");
  await create("project", "Website", async d => {
    await d.getByLabel("Space", { exact: true }).selectOption({ label: "Business" });
    await d.getByLabel("Area", { exact: true }).selectOption({ label: "Operations" });
    await d.getByLabel("Five new clients", { exact: true }).check();
    await d.getByLabel("Less admin", { exact: true }).check();
  });
  await create("project", "Referrals", async d => { await d.getByLabel("Five new clients", { exact: true }).check(); });
  let data = await graph();
  const goal = data.goals.find(g => g.name === "Five new clients");
  const project = data.projects.find(p => p.name === "Website");
  expect(goal.project_ids).toHaveLength(2);
  expect(project.goal_ids).toHaveLength(2);
  await tab("Goals");
  await expect(page.getByRole("progressbar")).toHaveAttribute("value", "0.2");
  await shot("productivity-goals-desktop.png");
  // Unsaved form inputs survive the real event refresh interval.
  await page.getByRole("button", { name: "Five new clients", exact: true }).click();
  await dialog().getByLabel("Name", { exact: true }).fill("Five clients, reviewed");
  await page.waitForTimeout(22000);
  await expect(dialog().getByLabel("Name", { exact: true })).toHaveValue("Five clients, reviewed");
  await dialog().getByRole("button", { name: "Save goal", exact: true }).click();
  await expect(dialog()).toHaveCount(0);
  // Planned work is distinct from its deadline and follows the project home.
  await page.getByRole("button", { name: "Work", exact: true }).click();
  await page.getByRole("button", { name: "New task", exact: true }).click();
  await dialog().getByLabel("Task", { exact: true }).fill("Review website copy");
  await dialog().getByLabel("Project", { exact: true }).selectOption({ label: "Website" });
  const today = new Date().toISOString().slice(0, 10);
  await dialog().getByLabel("Planned date", { exact: true }).fill(today);
  await dialog().getByLabel("Due date", { exact: true }).fill("2030-01-15");
  await dialog().getByRole("button", { name: "Save task", exact: true }).click();
  await expect(dialog()).toHaveCount(0);
  await page.getByLabel("Space filter").selectOption({ label: "Business" });
  await page.getByLabel("Goal filter").selectOption({ label: "Five clients, reviewed" });
  await expect(page.getByRole("button", { name: "Edit Review website copy", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Edit Preserved task", exact: true })).toHaveCount(0);
  // Authored notes can link to two projects, a goal and another note.
  await page.getByRole("button", { name: "Notes", exact: true }).click();
  await page.getByRole("button", { name: "New note", exact: true }).click();
  await dialog().getByLabel("Title", { exact: true }).fill("Research");
  await dialog().getByLabel("Note", { exact: true }).fill("Evidence supporting the plan.");
  await dialog().getByRole("button", { name: "Save note", exact: true }).click();
  await expect(dialog().getByRole("button", { name: "Find to-dos", exact: true })).toBeEnabled();
  await dialog().getByRole("button", { name: "Close note", exact: true }).click();
  await page.getByRole("button", { name: "New note", exact: true }).click();
  await dialog().getByLabel("Title", { exact: true }).fill("Meeting decisions");
  await dialog().locator("summary").filter({ hasText: "Connected goals, projects and notes" }).click();
  await dialog().getByLabel("Five clients, reviewed", { exact: true }).check();
  await dialog().getByLabel("Website", { exact: true }).check();
  await dialog().getByLabel("Referrals", { exact: true }).check();
  await expect(dialog().getByLabel("Research", { exact: true })).toBeVisible();
  await dialog().getByLabel("Research", { exact: true }).check();
  await dialog().getByRole("button", { name: "Save note", exact: true }).click();
  await expect(dialog().getByRole("button", { name: "Research", exact: true })).toBeEnabled();
  await dialog().getByRole("button", { name: "Research", exact: true }).click();
  await expect(dialog().getByRole("button", { name: "Meeting decisions", exact: true })).toBeVisible();
  await dialog().getByRole("button", { name: "Close note", exact: true }).click();
  // Mobile view/edit has no horizontal overflow.
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Open navigation", exact: true }).click();
  await page.getByRole("button", { name: "Goals & projects", exact: true }).click();
  await expect(page.getByRole("button", { name: "Five clients, reviewed", exact: true })).toBeVisible();
  await expect(page.locator(".sidebar")).not.toHaveClass(/open/);
  await page.waitForTimeout(300);
  await shot("productivity-goals-mobile.png");
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.getByRole("button", { name: "Five clients, reviewed", exact: true }).click();
  await expect(dialog().getByLabel("Name", { exact: true })).toBeVisible();
  await shot("productivity-edit-mobile.png");
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await dialog().getByRole("button", { name: "Close organization editor" }).click();
  const shown = await uiAction("ui_show", { view: "organize", entity_id: project.id });
  expect(shown.status).toBe("displayed");
  await expect(page.getByRole("tab", { name: "Projects", exact: true })).toHaveAttribute("aria-selected", "true");
  await expect(page.locator('[data-entity-id="' + project.id + '"]')).toHaveClass(/highlighted/);
  const filtered = await uiAction("ui_filter", { view: "all", goal_id: goal.id, space_id: project.space_id });
  expect(filtered.status).toBe("displayed");
  await expect(page.getByRole("button", { name: "Edit Review website copy", exact: true })).toBeVisible();
  expect(errors).toEqual([]);
  console.log("Many-to-many goals/projects, notes/backlinks, scoped task filters, unsaved refresh, mobile layout passed.");
} catch (error) { await shot("productivity-failure.png"); console.log(await page.locator("body").innerText()); throw error; } finally { await browser.close(); }
