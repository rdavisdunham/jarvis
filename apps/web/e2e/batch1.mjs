import { chromium, expect } from "@playwright/test";
import { writeFileSync } from "node:fs";
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
async function request(path, body) {
  return page.evaluate(
    async ({ path, body }) => {
      const boot = await (await fetch("/api/v1/bootstrap")).json();
      const r = await fetch("/api/v1" + path, {
        method: body ? "POST" : "GET",
        headers: {
          "Content-Type": "application/json",
          "X-CSRF-Token": boot.csrf,
        },
        ...(body ? { body: JSON.stringify(body) } : {}),
      });
      const result = await r.json();
      if (!r.ok) throw new Error(JSON.stringify(result));
      return result;
    },
    { path, body },
  );
}
const cmd = async (tool, args) =>
  (
    await request("/commands", {
      command_id: crypto.randomUUID(),
      tool,
      arguments: args,
    })
  ).data;
const ui = async (name, args = {}, status = "displayed") => {
  const r = await request("/__test_ui", { name, arguments: args });
  if (r.status !== status) throw new Error(name + ": " + JSON.stringify(r));
  return r;
};
const shot = (name) =>
  page.screenshot({
    path: new URL("../../../.runtime/batch1-" + name + ".png", import.meta.url)
      .pathname,
    fullPage: true,
  });
try {
  await page.goto(process.env.JARVIS_PLANNER_TEST_URL);
  await page.getByLabel("Pairing code").fill("planner-fixture");
  await page.locator(".login-card button.primary").click();
  await expect(
    page.getByRole("heading", { name: "Tasks", exact: true }),
  ).toBeVisible();
  const goal = await cmd("goal.create", { name: "A calmer week" });
  const project = await cmd("project.create", {
    name: "Home life",
    goal_ids: [goal.id],
  });
  const task = await cmd("task.create", {
    title: "Plan the week",
    notes: "A quiet Sunday ritual.\nKeep the original spacing.",
    project_id: project.id,
    tags: ["home"],
    due_date: "2030-01-12",
  });
  await page.reload();
  await ui("ui_show", { view: "all", entity_id: task.id });
  await expect(page.getByRole("dialog")).toContainText("A calmer week");
  await expect(
    page.getByRole("button", { name: /^(Edit|Save task)$/ }),
  ).toHaveCount(0);
  await shot("task-desktop");
  await page.getByRole("button", { name: "Change Task", exact: true }).click();
  await page
    .getByRole("textbox", { name: "Task", exact: true })
    .fill("Plan a better week");
  // Navigate by agent while a field is still focused, without an explicit save.
  await ui("ui_show", { view: "calendar" });
  await expect(page.getByRole("dialog")).toHaveCount(0);
  if ((await request("/tasks/" + task.id)).title !== "Plan a better week")
    throw new Error("Navigation lost a pending edit");
  await ui("ui_show", { view: "all", entity_id: task.id });
  await page
    .getByRole("button", { name: "Change Description", exact: true })
    .click();
  await page
    .getByLabel("Description", { exact: true })
    .fill("Do not save this");
  await page.keyboard.press("Escape");
  await ui("ui_show", { view: "notes" });
  if ((await request("/tasks/" + task.id)).notes !== task.notes)
    throw new Error("Escape saved text");
  // A stale inline field stays visible, does not overwrite the other user's change.
  await ui("ui_show", { view: "all", entity_id: task.id });
  await page
    .getByRole("button", { name: "Change Description", exact: true })
    .click();
  await page
    .getByLabel("Description", { exact: true })
    .fill("My pending context");
  const current = await request("/tasks/" + task.id);
  await cmd("task.update", {
    task_id: task.id,
    expected_revision: current.revision,
    notes: "Server's newer context",
  });
  await ui("ui_show", { view: "calendar" }, "failed");
  await expect(page.getByLabel("Description", { exact: true })).toHaveValue(
    "My pending context",
  );
  await page.keyboard.press("Escape");
  await ui("ui_show", { view: "all" });
  await ui("ui_filter", { view: "all", tag: "home", status: "active" });
  await ui("ui_workspace", {
    view: "all",
    layout: "board",
    group_by: "project",
    sort: "due",
  });
  await ui("ui_saved_view", {
    view_operation: "save",
    view_name: "Home board",
  });
  const views = await ui("ui_saved_view", { view_operation: "list" });
  const saved = views.data.items.find((x) => x.name === "Home board");
  if (!saved) throw new Error("No saved view");
  await ui("ui_search", { view: "all", query: "nothing here" });
  let ack = await ui("ui_workspace", { view: "all", layout: "timeline" });
  if (
    ack.data.observed.reported_visible_count !== 0 ||
    ack.data.observed.layout !== "timeline"
  )
    throw new Error("Wrong observed empty/layout state");
  await ui("ui_saved_view", {
    view_operation: "load",
    saved_view_id: saved.id,
  });
  await expect(page.locator(".board-card")).toHaveCount(1);
  const link = page.url();
  await page.goto(link);
  await expect(page.locator(".board-card")).toHaveCount(1);
  await ui("ui_saved_view", {
    view_operation: "delete",
    saved_view_id: saved.id,
  });
  await ui("ui_show", { view: "all", entity_id: task.id });
  await page.setViewportSize({ width: 390, height: 844 });
  await shot("task-mobile");
  if (
    await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth + 1,
    )
  )
    throw new Error("Mobile overflow");
  await ui("ui_editor", { operation: "patch", changes: { priority: 3 } });
  if ((await request("/tasks/" + task.id)).priority !== 3)
    throw new Error("Inline agent patch not saved");
  await ui("ui_show", { view: "notes" });
  const note = await cmd("note.create", {
    title: "Archived note",
    content: "Keep this",
  });
  await cmd("note.update", {
    note_id: note.id,
    expected_revision: note.revision,
    archived: true,
  });
  await ui("ui_workspace", { view: "notes", show_archived: true });
  await expect(page.getByText("Archived note", { exact: true })).toBeVisible();
  if (errors.length) throw new Error(errors.join("; "));
  writeFileSync(
    new URL("../../../.runtime/batch1-browser.json", import.meta.url),
    JSON.stringify({ passed: true, errors }, null, 2),
  );
  console.log(
    "Batch 1 browser acceptance passed: inline save/navigation, cancel/conflict, attribution, saved views/reload, exact empty/layout acknowledgement, archived notes and mobile.",
  );
} catch (e) {
  await shot("failure");
  console.log(errors);
  throw e;
} finally {
  await browser.close();
}
