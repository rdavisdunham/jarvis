import { chromium, expect } from "@playwright/test";
import { writeFileSync } from "node:fs";
const browser = await chromium.launch();
const context = await browser.newContext({
  viewport: { width: 1440, height: 1050 },
  hasTouch: true,
});
const page = await context.newPage();
const errors = [],
  checks = [];
page.on("pageerror", (e) => errors.push(e.message));
const base = process.env.JARVIS_PLANNER_TEST_URL;
const request = (path, body) =>
  page.evaluate(
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
      const data = await r.json();
      if (!r.ok) throw new Error(JSON.stringify(data));
      return data;
    },
    { path, body },
  );
const cmd = async (tool, args) =>
  (
    await request("/commands", {
      command_id: crypto.randomUUID(),
      tool,
      arguments: args,
    })
  ).data;
const ui = async (name, args = {}) => {
  const result = await request("/__test_ui", { name, arguments: args });
  if (result.status !== "displayed") throw new Error(JSON.stringify(result));
  return result;
};
const shot = (name) =>
  page.screenshot({
    path: new URL(
      "../../../.runtime/calendar-tasks-" + name + ".png",
      import.meta.url,
    ).pathname,
    fullPage: true,
  });
async function dragMouse(title, column) {
  const handle = page.getByRole("button", {
    name: "Move task " + title,
    exact: true,
  });
  await expect(handle).toBeEnabled();
  await handle.scrollIntoViewIfNeeded();
  const a = await handle.boundingBox(),
    b = await page.locator('[data-board-key="' + column + '"]').boundingBox();
  await page.mouse.move(a.x + a.width / 2, a.y + a.height / 2);
  await page.mouse.down();
  await page.mouse.move(b.x + b.width / 2, b.y + 180, { steps: 18 });
  await expect(page.locator(".board-drop-target")).toHaveAttribute(
    "data-board-key",
    column,
  );
  await page.mouse.up();
}
try {
  await page.goto(base);
  await page.getByLabel("Pairing code").fill("planner-fixture");
  await page.locator(".login-card button.primary").click();
  await expect(
    page.getByRole("heading", { name: "Tasks", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("tab", { name: "Today", exact: true }),
  ).toHaveAttribute("aria-selected", "true");
  await expect(
    page.locator("nav").getByRole("button", { name: "Tasks", exact: true }),
  ).toHaveCount(1);
  await expect(
    page.locator("nav").getByRole("button", { name: "Inbox", exact: true }),
  ).toHaveCount(0);
  const today = await page.evaluate(async () =>
    new Intl.DateTimeFormat("en-CA", {
      timeZone: (await (await fetch("/api/v1/bootstrap")).json()).preferences
        .timezone,
    }).format(new Date()),
  );
  const plus = (n) =>
    new Date(Date.parse(today + "T12:00:00Z") + n * 86400000)
      .toISOString()
      .slice(0, 10);
  const project = await cmd("project.create", { name: "Household" });
  const a = await cmd("task.create", {
    title: "Touch drag task",
    notes: "Keep the original notes.\nSecond line.",
    due_date: today,
    planned_date: today,
    priority: 2,
    tags: ["home"],
  });
  const b = await cmd("task.create", {
    title: "Keyboard drag task",
    due_date: plus(3),
    project_id: project.id,
  });
  const c = await cmd("task.create", { title: "Unscheduled capture" });
  const d = await cmd("task.create", {
    title: "Later task",
    due_date: plus(9),
    project_id: project.id,
  });
  await page.reload();
  await expect(
    page.locator(".task-row").filter({ hasText: a.title }),
  ).toHaveCount(1);
  await page.getByRole("tab", { name: "Inbox", exact: true }).click();
  await expect(
    page.locator(".task-row").filter({ hasText: c.title }),
  ).toHaveCount(1);
  await expect(
    page.locator(".task-row").filter({ hasText: b.title }),
  ).toHaveCount(0);
  await page.getByRole("tab", { name: "Next 7 days", exact: true }).click();
  await expect(
    page.locator(".task-row").filter({ hasText: b.title }),
  ).toHaveCount(1);
  await expect(
    page.locator(".task-row").filter({ hasText: d.title }),
  ).toHaveCount(0);
  await page.getByRole("tab", { name: "All", exact: true }).click();
  await expect(page.locator(".task-row")).toHaveCount(4);
  await page.reload();
  await expect(
    page.getByRole("tab", { name: "All", exact: true }),
  ).toHaveAttribute("aria-selected", "true");
  checks.push("One Tasks page, exact tab subsets, bookmark/reload");
  await ui("ui_workspace", {
    view: "all",
    layout: "board",
    group_by: "status",
  });
  await dragMouse(a.title, "in_progress");
  await expect
    .poll(async () => (await request("/tasks/" + a.id)).status)
    .toBe("in_progress");
  const keyboardHandle = page.getByRole("button", {
    name: "Move task " + b.title,
    exact: true,
  });
  await expect(keyboardHandle).toBeEnabled();
  await keyboardHandle.focus();
  await expect(keyboardHandle).toBeFocused();
  await page.keyboard.press("Space");
  await page.keyboard.press("ArrowRight");
  await page.keyboard.press("ArrowRight");
  await page.keyboard.press("Space");
  await expect
    .poll(async () => (await request("/tasks/" + b.id)).status)
    .toBe("waiting");

  // Cancelling a picked-up card must leave the saved status untouched.
  await expect(keyboardHandle).toBeEnabled();
  await keyboardHandle.focus();
  await page.keyboard.press("Space");
  await page.keyboard.press("ArrowRight");
  await page.keyboard.press("Escape");
  await expect(page.locator(".board-drop-target")).toHaveCount(0);
  if ((await request("/tasks/" + b.id)).status !== "waiting")
    throw new Error("Cancelled move changed the task");
  await page.locator(".board-scroll").evaluate((el) => (el.scrollLeft = 0));
  await shot("board-desktop");
  // Real touch input dispatch, not a call to the component's event handler.
  await page.setViewportSize({ width: 390, height: 844 });
  await ui("ui_workspace", {
    view: "all",
    layout: "board",
    group_by: "status",
  });
  await page.locator(".board-scroll").evaluate((el) => (el.scrollLeft = 0));
  const handle = page.getByRole("button", {
    name: "Move task " + c.title,
    exact: true,
  });
  await expect(handle).toBeEnabled();
  await handle.scrollIntoViewIfNeeded();
  const h = await handle.boundingBox(),
    target = await page.locator('[data-board-key="in_progress"]').boundingBox();
  const x0 = h.x + h.width / 2,
    y0 = h.y + h.height / 2,
    x1 = Math.min(365, target.x + 35),
    y1 = Math.max(350, Math.min(650, y0));
  const cdp = await context.newCDPSession(page);
  await cdp.send("Input.dispatchTouchEvent", {
    type: "touchStart",
    touchPoints: [{ x: x0, y: y0 }],
  });
  for (let i = 1; i <= 14; i++) {
    await cdp.send("Input.dispatchTouchEvent", {
      type: "touchMove",
      touchPoints: [
        { x: x0 + ((x1 - x0) * i) / 14, y: y0 + ((y1 - y0) * i) / 14 },
      ],
    });
    await page.waitForTimeout(25);
  }
  await expect(page.locator(".board-drag-preview")).toBeVisible();
  await page.waitForTimeout(250);
  await cdp.send("Input.dispatchTouchEvent", {
    type: "touchEnd",
    touchPoints: [],
  });
  await expect
    .poll(async () => (await request("/tasks/" + c.id)).status)
    .toBe("in_progress");
  await shot("board-mobile");
  checks.push(
    "Mouse, keyboard and real touch moves persist through task commands",
  );
  await page.setViewportSize({ width: 1440, height: 1050 });
  await ui("ui_workspace", {
    view: "all",
    layout: "board",
    group_by: "project",
  });
  await dragMouse(c.title, project.id);
  await expect
    .poll(async () => (await request("/tasks/" + c.id)).project_id)
    .toBe(project.id);
  checks.push("Project grouping drag changes canonical project link");
  const note = await cmd("note.create", {
    title: "Task reference",
    content: "Linked source",
    task_ids: [a.id],
  });
  const alert = await cmd("schedule.create", {
    title: "Check the touch task",
    kind: "reminder",
    task_id: a.id,
    when: today + "T19:00",
    timezone: "America/Chicago",
  });
  const appointmentDay = plus(1);
  const appointment = await cmd("planning.create", {
    title: "Dentist appointment",
    start: appointmentDay + "T09:00",
    end: appointmentDay + "T10:00",
    timezone: "America/Chicago",
    location: "12 Main Street",
    description: "Bring insurance card.\nParking behind the building.",
    busy: true,
  });
  const block = await cmd("planning.create", {
    title: "Focus on household",
    kind: "block",
    task_id: b.id,
    start: today + "T11:00",
    end: today + "T12:00",
    timezone: "America/Chicago",
    description: "Prepare everything",
  });
  await ui("ui_calendar", {
    date: today,
    calendar_view: "day",
    entity_id: a.id,
    open_details: true,
  });
  await expect(page.getByRole("dialog")).toContainText(
    "Keep the original notes.",
  );
  const info = await ui("ui_editor", {operation:"read"});
  if(!info.data.auto_save || !info.data.saved) throw new Error("Task is not a saved inline card");
  const before = await request("/tasks/"+a.id);
  await ui("ui_editor",{operation:"patch",changes:{title:"Touch task edited from calendar"}});
  const after = await request("/tasks/"+a.id);
  if(after.notes!==before.notes || JSON.stringify(after.tags)!==JSON.stringify(before.tags))throw new Error("Sparse task edit lost fields");
  await page.getByRole("button",{name:"Complete task",exact:true}).click();
  await expect.poll(async()=>(await request("/tasks/"+a.id)).status).toBe("completed");
  await expect.poll(async()=>(await request("/schedules/"+alert.id)).status).toBe("completed");
  await ui("ui_show",{view:"calendar"});
  checks.push(
    "Task calendar details, sparse Edit, task completion closes its alert",
  );
  await ui("ui_calendar", {
    date: appointmentDay,
    entity_id: appointment.id,
    open_details: true,
  });
  await expect(page.getByRole("dialog")).toContainText("12 Main Street");
  await expect(page.getByRole("dialog")).toContainText(
    "Parking behind the building.",
  );
  await expect(
    page.getByRole("button", { name: "Complete task", exact: true }),
  ).toHaveCount(0);
  await shot("event-details-desktop");
  await page
    .getByRole("button", { name: "Edit calendar item", exact: true })
    .click();
  await ui("ui_editor", {
    operation: "patch",
    changes: { location: "34 Oak Street" },
  });
  await ui("ui_editor", { operation: "save" });
  await expect
    .poll(
      async () =>
        (await request("/planning/" + appointment.id)).fields.location,
    )
    .toBe("34 Oak Street");
  await ui("ui_calendar", {
    date: today,
    entity_id: block.id,
    open_details: true,
  });
  await expect(page.getByRole("dialog")).toContainText("Linked task");
  await expect(
    page.getByRole("button", { name: "Complete task", exact: true }),
  ).toHaveCount(0);
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toHaveCount(0);
  // Month chips themselves open details instead of only selecting the day.
  await ui("ui_calendar", { date: appointmentDay, calendar_view: "month" });
  await page
    .getByRole("button", { name: "Event: Dentist appointment", exact: true })
    .click();
  await expect(page.getByRole("dialog")).toContainText("34 Oak Street");
  await page.setViewportSize({ width: 390, height: 844 });
  await shot("event-details-mobile");
  const editRect = await page
    .getByRole("button", { name: "Edit calendar item", exact: true })
    .boundingBox();
  const titleRect = await page.locator("#calendar-detail-title").boundingBox();
  if (editRect.y >= titleRect.y || editRect.x < 200)
    throw new Error("Edit button is not at top right");
  await ui("ui_editor", { operation: "close" });
  await ui("ui_workspace", { view: "all", layout: "list" });
  await ui("ui_form", { form: "reminder" });
  await expect(page.getByRole("dialog")).toContainText(
    "A reminder alerts you about a task",
  );
  await ui("ui_editor", { operation: "close" });
  checks.push(
    "Event/block details, upper-right Edit, clickable month chips, reminder explanation",
  );
  for (const view of ["today", "inbox", "week", "all", "calendar"]) {
    await ui("ui_show", { view });
    if (
      await page.evaluate(
        () => document.documentElement.scrollWidth > innerWidth + 1,
      )
    )
      throw new Error("Overflow in " + view);
  }
  if (errors.length) throw new Error(errors.join("\n"));
  writeFileSync(
    new URL("../../../.runtime/calendar-tasks-evidence.json", import.meta.url),
    JSON.stringify({ passed: true, checks, pageErrors: errors }, null, 2),
  );
  console.log("Calendar/task browser acceptance passed: " + checks.join("; "));
} catch (error) {
  await shot("failure");
  console.error(error);
  throw error;
} finally {
  await browser.close();
}
