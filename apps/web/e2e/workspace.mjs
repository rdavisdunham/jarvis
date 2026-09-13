import { chromium, expect } from "@playwright/test";
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
try {
  await page.goto(process.env.JARVIS_WORKSPACE_TEST_URL);
  await page.getByLabel("Pairing code").fill("workspace-fixture");
  await page.locator(".login-card button.primary").click();
  await page.getByRole("button", { name: "Goals & projects", exact: true }).click();
  await page.getByRole("tab", { name: "Projects", exact: true }).click();
  await page.getByRole("button", { name: "New project", exact: true }).click();
  await page.getByRole("dialog").getByLabel("Name", { exact: true }).fill("Workspace verification");
  await page.getByRole("button", { name: "Save project", exact: true }).click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await page.getByRole("button", { name: "Work", exact: true }).click();
  const today = await page.evaluate(async () => {
    const boot = await (await fetch("/api/v1/bootstrap")).json();
    return new Intl.DateTimeFormat("en-CA", {
      timeZone: boot.preferences.timezone,
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
    }).format(new Date());
  });
  await page.getByRole("button", { name: "New task", exact: true }).click();
  await page
    .getByRole("dialog")
    .getByLabel("Task", { exact: true })
    .fill("Calendar acceptance");
  await page.getByLabel("Due date", { exact: true }).fill(today);
  await page.getByLabel("Due time (optional)").fill("09:45");
  await page
    .getByRole("dialog")
    .getByLabel("Project", { exact: true })
    .selectOption({ label: "Workspace verification" });
  await page.getByLabel("Assignee", { exact: true }).fill("Eri");
  await page.getByLabel("Work type", { exact: true }).fill("Planning");
  await page.getByLabel("Tags", { exact: true }).fill("calendar, test");
  await page.getByRole("button", { name: "Save task", exact: true }).click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await page
    .getByRole("button", { name: "Edit Calendar acceptance", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Add linked reminder", exact: true })
    .click();
  await page.getByLabel("When", { exact: true }).fill(today + "T10:15");
  await page.getByLabel("Repeat", { exact: true }).selectOption("FREQ=DAILY");
  await page
    .getByRole("button", { name: "Save reminder", exact: true })
    .click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await expect(page.locator(".linked-reminder")).toHaveCount(1);
  await page.getByRole("button", { name: "Calendar", exact: true }).click();
  await expect(page.locator(".calendar-day")).toHaveCount(42);
  await expect(page.locator(".agenda-row")).toHaveCount(2);
  await expect(page.getByText("Task deadline", { exact: true })).toBeVisible();
  await expect(
    page.getByText("Reminder · Upcoming · Linked to task", { exact: true }),
  ).toBeVisible();
  await page.screenshot({
    path: new URL(
      "../../../.runtime/workspace-calendar-desktop.png",
      import.meta.url,
    ).pathname,
    fullPage: true,
  });
  await page.getByRole("button", { name: "Next month", exact: true }).click();
  await expect(page.locator(".calendar-heading h2")).not.toContainText(
    new Date(today + "T12:00:00Z").toLocaleDateString("en-US", {
      month: "long",
    }),
  );
  await page.getByRole("button", { name: "Today", exact: true }).last().click();
  await expect(page.locator(".agenda-row")).toHaveCount(2);
  await page.setViewportSize({ width: 390, height: 844 });
  if (
    await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)
  )
    throw new Error("Mobile calendar overflows");
  await page.screenshot({
    path: new URL(
      "../../../.runtime/workspace-calendar-mobile.png",
      import.meta.url,
    ).pathname,
    fullPage: true,
  });
  await page
    .locator(".agenda-row")
    .filter({ hasText: "Task deadline" })
    .click();
  await expect(page.getByLabel("Assignee", { exact: true })).toHaveValue("Eri");
  await expect(page.getByLabel("Tags", { exact: true })).toHaveValue(
    "calendar, test",
  );
  if (
    await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)
  )
    throw new Error("Task details overflow");
  await page.screenshot({
    path: new URL(
      "../../../.runtime/workspace-task-mobile.png",
      import.meta.url,
    ).pathname,
    fullPage: true,
  });
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "Close", exact: true })
    .click();
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.getByRole("button", { name: "Work", exact: true }).click();
  await page.getByLabel("Work kind filter").selectOption("reminder");
  await expect(page.locator(".work-reminder")).toHaveCount(1);
  await expect(page.locator(".work-item")).toHaveCount(0);
  await page.locator(".work-reminder .task-info").click();
  await page
    .getByLabel("Reminder", { exact: true })
    .fill("Linked calendar reminder");
  await page
    .getByRole("button", { name: "Save reminder", exact: true })
    .click();
  await expect(page.locator(".work-reminder")).toContainText(
    "Linked calendar reminder",
  );
  const uiAction = async (name, arguments_) =>
    page.evaluate(
      async ({ name, arguments_ }) => {
        const boot = await (await fetch("/api/v1/bootstrap")).json();
        const response = await fetch("/api/v1/__test_ui", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-CSRF-Token": boot.csrf,
          },
          body: JSON.stringify({ name, arguments: arguments_ }),
        });
        if (!response.ok) throw new Error("UI fixture request failed");
        return response.json();
      },
      { name, arguments_ },
    );
  const moved = await uiAction("ui_calendar", { date: today });
  if (moved.status !== "displayed")
    throw new Error("Calendar action was not acknowledged");
  await expect(page.locator(".calendar-heading h2")).toBeVisible();
  const filtered = await uiAction("ui_filter", {
    view: "calendar",
    work_kind: "reminder",
    project: "Workspace verification",
    status: "open",
  });
  if (filtered.status !== "displayed")
    throw new Error("Calendar filter was not acknowledged");
  await expect(page.locator(".agenda-row")).toHaveCount(1);
  await expect(page.locator(".agenda-row")).toContainText(
    "Linked calendar reminder",
  );
  await page.locator(".agenda-row").click();
  const rejected = await uiAction("ui_calendar", { date: today });
  if (rejected.status !== "failed")
    throw new Error("Unsaved reminder editor was not protected");
  await expect(page.getByRole("dialog")).toBeVisible();
  await page
    .getByRole("button", { name: "Close reminder", exact: true })
    .click();
  await page.getByRole("button", { name: "Settings", exact: true }).click();
  await expect(page.locator(".budget-holds summary")).toContainText(
    "1 sessions",
  );
  await page.locator(".budget-holds summary").click();
  await expect(page.getByText(/not confirmed charges/)).toBeVisible();
  if (errors.length) throw new Error(errors.join("; "));
  console.log(
    JSON.stringify({
      passed: true,
      project: true,
      taskMetadata: true,
      linkedReminder: true,
      calendar: true,
      mobile: true,
      holds: true,
    }),
  );
} catch (error) {
  await page.screenshot({
    path: new URL("../../../.runtime/workspace-failure.png", import.meta.url)
      .pathname,
    fullPage: true,
  });
  throw error;
} finally {
  await browser.close();
}
