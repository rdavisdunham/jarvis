import { chromium, expect } from "@playwright/test";
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
const shot = (name) =>
  page.screenshot({
    path: new URL("../../../.runtime/" + name, import.meta.url).pathname,
    fullPage: true,
  });
const uiAction = (name, args) =>
  page.evaluate(
    async ({ name, args }) => {
      const boot = await (await fetch("/api/v1/bootstrap")).json();
      const response = await fetch("/api/v1/__test_ui", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-CSRF-Token": boot.csrf,
        },
        body: JSON.stringify({ name, arguments: args }),
      });
      if (!response.ok) throw new Error("UI fixture request failed");
      return response.json();
    },
    { name, args },
  );
try {
  await page.goto(process.env.JARVIS_NOTES_TEST_URL);
  await page.getByLabel("Pairing code").fill("notes-fixture");
  await page.locator(".login-card button.primary").click();
  await page.getByRole("button", { name: "Notes", exact: true }).click();
  await page.getByRole("button", { name: "New note", exact: true }).click();
  await page.getByLabel("Title", { exact: true }).fill("Home planning");
  await page
    .getByLabel("Note", { exact: true })
    .fill("Call the dentist.\nBuy printer paper.");
  await page.getByLabel("Tags", { exact: true }).fill("home, errands");
  let dropped = false;
  const loseCommittedResponse = async (route) => {
    const body = route.request().postDataJSON();
    if (!dropped && body.tool === "note.create") {
      dropped = true;
      await route.fetch();
      await route.abort("connectionfailed");
    } else await route.continue();
  };
  await page.route("**/api/v1/commands", loseCommittedResponse);
  await page.getByRole("button", { name: "Save note", exact: true }).click();
  await expect(page.getByRole("dialog").getByRole("alert")).toContainText(
    "Connection lost",
  );
  await page.getByRole("button", { name: "Save note", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Find to-dos", exact: true }),
  ).toBeEnabled();
  await page.unroute("**/api/v1/commands", loseCommittedResponse);
  const savedNotes = await page.evaluate(
    async () => (await (await fetch("/api/v1/notes")).json()).items,
  );
  if (savedNotes.length !== 1)
    throw new Error("Retrying a lost response duplicated the note");
  await page.getByRole("button", { name: "Find to-dos", exact: true }).click();
  await expect(page.getByLabel("Suggested task 1")).toHaveValue(
    "Call the dentist",
  );
  await page
    .getByRole("button", { name: "Create selected tasks", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Create selected tasks", exact: true }),
  ).toHaveCount(0);
  await expect(page.locator(".note-linked-records")).toContainText(
    "Call the dentist",
  );
  await page.getByRole("button", { name: "Find to-dos", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Create selected tasks", exact: true }),
  ).toBeDisabled();
  await page.setViewportSize({ width: 390, height: 844 });
  if (
    await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)
  )
    throw new Error("Note editor overflows mobile");
  await shot("notes-editor-mobile.png");
  await page.getByRole("button", { name: "Close note", exact: true }).click();
  await shot("notes-list-mobile.png");
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.getByRole("button", { name: "Work", exact: true }).click();
  await page.getByRole("button", { name: "Select tasks", exact: true }).click();
  await page.getByLabel("Select Call the dentist", { exact: true }).check();
  await page.getByLabel("Select Buy printer paper", { exact: true }).check();
  await page
    .getByRole("button", { name: "Edit selected", exact: true })
    .click();
  await page.getByLabel("Change due date", { exact: true }).check();
  await page.getByLabel("Bulk due date").fill("2026-09-18");
  await page.getByLabel("Bulk priority").selectOption("2");
  await page
    .getByRole("button", { name: "Apply changes", exact: true })
    .click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  const records = await page.evaluate(
    async () => (await (await fetch("/api/v1/tasks")).json()).items,
  );
  const pair = records.filter((t) =>
    ["Call the dentist", "Buy printer paper"].includes(t.title),
  );
  if (
    pair.length !== 2 ||
    pair.some((t) => t.due_date !== "2026-09-18" || t.priority !== 2)
  )
    throw new Error("Bulk edits were not saved");
  await page
    .getByRole("button", { name: "Edit Call the dentist", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Home planning", exact: true })
    .click();
  await expect(page.getByLabel("Title", { exact: true })).toHaveValue(
    "Home planning",
  );
  const blocked = await uiAction("ui_calendar", { date: "2026-09-18" });
  if (blocked.status !== "failed")
    throw new Error("Open note editor did not block navigation");
  await page.getByRole("button", { name: "Close note", exact: true }).click();
  const selected = await uiAction("ui_select", {
    task_ids: pair.map((t) => t.id),
  });
  if (
    selected.status !== "displayed" ||
    selected.screen.selected_task_ids.length !== 2
  )
    throw new Error("Selection acknowledgement lost context");
  const notes = await page.evaluate(
    async () => (await (await fetch("/api/v1/notes")).json()).items,
  );
  const opened = await uiAction("ui_show", {
    view: "notes",
    entity_id: notes[0].id,
  });
  if (
    opened.status !== "displayed" ||
    opened.screen.selected_note_id !== notes[0].id
  )
    throw new Error("Note opening not acknowledged");
  await page.getByRole("button", { name: "Close note", exact: true }).click();
  await page.getByLabel("Search", { exact: true }).fill("dentist");
  await expect(page.locator(".note-card")).toHaveCount(1);
  await page
    .getByRole("button", { name: "Search by meaning", exact: true })
    .click();
  await expect(page.locator(".note-card")).toHaveCount(1);
  await expect(
    page.getByRole("status").filter({ hasText: "Loading notes" }),
  ).toHaveCount(0);
  let accidentalMeaningQueries = 0;
  page.on("request", (request) => {
    if (request.url().includes("/api/v1/notes/search?"))
      accidentalMeaningQueries++;
  });
  await page.getByLabel("Search", { exact: true }).fill("printer");
  await expect(page.locator(".note-card")).toHaveCount(1);
  await expect(
    page.getByRole("status").filter({ hasText: "Loading notes" }),
  ).toHaveCount(0);
  if (accidentalMeaningQueries)
    throw new Error("Typing unexpectedly requested a paid semantic query");
  await page.getByLabel("Search", { exact: true }).fill("");
  await shot("notes-list-desktop.png");
  await page.getByRole("button", { name: "Settings", exact: true }).click();
  await expect(
    page.getByText(
      "Cost tracking and spending limits are off during development.",
      { exact: false },
    ),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Save limit", exact: true }),
  ).toHaveCount(0);
  if (errors.length) throw new Error(errors.join("\n"));
  console.log(
    "Notes authoring/extraction/linking/search, mobile layouts, bulk changes, disabled cost UI and CopilotKit controls passed.",
  );
} finally {
  await browser.close();
}
