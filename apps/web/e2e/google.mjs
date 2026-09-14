import { chromium, expect } from "@playwright/test";
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 1050 } });
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
const base = process.env.JARVIS_GOOGLE_TEST_URL;
const request = (path, body = {}) =>
  page.evaluate(
    async ({ path, body }) => {
      const boot = await (await fetch("/api/v1/bootstrap")).json();
      const r = await fetch("/api/v1" + path, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-CSRF-Token": boot.csrf,
        },
        body: JSON.stringify(body),
      });
      if (!r.ok) throw new Error("Fixture request failed: " + path);
      return r.json();
    },
    { path, body },
  );
const ui = (name, args) => request("/__test_ui", { name, arguments: args });
const shot = (name) =>
  page.screenshot({
    path: new URL("../../../.runtime/" + name, import.meta.url).pathname,
    fullPage: true,
  });
try {
  // Exercise the real state/cookie redirect flow with Google's page/token response replaced.
  await page.route(
    "https://accounts.google.com/o/oauth2/v2/auth?**",
    async (route) => {
      const url = new URL(route.request().url());
      const code = JSON.stringify({
        sub: "fixture-subject",
        email: "owner@example.test",
        email_verified: true,
        nonce: url.searchParams.get("nonce"),
      });
      const callback = new URL(url.searchParams.get("redirect_uri"));
      callback.searchParams.set("state", url.searchParams.get("state"));
      callback.searchParams.set("code", code);
      await route.fulfill({
        status: 302,
        headers: { location: callback.href },
        body: "",
      });
    },
  );
  await page.goto(base);
  await page
    .getByRole("button", { name: "Sign in with Google", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Google", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("owner@example.test", { exact: true }),
  ).toBeVisible();
  await request("/__test_google_sync");
  await expect(page.getByLabel("Use calendar Personal")).toBeChecked();
  await expect(page.getByLabel("Use calendar Work calendar")).not.toBeChecked();
  await page.getByLabel("Use calendar Work calendar").check();
  await expect(page.getByLabel("Use calendar Work calendar")).toBeEnabled();
  await request("/__test_google_sync");
  await expect(page.getByLabel("Use calendar Work calendar")).toBeChecked();
  await page
    .getByRole("button", { name: "Enable Calendar editing", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Reconnect editing", exact: true }),
  ).toBeVisible();
  await request("/__test_google_sync");
  await shot("google-settings-desktop.png");
  const shown = await ui("ui_calendar", { date: "2026-09-18" });
  if (
    shown.status !== "displayed" ||
    shown.screen.calendar_date !== "2026-09-18"
  )
    throw new Error("Calendar UI was not acknowledged");
  await expect(
    page.locator(".agenda-row").filter({ hasText: "Dentist" }),
  ).toHaveCount(1);
  await expect(
    page.locator(".agenda-row").filter({ hasText: "Project review" }),
  ).toHaveCount(1);
  await page.getByText("Find an open time", { exact: true }).click();
  await page
    .getByRole("button", { name: "Check availability", exact: true })
    .click();
  await expect(page.locator(".available-times")).toContainText("10:00");
  // Exercise cached details before the fresh response: their raw Google date
  // objects must never be passed to the normalized time formatter.
  await page.route("**/api/v1/calendar/event-detail", async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 400));
    await route.continue();
  });
  await page.locator(".agenda-row").filter({ hasText: "Dentist" }).click();
  await expect(page.getByRole("dialog")).toBeVisible();

  await expect(
    page.getByText("Loading current event…", { exact: true }),
  ).toHaveCount(0, { timeout: 15000 });
  await expect(page.getByRole("dialog")).toContainText("Main office");
  await expect(
    page.getByRole("link", { name: "Join meeting", exact: true }),
  ).toBeVisible();
  await expect(
    page.locator("details").filter({ hasText: "Guests (1)" }),
  ).toHaveAttribute("open", "");
  await expect(page.getByRole("dialog")).toContainText("Fixture guest");
  await expect(
    page.getByRole("link", { name: "Agenda notes", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: "Open in Google Calendar" }),
  ).toHaveAttribute("rel", "noopener noreferrer");
  await page.unroute("**/api/v1/calendar/event-detail");
  const blocked = await ui("ui_show", { view: "notes" });
  if (blocked.status !== "failed")
    throw new Error("Open calendar event did not guard navigation");
  await page.getByRole("button", { name: "Close calendar event" }).click();
  const items = await page.evaluate(
    async () =>
      (
        await (
          await fetch("/api/v1/calendar?start=2026-09-18&end=2026-09-19")
        ).json()
      ).items,
  );
  const highlighted = await ui("ui_calendar", {
    date: "2026-09-18",
    entity_id: items.find((e) => e.title === "Dentist").entity_id,
  });
  if (highlighted.status !== "displayed")
    throw new Error("Event highlight failed");
  await expect(page.locator(".agenda-row.record-highlight")).toContainText(
    "Dentist",
  );
  await page.setViewportSize({ width: 390, height: 844 });
  await shot("google-calendar-mobile.png");
  if (
    await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)
  )
    throw new Error("Calendar overflows on mobile");
  // Reproduce the reported mobile jump across the real 30-second background refresh.
  await page.getByRole("button", { name: "Month", exact: true }).click();
  await request("/__test_google_dense");
  await expect(page.locator(".agenda-row")).toHaveCount(24);
  await page.locator(".agenda-row").last().scrollIntoViewIfNeeded();
  const scrollBefore = await page.evaluate(() => window.scrollY);
  if (scrollBefore < 300)
    throw new Error("Mobile scroll regression did not scroll into the agenda");
  const agendaNodes = await page.locator(".agenda-row").count();
  let calendarRequests = 0;
  page.on("request", (r) => {
    if (r.url().includes("/api/v1/calendar?")) calendarRequests++;
  });
  await page.route("**/api/v1/calendar?**", async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 500));
    await route.continue();
  });
  await page.waitForTimeout(32000);
  const scrollAfter = await page.evaluate(() => window.scrollY);
  if (
    Math.abs(scrollAfter - scrollBefore) > 3 ||
    (await page.locator(".agenda-row").count()) !== agendaNodes ||
    !calendarRequests
  )
    throw new Error(
      "Background calendar refresh moved the mobile agenda: " +
        scrollBefore +
        " -> " +
        scrollAfter,
    );
  await page.unroute("**/api/v1/calendar?**");
  await page.getByRole("button", { name: "Week", exact: true }).click();
  await expect(page.getByLabel("Week dates")).toBeVisible();
  await expect(page.locator(".calendar-day-agenda")).toHaveCount(7);
  await shot("google-calendar-week-mobile.png");
  await page.getByRole("button", { name: "Month", exact: true }).click();
  await page.locator('.calendar-number[data-day="2026-09-18"]').dblclick();
  await expect(
    page.getByRole("button", { name: "Day", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByLabel("Month dates")).toHaveCount(0);
  await expect(page.locator(".calendar-day-agenda")).toHaveCount(1);
  await shot("google-calendar-day-mobile.png");
  await page.getByRole("button", { name: "Next day", exact: true }).click();
  await expect(page.getByLabel("Agenda for 2026-09-19")).toBeVisible();
  await ui("ui_calendar", { date: "2026-09-18", calendar_view: "day" });
  await page.getByRole("button", { name: "New event", exact: true }).click();
  await expect(page.getByLabel("Event title")).toBeVisible();
  await page.getByLabel("Event title").fill("Mobile appointment");
  await page.getByLabel("Event start").fill("2026-09-18T15:00");
  await page.getByLabel("Event end").fill("2026-09-18T15:45");
  await page.getByLabel("Event location").fill("Home office");
  await page.getByLabel("Google copy").selectOption({ label: "Personal" });
  await shot("google-calendar-editor-mobile.png");
  if (
    await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)
  )
    throw new Error("Event editor overflows mobile");
  // A committed command with a lost response retries the same receipt and Google event ID.
  let lost = false;
  await page.route("**/api/v1/commands", async (route) => {
    const body = route.request().postDataJSON();
    if (!lost && body.tool === "planning.create") {
      lost = true;
      await route.fetch();
      await route.abort("connectionreset");
    } else await route.continue();
  });
  await page.getByRole("button", { name: "Save event", exact: true }).click();
  await expect(
    page.getByText(
      "The request did not finish. Retry the same change to check its saved receipt.",
      { exact: true },
    ),
  ).toBeVisible();
  await page.getByRole("button", { name: "Save event", exact: true }).click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await page.unroute("**/api/v1/commands");
  await expect(
    page.locator(".agenda-row").filter({ hasText: "Mobile appointment" }),
  ).toHaveCount(1);
  await page
    .locator(".agenda-row")
    .filter({ hasText: "Mobile appointment" })
    .click();
  await page
    .getByRole("button", { name: "Edit calendar item", exact: true })
    .click();
  await page.getByLabel("Event title").fill("Mobile appointment edited");
  await page.getByLabel("Event end").fill("2026-09-18T16:00");
  await page.getByRole("button", { name: "Save event", exact: true }).click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await expect(
    page
      .locator(".agenda-row")
      .filter({ hasText: "Mobile appointment edited" }),
  ).toHaveCount(1);
  await page
    .locator(".agenda-row")
    .filter({ hasText: "Mobile appointment edited" })
    .click();
  await page
    .getByRole("button", { name: "Edit calendar item", exact: true })
    .click();
  await page.getByRole("button", { name: "Cancel event", exact: true }).click();
  await page
    .getByRole("button", { name: "Confirm cancellation", exact: true })
    .click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await expect(
    page.locator(".agenda-row").filter({ hasText: "Mobile appointment" }),
  ).toHaveCount(0);
  await page.getByText("Recent calendar changes", { exact: true }).click();
  await expect(
    page.getByText("Confirmed by Google", { exact: true }),
  ).toHaveCount(3);
  // Linear uses the real app connection/command surfaces with a synthetic GraphQL provider.
  await ui("ui_workspace", {
    view: "settings",
    settings_section: "integrations",
  });
  await page.getByLabel("Linear API key").fill("fixture-key-browser");
  await page
    .getByRole("button", { name: "Connect Linear", exact: true })
    .click();
  await expect(page.getByText("Teams to sync", { exact: true })).toBeVisible();
  await page.getByLabel("Team", { exact: true }).check();
  await page
    .getByRole("button", { name: "Save sync scope", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Save sync scope", exact: true }),
  ).toBeEnabled();
  await ui("ui_show", { view: "all" });
  await page
    .getByRole("button", { name: "Edit Shared task", exact: true })
    .click();
  await expect(page.getByRole("dialog")).toContainText("ERI-1");
  await request("/__test_linear_change");
  await page.getByRole("button", { name: "Change Task", exact: true }).click();
  await page
    .getByRole("dialog")
    .getByLabel("Task", { exact: true })
    .fill("Local conflict version");
  await page.getByLabel("Task", { exact: true }).press("Enter");
  await expect(
    page.getByRole("button", { name: "Change Task", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Close task details" }).click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await page
    .getByRole("button", { name: "Edit Local conflict version", exact: true })
    .click();
  await expect(page.locator(".linear-task")).toContainText("conflict");
  await page
    .getByRole("button", { name: "Review Linear copy", exact: true })
    .click();
  await expect(page.locator(".sync-comparison")).toContainText(
    "Linear conflict version",
  );
  await shot("linear-conflict-mobile.png");
  await page
    .getByRole("button", { name: "Use Linear version", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Change Task", exact: true }),
  ).toContainText("Linear conflict version");
  await page.getByRole("button", { name: "Close task details" }).click();
  await page
    .getByRole("button", { name: "Edit Linear conflict version", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Reserve time for this task", exact: true })
    .click();
  await page.getByLabel("Event start").fill("2026-09-18T17:00");
  await page.getByLabel("Event end").fill("2026-09-18T18:00");
  await shot("planning-block-mobile.png");
  await page.getByRole("button", { name: "Save event", exact: true }).click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await ui("ui_calendar", { date: "2026-09-18", calendar_view: "day" });
  await expect(
    page
      .locator(".agenda-row")
      .filter({ hasText: "Work on Linear conflict version" }),
  ).toHaveCount(1);
  await ui("ui_show", { view: "all" });
  await page.getByRole("button", { name: "New task", exact: true }).click();
  await page
    .getByRole("dialog")
    .getByLabel("Task", { exact: true })
    .fill("Publish browser task");
  await page.getByRole("button", { name: "Save task", exact: true }).click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await page
    .getByRole("button", { name: "Edit Publish browser task", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Publish to Linear", exact: true })
    .click();
  await expect(page.locator(".linear-task")).toContainText("synced");
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "Close task details", exact: true })
    .click();
  await ui("ui_workspace", {
    view: "settings",
    settings_section: "integrations",
  });
  await shot("google-settings-mobile.png");
  if (
    await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)
  )
    throw new Error("Google settings overflow on mobile");
  await page
    .getByRole("button", { name: "Disconnect Calendar", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Disconnect now", exact: true })
    .click();
  await expect(
    page.getByText("Calendar disconnected. Google sign-in stays linked.", {
      exact: true,
    }),
  ).toBeVisible();
  await expect(page.getByLabel("Use calendar Personal")).toHaveCount(0);
  await Promise.all([
    page.waitForURL((url) => url.searchParams.get("google") === "connected"),
    page.getByRole("button", { name: "Connect Calendar", exact: true }).click(),
  ]);
  await expect(
    page.getByRole("button", { name: "Reconnect Calendar", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Google", exact: true }),
  ).toBeVisible();
  await request("/__test_google_sync");
  await expect(page.getByLabel("Use calendar Personal")).toBeChecked();
  await page
    .getByRole("button", { name: "Unlink Google sign-in", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Unlink account", exact: true })
    .click();
  await expect(page.getByLabel("Pairing code")).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Sign in with Google", exact: true }),
  ).toHaveCount(0);
  if (errors.length) throw new Error(errors.join("\n"));
  console.log(
    "Google consent/details, mobile views, 30-second scroll stability, local event publication/CRUD, lost-response retry, Linear import/publish/conflict review, task work blocks, CopilotKit and disconnect/unlink passed.",
  );
} catch (error) {
  console.log("Browser errors:", errors);
  console.log(
    "Google dialog at failure:",
    await page
      .getByRole("dialog")
      .textContent()
      .catch(() => "none"),
  );
  await shot("google-failure.png");
  throw error;
} finally {
  await browser.close();
}
