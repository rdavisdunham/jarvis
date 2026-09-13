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
  await page.locator(".agenda-row").filter({ hasText: "Dentist" }).click();
  await expect(page.getByRole("dialog")).toContainText("Main office");
  await expect(
    page.getByRole("link", { name: "Open in Google Calendar" }),
  ).toHaveAttribute("rel", "noopener noreferrer");
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
  await page.locator('[data-day="2026-09-18"]').dblclick();
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
  await shot("google-calendar-editor-mobile.png");
  if (
    await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)
  )
    throw new Error("Event editor overflows mobile");
  // A committed command with a lost response retries the same receipt and Google event ID.
  let lost = false;
  await page.route("**/api/v1/commands", async (route) => {
    const body = route.request().postDataJSON();
    if (!lost && body.tool === "calendar.create") {
      lost = true;
      await route.fetch();
      await route.abort("connectionreset");
    } else await route.continue();
  });
  await page.getByRole("button", { name: "Create event", exact: true }).click();
  await expect(
    page.getByText(
      "The request did not finish. Retry the same change to check its saved receipt.",
      { exact: true },
    ),
  ).toBeVisible();
  await page.getByRole("button", { name: "Create event", exact: true }).click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await page.unroute("**/api/v1/commands");
  await expect(
    page.locator(".agenda-row").filter({ hasText: "Mobile appointment" }),
  ).toHaveCount(1);
  await page
    .locator(".agenda-row")
    .filter({ hasText: "Mobile appointment" })
    .click();
  await page.getByRole("button", { name: "Edit event", exact: true }).click();
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
  await page.getByRole("button", { name: "Delete event", exact: true }).click();
  await page
    .getByRole("button", { name: "Confirm delete", exact: true })
    .click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await expect(
    page.locator(".agenda-row").filter({ hasText: "Mobile appointment" }),
  ).toHaveCount(0);
  await page.getByText("Recent calendar changes", { exact: true }).click();
  await expect(
    page.getByText("Confirmed by Google", { exact: true }),
  ).toHaveCount(3);
  await ui("ui_show", { view: "settings" });
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
    "Google consent/read/write, mobile views, 30-second scroll stability, lost-response save retry, event CRUD, CopilotKit and disconnect/unlink passed.",
  );
} finally {
  await browser.close();
}
