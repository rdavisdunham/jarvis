import { chromium, expect } from "@playwright/test";
import { readFileSync, writeFileSync } from "node:fs";
const root = new URL("../../../", import.meta.url);
const token = readFileSync(
  new URL(".runtime/pairing-code", root),
  "utf8",
).trim();
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
const evidence = { result: "failed", errors: [], reminder: null };
page.on("pageerror", (e) => evidence.errors.push(e.message));
try {
  await page.goto("https://davispc.tail957c2.ts.net:9443");
  await page.getByLabel("Pairing code").fill(token);
  await page.locator(".login-card button.primary").click();
  await page
    .getByRole("button", { name: "Start a private session", exact: true })
    .click();
  await expect(page.locator(".companion").getByRole("combobox")).toHaveCount(0);
  await expect(
    page.getByText("Your records live at home", { exact: true }),
  ).toHaveCount(0);
  await page
    .getByRole("button", { name: "Voice settings", exact: true })
    .click();
  await page.getByLabel("Voice provider", { exact: true }).selectOption("live");
  await page.getByLabel("Voice", { exact: true }).selectOption("willow");
  await page
    .getByLabel("Voice provider", { exact: true })
    .selectOption("realtime");
  await expect(page.getByLabel("Voice", { exact: true })).toHaveValue("marin");
  await expect(
    page.getByLabel("Voice", { exact: true }).locator('option[value="willow"]'),
  ).toHaveCount(0);
  await page.getByLabel("Voice provider", { exact: true }).selectOption("live");
  await expect(page.getByLabel("Voice", { exact: true })).toHaveValue("willow");
  evidence.providerVoices = true;
  evidence.reminder = await page.evaluate(async () => {
    const boot = await (await fetch("/api/v1/bootstrap")).json();
    const response = await fetch("/api/v1/commands", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRF-Token": boot.csrf,
      },
      body: JSON.stringify({
        command_id: crypto.randomUUID(),
        tool: "schedule.create",
        arguments: {
          title: "Synthetic polish check amber lake",
          when: new Date(Date.now() + 3600000).toISOString(),
          timezone: "America/Chicago",
        },
      }),
    });
    if (!response.ok) throw new Error("Reminder fixture could not save");
    return (await response.json()).data;
  });
  await page
    .getByLabel("Message Eridani")
    .fill(
      "Show me the reminder named Synthetic polish check amber lake. Open it on this site.",
    );
  await page.getByRole("button", { name: "Send message", exact: true }).click();
  const row = page.locator("#record-" + evidence.reminder.id);
  await expect(row).toHaveClass(/record-highlight/, { timeout: 45000 });
  evidence.agentNavigation = true;
  await page.screenshot({
    path: new URL(".runtime/reminder-highlight.png", root).pathname,
    fullPage: true,
  });
  await row
    .getByRole("button", {
      name: "Complete Synthetic polish check amber lake",
      exact: true,
    })
    .click();
  await page.getByRole("button", { name: "Completed", exact: true }).click();
  await expect(row).toContainText("Completed");
  evidence.completedHistory = true;
  const stored = await page.evaluate(
    async (id) => await (await fetch("/api/v1/schedules/" + id)).json(),
    evidence.reminder.id,
  );
  evidence.completionRetained =
    stored.status === "completed" && !!stored.completed_at;
  await page.getByRole("button", { name: "New chat", exact: true }).click();
  await expect(page.locator(".message")).toHaveCount(0);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Open Eridani", exact: true }).click();
  await expect(page.locator(".companion")).toBeVisible();
  evidence.mobileOverflow = await page.evaluate(
    () => document.documentElement.scrollWidth > innerWidth,
  );
  await page.screenshot({
    path: new URL(".runtime/minimal-chat-mobile.png", root).pathname,
    fullPage: true,
  });
  evidence.result =
    evidence.agentNavigation &&
    evidence.providerVoices &&
    evidence.completionRetained &&
    !evidence.mobileOverflow &&
    !evidence.errors.length
      ? "passed"
      : "failed";
} catch (error) {
  evidence.failure = error.message.split("\n")[0];
} finally {
  await browser.close();
  writeFileSync(
    new URL(".runtime/polish-evidence.json", root),
    JSON.stringify(evidence, null, 2),
  );
  console.log(JSON.stringify(evidence, null, 2));
}
if (evidence.result !== "passed") process.exitCode = 1;
