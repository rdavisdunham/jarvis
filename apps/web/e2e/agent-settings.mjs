import { chromium, expect } from "@playwright/test";
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 1050 } });
const errors = [];
page.on("pageerror", e => errors.push(e.message));
const boot = () => page.evaluate(async () => (await fetch("/api/v1/bootstrap")).json());
try {
  await page.goto(process.env.JARVIS_AGENT_TEST_URL);
  await page.getByLabel("Pairing code").fill("agent-fixture");
  await page.locator(".login-card button.primary").click();
  if (page.viewportSize().width < 768) await page.getByRole("button", { name: "Open navigation", exact: true }).click();
  await page.getByRole("button", { name: "Settings", exact: true }).click();
  const selector = page.getByLabel("Backend model", { exact: true });
  await expect(selector).toHaveValue("openai");
  await selector.selectOption("gemini");
  await expect.poll(async () => (await boot()).agent_provider).toBe("gemini");
  await expect(selector).toBeEnabled();
  await page.reload();
  if (page.viewportSize().width < 768) await page.getByRole("button", { name: "Open navigation", exact: true }).click();
  await page.getByRole("button", { name: "Settings", exact: true }).click();
  await expect(selector).toHaveValue("gemini");
  await page.setViewportSize({ width: 390, height: 844 });
  const section = page.locator("section").filter({ has: page.getByRole("heading", { name: "Task agent", exact: true }) });
  await section.scrollIntoViewIfNeeded();
  await section.screenshot({ path: "../../.runtime/gemini-settings-mobile.png" });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  // Simulate credentials becoming unavailable after the next server reload.
  await page.route("**/api/v1/bootstrap", async route => {
    const upstream = await route.fetch();
    const data = await upstream.json();
    data.agent_options.find(m => m.provider === "gemini").available = false;
    data.capabilities.chat = false;
    await route.fulfill({ response: upstream, json: data });
  });
  await page.reload();
  if (page.viewportSize().width < 768) await page.getByRole("button", { name: "Open navigation", exact: true }).click();
  await page.getByRole("button", { name: "Settings", exact: true }).click();
  await expect(selector.locator('option[value="gemini"]')).toHaveJSProperty("disabled", true);
  await expect(page.getByText("To try Gemini,", { exact: false })).toBeVisible();
  await selector.selectOption("openai");
  await expect.poll(async () => (await boot()).agent_provider).toBe("openai");
  expect(errors).toEqual([]);
  console.log("Agent Settings: desktop/mobile selection, persistence, unavailable-key guidance, and switching back passed.");
} finally {
  await browser.close();
}
