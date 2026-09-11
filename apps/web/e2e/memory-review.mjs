import { chromium, expect } from "@playwright/test";
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
const errors = [];
page.on("pageerror", error => errors.push(error.message));
try {
  await page.goto(process.env.JARVIS_REVIEW_TEST_URL);
  await page.getByLabel("Pairing code").fill("memory-review-fixture");
  await page.locator(".login-card button.primary").click();
  await page.getByRole("button", { name: "Memory", exact: true }).click();
  await expect(page.locator(".memory-review")).toHaveCount(1);
  await expect(page.locator(".memory-review")).toContainText("Haze");
  await expect(page.locator(".memory-review")).toContainText("Hayes");
  await page.setViewportSize({ width: 390, height: 844 });
  const menu = page.getByRole("button", { name: /close.*menu/i });
  if (await menu.count()) await menu.click();
  await expect(page.getByLabel("Correct fact")).toBeVisible();
  if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)) {
    throw new Error("Mobile layout overflows");
  }
  await page.screenshot({ path: new URL("../../../.runtime/memory-review-mobile.png", import.meta.url).pathname, fullPage: true });
  await page.getByLabel("Correct fact").fill("The user's cat is named Hayes.");
  await page.getByRole("button", { name: "Save corrected memory" }).click();
  await expect(page.locator(".memory-review")).toHaveCount(0);
  await expect(page.locator(".memory")).toHaveCount(1);
  await expect(page.locator(".memory")).toContainText("The user's cat is named Hayes.");
  const data = await (await page.request.get(process.env.JARVIS_REVIEW_TEST_URL + "/api/v1/memory")).json();
  if (data.reviews.length || data.items.length !== 1 || "legacy" in data) throw new Error("Unexpected saved memory state");
  await page.getByRole("button", { name: "Review now", exact: true }).click();
  await expect(page.getByRole("button", { name: "Review now", exact: true })).toBeDisabled();
  if (errors.length) throw new Error(errors.join("; "));
  console.log(JSON.stringify({result: "passed", mobile: true, correction: true, reviewNow: true, legacyRemoved: true}));
} finally { await browser.close(); }
