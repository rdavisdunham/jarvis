import { chromium, expect } from "@playwright/test";
import { writeFileSync } from "node:fs";
const browser = await chromium.launch();
const owner = await browser.newPage({
  viewport: { width: 1440, height: 1000 },
});
const guest = await browser.newPage({ viewport: { width: 390, height: 844 } });
const errors = [];
for (const p of [owner, guest])
  p.on("pageerror", (e) => errors.push(e.message));
const base = process.env.JARVIS_PLANNER_TEST_URL;
async function request(p, path, body) {
  return p.evaluate(
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
}
const cmd = async (p, tool, args) =>
  (
    await request(p, "/commands", {
      command_id: crypto.randomUUID(),
      tool,
      arguments: args,
    })
  ).data;
const ui = async (p, name, args = {}) => {
  const r = await request(p, "/__test_ui", { name, arguments: args });
  if (r.status !== "displayed") throw new Error(JSON.stringify(r));
  return r;
};
try {
  await owner.goto(base);
  await owner.getByLabel("Pairing code").fill("planner-fixture");
  await owner.locator(".login-card button.primary").click();
  await expect(
    owner.getByRole("heading", { name: "Tasks", exact: true }),
  ).toBeVisible();
  await cmd(owner, "task.create", { title: "Private owner task" });
  await ui(owner, "ui_workspace", {
    view: "settings",
    settings_section: "sharing",
  });
  await expect(
    owner.getByRole("heading", { name: "People & sharing" }),
  ).toBeVisible();
  await owner.getByLabel("Shared workspace name").fill("Household");
  await owner
    .getByRole("button", { name: "Create workspace", exact: true })
    .click();
  await expect(owner.getByLabel("Manage sharing")).not.toHaveValue("");
  await owner.getByLabel("Invite Google email").fill("guest@example.test");
  await owner
    .getByRole("button", { name: "Create invitation", exact: true })
    .click();
  await expect(
    owner.locator(".sharing-row").filter({ hasText: "guest@example.test" }),
  ).toBeVisible();
  const w = (await request(owner, "/accounts")).workspaces[0];
  await guest.goto(base);
  await guest.request.post(base + "/api/v1/auth/__test_guest_login");
  await guest.reload();
  await expect(
    guest.getByRole("heading", { name: "Tasks", exact: true }),
  ).toBeVisible();
  if ((await request(guest, "/tasks")).items.length)
    throw new Error("Private tasks leaked");
  await ui(guest, "ui_workspace", {
    view: "settings",
    settings_section: "sharing",
  });
  await guest.getByRole("button", { name: "Accept invitation" }).click();
  await expect(
    guest.getByText("Invitation accepted.", { exact: false }),
  ).toBeVisible();
  // Switch through the visible navigation selector, including on the phone.
  const guestSelect = guest.getByLabel("Active workspace");
  if (!(await guestSelect.isVisible()))
    await guest.getByRole("button", { name: "Open navigation" }).click();
  await guestSelect.selectOption(w.id);
  await expect(
    guest.getByRole("heading", { name: "Tasks", exact: true }),
  ).toBeVisible();
  await owner.getByLabel("Active workspace").selectOption(w.id);
  await expect(
    owner.getByRole("heading", { name: "Tasks", exact: true }),
  ).toBeVisible();
  const shared = await cmd(owner, "task.create", {
    title: "Shared household task",
    notes: "Shared details",
  });
  await ui(guest, "ui_show", { view: "all", entity_id: shared.id });
  await guest.getByRole("button", { name: "Change Task", exact: true }).click();
  await guest
    .getByLabel("Task", { exact: true })
    .fill("Guest updated shared task");
  await guest.getByLabel("Task", { exact: true }).press("Enter");
  await expect(
    guest.getByRole("button", { name: "Change Task", exact: true }),
  ).toContainText("Guest updated");
  if (
    (await request(owner, "/tasks/" + shared.id)).title !==
    "Guest updated shared task"
  )
    throw new Error("Shared edit lost");
  await guest.screenshot({
    path: "../../.runtime/accounts-card-mobile.png",
    fullPage: true,
  });
  await ui(owner, "ui_workspace", {
    view: "settings",
    settings_section: "sharing",
  });
  await owner.getByLabel("Manage sharing").selectOption(w.id);
  await owner.getByRole("button", { name: "Revoke access" }).click();
  // The open SSE stream clears the revoked card and returns this guest to Personal.
  await expect(guest.getByRole("dialog")).toHaveCount(0, { timeout: 10000 });
  await expect(
    guest.getByRole("heading", { name: "Tasks", exact: true }),
  ).toBeVisible();
  await expect(guest.locator(".task-row")).toHaveCount(0);
  if ((await request(guest, "/accounts")).active_workspace_id !== null)
    throw new Error("Revoked workspace still active");
  await owner.screenshot({
    path: "../../.runtime/accounts-sharing-desktop.png",
    fullPage: true,
  });
  if (
    await guest.evaluate(
      () => document.documentElement.scrollWidth > innerWidth + 1,
    )
  )
    throw new Error("Mobile overflow");
  if (errors.length) throw new Error(errors.join(";"));
  writeFileSync(
    "../../.runtime/accounts-browser.json",
    JSON.stringify({ passed: true, errors }, null, 2),
  );
  console.log(
    "Two-account browser acceptance passed: invitations, private isolation, shared inline edit, mobile workspace switching and live revocation.",
  );
} catch (e) {
  await guest.screenshot({
    path: "../../.runtime/accounts-failure-guest.png",
    fullPage: true,
  });
  await owner.screenshot({
    path: "../../.runtime/accounts-failure-owner.png",
    fullPage: true,
  });
  throw e;
} finally {
  await browser.close();
}
