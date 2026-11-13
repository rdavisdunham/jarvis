import { chromium, expect } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const base = process.env.JARVIS_FULL_STACK_URL;
if (!base || !/^http:\/\/(127\.0\.0\.1|localhost):\d+$/.test(base)) throw Error("Use the isolated local full-stack fixture URL");
const artifacts = resolve(process.env.ERIDANI_EVAL_ARTIFACT_DIR || "../../artifacts/full-stack");
mkdirSync(artifacts, { recursive: true });
const browser = await chromium.launch();
const page = await (await browser.newContext({ viewport: { width: 1440, height: 1000 } })).newPage();
const errors = [];
page.on("pageerror", error => errors.push(error.message));
async function api(path, body, method = body === undefined ? "GET" : "POST") {
  return page.evaluate(async ({ path, body, method }) => {
    const boot = await (await fetch("/api/v1/bootstrap")).json();
    const response = await fetch("/api/v1" + path, { method, headers: { "Content-Type": "application/json", "X-CSRF-Token": boot.csrf }, ...(body === undefined ? {} : { body: JSON.stringify(body) }) });
    const value = await response.json();
    if (!response.ok) throw Error(`${response.status} ${JSON.stringify(value)}`);
    return value;
  }, { path, body, method });
}
const command = async (tool, args) => (await api("/commands", { command_id: crypto.randomUUID(), tool, arguments: args })).data;
const shot = name => page.screenshot({ path: resolve(artifacts, name + ".png"), fullPage: false });
const evidence = [];
const mark = (scenario, details = {}) => { evidence.push({ scenario, passed: true, ...details }); console.log("ok: " + scenario); };
const view = page.getByRole("region", { name: "Agent run", exact: true });
const readRun = id => api(`/remotea/runs/${id}`);
const settled = async (id, state, reason) => expect.poll(async () => { const r = await readRun(id); return r.state + ":" + r.state_reason; },
  { timeout: 60000, message: `run ${id} to reach ${state}:${reason}` }).toBe(state + ":" + reason);
async function openRun(id) {
  await page.goto(`${base}?view=tasks&tab=all&run=${id}`);
  await expect(view).toBeVisible();
}
try {
  await page.goto(base);
  await page.getByLabel("Pairing code").fill(process.env.JARVIS_FULL_STACK_PAIRING);
  await page.locator(".login-card button.primary").click();
  await expect(page.getByRole("button", { name: "Tasks", exact: true })).toBeVisible();
  const host = (await api("/remotea/status")).hosts.find(h => h.connected);
  expect(host, "the remotea host is connected").toBeTruthy();

  // ---- 1. The route on a task, and the direction check ----
  const task = await command("task.create", { title: "Rename the helper" });
  await api("/remotea/targets", { host_id: host.id, host_target: process.env.JARVIS_FULL_STACK_TARGET, repo: process.env.JARVIS_FULL_STACK_REPO,
    worktree_root: process.env.JARVIS_FULL_STACK_WORKTREES, task_id: task.id });
  await page.goto(`${base}?view=tasks&tab=all`);
  await page.getByText("Rename the helper").first().click();
  const route = page.locator(".task-route");
  await expect(route).toBeVisible();
  // The readiness pass looked at the task by itself: it has a route and a reason before anyone chose one.
  await expect.poll(() => route.locator("select").inputValue(), { timeout: 20000 }).not.toBe("");
  // Choosing the repository made the pass look again: it no longer says none is chosen.
  await expect(route.locator("small")).not.toContainText("no repository is chosen", { timeout: 20000 });
  await shot("01-route-decided");
  await route.locator("select").selectOption("direction");
  await expect(route.locator("small")).toContainText("You chose this.");
  mark("the task shows its route and the owner can change it in one tap");

  const started = page.waitForResponse(r => r.url().endsWith("/remotea/runs") && r.request().method() === "POST");
  await page.locator(".remotea-run-task").getByRole("button", { name: "Run", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "Run with an agent", exact: true });
  await expect(dialog).toBeVisible();
  await shot("02-run-dialog");
  await expect(dialog.getByRole("button", { name: "Start", exact: true })).toBeEnabled({ timeout: 15000 });
  await dialog.getByRole("button", { name: "Start", exact: true }).click();
  const run = (await (await started).json()).id;
  await openRun(run);
  await expect(view.getByText("Direction check").first()).toBeVisible({ timeout: 45000 });
  await expect(view).toContainText("How it reads the task");
  await expect(view).toContainText("Anything to avoid?");
  await view.getByText("How it reads the task").scrollIntoViewIfNeeded();
  await shot("03-direction-check");
  await view.getByLabel("Your answer or a different direction").fill("Avoid the public API.");
  await view.getByRole("button", { name: "Send answer", exact: true }).click();
  await expect.poll(async () => (await readRun(run)).direction?.answer, { timeout: 30000 }).toBe("Avoid the public API.");
  // The revised plan starts by itself, the work is verified and judged; with no pull request the run stops for you.
  await settled(run, "needs_you", "pr_unknown");
  const done = await readRun(run);
  expect(done.history.filter(h => h.event === "plan_approved").map(h => h.actor.kind)).toEqual(["direction_check"]);
  await openRun(run);
  await expect(view).toContainText("pull request");
  await shot("04-direction-finished");
  mark("one answer to the brief starts the revised plan, which runs to a judged result", { run });

  // ---- 2. Auto-start from Settings ----
  await page.goto(`${base}?view=settings&section=remotea`);
  const toggle = page.getByLabel("Start agent-ready tasks by themselves");
  await expect(toggle).toBeVisible({ timeout: 20000 });
  // These switches save first and show the saved value, so click and wait for it (check() would click twice).
  await toggle.click();
  await expect(toggle).toBeChecked();
  const options = page.locator(".auto-start-options");
  await expect(options).toBeVisible();
  await options.locator("fieldset input[type=checkbox]").first().click();
  await expect(options.locator("fieldset input[type=checkbox]").first()).toBeChecked();
  await expect(page.getByText("On for 1 repository.")).toBeVisible();
  await options.scrollIntoViewIfNeeded();
  await shot("05-auto-start-settings");
  // The migrations row's hint keeps a readable width beside nothing (its three inputs sit on their own line).
  const hint = page.locator(".remotea-migration-row .setting-hint").first();
  if (await hint.count()) expect((await hint.boundingBox()).width).toBeGreaterThan(240);
  const allowed = (await api("/remotea/auto-start")).targets;
  expect(allowed.length).toBe(1);
  const labelled = await command("task.create", { title: "Add a greeting", tags: ["ai-assignable"] });
  await command("task.update", { task_id: labelled.id, expected_revision: labelled.revision, execution_target_id: allowed[0] });
  // Nothing else is done by hand: the pass routes it, auto-start starts it, the plan goes ahead, the judge reads it.
  let autoRun;
  await expect.poll(async () => { autoRun = (await api(`/remotea/runs?task_id=${labelled.id}`)).items?.[0] ?? (await api(`/remotea/runs?task_id=${labelled.id}`))[0]; return autoRun?.id ?? null; },
    { timeout: 60000, message: "auto-start to start a run for the labelled task" }).not.toBeNull();
  await settled(autoRun.id, "needs_you", "pr_unknown");
  const auto = await readRun(autoRun.id);
  expect(auto.auto_started).toBe(true);
  expect(auto.history.filter(h => h.event === "plan_approved").map(h => h.actor.kind)).toEqual(["auto_start"]);
  await openRun(autoRun.id);
  await expect(view).toContainText("Started by itself");
  await shot("06-auto-started-run");
  await page.goto(`${base}?view=tasks&tab=all`);
  await page.getByText("Add a greeting").first().click();
  await expect(page.locator(".task-route select")).toHaveValue("auto");
  mark("a labelled task runs from creation to a judged result with no owner action", { run: autoRun.id });
  // ---- 4. Talk it through: a run that opens as a conversation and plans when the owner says so ----
  const vague = await command("task.create", { title: "Make the greeting friendlier" });
  await command("task.update", { task_id: vague.id, expected_revision: vague.revision, execution_target_id: allowed[0] });
  await page.goto(`${base}?view=tasks&tab=all`);
  await page.getByText("Make the greeting friendlier").first().click();
  const talkStarted = page.waitForResponse(r => r.url().endsWith("/remotea/runs") && r.request().method() === "POST");
  await page.locator(".remotea-run-task").getByRole("button", { name: "Run", exact: true }).click();
  await expect(dialog).toBeVisible();
  await dialog.getByRole("button", { name: "Talk it through", exact: true }).click();
  await expect(dialog).toContainText("talks it through with you");
  await shot("07-run-dialog-talk");
  await expect(dialog.getByRole("button", { name: "Start", exact: true })).toBeEnabled({ timeout: 15000 });
  await dialog.getByRole("button", { name: "Start", exact: true }).click();
  const talkRequest = await talkStarted;
  expect(talkRequest.request().postDataJSON().brainstorm).toBe(true);
  const talk = (await talkRequest.json()).id;
  await settled(talk, "needs_you", "brainstorm");
  await openRun(talk);
  const card = view.getByRole("region", { name: "Talking it through", exact: true }).first();
  await expect(card).toContainText("what do you already have in mind", { timeout: 30000 });
  await card.scrollIntoViewIfNeeded();
  await shot("08-talking-it-through");
  await card.getByLabel("Your reply").fill("Make it warmer, option B.");
  await card.getByRole("button", { name: "Send reply", exact: true }).click();
  await expect(card).toContainText("Shall I plan it?", { timeout: 45000 });
  expect((await readRun(talk)).brainstorm).toMatchObject({ open: true, turns: 2 });
  await card.getByRole("button", { name: "Plan it", exact: true }).click();
  await settled(talk, "needs_you", "plan_approval");
  await openRun(talk);
  await expect(view.getByText("Final plan").first()).toBeVisible({ timeout: 30000 });
  await expect(view.getByRole("region", { name: "Talking it through", exact: true })).toHaveCount(0);
  expect((await readRun(talk)).brainstorm).toMatchObject({ open: false, ended_by: "owner" });
  await shot("09-plan-after-talking");
  mark("a run opens as a conversation, takes a reply, and plans when the owner says so", { run: talk });
  writeFileSync(resolve(artifacts, "evidence.json"), JSON.stringify(evidence, null, 2));
  console.log(`Full stack: ${evidence.length} scenarios passed.`);
} finally {
  await shot("99-last").catch(() => {});
  await browser.close();
}
if (errors.length) throw Error("Page errors: " + errors.join(" | "));
