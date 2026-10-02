import { afterEach, expect, it, vi } from "vitest";
import { asAgent, command } from "./api";

const uuid = /^[0-9a-f-]{36}$/;

function capture() {
  const bodies: { command_id: string }[] = [];
  vi.stubGlobal("fetch", async (_url: string, init: RequestInit) => {
    bodies.push(JSON.parse(String(init.body)));
    return new Response(JSON.stringify({ data: {}, command_id: "x" }), { status: 200 });
  });
  return bodies;
}

afterEach(() => vi.unstubAllGlobals());

it("tags commands issued during an assistant UI action as ui-agent", async () => {
  const bodies = capture();
  await asAgent("work-1:4", async () => {
    await new Promise((r) => setTimeout(r, 0));
    await command("record.update", { record_id: "r" }).send();
  });
  await command("record.update", { record_id: "r" }).send();
  expect(bodies[0].command_id).toMatch(/^ui-agent:work-1:4:[0-9a-f-]{36}$/);
  expect(bodies[1].command_id).toMatch(uuid);
});

it("leaves owner commands plain and restores the previous scope after errors", async () => {
  const bodies = capture();
  await asAgent(undefined, () => command("task.update", {}).send());
  await expect(asAgent("work-2:1", async () => { throw new Error("boom"); })).rejects.toThrow("boom");
  await command("task.update", {}).send();
  expect(bodies.map((b) => b.command_id).every((id) => uuid.test(id))).toBe(true);
  expect(command("task.update", {}).id.length).toBeLessThanOrEqual(100);
  await asAgent("w".repeat(150), async () => expect(command("x", {}).id.length).toBeLessThanOrEqual(100));
});
