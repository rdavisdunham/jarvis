// @vitest-environment jsdom
import { beforeEach, afterEach, expect, it, vi } from "vitest";
beforeEach(() => { localStorage.clear(); vi.resetModules(); });
afterEach(() => vi.unstubAllGlobals());
const response = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status });
it("persists an uncertain action ID through manual retry and reload", async () => {
  let client = await import("./api");
  const sent: Record<string, unknown>[] = [];
  vi.stubGlobal("fetch", async (_url: string, init: RequestInit) => { sent.push(JSON.parse(String(init.body))); throw new TypeError("lost response"); });
  await expect(client.postAction("/remotea/runs/r/message", { text: "Do it" }, { retries: 0 })).rejects.toMatchObject({ code: "NETWORK" });
  expect(client.actionReceipts().at(-1)?.status).toBe("uncertain");
  vi.resetModules(); client = await import("./api");
  vi.stubGlobal("fetch", async (_url: string, init: RequestInit) => { sent.push(JSON.parse(String(init.body))); return response({ receipt: { status: "saved" } }); });
  await client.postAction("/remotea/runs/r/message", { text: "Do it" }, { retries: 0 });
  expect(sent[1].action_id).toBe(sent[0].action_id);
  expect(client.actionReceipts().at(-1)?.status).toBe("saved");
});
it("queries receipt before replaying and only says delivered with host evidence", async () => {
  const client = await import("./api");
  vi.stubGlobal("fetch", async () => response({}));
  await client.postAction("/remotea/runs/r/pause", { expected_revision: 4 });
  const fetch = vi.fn(async (_url: string) => response({ status: "delivered", commands: [{ status: "done" }] })); vi.stubGlobal("fetch", fetch);
  await client.flushActions();
  expect(fetch).toHaveBeenCalledOnce(); expect(fetch.mock.calls[0][0]).toContain("/remotea/actions/");
  expect(client.actionReceipts().at(-1)?.status).toBe("delivered");
});
it("queues locally while definitely offline and replays the same action after a missing receipt", async () => {
  const client = await import("./api");
  vi.stubGlobal("navigator", { onLine: false });
  const fetch = vi.fn(); vi.stubGlobal("fetch", fetch);
  await expect(client.postAction("/remotea/runs/r/resume", { expected_revision: 4 }, { retries: 0 })).rejects.toMatchObject({ code: "NETWORK" });
  expect(fetch).not.toHaveBeenCalled(); const id = client.actionReceipts()[0].id;
  expect(client.actionReceipts()[0].status).toBe("queued");
  vi.stubGlobal("navigator", { onLine: true });
  const bodies: unknown[] = [];
  vi.stubGlobal("fetch", async (_url: string, init: RequestInit) => { if (init.method === "POST") { bodies.push(JSON.parse(String(init.body))); return response({}); } return response({ error: { code: "NOT_FOUND" } }, 404); });
  await client.flushActions();
  expect(bodies).toEqual([{ expected_revision: 4, action_id: id }]);
  expect(client.actionReceipts()[0].status).toBe("saved");
});
