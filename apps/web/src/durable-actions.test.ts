// @vitest-environment jsdom
import { beforeEach, afterEach, expect, it, vi } from "vitest";
beforeEach(() => { localStorage.clear(); vi.resetModules(); });
afterEach(() => vi.unstubAllGlobals());
const response = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status });
it("persists an uncertain action ID through manual retry and reload", async () => {
  let client = await import("./api"); client.setDurableScope("owner-a:personal");
  const sent: Record<string, unknown>[] = [];
  vi.stubGlobal("fetch", async (_url: string, init: RequestInit) => { sent.push(JSON.parse(String(init.body))); throw new TypeError("lost response"); });
  await expect(client.postAction("/remotea/runs/r/message", { text: "Do it" }, { retries: 0 })).rejects.toMatchObject({ code: "NETWORK" });
  expect(client.actionReceipts().at(-1)?.status).toBe("uncertain");
  vi.resetModules(); client = await import("./api"); client.setDurableScope("owner-a:personal");
  vi.stubGlobal("fetch", async (_url: string, init: RequestInit) => { sent.push(JSON.parse(String(init.body))); return response({ receipt: { status: "saved" } }); });
  await client.postAction("/remotea/runs/r/message", { text: "Do it" }, { retries: 0 });
  expect(sent[1].action_id).toBe(sent[0].action_id);
  expect(client.actionReceipts().at(-1)?.status).toBe("saved");
});
it("queries receipt before replaying and only says delivered with host evidence", async () => {
  const client = await import("./api"); client.setDurableScope("owner-a:personal");
  vi.stubGlobal("fetch", async () => response({}));
  await client.postAction("/remotea/runs/r/pause", { expected_revision: 4 });
  const fetch = vi.fn(async (_url: string) => response({ status: "delivered", commands: [{ status: "done" }] })); vi.stubGlobal("fetch", fetch);
  await client.flushActions();
  expect(fetch).toHaveBeenCalledOnce(); expect(fetch.mock.calls[0][0]).toContain("/remotea/actions/");
  expect(client.actionReceipts().at(-1)?.status).toBe("delivered");
});
it("queues locally while definitely offline and replays the same action after a missing receipt", async () => {
  const client = await import("./api"); client.setDurableScope("owner-a:personal");
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

it("keeps outboxes owner/workspace scoped and pauses replay when unbound", async () => {
  const client = await import("./api"); client.setDurableScope("owner-a:personal");
  vi.stubGlobal("navigator", { onLine: false });
  await expect(client.postAction("/remotea/runs/private/message", { text: "Private instructions" }, { retries: 0 })).rejects.toMatchObject({ code: "NETWORK" });
  client.setDurableScope("owner-b:personal");
  expect(client.actionReceipts()).toEqual([]);
  const fetch = vi.fn(); vi.stubGlobal("fetch", fetch);
  await client.flushActions(); expect(fetch).not.toHaveBeenCalled();
  client.setDurableScope(null); await client.flushActions(); expect(fetch).not.toHaveBeenCalled();
  client.setDurableScope("owner-a:personal"); expect(client.actionReceipts()[0].body.text).toBe("Private instructions");
  client.setDurableScope("owner-a:workspace-b"); expect(client.actionReceipts()).toEqual([]);
});
it("marks terminal server-only receipts Applied and stops polling them", async () => {
  const client = await import("./api"); client.setDurableScope("owner-a:personal");
  vi.stubGlobal("fetch", async () => response({ receipt: { status: "saved", terminal: true } }));
  await client.postAction("/remotea/runs/r/approve");
  expect(client.actionReceipts()[0].status).toBe("applied");
  const fetch = vi.fn(); vi.stubGlobal("fetch", fetch);
  await client.flushActions(); expect(fetch).not.toHaveBeenCalled();
});

it("unbinds replay when another tab logs out or changes account", async () => {
  const client = await import("./api"); client.setDurableScope("owner-a:personal");
  window.dispatchEvent(new StorageEvent("storage", { key: "eri-auth-scope", newValue: "owner-b:personal" }));
  expect(client.durableScope()).toBeNull();
  const fetch = vi.fn(); vi.stubGlobal("fetch", fetch);
  await client.flushActions(); expect(fetch).not.toHaveBeenCalled();
});

it("keeps a newer queued action visible when older receipts are polled", async () => {
  const client = await import("./api"); client.setDurableScope("owner-a:personal");
  vi.stubGlobal("fetch", async () => response({}));
  await client.postAction("/remotea/runs/r/first", {});
  const first = client.actionReceipts()[0].id;
  vi.stubGlobal("navigator", { onLine: false });
  await expect(client.postAction("/remotea/runs/r/message", { text: "Queued later" }, { retries: 0 })).rejects.toMatchObject({ code: "NETWORK" });
  const queued = client.actionReceipts().at(-1)!.id;
  vi.stubGlobal("navigator", { onLine: true });
  vi.stubGlobal("fetch", async (url: string) => { if (url.endsWith(first)) return response({ status: "saved", terminal: true }); throw new TypeError("receipt temporarily unreachable"); });
  await client.flushActions();
  expect(client.actionReceipts().at(-1)).toMatchObject({ id: queued, status: "queued" });
});
