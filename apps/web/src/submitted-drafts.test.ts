// @vitest-environment jsdom
import { afterEach, beforeEach, expect, it, vi } from "vitest";

beforeEach(() => { localStorage.clear(); vi.resetModules(); vi.useFakeTimers(); });
afterEach(() => { vi.clearAllTimers(); vi.useRealTimers(); vi.unstubAllGlobals(); });
const response = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status });
async function loaded() {
  const client = await import("./api"), drafts = await import("./run-drafts");
  client.setDurableScope("owner-a:personal");
  return { client, drafts };
}
async function queued(text = "Offline instruction") {
  const { client, drafts } = await loaded();
  drafts.writeDraft("composer:r1", text);
  const reference = drafts.submittedDraft("composer:r1")!;
  vi.stubGlobal("navigator", { onLine: false });
  await expect(client.postAction("/remotea/runs/r1/message", { text, mode: "queue" }, { draft: reference, retries: 0 })).rejects.toMatchObject({ code: "NETWORK" });
  return { client, drafts, reference, id: client.actionReceipts()[0].id };
}

it("clears only the acknowledged submitted draft after an offline queue and reload", async () => {
  const first = await queued();
  expect(first.drafts.readDraft("composer:r1", "").value).toBe("Offline instruction");
  vi.resetModules();
  const { client, drafts } = await loaded();
  expect(drafts.submittedDraft("composer:r1")).toEqual(first.reference);
  const bodies: unknown[] = [];
  vi.stubGlobal("navigator", { onLine: true });
  vi.stubGlobal("fetch", async (_url: string, init: RequestInit = {}) => {
    if (init.method === "POST") { bodies.push(JSON.parse(String(init.body))); return response({}); }
    return response({ error: { code: "NOT_FOUND" } }, 404);
  });
  await client.flushActions();
  expect(bodies).toEqual([{ text: "Offline instruction", mode: "queue", action_id: first.id }]);
  expect(client.actionReceipts()[0]).toMatchObject({ id: first.id, draft: first.reference, status: "saved" });
  expect(drafts.readDraft("composer:r1", "").value).toBe("");
  expect(drafts.submittedDraft("composer:r1")?.edit_id).not.toBe(first.reference.edit_id);
});

it.each(["A newer instruction", "Offline instruction"])("preserves a newer edit %j across reload when the original action is delivered", async newer => {
  const first = await queued();
  // A new edit has its own identity even when the user intentionally retypes identical text.
  first.drafts.writeDraft("composer:r1", newer);
  const edited = first.drafts.submittedDraft("composer:r1");
  expect(edited?.edit_id).not.toBe(first.reference.edit_id);
  vi.resetModules();
  const { client, drafts } = await loaded();
  vi.stubGlobal("navigator", { onLine: true });
  vi.stubGlobal("fetch", async () => response({ status: "delivered", terminal: true }));
  await client.flushActions();
  expect(client.actionReceipts()[0].status).toBe("delivered");
  expect(drafts.readDraft("composer:r1", "").value).toBe(newer);
  expect(drafts.submittedDraft("composer:r1")).toEqual(edited);
});

it.each(["uncertain", "failed"])("keeps the submitted draft when delivery is %s", async status => {
  const { client, drafts, reference } = await queued();
  vi.stubGlobal("navigator", { onLine: true });
  vi.stubGlobal("fetch", async () => response({ status, terminal: true }));
  await client.flushActions();
  expect(drafts.readDraft("composer:r1", "").value).toBe("Offline instruction");
  expect(drafts.submittedDraft("composer:r1")).toEqual(reference);
});

it("gives an intentionally resubmitted new draft its own action identity", async () => {
  const first = await queued();
  first.drafts.writeDraft("composer:r1", "Offline instruction");
  const newer = first.drafts.submittedDraft("composer:r1")!;
  await expect(first.client.postAction("/remotea/runs/r1/message", { text: "Offline instruction", mode: "queue" }, { draft: newer, retries: 0 })).rejects.toMatchObject({ code: "NETWORK" });
  const second = first.client.actionReceipts()[1];
  expect(second.id).not.toBe(first.id);
  expect(second.draft).toEqual(newer);
});
