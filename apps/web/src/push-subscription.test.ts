import { afterEach, expect, test, vi } from "vitest";
import { ensurePushSubscription, watchPushSubscription } from "./push-subscription";

afterEach(() => vi.unstubAllGlobals());

function setup(permission: string, existing: object | null) {
  const fetch = vi.fn(async () => ({ ok: true, json: async () => ({ registered: true }) }));
  const listeners: ((event: MessageEvent) => void)[] = [];
  const subscription = { toJSON: () => ({ endpoint: "https://push.example/new", keys: { p256dh: "a", auth: "b" } }) };
  const pushManager = {
    getSubscription: vi.fn(async () => existing),
    subscribe: vi.fn(async () => subscription),
  };
  vi.stubGlobal("fetch", fetch);
  vi.stubGlobal("Notification", { permission });
  vi.stubGlobal("navigator", {
    serviceWorker: {
      ready: Promise.resolve({ pushManager }),
      addEventListener: (_: string, f: (event: MessageEvent) => void) => listeners.push(f),
      removeEventListener: vi.fn(),
    },
  });
  return { fetch, pushManager, listeners };
}

test("granted permission re-posts the current subscription on boot", async () => {
  const existing = { toJSON: () => ({ endpoint: "https://push.example/old", keys: { p256dh: "a", auth: "b" } }) };
  const { fetch, pushManager } = setup("granted", existing);
  expect(await ensurePushSubscription("AQAB")).toBe(true);
  expect(pushManager.subscribe).not.toHaveBeenCalled();
  const [url, options] = fetch.mock.calls[0] as unknown as [string, RequestInit];
  expect(url).toBe("/api/v1/push");
  expect(options.credentials).toBe("same-origin");
  expect(JSON.parse(options.body as string).endpoint).toBe("https://push.example/old");
});

test("a lost subscription is recreated, and nothing happens without permission", async () => {
  const { fetch, pushManager } = setup("granted", null);
  expect(await ensurePushSubscription("AQAB")).toBe(true);
  expect(pushManager.subscribe).toHaveBeenCalledOnce();
  expect(fetch).toHaveBeenCalledOnce();
  const denied = setup("default", null);
  expect(await ensurePushSubscription("AQAB")).toBe(false);
  expect(denied.fetch).not.toHaveBeenCalled();
});

test("a service worker change message re-registers", async () => {
  const existing = { toJSON: () => ({ endpoint: "https://push.example/x", keys: { p256dh: "a", auth: "b" } }) };
  const { fetch, listeners } = setup("granted", existing);
  watchPushSubscription("AQAB");
  listeners[0]({ data: { type: "push-subscription-changed" } } as MessageEvent);
  await vi.waitFor(() => expect(fetch).toHaveBeenCalledOnce());
});
