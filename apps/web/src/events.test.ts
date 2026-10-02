import { afterEach, expect, test, vi } from "vitest";
import { createLoader, EVENT_SILENCE_MS, eventStreamOnline, signalScope, subscribeEvents } from "./events";
class Stream {
  static all: Stream[] = [];
  onopen?: () => void;
  onerror?: () => void;
  onmessage?: (event: { lastEventId: string }) => void;
  close = vi.fn();
  addEventListener() {}
  constructor(public url: string) {
    Stream.all.push(this);
  }
}
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  Stream.all = [];
});
test("a failed stream reconnects with its last cursor and refreshes after recovery", async () => {
  vi.useFakeTimers();
  vi.stubGlobal("EventSource", Stream);
  const refresh = vi.fn(),
    online = vi.fn();
  const stop = subscribeEvents(66, refresh, online);
  const first = Stream.all[0];
  first.onmessage?.({ lastEventId: "71" });
  first.onerror?.();
  expect(first.close).toHaveBeenCalledOnce();
  expect(online).toHaveBeenLastCalledWith(false);
  await vi.advanceTimersByTimeAsync(1000);
  expect(Stream.all[1].url).toBe("/api/v1/events?after=71");
  Stream.all[1].onopen?.();
  expect(online).toHaveBeenLastCalledWith(true);
  expect(refresh).toHaveBeenCalledTimes(2);
  stop();
});
test("unmount cancels a pending reconnect and ignores late callbacks", async () => {
  vi.useFakeTimers();
  vi.stubGlobal("EventSource", Stream);
  const refresh = vi.fn(),
    online = vi.fn();
  const stop = subscribeEvents(0, refresh, online);
  Stream.all[0].onerror?.();
  stop();
  Stream.all[0].onopen?.();
  await vi.advanceTimersByTimeAsync(16000);
  expect(Stream.all).toHaveLength(1);
  expect(refresh).not.toHaveBeenCalled();
});
test("each message is parsed once into a kind-tagged signal", () => {
  vi.stubGlobal("EventSource", Stream);
  const parse = vi.spyOn(JSON, "parse");
  const refresh = vi.fn();
  const stop = subscribeEvents(0, refresh, vi.fn());
  (Stream.all[0].onmessage as unknown as (e: object) => void)({ lastEventId: "5", data: '{"kind":"task.changed","entity_id":"t1"}' });
  expect(parse).toHaveBeenCalledTimes(1);
  expect(refresh).toHaveBeenLastCalledWith({ reason: "event", id: 5, kind: "task.changed", entity_id: "t1" });
  parse.mockRestore();
  stop();
});
test("a silent half-open stream is replaced after the heartbeat window", async () => {
  vi.useFakeTimers();
  vi.stubGlobal("EventSource", Stream);
  const online = vi.fn();
  const stop = subscribeEvents(3, vi.fn(), online);
  Stream.all[0].onopen?.();
  expect(eventStreamOnline()).toBe(true);
  await vi.advanceTimersByTimeAsync(EVENT_SILENCE_MS + 10);
  expect(Stream.all[0].close).toHaveBeenCalled();
  expect(online).toHaveBeenLastCalledWith(false);
  expect(eventStreamOnline()).toBe(false);
  await vi.advanceTimersByTimeAsync(1000);
  expect(Stream.all).toHaveLength(2);
  stop();
});
test("signal scope only invalidates what an event concerns", () => {
  expect(signalScope({ reason: "event", id: 1, kind: "memory.changed" })).toEqual({ load: false, revision: false, memory: true });
  expect(signalScope({ reason: "event", id: 1, kind: "note.changed" })).toEqual({ load: false, revision: true, memory: false });
  expect(signalScope({ reason: "event", id: 1, kind: "task.changed" })).toEqual({ load: true, revision: true, memory: false });
  expect(signalScope({ reason: "open" })).toEqual({ load: true, revision: true, memory: true });
});
function deferred() {
  let resolve!: (v: number) => void;
  const promise = new Promise<number>((r) => { resolve = r; });
  return { promise, resolve };
}
test("loader coalesces a burst into one load with a trailing re-run", async () => {
  vi.useFakeTimers();
  const runs: { revision: boolean; done: ReturnType<typeof deferred> }[] = [];
  const loader = createLoader(({ revision }) => {
    const done = deferred();
    runs.push({ revision, done });
    return done.promise;
  }, 250);
  const a = loader.request(), b = loader.request({ revision: true }), c = loader.request();
  await vi.advanceTimersByTimeAsync(249);
  expect(runs).toHaveLength(0);
  await vi.advanceTimersByTimeAsync(1);
  expect(runs).toHaveLength(1);
  expect(runs[0].revision).toBe(true);
  const d = loader.request(), e = loader.request();
  await vi.advanceTimersByTimeAsync(1000);
  expect(runs).toHaveLength(1);
  runs[0].done.resolve(10);
  await Promise.all([a, b, c]);
  await vi.advanceTimersByTimeAsync(0);
  expect(runs).toHaveLength(2);
  runs[1].done.resolve(11);
  await Promise.all([d, e]);
  expect(runs).toHaveLength(2);
});
test("loader drops feed signals its last snapshot already covers (own-write echo)", async () => {
  vi.useFakeTimers();
  const runs: ReturnType<typeof deferred>[] = [];
  const loader = createLoader(() => { const d = deferred(); runs.push(d); return d.promise; }, 250);
  const own = loader.request({ immediate: true, revision: true });
  expect(runs).toHaveLength(1);
  const echo = loader.request({ cursor: 42, revision: true });
  runs[0].resolve(42);
  await own; await echo;
  await vi.advanceTimersByTimeAsync(1000);
  expect(runs).toHaveLength(1);
  await loader.request({ cursor: 40 });
  expect(runs).toHaveLength(1);
  const later = loader.request({ cursor: 43 });
  await vi.advanceTimersByTimeAsync(250);
  expect(runs).toHaveLength(2);
  runs[1].resolve(43);
  await later;
});
test("a cancelled load's late response is marked stale", async () => {
  const runs: { current: () => boolean; done: ReturnType<typeof deferred> }[] = [];
  const loader = createLoader(({ current }) => { const done = deferred(); runs.push({ current, done }); return done.promise; });
  void loader.request({ immediate: true });
  loader.cancel();
  const next = loader.request({ immediate: true });
  expect(runs).toHaveLength(1);
  expect(runs[0].current()).toBe(false);
  runs[0].done.resolve(1);
  await new Promise((r) => setTimeout(r, 0));
  expect(runs).toHaveLength(2);
  expect(runs[1].current()).toBe(true);
  runs[1].done.resolve(2);
  await next;
});
