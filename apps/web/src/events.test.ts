import { afterEach, expect, test, vi } from "vitest";
import { subscribeEvents } from "./events";
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
