import { afterEach, beforeEach, expect, it, vi } from "vitest";

vi.mock("./api", () => ({post: vi.fn(async () => ({recorded: true}))}));
import {post} from "./api";
let doc: EventTarget & {visibilityState: string};
beforeEach(() => {
  vi.resetModules(); vi.clearAllMocks();
  doc = Object.assign(new EventTarget(), {visibilityState: "visible"});
  vi.stubGlobal("document", doc);
});
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

it("does not measure old history, background tabs, or expired requests", async () => {
  const m = await import("./clientLatency");
  await m.reportWorkRendered("old", 1);
  expect(post).not.toHaveBeenCalled();
  const clock = vi.spyOn(performance, "now").mockReturnValue(100);
  m.chatSendStarted("new"); doc.visibilityState = "hidden";
  await m.reportWorkRendered("new", 1);
  expect(post).not.toHaveBeenCalled();
  doc.visibilityState = "visible"; clock.mockReturnValue(300101);
  await m.reportWorkRendered("new", 1);
  expect(post).not.toHaveBeenCalled();
});

it("deduplicates simultaneous card/reply reports and allows a failed report retry", async () => {
  const m = await import("./clientLatency");
  const clock = vi.spyOn(performance, "now").mockReturnValue(100);
  m.chatSendStarted("new"); clock.mockReturnValue(240);
  await Promise.all([m.reportWorkRendered("new", 1), m.reportWorkRendered("new", 1, "work_reply_rendered")]);
  expect(post).toHaveBeenCalledTimes(1);
  expect(vi.mocked(post).mock.calls[0][1]).toMatchObject({elapsed_ms: 140, revision: 1});
  m.chatSendStarted("retry");
  vi.mocked(post).mockRejectedValueOnce(new Error("offline"));
  await m.reportWorkRendered("retry", 1);
  await m.reportWorkRendered("retry", 1, "work_reply_rendered");
  expect(post).toHaveBeenCalledTimes(3);
});

it("does not report a result that scrolls away before paint; resumes on foreground", async () => {
  const m = await import("./clientLatency");
  let callback: (entries: Array<{isIntersecting: boolean}>) => void = () => {};
  const frames = new Map<number, FrameRequestCallback>(); let seq = 0;
  vi.stubGlobal("IntersectionObserver", class {
    constructor(cb: typeof callback) { callback = cb; }
    observe() {} disconnect() {}
  });
  vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => {frames.set(++seq, cb); return seq;});
  vi.stubGlobal("cancelAnimationFrame", (id: number) => frames.delete(id));
  const paint = () => { const batch = [...frames.values()]; frames.clear(); batch.forEach(cb => cb(0)); };
  m.chatSendStarted("new");
  const stop = m.observeWorkRendered({isConnected: true} as HTMLElement, "new", 1);
  callback([{isIntersecting: true}]); paint();
  callback([{isIntersecting: false}]); paint();
  expect(post).not.toHaveBeenCalled();
  doc.visibilityState = "hidden"; callback([{isIntersecting: true}]); paint();
  expect(post).not.toHaveBeenCalled();
  doc.visibilityState = "visible"; doc.dispatchEvent(new Event("visibilitychange"));
  paint(); paint(); await Promise.resolve();
  expect(post).toHaveBeenCalledTimes(1);
  stop();
});

it("bounds memory when many unfinished requests accumulate", async () => {
  const m = await import("./clientLatency");
  for (let i=0;i<201;i++) m.chatSendStarted(String(i));
  await m.reportWorkRendered("0", 1);
  expect(post).not.toHaveBeenCalled();
  await m.reportWorkRendered("200", 1);
  expect(post).toHaveBeenCalledTimes(1);
});
