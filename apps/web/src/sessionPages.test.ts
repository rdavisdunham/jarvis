import { afterEach, describe, expect, it, vi } from "vitest";
import { sessionPages, watchSessionChanges, type SessionPage, type SessionPageState } from "./sessionPages";

type Item = { id: string; title: string };
const rows = (n: number) => Array.from({ length: n }, (_, i) => ({ id: String(i), title: "Session " + i }));
const page = (items: Item[], offset: number, limit: number): SessionPage<Item> => ({
  items: items.slice(offset, offset + limit), total: items.length,
  next_offset: offset + limit < items.length ? offset + limit : null,
});
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });

describe("session browsing", () => {
  it("reaches older sessions and preserves the loaded depth across refreshes", async () => {
    let items = rows(135);
    let state: SessionPageState<Item> | undefined;
    const pages = sessionPages(async (offset, limit) => page(items, offset, limit), s => { state = s; });
    await pages.refresh();
    expect(state?.data?.items).toHaveLength(50);
    await pages.more(); await pages.more();
    expect(state?.data?.items.map(r => r.id)).toEqual(items.map(r => r.id));
    expect(state?.data?.next_offset).toBeNull();
    items = items.map(item => item.id === "100" ? { ...item, title: "Updated session" } : item);
    await pages.refresh();
    expect(state?.data?.items).toHaveLength(135);
    expect(state?.data?.items[100].title).toBe("Updated session");
  });
  it("rebuilds the loaded range when a newly reported session shifts offsets", async () => {
    let items = rows(115);
    let state: SessionPageState<Item> | undefined;
    const pages = sessionPages(async (offset, limit) => page(items, offset, limit), s => { state = s; });
    await pages.refresh();
    items = [{ id: "new", title: "Just reported" }, ...items];
    await pages.more();
    expect(state?.data?.items.map(r => r.id)).toEqual(items.slice(0, 100).map(r => r.id));
    expect(new Set(state?.data?.items.map(r => r.id)).size).toBe(100);
    expect(state?.data?.next_offset).toBe(100);
  });
  it("retains visible sessions on a failed refresh and permits retry", async () => {
    let fail = false;
    let state: SessionPageState<Item> | undefined;
    const pages = sessionPages(async (offset, limit) => {
      if (fail) throw new Error("Connection lost");
      return page(rows(80), offset, limit);
    }, s => { state = s; });
    await pages.refresh();
    fail = true; await pages.more();
    expect(state?.data?.items).toHaveLength(50);
    expect(state?.error).toBe("Connection lost");
    expect(state?.loading).toBe(false);
    fail = false; await pages.more();
    expect(state?.data?.items).toHaveLength(80); expect(state?.error).toBe("");
  });
  it("ignores late results after leaving a search and coalesces overlapping refreshes", async () => {
    let resolve!: (value: SessionPage<Item>) => void;
    const read = vi.fn(() => new Promise<SessionPage<Item>>(r => { resolve = r; }));
    const publish = vi.fn();
    const pages = sessionPages(read, publish);
    const pending = pages.refresh();
    await pages.refresh();
    expect(read).toHaveBeenCalledTimes(1);
    pages.dispose();
    const calls = publish.mock.calls.length;
    resolve(page(rows(1), 0, 50)); await pending;
    expect(publish).toHaveBeenCalledTimes(calls);
  });
  it("refreshes visible tabs, pauses hidden tabs, and removes timers on close", () => {
    vi.useFakeTimers();
    const doc = Object.assign(new EventTarget(), { visibilityState: "visible" });
    vi.stubGlobal("document", doc); vi.stubGlobal("window", globalThis);
    const refresh = vi.fn();
    const stop = watchSessionChanges(refresh, false);
    vi.advanceTimersByTime(30_000); expect(refresh).toHaveBeenCalledTimes(1);
    doc.visibilityState = "hidden"; vi.advanceTimersByTime(60_000);
    expect(refresh).toHaveBeenCalledTimes(1);
    doc.visibilityState = "visible"; doc.dispatchEvent(new Event("visibilitychange"));
    expect(refresh).toHaveBeenCalledTimes(2);
    stop(); vi.advanceTimersByTime(30_000); doc.dispatchEvent(new Event("visibilitychange"));
    expect(refresh).toHaveBeenCalledTimes(2);
  });
  it("does not repeatedly run semantic searches while the search box is idle", () => {
    vi.useFakeTimers();
    const doc = Object.assign(new EventTarget(), { visibilityState: "visible" });
    vi.stubGlobal("document", doc); vi.stubGlobal("window", globalThis);
    const refresh = vi.fn(), stop = watchSessionChanges(refresh, true);
    vi.advanceTimersByTime(120_000); expect(refresh).not.toHaveBeenCalled();
    doc.dispatchEvent(new Event("visibilitychange")); expect(refresh).toHaveBeenCalledTimes(1);
    stop();
  });
});
