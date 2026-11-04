// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Meetings } from "./Meetings";
import { source } from "./test-fixtures";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

type Call = { url: string; init: RequestInit };
let host: HTMLDivElement, root: Root, calls: Call[], routes: Record<string, (call: Call) => [number, unknown]>;
beforeEach(() => {
  host = document.createElement("div"); document.body.append(host); root = createRoot(host); calls = [];
  routes = {
    "/api/v1/context/sources": () => [200, { items: [
      source(),
      source({ id: "src-2", title: "Hiring loop", status: "partial", processing: { stage: "extract", status: "partial", attempts: 1, updated_at: "",
        coverage: { "0": "extracted", "1": "failed" } } }),
      source({ id: "src-3", title: "Vendor call", status: "failed", last_error: { stage: "ingest", status: "failed", message: "Empty file.", retryable: false, at: "" } }),
    ], total: 3, limit: 30, offset: 0 }],
    "/api/v1/context/reviews": () => [200, { items: [{ source_id: "src-1", status: "pending", pending: 2, later: 0 }, { source_id: "src-3", status: "done", pending: 0, later: 0 }] }],
  };
  vi.stubGlobal("fetch", vi.fn(async (url: string, init: RequestInit = {}) => {
    const call = { url, init }; calls.push(call);
    const route = routes[url.split("?")[0]];
    const [status, body] = route ? route(call) : [404, { error: { code: "NOT_FOUND", message: "Not found." } }];
    return new Response(JSON.stringify(body), { status });
  }));
});
afterEach(() => { act(() => root.unmount()); host.remove(); vi.unstubAllGlobals(); });

async function render(node: React.ReactNode) {
  await act(async () => { root.render(node); });
  for (let i = 0; i < 3; i++) await act(async () => { await Promise.resolve(); });
}
const text = () => host.textContent ?? "";

describe("Meetings list", () => {
  it("lists meetings with status, partial coverage, failure kind and pending review counts", async () => {
    await render(<Meetings sourceId={null} onOpen={() => {}} onPerson={() => {}}/>);
    expect(text()).toContain("Pricing sync");
    expect(text()).toContain("Ready");
    expect(text()).toContain("2 to review");
    expect(text()).toContain("Partly read");
    expect(text()).toContain("1 of 2 passages read");
    expect(text()).toContain("Could not read");
    expect(text()).toContain("Empty file.");
    expect(text()).toContain("Reviewed");
    expect(calls.every(c => c.url.startsWith("/api/v1/context/"))).toBe(true);
  });

  it("still lists meetings when review counts are unavailable", async () => {
    delete routes["/api/v1/context/reviews"];
    await render(<Meetings sourceId={null} onOpen={() => {}} onPerson={() => {}}/>);
    expect(text()).toContain("Hiring loop");
    expect(text()).not.toContain("2 to review");
    expect(host.querySelector("[role=alert]")).toBeNull();
  });

  it("opens a meeting and filters by status", async () => {
    const onOpen = vi.fn();
    await render(<Meetings sourceId={null} onOpen={onOpen} onPerson={() => {}}/>);
    const row = [...host.querySelectorAll<HTMLButtonElement>(".meeting-row")].find(b => b.textContent?.includes("Hiring loop"))!;
    await act(async () => { row.click(); });
    expect(onOpen).toHaveBeenCalledWith("src-2");
    const select = host.querySelector<HTMLSelectElement>("select[aria-label='Meeting status']")!;
    await act(async () => { select.value = "failed"; select.dispatchEvent(new Event("change", { bubbles: true })); });
    await act(async () => { await Promise.resolve(); });
    expect(calls.some(c => c.url.includes("status=failed"))).toBe(true);
  });

  it("says when meeting memory is off instead of showing an error", async () => {
    routes["/api/v1/context/sources"] = () => [403, { error: { code: "CONTEXT_DISABLED", message: "Meeting context is not turned on." } }];
    await render(<Meetings sourceId={null} onOpen={() => {}} onPerson={() => {}}/>);
    expect(text()).toContain("Meeting memory is not on");
    expect(host.querySelector("[role=alert]")).toBeNull();
  });

  it("refreshes once after a burst of context changes", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    try {
      await render(<Meetings sourceId={null} onOpen={() => {}} onPerson={() => {}}/>);
      const lists = () => calls.filter(c => c.url.startsWith("/api/v1/context/sources")).length;
      const before = lists();
      for (let i = 0; i < 3; i++) window.dispatchEvent(new Event("eri-context-changed"));
      await act(async () => { vi.advanceTimersByTime(999); });
      expect(lists()).toBe(before);
      await act(async () => { vi.advanceTimersByTime(1); }); await act(async () => { await Promise.resolve(); });
      expect(lists()).toBe(before + 1);
    } finally { vi.useRealTimers(); }
  });
});
