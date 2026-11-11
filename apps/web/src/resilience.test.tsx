// @vitest-environment jsdom
/// <reference types="vite/client" />
// Outage vs signed out (api.ts, connectivity.ts, Outage.tsx), session expiry, and error boundaries.
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api, ApiError, isOutage, SESSION_EXPIRED } from "./api";
import { backoff, connectivity, probeNow, reportReachable, reportUnreachable, resetConnectivity } from "./connectivity";
import { ErrorBoundary } from "./ErrorBoundary";
import { OutageBanner } from "./Outage";
import { captureError } from "./monitoring";
vi.mock("./monitoring", async (original) => ({ ...(await original<typeof import("./monitoring")>()), captureError: vi.fn() }));
import appSource from "./App.tsx?raw";
import mainSource from "./main.tsx?raw";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement, root: Root;
beforeEach(() => {
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
  resetConnectivity(async () => false);
});
afterEach(() => {
  act(() => root.unmount());
  host.remove();
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  resetConnectivity();
});

const json = (status: number, body: unknown) => ({ ok: status < 400, status, json: async () => body, headers: new Headers() });

describe("api outcomes", () => {
  it("a network failure is an outage, not a sign-out", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    const expired = vi.fn();
    window.addEventListener(SESSION_EXPIRED, expired);
    const error = await api("/bootstrap").catch((e) => e);
    window.removeEventListener(SESSION_EXPIRED, expired);
    expect(isOutage(error)).toBe(true);
    expect(connectivity().reachable).toBe(false);
    expect(expired).not.toHaveBeenCalled();
  });

  it("a request its caller cancelled is not an outage", async () => {
    // The run drawer aborts its previous load whenever a newer one starts (remotea-model.ts latestOnly).
    const controller = new AbortController();
    vi.stubGlobal("fetch", vi.fn((_url: string, init: RequestInit) => new Promise((_resolve, reject) =>
      init.signal!.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError"))))));
    const pending = api("/remotea/runs/1", { signal: controller.signal }).catch((e: ApiError) => e);
    controller.abort();
    const error = (await pending) as ApiError;
    expect(error).toBeInstanceOf(ApiError);
    expect(error.code).toBe("ABORTED");
    expect(isOutage(error)).toBe(false);
    expect(connectivity().reachable).toBe(true);
    // Cancelled while the body was still arriving.
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ...json(200, {}), json: async () => { controller.abort(); throw new DOMException("Aborted", "AbortError"); } }));
    const late = new AbortController();
    const body = api("/remotea/runs/1", { signal: late.signal }).catch((e: ApiError) => e);
    late.abort();
    expect(((await body) as ApiError).code).toBe("ABORTED");
    expect(connectivity().reachable).toBe(true);
  });
  it("a 502 proxy page and a 503 are outages", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 502, json: async () => { throw new SyntaxError("html"); }, headers: new Headers() }));
    const bad = await api("/tasks").catch((e) => e);
    expect(isOutage(bad)).toBe(true);
    expect((bad as ApiError).status).toBe(502);
    resetConnectivity(async () => false);
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json(503, { error: { code: "MAINTENANCE", message: "Updating" } })));
    const maintenance = await api("/tasks").catch((e) => e);
    expect(isOutage(maintenance)).toBe(true);
    expect(connectivity().reachable).toBe(false);
  });

  it("a 401 ends the session; a wrong pairing code does not", async () => {
    const expired = vi.fn();
    window.addEventListener(SESSION_EXPIRED, expired);
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json(401, { error: { code: "NOT_AUTHORIZED", message: "Sign in to Eridani." } })));
    const error = await api("/tasks").catch((e) => e);
    expect(isOutage(error)).toBe(false);
    expect((error as ApiError).status).toBe(401);
    expect(expired).toHaveBeenCalledTimes(1);
    await api("/auth/login", { method: "POST" }).catch(() => {});
    expect(expired).toHaveBeenCalledTimes(1);
    window.removeEventListener(SESSION_EXPIRED, expired);
    expect(connectivity().reachable).toBe(true);
  });

  it("a 4xx answer means the API is reachable", async () => {
    reportUnreachable();
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json(404, { error: { code: "NOT_FOUND", message: "Gone" } })));
    await api("/tasks/x").catch(() => {});
    expect(connectivity().reachable).toBe(true);
  });
});

describe("connectivity backoff", () => {
  it("probes with growing delays until the API answers", async () => {
    vi.useFakeTimers();
    let up = false;
    const probe = vi.fn(async () => up);
    resetConnectivity(probe);
    reportUnreachable();
    await vi.advanceTimersByTimeAsync(backoff(0));
    expect(probe).toHaveBeenCalledTimes(1);
    expect(connectivity().attempts).toBe(1);
    await vi.advanceTimersByTimeAsync(backoff(1) - 1);
    expect(probe).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(probe).toHaveBeenCalledTimes(2);
    up = true;
    await probeNow();
    expect(connectivity()).toEqual({ reachable: true, nextProbeAt: null, attempts: 0 });
    await vi.advanceTimersByTimeAsync(60_000);
    expect(probe).toHaveBeenCalledTimes(3);
    expect(backoff(99)).toBe(30_000);
  });

  it("the banner shows only while unreachable and offers Retry now", async () => {
    await act(async () => root.render(<OutageBanner/>));
    expect(host.textContent).toBe("");
    await act(async () => reportUnreachable());
    expect(host.textContent).toContain("Can't reach Eridani");
    expect(host.querySelector("button")?.textContent).toBe("Retry now");
  });

  it("keeps a restored connection readable for six seconds and allows dismissal", async () => {
    vi.useFakeTimers();
    await act(async () => root.render(<OutageBanner/>));
    await act(async () => reportUnreachable());
    await act(async () => reportReachable());
    expect(host.textContent).toContain("Connection restored");
    await act(async () => vi.advanceTimersByTimeAsync(5999));
    expect(host.textContent).toContain("Connection restored");
    await act(async () => vi.advanceTimersByTimeAsync(1));
    expect(host.textContent).toBe("");
    await act(async () => reportUnreachable());
    await act(async () => reportReachable());
    await act(async () => host.querySelector("button")!.click());
    expect(host.textContent).toBe("");
  });

  it("keeps the outage visible if the connection fails again during the recovery notice", async () => {
    vi.useFakeTimers();
    await act(async () => root.render(<OutageBanner/>));
    await act(async () => reportUnreachable());
    await act(async () => reportReachable());
    await act(async () => vi.advanceTimersByTimeAsync(3000));
    await act(async () => reportUnreachable());
    await act(async () => vi.advanceTimersByTimeAsync(7000));
    expect(host.textContent).toContain("Can't reach Eridani");
    expect(host.querySelector("button")?.textContent).toBe("Retry now");
  });

  it("App keeps the outage apart from sign-in and refreshes after it", () => {
    expect(appSource).toContain("if (!boot && bootOutage) return <OutageScreen/>;");
    expect(appSource).toMatch(/if \(isOutage\(e\)\) \{[\s\S]*?setBootOutage\(true\)/);
    expect(appSource).toContain("<OutageBanner/>");
    expect(appSource).toContain("window.addEventListener(SESSION_EXPIRED, expired)");
  });
});

describe("error boundaries", () => {
  function Boom({ fail }: { fail: boolean }): React.ReactNode {
    if (fail) throw new Error("render failed");
    return <p>view ok</p>;
  }

  it("a crashing view shows a recovery notice while the rest of the page renders", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    await act(async () => root.render(<div>
      <nav>sidebar</nav>
      <ErrorBoundary name="tasks" resetKey="tasks"><Boom fail/></ErrorBoundary>
    </div>));
    expect(host.textContent).toContain("sidebar");
    expect(host.textContent).toContain("This view hit a problem");
    const labels = [...host.querySelectorAll("button")].map((b) => b.textContent);
    expect(labels).toEqual(["Try again", "Reload", "Report this"]);
    expect(captureError).toHaveBeenCalledWith(expect.any(Error), expect.objectContaining({ boundary: "tasks" }));
  });

  it("switching views clears the error", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    await act(async () => root.render(<ErrorBoundary name="tasks" resetKey="tasks"><Boom fail/></ErrorBoundary>));
    expect(host.textContent).toContain("This view hit a problem");
    await act(async () => root.render(<ErrorBoundary name="today" resetKey="today"><Boom fail={false}/></ErrorBoundary>));
    expect(host.textContent).toBe("view ok");
  });

  it("the app-level notice offers Reload and Report, and Report opens the bug dialog prefilled", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    await act(async () => root.render(<ErrorBoundary name="app" scope="app"><Boom fail/></ErrorBoundary>));
    expect(host.textContent).toContain("Eridani hit a problem");
    const report = [...host.querySelectorAll("button")].find((b) => b.textContent === "Report this")!;
    await act(async () => { report.click(); await new Promise((r) => setTimeout(r, 50)); });
    await act(async () => { await new Promise((r) => setTimeout(r, 50)); });
    const description = host.querySelector("textarea") as HTMLTextAreaElement | null;
    expect(description?.value).toBe("Eridani crashed: render failed");
  });

  it("wraps the whole app and the main view", () => {
    expect(mainSource).toContain('<ErrorBoundary name="app" scope="app">');
    expect(appSource).toContain("<ErrorBoundary name={view} resetKey={view}>");
  });
});
