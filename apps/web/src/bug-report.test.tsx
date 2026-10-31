// @vitest-environment jsdom
/// <reference types="vite/client" />
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";
import { collectContext, installDiagnostics, LIMITS, recordNavigation, redactPath, redactUrl, Ring, rings, safeStorage } from "./bug-report";
import { BugReportButton, isReportShortcut } from "./BugReport";
import { dataUrlBytes } from "./bug-report-capture";
import appSource from "./App.tsx?raw";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement, root: Root;
beforeEach(() => {
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
  localStorage.clear();
  Object.values(rings).forEach((ring) => ring.clear());
  vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => setTimeout(() => cb(0), 0));
});
afterEach(() => { act(() => root.unmount()); host.remove(); vi.unstubAllGlobals(); });
const settle = () => act(async () => { await new Promise((r) => setTimeout(r, 0)); });

describe("redaction", () => {
  it("drops query values and secrets, keeps paths, ids and deep-link parameters", () => {
    expect(redactPath("/tasks/11111111-1111-4111-8111-111111111111?status=open&q=dentist")).toBe("/tasks/11111111-1111-4111-8111-111111111111?status=…&q=…");
    expect(redactPath("/remotea/runs?task_id=abc#frag")).toBe("/remotea/runs?task_id=…");
    expect(redactUrl("https://app.example/?view=remotea&run=r-1&code=oauth&token=xyz")).toBe("https://app.example/?view=remotea&run=r-1&code=…&token=…");
    expect(redactUrl("/?view=<script>")).toBe("/?view=…");
    expect(redactPath("/x/eri_bot_abcdefghijklmnopqrstuvwxyz/y")).toBe("/x/[redacted]/y");
  });

  it("keeps storage key names but not key-like or long values", () => {
    localStorage.setItem("eri-density", "compact");
    localStorage.setItem("eridani-theme", "dark");
    localStorage.setItem("some-cache", "eyJhbGciOiJIUzI1NiJ9abcdefghijklmnop");
    localStorage.setItem("api_key", "short");
    localStorage.setItem("draft", "a long draft of a task. ".repeat(8));
    expect(safeStorage()).toEqual({ "eri-density": "compact", "eridani-theme": "dark", "some-cache": "[redacted]",
      api_key: "[redacted]", draft: "[192 characters]" });
  });
});

describe("ring buffers", () => {
  it("keep the newest entries up to their cap", () => {
    const ring = new Ring<number>(3);
    [1, 2, 3, 4, 5].forEach((n) => ring.push(n));
    expect(ring.list()).toEqual([3, 4, 5]);
  });

  it("record console errors, uncaught errors and rejections, capped at 50", () => {
    const logged = vi.fn(), quiet = { error: logged, warn: vi.fn() } as unknown as Console;
    const uninstall = installDiagnostics(window, quiet);
    try {
      for (let i = 0; i < 60; i++) quiet.error("boom", i);
      quiet.warn(new Error("careful: token=abcdefghijklmnop"));
      window.dispatchEvent(new ErrorEvent("error", { message: "Uncaught TypeError", filename: "https://app/x.js?v=1", lineno: 3, colno: 4 }));
      const rejection = new Event("unhandledrejection") as Event & { reason: unknown };
      rejection.reason = new Error("nope");
      window.dispatchEvent(rejection);
      const entries = rings.console.list();
      expect(entries).toHaveLength(LIMITS.console);
      expect(entries.at(-3)).toMatchObject({ level: "warn", message: "Error: careful: [redacted]" });
      expect(entries.at(-2)).toMatchObject({ level: "uncaught", message: "Uncaught TypeError", source: "https://app/x.js?v=…:3:4" });
      expect(entries.at(-1)).toMatchObject({ level: "unhandledrejection", message: "Error: nope" });
      expect(logged).toHaveBeenCalledTimes(60); // the original still logs
    } finally {
      uninstall();
    }
  });

  it("record api.ts requests without bodies or query values, capped at 40", async () => {
    vi.stubGlobal("fetch", async (url: string) => url.includes("fail")
      ? new Response(JSON.stringify({ error: { code: "NOT_FOUND", message: "gone" } }), { status: 404, headers: { "x-request-id": "req-9" } })
      : new Response(JSON.stringify({ ok: true }), { status: 200 }));
    for (let i = 0; i < 45; i++) await api("/tasks?query=secret-" + i);
    await api("/fail/123?x=1", { method: "POST", body: JSON.stringify({ password: "hunter2" }) }).catch(() => {});
    const entries = rings.requests.list();
    expect(entries).toHaveLength(LIMITS.requests);
    expect(entries[0]).toMatchObject({ method: "GET", path: "/tasks?query=…", status: 200 });
    expect(entries.at(-1)).toMatchObject({ method: "POST", path: "/fail/123?x=…", status: 404, error: "NOT_FOUND", request_id: "req-9" });
    expect(JSON.stringify(entries)).not.toContain("hunter2");
    expect(JSON.stringify(entries)).not.toContain("secret-");
  });

  it("record navigations once per change, capped at 20", () => {
    for (let i = 0; i < 25; i++) { recordNavigation("tasks", null, `/?view=tasks&q=x${i}&p=${i}`); recordNavigation("tasks", null, `/?view=tasks&q=x${i}&p=${i}`); }
    recordNavigation("remotea", "r-1", "/?view=remotea&run=r-1");
    recordNavigation("remotea", "r-1", "/?view=remotea&run=r-1");
    const entries = rings.navigations.list();
    expect(entries).toHaveLength(2); // query values are dropped, so the task pages read as one
    for (let i = 0; i < 25; i++) recordNavigation("view-" + i, null, "/");
    expect(rings.navigations.list()).toHaveLength(LIMITS.navigations);
    expect(entries.at(-1)).toMatchObject({ view: "remotea", url: "/?view=remotea&run=r-1", run_id: "r-1" });
    expect(entries.at(-2)).toMatchObject({ url: "/?view=tasks&q=…&p=…" });
  });
});

describe("the context", () => {
  it("names the page, build and device and carries the buffers", async () => {
    history.replaceState(null, "", "/?view=remotea&run=r-1&code=oauth-secret");
    rings.console.push({ level: "error", message: "x", at: "t" });
    const ctx = await collectContext({ view: "remotea", runId: "r-1", taskId: "t-1" });
    expect(ctx.page).toMatchObject({ view: "remotea", run_id: "r-1", task_id: "t-1", params: { view: "remotea", run: "r-1", code: "…" } });
    expect(ctx.page.url).toContain("?view=remotea&run=r-1&code=…");
    expect(ctx.build.version).toBeTruthy();
    expect(ctx.device).toMatchObject({ online: true, time_zone: expect.any(String), viewport: expect.stringMatching(/^\d+x\d+$/) });
    expect(ctx.errors).toHaveLength(1);
    expect(JSON.stringify(ctx)).not.toContain("oauth-secret");
    history.replaceState(null, "", "/");
  });
});

describe("Report a bug", () => {
  const shot = "data:image/jpeg;base64," + btoa("\xff\xd8\xffscreenshot");
  function stubFetch() {
    const calls: { method: string; url: string; body: Record<string, unknown> }[] = [];
    vi.stubGlobal("fetch", async (url: string, init: RequestInit = {}) => {
      calls.push({ method: init.method ?? "GET", url, body: init.body ? JSON.parse(String(init.body)) : undefined });
      return new Response(JSON.stringify({ id: "rep-1", task_id: "task-1", title: "Broken" }), { status: 201 });
    });
    return calls;
  }
  const button = () => host.querySelector<HTMLButtonElement>('button[aria-label="Report a bug"]');
  function type(el: HTMLTextAreaElement, value: string) {
    Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")!.set!.call(el, value);
    el.dispatchEvent(new Event("input", { bubbles: true }));
  }

  it("captures first, then posts the description, context and screenshot, and links the task", async () => {
    const calls = stubFetch(), opened = vi.fn();
    let seenButton: boolean | null = null;
    const capture = vi.fn(async () => { seenButton = !!button(); return { url: shot, error: null }; });
    act(() => root.render(<BugReportButton page={() => ({ view: "tasks", taskId: "t-9" })} onOpenTask={opened} captureShot={capture}/>));
    expect(button()!.className).toBe("bug-report-launcher");
    act(() => button()!.click());
    await settle(); await settle();
    expect(capture).toHaveBeenCalledOnce();
    expect(seenButton).toBe(false); // hidden while the page is drawn
    const dialog = host.querySelector('[role="dialog"]')!;
    expect(dialog.textContent).toContain("What went wrong?");
    expect(button()).toBeNull();
    expect(dialog.querySelector("img")!.getAttribute("src")).toBe(shot);
    expect(dialog.textContent).toContain("Details that will be attached");
    const send = [...dialog.querySelectorAll("button")].find((b) => b.textContent === "Send")!;
    expect(send.disabled).toBe(true);
    const [what, expected] = dialog.querySelectorAll("textarea");
    act(() => { type(what, "Run drawer is blank"); type(expected, "The timeline"); });
    act(() => send.click());
    await settle();
    expect(calls).toHaveLength(1);
    expect(calls[0]).toMatchObject({ method: "POST", url: "/api/v1/bug-reports", body: {
      description: "Run drawer is blank", expected: "The timeline", screenshot: shot, screenshot_error: null,
      context: { page: { view: "tasks", task_id: "t-9" } },
    } });
    expect(host.querySelector('[role="dialog"]')).toBeNull();
    expect(host.querySelector('[role="status"]')!.textContent).toContain("Reported. Thanks");
    act(() => [...host.querySelectorAll("button")].find((b) => b.textContent === "Open task")!.click());
    expect(opened).toHaveBeenCalledWith("task-1");
    expect(button()).not.toBeNull();
  });

  it("still sends when capture fails or the screenshot is left out", async () => {
    const calls = stubFetch();
    act(() => root.render(<BugReportButton page={() => ({ view: "today" })} onOpenTask={() => {}}
      captureShot={async () => ({ url: null, error: "capture failed: tainted canvas" })}/>));
    act(() => window.dispatchEvent(new KeyboardEvent("keydown", { altKey: true, shiftKey: true, code: "KeyB", key: "B" })));
    await settle(); await settle();
    const dialog = host.querySelector('[role="dialog"]')!;
    expect(dialog.textContent).toContain("No screenshot: capture failed: tainted canvas");
    act(() => type(dialog.querySelector("textarea")!, "Something"));
    act(() => [...dialog.querySelectorAll("button")].find((b) => b.textContent === "Send")!.click());
    await settle();
    expect(calls[0].body).toMatchObject({ screenshot: null, screenshot_error: "capture failed: tainted canvas" });
  });

  it("uses Alt+Shift+B and nothing else", () => {
    expect(isReportShortcut({ altKey: true, shiftKey: true, ctrlKey: false, metaKey: false, code: "KeyB" })).toBe(true);
    expect(isReportShortcut({ altKey: false, shiftKey: true, ctrlKey: true, metaKey: false, code: "KeyB" })).toBe(false);
  });

  it("is mounted by the shell on every page, beside Eri's launcher", () => {
    expect(appSource).toMatch(/<BugReportButton [\s\S]*?\/>\s*<ChatLauncher /);
    expect(dataUrlBytes("data:image/jpeg;base64,AAAA")).toBe(3);
  });
});
