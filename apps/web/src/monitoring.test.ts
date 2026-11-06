// @vitest-environment jsdom
import { afterEach, expect, it, vi } from "vitest";
import type { ErrorEvent } from "@sentry/react";
import { captureError, resetMonitoring, scrubBreadcrumb, scrubEvent, startMonitoring } from "./monitoring";

afterEach(() => { resetMonitoring(); vi.restoreAllMocks(); });

it("stays off without a DSN and never loads the SDK", async () => {
  expect(await startMonitoring("")).toBe(false);
  expect(await startMonitoring(null)).toBe(false);
  expect(() => captureError(new Error("x"))).not.toThrow();
});

it("drops cookies, headers, bodies, query values and console text", () => {
  const event = {
    request: { url: location.origin + "/?view=tasks&code=oauth-secret", cookies: { jarvis_session: "t" }, headers: { Cookie: "a", "X-CSRF-Token": "c" }, data: "{\"token\":1}", query_string: "code=oauth-secret" },
    user: { id: "u1", ip_address: "203.0.113.9", email: "a@b.c" },
    breadcrumbs: [
      { category: "console", message: "Task 'see the dentist' failed" },
      { category: "fetch", data: { url: "/api/v1/search?q=dentist", method: "GET", status_code: 200, body: "x" } },
      { category: "navigation", data: { from: "/?view=tasks&q=secret", to: "/?view=today" } },
    ],
  } as unknown as ErrorEvent;
  const out = scrubEvent(event);
  expect(out.request).toEqual({ url: "/?view=tasks&code=…" });
  expect(out.user).toEqual({ id: "u1" });
  expect(out.breadcrumbs).toHaveLength(2);
  expect(out.breadcrumbs![0].data).toEqual({ url: "/api/v1/search?q=…", method: "GET", status_code: 200 });
  expect(out.breadcrumbs![1].data).toEqual({ from: "/?view=tasks&q=…", to: "/?view=today" });
  expect(scrubBreadcrumb({ category: "fetch", data: { url: "https://o1.ingest.sentry.io/api/1/envelope/?sentry_key=k" } })!.data!.url)
    .toBe("https://o1.ingest.sentry.io/api/1/envelope/");
});
