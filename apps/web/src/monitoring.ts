/// <reference types="vite/client" />
/**
 * Browser error reporting (Sentry). Off unless a DSN is configured: VITE_SENTRY_DSN at build time, or
 * JARVIS_SENTRY_WEB_DSN served by /bootstrap (`sentry_dsn`). The SDK is a separate chunk that loads only
 * then. Reports never carry cookies, headers, request bodies, query values or console text, and the API's
 * CSP allows the DSN's ingest host only when that DSN is set (observability.py).
 */
import type { Breadcrumb, ErrorEvent } from "@sentry/react";
import { buildIdentity, redactText, redactUrl } from "./bug-report";

type Sentry = typeof import("@sentry/react");
let sdk: Sentry | null = null;
let starting: Promise<boolean> | null = null;

export const buildTimeDsn = (): string => import.meta.env.VITE_SENTRY_DSN?.trim() ?? "";

const cleanUrl = (url: string) => {
  try {
    const parsed = new URL(url, location.origin);
    return parsed.origin === location.origin ? redactUrl(parsed.pathname + parsed.search) : parsed.origin + parsed.pathname;
  } catch {
    return redactText(url.split("?")[0]);
  }
};

export function scrubEvent(event: ErrorEvent): ErrorEvent {
  if (event.request) {
    const { url } = event.request;
    event.request = { ...(url ? { url: cleanUrl(url) } : {}) };
  }
  if (event.user) event.user = event.user.id ? { id: event.user.id } : {};
  if (event.breadcrumbs) event.breadcrumbs = event.breadcrumbs.map(scrubBreadcrumb).filter((b): b is Breadcrumb => !!b);
  return event;
}

export function scrubBreadcrumb(crumb: Breadcrumb): Breadcrumb | null {
  // Console text can hold planner content; the Report a bug flow is where that is shared, by choice.
  if (crumb.category === "console") return null;
  if (!crumb.data) return crumb;
  const data: Record<string, unknown> = { ...crumb.data };
  for (const key of ["url", "from", "to"]) if (typeof data[key] === "string") data[key] = cleanUrl(data[key] as string);
  delete data["http.query"];
  delete data["http.fragment"];
  delete data.body;
  return { ...crumb, data };
}

/** Start reporting once; later DSNs are ignored. Resolves whether reporting is on. */
export function startMonitoring(dsn?: string | null): Promise<boolean> {
  const value = (dsn ?? "").trim();
  if (starting) return starting;
  if (!value) return Promise.resolve(false);
  starting = import("@sentry/react").then((Sentry) => {
    const build = buildIdentity();
    Sentry.init({
      dsn: value,
      release: build.commit !== "unknown" ? build.commit : undefined,
      environment: build.mode,
      dataCollection: {
        userInfo: false, cookies: false, httpHeaders: false, httpBodies: [], urlQueryParams: false, stackFrameVariables: false,
      },
      // Errors only: no tracing, replay or profiling.
      beforeSend: scrubEvent,
      beforeBreadcrumb: scrubBreadcrumb,
    });
    sdk = Sentry;
    return true;
  }).catch(() => {
    starting = null;
    return false;
  });
  return starting;
}

/** Report a caught error (error boundaries). A no-op while reporting is off. */
export function captureError(error: unknown, context: { boundary?: string; componentStack?: string | null } = {}) {
  if (!sdk) return;
  sdk.captureException(error, {
    tags: context.boundary ? { boundary: context.boundary } : undefined,
    contexts: context.componentStack ? { react: { componentStack: context.componentStack } } : undefined,
  });
}

/** Tests only. */
export function resetMonitoring() {
  sdk = null;
  starting = null;
}
