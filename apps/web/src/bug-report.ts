/**
 * Diagnostics for the Report a bug button (BugReport.tsx): ring buffers installed at app start and the
 * context collector. Everything here leaves secrets out: no headers, cookies, tokens, request or
 * response bodies, query values (except the deep-link parameters below) or storage values that look
 * like keys. The server scrubs the context again (`bug_reports.scrub`); docs/APP_FUNCTIONALITY.md
 * "Bug reports" lists what is sent.
 */
import { readTheme } from "./theme";

export class Ring<T> {
  private items: T[] = [];
  constructor(readonly cap: number) {}
  push(item: T) {
    this.items.push(item);
    if (this.items.length > this.cap) this.items.splice(0, this.items.length - this.cap);
  }
  list() {
    return [...this.items];
  }
  clear() {
    this.items = [];
  }
}

export type ConsoleEntry = { level: "error" | "warn" | "uncaught" | "unhandledrejection"; message: string; stack?: string; source?: string; at: string };
export type RequestEntry = { method: string; path: string; status: number; ms: number; error?: string; request_id?: string; at: string };
export type NavigationEntry = { view: string; url: string; run_id?: string | null; at: string };

export const LIMITS = { console: 50, requests: 40, navigations: 20 } as const;
export const rings = {
  console: new Ring<ConsoleEntry>(LIMITS.console),
  requests: new Ring<RequestEntry>(LIMITS.requests),
  navigations: new Ring<NavigationEntry>(LIMITS.navigations),
};

// A client copy of apps/api/jarvis/redaction.py PATTERNS (best effort; the server applies the canonical list).
const SECRET = new RegExp([
  String.raw`sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}`,
  String.raw`gh[pousr]_[A-Za-z0-9]{30,}`,
  String.raw`github_pat_[A-Za-z0-9_]{30,}`,
  String.raw`AKIA[0-9A-Z]{16}`,
  String.raw`xox[abpr]-[A-Za-z0-9-]{10,}`,
  String.raw`eri_bot_[A-Za-z0-9_-]{20,}`,
  String.raw`lin_api_[A-Za-z0-9]{20,}`,
  String.raw`[Bb][Ee][Aa][Rr][Ee][Rr]\s+[A-Za-z0-9._~+/=-]{20,}`,
  String.raw`eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}`,
  String.raw`-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|(?![\s\S]))`,
  String.raw`://[^/\s:@]+:[^/\s@]+@`,
  String.raw`(?:[Pp][Aa][Ss][Ss][Ww][Oo][Rr][Dd]|[Ss][Ee][Cc][Rr][Ee][Tt]|[Tt][Oo][Kk][Ee][Nn]|[Aa][Pp][Ii]_[Kk][Ee][Yy])\s*[=:]\s*\S{8,}`,
].join("|"), "g");
const SECRET_KEY = /authorization|cookie|token|secret|password|passwd|api[_-]?key|csrf|credential|bearer|private[_-]?key|session[_-]?id/i;
/** Long unbroken key-like strings (base64, hex, JWT pieces). */
const KEY_LIKE = /^[A-Za-z0-9_\-+/=.]{24,}$/;
/** Page parameters that only name a page, a section or a record id: kept in URLs, every other value is dropped. */
export const SAFE_PARAMS = new Set(["view", "run", "section", "activity", "task", "record", "note", "tab", "sharing", "google"]);

export const redactText = (text: string) => text.replace(SECRET, "[redacted]");
const trim = (text: string, max: number) => (text.length > max ? text.slice(0, max - 1) + "…" : text);

/** A path or URL with every query value replaced by "…" (parameter names and the path, ids included, stay). */
export function redactPath(path: string, keep: ReadonlySet<string> = new Set()) {
  return redactText(path).replace(/([?&])([^=&#\s]+)=([^&#\s]*)/g, (_m, sep: string, name: string, value: string) =>
    keep.has(name) && /^[\w.:-]{0,80}$/.test(value) ? `${sep}${name}=${value}` : `${sep}${name}=…`).replace(/#.*$/, "");
}
export const redactUrl = (url: string) => redactPath(url, SAFE_PARAMS);

function describe(value: unknown): string {
  if (value instanceof Error) return `${value.name}: ${value.message}`;
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value) ?? String(value);
  } catch {
    return String(value);
  }
}
const stackOf = (value: unknown) => (value instanceof Error && value.stack ? trim(redactText(value.stack.split("\n").slice(0, 10).join("\n")), 1500) : undefined);
const stamp = () => new Date().toISOString();

export function recordConsole(level: ConsoleEntry["level"], args: unknown[], source?: string) {
  const error = args.find((a) => a instanceof Error);
  rings.console.push({ level, message: trim(redactText(args.map(describe).join(" ")), 500), stack: stackOf(error), source, at: stamp() });
}

/** Called by api.ts for every request it makes. */
export function recordRequest(entry: Omit<RequestEntry, "at" | "path"> & { path: string }) {
  rings.requests.push({ ...entry, path: redactPath(entry.path), at: stamp() });
}

export function recordNavigation(view: string, runId?: string | null, href: string = location.pathname + location.search) {
  const last = rings.navigations.list().at(-1), url = redactUrl(href);
  if (last && last.view === view && last.url === url && (last.run_id ?? null) === (runId ?? null)) return;
  rings.navigations.push({ view, url, run_id: runId ?? null, at: stamp() });
}

let installed: (() => void) | null = null;
/** Install the console and error buffers once, at app start. Returns the uninstall function (tests). */
export function installDiagnostics(target: Window = window, con: Console = console) {
  if (installed) return installed;
  const originals = { error: con.error, warn: con.warn };
  for (const level of ["error", "warn"] as const) {
    con[level] = (...args: unknown[]) => {
      try {
        recordConsole(level, args);
      } catch {
        // Diagnostics never break logging.
      }
      originals[level].apply(con, args);
    };
  }
  const onError = (event: ErrorEvent) =>
    recordConsole("uncaught", [event.error ?? event.message], event.filename ? `${redactUrl(event.filename)}:${event.lineno}:${event.colno}` : undefined);
  const onRejection = (event: PromiseRejectionEvent) => recordConsole("unhandledrejection", [event.reason]);
  target.addEventListener("error", onError);
  target.addEventListener("unhandledrejection", onRejection);
  installed = () => {
    con.error = originals.error;
    con.warn = originals.warn;
    target.removeEventListener("error", onError);
    target.removeEventListener("unhandledrejection", onRejection);
    installed = null;
  };
  return installed;
}

declare const __ERIDANI_BUILD__: { version: string; commit: string; built_at: string; mode: string } | undefined;
export function buildIdentity() {
  return typeof __ERIDANI_BUILD__ === "undefined" ? { version: "unknown", commit: "unknown", built_at: "", mode: "unknown" } : __ERIDANI_BUILD__;
}

/** The service worker's VERSION (public/sw.js answers a `version` message), or null within 300 ms. */
async function serviceWorker() {
  const sw = typeof navigator !== "undefined" ? navigator.serviceWorker : undefined;
  const controller = sw?.controller;
  if (!controller) return { controlled: false };
  const version = await new Promise<string | null>((resolve) => {
    const channel = new MessageChannel(), timer = setTimeout(() => resolve(null), 300);
    channel.port1.onmessage = (event) => { clearTimeout(timer); resolve(typeof event.data === "string" ? event.data.slice(0, 40) : null); };
    try {
      controller.postMessage({ type: "version" }, [channel.port2]);
    } catch {
      clearTimeout(timer);
      resolve(null);
    }
  });
  return { controlled: true, state: controller.state, script: redactUrl(controller.scriptURL), version };
}

/** localStorage key names, with values only when short and not key-like. */
export function safeStorage(storage: Storage | undefined = typeof localStorage === "undefined" ? undefined : localStorage) {
  const out: Record<string, string> = {};
  try {
    for (let i = 0; storage && i < storage.length && i < 60; i++) {
      const key = storage.key(i);
      if (!key) continue;
      const value = storage.getItem(key) ?? "";
      out[trim(key, 80)] = SECRET_KEY.test(key) || KEY_LIKE.test(value) || redactText(value) !== value ? "[redacted]"
        : value.length > 60 ? `[${value.length} characters]` : value;
    }
  } catch {
    // Storage can be unavailable (private mode).
  }
  return out;
}

/** Names of the dialogs open right now (their label or heading). */
export function openDialogs(root: ParentNode = document) {
  return Array.from(root.querySelectorAll<HTMLElement>('[role="dialog"]'))
    .map((el) => {
      const labelled = el.getAttribute("aria-labelledby");
      return trim((el.getAttribute("aria-label") || (labelled && document.getElementById(labelled)?.textContent) || el.querySelector("h1,h2,h3")?.textContent || "dialog").trim(), 80);
    })
    .slice(0, 10);
}

export type PageState = { view: string; runId?: string | null; taskId?: string | null; recordId?: string | null };

export async function collectContext(page: PageState) {
  const params: Record<string, string> = {};
  new URLSearchParams(location.search).forEach((value, name) => {
    params[trim(name, 60)] = SAFE_PARAMS.has(name) && /^[\w.:-]{0,80}$/.test(value) ? value : "…";
  });
  const dark = document.documentElement.dataset.theme === "dark" ||
    (document.documentElement.dataset.theme !== "light" && typeof matchMedia === "function" && matchMedia("(prefers-color-scheme: dark)").matches);
  const standalone = typeof matchMedia === "function" && matchMedia("(display-mode: standalone)").matches;
  const zone = Intl.DateTimeFormat().resolvedOptions().timeZone;
  return {
    page: {
      view: page.view, url: redactUrl(location.origin + location.pathname + location.search), params,
      run_id: page.runId ?? null, task_id: page.taskId ?? null, record_id: page.recordId ?? null, dialogs: openDialogs(),
    },
    build: buildIdentity(),
    service_worker: await serviceWorker(),
    device: {
      user_agent: navigator.userAgent, platform: (navigator as Navigator & { userAgentData?: { platform?: string } }).userAgentData?.platform || navigator.platform,
      viewport: `${window.innerWidth}x${window.innerHeight}`, device_pixel_ratio: window.devicePixelRatio || 1,
      standalone, display_mode: standalone ? "standalone" : "browser", online: navigator.onLine,
      touch_points: navigator.maxTouchPoints ?? 0, theme: readTheme(), resolved_theme: dark ? "dark" : "light",
      locale: navigator.language, languages: [...(navigator.languages ?? [])].slice(0, 5),
      time: new Date().toISOString(), time_zone: zone, utc_offset_minutes: -new Date().getTimezoneOffset(),
    },
    storage: safeStorage(),
    errors: rings.console.list(),
    requests: rings.requests.list(),
    navigations: rings.navigations.list(),
  };
}
export type BugContext = Awaited<ReturnType<typeof collectContext>>;
