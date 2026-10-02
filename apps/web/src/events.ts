// Server change feed (SSE) plus the coalesced workspace loader that consumes it.

export type EventSignal =
  | { reason: "event"; id: number | null; kind: string; entity_id?: string }
  | { reason: "open" | "refresh" };

// The server writes an `event: refresh` roughly every 20 s (and an SSE comment heartbeat
// every ~5 s, which EventSource never surfaces). Silence well past that window means a
// half-open connection that the browser has not noticed, so reconnect explicitly.
export const EVENT_SILENCE_MS = 50_000;

let streamOnline = false;
/** True while an event stream is open and has been heard from recently. */
export function eventStreamOnline() {
  return streamOnline;
}
function setOnline(online: boolean) {
  if (streamOnline === online) return;
  streamOnline = online;
  if (typeof window !== "undefined") window.dispatchEvent(new CustomEvent("eri-events-status", { detail: online }));
}

// Explicit retries also recover from HTTP errors for which EventSource stops retrying.
export function subscribeEvents(
  after: number,
  refresh: (signal: EventSignal) => void,
  connection: (online: boolean) => void,
) {
  let cursor = after;
  let stopped = false;
  let delay = 1000;
  let stream: EventSource | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let silence: ReturnType<typeof setTimeout> | null = null;
  const report = (online: boolean) => {
    setOnline(online);
    connection(online);
  };
  const connect = () => {
    if (stopped) return;
    const current = new EventSource("/api/v1/events?after=" + cursor);
    stream = current;
    const active = () => !stopped && stream === current;
    const fail = () => {
      if (!active()) return;
      if (silence) clearTimeout(silence);
      silence = null;
      current.close();
      stream = null;
      report(false);
      timer = setTimeout(connect, delay);
      delay = Math.min(delay * 2, 15000);
    };
    const heard = () => {
      if (silence) clearTimeout(silence);
      silence = setTimeout(fail, EVENT_SILENCE_MS);
    };
    heard();
    current.onopen = () => {
      if (!active()) return;
      heard();
      delay = 1000;
      report(true);
      refresh({ reason: "open" });
    };
    current.onmessage = (event) => {
      if (!active()) return;
      heard();
      const received = Number(event.lastEventId);
      const id = event.lastEventId && Number.isSafeInteger(received) ? received : null;
      if (id !== null) cursor = Math.max(cursor, id);
      let payload: { kind?: unknown; entity_id?: unknown } = {};
      try {
        payload = JSON.parse(event.data || "{}") ?? {};
      } catch { /* malformed payloads still trigger a full refresh below */ }
      const kind = typeof payload.kind === "string" ? payload.kind : "";
      if (kind === "work.changed") {
        window.dispatchEvent(new Event("eri-work-changed"));
        return;
      }
      if (kind === "membership.changed")
        window.dispatchEvent(new Event("eri-accounts-changed"));
      refresh({
        reason: "event",
        id,
        kind,
        ...(typeof payload.entity_id === "string" ? { entity_id: payload.entity_id } : {}),
      });
    };
    current.addEventListener("refresh", () => {
      if (!active()) return;
      heard();
      refresh({ reason: "refresh" });
    });
    current.addEventListener("access_revoked", (event) => {
      if (!active()) return;
      if (silence) clearTimeout(silence);
      current.close();
      stopped = true;
      setOnline(false);
      let code = "ACCESS_REVOKED";
      try {
        code = JSON.parse((event as MessageEvent).data || "{}").code || code;
      } catch { /* keep the generic code */ }
      window.dispatchEvent(new CustomEvent("eri-access-ended", { detail: code }));
    });
    current.onerror = fail;
  };
  connect();
  return () => {
    stopped = true;
    if (timer) clearTimeout(timer);
    if (silence) clearTimeout(silence);
    stream?.close();
    stream = null;
    setOnline(false);
  };
}

// Which parts of the client a change-feed signal invalidates.
const LOAD_SKIP = new Set(["work.changed", "memory.changed", "note.indexed", "note.changed"]);
const REVISION_SKIP = new Set(["work.changed", "memory.changed", "notification.changed", "membership.changed"]);
export function signalScope(signal: EventSignal) {
  if (signal.reason === "open") return { load: true, revision: true, memory: true };
  if (signal.reason !== "event") return { load: true, revision: false, memory: false };
  if (!signal.kind) return { load: true, revision: true, memory: true };
  return {
    load: !LOAD_SKIP.has(signal.kind),
    revision: !REVISION_SKIP.has(signal.kind),
    memory: signal.kind === "memory.changed",
  };
}

export type LoadRequest = {
  /** Also bump dependent view revisions (notes, records, integrations). */
  revision?: boolean;
  /** Change-feed id that caused the request; already-loaded ids are dropped. */
  cursor?: number | null;
  /** Skip the debounce window (explicit user actions). */
  immediate?: boolean;
};
type Waiter = { request: LoadRequest; resolve: () => void; reject: (error: unknown) => void };
export type LoadRun = (options: { revision: boolean; current: () => boolean }) => Promise<number | void>;

/**
 * Single-flight, debounced loader. Requests arriving inside the debounce window or while a
 * load runs collapse into one trailing load. Each run returns the change-feed cursor its
 * snapshot covers; queued feed signals at or below it are already reflected and are dropped,
 * which also absorbs the echo of this client's own writes. `cancel()` invalidates the
 * running load so its late response is discarded (e.g. on a workspace change).
 */
export function createLoader(run: LoadRun, delay = 250) {
  let generation = 0;
  let running = false;
  let rerun = false;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let queue: Waiter[] = [];
  let loaded = -1;
  let revised = -1;
  const covered = (request: LoadRequest) =>
    request.cursor !== undefined && request.cursor !== null &&
    request.cursor <= loaded && (!request.revision || request.cursor <= revised);
  const settleCovered = () => {
    const done = queue.filter((w) => covered(w.request));
    queue = queue.filter((w) => !covered(w.request));
    done.forEach((w) => w.resolve());
  };
  const start = () => {
    if (timer) clearTimeout(timer);
    timer = null;
    if (running) {
      rerun = true;
      return;
    }
    settleCovered();
    if (!queue.length) return;
    const batch = queue;
    queue = [];
    rerun = false;
    running = true;
    const mine = ++generation;
    const revision = batch.some((w) => w.request.revision);
    const current = () => mine === generation;
    let outcome: Promise<number | void>;
    try {
      outcome = run({ revision, current });
    } catch (error) {
      outcome = Promise.reject(error);
    }
    outcome.then(
      (cursor) => {
        if (current() && typeof cursor === "number") {
          loaded = Math.max(loaded, cursor);
          if (revision) revised = Math.max(revised, cursor);
        }
        batch.forEach((w) => w.resolve());
      },
      (error) => batch.forEach((w) => w.reject(error)),
    ).finally(() => {
      running = false;
      if (rerun) start();
      else settleCovered();
    });
  };
  return {
    request(request: LoadRequest = {}) {
      if (covered(request)) return Promise.resolve();
      return new Promise<void>((resolve, reject) => {
        queue.push({ request, resolve, reject });
        if (request.immediate) start();
        else if (!timer && !(running && rerun)) timer = setTimeout(start, delay);
      });
    },
    cancel() {
      generation++;
      if (timer) clearTimeout(timer);
      timer = null;
      rerun = false;
      const pending = queue;
      queue = [];
      pending.forEach((w) => w.resolve());
    },
    /** Forget covered cursors, e.g. when the event scope changes. */
    reset() {
      loaded = -1;
      revised = -1;
    },
    get running() {
      return running;
    },
  };
}

// Device-bridge (/ui/sync) cadence. Server windows: screen context lives 60 s, a
// dispatched screen action 12 s with a ~10 s acknowledgement wait (device_bridge.py).
export const BRIDGE_ACTIVE_MS = 700;
export const BRIDGE_IDLE_MS = 3000;
export const BRIDGE_HIDDEN_MS = 20000;
export function bridgeInterval(active: boolean, hidden: boolean) {
  if (active) return BRIDGE_ACTIVE_MS;
  return hidden ? BRIDGE_HIDDEN_MS : BRIDGE_IDLE_MS;
}

/** Background poll cadence for views that also receive change-feed signals. */
export function fallbackInterval(online: boolean, fast: number, slow = 30000) {
  return online ? slow : fast;
}
