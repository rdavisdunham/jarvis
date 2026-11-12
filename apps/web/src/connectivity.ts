/**
 * Can this tab reach the Eridani API? An outage (no network, or the proxy/API answering 5xx) is not a
 * sign-out: the signed-in UI stays and a banner says "Can't reach Eridani" while this module probes with
 * backoff. api.ts reports every request's outcome here, and a failed
 * request is confirmed by one probe before it counts; a real 401 is handled separately (session expiry).
 */
export type Connectivity = {
  reachable: boolean;
  /** When the next automatic probe runs (epoch ms), while unreachable. */
  nextProbeAt: number | null;
  /** Consecutive failed probes since the outage began. */
  attempts: number;
};

// Ready means the API answers and its database is reachable (api.py /health/ready).
export const PROBE_URL = "/health/ready";
export const BACKOFF_MS = [1000, 2000, 4000, 8000, 15000, 30000];
export const backoff = (attempt: number) => BACKOFF_MS[Math.min(attempt, BACKOFF_MS.length - 1)];

let state: Connectivity = { reachable: true, nextProbeAt: null, attempts: 0 };
const listeners = new Set<(state: Connectivity) => void>();
let timer: ReturnType<typeof setTimeout> | null = null;
let probing = false;
let probe: () => Promise<boolean> = defaultProbe;
// A failed request is checked against the probe before it counts as an outage (reportFailure).
export const CONFIRM_MS = 5000;
let confirming = false;
let generation = 0;

async function defaultProbe() {
  try {
    const response = await fetch(PROBE_URL, { credentials: "same-origin", cache: "no-store" });
    if (response.status >= 500) return false;
    await response.json();
    return true;
  } catch {
    return false;
  }
}

function set(next: Connectivity) {
  state = next;
  listeners.forEach((listener) => listener(state));
}

function schedule() {
  if (timer) clearTimeout(timer);
  const delay = backoff(state.attempts);
  timer = setTimeout(() => void probeNow(), delay);
  set({ ...state, nextProbeAt: Date.now() + delay });
}

export function connectivity() {
  return state;
}

export function subscribeConnectivity(listener: (state: Connectivity) => void) {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
}

/** A request got an answer from the API (any status below 500). */
export function reportReachable() {
  if (state.reachable) return;
  if (timer) clearTimeout(timer);
  timer = null;
  set({ reachable: true, nextProbeAt: null, attempts: 0 });
}

/** The API is down: a failed request that the probe confirmed (reportFailure). */
export function reportUnreachable() {
  if (!state.reachable) return;
  set({ reachable: false, nextProbeAt: null, attempts: 0 });
  schedule();
}

/**
 * One request failed at the network or got a 5xx. That alone is not an outage: a single dropped request
 * or one failing endpoint must not raise the banner and the full reload that follows recovery. Ask the
 * probe first, and declare the outage only when it also fails (or gives no answer within CONFIRM_MS).
 */
export async function reportFailure() {
  if (!state.reachable || confirming) return;
  confirming = true;
  const started = generation;
  let timeout: ReturnType<typeof setTimeout> | undefined;
  let up = false;
  try {
    up = await Promise.race([
      probe(),
      new Promise<boolean>((resolve) => { timeout = setTimeout(() => resolve(false), CONFIRM_MS); }),
    ]);
  } finally {
    clearTimeout(timeout);
    if (started === generation) confirming = false;
  }
  if (started === generation && !up) reportUnreachable();
}

/** Probe now (the banner's Retry button, or the backoff timer). */
export async function probeNow() {
  if (state.reachable || probing) return;
  probing = true;
  if (timer) clearTimeout(timer);
  timer = null;
  try {
    if (await probe()) reportReachable();
    else if (!state.reachable) {
      set({ ...state, attempts: state.attempts + 1 });
      schedule();
    }
  } finally {
    probing = false;
  }
}

/** Tests: replace the probe and reset to reachable. */
export function resetConnectivity(replacement: () => Promise<boolean> = defaultProbe) {
  if (timer) clearTimeout(timer);
  timer = null;
  probing = false;
  confirming = false;
  generation += 1;
  probe = replacement;
  state = { reachable: true, nextProbeAt: null, attempts: 0 };
}
