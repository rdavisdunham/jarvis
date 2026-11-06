/**
 * Can this tab reach the Eridani API? An outage (no network, or the proxy/API answering 5xx) is not a
 * sign-out: the signed-in UI stays and a banner says "Can't reach Eridani" while this module probes with
 * backoff. api.ts reports every request's outcome here; a real 401 is handled separately (session expiry).
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

/** A request failed at the network or got a 5xx / non-API answer from the proxy. */
export function reportUnreachable() {
  if (!state.reachable) return;
  set({ reachable: false, nextProbeAt: null, attempts: 0 });
  schedule();
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
  probe = replacement;
  state = { reachable: true, nextProbeAt: null, attempts: 0 };
}
