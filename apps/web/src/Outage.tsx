/** "Can't reach Eridani": the boot screen and the signed-in banner for an outage (connectivity.ts). */
import { useEffect, useState, useSyncExternalStore } from "react";
import { BrandMark } from "./ux";
import { connectivity, probeNow, subscribeConnectivity } from "./connectivity";

/** Only the reachable flag, so a large consumer (App) re-renders on outage start and end, not on every probe. */
export function useReachable() {
  return useSyncExternalStore(subscribeConnectivity, () => connectivity().reachable, () => true);
}

export function useConnectivity() {
  return useSyncExternalStore(subscribeConnectivity, connectivity, connectivity);
}

function useRetryIn(nextProbeAt: number | null) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (nextProbeAt === null) return;
    setNow(Date.now());
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [nextProbeAt]);
  return nextProbeAt === null ? null : Math.max(0, Math.ceil((nextProbeAt - now) / 1000));
}

const retryText = (seconds: number | null) =>
  seconds ? `Retrying in ${seconds} s.` : "Retrying now…";

/** Shown instead of the app when the first load could not reach the API; it is not a sign-out. */
export function OutageScreen() {
  const state = useConnectivity();
  const seconds = useRetryIn(state.nextProbeAt);
  return <div className="splash outage-screen" role="alert">
    <BrandMark size={44}/>
    <h1>Can't reach Eridani</h1>
    <p>Your connection or Eridani may be down. You are still signed in. {retryText(seconds)}</p>
    <button type="button" className="btn btn-primary" onClick={() => void probeNow()}>Retry now</button>
  </div>;
}

/** Shown above a signed-in page while requests fail; the page and anything typed stay put. */
export function OutageBanner() {
  const state = useConnectivity();
  const seconds = useRetryIn(state.nextProbeAt);
  if (state.reachable) return null;
  return <div className="error-banner outage-banner" role="status">
    <span>Can't reach Eridani. Changes you make may not save until it reconnects. {retryText(seconds)}</span>
    <button type="button" onClick={() => void probeNow()}>Retry now</button>
  </div>;
}
