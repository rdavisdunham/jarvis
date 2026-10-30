import { useEffect } from "react";
import { REMOTEA_POLL_MS } from "./remotea-model";

/** Refresh on agent/session change-feed signals (coalesced), with a slower poll as the fallback.
 * Eager (the run panel, row buttons and the Runs badge use it), so it must not import Remotea.tsx. */
export function useAgentRefresh(refresh: () => void) {
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | null = null;
    const soon = () => { if (!timer) timer = setTimeout(() => { timer = null; refresh(); }, 300); };
    const poll = window.setInterval(() => { if (document.visibilityState === "visible") refresh(); }, REMOTEA_POLL_MS);
    window.addEventListener("eri-agents-changed", soon);
    document.addEventListener("visibilitychange", soon);
    return () => {
      if (timer) clearTimeout(timer);
      window.clearInterval(poll);
      window.removeEventListener("eri-agents-changed", soon);
      document.removeEventListener("visibilitychange", soon);
    };
  }, [refresh]);
}
