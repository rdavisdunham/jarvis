// Explicit retries also recover from HTTP errors for which EventSource stops retrying.
export function subscribeEvents(
  after: number,
  refresh: () => void,
  connection: (online: boolean) => void,
) {
  let cursor = after;
  let stopped = false;
  let delay = 1000;
  let stream: EventSource | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;
  const connect = () => {
    if (stopped) return;
    const current = new EventSource("/api/v1/events?after=" + cursor);
    stream = current;
    const active = () => !stopped && stream === current;
    current.onopen = () => {
      if (!active()) return;
      delay = 1000;
      connection(true);
      refresh();
    };
    current.onmessage = (event) => {
      if (!active()) return;
      const received = Number(event.lastEventId);
      if (Number.isSafeInteger(received)) cursor = Math.max(cursor, received);
      try {
        if(JSON.parse(event.data || "{}").kind==="membership.changed")
          window.dispatchEvent(new Event("eri-accounts-changed"));
      } catch { /* reconnect refresh still runs for malformed event payloads */ }
      refresh();
    };
    current.addEventListener("refresh", () => {
      if (active()) refresh();
    });
    current.addEventListener("access_revoked", (event) => {
      if (!active()) return;
      current.close();
      stopped = true;
      window.dispatchEvent(
        new CustomEvent("eri-access-ended", {
          detail:
            JSON.parse((event as MessageEvent).data || "{}").code ||
            "ACCESS_REVOKED",
        }),
      );
    });
    current.onerror = () => {
      if (!active()) return;
      current.close();
      stream = null;
      connection(false);
      timer = setTimeout(connect, delay);
      delay = Math.min(delay * 2, 15000);
    };
  };
  connect();
  return () => {
    stopped = true;
    if (timer) clearTimeout(timer);
    stream?.close();
    stream = null;
  };
}
