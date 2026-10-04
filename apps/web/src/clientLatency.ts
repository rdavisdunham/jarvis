import { post } from "./api";

// Only requests sent in this tab are measured. Reloaded history, other devices,
// background tabs and requests older than five minutes stay unmeasured.
const starts = new Map<string, number>();
const reported = new Set<string>();
const MAX_AGE_MS = 300_000;

export function chatSendStarted(requestId: string) {
  const now = performance.now();
  for (const [key, at] of starts) if (now - at > MAX_AGE_MS) starts.delete(key);
  if (starts.size >= 200) starts.delete(starts.keys().next().value!);
  starts.set(requestId, now);
}

export async function reportWorkRendered(requestId: string, revision: number,
  stage: "work_card_rendered" | "work_reply_rendered" = "work_card_rendered") {
  const key = requestId + ":" + revision;
  const start = starts.get(requestId);
  if (start === undefined || reported.has(key) || document.visibilityState !== "visible") return;
  const elapsed = performance.now() - start;
  if (elapsed < 0 || elapsed > MAX_AGE_MS) { starts.delete(requestId); return; }
  if (reported.size >= 400) reported.delete(reported.values().next().value!);
  reported.add(key);
  try {
    await post("/work/" + encodeURIComponent(requestId) + "/latency", {
      stage, revision, elapsed_ms: elapsed,
    });
    starts.delete(requestId);
  } catch {
    reported.delete(key);
  }
}

export function observeWorkRendered(element: HTMLElement, requestId: string, revision: number,
  stage: "work_card_rendered" | "work_reply_rendered" = "work_card_rendered") {
  if (!starts.has(requestId) || typeof IntersectionObserver === "undefined") return () => {};
  let first = 0, second = 0, visible = false, stopped = false;
  const schedule = () => {
    cancelAnimationFrame(first); cancelAnimationFrame(second);
    if (!visible || stopped || document.visibilityState !== "visible") return;
    first = requestAnimationFrame(() => {
      second = requestAnimationFrame(() => {
        if (!stopped && visible && element.isConnected && document.visibilityState === "visible")
          void reportWorkRendered(requestId, revision, stage);
      });
    });
  };
  const observer = new IntersectionObserver(entries => {
    visible = entries.some(entry => entry.isIntersecting);
    schedule();
  });
  observer.observe(element);
  document.addEventListener("visibilitychange", schedule);
  return () => {
    stopped = true; observer.disconnect();
    cancelAnimationFrame(first); cancelAnimationFrame(second);
    document.removeEventListener("visibilitychange", schedule);
  };
}
