import { useEffect, useState } from "react";

/** Dani at work, in the chat: a thinking bubble until her reply starts, then the reply as it is written.
 *
 * The worker streams the model's text (work_stream.py); /api/v1/events passes it on as `work_delta` events and
 * events.ts re-dispatches each as `eri-work-delta`. The preview is never stored: when the request finishes, the
 * saved reply (mergeWorkReplies) takes its place. A new model round (after tool calls) starts a fresh preview. */

export type WorkDelta = { w: string; r: number; s: number; t: string };
type Stream = { round: number; text: string };
export const WORK_DELTA = "eri-work-delta";

/** Streamed text by work id, for the given still-active requests (others are dropped). */
export function useReplyStreams(activeIds: string[]): Record<string, string> {
  const [streams, setStreams] = useState<Record<string, Stream>>({});
  const key = activeIds.join(",");
  useEffect(() => {
    const listen = (event: Event) => {
      const delta = (event as CustomEvent<WorkDelta>).detail;
      if (!delta || typeof delta.w !== "string" || typeof delta.t !== "string") return;
      setStreams(current => {
        const previous = current[delta.w];
        if (previous && delta.r < previous.round) return current; // A late batch from an earlier round.
        const text = previous && previous.round === delta.r ? previous.text + delta.t : delta.t;
        return { ...current, [delta.w]: { round: delta.r, text } };
      });
    };
    window.addEventListener(WORK_DELTA, listen);
    return () => window.removeEventListener(WORK_DELTA, listen);
  }, []);
  useEffect(() => {
    const keep = new Set(key ? key.split(",") : []);
    setStreams(current => Object.keys(current).every(id => keep.has(id)) ? current
      : Object.fromEntries(Object.entries(current).filter(([id]) => keep.has(id))));
  }, [key]);
  return Object.fromEntries(Object.entries(streams).map(([id, s]) => [id, s.text]));
}

export function ThinkingDots() {
  return <span className="thinking-dots" role="status" aria-label="Dani is thinking"><span /><span /><span /></span>;
}

/** One bubble per request Dani is still working on: thinking dots, then her reply streaming in. `sending`
 * covers the moment before the request exists. */
export function PendingReplies({ ids, streams, sending }: { ids: string[]; streams: Record<string, string>; sending: boolean }) {
  const bubbles = ids.length ? ids : sending ? ["sending"] : [];
  return <>{bubbles.map(id => {
    const text = streams[id]?.trimStart() ?? "";
    return <div className="message assistant pending-reply" key={"pending:" + id} aria-live="polite" aria-busy="true">
      <span className="message-label"><span className="eri-orb" aria-hidden="true" />Dani</span>
      {text ? <p>{text}<span className="stream-caret" aria-hidden="true" /></p> : <ThinkingDots />}
    </div>;
  })}</>;
}
