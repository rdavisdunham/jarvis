import { useEffect, useRef } from "react";

type Entry<T> = {snapshot: T; key: string; page: string; layers: string[]; url: string; scroll: number};
const frames = () => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve())));

/** Keep browser Back inside the SPA. Only opaque indices, never record bodies, enter history.state. */
export function useAppHistory<T>({enabled, snapshot, page, layers, url, restore, onError}: {
  enabled: boolean; snapshot: T; page: string; layers: string[]; url: string;
  restore: (state: T) => Promise<void>; onError: (message: string) => void;
}) {
  const latest = useRef({snapshot, page, layers, url, restore, onError});
  latest.current = {snapshot, page, layers, url, restore, onError};
  const store = useRef({session: crypto.randomUUID(), entries: new Map<number, Entry<T>>(), index: 0,
    initialized: false, restoring: false, rollback: false, consuming: null as number | null, pending: null as number | null});
  useEffect(() => {
    if (!enabled) {
      // Never carry navigation snapshots into a subsequent sign-in.
      store.current = {session: crypto.randomUUID(), entries: new Map(), index: 0,
        initialized: false, restoring: false, rollback: false, consuming: null, pending: null};
      return;
    }
    const state = store.current;
    const scroll = () => {
      const entry = state.entries.get(state.index);
      if (entry && !state.restoring) entry.scroll = document.querySelector(".content")?.scrollTop ?? window.scrollY;
    };
    const pop = async (event: PopStateEvent) => {
      const marker = event.state?.eriNavigation;
      if (!marker || marker.session !== state.session || !state.entries.has(marker.index)) return;
      if (state.rollback) {state.rollback = false; state.restoring = false; return;}
      if (state.consuming === marker.index) {
        // The UI already closed the layer. Do not replay old filters over an Eri navigation.
        state.consuming = null; state.index = marker.index;
        const value = latest.current;
        state.entries.set(state.index, {snapshot: value.snapshot, page: value.page, layers: [...value.layers],
          key: value.page + "|" + value.layers.join("|"), url: value.url,
          scroll: state.entries.get(state.index)?.scroll ?? 0});
        history.replaceState({...history.state, eriNavigation: {session: state.session, index: state.index}}, "", value.url);
        state.restoring = false;
        return;
      }
      state.consuming = null;
      state.pending = marker.index;
      if (state.restoring) return;
      state.restoring = true;
      while (state.pending !== null) {
        const targetIndex = state.pending;
        state.pending = null;
        const target = state.entries.get(targetIndex)!;
        try {
          await latest.current.restore(target.snapshot);
          await frames();
          state.index = targetIndex;
          document.querySelector(".content")?.scrollTo({top: target.scroll, behavior: "instant"});
          window.scrollTo({top: target.scroll, behavior: "instant"});
        } catch (error) {
          latest.current.onError((error as Error).message || "Finish saving before leaving this view.");
          const actual = history.state?.eriNavigation?.index ?? targetIndex;
          state.pending = null;
          if (actual !== state.index) {state.rollback = true; history.go(state.index - actual); return;}
        }
      }
      state.restoring = false;
    };
    window.addEventListener("popstate", pop);
    document.addEventListener("scroll", scroll, true);
    const oldRestoration = history.scrollRestoration;
    history.scrollRestoration = "manual";
    return () => {window.removeEventListener("popstate", pop); document.removeEventListener("scroll", scroll, true); history.scrollRestoration = oldRestoration;};
  }, [enabled]);
  // Defer one frame so editor mount/unmount and page state settle as one navigation.
  useEffect(() => {
    if (!enabled) return;
    const frame = requestAnimationFrame(() => {
      const state = store.current;
      if (state.restoring) return;
      const value = latest.current;
      const key = value.page + "|" + value.layers.join("|");
      const previous = state.entries.get(state.index);
      const entry: Entry<T> = {snapshot: value.snapshot, key, page: value.page, layers: [...value.layers], url: value.url, scroll: previous?.scroll ?? 0};
      const mark = (index: number) => ({...history.state, eriNavigation: {session: state.session, index}});
      if (!state.initialized) {
        state.initialized = true;
        state.entries.set(0, entry);
        history.replaceState(mark(0), "", value.url);
      } else if (previous?.key === key) {
        state.entries.set(state.index, entry);
        history.replaceState(mark(state.index), "", value.url);
      } else {
        // Closing a sheet with X should consume its entry, just like browser Back.
        const closing = previous?.page === value.page && previous.layers.length > value.layers.length
          && value.layers.every(layer => previous.layers.includes(layer));
        const ancestor = closing ? [...state.entries].reverse().find(([i, e]) => i < state.index && e.key === key) : undefined;
        if (ancestor) {state.consuming = ancestor[0]; state.restoring = true; history.go(ancestor[0] - state.index); return;}
        for (const index of state.entries.keys()) if (index > state.index) state.entries.delete(index);
        state.index++;
        entry.scroll = previous?.page === value.page ? previous.scroll : 0;
        state.entries.set(state.index, entry);
        history.pushState(mark(state.index), "", value.url);
        if (previous?.page !== value.page) {document.querySelector(".content")?.scrollTo(0,0); window.scrollTo(0,0);}
      }
    });
    return () => cancelAnimationFrame(frame);
  });
}
