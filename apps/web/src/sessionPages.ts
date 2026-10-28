/** Paginated session snapshots. A refresh replaces the loaded range without losing its depth. */
export type SessionPage<T> = { items: T[]; total: number; next_offset: number | null };
export type SessionPageState<T> = { data: SessionPage<T> | null; loading: boolean; error: string };

export function sessionPages<T extends { id: string }>(
  read: (offset: number, limit: number) => Promise<SessionPage<T>>,
  publish: (state: SessionPageState<T>) => void,
) {
  let active = true, loading = false;
  let data: SessionPage<T> | null = null;
  const emit = (error = "") => { if (active) publish({ data, loading, error }); };
  async function load(more = false) {
    if (!active || loading || (more && data?.next_offset == null)) return;
    loading = true; emit();
    try {
      // Re-read the loaded range when extending it too: newly inserted sessions must not
      // cause offset pagination to duplicate an item or skip an older one.
      const wanted = Math.max(50, data?.items.length ?? 0) + (more ? 50 : 0);
      const items = new Map<string, T>();
      let offset = 0, total = 0, next: number | null = 0;
      while (active && next !== null && items.size < wanted) {
        const page = await read(offset, Math.min(100, wanted - items.size));
        if (!active) return;
        for (const item of page.items) items.set(item.id, item);
        total = page.total;
        next = page.next_offset;
        if (next !== null && next <= offset) throw new Error("Session pagination did not advance. Refresh to try again.");
        offset = next ?? offset;
      }
      if (active) data = { items: [...items.values()], total, next_offset: next };
      loading = false; emit();
    } catch (error) {
      loading = false; emit(error instanceof Error ? error.message : "Could not load sessions. Try again.");
    }
  }
  emit();
  return { refresh: () => load(), more: () => load(true), dispose: () => { active = false; } };
}

/** Do not poll a hidden tab or spend semantic-search calls on an idle search. */
export function watchSessionChanges(refresh: () => void, searching: boolean) {
  const update = () => { if (document.visibilityState === "visible") refresh(); };
  const timer = searching ? undefined : window.setInterval(update, 30_000);
  document.addEventListener("visibilitychange", update);
  // Host-reported session changes (agent link) arrive on the change feed.
  let pending: ReturnType<typeof setTimeout> | undefined;
  const changed = () => { if (!searching && pending === undefined) pending = setTimeout(() => { pending = undefined; update(); }, 500); };
  const feed = typeof window.addEventListener === "function" ? window : null;
  feed?.addEventListener("eri-agents-changed", changed);
  return () => {
    if (timer !== undefined) window.clearInterval(timer);
    if (pending !== undefined) clearTimeout(pending);
    feed?.removeEventListener("eri-agents-changed", changed);
    document.removeEventListener("visibilitychange", update);
  };
}
