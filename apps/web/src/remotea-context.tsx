import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { api, ApiError, post } from "./api";
import { useAgentRefresh } from "./remotea-refresh";
import {
  needsTargetData, PANEL_STORAGE_KEY, restorePanel, runsByTask, storedPanel,
  type NeedsTarget, type PanelState, type PendingControl, type RemoteaStatus, type Run,
} from "./remotea-model";

/** Per proposal card: the last refusal and the target picker's data (null: the picker was cancelled). */
export type PendingCardState = { busy: boolean; error: string; need: NeedsTarget | null | undefined };

export type RemoteaData = {
  enabled: boolean;
  status: RemoteaStatus | null;
  /** Runs that have not settled, newest first (GET /remotea/runs?active=true). */
  activeRuns: Run[];
  runsByTask: ReadonlyMap<string, Run>;
  pending: PendingControl[];
  cards: Record<string, PendingCardState>;
  error: string;
  panel: PanelState;
  /** The task whose Run dialog is open. */
  start: string | null;
};
export type RemoteaActions = {
  refresh: () => void;
  /** App reports boot.preferences.remotea_enabled; nothing is fetched while off. */
  setEnabled: (enabled: boolean) => void;
  openRun: (id: string) => void;
  closePanel: () => void;
  setCollapsed: (collapsed: boolean) => void;
  openStart: (taskId: string) => void;
  closeStart: () => void;
  /** Confirm or decline one of Eri's proposals; a confirmed start returns its run. */
  decide: (id: string, decision: "confirm" | "decline", target?: string) => Promise<Run | null>;
  cancelPicker: (id: string) => void;
};

const OFF: RemoteaData = { enabled: false, status: null, activeRuns: [], runsByTask: new Map(), pending: [], cards: {}, error: "",
  panel: { run: null, collapsed: false }, start: null };
const noop = () => {};
const NO_ACTIONS: RemoteaActions = { refresh: noop, setEnabled: noop, openRun: noop, closePanel: noop, setCollapsed: noop, openStart: noop, closeStart: noop,
  decide: async () => null, cancelPicker: noop };

export const RemoteaDataContext = createContext<RemoteaData>(OFF);
export const RemoteaActionsContext = createContext<RemoteaActions>(NO_ACTIONS);
export const useRemotea = () => useContext(RemoteaDataContext);
export const useRemoteaActions = () => useContext(RemoteaActionsContext);

function readStored() {
  try { return sessionStorage.getItem(PANEL_STORAGE_KEY); } catch { return null; }
}
function writeStored(value: string | null) {
  try { if (value) sessionStorage.setItem(PANEL_STORAGE_KEY, value); else sessionStorage.removeItem(PANEL_STORAGE_KEY); } catch { /* private mode */ }
}
const message = (e: unknown) => (e as Error).message || "Something went wrong.";

/** One place that knows remotea's state for the whole app: the server, active runs, Eri's open proposals and which
 * run the side panel shows. Rows, the Runs button, the chat drawer and the remotea page read it; nothing else polls. */
export function RemoteaProvider({ children }: { children: ReactNode }) {
  const [enabled, setEnabled] = useState(false);
  const [status, setStatus] = useState<RemoteaStatus | null>(null);
  const [activeRuns, setActiveRuns] = useState<Run[]>([]);
  const [pending, setPending] = useState<PendingControl[]>([]);
  const [cards, setCards] = useState<Record<string, PendingCardState>>({});
  const [error, setError] = useState("");
  const [panel, setPanel] = useState<PanelState>(() => restorePanel(location.search, readStored()));
  const [start, setStart] = useState<string | null>(null);

  const refresh = useCallback(() => {
    if (!enabled) return;
    void Promise.allSettled([
      api<RemoteaStatus>("/remotea/status"),
      api<{ items: Run[] }>("/remotea/runs?active=true&limit=100"),
      api<{ items: PendingControl[] }>("/remotea/pending"),
    ]).then(([s, r, p]) => {
      if (s.status === "fulfilled") setStatus(s.value);
      if (r.status === "fulfilled") setActiveRuns(r.value.items);
      if (p.status === "fulfilled") {
        setPending(p.value.items);
        const open = new Set(p.value.items.map(item => item.id));
        setCards(current => Object.fromEntries(Object.entries(current).filter(([id]) => open.has(id))));
      }
      const failed = [s, r, p].find((x): x is PromiseRejectedResult => x.status === "rejected");
      setError(failed ? message(failed.reason) : "");
    });
  }, [enabled]);
  useEffect(() => { if (!enabled) { setStatus(null); setActiveRuns([]); setPending([]); } refresh(); }, [enabled, refresh]);
  useAgentRefresh(refresh);

  const byTask = useMemo(() => runsByTask(activeRuns), [activeRuns]);
  const actions = useMemo<RemoteaActions>(() => ({
    refresh,
    setEnabled,
    openRun: id => { setPanel({ run: id, collapsed: false }); writeStored(null); },
    closePanel: () => setPanel(p => ({ ...p, run: null })),
    setCollapsed: collapsed => setPanel(p => { const next = { ...p, collapsed }; writeStored(storedPanel(next)); return next; }),
    openStart: taskId => setStart(taskId),
    closeStart: () => setStart(null),
    cancelPicker: id => setCards(c => ({ ...c, [id]: { busy: false, error: "", need: null } })),
    decide: async (id, decision, target) => {
      setCards(c => ({ ...c, [id]: { busy: true, error: "", need: c[id]?.need } }));
      try {
        const result = await post<{ run?: Run | null }>(`/remotea/pending/${id}/${decision}`, target ? { execution_target_id: target } : {});
        setCards(c => { const { [id]: _, ...rest } = c; return rest; });
        return result?.run ?? null;
      } catch (e) {
        const need = e instanceof ApiError && e.code === "NEEDS_TARGET" ? needsTargetData(e.data) : null;
        setCards(c => ({ ...c, [id]: { busy: false, error: need ? "" : message(e), need: need ?? c[id]?.need } }));
        return null;
      } finally { refresh(); }
    },
  }), [refresh]);
  const data = useMemo<RemoteaData>(() => ({ enabled, status, activeRuns, runsByTask: byTask, pending, cards, error, panel, start }),
    [enabled, status, activeRuns, byTask, pending, cards, error, panel, start]);
  return <RemoteaActionsContext.Provider value={actions}><RemoteaDataContext.Provider value={data}>{children}</RemoteaDataContext.Provider></RemoteaActionsContext.Provider>;
}
