import { lazy, Suspense, useEffect, useId, useLayoutEffect, useRef, useState, type CSSProperties } from "react";
import { ChevronLeft, ChevronRight, X } from "lucide-react";
import { useDialogFocus } from "./ux";
import { railDot } from "./remotea-model";
import { useRemotea, useRemoteaActions } from "./remotea-context";
import "./runs.css";

// The drawer body stays in the remotea chunk; the panel shell is eager so the column exists on every page.
const RunDrawer = lazy(() => import("./Remotea").then(m => ({ default: m.RunDrawer })));

/** On phones the open panel is a sheet: trap focus, Escape collapses it. */
function SheetFocus({ onEscape }: { onEscape: () => void }) {
  useDialogFocus({ onEscape });
  return null;
}

/** The persistent right-side run panel: a layout column beside the page (collapsible to a rail) on wide screens,
 * a full-height sheet that collapses to an edge pill on phones. Stays mounted across pages. */
export function RunPanel({ phone }: { phone: boolean }) {
  const { panel, activeRuns } = useRemotea();
  const { closePanel, setCollapsed } = useRemoteaActions();
  const bodyId = useId();
  const root = useRef<HTMLElement>(null), rail = useRef<HTMLButtonElement>(null);
  const opener = useRef<HTMLElement | null>(null);
  const previousRun = useRef<string | null>(null);
  useLayoutEffect(() => {
    if (panel.run && !previousRun.current) opener.current = document.activeElement as HTMLElement | null;
    previousRun.current = panel.run;
  }, [panel.run]);
  useEffect(() => {
    if (!panel.run) return;
    if (panel.collapsed) rail.current?.focus({ preventScroll: true });
    else {
      // The eager shell remains focusable while the drawer chunk is loading.
      (root.current?.querySelector<HTMLElement>(".run-drawer-close") ?? root.current)?.focus({ preventScroll: true });
    }
  }, [panel.run, panel.collapsed]);
  const close = () => {
    closePanel();
    if (opener.current?.isConnected) opener.current.focus({ preventScroll: true });
  };
  useEffect(() => {
    const root = document.querySelector<HTMLElement>(".app-shell");
    if (root) root.dataset.runPanel = panel.run ? (panel.collapsed ? "rail" : "open") : "closed";
    return () => { if (root) root.dataset.runPanel = "closed"; };
  }, [panel.run, panel.collapsed]);
  const [viewport, setViewport] = useState<{ height: number; top: number } | null>(null);
  useEffect(() => {
    const visible = window.visualViewport;
    if (!phone || !visible || !panel.run || panel.collapsed) { setViewport(null); return; }
    const update = () => setViewport({ height: visible.height, top: visible.offsetTop });
    update();
    visible.addEventListener("resize", update);
    visible.addEventListener("scroll", update);
    return () => { visible.removeEventListener("resize", update); visible.removeEventListener("scroll", update); };
  }, [phone, panel.run, panel.collapsed]);
  if (!panel.run) return null;
  const run = activeRuns.find(r => r.id === panel.run);
  const dot = railDot(run);
  const open = !panel.collapsed;
  return <aside ref={root} tabIndex={-1} className="run-panel" data-state={open ? "open" : "rail"} aria-label="Agent run panel"
    style={viewport ? { "--run-viewport-height": `${viewport.height}px`, "--run-viewport-top": `${viewport.top}px` } as CSSProperties : undefined}
    {...(phone && open ? { role: "dialog", "aria-modal": true } : {})}>
    {!open && <button type="button" ref={rail} className={"run-panel-rail tone-" + dot.tone} aria-expanded={false} aria-controls={bodyId}
      aria-label={"Expand run panel, " + dot.label} title={dot.label} onClick={() => setCollapsed(false)}>
      <ChevronLeft size={18} aria-hidden="true"/><i className="run-dot" aria-hidden="true"/><span className="run-panel-rail-label">{dot.label}</span>
    </button>}
    {open && <button type="button" className="run-panel-collapse" aria-label="Collapse run panel" title="Collapse run panel"
      aria-expanded={true} aria-controls={bodyId} onClick={() => setCollapsed(true)}>
      <ChevronRight size={20} aria-hidden="true"/>
    </button>}
    <div id={bodyId} className="run-panel-body" hidden={!open}>
      {phone && open && <SheetFocus onEscape={() => setCollapsed(true)}/>}
      <Suspense fallback={<><button type="button" className="btn-icon run-panel-loading-close" aria-label="Close run" title="Close run" onClick={close}><X size={20} aria-hidden="true"/></button><p role="status" className="run-panel-loading">Loading run…</p></>}>
        {/* Keyed by run: switching runs remounts the drawer, so draft text, busy state, scroll and in-flight
            requests never carry over from the previous run. */}
        <RunDrawer key={panel.run} id={panel.run} onClose={close}/>
      </Suspense>
    </div>
  </aside>;
}
