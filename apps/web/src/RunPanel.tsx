import { lazy, Suspense, useEffect } from "react";
import { ChevronLeft } from "lucide-react";
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
  useEffect(() => {
    const root = document.querySelector<HTMLElement>(".app-shell");
    if (root) root.dataset.runPanel = panel.run ? (panel.collapsed ? "rail" : "open") : "closed";
    return () => { if (root) root.dataset.runPanel = "closed"; };
  }, [panel.run, panel.collapsed]);
  if (!panel.run) return null;
  const run = activeRuns.find(r => r.id === panel.run);
  const dot = railDot(run);
  const open = !panel.collapsed;
  return <aside className="run-panel" data-state={open ? "open" : "rail"} aria-label="Agent run panel"
    {...(phone && open ? { role: "dialog", "aria-modal": true } : {})}>
    {!open && <button type="button" className={"run-panel-rail tone-" + dot.tone} aria-expanded={false}
      aria-label={"Expand run panel, " + dot.label} title={dot.label} onClick={() => setCollapsed(false)}>
      <ChevronLeft size={18} aria-hidden="true"/><i className="run-dot" aria-hidden="true"/><span className="run-panel-rail-label">{dot.label}</span>
    </button>}
    {(open || !phone) && <div className="run-panel-body" hidden={!open}>
      {phone && open && <SheetFocus onEscape={() => setCollapsed(true)}/>}
      <Suspense fallback={<p role="status" className="run-panel-loading">Loading run…</p>}>
        {/* Keyed by run: switching runs remounts the drawer, so draft text, busy state, scroll and in-flight
            requests never carry over from the previous run. */}
        <RunDrawer key={panel.run} id={panel.run} onClose={closePanel} onCollapse={() => setCollapsed(true)}/>
      </Suspense>
    </div>}
  </aside>;
}
