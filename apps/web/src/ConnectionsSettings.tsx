import { lazy, Suspense, useState, type ReactNode } from "react";
import { Tabs } from "./ux";
import { connectionTabs, type ConnectionTab } from "./connection-tabs";
// Meeting sources sit under Connections → Meetings; the panel hides itself when meeting context is off or the workspace is shared.
const SourcesSettings = lazy(() => import("./context/SourcesSettings").then(m => ({ default: m.SourcesSettings })));

/** Settings → Connections, one tab at a time (agent keys, remotea, Linear, Google, meeting sources). A tab shows
 * only when its panel is given; a tab the URL asks for that is not available here falls back to the first. A tab
 * mounts on its first visit and then stays mounted (hidden), so a key shown once survives a look at another tab. */
export function ConnectionsSettings({ tab, onTab, panels, meetings = false }: {
  tab: ConnectionTab; onTab: (tab: ConnectionTab) => void;
  panels: Partial<Record<Exclude<ConnectionTab, "meetings">, ReactNode>>;
  /** Show the Meetings tab (personal workspace with meeting context on). */
  meetings?: boolean;
}) {
  const items = connectionTabs.filter(item => item.id === "meetings" ? meetings : panels[item.id] != null);
  const current = items.find(item => item.id === tab)?.id ?? items[0]?.id;
  const [visited, setVisited] = useState<ReadonlySet<ConnectionTab>>(() => new Set(current ? [current] : []));
  if (current && !visited.has(current)) setVisited(new Set([...visited, current]));
  if (!current) return null;
  return <div className="connections-settings">
    <Tabs id="connection-tab" label="Connections" className="tabs connection-tabs" panel="connection-panel"
      value={current} onChange={onTab} items={items}/>
    <div id="connection-panel" className="connection-panel" role="tabpanel" aria-labelledby={"connection-tab-" + current}>
      {items.filter(item => item.id === current || visited.has(item.id)).map(item =>
        <div key={item.id} className="connection-tab-body" data-connection={item.id} hidden={item.id !== current}>
          <Suspense fallback={<p className="subtle" role="status">Loading…</p>}>
            {item.id === "meetings" ? <SourcesSettings/> : panels[item.id]}
          </Suspense>
        </div>)}
    </div>
  </div>;
}
