import type { KeyboardEvent, PointerEvent as ReactPointerEvent, ReactNode } from "react";
import {
  chooseLabels, dueState, itemRadius, itemState, markKind, markName, project, visibleNodes,
  type AtlasIndex, type AtlasLayout, type AtlasLink, type PackedNode, type View,
} from "./atlas-model";
import { reviewLensDim } from "./review-model";

export type Lens = "status" | "due" | "review" | "type";
export type DragState = { id: string; legal: Set<string>; target: string | null; pt: [number, number] | null };

/** The zoomable map. Pure rendering: every interaction is reported to the owner. Colours come from CSS classes
 * so light, dark and lenses switch without reading tokens in script. */
export function AtlasMap({ layout, index, view, width, height, focus, selected, picked, active, lens, litType, links, drag, today,
  onMarkPointerDown, onMarkClick, onBackgroundPointerDown, onBackgroundClick, onFlyOut, onMarkKey, onMarkFocus }: {
  layout: AtlasLayout; index: AtlasIndex; view: View; width: number; height: number; focus: PackedNode; selected: string | null; picked: Set<string>; active: string | null;
  lens: Lens; litType: string | null; links: AtlasLink[]; drag: DragState | null; today: string;
  onMarkPointerDown: (e: ReactPointerEvent, n: PackedNode) => void; onMarkClick: (n: PackedNode, additive: boolean) => void;
  onBackgroundPointerDown: (e: ReactPointerEvent) => void; onBackgroundClick: () => void; onFlyOut: () => void;
  onMarkKey: (e: KeyboardEvent, n: PackedNode) => void; onMarkFocus: (n: PackedNode) => void;
}) {
  const { k, x, y } = project(view, width, height);
  const nodes = visibleNodes(layout, view, width, height);
  const labels = chooseLabels(nodes, focus, view, width, height, selected);
  const tabbable = active && nodes.some(n => n.id === active) ? active : nodes.find(n => n.parent === focus)?.id ?? null;
  const lit = (typeId: string) => !litType || typeId === litType;
  const dueDim = (n: PackedNode) => lens === "due" && !n.region && n.record?.work && dueState(n.record, today) === "none";
  const reviewDim = (n: PackedNode) => lens === "review" && reviewLensDim(n.record?.review_due, index.reviews.get(n.id) ?? 0);
  /** Review-due mark: a small amber dot on the mark's upper right, with a pulse that reduced motion turns off. */
  const pulse = (n: PackedNode, cx: number, cy: number, r: number) => {
    if (!n.record?.review_due) return null;
    const px = cx + r * Math.SQRT1_2, py = cy - r * Math.SQRT1_2;
    return <g className="atlas-review" aria-hidden="true"><circle className="atlas-pulse" cx={px} cy={py} r={5}/><circle className="atlas-review-dot" cx={px} cy={py} r={4.5}/></g>;
  };
  const regionClass = (n: PackedNode) => {
    const parts = ["atlas-mark", "atlas-region"];
    if (litType) parts.push(lit(n.record!.type_id) ? "is-lit" : "is-faint");
    else if (reviewDim(n)) parts.push("is-faint");
    if (n.record?.review_due) parts.push("is-review-due");
    if (drag) parts.push(drag.legal.has(n.id) ? (drag.target === n.id ? "is-target" : "is-legal") : n.id === drag.id ? "is-dragged" : "is-illegal");
    if (n.id === selected || picked.has(n.id)) parts.push("is-selected");
    return parts.join(" ");
  };
  const progressRing = (n: PackedNode, r: number, cx: number, cy: number) => {
    const p = index.progress.get(n.id);
    if (!p?.total || r < 18) return null;
    const ring = r - 3, c = 2 * Math.PI * ring, share = p.done / p.total;
    return <g className="atlas-progress" aria-hidden="true">
      <circle cx={cx} cy={cy} r={ring} className="atlas-progress-track"/>
      {share > 0 && <circle cx={cx} cy={cy} r={ring} className="atlas-progress-done" strokeDasharray={`${c * share} ${c}`} transform={`rotate(-90 ${cx} ${cy})`}/>}
    </g>;
  };
  const markFor = (n: PackedNode, r: number, cx: number, cy: number): ReactNode => {
    const rec = n.record!, kind = markKind(rec, index.types.get(rec.type_id));
    if (kind === "note") return <rect className="atlas-glyph" x={cx - r * 0.82} y={cy - r * 0.82} width={r * 1.64} height={r * 1.64} rx={r * 0.32}/>;
    if (kind === "goal") return <><circle className="atlas-glyph" cx={cx} cy={cy} r={r}/><circle className="atlas-glyph-core" cx={cx} cy={cy} r={r * 0.45}/></>;
    if (kind === "other") return <rect className="atlas-glyph" x={cx - r * 0.7} y={cy - r * 0.7} width={r * 1.4} height={r * 1.4} rx={2} transform={`rotate(45 ${cx} ${cy})`}/>;
    const state = itemState(rec, today);
    return <>
      <circle className="atlas-glyph" cx={cx} cy={cy} r={r}/>
      {state === "in_progress" && r > 4 && <path className="atlas-glyph-half" d={`M${cx},${cy - (r - 2.5)}A${r - 2.5},${r - 2.5} 0 0 0 ${cx},${cy + (r - 2.5)}Z`}/>}
    </>;
  };
  const itemClass = (n: PackedNode) => {
    const rec = n.record!, kind = markKind(rec, index.types.get(rec.type_id));
    const parts = ["atlas-mark", "atlas-item", "kind-" + kind, "lens-" + lens];
    const state = itemState(rec, today);
    if (state) parts.push("state-" + state);
    parts.push("due-" + dueState(rec, today));
    if (rec.review_due) parts.push("is-review-due");
    if ((litType && !lit(rec.type_id)) || dueDim(n) || reviewDim(n)) parts.push("is-dim");
    else if (litType) parts.push("is-lit");
    if (drag) { if (drag.id === rec.id) parts.push("is-dragged"); else if (drag.legal.has(rec.id)) parts.push(drag.target === rec.id ? "is-target" : "is-legal"); }
    if (n.id === selected || picked.has(n.id)) parts.push("is-selected");
    return parts.join(" ");
  };
  const common = (n: PackedNode) => ({
    "data-id": n.id, role: "button", tabIndex: n.id === tabbable ? 0 : -1, "aria-label": markName(n.record!, index, today),
    "aria-current": n.id === selected ? ("true" as const) : undefined,
    onPointerDown: (e: ReactPointerEvent) => onMarkPointerDown(e, n),
    onClick: (e: React.MouseEvent) => { e.stopPropagation(); onMarkClick(n, e.shiftKey || e.metaKey || e.ctrlKey); },
    onKeyDown: (e: KeyboardEvent) => onMarkKey(e, n), onFocus: () => onMarkFocus(n),
  });
  // Link endpoints use the record, or the nearest placed ancestor when it is a moon or off-map.
  const anchor = (id: string): [number, number] | null => {
    let n = layout.byId.get(id) ?? null;
    if (!n) { let p = index.byId.get(id)?.parent_id ?? null; while (p && !n) { n = layout.byId.get(p) ?? null; p = index.byId.get(p)?.parent_id ?? null; } }
    return n ? [x(n.x), y(n.y)] : null;
  };
  const shown = selected ? links.filter(l => l.source_id === selected || l.target_id === selected) : [];
  return <svg className="atlas-canvas" width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="group" aria-label="Map of records">
    <defs>
      <linearGradient id="atlas-orbit" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" style={{ stopColor: "var(--iri-1)" }}/><stop offset=".33" style={{ stopColor: "var(--iri-2)" }}/>
        <stop offset=".66" style={{ stopColor: "var(--iri-3)" }}/><stop offset="1" style={{ stopColor: "var(--iri-4)" }}/>
      </linearGradient>
      <marker id="atlas-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0L10,5L0,10z" className="atlas-arrow"/></marker>
    </defs>
    <rect className="atlas-bg" width={width} height={height} onPointerDown={onBackgroundPointerDown} onClick={onBackgroundClick} onDoubleClick={onFlyOut}/>
    {nodes.map(n => {
      const rec = n.record!, cx = x(n.x), cy = y(n.y);
      if (n.region) {
        const r = n.r * k, empty = !n.children.length;
        return <g key={n.id} className={regionClass(n) + (empty ? " is-empty" : "")} {...common(n)} onDoubleClick={n === focus ? (e => { e.stopPropagation(); onFlyOut(); }) : undefined}>
          <circle className={"atlas-body depth-" + Math.min(4, Math.max(1, n.depth - focus.depth + (focus === layout.root ? 0 : 1)))} cx={cx} cy={cy} r={r}/>
          {progressRing(n, r, cx, cy)}
          <circle className="atlas-focusring" cx={cx} cy={cy} r={r + 3}/>
          {pulse(n, cx, cy, r)}
        </g>;
      }
      const r = itemRadius(n, k), moons = n.moons.slice(0, 12), orbit = r + 8;
      return <g key={n.id} className={itemClass(n)} {...common(n)}>
        <circle className="atlas-hit" cx={cx} cy={cy} r={Math.max(r + (moons.length ? 10 : 4), 9)}/>
        {markFor(n, r, cx, cy)}
        {!!moons.length && r >= 4 && <g className="atlas-moons" aria-hidden="true" style={{ transformOrigin: `${cx}px ${cy}px` }}>
          <circle className="atlas-moon-path" cx={cx} cy={cy} r={orbit}/>
          {moons.map((m, i) => { const a = i / moons.length * 2 * Math.PI - Math.PI / 2; const s = itemState(m, today);
            return <circle key={m.id} className={"atlas-moon state-" + (s ?? "none")} cx={cx + Math.cos(a) * orbit} cy={cy + Math.sin(a) * orbit} r={Math.max(2.4, Math.min(3.6, r / 4))}/>; })}
        </g>}
        {(n.id === selected || picked.has(n.id)) && <circle className="atlas-selring" cx={cx} cy={cy} r={(moons.length ? orbit + 6 : r + 5)}/>}
        {pulse(n, cx, cy, (moons.length ? orbit : r) + 3)}
        <circle className="atlas-focusring" cx={cx} cy={cy} r={(moons.length ? orbit + 6 : r + 5)}/>
      </g>;
    })}
    {focus !== layout.root && <circle className="atlas-halo" cx={x(focus.x)} cy={y(focus.y)} r={focus.r * k + 5} aria-hidden="true"/>}
    {shown.map(l => {
      const a = anchor(l.source_id), b = anchor(l.target_id);
      if (!a || !b || (a[0] === b[0] && a[1] === b[1])) return null;
      const d = Math.hypot(b[0] - a[0], b[1] - a[1]);
      // Bend sideways, and stop short of each mark so the arrow and label stay readable.
      const nx = -(b[1] - a[1]) / d, ny = (b[0] - a[0]) / d, bend = d * 0.28;
      const cx = (a[0] + b[0]) / 2 + nx * bend, cy = (a[1] + b[1]) / 2 + ny * bend;
      const trim = (p: [number, number], q: [number, number], by: number): [number, number] => { const len = Math.hypot(q[0] - p[0], q[1] - p[1]) || 1; return [p[0] + (q[0] - p[0]) / len * by, p[1] + (q[1] - p[1]) / len * by]; };
      const pad = Math.min(24, d / 4), s0 = trim(a, [cx, cy], pad), s1 = trim(b, [cx, cy], pad);
      const lx = (s0[0] + 2 * cx + s1[0]) / 4, ly = (s0[1] + 2 * cy + s1[1]) / 4;
      return <g key={l.id} className={"atlas-link is-" + l.behavior} aria-hidden="true">
        <path d={`M${s0[0]},${s0[1]}Q${cx},${cy} ${s1[0]},${s1[1]}`} markerEnd={l.behavior === "blocks" ? "url(#atlas-arrow)" : undefined}/>
        <text className="atlas-label is-caption is-link" x={lx + nx * 10} y={ly + ny * 10 + 4} textAnchor="middle">{l.label.toLowerCase()}</text>
      </g>;
    })}
    {labels.map(l => {
      const n = layout.byId.get(l.id)!; const p = index.progress.get(l.id); const r = n.r * k;
      const caption = l.kind === "region" && r > 60;
      return <g key={"label-" + l.id} className="atlas-labels" aria-hidden="true">
        <text className={"atlas-label" + (l.kind === "item" ? " is-item" : "")} x={l.x} y={caption ? l.y - 2 : l.y} textAnchor="middle">{l.text}</text>
        {caption && <text className="atlas-label is-caption" x={l.x} y={l.y + 12} textAnchor="middle">
          <tspan>{index.types.get(n.record!.type_id)?.name}</tspan>{!!p?.total && <tspan dx="10">{p.done} of {p.total} done</tspan>}
        </text>}
      </g>;
    })}
    {drag?.pt && <circle className="atlas-ghost" cx={drag.pt[0]} cy={drag.pt[1]} r={9} aria-hidden="true"/>}
  </svg>;
}
