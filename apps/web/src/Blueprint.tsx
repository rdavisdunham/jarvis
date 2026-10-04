import { useRef, useState, type KeyboardEvent } from "react";
import { hierarchy, tree } from "d3-hierarchy";
import type { Schema } from "./structure-types";
import { presentation } from "./field-library";
import { CAPABILITY_GLYPHS, CONTAINER_GLYPH, REVIEW_GLYPH, alsoAllowedIn, diagramParent, nestsItself } from "./blueprint-model";
import { intervalLabel, reviews } from "./review-model";

type Node = { id: string; kids: Node[] };
/** "Also allowed in A, B +3", trimmed to the node width; the full list is in the title and accessible name. */
function alsoLine(names: string[], width: number) {
  const budget = Math.floor(width / 5.6), lead = width < 160 ? "Also in " : "Also allowed in ";
  let shown = 0, text = lead;
  while (shown < names.length) {
    const rest = names.length - shown - 1, next = text + (shown ? ", " : "") + names[shown];
    if (next.length + (rest ? 4 : 0) > budget && shown) break;
    text = next; shown++;
  }
  return shown < names.length ? `${text} +${names.length - shown}` : text;
}
/** The rules as a constellation: each type once along its diagram path, with behaviors, field count,
 * the nests-itself loop and every other allowed home spelled out. */
export function Blueprint({ schema, width, height, selectedType, recordType, hoverType, allHomes, canDesign, counts, onHover, onSelect, onDropType }: {
  schema: Schema; width: number; height: number; selectedType: string | null; recordType: string | null; hoverType: string | null; allHomes: boolean; canDesign: boolean;
  counts: Map<string, number>; onHover: (id: string | null) => void; onSelect: (id: string) => void; onDropType: (type: string, target: string, x: number, y: number) => void;
}) {
  const svg = useRef<SVGSVGElement>(null);
  const overRef = useRef<string | null>(null);
  const [drag, setDrag] = useState<{ type: string; over: string | null } | null>(null);
  const types = schema.types.filter(t => !t.archived);
  const live = new Set(types.map(t => t.id));
  const parents = new Map(presentation(schema).map(p => [p.type_id, p.parent_type_id && live.has(p.parent_type_id) ? p.parent_type_id : null]));
  const build = (id: string, seen: Set<string>): Node => ({ id, kids: types.filter(t => (parents.get(t.id) ?? null) === (id === "" ? null : id) && !seen.has(t.id)).map(t => build(t.id, new Set([...seen, t.id]))) });
  const root = hierarchy<Node>(build("", new Set()), d => d.kids);
  const vertical = width < 620;
  // Fit the columns to the pane where possible; very deep diagrams scroll instead of shrinking further.
  const columns = Math.max(1, root.height), gap = 44;
  const nodeW = vertical ? Math.min(168, (width - 40) / 2.2) : Math.max(132, Math.min(184, (width - 32 - (columns - 1) * gap) / columns)), nodeH = 60;
  if (vertical) tree<Node>().nodeSize([nodeW + 20, nodeH + 50])(root); else tree<Node>().nodeSize([nodeH + 44, nodeW + gap])(root);
  const ds = root.descendants().filter(d => d.data.id);
  const P = new Map<string, [number, number]>();
  let vbH = height;
  if (ds.length) {
    if (vertical) {
      const xs = ds.map(d => d.x!), ys = ds.map(d => d.y!);
      const ox = (width - (Math.max(...xs) - Math.min(...xs) + nodeW)) / 2 - Math.min(...xs), oy = 40 - Math.min(...ys);
      ds.forEach(d => P.set(d.data.id, [d.x! + ox, d.y! + oy]));
      vbH = Math.max(height, Math.max(...ys) + oy + 70);
    } else {
      const xs = ds.map(d => d.y!), ys = ds.map(d => d.x!);
      const spanX = Math.max(...xs) - Math.min(...xs) + nodeW, spanY = Math.max(...ys) - Math.min(...ys);
      const ox = Math.max(16, (width - spanX) / 2) - Math.min(...xs), oy = Math.max(nodeH / 2 + 34, (height - spanY) / 2 + 6) - Math.min(...ys);
      ds.forEach(d => P.set(d.data.id, [d.y! + ox, d.x! + oy]));
      vbH = Math.max(height, Math.max(...ys) + oy + nodeH / 2 + 30);
    }
  }
  const vbW = Math.max(width, ...[...P.values()].map(p => p[0] + nodeW + 16));
  const edge = (a: [number, number], b: [number, number], bend: number) => vertical
    ? `M${a[0] + nodeW / 2},${a[1] + nodeH / 2}C${a[0] + nodeW / 2},${a[1] + nodeH / 2 + bend} ${b[0] + nodeW / 2},${b[1] - nodeH / 2 - bend} ${b[0] + nodeW / 2},${b[1] - nodeH / 2}`
    : `M${a[0] + nodeW},${a[1]}C${a[0] + nodeW + bend},${a[1]} ${b[0] - bend},${b[1]} ${b[0]},${b[1]}`;
  const homesOfRecord = recordType ? new Set(schema.types.find(t => t.id === recordType)?.parent_types ?? []) : new Set<string>();
  const start = (e: React.PointerEvent, id: string) => {
    if (!canDesign || e.button !== 0) return;
    const origin = [e.clientX, e.clientY]; let moved = false;
    const move = (ev: PointerEvent) => {
      if (!moved && Math.hypot(ev.clientX - origin[0], ev.clientY - origin[1]) < 6) return;
      moved = true;
      const box = svg.current!.getBoundingClientRect();
      const px = (ev.clientX - box.left) * vbW / box.width, py = (ev.clientY - box.top) * vbH / box.height;
      const over = [...P].find(([o, [x, y]]) => o !== id && px >= x && px <= x + nodeW && Math.abs(py - y) <= nodeH / 2)?.[0] ?? null;
      overRef.current = over;
      setDrag(d => d?.over === over ? d : { type: id, over });
    };
    const up = (ev: PointerEvent) => {
      window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up);
      setDrag(null);
      if (moved && overRef.current) onDropType(id, overRef.current, ev.clientX, ev.clientY);
      overRef.current = null;
    };
    window.addEventListener("pointermove", move); window.addEventListener("pointerup", up);
  };
  const key = (e: KeyboardEvent, id: string) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect(id); } };
  return <svg ref={svg} className="blueprint-canvas" width={vbW} height={vbH} viewBox={`0 0 ${vbW} ${vbH}`} role="group" aria-label="Type blueprint">
    {allHomes && types.flatMap(t => t.parent_types.filter(h => h !== t.id && h !== diagramParent(schema, t.id) && P.has(h) && P.has(t.id)).map(h =>
      <path key={"home-" + t.id + h} className="blueprint-edge is-allowed" d={edge(P.get(h)!, P.get(t.id)!, 64)}/>))}
    {ds.filter(d => d.parent?.data.id).map(d => <path key={"edge-" + d.data.id} className="blueprint-edge" d={edge(P.get(d.parent!.data.id)!, P.get(d.data.id)!, 26)}/>)}
    {ds.map(d => {
      const t = schema.types.find(x => x.id === d.data.id)!, [x, y] = P.get(t.id)!;
      const also = alsoAllowedIn(schema, t.id), glyphs = CAPABILITY_GLYPHS.filter(([c]) => t.capabilities.includes(c));
      const container = (t.opens_as ?? "auto") === "container", fields = t.fields.filter(f => !f.archived).length, reviewed = reviews(t);
      const cls = ["blueprint-node", t.id === selectedType && "is-selected", t.id === recordType && "is-record", homesOfRecord.has(t.id) && t.id !== recordType && "is-home",
        hoverType === t.id && "is-hover", drag?.over === t.id && "is-drop", drag?.type === t.id && "is-dragged"].filter(Boolean).join(" ");
      const label = [`Type ${t.name}`, `${counts.get(t.id) ?? 0} records`, `${fields} fields`, nestsItself(t) ? "nests itself" : "", reviewed ? "reviewed " + intervalLabel(t.review!.every).toLowerCase() : "", also.length ? "also allowed in " + also.map(h => h.name).join(", ") : ""].filter(Boolean).join(", ");
      return <g key={t.id} className={cls} data-type={t.id} role="button" tabIndex={0} aria-label={label} aria-pressed={t.id === selectedType}
        onPointerEnter={() => onHover(t.id)} onPointerLeave={() => onHover(null)} onFocus={() => onHover(t.id)} onBlur={() => onHover(null)}
        onPointerDown={e => start(e, t.id)} onClick={() => onSelect(t.id)} onKeyDown={e => key(e, t.id)}>
        <rect x={x} y={y - nodeH / 2} width={nodeW} height={nodeH} rx={12}/>
        <text className="blueprint-name" x={x + 12} y={y - 6}>{t.name}</text>
        <text className="blueprint-small" x={x + nodeW - 10} y={y - 6} textAnchor="end">{fields} {fields === 1 ? "field" : "fields"}</text>
        {glyphs.map(([c, g, name], i) => <text key={c} className="blueprint-glyph" x={x + 12 + i * 17} y={y + 16}><title>{name}</title>{g}</text>)}
        {container && <text className="blueprint-glyph is-container" x={x + 12 + glyphs.length * 17} y={y + 16}><title>{CONTAINER_GLYPH[1]}</title>{CONTAINER_GLYPH[0]}</text>}
        {reviewed && <text className="blueprint-glyph is-review" x={x + 12 + (glyphs.length + (container ? 1 : 0)) * 17} y={y + 16}><title>{REVIEW_GLYPH[1] + ": " + intervalLabel(t.review!.every).toLowerCase()}</title>{REVIEW_GLYPH[0]}</text>}
        {nestsItself(t) && <><path className="blueprint-loop" d={`M${x + nodeW - 30},${y - nodeH / 2} a8,8 0 1 1 14,0`}/>
          <text className="blueprint-small is-violet" x={x + nodeW - 10} y={y + 16} textAnchor="end">nests itself</text></>}
        {!!also.length && <text className="blueprint-small" x={x + 4} y={y + nodeH / 2 + 16}><title>{"Also allowed in " + also.map(h => h.name).join(", ")}</title>{alsoLine(also.map(h => h.name), nodeW)}</text>}
      </g>;
    })}
  </svg>;
}
