/** Pure Atlas logic: indexing, legal homes, move choices, circle packing, culling and labels.
 * Kept free of React and the DOM so it can be unit tested and timed. */
import { hierarchy, pack } from "d3-hierarchy";

export type AtlasRecord = {
  id: string; title: string; type_id: string; parent_id: string | null; depth: number; revision: number; sort_order: number;
  opens_as: "container" | "item"; status_meaning: string | null; work: boolean; due_date: string | null; planned_date: string | null;
  review_due?: boolean; next_review_at?: string | null;
};
export type AtlasLink = { id: string; source_id: string; target_id: string; relationship_id: string; label: string; behavior: "related" | "blocks" };
export type AtlasType = { id: string; name: string; plural: string; capabilities: string[]; parent_types: string[]; opens_as?: string; review?: { enabled: boolean; every: string }; archived?: boolean };
export type AtlasData = { schema_revision: number; types: AtlasType[]; records: AtlasRecord[]; links: AtlasLink[]; total: number; truncated: boolean; limit: number };
export type AtlasIndex = { links: AtlasLink[]; byId: Map<string, AtlasRecord>; children: Map<string | null, AtlasRecord[]>; types: Map<string, AtlasType>; progress: Map<string, { done: number; total: number }>;
  /** Due reviews among each record's descendants (never counting the record itself). */
  reviews: Map<string, number> };

export const ROOT = "root";
/** A synthetic region for records without a home; it is never a move target. */
export const UNFILED = "unfiled";
export const unfiledRecord = (): AtlasRecord => ({ id: UNFILED, title: "Unfiled", type_id: "", parent_id: null, depth: 0, revision: 0, sort_order: 1e9,
  opens_as: "container", status_meaning: null, work: false, due_date: null, planned_date: null });
export const isRegion = (r: Pick<AtlasRecord, "opens_as"> | null | undefined) => r?.opens_as === "container";

export function indexAtlas(data: Pick<AtlasData, "records" | "types"> & { links?: AtlasLink[] }): AtlasIndex {
  const byId = new Map(data.records.map(r => [r.id, r]));
  const children = new Map<string | null, AtlasRecord[]>();
  for (const r of data.records) {
    const parent = r.parent_id && byId.has(r.parent_id) ? r.parent_id : null;
    const list = children.get(parent);
    if (list) list.push(r); else children.set(parent, [r]);
  }
  for (const list of children.values()) list.sort((a, b) => a.sort_order - b.sort_order || a.title.localeCompare(b.title));
  const index: AtlasIndex = { links: data.links ?? [], byId, children, types: new Map(data.types.map(t => [t.id, t])), progress: new Map(), reviews: new Map() };
  // Post-order totals: work done/total and due reviews over each record's descendants (never its own state).
  const visit = (id: string, seen: Set<string>): { done: number; total: number; due: number } => {
    let done = 0, total = 0, due = 0;
    for (const child of children.get(id) ?? []) {
      if (seen.has(child.id)) continue;
      seen.add(child.id);
      if (child.work) { total++; if (child.status_meaning === "completed") done++; }
      if (child.review_due) due++;
      const sub = visit(child.id, seen);
      done += sub.done; total += sub.total; due += sub.due;
    }
    index.progress.set(id, { done, total });
    if (due) index.reviews.set(id, due);
    return { done, total, due };
  };
  const seen = new Set<string>();
  for (const r of children.get(null) ?? []) { seen.add(r.id); visit(r.id, seen); }
  return index;
}

export const homeOf = (index: AtlasIndex, r: AtlasRecord) => (r.parent_id ? index.byId.get(r.parent_id) ?? null : null);
/** Root-first ancestors, excluding the record itself. */
export function ancestors(index: AtlasIndex, id: string): AtlasRecord[] {
  const out: AtlasRecord[] = [];
  let current = index.byId.get(id);
  const seen = new Set<string>();
  while (current?.parent_id && !seen.has(current.parent_id)) {
    seen.add(current.parent_id);
    const parent = index.byId.get(current.parent_id);
    if (!parent) break;
    out.unshift(parent);
    current = parent;
  }
  return out;
}
export function subtreeIds(index: AtlasIndex, id: string): Set<string> {
  const out = new Set<string>([id]);
  const todo = [id];
  while (todo.length) for (const child of index.children.get(todo.pop()!) ?? []) if (!out.has(child.id)) { out.add(child.id); todo.push(child.id); }
  return out;
}

/** Records this record may move into: an allowed home type, not itself, not its own subtree, not its current home. */
export function legalTargets(record: AtlasRecord, index: AtlasIndex): Set<string> {
  const allowed = new Set(index.types.get(record.type_id)?.parent_types ?? []);
  const own = subtreeIds(index, record.id);
  const out = new Set<string>();
  for (const r of index.byId.values()) if (allowed.has(r.type_id) && !own.has(r.id) && r.id !== record.parent_id) out.add(r.id);
  return out;
}

export type MoveOption = { mode: "subtree" | "item"; label: string; note: string };
const count = (n: number, one: string, many = one + "s") => `${n} ${n === 1 ? one : many}`;
/** The two contents choices offered after a drop or a keyboard "Move to…". */
export function moveChoices(record: AtlasRecord, index: AtlasIndex): MoveOption[] {
  const kids = index.children.get(record.id) ?? [];
  const inside = subtreeIds(index, record.id).size - 1;
  const old = homeOf(index, record);
  const options: MoveOption[] = [{ mode: "subtree", label: "Move with contents", note: inside ? `${count(inside, "item")} inside move along` : "Nothing inside to move" }];
  if (kids.length) options.push({ mode: "item", label: "Move only this record", note: `Its ${count(kids.length, "item")} stay in ${old ? old.title : "Unfiled"}` });
  return options;
}

/** No choice to make: a record with nothing inside, or a task, meaning a work record that opens as an item
 *  (its subtasks always travel with it). Containers such as projects still ask. */
export function movesWhole(record: AtlasRecord, index: AtlasIndex): boolean {
  return !(index.children.get(record.id)?.length) || (record.work && record.opens_as === "item");
}

export type MarkKind = "region" | "task" | "note" | "goal" | "other";
export function markKind(r: AtlasRecord, type?: AtlasType): MarkKind {
  if (isRegion(r)) return "region";
  const caps = type?.capabilities ?? [];
  if (caps.includes("metric")) return "goal";
  if (r.work || caps.includes("work")) return "task";
  if (caps.includes("content")) return "note";
  return "other";
}
export type ItemState = "done" | "overdue" | "in_progress" | "open" | "waiting" | "cancelled";
export function itemState(r: AtlasRecord, today: string): ItemState | null {
  const m = r.status_meaning;
  if (!m) return null;
  if (m === "completed") return "done";
  if (m === "cancelled") return "cancelled";
  if (r.due_date && r.due_date < today) return "overdue";
  if (m === "in_progress") return "in_progress";
  if (m === "open") return "open";
  return "waiting";
}
export function dueState(r: AtlasRecord, today: string): "overdue" | "today" | "upcoming" | "none" {
  if (!r.due_date || ["completed", "cancelled"].includes(r.status_meaning ?? "")) return "none";
  return r.due_date < today ? "overdue" : r.due_date === today ? "today" : "upcoming";
}
export const stateLabel: Record<ItemState, string> = { done: "done", overdue: "overdue", in_progress: "in progress", open: "open", waiting: "waiting", cancelled: "cancelled" };

// ---- Layout --------------------------------------------------------------------------------
export type PackedNode = { id: string; record: AtlasRecord | null; x: number; y: number; r: number; depth: number; parent: PackedNode | null; children: PackedNode[]; region: boolean; moons: AtlasRecord[] };
export type AtlasLayout = { root: PackedNode; nodes: PackedNode[]; byId: Map<string, PackedNode> };
type Datum = { record: AtlasRecord | null; kids?: Datum[]; moons?: AtlasRecord[] };
export const LAYOUT_SIZE = 1000;

/** Regions (containers) nest; items are leaves whose own contents orbit as moons. */
export function layoutAtlas(index: AtlasIndex, size = LAYOUT_SIZE): AtlasLayout {
  const seen = new Set<string>();
  const build = (r: AtlasRecord): Datum => {
    seen.add(r.id);
    const kids = (index.children.get(r.id) ?? []).filter(k => !seen.has(k.id));
    if (isRegion(r)) return { record: r, kids: kids.map(build) };
    return { record: r, moons: kids };
  };
  const roots = index.children.get(null) ?? [];
  // Unfiled items gather in one region so a large inbox doesn't shrink every top-level home.
  const loose = roots.filter(r => !isRegion(r));
  const top = roots.filter(r => isRegion(r)).map(build);
  if (loose.length) top.push({ record: unfiledRecord(), kids: loose.map(build) });
  const tree = hierarchy<Datum>({ record: null, kids: top }, d => d.kids)
    .sum(d => d.kids ? (d.kids.length ? 0 : 1.6) : 1 + Math.min(12, d.moons?.length ?? 0) * 0.3)
    .sort((a, b) => (b.value ?? 0) - (a.value ?? 0));
  const packed = pack<Datum>().size([size, size]).padding(d => d.depth === 0 ? 14 : Math.max(1.2, 9 / d.depth))(tree);
  const nodes: PackedNode[] = [];
  const byId = new Map<string, PackedNode>();
  const convert = (d: typeof packed, parent: PackedNode | null): PackedNode => {
    const record = d.data.record;
    const node: PackedNode = { id: record?.id ?? ROOT, record, x: d.x, y: d.y, r: d.r, depth: d.depth, parent, children: [], region: !record || isRegion(record), moons: d.data.moons ?? [] };
    nodes.push(node); byId.set(node.id, node);
    node.children = (d.children ?? []).map(c => convert(c, node));
    return node;
  };
  const root = convert(packed, null);
  return { root, nodes, byId };
}

/** The placed node that stands for a record: itself, or the nearest placed ancestor (moons and deeper). */
export function placedNode(layout: AtlasLayout, index: AtlasIndex, id: string | null): PackedNode {
  if (!id) return layout.root;
  const own = layout.byId.get(id);
  if (own) return own;
  for (const a of ancestors(index, id).reverse()) { const n = layout.byId.get(a.id); if (n) return n; }
  return layout.root;
}
/** Where to fly for a record: a region opens itself; anything else opens its nearest region and is selected. */
export function flyTarget(layout: AtlasLayout, index: AtlasIndex, id: string): { focus: string; select: string } {
  let node: PackedNode | null = placedNode(layout, index, id);
  if (node.id === id && node.region) return { focus: id, select: id };
  while (node && !node.region) node = node.parent;
  return { focus: node?.id ?? ROOT, select: id };
}

// ---- Viewport ------------------------------------------------------------------------------
export type View = [number, number, number];
export const viewFor = (n: Pick<PackedNode, "x" | "y" | "r">): View => [n.x, n.y, n.r * 2.15];
/** Vertical room kept for the overlaid path row (top) and legend (bottom), so a fitted region clears both. */
export const FRAME_Y = 40;
export const scaleOf = (view: View, w: number, h: number) => Math.min(w, Math.max(h - 2 * FRAME_Y, h * 0.7)) / view[2];
export function project(view: View, w: number, h: number) {
  const k = scaleOf(view, w, h);
  return { k, x: (x: number) => (x - view[0]) * k + w / 2, y: (y: number) => (y - view[1]) * k + h / 2 };
}
/** Smallest screen radius at which a region shows its contents, and the item mark radius. */
export const OPEN_RADIUS = 22;
export const itemRadius = (n: PackedNode, k: number) => Math.max(3, Math.min(n.r * k * (n.moons.length ? 0.5 : 0.62), 20));

/** Semantic zoom: nodes that intersect the viewport and are large enough to read, parents first. */
export function visibleNodes(layout: AtlasLayout, view: View, w: number, h: number): PackedNode[] {
  const { k, x, y } = project(view, w, h);
  const out: PackedNode[] = [];
  const walk = (n: PackedNode) => {
    const r = n.r * k, cx = x(n.x), cy = y(n.y);
    if (cx + r < -40 || cx - r > w + 40 || cy + r < -40 || cy - r > h + 40) return;
    if (n !== layout.root) {
      if (n.region ? r < 1.5 : r < 1.2) return;
      out.push(n);
    }
    if (n.region && (n === layout.root || r >= OPEN_RADIUS)) n.children.forEach(walk);
  };
  walk(layout.root);
  return out;
}

export const trimLabel = (s: string, n: number) => { const max = Math.max(6, Math.floor(n)); return s.length > max ? s.slice(0, max - 1) + "…" : s; };
type Box = { x0: number; y0: number; x1: number; y1: number };
export type Label = { id: string; text: string; x: number; y: number; kind: "region" | "item" };
/** Label budget: the focused level's children first (largest first), the selection always, no overlaps. */
export function chooseLabels(nodes: PackedNode[], focus: PackedNode, view: View, w: number, h: number, selected: string | null, budget = 40): Label[] {
  const { k, x, y } = project(view, w, h);
  const scored: { n: PackedNode; score: number }[] = [];
  for (const n of nodes) {
    if (n === focus || !n.record) continue;
    const r = n.r * k;
    const near = n.parent === focus;
    if (n.id === selected) scored.push({ n, score: 1e9 });
    else if (n.region && (near || (r > 55 && n.depth <= focus.depth + 2))) scored.push({ n, score: r * (near ? 2 : 1) });
    else if (!n.region && near && r * 2 / 6.4 >= 6) scored.push({ n, score: r });
  }
  scored.sort((a, b) => b.score - a.score);
  const boxes: Box[] = [], labels: Label[] = [];
  for (const { n } of scored) {
    if (labels.length >= budget) break;
    const r = n.r * k, cx = x(n.x), cy = y(n.y);
    const region = n.region;
    const room = region ? Math.max(12, r / 3.4) : Math.max(8, r * 2 / 6.2);
    const text = n.id === selected ? trimLabel(n.record!.title, 48) : trimLabel(n.record!.title, room);
    const width = text.length * (region ? 7.2 : 6.4) + 8;
    const ly = region ? cy - r - (r > 60 ? 18 : 6) : cy + itemRadius(n, k) + (n.moons.length ? 22 : 14);
    const box = { x0: cx - width / 2, x1: cx + width / 2, y0: ly - 13, y1: ly + 4 };
    if (n.id !== selected && boxes.some(b => b.x0 < box.x1 && box.x0 < b.x1 && b.y0 < box.y1 && box.y0 < b.y1)) continue;
    boxes.push(box);
    labels.push({ id: n.id, text, x: cx, y: ly, kind: region ? "region" : "item" });
  }
  return labels;
}

/** Arrow-key order among visible siblings: reading order (rows, then left to right). */
export function siblingOrder(nodes: PackedNode[]): PackedNode[] {
  return [...nodes].sort((a, b) => Math.round(a.y / 24) - Math.round(b.y / 24) || a.x - b.x);
}

export function searchRecords(index: AtlasIndex, query: string, limit = 8): AtlasRecord[] {
  const q = query.trim().toLowerCase();
  if (!q) return [];
  const hits: [number, AtlasRecord][] = [];
  for (const r of index.byId.values()) {
    const t = r.title.toLowerCase(), at = t.indexOf(q);
    if (at >= 0) hits.push([at === 0 ? 0 : t.includes(" " + q) ? 1 : 2, r]);
  }
  return hits.sort((a, b) => a[0] - b[0] || a[1].title.localeCompare(b[1].title)).slice(0, limit).map(h => h[1]);
}

/** Accessible name for a mark: type, title and its state, as separate phrases (no meta strings). */
export function markName(r: AtlasRecord, index: AtlasIndex, today: string): string {
  if (r.id === UNFILED) return `Unfiled: ${count(index.children.get(null)?.filter(x => !isRegion(x)).length ?? 0, "record")} without a home`;
  const type = index.types.get(r.type_id);
  const parts = [`${type?.name ?? "Record"}: ${r.title}`];
  const state = itemState(r, today);
  if (state) parts.push(stateLabel[state]);
  if (r.due_date && state !== "done") parts.push("due " + r.due_date);
  if (r.review_due) parts.push("review due");
  const p = index.progress.get(r.id);
  if (isRegion(r) && p?.total) parts.push(`${p.done} of ${p.total} done`);
  const reviewsInside = index.reviews.get(r.id) ?? 0;
  if (isRegion(r) && reviewsInside) parts.push(count(reviewsInside, "review") + " due inside");
  const inside = index.children.get(r.id)?.length ?? 0;
  if (!isRegion(r) && inside) parts.push(count(inside, "item") + " inside");
  return parts.join(", ");
}
