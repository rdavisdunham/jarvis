import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent as ReactPointerEvent } from "react";
import { ChevronRight, Search } from "lucide-react";
import { interpolateZoom } from "d3-interpolate";
import { api, command, post } from "./api";
import { AtlasMap, type DragState, type Lens } from "./AtlasMap";
import { Blueprint } from "./Blueprint";
import { AddHere, MoveChoiceList, RecordInspector, TypeInspector, type TemplateHooks } from "./AtlasInspector";
import { TypeMoveChoices } from "./TypeMap";
import {
  ROOT, ancestors, flyTarget, indexAtlas, isRegion, itemRadius, layoutAtlas, legalTargets, moveChoices, project, scaleOf, searchRecords,
  siblingOrder, viewFor, visibleNodes, type AtlasData, type AtlasIndex, type AtlasRecord, type MoveOption, type PackedNode, type View,
} from "./atlas-model";
import { allowAndPlace, draftChanges, typeMoveChoice } from "./blueprint-model";
import { placeType } from "./field-library";
import type { Schema } from "./structure-types";
import { instantiate, useTemplates } from "./Templates";
import type { InstantiatePreview } from "./template-model";
import "./atlas.css";

type Mode = "atlas" | "both" | "blueprint";
type Popup =
  | { kind: "move"; x: number; y: number; record: AtlasRecord; target: string | null; targetTitle: string }
  | { kind: "type"; x: number; y: number; type: string; target: string };
type Toast = { message: string; undo?: () => Promise<void> } | null;
const MODES: [Mode, string][] = [["atlas", "Map"], ["both", "Both"], ["blueprint", "Blueprint"]];
const LENSES: [Lens, string][] = [["status", "Status"], ["due", "Due"], ["review", "Reviews due"], ["type", "Type"]];
const reducedMotion = () => typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
const stored = (key: string, fallback: string) => { try { return localStorage.getItem(key) ?? fallback; } catch { return fallback; } };
const store = (key: string, value: string) => { try { localStorage.setItem(key, value); } catch { /* private mode */ } };

function useSize<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => setSize(s => (s.width === el.clientWidth && s.height === el.clientHeight ? s : { width: el.clientWidth, height: el.clientHeight }));
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  });
  return [ref, size] as const;
}

/** Atlas: your actual records as a zoomable star map, with the Blueprint of types beside it.
 * Moves use record.contents preview/apply, creates use record.create, and type edits stay a draft
 * until they are reviewed in Types & fields (structure preview → apply). */
export default function AtlasView({ schema, canEdit, canDesign, focus, target, refresh, today, onFocus, onOpen, onBrowse, onChanged, onEditTypes, onVisible }: {
  schema: Schema; canEdit: boolean; canDesign: boolean; focus: string; target?: { nonce: string; record_id?: string } | null; refresh: number; today: string;
  onFocus: (id: string) => void; onOpen: (id: string) => void; onBrowse: (id: string) => void; onChanged: () => Promise<void>;
  onEditTypes: (draft: Schema | null, type?: string) => void; onVisible?: (ids: string[]) => void;
}) {
  const [data, setData] = useState<AtlasData | null>(null), [error, setError] = useState("");
  const [mode, setModeState] = useState<Mode>(() => (["atlas", "both", "blueprint"].includes(stored("eridani-atlas-mode", "atlas")) ? stored("eridani-atlas-mode", "atlas") as Mode : "atlas"));
  const [lens, setLens] = useState<Lens>("status");
  const [focusId, setFocusId] = useState(focus || ROOT);
  const [selected, setSelected] = useState<string | null>(null), [selectedType, setSelectedType] = useState<string | null>(null);
  const [hoverType, setHoverType] = useState<string | null>(null), [allHomes, setAllHomes] = useState(false);
  const [active, setActive] = useState<string | null>(null), [view, setView] = useState<View>([500, 500, 1075]);
  const [drag, setDrag] = useState<DragState | null>(null), [popup, setPopup] = useState<Popup | null>(null);
  const [toast, setToast] = useState<Toast>(null), [busy, setBusy] = useState(false);
  const [query, setQuery] = useState(""), [draft, setDraft] = useState<Schema | null>(null);
  const { templates } = useTemplates(refresh);
  const [mapRef, mapSize] = useSize<HTMLDivElement>(), [blueRef, blueSize] = useSize<HTMLDivElement>();
  const searchRef = useRef<HTMLInputElement>(null), animation = useRef(0), suppressClick = useRef(false), keyboard = useRef(false);
  const viewRef = useRef(view); viewRef.current = view;

  const load = useCallback(async () => { const next = await api<AtlasData>("/structure/atlas"); setData(next); setError(""); return next; }, []);
  useEffect(() => { void load().catch(e => setError(e.message)); }, [load, refresh]);
  useEffect(() => setDraft(null), [schema.revision]);
  const index = useMemo(() => (data ? indexAtlas(data) : null), [data]);
  const layout = useMemo(() => (index ? layoutAtlas(index) : null), [index]);
  const focusNode = layout?.byId.get(focusId) ?? layout?.root ?? null;

  const setMode = (m: Mode) => { setModeState(m); store("eridani-atlas-mode", m); };
  const flyTo = useCallback((id: string, { animate = true, report = true }: { animate?: boolean; report?: boolean } = {}) => {
    if (!layout) return;
    const node = layout.byId.get(id) ?? layout.root;
    const region = node.region ? node : (node.parent ?? layout.root);
    setFocusId(region.id);
    if (report) onFocus(region.id === ROOT ? "" : region.id);
    const to = viewFor(region), from = viewRef.current;
    cancelAnimationFrame(animation.current);
    if (!animate || reducedMotion()) { setView(to); return; }
    const zoom = interpolateZoom(from, to), duration = Math.min(900, Math.max(250, zoom.duration * 0.6)), started = performance.now();
    const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
    const step = (now: number) => { const t = Math.min(1, (now - started) / duration); setView(zoom(ease(t)) as View); if (t < 1) animation.current = requestAnimationFrame(step); };
    animation.current = requestAnimationFrame(step);
  }, [layout, onFocus]);
  useEffect(() => () => cancelAnimationFrame(animation.current), []);
  // Keep the camera on the focused region whenever the layout changes (load, move, create).
  useEffect(() => { if (layout) { const node = layout.byId.get(focusId) ?? layout.root; setView(viewFor(node)); if (node === layout.root && focusId !== ROOT) setFocusId(ROOT); } }, [layout]); // eslint-disable-line react-hooks/exhaustive-deps
  // Breadcrumbs, history and Eri set the focus from outside.
  useEffect(() => { if (layout && (focus || ROOT) !== focusId) flyTo(focus || ROOT, { report: false }); }, [focus, layout]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!target?.record_id || !layout || !index) return;
    const goal = flyTarget(layout, index, target.record_id);
    flyTo(goal.focus); setSelected(goal.select); setSelectedType(null); setActive(goal.select);
  }, [target?.nonce, layout]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!focusNode || !onVisible) return;
    onVisible(focusNode.children.map(n => n.id).slice(0, 60));
  }, [focusNode, onVisible]);
  useEffect(() => {
    const key = (e: globalThis.KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k" && searchRef.current) { e.preventDefault(); e.stopPropagation(); searchRef.current.focus(); }
      if (e.key === "Escape") { setPopup(null); setDrag(null); }
    };
    window.addEventListener("keydown", key, true);
    return () => window.removeEventListener("keydown", key, true);
  }, []);
  useEffect(() => { if (!toast) return; const timer = setTimeout(() => setToast(null), 6000); return () => clearTimeout(timer); }, [toast]);
  useEffect(() => {
    if (!keyboard.current || !active) return;
    keyboard.current = false;
    requestAnimationFrame(() => mapRef.current?.querySelector<SVGGElement>(`[data-id="${CSS.escape(active)}"]`)?.focus());
  });

  const select = (id: string | null) => { setSelected(id); setSelectedType(null); if (id) setActive(id); };
  const selectType = (id: string) => { setSelectedType(id); setSelected(null); };
  const visible = layout && mapSize.width ? visibleNodes(layout, view, mapSize.width, mapSize.height) : [];

  // ---- Mutations: existing commands only -------------------------------------------------
  const move = async (record: AtlasRecord, targetId: string | null, mode: MoveOption["mode"]) => {
    if (!data || !index) return;
    setBusy(true); setPopup(null);
    try {
      const args = { record_id: record.id, expected_revision: record.revision, schema_revision: data.schema_revision, operation: "move" as const, mode, parent_id: targetId };
      const preview = await post<{ preview_hash: string; issues: string[] }>("/structure/contents/preview", args);
      if (preview.issues.length) { setToast({ message: preview.issues[0] }); return; }
      const run = command("record.contents", { ...args, preview_hash: preview.preview_hash });
      await run.send();
      const where = targetId ? index.byId.get(targetId)?.title ?? "its new home" : "Unfiled";
      setToast({
        message: `Moved ${record.title} to ${where}${mode === "item" ? " without its contents" : ""}.`,
        undo: async () => { await command("record.restore_contents", { source_command_id: run.id }).send(); await load(); await onChanged(); setToast({ message: "Move undone." }); },
      });
      select(record.id);
      await load(); await onChanged();
    } catch (e) { setToast({ message: (e as Error).message }); }
    finally { setBusy(false); }
  };
  const create = async (type: string, title: string, parent: string | null) => {
    if (!data) return;
    setBusy(true);
    try {
      const made = (await command<{ id: string }>("record.create", { type_id: type, title, schema_revision: data.schema_revision, ...(parent ? { parent_id: parent } : {}) }).send()).data;
      await load(); await onChanged(); select(made.id);
      setToast({ message: `Added ${title}.` });
    } catch (e) { setToast({ message: (e as Error).message }); }
    finally { setBusy(false); }
  };

  const fromTemplate = async (preview: InstantiatePreview, title: string) => {
    setBusy(true);
    try {
      const made = await instantiate(preview, title);
      await load(); await onChanged(); select(made.result.id);
      setToast({
        message: `Created ${made.result.title} from the ${made.result.name} template.`,
        undo: async () => { await made.undo(); await load(); await onChanged(); setToast({ message: "Undone. The new records were archived." }); },
      });
    } finally { setBusy(false); }
  };
  const templateHooks = { templates, onTemplate: fromTemplate };

  // ---- Pointer: click to fly or select, drag to re-home ---------------------------------
  const svgPoint = (e: { clientX: number; clientY: number }): [number, number] => {
    const box = mapRef.current!.getBoundingClientRect();
    return [e.clientX - box.left, e.clientY - box.top];
  };
  const startDrag = (e: ReactPointerEvent, node: PackedNode) => {
    if (e.button !== 0 || !node.record || !layout || !index || node === focusNode) return;
    const record = node.record, origin = [e.clientX, e.clientY];
    let moved = false, legal: Set<string> | null = null, over: string | null = null;
    const moveHandler = (ev: PointerEvent) => {
      if (!moved && Math.hypot(ev.clientX - origin[0], ev.clientY - origin[1]) < 6) return;
      moved = true; legal ??= legalTargets(record, index);
      const pt = svgPoint(ev), { k, x, y } = project(viewRef.current, mapSize.width, mapSize.height);
      let best: { id: string; r: number } | null = null;
      for (const n of visibleNodes(layout, viewRef.current, mapSize.width, mapSize.height)) {
        if (!legal.has(n.id)) continue;
        const r = n.region ? n.r * k : itemRadius(n, k) + 8;
        if (Math.hypot(pt[0] - x(n.x), pt[1] - y(n.y)) <= r && (!best || r < best.r)) best = { id: n.id, r };
      }
      over = best?.id ?? null;
      setDrag({ id: record.id, legal, target: over, pt });
    };
    const up = (ev: PointerEvent) => {
      window.removeEventListener("pointermove", moveHandler); window.removeEventListener("pointerup", up);
      setDrag(null);
      if (!moved) return;
      suppressClick.current = true; setTimeout(() => { suppressClick.current = false; }, 0);
      if (over) setPopup({ kind: "move", x: ev.clientX, y: ev.clientY, record, target: over, targetTitle: index.byId.get(over)?.title ?? "" });
      else {
        const homes = (index.types.get(record.type_id)?.parent_types ?? []).map(h => index.types.get(h)?.plural).filter(Boolean);
        setToast({ message: `Drop ${record.title} on a highlighted home. ${index.types.get(record.type_id)?.plural ?? "Records"} can live in ${homes.join(", ") || "the top level only"}.` });
      }
    };
    window.addEventListener("pointermove", moveHandler); window.addEventListener("pointerup", up);
  };
  const clickMark = (node: PackedNode) => {
    if (suppressClick.current) return;
    if (node.region && node !== focusNode) { flyTo(node.id); select(node.id); return; }
    select(node.id);
  };
  const flyOut = () => { if (focusNode?.parent) flyTo(focusNode.parent.id); };
  const markKey = (e: KeyboardEvent, node: PackedNode) => {
    const siblings = siblingOrder(visible.filter(n => n.parent === node.parent));
    const at = siblings.findIndex(n => n.id === node.id);
    if (["ArrowRight", "ArrowDown", "ArrowLeft", "ArrowUp"].includes(e.key)) {
      e.preventDefault();
      const next = siblings[(at + (e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : siblings.length - 1)) % siblings.length];
      if (next) { keyboard.current = true; setActive(next.id); }
    } else if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      if (node.region && node !== focusNode) { flyTo(node.id); select(node.id); keyboard.current = true; setActive(node.children[0]?.id ?? node.id); }
      else select(node.id);
    } else if (e.key === "Backspace") {
      e.preventDefault();
      if (focusNode?.parent) { keyboard.current = true; setActive(focusNode.id); flyTo(focusNode.parent.id); }
    } else if (e.key === "Escape") select(null);
  };

  // ---- Blueprint drafts -------------------------------------------------------------------
  const working = draft ?? schema;
  const changeDraft = (next: Schema) => setDraft(draftChanges(schema, next) ? next : null);
  const typeDrop = (type: string, targetId: string, x: number, y: number) => {
    const choice = typeMoveChoice(working, type, targetId);
    if (choice.cycle) { setToast({ message: "That would put a type inside its own branch. Choose another place." }); return; }
    setPopup({ kind: "type", x, y, type, target: targetId });
  };
  const place = (type: string, targetId: string) => {
    const choice = typeMoveChoice(working, type, targetId);
    if (choice.allowed && !choice.cycle) { changeDraft(placeType(working, type, targetId)); setToast({ message: "Diagram rearranged in your draft. No records move." }); }
    else typeDrop(type, targetId, innerWidth / 2, innerHeight / 3);
  };

  if (error && !data) return <p role="alert" className="work-error">{error}</p>;
  if (!data || !index || !layout || !focusNode) return <p role="status" className="work-loading">Loading the Atlas…</p>;
  const record = selected ? index.byId.get(selected) ?? null : null;
  const path = focusNode === layout.root ? [] : [...ancestors(index, focusNode.id), focusNode.record!];
  const litType = hoverType ?? selectedType;
  const counts = new Map<string, number>();
  for (const r of data.records) counts.set(r.type_id, (counts.get(r.type_id) ?? 0) + 1);
  const results = searchRecords(index, query);
  const changes = draft ? draftChanges(schema, draft) : 0;
  const scale = mapSize.width ? scaleOf(view, mapSize.width, mapSize.height) : 1;
  const legend: Record<Lens, [string, string][]> = {
    status: [["open", "Open"], ["in_progress", "In progress"], ["done", "Done"], ["overdue", "Overdue"]],
    due: [["overdue", "Overdue"], ["today", "Due today"], ["upcoming", "Has a due date"]],
    review: [],
    type: [["work", "Work"], ["note", "Note"], ["goal", "Goal"]],
  };
  const popupStyle = popup ? { left: Math.max(16, Math.min(popup.x + 8, innerWidth - 336)), top: Math.max(16, Math.min(popup.y + 8, innerHeight - 260)) } : undefined;

  return <section className="atlas" aria-label="Atlas">
    <div className="atlas-bar">
      <div className="segmented" role="group" aria-label="Atlas view">{MODES.map(([id, label]) => <button key={id} type="button" aria-pressed={mode === id} onClick={() => setMode(id)}>{label}</button>)}</div>
      {mode !== "blueprint" && <div className="atlas-lenses" role="group" aria-label="Lens">{LENSES.map(([id, label]) => <button key={id} type="button" className="atlas-lens" aria-pressed={lens === id} onClick={() => setLens(id)}>{label}</button>)}</div>}
      <div className="atlas-search">
        <Search size={16} aria-hidden="true"/>
        <input ref={searchRef} type="search" aria-label="Fly to a record" placeholder="Fly to a record" value={query} autoComplete="off"
          onChange={e => setQuery(e.target.value)} onKeyDown={e => { if (e.key === "Enter" && results[0]) { e.preventDefault(); const goal = flyTarget(layout, index, results[0].id); flyTo(goal.focus); select(goal.select); setQuery(""); } if (e.key === "Escape") setQuery(""); }}/>
        <kbd aria-hidden="true">{/Mac|iP(hone|ad)/.test(navigator.platform) ? "⌘K" : "Ctrl K"}</kbd>
        {!!results.length && <ul className="atlas-results" aria-label="Matching records">{results.map(r => <li key={r.id}>
          <button type="button" onClick={() => { const goal = flyTarget(layout, index, r.id); flyTo(goal.focus); select(goal.select); setQuery(""); }}><span>{r.title}</span><small>{index.types.get(r.type_id)?.name}</small></button>
        </li>)}</ul>}
      </div>
    </div>
    {!!changes && <div className="atlas-draft" role="status"><span>Your structure draft has {changes} {changes === 1 ? "change" : "changes"}. Nothing applies until you review it.</span>
      <button type="button" className="btn btn-primary btn-sm" onClick={() => onEditTypes(draft, selectedType ?? undefined)}>Review and apply</button>
      <button type="button" className="btn btn-ghost btn-sm" onClick={() => setDraft(null)}>Discard draft</button></div>}
    {data.truncated && <p className="atlas-note">Showing the first {data.records.length} of {data.total} records, upper levels first. Browse lists everything.</p>}
    <div className="atlas-work">
      <div className={"atlas-stage is-" + mode}>
        {mode !== "blueprint" && <div className="atlas-pane">
          <div className="atlas-pane-head">
            <nav className="atlas-crumbs" aria-label="Atlas path">
              <button type="button" onClick={() => flyTo(ROOT)} aria-current={focusNode === layout.root ? "location" : undefined}>Workspace</button>
              {path.map(p => <span key={p.id}><ChevronRight size={13} aria-hidden="true"/><button type="button" onClick={() => flyTo(p.id)} aria-current={p.id === focusNode.id ? "location" : undefined}>{p.title}</button></span>)}
            </nav>
            <span className="atlas-pane-title">Your records</span>
          </div>
          <div className="atlas-map" ref={mapRef}>
            {mapSize.width > 0 && <AtlasMap layout={layout} index={index} view={view} width={mapSize.width} height={mapSize.height} focus={focusNode}
              selected={selected} active={active} lens={lens} litType={litType} links={data.links} drag={drag} today={today} canDrag={canEdit && !busy}
              onMarkPointerDown={startDrag} onMarkClick={clickMark} onBackgroundClick={() => select(null)} onFlyOut={flyOut} onMarkKey={markKey} onMarkFocus={n => setActive(n.id)}/>}
            {!data.records.length && <div className="atlas-empty"><p>Nothing here yet. Add a space or client to start your map.</p></div>}
          </div>
          <div className="atlas-legend" aria-hidden={lens === "review" ? undefined : "true"}>
            {lens === "review" ? <span>Reviews due appear here once types have a review cadence.</span> : legend[lens].map(([id, label]) => <span key={id}><i className={"atlas-key is-" + id}/>{label}</span>)}
            {lens !== "review" && <span><i className="atlas-key is-ring"/>Ring shows work done</span>}
            <span className="atlas-zoom tabular">{Math.round(scale * 100)}%</span>
          </div>
        </div>}
        {mode !== "atlas" && <div className="atlas-pane">
          <div className="atlas-pane-head"><span className="atlas-pane-title">Blueprint</span><span className="atlas-pane-sub">Types and where they may live</span>
            <label className="atlas-toggle"><input type="checkbox" checked={allHomes} onChange={e => setAllHomes(e.target.checked)}/>Show every allowed home</label></div>
          <div className="atlas-blueprint" ref={blueRef}>
            {blueSize.width > 0 && <Blueprint schema={working} width={blueSize.width} height={blueSize.height} selectedType={selectedType} recordType={record?.type_id ?? null}
              hoverType={hoverType} allHomes={allHomes} canDesign={canDesign} counts={counts} onHover={setHoverType} onSelect={selectType} onDropType={typeDrop}/>}
          </div>
        </div>}
      </div>
      <aside className="atlas-inspector" aria-label="Inspector" aria-live="polite">
        {selectedType ? <TypeInspector typeId={selectedType} draft={working} saved={schema} records={data.records} index={index} canDesign={canDesign}
            onDraft={changeDraft} onFly={id => { const goal = flyTarget(layout, index, id); flyTo(goal.focus); select(goal.select); }} onEditTypes={type => onEditTypes(draft, type)} onPlace={place}/>
          : record ? <RecordInspector record={record} index={index} schema={schema} today={today} canEdit={canEdit} focused={record.id === focusNode.id} busy={busy}
            onFly={id => flyTo(id)} onSelect={id => { const goal = flyTarget(layout, index, id); flyTo(goal.focus); select(goal.select); }} onOpen={onOpen} onBrowse={onBrowse}
            onType={id => { selectType(id); if (mode === "atlas") setMode("both"); }} onMove={move} onCreate={create} templates={templateHooks}/>
          : <FocusSummary focus={focusNode} index={index} canEdit={canEdit} schema={schema} busy={busy} onCreate={create} templates={templateHooks}/>}
      </aside>
    </div>
    {popup && <div className="atlas-popover" style={popupStyle} role="dialog" aria-label={popup.kind === "move" ? "Move choice" : "Type placement choice"}>
      {popup.kind === "move"
        ? <MoveChoiceList title={`Move ${popup.record.title} into ${popup.targetTitle}`} options={moveChoices(popup.record, index)} busy={busy}
            onCancel={() => setPopup(null)} onChoose={o => void move(popup.record, popup.target, o.mode)}/>
        : <TypeMoveChoices choice={typeMoveChoice(working, popup.type, popup.target)} onCancel={() => setPopup(null)}
            onRearrange={() => { changeDraft(placeType(working, popup.type, popup.target)); setPopup(null); setToast({ message: "Diagram rearranged in your draft. No records move." }); }}
            onAllow={() => { changeDraft(allowAndPlace(working, popup.type, popup.target)); setPopup(null); setToast({ message: "Allowed home added to your draft. Review and apply to save it." }); }}/>}
    </div>}
    {toast && <div className="toast" role="status"><span>{toast.message}</span>{toast.undo && <button type="button" disabled={busy} onClick={() => { const undo = toast.undo!; setToast(null); setBusy(true); void undo().catch(e => setToast({ message: e.message })).finally(() => setBusy(false)); }}>Undo</button>}</div>}
  </section>;

}

function FocusSummary({ focus: node, index, canEdit: editable, schema: s, busy: working, onCreate, templates }: { focus: PackedNode; index: AtlasIndex; canEdit: boolean; schema: Schema; busy: boolean; onCreate: (type: string, title: string, parent: string | null) => Promise<void>; templates: TemplateHooks }) {
  const rec = node.record, p = rec ? index.progress.get(rec.id) : null;
  const regions = node.children.filter(c => c.region).length, items = node.children.length - regions;
  return <>
    <div className="atlas-kicker"><span className="chip chip-type">{rec ? index.types.get(rec.type_id)?.name ?? "Group" : "Workspace"}</span></div>
    <h2 className="atlas-title">{rec ? rec.title : "Your workspace"}</h2>
    <dl className="atlas-facts">
      <div><dt>Regions inside</dt><dd className="tabular">{regions}</dd></div>
      <div><dt>Items inside</dt><dd className="tabular">{items}</dd></div>
      {!!p?.total && <div><dt>Work done</dt><dd className="tabular">{p.done} of {p.total}</dd></div>}
    </dl>
    <ul className="atlas-tips">
      <li>Click a region to fly in. Double-click empty space to fly out.</li>
      {editable && <li>Drag a record onto a highlighted home to move it.</li>}
      <li>Use the arrow keys between marks, Enter to open, Backspace to go back.</li>
    </ul>
    {editable && (rec ? isRegion(rec) : true) && <AddHere parent={rec} schema={s} busy={working} onCreate={onCreate} templates={templates}/>}
  </>;
}

