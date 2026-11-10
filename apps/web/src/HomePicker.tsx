import { useEffect, useMemo, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { ChevronDown, ChevronRight, Clock, FolderInput, Sparkles } from "lucide-react";
import { api } from "./api";
import { Popover } from "./ux";
import type { CustomRecord, Schema } from "./structure-types";
import type { PlacementData } from "./PlacementNote";
import "./home-picker.css";
import { groupByType, homeRef, offersSection, pathOf, recentHomes, refusal, saveRecentHome, type HomeRef } from "./home-picks";

type Choose = (home: HomeRef | null) => void;
type Page = { items: CustomRecord[]; next_offset?: number | null };

/** Main home picker: a floating "Move to…" panel with the current path, suggested and recent homes, a lazy tree
 * of homes, collapsed Tasks/Notes, and search over everything. `onChoose` gets the chosen home's id (null to
 * remove it from its home) and the home itself. */
export function HomePicker({row, schema, disabled, onChoose}: {row: CustomRecord; schema: Schema; disabled: boolean; onChoose: (id: string | null, home: HomeRef | null) => void}) {
  return <Popover floating label="Choose main home" title={pathOf(row) || "Unfiled"} disabled={disabled}
    buttonClassName="prop-text home-trigger" panelClassName="home-picker"
    button={<><FolderInput size={14} aria-hidden="true"/><span>{row.home.at(-1)?.title ?? "Unfiled"}</span></>}>
    {close => <HomePanel row={row} schema={schema} onChoose={home => {
      close();
      if (home) saveRecentHome(home);
      if ((home?.id ?? null) !== row.parent_id) onChoose(home?.id ?? null, home);
    }}/>}
  </Popover>;
}

const ROW = "[data-home-row]:not(:disabled)";
function HomePanel({row, schema, onChoose}: {row: CustomRecord; schema: Schema; onChoose: Choose}) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<CustomRecord[] | null>(null);
  const [error, setError] = useState("");
  const [suggestion, setSuggestion] = useState<{ home: HomeRef; reason: string } | null>(null);
  const panel = useRef<HTMLDivElement>(null);
  const recent = useMemo(() => recentHomes().filter(h => h.id !== row.parent_id && h.id !== row.id), [row.id, row.parent_id]);
  useEffect(() => {
    if (!row.task_id) return;
    let live = true;
    void api<PlacementData>(`/structure/records/${row.id}/placement`).then(d => {
      const s = d.suggestion;
      if (live && s && s.home_id !== row.parent_id) setSuggestion({ home: { id: s.home_id, title: s.title, type_id: s.type_id, path: "" }, reason: s.reason });
    }).catch(() => {});
    return () => { live = false; };
  }, [row.id, row.task_id, row.parent_id]);
  useEffect(() => {
    const text = query.trim();
    if (!text) { setResults(null); return; }
    let live = true;
    const timer = setTimeout(() => {
      void api<Page>("/structure/records?" + new URLSearchParams({ query: text, limit: "30" }))
        .then(r => { if (live) { setResults(r.items); setError(""); } }).catch(e => { if (live) setError((e as Error).message); });
    }, 180);
    return () => { live = false; clearTimeout(timer); };
  }, [query]);
  // ↑/↓ walk the visible rows (the search box is the top); → and ← open and close tree branches (TreeNode).
  const walk = (e: KeyboardEvent) => {
    if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
    const stops = [panel.current?.querySelector<HTMLElement>("input[type=search]"), ...Array.from(panel.current?.querySelectorAll<HTMLElement>(ROW) ?? [])].filter((x): x is HTMLElement => !!x);
    const at = stops.indexOf(document.activeElement as HTMLElement);
    const next = stops[Math.min(stops.length - 1, Math.max(0, at + (e.key === "ArrowDown" ? 1 : -1)))];
    if (next) { e.preventDefault(); next.focus(); }
  };
  const choice = (home: HomeRef, why: string, detail: ReactNode, icon?: ReactNode) =>
    <Choice key={home.id} home={home} why={why} detail={detail} icon={icon} onChoose={onChoose}/>;
  const known = (home: HomeRef) => refusal(schema, row, { ...home, home: [] });
  return <div ref={panel} className="home-panel" onKeyDown={walk}>
    <input type="search" aria-label="Find a home" placeholder="Search everything…" value={query} onChange={e => setQuery(e.target.value)}/>
    {error && <p role="alert" className="home-error">{error}</p>}
    {query.trim() ? <div className="home-results" aria-live="polite">
      {results && !results.length && <p className="footnote">Nothing matches “{query.trim()}”.</p>}
      {results && groupByType(results.filter(r => r.id !== row.id)).map(g => <section key={g.type} className="home-group" aria-label={g.type}>
        <h4>{g.type}</h4>
        {g.rows.map(r => choice(homeRef(r), refusal(schema, row, r), pathOf(r) || "Top level"))}
      </section>)}
    </div> : <>
      <section className="home-group" aria-label="Now">
        <h4>Now</h4>
        <div className="home-now"><span>{pathOf(row) || "Unfiled"}</span>
          {row.parent_id && <button type="button" className="text-button" data-home-row onClick={() => onChoose(null)}>Remove from home</button>}</div>
      </section>
      {(suggestion || recent.length > 0) && <section className="home-group" aria-label="Suggested">
        <h4>Suggested</h4>
        {suggestion && choice(suggestion.home, known(suggestion.home), suggestion.reason || "Suggested by meaning", <Sparkles size={13} aria-hidden="true"/>)}
        {recent.filter(h => h.id !== suggestion?.home.id).map(h => choice(h, known(h), h.path || "Top level", <Clock size={13} aria-hidden="true"/>))}
      </section>}
      <section className="home-group" aria-label="Homes">
        <h4>Homes</h4>
        <div role="tree" aria-label="Homes"><Tree parent={null} depth={0} row={row} schema={schema} onChoose={onChoose}/></div>
      </section>
      {offersSection(schema, row, "work") && <Section label="Tasks" capability="work" row={row} schema={schema} onChoose={onChoose}/>}
      {offersSection(schema, row, "content") && <Section label="Notes" capability="content" row={row} schema={schema} onChoose={onChoose}/>}
    </>}
  </div>;
}

function Choice({home, why, detail, icon, onChoose, onKeyDown}: {home: HomeRef; why: string; detail: ReactNode; icon?: ReactNode; onChoose: Choose; onKeyDown?: (e: KeyboardEvent) => void}) {
  return <button type="button" className="home-choice" data-home-row disabled={!!why} title={why || undefined}
    aria-label={"Use " + home.title + " as home"} onKeyDown={onKeyDown} onClick={() => onChoose(home)}>
    <span className="home-choice-title">{icon}{home.title}</span>
    <small>{why || detail}</small>
  </button>;
}

function Tree({parent, depth, row, schema, onChoose}: {parent: string | null; depth: number; row: CustomRecord; schema: Schema; onChoose: Choose}) {
  const [items, setItems] = useState<CustomRecord[] | null>(null), [error, setError] = useState("");
  useEffect(() => {
    let live = true;
    const params = new URLSearchParams({ section: "groups", limit: "100" });
    if (parent) params.set("parent_id", parent);
    void api<Page>("/structure/browse?" + params).then(r => { if (live) setItems(r.items); }).catch(e => { if (live) setError((e as Error).message); });
    return () => { live = false; };
  }, [parent]);
  if (error) return <p role="alert" className="home-error">{error}</p>;
  if (!items) return <p className="footnote home-loading" style={{ paddingLeft: depth * 16 + 24 }}>Loading…</p>;
  const shown = items.filter(i => i.id !== row.id);
  if (!shown.length) return <p className="footnote home-loading" style={{ paddingLeft: depth * 16 + 24 }}>{depth ? "No homes inside." : "No homes yet."}</p>;
  return <>{shown.map(i => <TreeNode key={i.id} item={i} depth={depth} row={row} schema={schema} onChoose={onChoose}/>)}</>;
}

function TreeNode({item, depth, row, schema, onChoose}: {item: CustomRecord; depth: number; row: CustomRecord; schema: Schema; onChoose: Choose}) {
  const [open, setOpen] = useState(false);
  const branches = (item.contents?.direct ?? 0) + (item.contents?.context_direct ?? 0) > 0;
  const keys = (e: KeyboardEvent) => {
    if (e.key === "ArrowRight" && branches && !open) { e.preventDefault(); setOpen(true); }
    else if (e.key === "ArrowLeft" && open) { e.preventDefault(); setOpen(false); }
  };
  return <div role="treeitem" aria-level={depth + 1} aria-expanded={branches ? open : undefined} aria-selected={item.id === row.parent_id}>
    <div className={"home-tree-row" + (item.id === row.parent_id ? " is-current" : "")} style={{ paddingLeft: depth * 16 }}>
      {branches ? <button type="button" className="home-twisty" tabIndex={-1} aria-label={(open ? "Collapse " : "Expand ") + item.title} onClick={() => setOpen(!open)}>
        {open ? <ChevronDown size={14}/> : <ChevronRight size={14}/>}</button> : <span className="home-twisty" aria-hidden="true"/>}
      <Choice home={homeRef(item)} why={refusal(schema, row, item)} detail={item.type_name} onChoose={onChoose} onKeyDown={keys}/>
    </div>
    {open && <div role="group"><Tree parent={item.id} depth={depth + 1} row={row} schema={schema} onChoose={onChoose}/></div>}
  </div>;
}

/** Tasks or Notes as homes: collapsed until asked for, newest first, 20 at a time. */
function Section({label, capability, row, schema, onChoose}: {label: string; capability: "work" | "content"; row: CustomRecord; schema: Schema; onChoose: Choose}) {
  const [open, setOpen] = useState(false), [items, setItems] = useState<CustomRecord[]>([]), [next, setNext] = useState<number | null>(0), [error, setError] = useState("");
  const load = (offset: number) => void api<Page>("/structure/records?" + new URLSearchParams({ capability, limit: "20", offset: String(offset) }))
    .then(r => { setItems(old => [...(offset ? old : []), ...r.items]); setNext(r.next_offset ?? null); }).catch(e => setError((e as Error).message));
  return <section className="home-group home-section" aria-label={label}>
    <button type="button" className="home-section-head" data-home-row aria-expanded={open} onClick={() => { if (!open && !items.length) load(0); setOpen(!open); }}>
      {open ? <ChevronDown size={14}/> : <ChevronRight size={14}/>}{label}</button>
    {open && <>
      {items.filter(i => i.id !== row.id).map(i => <Choice key={i.id} home={homeRef(i)} why={refusal(schema, row, i)} detail={pathOf(i) || "Top level"} onChoose={onChoose}/>)}
      {error && <p role="alert" className="home-error">{error}</p>}
      {next !== null && items.length > 0 && <button type="button" className="text-button home-more" onClick={() => load(next)}>More {label.toLowerCase()}</button>}
    </>}
  </section>;
}
