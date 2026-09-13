import { useEffect, useRef, useState } from "react";
import { Plus, Target, Folder, ArrowRight, X } from "lucide-react";
import { useDialogFocus } from "./components";
import type { Project } from "./types";
import type { Organization, Home, Goal, Space, Area, Actor, OrganizationFilter } from "./productivity";

type Mutate = (tool: string, args: unknown, message: string) => Promise<unknown>;
type Kind = "goal" | "project" | "area" | "space" | "actor";
type RecordRow = { id: string; name: string; description?: string; revision: number; archived: boolean;
  space_id?: string | null; area_id?: string | null; status?: string; success_criteria?: string;
  horizon?: string; parent_goal_id?: string | null; target_date?: string | null; start_date?: string | null;
  metric_unit?: string; metric_baseline?: number; metric_current?: number | null; metric_target?: number | null;
  project_ids?: string[]; goal_ids?: string[]; kind?: string };

export function LinkPicker({ label, items, selected, onChange }: {
  label: string; items: { id: string; name: string; archived?: boolean }[]; selected: string[]; onChange: (ids: string[]) => void;
}) {
  const [query, setQuery] = useState("");
  return <fieldset className="relationship-picker"><legend>{label} · {selected.length}</legend>
    {items.length > 6 && <input aria-label={"Find " + label.toLowerCase()} placeholder="Find a record…" value={query} onChange={e => setQuery(e.target.value)} />}
    <div className="relationship-options">
      {items.filter(i => (!i.archived || selected.includes(i.id)) && (selected.includes(i.id) || i.name.toLowerCase().includes(query.toLowerCase()))).map(i =>
        <label key={i.id}><input type="checkbox" checked={selected.includes(i.id)} onChange={e => onChange(e.target.checked ? [...selected, i.id] : selected.filter(id => id !== i.id))} />{i.name}{i.archived ? " (archived)" : ""}</label>)}
      {!items.length && <span className="footnote">No records yet.</span>}
    </div>
  </fieldset>;
}

export function HomeFields({ value, onChange, organization, disabled = false }: {
  value: Home; onChange: (home: Home) => void; organization: Organization; disabled?: boolean;
}) {
  return <div className="form-grid">
    <label>Space<select aria-label="Space" disabled={disabled} value={value.space_id ?? ""} onChange={e => onChange({ space_id: e.target.value || null, area_id: null })}>
      <option value="">Unclassified</option>{organization.spaces.filter(s => !s.archived || s.id === value.space_id).map(s => <option key={s.id} value={s.id}>{s.name}</option>)}
    </select></label>
    <label>Area<select aria-label="Area" disabled={disabled} value={value.area_id ?? ""} onChange={e => {
      const area = organization.areas.find(a => a.id === e.target.value);
      onChange({ space_id: area?.space_id ?? value.space_id, area_id: area?.id ?? null });
    }}>
      <option value="">No area</option>{organization.areas.filter(a => (!a.archived || a.id === value.area_id) && (!value.space_id || a.space_id === value.space_id)).map(a => <option key={a.id} value={a.id}>{a.name}</option>)}
    </select></label>
  </div>;
}

export function OrganizationFilters({ organization, value, onChange }: {
  organization: Organization; value: OrganizationFilter; onChange: (v: OrganizationFilter) => void;
}) {
  return <>
    <label>Space<select aria-label="Space filter" value={value.space} onChange={e => onChange({ space: e.target.value, area: "", goal: "" })}>
      <option value="">All spaces</option>{organization.spaces.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}
    </select></label>
    <label>Area<select aria-label="Area filter" value={value.area} onChange={e => onChange({ ...value, area: e.target.value })}>
      <option value="">All areas</option>{organization.areas.filter(a => !value.space || a.space_id === value.space).map(a => <option key={a.id} value={a.id}>{a.name}</option>)}
    </select></label>
    <label>Goal<select aria-label="Goal filter" value={value.goal} onChange={e => onChange({ ...value, goal: e.target.value })}>
      <option value="">All goals</option>{organization.goals.map(g => <option key={g.id} value={g.id}>{g.name}</option>)}
    </select></label>
  </>;
}

export function ProductivityPage({ organization, busy, mutate, onProject, onNote, query, highlight, onEditing }: {
  organization: Organization; busy: boolean; mutate: Mutate; onProject: (p: Project) => void; onNote: (id: string) => void;
  query: string; highlight?: string | null; onEditing: (editing: boolean) => void;
}) {
  const [tab, setTab] = useState<Kind>("goal");
  const [editor, setEditor] = useState<{ kind: Kind; row?: RecordRow } | null>(null);
  const [space, setSpace] = useState("");
  const [archived, setArchived] = useState(false);
  const collections = { goal: organization.goals, project: organization.projects, area: organization.areas, space: organization.spaces, actor: organization.actors };
  const rows: RecordRow[] = collections[tab];
  const visible = rows.filter(r => r.archived === archived && (!space || tab === "space" || tab === "actor" || r.space_id === space) && (r.name + " " + (r.description ?? "")).toLowerCase().includes(query.toLowerCase()));
  const edit = (kind: Kind, row?: RecordRow) => { setEditor({ kind, row }); onEditing(true); };
  const close = () => { setEditor(null); onEditing(false); };
  const seenHighlight = useRef<string | null>(null);
  useEffect(() => {
    if (!highlight || seenHighlight.current === highlight) return;
    for (const kind of ["goal", "project", "area", "space", "actor"] as Kind[]) {
      const found = collections[kind].find(r => r.id === highlight);
      if (found) { seenHighlight.current = highlight; setTab(kind); setSpace(""); setArchived(found.archived); break; }
    }
  }, [highlight, organization]);
  useEffect(() => () => onEditing(false), [onEditing]);
  const labels = { goal: "Goals", project: "Projects", area: "Areas", space: "Spaces", actor: "People & agents" };
  return <section className="productivity-page" aria-label="Goals and projects">
    <div className="organization-tabs" role="tablist" aria-label="Organization">
      {(Object.keys(labels) as Kind[]).map(kind => <button key={kind} role="tab" aria-selected={tab === kind} onClick={() => setTab(kind)}>{labels[kind]}</button>)}
    </div>
    <div className="workspace-actions">
      <label>Space<select aria-label="Organization space" value={space} onChange={e => setSpace(e.target.value)}>
        <option value="">All spaces</option>{organization.spaces.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}
      </select></label>
      <label className="inline-check"><input type="checkbox" checked={archived} onChange={e => setArchived(e.target.checked)} />Archived</label>
      <button className="primary" onClick={() => edit(tab)}><Plus size={17} />New {tab === "actor" ? "assignee" : tab}</button>
    </div>
    <p className="footnote">{tab === "goal" ? "Track the outcome you want. Connect the projects that support it." : tab === "project" ? "Organize a finite body of work and connect it to your goals." : tab === "area" ? "Ongoing responsibilities, such as Health, Home or Operations." : tab === "actor" ? "Assignment records responsibility. Assigning work to Eri does not start an agent run." : "Personal and business contexts. These spaces are private to you."}</p>
    <div className="organization-list">
      {visible.map(row => {
        const goal = tab === "goal" ? row as Goal : null;
        const project = tab === "project" ? row as Project : null;
        const linked = goal ? organization.projects.filter(p => goal.project_ids.includes(p.id)) : project ? organization.goals.filter(g => project.goal_ids?.includes(g.id)) : [];
        const notes = goal?.notes ?? project?.notes ?? [];
        const home = [organization.spaces.find(s => s.id === row.space_id)?.name, organization.areas.find(a => a.id === row.area_id)?.name].filter(Boolean).join(" / ");
        return <article className={"organization-row " + (highlight === row.id ? "highlighted" : "")} key={row.id} data-entity-id={row.id}>
          <div className="organization-row-heading">
            <button className="organization-title" onClick={() => edit(tab, row)}>{tab === "goal" ? <Target size={19} /> : <Folder size={19} />}<strong>{row.name}</strong></button>
            <span className="record-status">{(row.status ?? row.kind ?? "").replaceAll("_", " ")}</span>
          </div>
          {home && <div className="eyebrow">{home}</div>}
          {row.description && <p>{row.description}</p>}
          {row.success_criteria && <p className="success-criteria">Success: {row.success_criteria}</p>}
          {row.target_date && <span className="footnote">Target: {row.target_date}</span>}
          {goal?.parent_goal_id && <p className="footnote">Supports goal: {organization.goals.find(g => g.id === goal.parent_goal_id)?.name}</p>}
          {goal && goal.metric_target !== null && <div className="goal-metric">
            <span>{goal.metric_current ?? "—"} / {goal.metric_target} {goal.metric_unit}</span>
            {goal.progress !== null && <progress aria-label={"Outcome progress: " + goal.name} max={1} value={goal.progress} />}
          </div>}
          {project && <p className="footnote">{project.completed_task_count ?? 0} of {project.task_count ?? 0} tasks completed</p>}
          {!!linked.length && <div className="organization-links">{linked.map(link => <button key={link.id} className="text-button" onClick={() => edit(goal ? "project" : "goal", link)}>{link.name}<ArrowRight size={13} /></button>)}</div>}
          {!!notes.length && <div className="organization-links">{notes.map(note => <button key={note.id} className="text-button" onClick={() => onNote(note.id)}>Note: {note.title}</button>)}</div>}
          <div className="organization-row-actions">
            <button className="text-button" onClick={() => edit(tab, row)}>Edit</button>
            {project && <button className="text-button" onClick={() => onProject(project)}>View tasks<ArrowRight size={14} /></button>}
            <button className="text-button" disabled={busy} onClick={() => void mutate(tab + ".update", { [tab + "_id"]: row.id, expected_revision: row.revision, archived: !row.archived }, row.archived ? "Restored" : "Archived")}>{row.archived ? "Restore" : "Archive"}</button>
          </div>
        </article>;
      })}
      {!visible.length && <p className="empty-state">No {labels[tab].toLowerCase()} here yet.</p>}
    </div>
    {editor && <OrganizationEditor key={editor.kind + ":" + (editor.row?.id ?? "new")} {...editor} organization={organization} busy={busy} mutate={mutate} onClose={close} />}
  </section>;
}

function OrganizationEditor({ kind, row, organization, busy, mutate, onClose }: {
  kind: Kind; row?: RecordRow; organization: Organization; busy: boolean; mutate: Mutate; onClose: () => void;
}) {
  useDialogFocus();
  const [home, setHome] = useState<Home>({ space_id: row?.space_id, area_id: row?.area_id });
  const [links, setLinks] = useState(row?.project_ids ?? row?.goal_ids ?? []);
  const [localError, setLocalError] = useState("");
  const isGoal = kind === "goal", isProject = kind === "project";
  return <div className="modal-backdrop"><form className="dialog organization-editor" role="dialog" aria-modal="true" aria-labelledby="organization-title" onSubmit={async e => {
    e.preventDefault(); setLocalError("");
    const form = new FormData(e.currentTarget);
    const text = (key: string) => String(form.get(key) ?? "").trim();
    const args: Record<string, unknown> = { name: text("name") };
    if (kind !== "actor") args.description = text("description");
    if (isGoal || isProject) Object.assign(args, home, { space_id: home.space_id || null, area_id: home.area_id || null, status: text("status"), success_criteria: text("success_criteria"), target_date: text("target_date") || null });
    if (kind === "area") args.space_id = text("space_id");
    if (isProject) Object.assign(args, { start_date: text("start_date") || null, goal_ids: links });
    if (isGoal) Object.assign(args, { project_ids: links, parent_goal_id: text("parent_goal_id") || null, horizon: text("horizon"), metric_unit: text("metric_unit"), metric_baseline: Number(text("metric_baseline") || 0), metric_current: text("metric_current") ? Number(text("metric_current")) : null, metric_target: text("metric_target") ? Number(text("metric_target")) : null });
    if (kind === "actor" && !row) args.kind = text("kind");
    if (row) Object.assign(args, { [kind + "_id"]: row.id, expected_revision: row.revision });
    const result = await mutate(kind + (row ? ".update" : ".create"), args, (isGoal ? "Goal" : isProject ? "Project" : "Record") + " saved");
    if (result) onClose(); else setLocalError("Could not save. Check the error message; your changes are still here. Close and reopen to load a newer revision if needed.");
  }}>
    <div className="dialog-heading"><h2 id="organization-title">{row ? "Edit" : "New"} {kind === "actor" ? "assignee" : kind}</h2><button type="button" className="icon-button" aria-label="Close organization editor" onClick={onClose}><X size={20} /></button></div>
    {localError && <p role="alert" className="error-banner">{localError}</p>}
    <label>Name<input name="name" required maxLength={kind === "actor" ? 100 : 200} defaultValue={row?.name ?? ""} autoFocus /></label>
    {kind !== "actor" && <label>Description<textarea name="description" rows={3} maxLength={10000} defaultValue={row?.description ?? ""} /></label>}
    {(isGoal || isProject) && <><HomeFields value={home} onChange={setHome} organization={organization} />
      <label>Success criteria<textarea name="success_criteria" maxLength={10000} rows={2} defaultValue={row?.success_criteria ?? ""} placeholder={isGoal ? "How will you know this outcome is achieved?" : "What must be delivered to finish this project?"} /></label>
      <div className="form-grid"><label>Status<select aria-label="Status" name="status" defaultValue={row?.status ?? "planned"}>{(isGoal ? ["planned", "active", "on_hold", "achieved", "abandoned"] : ["planned", "active", "on_hold", "completed", "cancelled"]).map(s => <option key={s} value={s}>{s.replaceAll("_", " ")}</option>)}</select></label>
      <label>Target date<input name="target_date" type="date" defaultValue={row?.target_date ?? ""} /></label>
      {isProject && <label>Start date<input name="start_date" type="date" defaultValue={row?.start_date ?? ""} /></label>}
      {isGoal && <><label>Horizon<select aria-label="Horizon" name="horizon" defaultValue={row?.horizon ?? "unspecified"}><option value="unspecified">Unspecified</option><option value="short_term">Short term</option><option value="long_term">Long term</option></select></label>
      <label>Parent goal<select aria-label="Parent goal" name="parent_goal_id" defaultValue={row?.parent_goal_id ?? ""}><option value="">No parent goal</option>{organization.goals.filter(g => g.id !== row?.id && (!g.archived || g.id === row?.parent_goal_id)).map(g => <option key={g.id} value={g.id}>{g.name}</option>)}</select></label></>}
      </div></>}
    {isGoal && <details className="metric-editor" open={row?.metric_target != null || undefined}><summary>Optional outcome metric</summary><div className="form-grid">
      <label>Unit<input name="metric_unit" maxLength={80} defaultValue={row?.metric_unit ?? ""} placeholder="clients, km, hours…" /></label>
      <label>Starting value<input name="metric_baseline" type="number" step="any" defaultValue={row?.metric_baseline ?? 0} /></label>
      <label>Current value<input name="metric_current" type="number" step="any" defaultValue={row?.metric_current ?? ""} /></label>
      <label>Target value<input name="metric_target" type="number" step="any" defaultValue={row?.metric_target ?? ""} /></label>
    </div><p className="footnote">Tracks the outcome independently of completed tasks.</p></details>}
    {(isGoal || isProject) && <LinkPicker label={isGoal ? "Supporting projects" : "Supported goals"} items={isGoal ? organization.projects : organization.goals} selected={links} onChange={setLinks} />}
    {kind === "area" && <label>Space<select aria-label="Space" name="space_id" required defaultValue={row?.space_id ?? ""}><option value="">Choose a space</option>{organization.spaces.filter(s => !s.archived || s.id === row?.space_id).map(s => <option key={s.id} value={s.id}>{s.name}</option>)}</select></label>}
    {kind === "actor" && !row && <label>Type<select aria-label="Type" name="kind" defaultValue="person"><option value="person">Person</option><option value="agent">Agent</option></select></label>}
    <div className="dialog-actions"><button type="button" className="secondary" onClick={onClose}>Cancel</button><button className="primary" disabled={busy}>Save {kind === "actor" ? "assignee" : kind}</button></div>
  </form></div>;
}
