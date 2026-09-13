import { z } from "zod";
import { useEditor, nullableId, choice } from "./editor-control";
import { LayoutSwitch, Timeline } from "./WorkViews";
import type { WorkLayout, TimelineSpan } from "./work-views";
import { useEffect, useRef, useState } from "react";
import { Plus, Target, Folder, ArrowRight, X } from "lucide-react";
import { useDialogFocus } from "./components";
import type { Project } from "./types";
import type {
  Organization,
  Home,
  Goal,
  Space,
  Area,
  Actor,
  OrganizationFilter,
} from "./productivity";

type Mutate = (
  tool: string,
  args: unknown,
  message: string,
) => Promise<unknown>;
export type Kind = "goal" | "project" | "area" | "space" | "actor";
type RecordRow = {
  id: string;
  name: string;
  description?: string;
  revision: number;
  archived: boolean;
  space_id?: string | null;
  area_id?: string | null;
  status?: string;
  success_criteria?: string;
  horizon?: string;
  parent_goal_id?: string | null;
  target_date?: string | null;
  start_date?: string | null;
  metric_unit?: string;
  metric_baseline?: number;
  metric_current?: number | null;
  metric_target?: number | null;
  project_ids?: string[];
  goal_ids?: string[];
  kind?: string;
};

export function LinkPicker({
  label,
  items,
  selected,
  onChange,
}: {
  label: string;
  items: { id: string; name: string; archived?: boolean }[];
  selected: string[];
  onChange: (ids: string[]) => void;
}) {
  const [query, setQuery] = useState("");
  return (
    <fieldset className="relationship-picker">
      <legend>
        {label} · {selected.length}
      </legend>
      {items.length > 6 && (
        <input
          aria-label={"Find " + label.toLowerCase()}
          placeholder="Find a record…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      )}
      <div className="relationship-options">
        {items
          .filter(
            (i) =>
              (!i.archived || selected.includes(i.id)) &&
              (selected.includes(i.id) ||
                i.name.toLowerCase().includes(query.toLowerCase())),
          )
          .map((i) => (
            <label key={i.id}>
              <input
                type="checkbox"
                checked={selected.includes(i.id)}
                onChange={(e) =>
                  onChange(
                    e.target.checked
                      ? [...selected, i.id]
                      : selected.filter((id) => id !== i.id),
                  )
                }
              />
              {i.name}
              {i.archived ? " (archived)" : ""}
            </label>
          ))}
        {!items.length && <span className="footnote">No records yet.</span>}
      </div>
    </fieldset>
  );
}

export function HomeFields({
  value,
  onChange,
  organization,
  disabled = false,
}: {
  value: Home;
  onChange: (home: Home) => void;
  organization: Organization;
  disabled?: boolean;
}) {
  return (
    <div className="form-grid">
      <label>
        Space
        <select
          aria-label="Space"
          disabled={disabled}
          value={value.space_id ?? ""}
          onChange={(e) =>
            onChange({ space_id: e.target.value || null, area_id: null })
          }
        >
          <option value="">Unclassified</option>
          {organization.spaces
            .filter((s) => !s.archived || s.id === value.space_id)
            .map((s) => (
              <option key={s.id} value={s.id}>
                {s.name}
              </option>
            ))}
        </select>
      </label>
      <label>
        Area
        <select
          aria-label="Area"
          disabled={disabled}
          value={value.area_id ?? ""}
          onChange={(e) => {
            const area = organization.areas.find(
              (a) => a.id === e.target.value,
            );
            onChange({
              space_id: area?.space_id ?? value.space_id,
              area_id: area?.id ?? null,
            });
          }}
        >
          <option value="">No area</option>
          {organization.areas
            .filter(
              (a) =>
                (!a.archived || a.id === value.area_id) &&
                (!value.space_id || a.space_id === value.space_id),
            )
            .map((a) => (
              <option key={a.id} value={a.id}>
                {a.name}
              </option>
            ))}
        </select>
      </label>
    </div>
  );
}

export function OrganizationFilters({
  organization,
  value,
  onChange,
}: {
  organization: Organization;
  value: OrganizationFilter;
  onChange: (v: OrganizationFilter) => void;
}) {
  return (
    <>
      <label>
        Space
        <select
          aria-label="Space filter"
          value={value.space}
          onChange={(e) =>
            onChange({ space: e.target.value, area: "", goal: "" })
          }
        >
          <option value="">All spaces</option>
          {organization.spaces.map((s) => (
            <option key={s.id} value={s.id}>
              {s.name}
            </option>
          ))}
        </select>
      </label>
      <label>
        Area
        <select
          aria-label="Area filter"
          value={value.area}
          onChange={(e) => onChange({ ...value, area: e.target.value })}
        >
          <option value="">All areas</option>
          {organization.areas
            .filter((a) => !value.space || a.space_id === value.space)
            .map((a) => (
              <option key={a.id} value={a.id}>
                {a.name}
              </option>
            ))}
        </select>
      </label>
      <label>
        Goal
        <select
          aria-label="Goal filter"
          value={value.goal}
          onChange={(e) => onChange({ ...value, goal: e.target.value })}
        >
          <option value="">All goals</option>
          {organization.goals.map((g) => (
            <option key={g.id} value={g.id}>
              {g.name}
            </option>
          ))}
        </select>
      </label>
    </>
  );
}

export function ProductivityPage({
  organization,
  busy,
  mutate,
  onProject,
  onNote,
  query,
  highlight,
  onEditing,
  tab,
  onTab,
  layout,
  onLayout,
  archived,
  onArchived,
  space,
  onSpace,
  editorRequest,
  onEditorRequestHandled,
  onVisible,
  today,
  timelineDate,
  timelineSpan,
  onTimelineDate,
  onTimelineSpan,
}: {
  organization: Organization;
  busy: boolean;
  mutate: Mutate;
  onProject: (p: Project) => void;
  onNote: (id: string) => void;
  query: string;
  highlight?: string | null;
  onEditing: (editing: boolean) => void;
  tab: Kind;
  onTab: (v: Kind) => void;
  layout: WorkLayout;
  onLayout: (v: WorkLayout) => void;
  archived: boolean;
  onArchived: (v: boolean) => void;
  space: string;
  onSpace: (v: string) => void;
  editorRequest: { kind: Kind; id?: string; sequence: number } | null;
  onEditorRequestHandled: () => void;
  onVisible: (ids: string[]) => void;
  today: string;
  timelineDate: string;
  timelineSpan: TimelineSpan;
  onTimelineDate: (v: string) => void;
  onTimelineSpan: (v: TimelineSpan) => void;
}) {
  const [editor, setEditor] = useState<{ kind: Kind; row?: RecordRow } | null>(
    null,
  );
  const collections = {
    goal: organization.goals,
    project: organization.projects,
    area: organization.areas,
    space: organization.spaces,
    actor: organization.actors,
  };
  const rows: RecordRow[] = collections[tab];
  const visible = rows
    .filter(
      (r) =>
        r.archived === archived &&
        (!space ||
          tab === "space" ||
          tab === "actor" ||
          r.space_id === space) &&
        (r.name + " " + (r.description ?? ""))
          .toLowerCase()
          .includes(query.toLowerCase()),
    )
    .sort((a, b) => a.name.localeCompare(b.name));
  const visibleKey = visible
    .slice(0, 60)
    .map((r) => r.id)
    .join(",");
  useEffect(
    () => onVisible(visibleKey ? visibleKey.split(",") : []),
    [visibleKey, onVisible],
  );
  const edit = (kind: Kind, row?: RecordRow) => {
    setEditor({ kind, row });
    onEditing(true);
  };
  const close = () => {
    setEditor(null);
    onEditing(false);
  };
  const seenRequest = useRef(0);
  useEffect(() => {
    if (!editorRequest || seenRequest.current === editorRequest.sequence)
      return;
    seenRequest.current = editorRequest.sequence;
    edit(
      editorRequest.kind,
      editorRequest.id
        ? collections[editorRequest.kind].find((r) => r.id === editorRequest.id)
        : undefined,
    );
    onEditorRequestHandled();
  }, [editorRequest]);
  const seenHighlight = useRef<string | null>(null);
  useEffect(() => {
    if (!highlight || seenHighlight.current === highlight) return;
    for (const kind of [
      "goal",
      "project",
      "area",
      "space",
      "actor",
    ] as Kind[]) {
      const found = collections[kind].find((r) => r.id === highlight);
      if (found) {
        seenHighlight.current = highlight;
        onTab(kind);
        onSpace("");
        onArchived(found.archived);
        break;
      }
    }
  }, [highlight, organization]);
  useEffect(() => () => onEditing(false), [onEditing]);
  const labels = {
    goal: "Goals",
    project: "Projects",
    area: "Areas",
    space: "Spaces",
    actor: "People & agents",
  };
  function card(row: RecordRow) {
    const goal = tab === "goal" ? (row as Goal) : null,
      project = tab === "project" ? (row as Project) : null;
    const linked = goal
      ? organization.projects.filter((p) => goal.project_ids.includes(p.id))
      : project
        ? organization.goals.filter((g) => project.goal_ids?.includes(g.id))
        : [];
    const notes = goal?.notes ?? project?.notes ?? [];
    const home = [
      organization.spaces.find((s) => s.id === row.space_id)?.name,
      organization.areas.find((a) => a.id === row.area_id)?.name,
    ]
      .filter(Boolean)
      .join(" / ");
    return (
      <article
        className={
          "organization-row " + (highlight === row.id ? "highlighted" : "")
        }
        key={row.id}
        data-entity-id={row.id}
        id={"record-" + row.id}
        draggable={tab === "project" && layout === "board" && !busy}
        onDragStart={(e) =>
          e.dataTransfer.setData("application/eridani-project", row.id)
        }
      >
        <div className="organization-row-heading">
          <button className="organization-title" onClick={() => edit(tab, row)}>
            {tab === "goal" ? <Target size={16} /> : <Folder size={16} />}
            <strong>{row.name}</strong>
          </button>
          <span className="record-status">
            {(row.status ?? row.kind ?? "").replaceAll("_", " ")}
          </span>
        </div>
        <div className="record-meta">
          {home && <span>{home}</span>}
          {row.target_date && <span>Target {row.target_date}</span>}
          {project && (
            <span>
              {project.completed_task_count ?? 0}/{project.task_count ?? 0}{" "}
              tasks
            </span>
          )}
          {!!linked.length && (
            <span>
              {linked.length} {goal ? "projects" : "goals"}
            </span>
          )}
          {!!notes.length && <span>{notes.length} notes</span>}
        </div>
        {goal && goal.metric_target != null && (
          <div className="goal-metric">
            <span>
              {goal.metric_current ?? "—"} / {goal.metric_target}{" "}
              {goal.metric_unit}
            </span>
            {goal.progress !== null && (
              <progress
                aria-label={"Outcome progress: " + goal.name}
                max={1}
                value={goal.progress}
              />
            )}
          </div>
        )}
        {(row.description ||
          row.success_criteria ||
          linked.length > 0 ||
          notes.length > 0) && (
          <details className="record-details">
            <summary>Details & links</summary>
            {row.description && <p>{row.description}</p>}
            {row.success_criteria && <p>Success: {row.success_criteria}</p>}
            {!!linked.length && (
              <div className="organization-links">
                {linked.map((link) => (
                  <button
                    key={link.id}
                    className="text-button"
                    onClick={() => edit(goal ? "project" : "goal", link)}
                  >
                    {link.name}
                    <ArrowRight size={13} />
                  </button>
                ))}
              </div>
            )}
            {!!notes.length && (
              <div className="organization-links">
                {notes.map((note) => (
                  <button
                    key={note.id}
                    className="text-button"
                    onClick={() => onNote(note.id)}
                  >
                    {note.title}
                  </button>
                ))}
              </div>
            )}
          </details>
        )}
        <div className="organization-row-actions">
          <button className="text-button" onClick={() => edit(tab, row)}>
            Edit
          </button>
          {project && (
            <button className="text-button" onClick={() => onProject(project)}>
              Tasks
              <ArrowRight size={14} />
            </button>
          )}
          {tab === "project" && layout === "board" && (
            <select
              aria-label={"Status for " + row.name}
              value={row.status}
              disabled={busy}
              onChange={(e) =>
                void mutate(
                  "project.update",
                  {
                    project_id: row.id,
                    expected_revision: row.revision,
                    status: e.target.value,
                  },
                  "Project moved",
                )
              }
            >
              {["planned", "active", "on_hold", "completed", "cancelled"].map(
                (v) => (
                  <option key={v} value={v}>
                    {v.replaceAll("_", " ")}
                  </option>
                ),
              )}
            </select>
          )}
          <button
            className="text-button archive-action"
            disabled={busy}
            onClick={() =>
              void mutate(
                tab + ".update",
                {
                  [tab + "_id"]: row.id,
                  expected_revision: row.revision,
                  archived: !row.archived,
                },
                row.archived ? "Restored" : "Archived",
              )
            }
          >
            {row.archived ? "Restore" : "Archive"}
          </button>
        </div>
      </article>
    );
  }
  return (
    <section className="productivity-page" aria-label="Goals and projects">
      <div
        className="organization-tabs"
        role="tablist"
        aria-label="Organization"
      >
        {(Object.keys(labels) as Kind[]).map((kind) => (
          <button
            key={kind}
            role="tab"
            aria-selected={tab === kind}
            onClick={() => onTab(kind)}
          >
            {labels[kind]}
            <span>{collections[kind].filter((r) => !r.archived).length}</span>
          </button>
        ))}
      </div>
      <div className="workspace-actions">
        {tab === "project" && (
          <LayoutSwitch
            value={layout}
            onChange={onLayout}
            label="Project view"
          />
        )}
        <label>
          Space
          <select
            aria-label="Organization space"
            value={space}
            onChange={(e) => onSpace(e.target.value)}
          >
            <option value="">All spaces</option>
            {organization.spaces.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name}
              </option>
            ))}
          </select>
        </label>
        <label className="inline-check">
          <input
            type="checkbox"
            checked={archived}
            onChange={(e) => onArchived(e.target.checked)}
          />
          Archived
        </label>
        <button className="primary compact" onClick={() => edit(tab)}>
          <Plus size={16} />
          New {tab === "actor" ? "assignee" : tab}
        </button>
      </div>
      {tab === "project" && layout === "timeline" ? (
        <Timeline
          label="Project timeline"
          items={visible.map((r) => ({
            id: r.id,
            title: r.name,
            start: r.start_date ?? null,
            end: r.target_date ?? null,
            status: r.status,
            range: true,
            subtitle: r.status?.replaceAll("_", " "),
          }))}
          start={timelineDate || today}
          span={timelineSpan}
          onStart={onTimelineDate}
          onSpan={onTimelineSpan}
          today={today}
          onOpen={(id) =>
            edit(
              "project",
              visible.find((r) => r.id === id),
            )
          }
        />
      ) : tab === "project" && layout === "board" ? (
        <div className="board-scroll" aria-label="Project board" tabIndex={0}>
          <div className="task-board">
            {["planned", "active", "on_hold", "completed", "cancelled"].map(
              (status) => (
                <section
                  className="board-column"
                  key={status}
                  onDragOver={(e) => {
                    if (!busy) e.preventDefault();
                  }}
                  onDrop={(e) => {
                    e.preventDefault();
                    const row = visible.find(
                      (r) =>
                        r.id ===
                        e.dataTransfer.getData("application/eridani-project"),
                    );
                    if (row && !busy && row.status !== status)
                      void mutate(
                        "project.update",
                        {
                          project_id: row.id,
                          expected_revision: row.revision,
                          status,
                        },
                        "Project moved",
                      );
                  }}
                >
                  <h3>
                    {status.replaceAll("_", " ")}
                    <span>
                      {visible.filter((r) => r.status === status).length}
                    </span>
                  </h3>
                  {visible.filter((r) => r.status === status).map(card)}
                  {!visible.some((r) => r.status === status) && (
                    <p className="board-empty">No projects</p>
                  )}
                </section>
              ),
            )}
          </div>
        </div>
      ) : (
        <div className="organization-list">{visible.map(card)}</div>
      )}
      {!visible.length && (
        <p className="compact-empty">
          No {labels[tab].toLowerCase()} match this view.
        </p>
      )}
      {editor && (
        <OrganizationEditor
          key={editor.kind + ":" + (editor.row?.id ?? "new")}
          {...editor}
          organization={organization}
          busy={busy}
          mutate={mutate}
          onClose={close}
        />
      )}
    </section>
  );
}

function OrganizationEditor({
  kind,
  row,
  organization,
  busy,
  mutate,
  onClose,
}: {
  kind: Kind;
  row?: RecordRow;
  organization: Organization;
  busy: boolean;
  mutate: Mutate;
  onClose: () => void;
}) {
  useDialogFocus();
  const isGoal = kind === "goal",
    isProject = kind === "project";
  const initial: Record<string, unknown> = { name: row?.name ?? "" };
  const shape: Record<string, z.ZodType> = {
    name: z
      .string()
      .min(1)
      .max(kind === "actor" ? 100 : 200),
  };
  const add = (key: string, value: unknown, schema: z.ZodType) => {
    initial[key] = value;
    shape[key] = schema;
  };
  if (kind !== "actor")
    add("description", row?.description ?? "", z.string().max(10000));
  if (isGoal || isProject) {
    add(
      "space_id",
      row?.space_id ?? null,
      nullableId(organization.spaces.map((s) => s.id)),
    );
    add(
      "area_id",
      row?.area_id ?? null,
      nullableId(organization.areas.map((a) => a.id)),
    );
    add("success_criteria", row?.success_criteria ?? "", z.string().max(10000));
    add(
      "status",
      row?.status ?? "planned",
      choice(
        isGoal
          ? ["planned", "active", "on_hold", "achieved", "abandoned"]
          : ["planned", "active", "on_hold", "completed", "cancelled"],
      ),
    );
    add("target_date", row?.target_date ?? null, z.string().date().nullable());
  }
  if (kind === "area")
    add(
      "space_id",
      row?.space_id ?? null,
      nullableId(organization.spaces.map((s) => s.id)),
    );
  if (isProject) {
    add("start_date", row?.start_date ?? null, z.string().date().nullable());
    add(
      "goal_ids",
      row?.goal_ids ?? [],
      z.array(choice(organization.goals.map((g) => g.id))).max(100),
    );
  }
  if (isGoal) {
    add(
      "project_ids",
      row?.project_ids ?? [],
      z.array(choice(organization.projects.map((p) => p.id))).max(100),
    );
    add(
      "parent_goal_id",
      row?.parent_goal_id ?? null,
      nullableId(
        organization.goals.filter((g) => g.id !== row?.id).map((g) => g.id),
      ),
    );
    add(
      "horizon",
      row?.horizon ?? "unspecified",
      choice(["unspecified", "short_term", "long_term"]),
    );
    add("metric_unit", row?.metric_unit ?? "", z.string().max(80));
    add("metric_baseline", row?.metric_baseline ?? 0, z.number().finite());
    add(
      "metric_current",
      row?.metric_current ?? null,
      z.number().finite().nullable(),
    );
    add(
      "metric_target",
      row?.metric_target ?? null,
      z.number().finite().nullable(),
    );
  }
  if (kind === "actor" && !row)
    add("kind", "person", choice(["person", "agent"]));
  const [values, setValues] = useState(initial),
    [localError, setLocalError] = useState("");
  const set = (key: string, value: unknown) =>
    setValues((v) => ({ ...v, [key]: value }));
  const schema = z.object(shape);
  async function save() {
    const parsed = schema.safeParse(values);
    if (!parsed.success) {
      setLocalError(
        parsed.error.issues
          .map((i) => i.path.join(".") + ": " + i.message)
          .join("; "),
      );
      return;
    }
    const args = row
      ? Object.fromEntries(
          Object.entries(values).filter(
            ([k, v]) => JSON.stringify(v) !== JSON.stringify(initial[k]),
          ),
        )
      : values;
    if (row && !Object.keys(args).length) {
      onClose();
      return row;
    }
    const result = await mutate(
      kind + (row ? ".update" : ".create"),
      row
        ? { ...args, [kind + "_id"]: row.id, expected_revision: row.revision }
        : args,
      "Saved",
    );
    if (result) onClose();
    else setLocalError("Could not save. Your changes are still here.");
    return result;
  }
  useEditor({
    kind,
    record_id: row?.id,
    dirty: JSON.stringify(values) !== JSON.stringify(initial),
    busy,
    schema,
    values,
    patch: (patch) => setValues((v) => ({ ...v, ...patch })),
    save,
    close: onClose,
  });
  const input = (key: string, label: string, type = "text") => (
    <label key={key}>
      {label}
      <input
        aria-label={label}
        type={type}
        value={String(values[key] ?? "")}
        step={type === "number" ? "any" : undefined}
        required={key === "name"}
        maxLength={key === "name" ? 200 : 10000}
        onChange={(e) =>
          set(
            key,
            type === "number"
              ? e.target.value
                ? Number(e.target.value)
                : null
              : type === "date"
                ? e.target.value || null
                : e.target.value,
          )
        }
      />
    </label>
  );
  const select = (
    key: string,
    label: string,
    options: { id: string; name: string }[],
    nullable = false,
  ) => (
    <label>
      {label}
      <select
        aria-label={label}
        value={String(values[key] ?? "")}
        onChange={(e) =>
          set(key, nullable ? e.target.value || null : e.target.value)
        }
      >
        {nullable && <option value="">None</option>}
        {options.map((v) => (
          <option key={v.id} value={v.id}>
            {v.name}
          </option>
        ))}
      </select>
    </label>
  );
  const enums = (values: string[]) =>
    values.map((v) => ({ id: v, name: v.replaceAll("_", " ") }));
  return (
    <div className="modal-backdrop">
      <form
        className="dialog organization-editor"
        role="dialog"
        aria-modal="true"
        aria-labelledby="organization-title"
        onSubmit={(e) => {
          e.preventDefault();
          void save();
        }}
      >
        <div className="dialog-heading">
          <h2 id="organization-title">
            {row ? "Edit" : "New"} {kind === "actor" ? "assignee" : kind}
          </h2>
          <button
            type="button"
            className="icon-button"
            aria-label="Close organization editor"
            onClick={onClose}
          >
            <X size={20} />
          </button>
        </div>
        {localError && (
          <p role="alert" className="error-banner">
            {localError}
          </p>
        )}
        {input("name", "Name")}
        {kind !== "actor" && (
          <label>
            Description
            <textarea
              rows={3}
              value={String(values.description)}
              maxLength={10000}
              onChange={(e) => set("description", e.target.value)}
            />
          </label>
        )}
        {(isGoal || isProject) && (
          <>
            <HomeFields
              organization={organization}
              value={values as Home}
              onChange={(h) => setValues((v) => ({ ...v, ...h }))}
            />
            <label>
              Success criteria
              <textarea
                rows={2}
                maxLength={10000}
                value={String(values.success_criteria)}
                onChange={(e) => set("success_criteria", e.target.value)}
              />
            </label>
            <div className="form-grid">
              {select(
                "status",
                "Status",
                enums(
                  isGoal
                    ? ["planned", "active", "on_hold", "achieved", "abandoned"]
                    : [
                        "planned",
                        "active",
                        "on_hold",
                        "completed",
                        "cancelled",
                      ],
                ),
              )}
              {input("target_date", "Target date", "date")}
              {isProject && input("start_date", "Start date", "date")}
              {isGoal && (
                <>
                  {select(
                    "horizon",
                    "Horizon",
                    enums(["unspecified", "short_term", "long_term"]),
                  )}
                  {select(
                    "parent_goal_id",
                    "Parent goal",
                    organization.goals.filter((g) => g.id !== row?.id),
                    true,
                  )}
                </>
              )}
            </div>
          </>
        )}
        {isGoal && (
          <details
            className="metric-editor"
            open={values.metric_target != null || undefined}
          >
            <summary>Outcome metric</summary>
            <div className="form-grid">
              {input("metric_unit", "Unit")}
              {input("metric_baseline", "Starting value", "number")}
              {input("metric_current", "Current value", "number")}
              {input("metric_target", "Target value", "number")}
            </div>
          </details>
        )}
        {(isGoal || isProject) && (
          <LinkPicker
            label={isGoal ? "Supporting projects" : "Supported goals"}
            items={isGoal ? organization.projects : organization.goals}
            selected={values[isGoal ? "project_ids" : "goal_ids"] as string[]}
            onChange={(ids) => set(isGoal ? "project_ids" : "goal_ids", ids)}
          />
        )}
        {kind === "area" &&
          select("space_id", "Space", organization.spaces, true)}
        {kind === "actor" &&
          !row &&
          select("kind", "Type", enums(["person", "agent"]))}
        <div className="dialog-actions">
          <button type="button" className="secondary" onClick={onClose}>
            Cancel
          </button>
          <button className="primary" disabled={busy}>
            Save
          </button>
        </div>
      </form>
    </div>
  );
}
