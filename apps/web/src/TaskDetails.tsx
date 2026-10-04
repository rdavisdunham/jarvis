import {QuickListDetail} from "./QuickLists";
import { SourceDetails, SourceNotes, useInlineGuard } from "./SourceDetails";
import { RecordTools } from "./record-links";
import { humanLabel, SchedulingHelp, Dialog } from "./ux";
import { lazy, Suspense, useEffect, useRef, useState } from "react";
import { Check, X, ChevronRight, ListTodo, Bell, FileText, CornerLeftUp, ExternalLink, Plus } from "lucide-react";
import { Prop, PriorityControl } from "./RecordCard";
import { ReviewProps } from "./Reviews";
import type { CustomRecord } from "./structure-types";
import "./details.css";
import { z } from "zod";
import { api } from "./api";
import { timeLabel, recurrenceLabel } from "./components";
import { useEditor, choice, nullableId, tagsField } from "./editor-control";
import type { Task, Schedule } from "./types";
import type { Organization } from "./productivity";
import type { NoteRecord } from "./Notes";
const LinearTask = lazy(() => import("./Linear").then((m) => ({ default: m.LinearTask })));
type Props = {
  id: string;
  canEdit?: boolean;
  organization: Organization;
  tasks: Task[];
  schedules: Schedule[];
  zone: string;
  mutate: (tool: string, args: unknown, message: string) => Promise<unknown>;
  onClose: () => void;
  onTask: (task: Task) => void;
  onNote: (id: string) => void;
  onNewNote: (task: Task) => void;
  onReminder: (schedule: Schedule | null, task: Task) => void;
  onBlock: (task: Task) => void;
};
export function TaskDetails(p:Props){
 const [quick,setQuick]=useState<boolean|null>(p.tasks.find(t=>t.id===p.id)?.is_quick_list??null);
 useEffect(()=>{let active=true;setQuick(p.tasks.find(t=>t.id===p.id)?.is_quick_list??null);void api<Task>("/tasks/"+p.id).then(t=>{if(active)setQuick(!!t.is_quick_list);}).catch(()=>{if(active)setQuick(false);});return()=>{active=false;};},[p.id]);
 if(quick)return <QuickListDetail refresh={p.tasks.filter(t=>t.id===p.id||t.parent_task_id===p.id).reduce((sum,t)=>sum+t.revision,0)} id={p.id} canEdit={p.canEdit} today={new Intl.DateTimeFormat("en-CA",{timeZone:p.zone}).format(new Date())} onClose={p.onClose} onChanged={()=>window.dispatchEvent(new Event("eri-quick-changed"))}/>;
 return <StandardTaskDetails {...p}/>;
}
function StandardTaskDetails(p: Props) {
  const [customHome,setCustomHome]=useState<CustomRecord|null>(null);
  useEffect(()=>{void api<CustomRecord>("/structure/by-core/task/"+p.id).then(setCustomHome);},[p.id]);
  const annotation=useInlineGuard();
  const [task, setTask] = useState<Task | null>(null),
    [notes, setNotes] = useState<NoteRecord[]>([]);
  const current = useRef<Task | null>(null),
    flight = useRef<Promise<unknown> | null>(null);
  const [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  const [editing, setEditing] = useState<string | null>(null),
    [draft, setDraft] = useState("");
  const [notesError, setNotesError] = useState("");
  const [savedFields, setSavedFields] = useState<string[]>([]);
  const [comparison, setComparison] = useState<Task | null>(null);
  const failedPatch = useRef<Record<string, unknown> | null>(null);
  const pending = useRef<{ field: string; value: string } | null>(null);
  const card = useRef<HTMLElement>(null);
  useEffect(() => {
    card.current
      ?.querySelector<HTMLButtonElement>("[aria-label='Close task details']")
      ?.focus();
  }, []);
  async function reload(compare = false) {
    const value = await api<Task>("/tasks/" + p.id);
    if (compare) setComparison(value);
    else if (!pending.current && !flight.current) {current.current = value;setTask(value);}
  }
  useEffect(() => {
    let live = true;
    void api<Task>("/tasks/" + p.id)
      .then((t) => {
        if (live) {
          current.current = t;
          setTask(t);
        }
      })
      .catch((e) => {
        if (live) setError(e.message);
      });
    void api<{ items: NoteRecord[] }>("/notes?task_id=" + p.id)
      .then((r) => {
        if (live) setNotes(r.items);
      })
      .catch(() => {
        if (live) setNotesError("Linked notes could not be loaded.");
      });
    return () => {
      live = false;
    };
  }, [p.id]);
  useEffect(() => {
    const fresh = p.tasks.find((t) => t.id === p.id);
    if (
      fresh &&
      current.current &&
      fresh.revision > current.current.revision &&
      !pending.current &&
      !flight.current
    ) {
      current.current = fresh;
      setTask(fresh);
    }
  }, [p.tasks, p.id]);
  async function save(changes: Record<string, unknown>) {
    if (p.canEdit === false)
      throw new Error("You have view access to this workspace.");
    if (flight.current) await flight.current;
    const t = current.current;
    if (!t) throw new Error("Task details are still loading.");
    const patch = { ...changes };
    for (const key of [
      "project_id",
      "space_id",
      "area_id",
      "parent_task_id",
      "planned_date",
      "due_date",
      "due_time",
      "due_timezone",
    ])
      if (patch[key] === "") patch[key] = null;
    if ("due_date" in patch && !patch.due_date) {
      patch.due_time = null;
      patch.due_timezone = null;
    }
    if (patch.due_time && !patch.due_timezone && !t.due_timezone)
      patch.due_timezone = p.zone;
    const changed = Object.fromEntries(
      Object.entries(patch).filter(
        ([k, v]) =>
          JSON.stringify(t[k as keyof Task] ?? null) !== JSON.stringify(v),
      ),
    );
    if (!Object.keys(changed).length) return { unchanged: true };
    setBusy(true);
    setError("");
    const operation = (async () => {
      const result = await p.mutate(
        "task.update",
        { task_id: t.id, expected_revision: t.revision, ...changed },
        "Task updated",
      );
      if (!result)
        throw new Error(
          "This change was not saved. Review the error, or reload the current values before trying again.",
        );
      const updated = result as Task;
      current.current = updated;
      setTask(updated);
      setSavedFields(Object.keys(changed));
      failedPatch.current = null;
      setComparison(null);
      return result;
    })();
    flight.current = operation;
    try {
      return await operation;
    } catch (e) {
      failedPatch.current = changed;
      setError((e as Error).message);
      throw e;
    } finally {
      flight.current = null;
      setBusy(false);
    }
  }
  async function finish() {
    await annotation.flush();
    if (flight.current) await flight.current;
    const item = pending.current;
    if (!item) return;
    let value: unknown = item.value;
    if (item.field === "tags")
      value = item.value
        .split(",")
        .map((t) => t.trim())
        .filter(Boolean);
    if (item.field === "estimate_minutes")
      value = item.value === "" ? null : Number(item.value);
    await save({ [item.field]: value });
    if (pending.current === item) {
      pending.current = null;
      setEditing(null);
    }
  }
  async function leave(next: () => void) {
    try {
      await finish();
      next();
    } catch {
      /* retain the failed field */
    }
  }
  const close = () => {
    void leave(p.onClose);
  };
  useEffect(() => {
    const escape = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.preventDefault();
      e.stopImmediatePropagation();
      if (pending.current) {
        pending.current = null;
        setEditing(null);
        setError("");
      } else if (!flight.current) close();
    };
    document.addEventListener("keydown", escape, true);
    return () => document.removeEventListener("keydown", escape, true);
  }, [p.onClose]);
  const schema = z.object({
    title: z.string().min(1).max(500),
    notes: z.string().max(10000),
    status: choice([
      "backlog",
      "open",
      "in_progress",
      "waiting",
      "deferred",
      "completed",
      "cancelled",
    ]),
    deadline_alert:z.enum(["default","on","off"]),
    alert_urgent:z.boolean(),
    priority: z.number().int().min(0).max(3),
    project_id: nullableId(p.organization.projects.map((x) => x.id)),
    space_id: nullableId(p.organization.spaces.map((x) => x.id)),
    area_id: nullableId(p.organization.areas.map((x) => x.id)),
    parent_task_id: nullableId(
      p.tasks.filter((x) => x.id !== p.id).map((x) => x.id),
    ),
    assignee_id: choice(p.organization.actors.map((x) => x.id)),
    work_type: z.string().max(80),
    tags: tagsField,
    planned_date: z.string().date().or(z.literal("")).nullable(),
    due_date: z.string().date().or(z.literal("")).nullable(),
    due_time: z
      .string()
      .regex(/^$|^\d{2}:\d{2}(:\d{2})?$/)
      .nullable(),
    due_timezone: z.string().max(100).nullable(),
    estimate_minutes: z.number().int().min(1).max(100000).nullable(),
  });
  useEditor({
    kind: "task",
    record_id: p.id,
    mode: "detail",
    auto_save: true,
    dirty: !!pending.current || annotation.dirty,
    // Nested annotation saves are awaited by finish; do not block navigation before it runs.
    busy: busy || !task,
    schema,
    values: task
      ? Object.fromEntries(
          Object.keys(schema.shape).map((k) => [
            k,
            task[k as keyof Task] ?? null,
          ]),
        )
      : {},
    beforeLeave: finish,
    discard: () => {
      pending.current = null;
      setEditing(null);
      p.onClose();
    },
    patch: async (v) => {
      await finish();
      return save(v);
    },
    close,
  });
  const shownValue = (field: keyof Task) => {
    const value = task?.[field];
    return Array.isArray(value) ? value.join(", ") : String(value ?? "");
  };
  // Friendly read-only presentation; editing always uses the raw value.
  const pretty = (field: keyof Task, raw: string) => {
    if (!raw) return "";
    if (field === "planned_date" || field === "due_date") {
      const [y, m, d] = raw.split("-").map(Number);
      if (y && m && d)
        return new Date(y, m - 1, d).toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric", year: y === new Date().getFullYear() ? undefined : "numeric" });
    }
    if (field === "due_time") {
      const [h, m] = raw.split(":").map(Number);
      if (!Number.isNaN(h) && !Number.isNaN(m))
        return new Date(2000, 0, 1, h, m).toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
    }
    if (field === "estimate_minutes") return raw + " min";
    return raw;
  };
  const start = (field: keyof Task) =>
    void leave(() => {
      const shown = shownValue(field);
      setEditing(field);
      setDraft(shown);
      pending.current = { field, value: shown };
    });
  const text = (
    field: keyof Task,
    label: string,
    type = "text",
    variant: "title" | "body" | "prop" = "prop",
    empty = "Empty",
  ) => {
    const shown = shownValue(field);
    const change = (value: string) => {
      failedPatch.current = null;
      setDraft(value);
      pending.current = { field, value };
    };
    const blur = () => {
      if (!failedPatch.current) void finish().catch(() => {});
    };
    const inputClass = variant === "title" ? "detail-title" : variant === "body" ? "detail-body" : undefined;
    const buttonClass = variant === "title" ? "detail-title-button" : variant === "body" ? "detail-body-button" : "prop-text";
    return (
      <>
        {editing === field ? (
          type === "textarea" ? (
            <textarea
              aria-label={label}
              className={inputClass}
              autoFocus
              rows={7}
              value={draft}
              onChange={(e) => change(e.target.value)}
              onBlur={blur}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
                  e.preventDefault();
                  void finish().catch(() => {});
                }
              }}
            />
          ) : (
            <input
              aria-label={label}
              className={inputClass}
              autoFocus
              type={type}
              value={draft}
              min={type === "number" ? 1 : undefined}
              onChange={(e) => change(e.target.value)}
              onBlur={blur}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  void finish().catch(() => {});
                }
              }}
            />
          )
        ) : (
          <button
            className={buttonClass + (!shown ? " empty" : "")}
            aria-label={"Change " + label}
            disabled={busy || p.canEdit === false}
            onClick={() => start(field)}
          >
            {pretty(field, shown) || empty}
          </button>
        )}
        {editing === field && (
          <small className="prop-help">
            {type === "textarea" ? "Press Ctrl+Enter or leave the field to save. Escape cancels." : "Press Enter or leave the field to save. Escape cancels."}
          </small>
        )}
        {savedFields.includes(field) && !editing && !busy && !error && variant === "prop" && (
          <small className="prop-saved">Saved</small>
        )}
      </>
    );
  };
  const select = (
    field: keyof Task,
    label: string,
    options: { id: string; name: string }[],
    disabled = false,
  ) => (
    <select
      aria-label={label}
      value={String(task?.[field] ?? "")}
      disabled={busy || disabled || p.canEdit === false}
      onChange={(e) => {
        const value = e.target.value;
        void leave(() => {
          void save({
            [field]: field === "priority" ? Number(value) : value,
          }).catch(() => {});
        });
      }}
    >
      {options.map((o) => (
        <option key={o.id} value={o.id}>
          {o.name === "owner" ? "Me" : o.name}
        </option>
      ))}
    </select>
  );
  const children = p.tasks.filter(
    (t) => t.parent_task_id === p.id && !t.archived,
  );
  const parent = p.tasks.find((t) => t.id === task?.parent_task_id);
  const reminders = p.schedules.filter((s) => s.task_id === p.id);
  const locked = busy || p.canEdit === false;
  const home = customHome?.home.map((h) => h.title).join(" / ");
  return (
    <Dialog onBackdrop={() => close()}
        dialogRef={card}
        className="detail-card task-detail-card"
        aria-label="Task details"
      >
        <header className="detail-head">
          <span className="detail-kind">
            <ListTodo size={16} />
            {task?.is_template ? "Routine template" : "Task"}
          </span>
          {home && <span className="chip chip-home detail-path" title={home}>{home}</span>}
          <span role="status" className={"detail-save-state" + (busy ? " busy" : error ? " attention" : "")}>
            {busy
              ? "Saving…"
              : error
                ? "Change needs attention"
                : p.canEdit === false
                  ? "View only"
                  : savedFields.length ? "Saved" : "Click any field to edit"}
          </span>
          <button
            className="btn-icon"
            aria-label="Close task details"
            disabled={busy}
            onClick={close}
          >
            <X size={20} />
          </button>
        </header>
        {error && (
          <div role="alert" className="detail-alert">
            <span>{error}</span>
            <button
              className="btn btn-sm"
              onClick={() => void reload(true).catch((e) => setError(e.message))}
            >
              Compare current values
            </button>
          </div>
        )}
        {comparison && <section className="detail-conflict" aria-label="Compare task changes"><h3>Your draft is still here</h3><p>Compare the saved version before choosing which values to keep.</p>
          <div className="conflict-table">
            <span className="conflict-head">Field</span><span className="conflict-head">Saved</span><span className="conflict-head">Your change</span>
            {Object.entries(failedPatch.current ?? {}).map(([field, value]) => <div key={field} style={{ display: "contents" }}><strong>{humanLabel(field)}</strong><span>{String(comparison[field as keyof Task] ?? "Empty")}</span><span className="conflict-mine">{String(value ?? "Empty")}</span></div>)}
          </div>
          <div className="detail-actions">
            <button className="btn" disabled={busy} onClick={() => {current.current = comparison; setTask(comparison); pending.current = null; failedPatch.current = null; setEditing(null); setComparison(null); setError("");}}>Use saved values</button>
            <button className="btn btn-primary" disabled={busy} onClick={() => {const patch = failedPatch.current; current.current = comparison; setTask(comparison); if (patch) void save(patch).then(() => {pending.current = null; setEditing(null);}).catch(() => {});}}>Apply my change to this version</button>
          </div>
        </section>}
        {!task ? (
          <p role="status" className="detail-main">Loading task…</p>
        ) : (
          <div className="detail-grid">
            <div className="detail-main">
              {text("title", "Task", "text", "title", "Add a title")}
              {text("notes", "Description", "textarea", "body", "Add details")}
              <SourceDetails source={task.source}/><SourceNotes kind="task" id={p.id} canEdit={p.canEdit!==false} onGuard={annotation.update}/>
              <RecordTools kind="task" id={p.id}/>
              {(parent || !!children.length) && (
                <section className="detail-section" aria-label="Related tasks">
                  <div className="detail-section-head"><h3>Related tasks</h3></div>
                  {parent && (
                    <button
                      className="detail-link-row"
                      onClick={() => void leave(() => p.onTask(parent))}
                    >
                      <CornerLeftUp size={15} />
                      <span className="detail-link-title">{parent.title}</span>
                      <span className="chip">Parent</span>
                    </button>
                  )}
                  {children.map((t) => (
                    <button
                      key={t.id}
                      className="detail-link-row"
                      onClick={() => void leave(() => p.onTask(t))}
                    >
                      <ChevronRight size={15} />
                      <span className="detail-link-title">{t.title}</span>
                      <span className={"chip" + (t.status === "completed" ? " chip-done" : "")}>{humanLabel(t.status)}</span>
                    </button>
                  ))}
                </section>
              )}
              <section className="detail-section" aria-label="Linked notes">
                <div className="detail-section-head">
                  <h3>Linked notes</h3>
                  {!!notes.length && <span className="detail-count">{notes.length}</span>}
                  <button
                    className="btn btn-ghost btn-sm"
                    disabled={locked}
                    onClick={() =>
                      void leave(() => p.onNewNote(current.current!))
                    }
                  >
                    <Plus size={15} />
                    New linked note
                  </button>
                </div>
                {notesError && <p role="alert" className="field-error">{notesError}</p>}
                {notes.length ? (
                  notes.map((n) => (
                    <button
                      className="detail-link-row"
                      key={n.id}
                      onClick={() => void leave(() => p.onNote(n.id))}
                    >
                      <FileText size={15} />
                      <span className="detail-link-title">{n.title}</span>
                    </button>
                  ))
                ) : (
                  <p className="detail-empty">No linked notes yet.</p>
                )}
              </section>
              <section className="detail-section" aria-label="Schedule">
                <div className="detail-section-head">
                  <h3>Reminders and time</h3>
                  {!!reminders.length && <span className="detail-count">{reminders.length}</span>}
                  <button
                    className="btn btn-ghost btn-sm"
                    disabled={locked}
                    onClick={() =>
                      void leave(() => p.onReminder(null, current.current!))
                    }
                  >
                    <Plus size={15} />
                    Add reminder
                  </button>
                </div>
                {reminders.length ? (
                  reminders.map((s) => (
                    <div className="detail-link-row" key={s.id}>
                      <button
                        className="detail-link-open"
                        onClick={() =>
                          void leave(() =>
                            p.onReminder(s, current.current!),
                          )
                        }
                      >
                        <Bell size={15} />
                        <span className="detail-link-title">{s.title}</span>
                      </button>
                      <span className="detail-meta">
                        {s.next_run_at && <span className="chip">{timeLabel(s.next_run_at, s.timezone)}</span>}
                        <span className="chip">{recurrenceLabel(s.recurrence)}</span>
                        {s.status !== "active" && <span className="chip">{humanLabel(s.status)}</span>}
                      </span>
                    </div>
                  ))
                ) : (
                  <p className="detail-empty">No reminders for this task.</p>
                )}
                {!task.is_template && (
                  <div className="detail-actions">
                    <button
                      className="btn btn-soft btn-sm"
                      disabled={locked}
                      onClick={() =>
                        void leave(() => p.onBlock(current.current!))
                      }
                    >
                      Reserve time for this task
                    </button>
                  </div>
                )}
                <SchedulingHelp />
              </section>
              <Suspense fallback={null}><LinearTask
                task={task}
                mutate={p.mutate}
                onChanged={() =>
                  void reload().catch((e) => setError(e.message))
                }
              /></Suspense>
            </div>
            <aside
              className="detail-props"
              aria-label="Task properties"
            >
              <div className="prop-list">
                {customHome && <Prop label="Main home">
                  <button className="prop-text" title="Open the organization card with custom fields" disabled={busy} onClick={()=>void leave(()=>{p.onClose();window.dispatchEvent(new CustomEvent("eri-open-custom-record",{detail:{id:customHome.id}}));})}>{home||"Unfiled"}</button>
                </Prop>}
                <Prop label="Status">{select(
                  "status",
                  "Status",
                  [
                    "backlog",
                    "open",
                    "in_progress",
                    "waiting",
                    "deferred",
                    "completed",
                    "cancelled",
                  ].map((id) => ({ id, name: humanLabel(id) })),
                  !!task.is_template,
                )}</Prop>
                <Prop label="Due" hint="When it must be finished. A deadline does not send an alert by itself.">
                  <div className="prop-inline">
                    <span className="prop-date">{text("due_date", "Due date", "date", "prop", "No deadline")}</span>
                    {task.due_date && <span className="prop-time">{text("due_time", "Due time", "time", "prop", "Any time")}</span>}
                    {task.due_time && <span className="prop-zone">{text("due_timezone", "Time zone")}</span>}
                  </div>
                </Prop>
                <Prop label="Planned day" hint="When you intend to work on it.">{text("planned_date", "Planned date", "date", "prop", "Not planned")}</Prop>
                <Prop label="Priority">
                  <PriorityControl value={task.priority ?? 0} disabled={locked} onChange={(value) => void leave(() => { void save({ priority: value }).catch(() => {}); })}/>
                </Prop>
                <Prop label="Estimate">{text("estimate_minutes", "Estimate (minutes)", "number")}</Prop>
                <Prop label="Assigned to">{select("assignee_id", "Assignee", p.organization.actors)}</Prop>
                <div className="prop-divider"/>
                <Prop label="Deadline alert">{select("deadline_alert","Deadline alert",[{id:"default",name:"Use my setting"},{id:"on",name:"On"},{id:"off",name:"Off"}])}</Prop>
                <Prop label="Urgent"><label className="prop-switch"><input type="checkbox" className="switch" role="switch" aria-label="Urgent alert" checked={!!task.alert_urgent} disabled={locked} onChange={e=>void save({alert_urgent:e.target.checked}).catch(()=>{})}/><small>Bypasses quiet hours</small></label></Prop>
                {customHome && <ReviewProps row={customHome} canEdit={p.canEdit !== false && !locked} onRecord={setCustomHome}/>}
              </div>
              <div className="detail-props-foot">
                {!task.is_template && (
                  <button
                    className="btn btn-soft btn-sm"
                    disabled={locked}
                    onClick={() =>
                      void leave(() => {
                        void save({
                          status:
                            task.status === "completed"
                              ? "open"
                              : "completed",
                        }).catch(() => {});
                      })
                    }
                  >
                    <Check size={15} />
                    {task.status === "completed"
                      ? "Reopen task"
                      : "Complete task"}
                  </button>
                )}
                <button
                  className="btn btn-ghost btn-sm"
                  disabled={locked}
                  onClick={() =>
                    void leave(() => {
                      void save({ archived: !task.archived })
                        .then(p.onClose)
                        .catch(() => {});
                    })
                  }
                >
                  {task.archived ? "Restore task" : "Archive task"}
                </button>
                {task.external?.url && /^https?:\/\//.test(task.external.url) && (
                  <a className="btn btn-ghost btn-sm" href={task.external.url} target="_blank" rel="noreferrer">
                    <ExternalLink size={14} />
                    Open {task.external.identifier || "in Linear"}
                  </a>
                )}
              </div>
            </aside>
          </div>
        )}
      </Dialog>
  );
}
