import { RecordTools } from "./record-links";
import { Tabs, humanLabel, priorityLabels, SchedulingHelp } from "./ux";
import { useEffect, useRef, useState } from "react";
import { Check, X, ChevronRight, ListTodo, Bell, FileText } from "lucide-react";
import { z } from "zod";
import { api } from "./api";
import { useDialogFocus, timeLabel, recurrenceLabel } from "./components";
import { useEditor, choice, nullableId, tagsField } from "./editor-control";
import { LinearTask } from "./Linear";
import type { Task, Schedule } from "./types";
import type { Organization } from "./productivity";
import type { NoteRecord } from "./Notes";
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
export function TaskDetails(p: Props) {
  useDialogFocus();
  const [task, setTask] = useState<Task | null>(null),
    [notes, setNotes] = useState<NoteRecord[]>([]);
  const current = useRef<Task | null>(null),
    flight = useRef<Promise<unknown> | null>(null);
  const [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [tab, setTab] = useState("overview");
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
      } else if (!flight.current) p.onClose();
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
    dirty: !!pending.current,
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
  const text = (field: keyof Task, label: string, type = "text") => {
    const value = task?.[field];
    const shown = Array.isArray(value) ? value.join(", ") : (value ?? "");
    return (
      <div
        className={
          "inline-field " + (field === "title" ? "task-card-title" : "")
        }
      >
        {field !== "title" && <span className="field-label">{label}</span>}
        {editing === field ? (
          type === "textarea" ? (
            <textarea
              aria-label={label}
              autoFocus
              rows={7}
              value={draft}
              onChange={(e) => {
                failedPatch.current = null;
                setDraft(e.target.value);
                pending.current = { field, value: e.target.value };
              }}
              onBlur={() => { if (!failedPatch.current) void finish().catch(() => {}); }}
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
              autoFocus
              type={type}
              value={draft}
              min={type === "number" ? 1 : undefined}
              onChange={(e) => {
                failedPatch.current = null;
                setDraft(e.target.value);
                pending.current = { field, value: e.target.value };
              }}
              onBlur={() => { if (!failedPatch.current) void finish().catch(() => {}); }}
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
            className={"inline-value " + (!shown ? "empty" : "")}
            aria-label={"Change " + label}
            disabled={busy || p.canEdit === false}
            onClick={() =>
              void leave(() => {
                setEditing(field);
                setDraft(String(shown));
                pending.current = { field, value: String(shown) };
              })
            }
          >
            {String(shown) || "Add " + label.toLowerCase()}
          </button>
        )}
        {editing === field && <small className="inline-help">{type === "textarea" ? "Ctrl/⌘ + Enter" : "Enter"} or leave the field to save · Escape cancels this field.</small>}
        {savedFields.includes(field) && !editing && !busy && !error && <small className="field-saved">Saved</small>}
      </div>
    );
  };
  const select = (
    field: keyof Task,
    label: string,
    options: { id: string; name: string }[],
    disabled = false,
  ) => (
    <label className="inline-field">
      <span className="field-label">{label}</span>
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
            {o.name}
          </option>
        ))}
      </select>
    </label>
  );
  const project = p.organization.projects.find(
    (x) => x.id === task?.project_id,
  );
  const goals = p.organization.goals.filter((g) =>
    project?.goal_ids?.includes(g.id),
  );
  const children = p.tasks.filter(
    (t) => t.parent_task_id === p.id && !t.archived,
  );
  const parent = p.tasks.find((t) => t.id === task?.parent_task_id);
  const reminders = p.schedules.filter((s) => s.task_id === p.id);
  return (
    <div
      className="modal-backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget) close();
      }}
    >
      <section
        ref={card}
        className="dialog task-detail-card"
        role="dialog"
        aria-modal="true"
        aria-label="Task details"
      >
        <header className="task-card-toolbar">
          <span>
            <ListTodo size={16} />{" "}
            {task?.is_template ? "Routine template" : "Task"}
          </span>
          <span role="status" className="footnote">
            {busy
              ? "Saving…"
              : error
                ? "Change needs attention"
                : p.canEdit === false
                  ? "View only"
                  : savedFields.length ? "Saved" : "Click any field to edit"}
          </span>
          <button
            className="icon-button"
            aria-label="Close task details"
            disabled={busy}
            onClick={close}
          >
            <X size={20} />
          </button>
        </header>
        <RecordTools kind="task" id={p.id}/>
        {error && (
          <div role="alert" className="error-banner">
            {error}{" "}
            <button
              className="text-button"
              onClick={() => void reload(true).catch((e) => setError(e.message))}
            >
              Compare current values
            </button>
          </div>
        )}
        {comparison && <section className="conflict-comparison" aria-label="Compare task changes"><h3>Your draft is still here</h3><p>Compare the saved version before choosing which values to keep.</p>
          {Object.entries(failedPatch.current ?? {}).map(([field, value]) => <div key={field}><strong>{humanLabel(field)}</strong><p>Saved: {String(comparison[field as keyof Task] ?? "Empty")}</p><p>Your change: {String(value ?? "Empty")}</p></div>)}
          <button className="secondary" disabled={busy} onClick={() => {current.current = comparison; setTask(comparison); pending.current = null; failedPatch.current = null; setEditing(null); setComparison(null); setError("");}}>Use saved values</button>
          <button className="primary" disabled={busy} onClick={() => {const patch = failedPatch.current; current.current = comparison; setTask(comparison); if (patch) void save(patch).then(() => {pending.current = null; setEditing(null);}).catch(() => {});}}>Apply my change to this version</button>
        </section>}
        {!task ? (
          <p role="status">Loading task…</p>
        ) : (
          <div className="task-detail-grid">
            <main className="task-detail-main">
              {text("title", "Task")}
              <Tabs id="task-detail-tab" label="Task detail sections" className="task-card-tabs" panel="task-detail-panel" value={tab} onChange={id => leave(() => setTab(id))}
                items={[{id:"overview",label:"Overview"},{id:"notes",label:"Linked notes" + (notes.length ? " " + notes.length : "")},{id:"reminders",label:"Reminders" + (reminders.length ? " " + reminders.length : "")}]} />
              <div id="task-detail-panel" role="tabpanel" aria-labelledby={"task-detail-tab-" + tab}>
                {tab === "overview" && (
                  <>
                    {text("notes", "Description", "textarea")}
                    <div className="task-date-grid">
                      {text("planned_date", "Planned date", "date")}
                      {text("due_date", "Due date", "date")}
                      {task.due_date && text("due_time", "Due time", "time")}
                      {task.due_time && text("due_timezone", "Time zone")}
                      {text("estimate_minutes", "Estimate (minutes)", "number")}
                    </div>
                    <SchedulingHelp />
                    {reminders.length > 0 && <p className="footnote">This task has {reminders.length} reminder{reminders.length === 1 ? "" : "s"}. Its planned day, deadline and reminders all refer to this same task.</p>}
                    {parent && (
                      <section>
                        <h3>Parent task</h3>
                        <button
                          className="related-record"
                          onClick={() => void leave(() => p.onTask(parent))}
                        >
                          {parent.title}
                          <ChevronRight size={15} />
                        </button>
                      </section>
                    )}
                    {!!children.length && (
                      <section>
                        <h3>Subtasks</h3>
                        {children.map((t) => (
                          <button
                            key={t.id}
                            className="related-record"
                            onClick={() => void leave(() => p.onTask(t))}
                          >
                            {t.title}
                            <span>{t.status.replaceAll("_", " ")}</span>
                          </button>
                        ))}
                      </section>
                    )}
                    <LinearTask
                      task={task}
                      mutate={p.mutate}
                      onChanged={() =>
                        void reload().catch((e) => setError(e.message))
                      }
                    />
                    {!task.is_template && (
                      <button
                        className="secondary compact"
                        disabled={busy || p.canEdit === false}
                        onClick={() =>
                          void leave(() => p.onBlock(current.current!))
                        }
                      >
                        Reserve time for this task
                      </button>
                    )}
                    {!task.is_template && (
                      <button
                        className="secondary compact"
                        disabled={busy || p.canEdit === false}
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
                        <Check size={16} />
                        {task.status === "completed"
                          ? "Reopen task"
                          : "Complete task"}
                      </button>
                    )}
                  </>
                )}
                {tab === "notes" && (
                  <section>
                    <h3>
                      <FileText size={16} /> Linked notes
                    </h3>
                    <button
                      className="secondary compact"
                      disabled={busy || p.canEdit === false}
                      onClick={() =>
                        void leave(() => p.onNewNote(current.current!))
                      }
                    >
                      New linked note
                    </button>
                    {notesError && <p role="alert">{notesError}</p>}
                    {notes.length ? (
                      notes.map((n) => (
                        <button
                          className="related-record"
                          key={n.id}
                          onClick={() => void leave(() => p.onNote(n.id))}
                        >
                          {n.title}
                          <ChevronRight size={15} />
                        </button>
                      ))
                    ) : (
                      <p className="footnote">No linked notes yet.</p>
                    )}
                  </section>
                )}
                {tab === "reminders" && (
                  <section>
                    <h3>
                      <Bell size={16} /> Reminders
                    </h3>
                    <button
                      className="secondary compact"
                      disabled={busy || p.canEdit === false}
                      onClick={() =>
                        void leave(() => p.onReminder(null, current.current!))
                      }
                    >
                      Add reminder
                    </button>
                    {reminders.length ? (
                      reminders.map((s) => (
                        <div className="reminder-detail" key={s.id}>
                          <button
                            className="text-button"
                            onClick={() =>
                              void leave(() =>
                                p.onReminder(s, current.current!),
                              )
                            }
                          >
                            {s.title}
                          </button>
                          <p>
                            {s.status} · {recurrenceLabel(s.recurrence)}
                            {s.next_run_at
                              ? " · " + timeLabel(s.next_run_at, s.timezone)
                              : ""}
                          </p>
                        </div>
                      ))
                    ) : (
                      <p className="footnote">No reminders for this task.</p>
                    )}
                  </section>
                )}
              </div>
            </main>
            <aside
              className="task-detail-properties"
              aria-label="Task properties"
            >
              {select(
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
              )}
              {select(
                "priority",
                "Priority",
                priorityLabels.map((name, i) => ({
                  id: String(i),
                  name,
                })),
              )}
              {select("assignee_id", "Assignee", p.organization.actors)}
              {select("project_id", "Project", [
                { id: "", name: "No project" },
                ...p.organization.projects.filter(
                  (x) => !x.archived || x.id === task.project_id,
                ),
              ])}
              {select(
                "space_id",
                "Space",
                [{ id: "", name: "No space" }, ...p.organization.spaces],
                !!task.project_id,
              )}
              {select(
                "area_id",
                "Area",
                [
                  { id: "", name: "No area" },
                  ...p.organization.areas.filter(
                    (x) => x.space_id === task.space_id,
                  ),
                ],
                !!task.project_id,
              )}
              {select("parent_task_id", "Parent task", [
                { id: "", name: "No parent" },
                ...p.tasks
                  .filter((t) => t.id !== p.id && !t.archived)
                  .map((t) => ({ id: t.id, name: t.title })),
              ])}
              <div className="inline-field">
                <span className="field-label">Supporting goals</span>
                {goals.length ? (
                  goals.map((g) => (
                    <span className="attribution-chip" key={g.id}>
                      {g.name}
                    </span>
                  ))
                ) : (
                  <span className="footnote">Linked through a project</span>
                )}
              </div>
              {text("work_type", "Work type")}
              {text("tags", "Tags")}
              <button
                className="text-button"
                disabled={busy || p.canEdit === false}
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
                <a href={task.external.url} target="_blank" rel="noreferrer">
                  Open {task.external.identifier || "in Linear"}
                </a>
              )}
            </aside>
          </div>
        )}
      </section>
    </div>
  );
}
