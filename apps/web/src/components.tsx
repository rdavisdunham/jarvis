import { useBodyLock, humanLabel, priorityLabels, SchedulingHelp } from "./ux";
import { z } from "zod";
import { useEditor, nullableId, choice, tagsField } from "./editor-control";
import { HomeFields } from "./Productivity";
import { emptyOrganization, type Organization } from "./productivity";
import { BudgetHolds } from "./BudgetHolds";
import { useEffect, useState, type ReactNode } from "react";
import {
  CalendarDays,
  Check,
  ChevronRight,
  Clock3,
  Download,
  MoreHorizontal,
  Plus,
  Shield,
  Trash2,
  X,
  Brain,
} from "lucide-react";
import type { Bootstrap, Task, Project, Schedule } from "./types";
export function useDialogFocus() {
  useBodyLock(true);
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    const trap = (event: KeyboardEvent) => {
      if (event.key !== "Tab" || event.defaultPrevented) return;
      const dialog = Array.from(document.querySelectorAll<HTMLElement>('[role="dialog"]')).filter(el => el.getClientRects().length).at(-1);
      const items = Array.from(
        dialog?.querySelectorAll<HTMLElement>(
          'button:not(:disabled), input:not(:disabled), textarea:not(:disabled), select:not(:disabled), a[href], summary, [tabindex="0"]',
        ) ?? [],
      ).filter(el => el.getClientRects().length && el.tabIndex >= 0);
      const first = items[0],
        last = items.at(-1);
      if (event.shiftKey && (document.activeElement === first || !dialog?.contains(document.activeElement))) {
        event.preventDefault();
        last?.focus();
      } else if (!event.shiftKey && (document.activeElement === last || !dialog?.contains(document.activeElement))) {
        event.preventDefault();
        first?.focus();
      }
    };
    document.addEventListener("keydown", trap);
    return () => {
      document.removeEventListener("keydown", trap);
      if (previous?.isConnected) previous.focus({preventScroll: true});
    };
  }, []);
}
export function dayInZone(zone: string) {
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: zone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(new Date());
}
export function timeLabel(value: string, zone: string) {
  return new Intl.DateTimeFormat(undefined, {
    timeZone: zone,
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(value));
}
export function recurrenceLabel(rule: string | null) {
  if (!rule) return "One time";
  if (rule.includes("BYDAY="))
    return (
      "Every " + rule.split("BYDAY=")[1].split(";")[0].replaceAll(",", ", ")
    );
  if (rule.includes("WEEKLY")) return "Weekly";
  if (rule.includes("MONTHLY")) return "Monthly";
  return "Daily";
}
function relativeDate(value: string, today: string) {
  if (value === today) return "Today";
  return new Date(value + "T12:00:00").toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}
export function TaskRow({
  task,
  today,
  busy,
  onToggle,
  onOpen,
  selected,
  onSelect,
}: {
  selected?: boolean;
  onSelect?: () => void;
  task: Task;
  today: string;
  busy: boolean;
  onToggle: () => void;
  onOpen: () => void;
}) {
  const done = task.status === "completed";
  return (
    <div className={"task-row " + (done ? "done" : "")}>
      {onSelect && (
        <input
          className="task-selection"
          type="checkbox"
          aria-label={"Select " + task.title}
          checked={selected}
          onChange={onSelect}
        />
      )}
      <button
        className="task-check"
        disabled={busy}
        aria-label={
          (task.is_template
            ? "Open routine "
            : done
              ? "Reopen "
              : "Complete ") + task.title
        }
        onClick={task.is_template ? onOpen : onToggle}
      >
        {task.is_template ? "↻" : done ? <Check size={15} /> : null}
      </button>
      <button className="task-info" onClick={onOpen}>
        <span className="task-title">
          {task.title}
          {task.external?.identifier && (
            <small> · {task.external.identifier}</small>
          )}
          {task.is_template && <small> · Routine</small>}
        </span>
        {(task.notes ||
          task.project ||
          task.assignee !== "owner" ||
          task.work_type ||
          task.tags?.length ||
          !["open", "completed"].includes(task.status)) && (
          <span className="task-meta">
            {task.project && <span>{task.project}</span>}
            {task.assignee && task.assignee !== "owner" && (
              <span>Assigned: {task.assignee}</span>
            )}
            {task.work_type && <span>{task.work_type}</span>}
            {!!task.tags?.length && (
              <span>{task.tags.map((t) => "#" + t).join(" ")}</span>
            )}
            {!["open", "completed"].includes(task.status) && (
              <span>{task.status.replace("_", " ")}</span>
            )}
            {task.notes && <span>{task.notes.slice(0, 80)}</span>}
          </span>
        )}
      </button>
      {task.priority > 0 && (
        <span
          className={"priority p" + task.priority}
          title={"Priority " + task.priority}
        >
          {"!".repeat(task.priority)}
        </span>
      )}
      {task.planned_date && (
        <span className="due">
          Planned {relativeDate(task.planned_date, today)}
        </span>
      )}
      {task.due_date && (
        <button
          className={"due " + (task.due_date < today && !done ? "overdue" : "")}
          onClick={onOpen}
        >
          <CalendarDays size={13} />
          {relativeDate(task.due_date, today)}
          {task.due_time && (
            <span title={task.due_timezone ?? undefined}>
              {" "}
              · {task.due_time.slice(0, 5)}
            </span>
          )}
        </button>
      )}
      <button
        className="icon-button task-more"
        aria-label={"Edit " + task.title}
        onClick={onOpen}
      >
        <MoreHorizontal size={17} />
      </button>
    </div>
  );
}
export function TaskDialog({
  organization = emptyOrganization,
  linkedNotes,
  timezone,
  projects,
  tasks,
  reminders,
  onReminder,
  onAddReminder,
  error,
  task,
  busy,
  onClose,
  onSave,
  onArchive,
}: {
  organization?: Organization;
  linkedNotes?: ReactNode;
  timezone: string;
  projects: Project[];
  tasks: Task[];
  reminders: Schedule[];
  onReminder: (schedule: Schedule) => void;
  onAddReminder: () => void;
  error?: string;
  task: Task;
  busy: boolean;
  onClose: () => void;
  onSave: (args: unknown) => Promise<unknown>;
  onArchive: () => Promise<void>;
}) {
  useDialogFocus();
  const [draft, setDraft] = useState(task);
  useEffect(() => setDraft(task), [task]);
  async function save() {
    if (!draft.title.trim()) return;
    return onSave({
      task_id: task.id,
      expected_revision: task.revision,
      title: draft.title,
      notes: draft.notes,
      project_id: draft.project_id || null,
      parent_task_id: draft.parent_task_id || null,
      assignee: draft.assignee || "owner",
      work_type: draft.work_type || "",
      tags: draft.tags ?? [],
      space_id: draft.project_id ? undefined : draft.space_id || null,
      area_id: draft.project_id ? undefined : draft.area_id || null,
      planned_date: draft.planned_date || null,
      estimate_minutes: draft.estimate_minutes ?? null,
      due_date: draft.due_date || null,
      due_time: draft.due_date ? draft.due_time || null : null,
      due_timezone:
        draft.due_date && draft.due_time
          ? draft.due_timezone || timezone
          : null,
      priority: draft.priority,
      status: draft.status,
    });
  }
  const editorSchema = z.object({
    title: z.string().min(1).max(500),
    notes: z.string().max(10000),
    priority: z.number().int().min(0).max(3),
    status: choice(
      task.id === "new"
        ? ["open", "backlog"]
        : [
            "backlog",
            "open",
            "in_progress",
            "waiting",
            "deferred",
            "completed",
            "cancelled",
          ],
    ),
    project_id: nullableId(projects.map((p) => p.id)),
    parent_task_id: nullableId(
      tasks.filter((t) => t.id !== task.id).map((t) => t.id),
    ),
    space_id: nullableId(organization.spaces.map((s) => s.id)),
    area_id: nullableId(organization.areas.map((a) => a.id)),
    assignee: choice([
      "owner",
      ...organization.actors.map((a) => a.name),
      task.assignee,
    ]),
    work_type: z.string().max(80),
    tags: tagsField,
    planned_date: z.string().date().or(z.literal("")).nullable(),
    due_date: z.string().date().or(z.literal("")).nullable(),
    due_time: z
      .string()
      .regex(/^$|^\\d{2}:\\d{2}(:\\d{2})?$/)
      .nullable(),
    due_timezone: z.string().max(100).nullable(),
    estimate_minutes: z.number().int().min(1).max(100000).nullable(),
  });
  useEditor({
    kind: "task",
    record_id: task.id === "new" ? null : task.id,
    dirty: JSON.stringify(draft) !== JSON.stringify(task),
    busy,
    schema: editorSchema,
    values: Object.fromEntries(
      Object.keys(editorSchema.shape).map((k) => [
        k,
        draft[k as keyof Task] ?? null,
      ]),
    ),
    patch: (v) => setDraft((d) => ({ ...d, ...v })),
    save,
    close: onClose,
  });
  return (
    <div
      className="modal-backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <form
        className="dialog task-create-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="task-dialog-title"
        onSubmit={(e) => {
          e.preventDefault();
          void save();
        }}
      >
        <div className="dialog-heading">
          <h2 id="task-dialog-title">{task.id === "new" ? "New task" : "Task details"}</h2>
          <button
            className="icon-button"
            type="button"
            aria-label="Close"
            onClick={onClose}
          >
            <X size={20} />
          </button>
        </div>
        {error && (
          <p className="error-banner" role="alert">
            {error}
          </p>
        )}
        <label>
          Task
          <input
            autoFocus
            required
            maxLength={500}
            value={draft.title}
            onChange={(e) => setDraft({ ...draft, title: e.target.value })}
          />
        </label>
        <div className="capture-date"><label>Planned day<input type="date" value={draft.planned_date ?? ""} onChange={e => setDraft({...draft, planned_date: e.target.value || null})}/></label>
          {draft.planned_date && <button type="button" className="text-button" onClick={() => setDraft({...draft, planned_date: null})}>Clear planned day</button>}</div>
        <details className="capture-options" open={task.id !== "new" || undefined}><summary>Details & organization</summary>
        <label>
          Notes
          <textarea
            value={draft.notes}
            onChange={(e) => setDraft({ ...draft, notes: e.target.value })}
            rows={4}
            placeholder="A little more context…"
          />
        </label>
        <HomeFields
          organization={organization}
          disabled={!!draft.project_id}
          value={projects.find((p) => p.id === draft.project_id) ?? draft}
          onChange={(home) => setDraft({ ...draft, ...home })}
        />
        <div className="form-grid">
          <label>
            Estimate (minutes)
            <input
              type="number"
              min={1}
              max={100000}
              value={draft.estimate_minutes ?? ""}
              onChange={(e) =>
                setDraft({
                  ...draft,
                  estimate_minutes: e.target.value
                    ? Number(e.target.value)
                    : null,
                })
              }
            />
          </label>
          <label>
            Due date
            <input
              type="date"
              value={draft.due_date ?? ""}
              onChange={(e) => setDraft({ ...draft, due_date: e.target.value })}
            />
          </label>
          <label>
            Due time (optional)
            <input
              type="time"
              disabled={!draft.due_date}
              value={draft.due_time?.slice(0, 5) ?? ""}
              onChange={(e) =>
                setDraft({ ...draft, due_time: e.target.value || null })
              }
            />
          </label>
          {draft.due_time && (
            <label>
              Due time zone
              <input
                value={draft.due_timezone ?? timezone}
                onChange={(e) =>
                  setDraft({ ...draft, due_timezone: e.target.value })
                }
              />
            </label>
          )}
          <label>
            Priority
            <select
              aria-label="Priority"
              value={draft.priority}
              onChange={(e) =>
                setDraft({ ...draft, priority: +e.target.value })
              }
            >
              {priorityLabels.map((label, value) => <option key={value} value={value}>{label}</option>)}
            </select>
          </label>
          <label>
            Project
            <select
              aria-label="Project"
              value={draft.project_id ?? ""}
              onChange={(e) =>
                setDraft({ ...draft, project_id: e.target.value || null })
              }
            >
              <option value="">No project</option>
              {projects
                .filter((p) => !p.archived || p.id === draft.project_id)
                .map((p) => (
                  <option value={p.id} key={p.id}>
                    {p.name}
                  </option>
                ))}
            </select>
          </label>
          <label>
            Parent task
            <select
              aria-label="Parent task"
              value={draft.parent_task_id ?? ""}
              onChange={(e) =>
                setDraft({ ...draft, parent_task_id: e.target.value || null })
              }
            >
              <option value="">Top-level task</option>
              {tasks
                .filter((t) => t.id !== draft.id)
                .map((t) => (
                  <option value={t.id} key={t.id}>
                    {t.title}
                  </option>
                ))}
            </select>
          </label>
          <label>
            Assignee
            <input
              value={draft.assignee ?? "owner"}
              maxLength={100}
              required
              onChange={(e) => setDraft({ ...draft, assignee: e.target.value })}
              list="assignee-options"
            />
            <datalist id="assignee-options">
              {organization.actors.length ? (
                organization.actors
                  .filter((a) => !a.archived)
                  .map((a) => <option key={a.id} value={a.name} />)
              ) : (
                <>
                  <option value="owner" />
                  <option value="Eri" />
                </>
              )}
            </datalist>
          </label>
          <label>
            Work type
            <input
              value={draft.work_type ?? ""}
              maxLength={80}
              placeholder="Research, admin, development…"
              onChange={(e) =>
                setDraft({ ...draft, work_type: e.target.value })
              }
            />
          </label>
          <label>
            Tags
            <input
              value={(draft.tags ?? []).join(", ")}
              maxLength={800}
              placeholder="Separate with commas"
              onChange={(e) =>
                setDraft({
                  ...draft,
                  tags: e.target.value.split(",").map((t) => t.trim()),
                })
              }
            />
          </label>
          <label>
            Status
            <select
              aria-label="Status"
              value={draft.status}
              onChange={(e) => setDraft({ ...draft, status: e.target.value })}
            >
              {(task.id === "new"
                ? ["open", "backlog"]
                : [
                    "backlog",
                    "open",
                    "in_progress",
                    "waiting",
                    "deferred",
                    "completed",
                    "cancelled",
                  ]
              ).map((s) => (
                <option key={s} value={s}>
                  {humanLabel(s)}
                </option>
              ))}
            </select>
          </label>
        </div>
        <p className="footnote">
          Assigning a task organizes it; it does not start an agent. A deadline
          does not send a notification.
        </p>
        </details>
        <SchedulingHelp />
        {linkedNotes && (
          <fieldset
            className="linked-notes-fieldset"
            disabled={JSON.stringify(draft) !== JSON.stringify(task)}
          >
            {linkedNotes}
          </fieldset>
        )}
        {task.id !== "new" && (
          <section className="task-reminders">
            <h3>Reminders</h3>
            {reminders.map((r) => (
              <button
                type="button"
                key={r.id}
                className="text-button"
                disabled={JSON.stringify(draft) !== JSON.stringify(task)}
                onClick={() => onReminder(r)}
              >
                <Clock3 size={14} />
                {r.title} · {r.status}
              </button>
            ))}
            <button
              type="button"
              className="text-button"
              disabled={JSON.stringify(draft) !== JSON.stringify(task)}
              onClick={onAddReminder}
            >
              <Plus size={14} />
              Add linked reminder
            </button>
            {JSON.stringify(draft) !== JSON.stringify(task) && (
              <p className="footnote">
                Save your edits before opening a reminder.
              </p>
            )}
          </section>
        )}
        <div className="dialog-actions">
          <button
            hidden={task.id === "new"}
            type="button"
            className="text-button danger"
            onClick={() => void onArchive()}
            disabled={busy}
          >
            <Trash2 size={15} />
            Archive
          </button>
          <button
            type="submit"
            className="primary"
            disabled={busy || !draft.title.trim()}
          >
            {busy ? "Saving…" : task.id === "new" ? "Create task" : "Save task"}
          </button>
        </div>
      </form>
    </div>
  );
}
export function ReminderDialog({
  zone,
  busy,
  onClose,
  onSave,
}: {
  zone: string;
  busy: boolean;
  onClose: () => void;
  onSave: (args: unknown) => Promise<void>;
}) {
  useDialogFocus();
  const [title, setTitle] = useState(""),
    [when, setWhen] = useState(""),
    [repeat, setRepeat] = useState(""),
    [kind, setKind] = useState("reminder");
  return (
    <div
      className="modal-backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <form
        className="dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="reminder-title"
        onSubmit={(e) => {
          e.preventDefault();
          void onSave({
            title,
            when,
            timezone: zone,
            recurrence: repeat || null,
            kind,
            original_words: title,
          });
        }}
      >
        <div className="dialog-heading">
          <h2 id="reminder-title">A nudge for later</h2>
          <button
            className="icon-button"
            type="button"
            aria-label="Close"
            onClick={onClose}
          >
            <X size={20} />
          </button>
        </div>
        <label>
          Remind me to
          <input
            autoFocus
            required
            value={title}
            maxLength={500}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="Call Josh, water the plants…"
          />
        </label>
        <label>
          When
          <input
            required
            type="datetime-local"
            aria-label="When"
            value={when}
            onChange={(e) => setWhen(e.target.value)}
          />
          <span className="field-hint">{zone}</span>
        </label>
        <label>
          Repeat
          <select
            value={repeat}
            onChange={(e) => {
              setRepeat(e.target.value);
              if (!e.target.value) setKind("reminder");
            }}
          >
            <option value="">Just once</option>
            <option value="FREQ=DAILY">Every day</option>
            <option value="FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR">Weekdays</option>
            <option value="FREQ=WEEKLY">Every week</option>
            <option value="FREQ=MONTHLY">Every month</option>
          </select>
        </label>
        {repeat && (
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={kind === "recurring_task"}
              onChange={(e) =>
                setKind(e.target.checked ? "recurring_task" : "reminder")
              }
            />
            <span>
              Create a new task each time
              <small>Completing today’s task keeps tomorrow’s routine.</small>
            </span>
          </label>
        )}
        <div className="dialog-actions">
          <span className="footnote">Saved even when the app is closed.</span>
          <button className="primary" disabled={busy || !title.trim() || !when}>
            {busy ? "Saving…" : "Save reminder"}
          </button>
        </div>
      </form>
    </div>
  );
}
export function MemoryCapture({
  busy,
  onCapture,
}: {
  busy: boolean;
  onCapture: (content: string) => Promise<unknown>;
}) {
  const [text, setText] = useState("");
  return (
    <form
      className="quick-add"
      onSubmit={async (e) => {
        e.preventDefault();
        if (await onCapture(text)) setText("");
      }}
    >
      <Brain size={20} />
      <input
        value={text}
        onChange={(e) => setText(e.target.value)}
        aria-label="New memory"
        placeholder="Something you'd like me to remember…"
      />
      <button disabled={busy || !text.trim()}>
        Remember
        <Plus size={15} />
      </button>
    </form>
  );
}
export function SettingsPanel({
  boot,
  busy,
  onSave,
  onPush,
  section,
}: {
  section: "profile" | "voice" | "integrations" | "privacy" | "system" | "organization" | "notifications";
  boot: Bootstrap;
  busy: boolean;
  onSave: (args: unknown) => Promise<void>;
  onPush: () => Promise<void>;
}) {
  const p = boot.preferences;
  return (
    <div className="settings-sections">
      <section hidden={section !== "profile"}>
        <h2>Your name</h2>
        <form
          className="setting-row"
          onSubmit={async (e) => {
            e.preventDefault();
            const preferred_name = String(
              new FormData(e.currentTarget).get("preferred_name") ?? "",
            ).trim();
            if (preferred_name) await onSave({ preferred_name });
          }}
        >
          <label>
            What should Eri call you?
            <input
              key={p.preferred_name}
              name="preferred_name"
              aria-label="Preferred name"
              defaultValue={p.preferred_name}
              required
              maxLength={80}
              autoComplete="nickname"
            />
          </label>
          <button className="secondary" disabled={busy}>
            Save name
          </button>
        </form>
      </section>
      <section hidden={section !== "privacy"}>
        <h2>Conversation & memory</h2>
        <p>
          Choose what stays with you. New settings apply to new conversations.
        </p>
        {[
          {
            key: "history_enabled",
            label: "Conversation history",
            description: "Keep the words you exchange with Eridani.",
          },
          {
            key: "memory_learning",
            label: "Learn from conversations",
            description:
              "Automatically extract useful facts and preferences, with sources and semantic search.",
          },
          {
            key: "deep_sleep_enabled",
            label: "Weekly deep sleep",
            description:
              "Review memories on Sundays at 3 AM in your home time zone. Ask before resolving uncertain names.",
          },
        ].map((item) => (
          <label className="setting-row" key={item.key}>
            <span>
              <strong>{item.label}</strong>
              <small>{item.description}</small>
            </span>
            <input
              className="switch"
              type="checkbox"
              role="switch"
              checked={
                p[
                  item.key as
                    | "history_enabled"
                    | "memory_learning"
                    | "deep_sleep_enabled"
                ]
              }
              disabled={busy}
              onChange={(e) => void onSave({ [item.key]: e.target.checked })}
            />
          </label>
        ))}
        <label className="setting-row">
          <span>
            <strong>Keep ordinary history</strong>
            <small>Explicitly saved memories stay until you delete them.</small>
          </span>
          <select
            value={p.history_days}
            onChange={(e) => void onSave({ history_days: +e.target.value })}
          >
            <option value={0}>Until I delete it</option>
            <option value={30}>30 days</option>
            <option value={90}>90 days</option>
            <option value={365}>1 year</option>
          </select>
        </label>
        <div className="privacy-callout">
          <Shield size={18} />
          <p>
            Eridani does not save raw audio. Conversation history follows these
            settings; tasks and notes save separately. Cloud models process what
            you send.
          </p>
        </div>
      </section>
      <section hidden={section !== "notifications"}>
        <h2>Reminder delivery</h2>
        <label className="setting-row">
          <span>
            <strong>Default reminder time</strong>
            <small>Used when you give a date without a time.</small>
          </span>
          <select
            value={p.default_reminder_hour}
            onChange={(e) =>
              void onSave({ default_reminder_hour: +e.target.value })
            }
          >
            {Array.from({ length: 24 }, (_, i) => (
              <option value={i} key={i}>
                {String(i).padStart(2, "0")}:00
              </option>
            ))}
          </select>
        </label>
        <div className="setting-row">
          <span>
            <strong>Phone notifications</strong>
            <small>Receive a nudge with the app closed.</small>
          </span>
          <button className="secondary" onClick={() => void onPush()}>
            Enable on this device
          </button>
        </div>
        <label className="setting-row">
          <span>
            <strong>Show reminder details</strong>
            <small>Include reminder text on your lock screen.</small>
          </span>
          <input
            type="checkbox"
            className="switch"
            role="switch"
            checked={p.detailed_notifications}
            onChange={(e) =>
              void onSave({ detailed_notifications: e.target.checked })
            }
          />
        </label>
        <p className="footnote">Home time zone: {p.timezone}</p>
      </section>
      <section hidden={section !== "system"}>
        <h2>Task agent</h2>
        <label className="setting-row">
          <span>
            <strong>Backend model</strong>
            <small>Used for text chat and tasks delegated by GPT-Live.</small>
          </span>
          <select
            aria-label="Backend model"
            value={boot.agent_profile}
            disabled={busy}
            onChange={(e) => void onSave({ agent_profile: e.target.value })}
          >
            {boot.agent_options.map((model) => (
              <option
                key={model.id}
                value={model.id}
                disabled={!model.available}
              >
                {model.label}
                {model.available ? "" : " · API key needed"}
              </option>
            ))}
          </select>
        </label>
        <p className="footnote">
          Changes apply to the next request on all your devices. Voice selection
          is separate. Automatic memory learning and note extraction still use
          OpenAI.
        </p>
        {boot.agent_profile === "luna" && (
          <p className="footnote">
            Luna uses low reasoning effort for task work, including tasks
            delegated during GPT-Live conversations.
          </p>
        )}
        {boot.agent_options.some(
          (model) => model.provider === "gemini" && !model.available,
        ) && (
          <p className="footnote">
            To try Gemini, add GEMINI_API_KEY to the server .env file, then
            recreate the API and worker containers.
          </p>
        )}
      </section>
      <section hidden={section !== "system"}>
        <h2>Model usage</h2>
        {boot.budget.tracking_enabled === false ? (
          <p className="footnote">
            Cost tracking and spending limits are off during development.
            Monitor usage in{" "}
            <a
              href="https://platform.openai.com/usage"
              target="_blank"
              rel="noreferrer"
            >
              OpenAI Usage
            </a>{" "}
            and{" "}
            <a
              href="https://aistudio.google.com/usage"
              target="_blank"
              rel="noreferrer"
            >
              Google AI Studio
            </a>
            .
          </p>
        ) : (
          <>
            <form
              className="setting-row"
              onSubmit={(event) => {
                event.preventDefault();
                const data = new FormData(event.currentTarget);
                void onSave({ monthly_budget_usd: Number(data.get("limit")) });
              }}
            >
              <label>
                Monthly limit ($)
                <input
                  name="limit"
                  type="number"
                  min="0"
                  max="10000"
                  step="1"
                  defaultValue={boot.budget.limit_usd}
                  key={boot.budget.limit_usd}
                  style={{ width: "100px", marginTop: "8px" }}
                />
              </label>
              <button className="secondary" disabled={busy}>
                Save limit
              </button>
            </form>
            <div className="budget-line">
              <strong>
                ${boot.budget.spent_usd.toFixed(2)}
                <span> / ${boot.budget.limit_usd.toFixed(0)} this month</span>
              </strong>
              <span>
                ${boot.budget.active_reserved_usd.toFixed(2)} reserved for
                active work
              </span>
            </div>
            {boot.budget.uncertain_usd > 0 && (
              <p className="footnote">
                ${boot.budget.uncertain_usd.toFixed(2)} is held for sessions
                with unconfirmed final usage.
              </p>
            )}
            <BudgetHolds />
            <p className="footnote">
              At this month's pace: about $
              {boot.budget.projected_month_usd.toFixed(2)} this month.
            </p>
            {boot.budget.budget_mode !== "normal" && (
              <p role="status" className="footnote">
                {["defer_optional", "paused"].includes(boot.budget.budget_mode)
                  ? "Optional memory processing is paused near your limit. Saved tasks and reminders still work."
                  : "Your usage and reservations have reached 80% of the monthly limit."}
              </p>
            )}
            <progress
              max={Math.max(1, boot.budget.limit_usd)}
              value={boot.budget.spent_usd + boot.budget.reserved_usd}
            />
            <p className="footnote">
              Usage is estimated from provider reports. Tasks and reminders keep
              working when model spending stops.
            </p>
          </>
        )}
      </section>
      <section hidden={section !== "system"}>
        <h2>Your data</h2>
        <p className="footnote">
          {boot.last_backup_at
            ? "Last encrypted backup: " +
              timeLabel(boot.last_backup_at, p.timezone)
            : "First backup is pending. Check the backup service if this persists."}
        </p>
        <div className="setting-row">
          <span>
            <strong>Take it with you</strong>
            <small>Download your tasks, reminders, sources and memories.</small>
          </span>
          <a href="/api/v1/export" className="secondary">
            <Download size={15} />
            Export JSON
          </a>
        </div>
        <a className="text-button" href="/api/v1/export?format=csv">
          Download tasks as CSV
          <ChevronRight size={14} />
        </a>
      </section>
    </div>
  );
}
