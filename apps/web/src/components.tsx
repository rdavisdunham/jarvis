import { useEffect, useState } from "react";
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
import type { Bootstrap, Task } from "./types";
function useDialogFocus() {
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    const trap = (event: KeyboardEvent) => {
      if (event.key !== "Tab") return;
      const dialog = document.querySelector('[role="dialog"]');
      const items = Array.from(
        dialog?.querySelectorAll<HTMLElement>(
          'button:not(:disabled), input:not(:disabled), textarea, select, [tabindex="0"]',
        ) ?? [],
      );
      const first = items[0],
        last = items.at(-1);
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last?.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first?.focus();
      }
    };
    document.addEventListener("keydown", trap);
    return () => {
      document.removeEventListener("keydown", trap);
      previous?.focus();
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
}: {
  task: Task;
  today: string;
  busy: boolean;
  onToggle: () => void;
  onOpen: () => void;
}) {
  const done = task.status === "completed";
  return (
    <div className={"task-row " + (done ? "done" : "")}>
      <button
        className="task-check"
        disabled={busy}
        aria-label={(done ? "Reopen " : "Complete ") + task.title}
        onClick={onToggle}
      >
        {done ? <Check size={15} /> : null}
      </button>
      <button className="task-info" onClick={onOpen}>
        <span className="task-title">{task.title}</span>
        {(task.notes ||
          task.project ||
          !["open", "completed"].includes(task.status)) && (
          <span className="task-meta">
            {task.project && <span>{task.project}</span>}
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
      {task.due_date && (
        <button
          className={"due " + (task.due_date < today && !done ? "overdue" : "")}
          onClick={onOpen}
        >
          <CalendarDays size={13} />
          {relativeDate(task.due_date, today)}
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
  task,
  busy,
  onClose,
  onSave,
  onArchive,
}: {
  task: Task;
  busy: boolean;
  onClose: () => void;
  onSave: (args: unknown) => Promise<void>;
  onArchive: () => Promise<void>;
}) {
  useDialogFocus();
  const [draft, setDraft] = useState(task);
  useEffect(() => setDraft(task), [task]);
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
        aria-labelledby="task-dialog-title"
        onSubmit={(e) => {
          e.preventDefault();
          void onSave({
            task_id: task.id,
            expected_revision: task.revision,
            title: draft.title,
            notes: draft.notes,
            project: draft.project || null,
            due_date: draft.due_date || null,
            priority: draft.priority,
            status: draft.status,
          });
        }}
      >
        <div className="dialog-heading">
          <h2 id="task-dialog-title">Task details</h2>
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
          Task
          <input
            autoFocus
            required
            maxLength={500}
            value={draft.title}
            onChange={(e) => setDraft({ ...draft, title: e.target.value })}
          />
        </label>
        <label>
          Notes
          <textarea
            value={draft.notes}
            onChange={(e) => setDraft({ ...draft, notes: e.target.value })}
            rows={4}
            placeholder="A little more context…"
          />
        </label>
        <div className="form-grid">
          <label>
            Due date
            <input
              type="date"
              value={draft.due_date ?? ""}
              onChange={(e) => setDraft({ ...draft, due_date: e.target.value })}
            />
          </label>
          <label>
            Priority
            <select
              aria-label="Priority"
              value={draft.priority}
              onChange={(e) =>
                setDraft({ ...draft, priority: +e.target.value })
              }
            >
              <option value={0}>Normal</option>
              <option value={1}>Low</option>
              <option value={2}>Medium</option>
              <option value={3}>High</option>
            </select>
          </label>
          <label>
            Project
            <input
              value={draft.project ?? ""}
              onChange={(e) => setDraft({ ...draft, project: e.target.value })}
              placeholder="Inbox"
            />
          </label>
          <label>
            Status
            <select
              aria-label="Status"
              value={draft.status}
              onChange={(e) => setDraft({ ...draft, status: e.target.value })}
            >
              {[
                "open",
                "in_progress",
                "waiting",
                "deferred",
                "completed",
                "cancelled",
              ].map((s) => (
                <option key={s} value={s}>
                  {s.replace("_", " ")}
                </option>
              ))}
            </select>
          </label>
        </div>
        <p className="footnote">
          A due date doesn’t create a notification. Set a reminder separately.
        </p>
        <div className="dialog-actions">
          <button
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
            {busy ? "Saving…" : "Save task"}
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
}: {
  boot: Bootstrap;
  busy: boolean;
  onSave: (args: unknown) => Promise<void>;
  onPush: () => Promise<void>;
}) {
  const p = boot.preferences;
  return (
    <div className="settings-sections">
      <section>
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
              "Remember explicitly stated preferences with their sources.",
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
              checked={p[item.key as "history_enabled" | "memory_learning"]}
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
            Raw audio is never recorded by Eridani. Private sessions keep task
            receipts, but no conversation history. Cloud processing still
            applies.
          </p>
        </div>
      </section>
      <section>
        <h2>Reminders</h2>
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
      <section>
        <h2>Model usage</h2>
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
          <span>${boot.budget.reserved_usd.toFixed(2)} reserved</span>
        </div>
        <progress
          max={Math.max(1, boot.budget.limit_usd)}
          value={boot.budget.spent_usd + boot.budget.reserved_usd}
        />
        <p className="footnote">
          Usage is estimated from provider reports. Tasks and reminders keep
          working when model spending stops.
        </p>
      </section>
      <section>
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
