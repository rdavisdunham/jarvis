import { z } from "zod";
import { useEditor, choice } from "./editor-control";
import { useState } from "react";
import { X } from "lucide-react";
import { useDialogFocus } from "./components";
import type { Task, Project } from "./types";

export function BulkTaskDialog({
  tasks,
  projects,
  busy,
  error,
  onClose,
  onSave,
}: {
  tasks: Task[];
  projects: Project[];
  busy: boolean;
  error: string;
  onClose: () => void;
  onSave: (items: Record<string, unknown>[]) => Promise<unknown>;
}) {
  useDialogFocus();
  const [status, setStatus] = useState(""),
    [project, setProject] = useState("unchanged");
  const [setDate, changeDate] = useState(false),
    [date, changeDay] = useState("");
  const [assignee, setAssignee] = useState(""),
    [priority, setPriority] = useState("");
  const hasChanges =
    !!status ||
    project !== "unchanged" ||
    setDate ||
    !!assignee.trim() ||
    !!priority;
  async function save() {
    if (!hasChanges || !tasks.length) return;
    const changes: Record<string, unknown> = {};
    if (status) changes.status = status;
    if (project !== "unchanged") changes.project_id = project || null;
    if (setDate) changes.due_date = date || null;
    if (assignee.trim()) changes.assignee = assignee.trim();
    if (priority) changes.priority = Number(priority);
    return onSave(
      tasks.map((t) => ({
        task_id: t.id,
        expected_revision: t.revision,
        ...changes,
      })),
    );
  }
  useEditor({
    kind: "bulk",
    dirty: hasChanges,
    busy,
    schema: z.object({
      status: choice([
        "",
        "backlog",
        "open",
        "in_progress",
        "waiting",
        "deferred",
        "completed",
        "cancelled",
      ]),
      project_id: choice(["unchanged", "", ...projects.map((p) => p.id)]),
      change_due_date: z.boolean(),
      due_date: z.string().date().or(z.literal("")),
      assignee: z.string().max(100),
      priority: choice(["", "0", "1", "2", "3"]),
    }),
    values: {
      status,
      project_id: project,
      change_due_date: setDate,
      due_date: date,
      assignee,
      priority,
    },
    save,
    close: onClose,
    patch: (v) => {
      if ("status" in v) setStatus(v.status as string);
      if ("project_id" in v) setProject(v.project_id as string);
      if ("change_due_date" in v) changeDate(v.change_due_date as boolean);
      if ("due_date" in v) changeDay(v.due_date as string);
      if ("assignee" in v) setAssignee(v.assignee as string);
      if ("priority" in v) setPriority(v.priority as string);
    },
  });
  return (
    <div className="modal-backdrop">
      <form
        className="dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="bulk-title"
        onSubmit={(e) => {
          e.preventDefault();
          void save();
        }}
      >
        <div className="dialog-heading">
          <h2 id="bulk-title">Edit {tasks.length} tasks</h2>
          <button
            type="button"
            className="icon-button"
            aria-label="Close bulk editor"
            onClick={onClose}
            disabled={busy}
          >
            <X size={20} />
          </button>
        </div>
        {error && (
          <p role="alert" className="error-banner">
            {error}
          </p>
        )}
        <p className="footnote">Only the fields you choose will change.</p>
        <div className="bulk-task-names">
          {tasks.map((t) => (
            <span key={t.id}>{t.title}</span>
          ))}
        </div>
        <div className="form-grid">
          <label>
            Status
            <select
              aria-label="Bulk status"
              value={status}
              onChange={(e) => setStatus(e.target.value)}
            >
              <option value="">Keep current</option>
              {[
                "backlog",
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
          <label>
            Project
            <select
              aria-label="Bulk project"
              value={project}
              onChange={(e) => setProject(e.target.value)}
            >
              <option value="unchanged">Keep current</option>
              <option value="">No project</option>
              {projects
                .filter((p) => !p.archived)
                .map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                  </option>
                ))}
            </select>
          </label>
          <label>
            Assignee
            <input
              value={assignee}
              onChange={(e) => setAssignee(e.target.value)}
              placeholder="Keep current"
              maxLength={100}
            />
          </label>
          <label>
            Priority
            <select
              aria-label="Bulk priority"
              value={priority}
              onChange={(e) => setPriority(e.target.value)}
            >
              <option value="">Keep current</option>
              {[0, 1, 2, 3].map((n) => (
                <option key={n} value={n}>
                  {n === 0 ? "None" : "Priority " + n}
                </option>
              ))}
            </select>
          </label>
        </div>
        <label className="note-check">
          <input
            type="checkbox"
            checked={setDate}
            onChange={(e) => changeDate(e.target.checked)}
          />
          Change due date
        </label>
        {setDate && (
          <label>
            New due date
            <input
              aria-label="Bulk due date"
              type="date"
              value={date}
              onChange={(e) => changeDay(e.target.value)}
            />
            <small>
              Leave blank to clear the deadline. Existing due times stay with a
              new date.
            </small>
          </label>
        )}
        <div className="dialog-actions">
          <button
            className="primary"
            disabled={busy || !hasChanges || !tasks.length}
          >
            Apply changes
          </button>
        </div>
      </form>
    </div>
  );
}
