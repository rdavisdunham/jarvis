import { useState } from "react";
import { X, Check, Clock3 } from "lucide-react";
import { useDialogFocus } from "./components";
import { localDateTime } from "./workspace";
import type { Schedule, Task, Project, Notice } from "./types";
export function ScheduleDialog({
  schedule,
  initialDate,
  linkedTask,
  zone,
  tasks,
  projects,
  notices,
  busy,
  error,
  onClose,
  mutate,
}: {
  schedule: Schedule | null;
  initialDate?: string;
  linkedTask?: Task | null;
  zone: string;
  tasks: Task[];
  projects: Project[];
  notices: Notice[];
  busy: boolean;
  error?: string;
  onClose: () => void;
  mutate: (tool: string, args: unknown, message: string) => Promise<unknown>;
}) {
  useDialogFocus();
  const [title, setTitle] = useState(
    schedule?.title ?? linkedTask?.title ?? "",
  );
  const [timezone, setZone] = useState(schedule?.timezone ?? zone);
  const [when, setWhen] = useState(
    schedule
      ? localDateTime(
          schedule.next_run_at ?? schedule.anchor_at,
          schedule.timezone,
        )
      : initialDate
        ? initialDate + "T10:00"
        : "",
  );
  const [repeat, setRepeat] = useState(schedule?.recurrence ?? "");
  const kind = schedule?.kind ?? "reminder";
  const [taskId, setTaskId] = useState(
    schedule?.task_id ?? linkedTask?.id ?? "",
  );
  const [projectId, setProjectId] = useState(
    schedule?.project_id ?? linkedTask?.project_id ?? "",
  );
  const inactive = !!schedule && schedule.status !== "active";
  const outstanding = notices.find(
    (n) => n.schedule_id === schedule?.id && !n.completed_at,
  );
  const run = async (tool: string, args: unknown, message: string) => {
    if (await mutate(tool, args, message)) onClose();
  };
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
        aria-labelledby="schedule-heading"
        onSubmit={(e) => {
          e.preventDefault();
          if (inactive)
            void run(
              "schedule.reschedule",
              {
                schedule_id: schedule!.id,
                expected_revision: schedule!.revision,
                when,
              },
              "Reminder rescheduled",
            );
          else {
            const values: Record<string, unknown> = {
              title,
              when,
              timezone,
              recurrence: repeat || null,
              task_id: taskId || null,
              project_id: projectId || null,
            };
            if (schedule) {
              if (
                timezone === schedule.timezone &&
                when ===
                  localDateTime(
                    schedule.next_run_at ?? schedule.anchor_at,
                    schedule.timezone,
                  )
              ) {
                delete values.when;
                delete values.timezone;
              }
              if (repeat === (schedule.recurrence ?? ""))
                delete values.recurrence;
              if (taskId === (schedule.task_id ?? "")) delete values.task_id;
              if (projectId === (schedule.project_id ?? ""))
                delete values.project_id;
            }
            void run(
              schedule ? "schedule.update" : "schedule.create",
              {
                ...(schedule
                  ? {
                      schedule_id: schedule.id,
                      expected_revision: schedule.revision,
                    }
                  : { kind }),
                ...values,
              },
              "Reminder saved",
            );
          }
        }}
      >
        <div className="dialog-heading">
          <h2 id="schedule-heading">
            {schedule ? "Reminder details" : "New reminder"}
          </h2>
          <button
            className="icon-button"
            type="button"
            aria-label="Close reminder"
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
          Reminder
          <input
            autoFocus
            required
            maxLength={500}
            value={title}
            disabled={inactive}
            onChange={(e) => setTitle(e.target.value)}
          />
        </label>
        {inactive && (
          <p className="footnote">
            This reminder is {schedule?.status}. Choose a new time to reactivate
            it.
          </p>
        )}
        <div className="form-grid">
          <label>
            When
            <input
              type="datetime-local"
              required
              value={when}
              onChange={(e) => setWhen(e.target.value)}
            />
          </label>
          <label>
            Time zone
            <input
              required
              value={timezone}
              disabled={inactive}
              onChange={(e) => setZone(e.target.value)}
            />
          </label>
          <label>
            Repeat
            <select
              aria-label="Repeat"
              value={repeat}
              disabled={inactive}
              onChange={(e) => {
                setRepeat(e.target.value);
              }}
            >
              <option value="">Just once</option>
              <option value="FREQ=DAILY">Every day</option>
              <option value="FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR">Weekdays</option>
              <option value="FREQ=WEEKLY">Weekly</option>
              <option value="FREQ=MONTHLY">Monthly</option>
              {repeat &&
                ![
                  "FREQ=DAILY",
                  "FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR",
                  "FREQ=WEEKLY",
                  "FREQ=MONTHLY",
                ].includes(repeat) && <option value={repeat}>{repeat}</option>}
            </select>
          </label>
          <label>
            Linked task
            <select
              aria-label="Linked task"
              value={taskId}
              disabled={inactive || kind === "recurring_task"}
              onChange={(e) => setTaskId(e.target.value)}
            >
              {!schedule && (
                <option value="">Create a task with this alert</option>
              )}
              {tasks
                .filter((t) => !t.archived)
                .map((t) => (
                  <option key={t.id} value={t.id}>
                    {t.title}
                  </option>
                ))}
            </select>
          </label>
          {!taskId && (
            <label>
              Project
              <select
                aria-label="Project"
                value={projectId}
                disabled={inactive}
                onChange={(e) => setProjectId(e.target.value)}
              >
                <option value="">No project</option>
                {projects
                  .filter((p) => !p.archived || p.id === projectId)
                  .map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.name}
                    </option>
                  ))}
              </select>
            </label>
          )}
        </div>
        <p className="footnote">
          {taskId
            ? "Completing this task closes its alerts. Repeat alerts on an existing task stop when it is done."
            : repeat
              ? "Each occurrence creates a task with its own completion history."
              : "This creates one task with an alert. The alert time is separate from its deadline."}
        </p>
        <div className="dialog-actions">
          {schedule &&
            ["active", "finished"].includes(schedule.status) &&
            (!schedule.recurrence || outstanding) && (
              <button
                type="button"
                className="text-button"
                disabled={busy}
                onClick={() =>
                  void run(
                    schedule.recurrence
                      ? "notification.complete"
                      : "schedule.complete",
                    schedule.recurrence
                      ? { notification_id: outstanding!.id }
                      : {
                          schedule_id: schedule.id,
                          expected_revision: schedule.revision,
                        },
                    "Reminder completed",
                  )
                }
              >
                <Check size={16} />
                {schedule.recurrence ? "Done this time" : "Complete"}
              </button>
            )}
          {schedule?.status === "active" && (
            <button
              className="text-button danger"
              disabled={busy}
              type="button"
              onClick={() =>
                void run(
                  "schedule.cancel",
                  {
                    schedule_id: schedule.id,
                    expected_revision: schedule.revision,
                  },
                  "Reminder cancelled",
                )
              }
            >
              Cancel {schedule.recurrence ? "series" : "reminder"}
            </button>
          )}
          <button className="primary" disabled={busy || !title.trim() || !when}>
            <Clock3 size={16} />
            {inactive ? "Reschedule" : "Save reminder"}
          </button>
        </div>
      </form>
    </div>
  );
}
