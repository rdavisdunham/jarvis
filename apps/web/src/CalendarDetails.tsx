import { useEffect, useRef, useState } from "react";
import { z } from "zod";
import {
  Check,
  Pencil,
  X,
  CalendarDays,
  ListTodo,
  ExternalLink,
} from "lucide-react";
import { api } from "./api";
import { useDialogFocus, timeLabel, recurrenceLabel } from "./components";
import { useEditor } from "./editor-control";
import {
  calendarKind,
  completableCalendarTask,
  dateLabel,
} from "./calendar-presentation";
import { shiftDate } from "./workspace";
import type { CalendarEntry, Task, Schedule } from "./types";
import type { Organization } from "./productivity";
import type { NoteRecord } from "./Notes";
import type { PlanningRecord } from "./PlanningDialog";

export function CalendarDetails({
  event,
  organization,
  allTasks,
  allSchedules,
  zone,
  onClose,
  onEdit,
  onTask,
  onNote,
  onComplete,
}: {
  event: CalendarEntry;
  organization: Organization;
  allTasks: Task[];
  allSchedules: Schedule[];
  zone: string;
  onClose: () => void;
  onEdit: (
    event: CalendarEntry,
    task: Task | null,
    schedule: Schedule | null,
  ) => void;
  onTask: (task: Task) => void;
  onNote: (id: string) => void;
  onComplete: (task: Task) => Promise<unknown>;
}) {
  useDialogFocus();
  const card = useRef<HTMLElement>(null);
  useEffect(() => {
    card.current
      ?.querySelector<HTMLButtonElement>("button:not(:disabled)")
      ?.focus();
  }, []);
  const [task, setTask] = useState<Task | null>(null),
    [schedule, setSchedule] = useState<Schedule | null>(null);
  const [record, setRecord] = useState<PlanningRecord | null>(null),
    [notes, setNotes] = useState<NoteRecord[]>([]);
  const [loading, setLoading] = useState(true),
    [saving, setSaving] = useState(false),
    [error, setError] = useState("");
  const [retry, setRetry] = useState(0),
    [notesError, setNotesError] = useState("");
  const local = event.kind === "event" || event.kind === "block";
  useEffect(() => {
    let active = true;
    setLoading(true);
    setError("");
    setNotesError("");
    setNotes([]);
    const load = async () => {
      let foundTask: Task | null = null,
        foundSchedule: Schedule | null = null,
        foundRecord: PlanningRecord | null = null;
      if (local)
        foundRecord = await api<PlanningRecord>("/planning/" + event.entity_id);
      else if (event.kind !== "task")
        foundSchedule = await api<Schedule>("/schedules/" + event.entity_id);
      const id =
        foundRecord?.task_id ??
        (event.kind === "task"
          ? event.entity_id
          : (event.task_id ?? foundSchedule?.task_id));
      if (id) foundTask = await api<Task>("/tasks/" + id);
      if (!active) return;
      setTask(foundTask);
      setSchedule(foundSchedule);
      setRecord(foundRecord);
      setLoading(false);
      if (foundTask)
        void api<{ items: NoteRecord[] }>("/notes?task_id=" + foundTask.id)
          .then((data) => {
            if (active) setNotes(data.items);
          })
          .catch((e) => {
            if (active) setNotesError(e.message);
          });
    };
    void load().catch((e) => {
      if (active) {
        setError(e.message);
        setLoading(false);
      }
    });
    return () => {
      active = false;
    };
  }, [event.id, retry, local]);
  useEffect(() => {
    const escape = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.preventDefault();
      e.stopImmediatePropagation();
      if (!saving) onClose();
    };
    document.addEventListener("keydown", escape, true);
    return () => document.removeEventListener("keydown", escape, true);
  }, [saving, onClose]);
  const parent = allTasks.find((t) => t.id === task?.parent_task_id);
  const children = allTasks.filter(
    (t) => t.parent_task_id === task?.id && !t.archived,
  );
  const alerts = allSchedules.filter(
    (s) => s.task_id === task?.id && s.id !== schedule?.id,
  );
  const fields = record?.fields;
  const kind = calendarKind(event);
  const title =
    fields?.title ??
    (event.kind === "task" ? task?.title : null) ??
    schedule?.title ??
    event.title;
  const canEdit =
    !loading &&
    !error &&
    (local ? !!record : event.kind === "task" ? !!task : !!schedule);
  const completion = completableCalendarTask(event, task);
  const project = organization.projects.find(
    (p) => p.id === (task?.project_id ?? event.project_id),
  );
  const space = organization.spaces.find(
    (p) => p.id === (task?.space_id ?? project?.space_id),
  );
  const area = organization.areas.find(
    (p) => p.id === (task?.area_id ?? project?.area_id),
  );
  const goals = organization.goals.filter((g) =>
    project?.goal_ids?.includes(g.id),
  );
  const summary = {
    kind: event.kind,
    title,
    date: event.date,
    at: event.at,
    status: task?.status ?? event.status,
    task,
    schedule,
    appointment: record,
    notes: notes.map((n) => ({ id: n.id, title: n.title })),
  };
  useEditor({
    kind: local ? "event" : event.kind === "task" ? "task" : "reminder",
    record_id: event.entity_id,
    mode: "detail",
    dirty: false,
    busy: loading || saving,
    schema: z.object({}),
    values: summary,
    close: onClose,
    patch: () => {
      throw new Error("This is a saved detail card. Open its form to edit.");
    },
  });
  async function complete() {
    if (!task || !completion) return;
    setSaving(true);
    setError("");
    try {
      const result = await onComplete(task);
      if (result) onClose();
      else
        setError(
          "The task could not be updated. Reload its details before trying again.",
        );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  }
  const appointmentTime = fields
    ? fields.all_day
      ? dateLabel(fields.start) +
        (shiftDate(fields.end, -1) !== fields.start
          ? " – " + dateLabel(shiftDate(fields.end, -1))
          : "") +
        " · All day"
      : timeLabel(fields.start, fields.timezone) +
        " – " +
        timeLabel(fields.end, fields.timezone)
    : null;
  return (
    <div
      className="modal-backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget && !saving) onClose();
      }}
    >
      <section
        className="dialog calendar-detail-card"
        ref={card}
        tabIndex={-1}
        role="dialog"
        aria-modal="true"
        aria-labelledby="calendar-detail-title"
      >
        <div className="calendar-detail-toolbar">
          <span className={"calendar-type-badge " + event.kind}>
            {kind.label}
          </span>
          <div>
            <button
              className="icon-button detail-edit"
              aria-label="Edit calendar item"
              title="Edit"
              disabled={!canEdit || saving}
              onClick={() => onEdit(event, task, schedule)}
            >
              <Pencil size={18} />
              <span>Edit</span>
            </button>
            <button
              className="icon-button"
              aria-label="Close calendar details"
              disabled={saving}
              onClick={onClose}
            >
              <X size={20} />
            </button>
          </div>
        </div>
        <h2 id="calendar-detail-title">{title}</h2>
        <p className="footnote">{kind.hint}</p>
        {loading && <p role="status">Loading details…</p>}
        {error && (
          <p className="error-banner" role="alert">
            {error}{" "}
            <button onClick={() => setRetry((n) => n + 1)}>
              Reload details
            </button>
          </p>
        )}
        <div className="detail-time">
          <CalendarDays size={18} />
          <span>
            {appointmentTime ??
              (event.at ? timeLabel(event.at, zone) : dateLabel(event.date))}
          </span>
        </div>
        <p className="footnote">
          {fields?.timezone ?? schedule?.timezone ?? task?.due_timezone ?? zone}
          {!event.at && !local ? " · Date only" : ""}
        </p>
        {fields && (
          <>
            {fields.location && (
              <section>
                <h3>Location</h3>
                <p>{fields.location}</p>
              </section>
            )}
            {fields.description && (
              <section>
                <h3>Details</h3>
                <p className="event-description">{fields.description}</p>
              </section>
            )}
            <p className="footnote">
              {fields.busy ? "Blocks availability" : "Free time"}
              {record?.google_calendar_id
                ? " · Google: " + record.google_state.replaceAll("_", " ")
                : " · Eridani calendar"}
            </p>
            {record?.write_message && (
              <p role="status">{record.write_message}</p>
            )}
          </>
        )}
        {schedule && (
          <dl className="detail-facts">
            <div>
              <dt>Alert status</dt>
              <dd>{schedule.status.replaceAll("_", " ")}</dd>
            </div>
            <div>
              <dt>Repeat</dt>
              <dd>
                {schedule.recurrence
                  ? recurrenceLabel(schedule.recurrence)
                  : "Does not repeat"}
              </dd>
            </div>
            {schedule.next_run_at && (
              <div>
                <dt>Next alert</dt>
                <dd>{timeLabel(schedule.next_run_at, schedule.timezone)}</dd>
              </div>
            )}
          </dl>
        )}
        {task && (
          <section className="calendar-task-details">
            {event.kind !== "task" && (
              <h3>
                <ListTodo size={16} />
                {task.is_template ? "Routine template" : "Linked task"}:{" "}
                {task.title}
              </h3>
            )}
            <dl className="detail-facts">
              <div>
                <dt>Status</dt>
                <dd>{task.status.replaceAll("_", " ")}</dd>
              </div>
              <div>
                <dt>Assignee</dt>
                <dd>
                  {organization.actors.find((a) => a.id === task.assignee_id)
                    ?.name ?? task.assignee}
                </dd>
              </div>
              <div>
                <dt>Priority</dt>
                <dd>
                  {["Normal", "Low", "Medium", "High"][task.priority] ??
                    task.priority}
                </dd>
              </div>
              {task.planned_date && (
                <div>
                  <dt>Planned</dt>
                  <dd>{dateLabel(task.planned_date)}</dd>
                </div>
              )}
              {task.due_date && (
                <div>
                  <dt>Deadline</dt>
                  <dd>
                    {dateLabel(task.due_date)}
                    {task.due_time
                      ? " · " +
                        task.due_time.slice(0, 5) +
                        " · " +
                        task.due_timezone
                      : ""}
                  </dd>
                </div>
              )}
              {!!task.estimate_minutes && (
                <div>
                  <dt>Estimate</dt>
                  <dd>{task.estimate_minutes} minutes</dd>
                </div>
              )}
              {project && (
                <div>
                  <dt>Project</dt>
                  <dd>{project.name}</dd>
                </div>
              )}
              {space && (
                <div>
                  <dt>Space</dt>
                  <dd>{space.name}</dd>
                </div>
              )}
              {area && (
                <div>
                  <dt>Area</dt>
                  <dd>{area.name}</dd>
                </div>
              )}
              {!!goals.length && (
                <div>
                  <dt>Goals</dt>
                  <dd>{goals.map((g) => g.name).join(", ")}</dd>
                </div>
              )}
              {task.work_type && (
                <div>
                  <dt>Work type</dt>
                  <dd>{task.work_type}</dd>
                </div>
              )}
              {!!task.tags.length && (
                <div>
                  <dt>Tags</dt>
                  <dd>{task.tags.map((t) => "#" + t).join(" ")}</dd>
                </div>
              )}
              {task.completed_at && (
                <div>
                  <dt>Completed</dt>
                  <dd>{timeLabel(task.completed_at, zone)}</dd>
                </div>
              )}
            </dl>
            {parent && (
              <section>
                <h3>Parent task</h3>
                <button className="text-button" onClick={() => onTask(parent)}>
                  {parent.title}
                </button>
              </section>
            )}
            {!!children.length && (
              <section>
                <h3>Subtasks</h3>
                {children.map((child) => (
                  <button
                    key={child.id}
                    className="text-button linked-note-detail"
                    onClick={() => onTask(child)}
                  >
                    {child.title} · {child.status.replaceAll("_", " ")}
                  </button>
                ))}
              </section>
            )}
            {!!alerts.length && (
              <section>
                <h3>Task reminders</h3>
                {alerts.map((alert) => (
                  <p key={alert.id}>
                    {alert.title} · {alert.status.replaceAll("_", " ")}
                    {alert.next_run_at
                      ? " · " + timeLabel(alert.next_run_at, alert.timezone)
                      : ""}
                    {alert.recurrence
                      ? " · " + recurrenceLabel(alert.recurrence)
                      : ""}
                  </p>
                ))}
              </section>
            )}
            {task.notes && (
              <section>
                <h3>Task notes</h3>
                <p className="event-description">{task.notes}</p>
              </section>
            )}
            {!!notes.length && (
              <section>
                <h3>Linked notes</h3>
                {notes.map((note) => (
                  <button
                    className="text-button linked-note-detail"
                    key={note.id}
                    onClick={() => onNote(note.id)}
                  >
                    {note.title}
                  </button>
                ))}
              </section>
            )}
            {notesError && (
              <p role="status">
                Linked notes could not load. Reopen this card to retry.
              </p>
            )}
            {task.external?.provider && (
              <p className="footnote">
                {task.external.identifier} ·{" "}
                {task.external.sync_state?.replaceAll("_", " ")}
                {task.external.url &&
                  /^https?:\/\//i.test(task.external.url) && (
                    <a
                      href={task.external.url}
                      target="_blank"
                      rel="noreferrer"
                    >
                      {" "}
                      Open in Linear <ExternalLink size={12} />
                    </a>
                  )}
              </p>
            )}
            {completion && (
              <button
                className="primary complete-calendar-task"
                disabled={saving || loading}
                onClick={() => void complete()}
              >
                <Check size={17} />
                {saving
                  ? "Saving…"
                  : task.status === "completed"
                    ? "Reopen task"
                    : "Complete task"}
              </button>
            )}
            {!completion && (
              <button
                className="secondary compact"
                onClick={() => onTask(task)}
              >
                Open {task.is_template ? "routine template" : "linked task"}
              </button>
            )}
          </section>
        )}
        {event.kind === "routine" && event.projected && (
          <p className="footnote">
            This occurrence becomes completable when its scheduled task is
            created.
          </p>
        )}
      </section>
    </div>
  );
}
