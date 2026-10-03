import { SourceDetails, SourceNotes, useInlineGuard } from "./SourceDetails";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { z } from "zod";
import {
  Bell,
  Check,
  Pencil,
  X,
  CalendarDays,
  FileText,
  ListTodo,
  ExternalLink,
} from "lucide-react";
import { KindIcon } from "./CalendarView";
import "./calendar.css";
import { api } from "./api";
import { useDialogFocus, timeLabel, recurrenceLabel } from "./components";
import { useEditor } from "./editor-control";
import {
  calendarKind,
  clockLabel,
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
  onClose: closeParent,
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
  const annotation=useInlineGuard();
  const onClose=()=>void annotation.flush().then(closeParent).catch(()=>{});
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
    dirty: annotation.dirty,
    busy: loading || saving,
    schema: z.object({}),
    values: summary,
    beforeLeave: annotation.flush,
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
          : "")
      : sameDayRange(fields.start, fields.end, fields.timezone)
    : null;
  const shownZone =
    fields?.timezone ?? schedule?.timezone ?? task?.due_timezone ?? zone;
  const allDay = fields ? fields.all_day : !event.at;
  const human = (value: string) => {
    const text = value.replaceAll("_", " ");
    return text.charAt(0).toUpperCase() + text.slice(1);
  };
  const props: [string, ReactNode][] = [];
  if (fields) {
    props.push(["Availability", fields.busy ? "Busy" : "Free"]);
    props.push([
      "Calendar",
      record?.google_calendar_id
        ? "Google, " + record.google_state.replaceAll("_", " ")
        : "Eridani",
    ]);
  }
  if (schedule) {
    props.push(["Alert status", human(schedule.status)]);
    props.push([
      "Repeat",
      schedule.recurrence ? recurrenceLabel(schedule.recurrence) : "Does not repeat",
    ]);
    if (schedule.next_run_at)
      props.push(["Next alert", timeLabel(schedule.next_run_at, schedule.timezone)]);
  }
  if (task) {
    props.push(["Status", human(task.status)]);
    props.push([
      "Assignee",
      organization.actors.find((a) => a.id === task.assignee_id)?.name ??
        task.assignee,
    ]);
    props.push([
      "Priority",
      ["Normal", "Low", "Medium", "High"][task.priority] ?? task.priority,
    ]);
    if (task.planned_date) props.push(["Planned", dateLabel(task.planned_date)]);
    if (task.due_date)
      props.push([
        "Deadline",
        <>
          {dateLabel(task.due_date)}
          {task.due_time && (
            <>
              {", " + task.due_time.slice(0, 5)}
              <small className="cal-prop-sub">{task.due_timezone}</small>
            </>
          )}
        </>,
      ]);
    if (task.estimate_minutes)
      props.push(["Estimate", task.estimate_minutes + " minutes"]);
    if (project) props.push(["Project", project.name]);
    if (space) props.push(["Space", space.name]);
    if (area) props.push(["Area", area.name]);
    if (goals.length) props.push(["Goals", goals.map((g) => g.name).join(", ")]);
    if (task.work_type) props.push(["Work type", task.work_type]);
    if (task.tags.length)
      props.push([
        "Tags",
        <span className="chip-row">
          {task.tags.map((t) => (
            <span className="chip" key={t}>
              #{t}
            </span>
          ))}
        </span>,
      ]);
    if (task.completed_at)
      props.push(["Completed", timeLabel(task.completed_at, zone)]);
    if (parent)
      props.push([
        "Parent task",
        <button className="text-button cal-prop-link" onClick={() => onTask(parent)}>
          {parent.title}
        </button>,
      ]);
  }
  return (
    <div
      className="modal-backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget && !saving) onClose();
      }}
    >
      <section
        className="dialog cal-dialog calendar-detail-card"
        ref={card}
        tabIndex={-1}
        role="dialog"
        aria-modal="true"
        aria-labelledby="calendar-detail-title"
      >
        <div className="dialog-heading">
          <div className="cal-dialog-heading-text">
            <span className={"chip calendar-type-badge cal-kind-" + event.kind}>
              <KindIcon kind={event.kind} size={13} />
              {kind.label}
            </span>
            <h2 id="calendar-detail-title">{title}</h2>
          </div>
          <div className="cal-dialog-heading-actions">
            <button
              className="btn btn-ghost btn-sm detail-edit"
              aria-label="Edit calendar item"
              title="Edit"
              disabled={!canEdit || saving}
              onClick={() => onEdit(event, task, schedule)}
            >
              <Pencil size={15} aria-hidden="true" />
              <span>Edit</span>
            </button>
            <button
              className="btn-icon"
              aria-label="Close calendar details"
              disabled={saving}
              onClick={onClose}
            >
              <X size={20} />
            </button>
          </div>
        </div>
        <p className="cal-dialog-hint">{kind.hint}</p>
        {loading && (
          <p role="status" className="cal-loading-line">
            Loading details…
          </p>
        )}
        {error && (
          <p className="error-banner" role="alert">
            {error}{" "}
            <button onClick={() => setRetry((n) => n + 1)}>
              Reload details
            </button>
          </p>
        )}
        <div className={"cal-detail-layout" + (props.length ? "" : " single")}>
                      <div className="cal-detail-main">
              <SourceDetails source={event.source}/>
              {local&&<SourceNotes kind="planning" id={event.entity_id} canEdit={canEdit} onGuard={annotation.update}/> }
            <div className="cal-when">
              <CalendarDays size={18} aria-hidden="true" />
              <div>
                {appointmentTime ??
                  (event.at ? timeLabel(event.at, zone) : dateLabel(event.date))}
                <small>
                  {allDay && !fields ? "Date only, " : allDay ? "All day, " : ""}
                  {shownZone}
                </small>
              </div>
            </div>
            {fields?.location && (
              <section className="cal-section">
                <h3>Location</h3>
                <p>{fields.location}</p>
              </section>
            )}
            {fields?.description && (
              <section className="cal-section">
                <h3>Details</h3>
                <p className="cal-prose event-description">{fields.description}</p>
              </section>
            )}
            {record?.write_message && (
              <p role="status" className="cal-notice">
                {record.write_message}
              </p>
            )}
            {task && event.kind !== "task" && (
              <section className="cal-section">
                <h3>{task.is_template ? "Routine template" : "Linked task"}</h3>
                <div className="cal-links">
                  <button onClick={() => onTask(task)}>
                    <ListTodo size={16} aria-hidden="true" />
                    {task.title}
                  </button>
                </div>
              </section>
            )}
            {task?.notes && (
              <section className="cal-section">
                <h3>Task notes</h3>
                <p className="cal-prose event-description">{task.notes}</p>
              </section>
            )}
            {!!children.length && (
              <section className="cal-section">
                <h3>Subtasks</h3>
                <div className="cal-links">
                  {children.map((child) => (
                    <button
                      key={child.id}
                      className="linked-note-detail"
                      onClick={() => onTask(child)}
                    >
                      {child.title}
                      <span className="chip">{human(child.status)}</span>
                    </button>
                  ))}
                </div>
              </section>
            )}
            {!!alerts.length && (
              <section className="cal-section">
                <h3>Task reminders</h3>
                <div className="cal-links">
                  {alerts.map((alert) => (
                    <div key={alert.id}>
                      <Bell size={15} aria-hidden="true" />
                      {alert.title}
                      <span className="chip-row cal-links-meta">
                        {alert.next_run_at && (
                          <span className="chip">
                            {timeLabel(alert.next_run_at, alert.timezone)}
                          </span>
                        )}
                        {alert.recurrence && (
                          <span className="chip">
                            {recurrenceLabel(alert.recurrence)}
                          </span>
                        )}
                        <span className="chip">{human(alert.status)}</span>
                      </span>
                    </div>
                  ))}
                </div>
              </section>
            )}
            {!!notes.length && (
              <section className="cal-section">
                <h3>Linked notes</h3>
                <div className="cal-links">
                  {notes.map((note) => (
                    <button
                      className="linked-note-detail"
                      key={note.id}
                      onClick={() => onNote(note.id)}
                    >
                      <FileText size={15} aria-hidden="true" />
                      {note.title}
                    </button>
                  ))}
                </div>
              </section>
            )}
            {notesError && (
              <p role="status" className="cal-form-hint">
                Linked notes could not load. Reopen this card to retry.
              </p>
            )}
            {task?.external?.provider && (
              <p className="cal-form-hint">
                <span className="chip">{task.external.identifier}</span>{" "}
                <span className="chip">
                  {human(task.external.sync_state ?? "")}
                </span>
                {task.external.url &&
                  /^https?:\/\//i.test(task.external.url) && (
                    <a
                      className="text-button"
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
            {event.kind === "routine" && event.projected && (
              <p className="cal-notice">
                This occurrence becomes completable when its scheduled task is
                created.
              </p>
            )}
          </div>
          {!!props.length && (
            <aside className="cal-aside" aria-label="Properties">
              <dl className="cal-props detail-facts">
                {props.map(([term, value]) => (
                  <div key={term}>
                    <dt>{term}</dt>
                    <dd>{value}</dd>
                  </div>
                ))}
              </dl>
            </aside>
          )}
        </div>
        {task && (
          <div className="dialog-actions">
            {completion ? (
              <>
                <button
                  className="btn btn-ghost"
                  onClick={() => onTask(task)}
                  disabled={saving}
                >
                  Open task
                </button>
                <button
                  className="primary complete-calendar-task"
                  disabled={saving || loading}
                  onClick={() => void complete()}
                >
                  <Check size={17} aria-hidden="true" />
                  {saving
                    ? "Saving…"
                    : task.status === "completed"
                      ? "Reopen task"
                      : "Complete task"}
                </button>
              </>
            ) : (
              <button className="secondary" onClick={() => onTask(task)}>
                Open {task.is_template ? "routine template" : "linked task"}
              </button>
            )}
          </div>
        )}
      </section>
    </div>
  );
}

/** "Fri, Oct 2, 10:00 AM – 11:00 AM" for a same-day range, full stamps otherwise. */
function sameDayRange(start: string, end: string, zone: string) {
  const day = (value: string) =>
    new Intl.DateTimeFormat("en-CA", { timeZone: zone }).format(new Date(value));
  if (day(start) !== day(end))
    return timeLabel(start, zone) + " – " + timeLabel(end, zone);
  const date = new Intl.DateTimeFormat(undefined, {
    timeZone: zone,
    weekday: "short",
    month: "short",
    day: "numeric",
  }).format(new Date(start));
  return date + ", " + clockLabel(start, zone) + " – " + clockLabel(end, zone);
}
