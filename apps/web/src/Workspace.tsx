import { useEffect, useMemo, useState } from "react";
import {
  CalendarDays,
  ChevronLeft,
  ChevronRight,
  Clock3,
  Plus,
  Repeat2,
  Check,
} from "lucide-react";
import { Availability } from "./Availability";
import type { GoogleStatus } from "./GoogleSettings";
import { api } from "./api";
import { TaskRow, timeLabel } from "./components";
import {
  dateKey,
  matchesStatus,
  monthDays,
  scheduleProject,
  shiftMonth,
} from "./workspace";
import type { Task, Schedule, Notice, Project, CalendarEntry } from "./types";

type Props = {
  onGoogleEvent: (event: CalendarEntry) => void;
  selecting: boolean;
  selectedIds: string[];
  onSelecting: (value: boolean) => void;
  onSelection: (ids: string[]) => void;
  onBulk: () => void;
  highlight?: string | null;
  calendar: boolean;
  day: string;
  onDay: (day: string) => void;
  today: string;
  tasks: Task[];
  schedules: Schedule[];
  notices: Notice[];
  projects: Project[];
  zone: string;
  query: string;
  status: string;
  project: string;
  kind: string;
  busy: boolean;
  onTask: (task: Task) => void;
  onSchedule: (schedule: Schedule) => void;
  toggle: (task: Task) => void;
  createTask: (date?: string) => void;
  createReminder: (date?: string) => void;
  mutate: (tool: string, args: unknown, message: string) => Promise<unknown>;
  onVisible: (ids: string[]) => void;
};
export function Workspace(p: Props) {
  const [calendarData, setCalendarData] = useState<{
    items: CalendarEntry[];
    truncated: boolean;
    google?: GoogleStatus;
  } | null>(null);
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  const days = useMemo(() => monthDays(p.day), [p.day.slice(0, 7)]);
  const from = days[0],
    end = dateKey(
      new Date(new Date(days[41] + "T12:00:00Z").getTime() + 86400000),
    );
  useEffect(() => {
    if (!p.calendar) return;
    let current = true;
    setCalendarData(null);
    setError("");
    api<{ items: CalendarEntry[]; truncated: boolean; google?: GoogleStatus }>(
      "/calendar?start=" +
        from +
        "&end=" +
        end +
        "&timezone=" +
        encodeURIComponent(p.zone),
    )
      .then((data) => {
        if (current) setCalendarData(data);
      })
      .catch((e) => {
        if (current) setError(e.message);
      });
    return () => {
      current = false;
    };
  }, [p.calendar, from, end, p.zone, p.tasks, p.schedules, p.notices, retry]);
  const query = p.query.trim().toLocaleLowerCase();
  const selectedProject = p.projects.find(
    (project) => project.name === p.project,
  )?.id;
  const taskMatches = (t: Task) =>
    matchesStatus(t.status, p.status) &&
    (!p.project || t.project === p.project) &&
    (!query ||
      [t.title, t.notes, t.project, t.assignee, t.work_type, ...(t.tags ?? [])]
        .join(" ")
        .toLocaleLowerCase()
        .includes(query));
  const tasks = p.tasks
    .filter(
      (t) =>
        taskMatches(t) ||
        (p.kind === "all" &&
          matchesStatus(t.status, p.status) &&
          (!p.project || t.project === p.project) &&
          !!query &&
          p.schedules.some(
            (s) =>
              s.task_id === t.id && s.title.toLocaleLowerCase().includes(query),
          )),
    )
    .sort(
      (a, b) =>
        b.priority - a.priority ||
        (a.due_date ?? "9999").localeCompare(b.due_date ?? "9999") ||
        (a.due_time ?? "").localeCompare(b.due_time ?? "") ||
        a.title.localeCompare(b.title),
    );
  const reminders = p.schedules
    .filter(
      (s) =>
        (!s.task_id || p.kind === "reminder") &&
        matchesStatus(s.status, p.status) &&
        (!p.project || scheduleProject(s, p.tasks, p.projects) === p.project) &&
        (!query ||
          [s.title, scheduleProject(s, p.tasks, p.projects)]
            .join(" ")
            .toLocaleLowerCase()
            .includes(query)),
    )
    .sort((a, b) =>
      (a.next_run_at ?? a.anchor_at).localeCompare(
        b.next_run_at ?? b.anchor_at,
      ),
    );
  const events = (calendarData?.items ?? []).filter(
    (e) =>
      matchesStatus(e.status, p.status) &&
      (p.kind === "all" ||
        (p.kind === "task"
          ? e.kind === "task"
          : e.kind === "reminder" || e.kind === "routine")) &&
      (!p.project || e.project_id === selectedProject) &&
      (!query || e.title.toLocaleLowerCase().includes(query)),
  );
  const visible = p.calendar
    ? events.filter((e) => e.date === p.day).map((e) => e.entity_id)
    : [
        ...(p.kind !== "reminder" ? tasks.map((t) => t.id) : []),
        ...(p.kind !== "task" ? reminders.map((s) => s.id) : []),
      ];
  const visibleKey = visible.slice(0, 60).join(",");
  useEffect(
    () => p.onVisible(visibleKey ? visibleKey.split(",") : []),
    [visibleKey, p.onVisible],
  );
  const agenda = events.filter((e) => e.date === p.day);
  const openEntry = (e: CalendarEntry) => {
    if (e.kind === "google") {
      p.onGoogleEvent(e);
      return;
    }
    if (e.kind === "task") {
      const task = p.tasks.find((t) => t.id === e.entity_id);
      if (task) p.onTask(task);
    } else {
      const schedule = p.schedules.find((s) => s.id === e.entity_id);
      if (schedule) p.onSchedule(schedule);
    }
  };
  const reminderRow = (s: Schedule, linked = false) => {
    const outstanding = p.notices.find(
      (n) => n.schedule_id === s.id && !n.completed_at,
    );
    const completed = s.status === "completed";
    return (
      <article
        className={
          "work-reminder " +
          (linked ? "linked-reminder " : "") +
          (completed ? "done" : "") +
          (p.highlight === s.id ? " record-highlight" : "")
        }
        key={s.id}
        id={"record-" + s.id}
        tabIndex={-1}
      >
        <span className="item-icon">
          {s.recurrence ? <Repeat2 size={17} /> : <Clock3 size={17} />}
        </span>
        <button className="task-info" onClick={() => p.onSchedule(s)}>
          <span className="task-title">{s.title}</span>
          <span className="task-meta">
            {s.status === "finished"
              ? "Delivered · waiting for you"
              : s.status === "completed"
                ? "Completed"
                : s.status === "cancelled"
                  ? "Cancelled"
                  : timeLabel(s.next_run_at ?? s.anchor_at, p.zone)}
            {s.recurrence ? " · Repeats" : ""}
            {!linked && scheduleProject(s, p.tasks, p.projects)
              ? " · " + scheduleProject(s, p.tasks, p.projects)
              : ""}
          </span>
        </button>
        {["active", "finished"].includes(s.status) &&
          (!s.recurrence || outstanding) && (
            <button
              className="icon-button"
              disabled={p.busy}
              aria-label={
                (s.recurrence ? "Complete occurrence of " : "Complete ") +
                s.title
              }
              onClick={() =>
                void p.mutate(
                  s.recurrence ? "notification.complete" : "schedule.complete",
                  s.recurrence
                    ? { notification_id: outstanding!.id }
                    : { schedule_id: s.id, expected_revision: s.revision },
                  s.recurrence
                    ? "Occurrence completed · routine continues"
                    : "Reminder completed",
                )
              }
            >
              <Check size={17} />
            </button>
          )}
      </article>
    );
  };
  return (
    <section
      className="work-space-panel"
      aria-label={p.calendar ? "Calendar workspace" : "Work workspace"}
    >
      <div className="workspace-actions">
        <div className="record-tabs">
          <span className="footnote">
            {p.calendar ? "Your calendar" : "Tasks & reminders"}
          </span>
        </div>
        <div className="workspace-add">
          {!p.calendar && p.kind !== "reminder" && (
            <button
              className="text-button"
              onClick={() => p.onSelecting(!p.selecting)}
            >
              {p.selecting ? "Done selecting" : "Select tasks"}
            </button>
          )}
          <button
            className="secondary compact"
            onClick={() => p.createReminder(p.calendar ? p.day : undefined)}
          >
            <Clock3 size={15} />
            Reminder
          </button>
          <button
            className="primary compact"
            onClick={() => p.createTask(p.calendar ? p.day : undefined)}
          >
            <Plus size={16} />
            New task
          </button>
        </div>
      </div>
      {p.selecting && !p.calendar && (
        <div className="bulk-toolbar">
          <label>
            <input
              type="checkbox"
              aria-label="Select visible tasks"
              checked={
                !!tasks.length &&
                tasks.slice(0, 100).every((t) => p.selectedIds.includes(t.id))
              }
              onChange={(e) =>
                p.onSelection(
                  e.target.checked ? tasks.slice(0, 100).map((t) => t.id) : [],
                )
              }
            />
            Select visible tasks
          </label>
          <span>{p.selectedIds.length} selected</span>
          <button
            className="secondary compact"
            disabled={!p.selectedIds.length || p.busy}
            onClick={p.onBulk}
          >
            Edit selected
          </button>
          {tasks.length > 100 && (
            <small>Select up to 100 tasks at a time.</small>
          )}
        </div>
      )}
      {p.calendar ? (
        <>
          <div className="calendar-heading">
            <h2>
              {new Date(p.day.slice(0, 7) + "-01T12:00:00Z").toLocaleDateString(
                undefined,
                { month: "long", year: "numeric", timeZone: "UTC" },
              )}
            </h2>
            <div>
              <button
                className="icon-button"
                aria-label="Previous month"
                onClick={() => p.onDay(shiftMonth(p.day, -1))}
              >
                <ChevronLeft size={18} />
              </button>
              <button className="text-button" onClick={() => p.onDay(p.today)}>
                Today
              </button>
              <button
                className="icon-button"
                aria-label="Next month"
                onClick={() => p.onDay(shiftMonth(p.day, 1))}
              >
                <ChevronRight size={18} />
              </button>
            </div>
          </div>
          <p className="footnote calendar-zone">
            {p.zone} · Dates without a time stay all-day.
          </p>
          {calendarData?.google?.calendar_enabled && (
            <p className="footnote calendar-sync-row" role="status">
              {calendarData.google.syncing
                ? "Google calendars are syncing…"
                : calendarData.google.status === "needs_reconnect"
                  ? "Reconnect Google Calendar in Settings."
                  : calendarData.google.stale ||
                      calendarData.google.status === "error"
                    ? "Google events may be out of date. Check the connection in Settings."
                    : "Google synced " +
                      new Date(
                        calendarData.google.last_sync_at!,
                      ).toLocaleString()}
            </p>
          )}
          {error && (
            <p className="error-banner" role="alert">
              {error}{" "}
              <button onClick={() => setRetry((n) => n + 1)}>
                Retry calendar
              </button>
            </p>
          )}
          {!calendarData && !error && (
            <p role="status" className="footnote">
              Loading calendar…
            </p>
          )}
          <div className="calendar-grid" aria-label="Month dates">
            {["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"].map((d) => (
              <div className="calendar-weekday" key={d}>
                {d}
              </div>
            ))}
            {days.map((day) => {
              const entries = events.filter((e) => e.date === day);
              return (
                <button
                  key={day}
                  aria-label={day + ", " + entries.length + " items"}
                  aria-pressed={day === p.day}
                  className={
                    "calendar-day " +
                    (day === p.day ? "selected " : "") +
                    (day === p.today ? "is-today " : "") +
                    (day.slice(0, 7) !== p.day.slice(0, 7) ? "outside" : "")
                  }
                  onClick={() => p.onDay(day)}
                  onKeyDown={(event) => {
                    const step = (
                      {
                        ArrowLeft: -1,
                        ArrowRight: 1,
                        ArrowUp: -7,
                        ArrowDown: 7,
                      } as Record<string, number>
                    )[event.key];
                    if (step) {
                      event.preventDefault();
                      const d = new Date(day + "T12:00:00Z");
                      d.setUTCDate(d.getUTCDate() + step);
                      const next = dateKey(d);
                      p.onDay(next);
                      setTimeout(
                        () =>
                          document
                            .querySelector<HTMLButtonElement>(
                              '[data-day="' + next + '"]',
                            )
                            ?.focus(),
                        0,
                      );
                    }
                  }}
                  data-day={day}
                >
                  <span className="calendar-number">
                    {Number(day.slice(-2))}
                  </span>
                  <span className="calendar-labels">
                    {entries.slice(0, 2).map((e) => (
                      <span
                        key={e.id}
                        className={
                          "calendar-chip " +
                          e.kind +
                          (e.status === "completed" ? " completed" : "")
                        }
                      >
                        {e.title}
                      </span>
                    ))}
                    {entries.length > 2 && (
                      <small>+{entries.length - 2} more</small>
                    )}
                  </span>
                  <span className="calendar-dots" aria-hidden="true">
                    {entries.slice(0, 3).map((e) => (
                      <i className={e.kind} key={e.id} />
                    ))}
                    {entries.length > 3 && <small>+{entries.length - 3}</small>}
                  </span>
                </button>
              );
            })}
          </div>
          {calendarData?.truncated && (
            <p role="status">
              Some items could not be shown. Try a shorter range with Eri, or
              check Google Calendar for the complete schedule.
            </p>
          )}
          <div className="section-head agenda-heading">
            <h2>
              {new Date(p.day + "T12:00:00Z").toLocaleDateString(undefined, {
                weekday: "long",
                month: "short",
                day: "numeric",
                timeZone: "UTC",
              })}
              <span>{agenda.length}</span>
            </h2>
          </div>
          <Availability
            key={p.day + ":" + p.zone}
            day={p.day}
            timezone={p.zone}
            enabled={!!calendarData?.google?.calendar_enabled}
          />
          <div className="calendar-agenda">
            {agenda.map((e) => (
              <button
                key={e.id}
                className={
                  "agenda-row " +
                  (e.status === "completed" ? "done" : "") +
                  (p.highlight === e.entity_id ? " record-highlight" : "")
                }
                id={"record-" + e.entity_id}
                onClick={() => openEntry(e)}
              >
                <span className={"agenda-kind " + e.kind}>
                  {e.kind === "task" ? (
                    <CalendarDays size={17} />
                  ) : e.kind === "routine" ? (
                    <Repeat2 size={17} />
                  ) : (
                    <Clock3 size={17} />
                  )}
                </span>
                <span className="agenda-time">
                  {e.at
                    ? new Intl.DateTimeFormat(undefined, {
                        timeZone: p.zone,
                        hour: "numeric",
                        minute: "2-digit",
                      }).format(new Date(e.at))
                    : "All day"}
                </span>
                <span className="grow">
                  <strong>{e.title}</strong>
                  <small>
                    {e.kind === "google"
                      ? e.calendar_title + " · Google"
                      : e.kind === "task"
                        ? "Task deadline"
                        : e.kind === "routine"
                          ? "Repeating task"
                          : "Reminder"}
                    {e.projected && e.kind !== "google" ? " · Upcoming" : ""}
                    {!!e.conflicts?.length &&
                      " · Deadline during " + e.conflicts.join(", ")}
                    {e.status === "completed" ? " · Completed" : ""}
                    {e.task_id && e.kind !== "task" ? " · Linked to task" : ""}
                  </small>
                </span>
                <ChevronRight size={15} />
              </button>
            ))}
            {calendarData && !agenda.length && (
              <div className="calendar-empty">
                Nothing {query || p.project ? "matching your filters " : ""}
                scheduled for this day.
              </div>
            )}
          </div>
        </>
      ) : (
        <>
          {p.kind !== "reminder" &&
            tasks.map((task) => (
              <div className="work-item" key={task.id}>
                {task.parent_task_id && (
                  <button
                    className="parent-link text-button"
                    onClick={() => {
                      const parent = p.tasks.find(
                        (t) => t.id === task.parent_task_id,
                      );
                      if (parent) p.onTask(parent);
                    }}
                  >
                    ↳{" "}
                    {p.tasks.find((t) => t.id === task.parent_task_id)?.title ??
                      "Parent task"}
                  </button>
                )}
                <TaskRow
                  task={task}
                  selected={p.selectedIds.includes(task.id)}
                  onSelect={
                    p.selecting
                      ? () =>
                          p.onSelection(
                            p.selectedIds.includes(task.id)
                              ? p.selectedIds.filter((id) => id !== task.id)
                              : [...p.selectedIds, task.id].slice(0, 100),
                          )
                      : undefined
                  }
                  today={p.today}
                  busy={p.busy}
                  onToggle={() => p.toggle(task)}
                  onOpen={() => p.onTask(task)}
                />
                {p.schedules
                  .filter(
                    (s) =>
                      s.task_id === task.id &&
                      matchesStatus(s.status, p.status),
                  )
                  .map((s) => reminderRow(s, true))}
              </div>
            ))}
          {p.kind !== "task" && reminders.map((s) => reminderRow(s))}

          {!visible.length && (
            <div className="empty-state">
              <CalendarDays size={28} />
              <h3>Room for what comes next.</h3>
              <p>Add a task or reminder, or adjust your filters.</p>
            </div>
          )}
        </>
      )}
    </section>
  );
}
