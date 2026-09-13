import { useEffect, useMemo, useState } from "react";
import { CalendarDays, Clock3, Plus, Repeat2, Check } from "lucide-react";
import { CalendarView } from "./CalendarView";
import type { GoogleStatus } from "./GoogleSettings";
import { api } from "./api";
import { TaskRow, timeLabel } from "./components";
import {
  calendarRange,
  type CalendarMode,
  matchesStatus,
  scheduleProject,
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
  calendarMode: CalendarMode;
  onCalendarMode: (mode: CalendarMode) => void;
  onCreateEvent: (day: string) => void;
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
    rangeKey: string;
    items: CalendarEntry[];
    truncated: boolean;
    google?: GoogleStatus;
    warnings?: { calendar: string; reason: string }[];
  } | null>(null);
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  const { days, from, end } = useMemo(
    () => calendarRange(p.day, p.calendarMode),
    [p.day, p.calendarMode],
  );
  const rangeKey = from + ":" + end + ":" + p.zone;
  const currentData = calendarData?.rangeKey === rangeKey ? calendarData : null;
  useEffect(() => {
    if (!p.calendar) return;
    let current = true;
    // Preserve rows and page height while refreshing this same date range.
    // A genuinely different range uses its own loading state.
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
        if (current) setCalendarData({ ...data, rangeKey });
      })
      .catch((e) => {
        if (current) setError(e.message);
      });
    return () => {
      current = false;
    };
  }, [
    p.calendar,
    from,
    end,
    p.zone,
    p.tasks,
    p.schedules,
    p.notices,
    retry,
    rangeKey,
  ]);
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
  const events = (currentData?.items ?? []).filter(
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
    ? events
        .filter((e) => p.calendarMode === "week" || e.date === p.day)
        .map((e) => e.entity_id)
    : [
        ...(p.kind !== "reminder" ? tasks.map((t) => t.id) : []),
        ...(p.kind !== "task" ? reminders.map((s) => s.id) : []),
      ];
  const visibleKey = visible.slice(0, 60).join(",");
  useEffect(
    () => p.onVisible(visibleKey ? visibleKey.split(",") : []),
    [visibleKey, p.onVisible],
  );
  const openEntry = (e: CalendarEntry) => {
    if (["google", "event", "block"].includes(e.kind)) {
      p.onGoogleEvent(e);
      return;
    }
    if (e.kind === "routine" && e.notification_id && e.task_id) {
      const task = p.tasks.find((t) => t.id === e.task_id);
      if (task) {
        p.onTask(task);
        return;
      }
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
            {p.calendar ? "Your calendar" : "Tasks and alerts"}
          </span>
        </div>
        <div className="workspace-add">
          {p.calendar && (
            <button
              className="primary compact"
              onClick={() => p.onCreateEvent(p.day)}
            >
              <Plus size={17} /> New event
            </button>
          )}
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
        <CalendarView
          day={p.day}
          today={p.today}
          zone={p.zone}
          mode={p.calendarMode}
          days={days}
          events={events}
          loaded={!!currentData}
          truncated={!!currentData?.truncated}
          warnings={currentData?.warnings}
          google={currentData?.google}
          error={error}
          highlight={p.highlight}
          filtered={!!(query || p.project)}
          onDay={p.onDay}
          onMode={p.onCalendarMode}
          onEntry={openEntry}
          onRetry={() => setRetry((n) => n + 1)}
        />
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
