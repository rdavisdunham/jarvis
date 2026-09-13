import {
  matchesOrganization,
  type Organization,
  type OrganizationFilter,
} from "./productivity";
import { useEffect, useMemo, useState } from "react";
import { CalendarDays, Clock3, Plus, Repeat2, Check } from "lucide-react";
import { LayoutSwitch, TaskBoard, Timeline } from "./WorkViews";
import {
  matchesTaskFilters,
  sortTasks,
  type TaskFilters,
  type WorkLayout,
  type WorkSort,
  type WorkGroup,
  type TimelineSpan,
} from "./work-views";
import { CalendarView } from "./CalendarView";
import type { GoogleStatus } from "./GoogleSettings";
import { api } from "./api";
import { TaskRow, timeLabel } from "./components";
import {
  calendarRange,
  type CalendarMode,
  matchesStatus,
  scheduleProject,
  shiftDate,
} from "./workspace";
import type { Task, Schedule, Notice, Project, CalendarEntry } from "./types";

type Props = {
  preset?: "all" | "today" | "inbox" | "week";
  layout: WorkLayout;
  onLayout: (layout: WorkLayout) => void;
  sort: WorkSort;
  group: WorkGroup;
  taskFilters: TaskFilters;
  timelineDate: string;
  timelineSpan: TimelineSpan;
  onTimelineDate: (date: string) => void;
  onTimelineSpan: (span: TimelineSpan) => void;

  organization: Organization;
  organizationFilter: OrganizationFilter;
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
    matchesTaskFilters(t, p.taskFilters) &&
    matchesStatus(t.status, p.status) &&
    (!p.project || t.project === p.project) &&
    (!query ||
      [t.title, t.notes, t.project, t.assignee, t.work_type, ...(t.tags ?? [])]
        .join(" ")
        .toLocaleLowerCase()
        .includes(query));
  const matchedTasks = p.tasks
    .filter((t) => !t.archived)
    .filter((t) =>
      p.preset === "inbox"
        ? !t.project_id && !t.space_id && !t.area_id
        : p.preset === "today"
          ? !!(
              (t.planned_date && t.planned_date <= p.today) ||
              (t.due_date && t.due_date <= p.today)
            )
          : p.preset === "week"
            ? !!(
                (t.planned_date && t.planned_date <= shiftDate(p.today, 6)) ||
                (t.due_date && t.due_date <= shiftDate(p.today, 6))
              )
            : true,
    )
    .filter((t) => matchesOrganization(t, p.organizationFilter, p.organization))
    .filter(
      (t) =>
        taskMatches(t) ||
        (p.kind === "all" &&
          matchesTaskFilters(t, p.taskFilters) &&
          matchesStatus(t.status, p.status) &&
          (!p.project || t.project === p.project) &&
          !!query &&
          p.schedules.some(
            (s) =>
              s.task_id === t.id && s.title.toLocaleLowerCase().includes(query),
          )),
    );
  const tasks = sortTasks(matchedTasks, p.sort);
  const hasTaskFilters = Object.values(p.taskFilters).some(Boolean);
  const reminders = p.schedules
    .filter(
      (s) =>
        (!p.preset || p.preset === "all") &&
        (!hasTaskFilters ||
          !!p.tasks.find(
            (t) => t.id === s.task_id && matchesTaskFilters(t, p.taskFilters),
          )),
    )
    .filter((s) =>
      matchesOrganization(
        p.tasks.find((t) => t.id === s.task_id) ?? { project_id: s.project_id },
        p.organizationFilter,
        p.organization,
      ),
    )
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
      (!hasTaskFilters ||
        !!p.tasks.find(
          (t) =>
            (t.id === e.task_id ||
              (e.kind === "task" && t.id === e.entity_id)) &&
            matchesTaskFilters(t, p.taskFilters),
        )) &&
      matchesOrganization(
        p.tasks.find((t) => t.id === e.task_id) ?? e,
        p.organizationFilter,
        p.organization,
      ) &&
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
        {p.calendar ? (
          <span className="footnote">Calendar</span>
        ) : p.kind === "reminder" ? (
          <span className="footnote">Task alerts</span>
        ) : (
          <LayoutSwitch value={p.layout} onChange={p.onLayout} />
        )}
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
              onClick={() => {
                if (!p.selecting) p.onLayout("list");
                p.onSelecting(!p.selecting);
              }}
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
          filtered={
            !!(
              query ||
              p.project ||
              hasTaskFilters ||
              Object.values(p.organizationFilter).some(Boolean) ||
              p.status !== "all" ||
              p.kind !== "all"
            )
          }
          onDay={p.onDay}
          onMode={p.onCalendarMode}
          onEntry={openEntry}
          onRetry={() => setRetry((n) => n + 1)}
        />
      ) : p.layout === "board" && p.kind !== "reminder" ? (
        <TaskBoard
          tasks={tasks}
          allTasks={p.tasks}
          busy={p.busy}
          group={p.group}
          onOpen={p.onTask}
          highlight={p.highlight}
          onStatus={(task, status) =>
            p.mutate(
              "task.update",
              { task_id: task.id, expected_revision: task.revision, status },
              "Task status updated",
            )
          }
        />
      ) : p.layout === "timeline" && p.kind !== "reminder" ? (
        <Timeline
          items={tasks.map((t) => ({
            id: t.id,
            title: t.title,
            start: t.planned_date ?? null,
            end: t.due_date,
            subtitle: t.project ?? t.status.replaceAll("_", " "),
          }))}
          start={p.timelineDate}
          span={p.timelineSpan}
          onStart={p.onTimelineDate}
          onSpan={p.onTimelineSpan}
          today={p.today}
          onOpen={(id) => {
            const task = tasks.find((t) => t.id === id);
            if (task) p.onTask(task);
          }}
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
