import {
  CalendarDays,
  ChevronLeft,
  ChevronRight,
  List,
  Columns3,
  ChartNoAxesGantt,
} from "lucide-react";
import type { Task } from "./types";
import { TaskRow } from "./components";
import { shiftDate } from "./workspace";
import {
  dayOffset,
  groupedTasks,
  statuses,
  timelineWindow,
  type WorkLayout,
  type WorkGroup,
  type TimelineSpan,
} from "./work-views";

export function LayoutSwitch({
  value,
  onChange,
  label = "Work view",
}: {
  value: WorkLayout;
  onChange: (v: WorkLayout) => void;
  label?: string;
}) {
  const icons = { list: List, board: Columns3, timeline: ChartNoAxesGantt };
  return (
    <div className="layout-switch" role="group" aria-label={label}>
      {(["list", "board", "timeline"] as const).map((layout) => {
        const Icon = icons[layout];
        return (
          <button
            key={layout}
            aria-pressed={value === layout}
            onClick={() => onChange(layout)}
          >
            <Icon size={15} />
            <span>{layout.charAt(0).toUpperCase() + layout.slice(1)}</span>
          </button>
        );
      })}
    </div>
  );
}
type TaskProps = {
  tasks: Task[];
  allTasks: Task[];
  busy: boolean;
  onOpen: (t: Task) => void;
  onStatus: (t: Task, status: string) => Promise<unknown>;
  highlight?: string | null;
  group: WorkGroup;
};
export function TaskBoard(p: TaskProps) {
  const groups = groupedTasks(p.tasks, p.group);
  return (
    <div className="board-scroll" aria-label="Task board" tabIndex={0}>
      <div className="task-board">
        {groups.map((group) => (
          <section
            className="board-column"
            key={group.key}
            onDragOver={(e) => {
              if (p.group === "status" && !p.busy) e.preventDefault();
            }}
            onDrop={(e) => {
              e.preventDefault();
              if (p.group !== "status" || p.busy) return;
              const task = p.tasks.find(
                (t) =>
                  t.id === e.dataTransfer.getData("application/eridani-task"),
              );
              if (task && task.status !== group.key)
                void p.onStatus(task, group.key);
            }}
          >
            <h3>
              {group.label}
              <span>{group.tasks.length}</span>
            </h3>
            {group.tasks.map((task) => (
              <article
                className={
                  "board-card " +
                  (p.highlight === task.id ? "record-highlight" : "")
                }
                id={"record-" + task.id}
                key={task.id}
                draggable={!p.busy && p.group === "status" && !task.is_template}
                onDragStart={(e) =>
                  e.dataTransfer.setData("application/eridani-task", task.id)
                }
              >
                <button className="board-title" onClick={() => p.onOpen(task)}>
                  {task.title}
                </button>
                <div className="board-meta">
                  {task.project && <span>{task.project}</span>}
                  {task.priority > 0 && (
                    <span className={"priority p" + task.priority}>
                      {"!".repeat(task.priority)}
                    </span>
                  )}
                  {task.assignee !== "owner" && <span>{task.assignee}</span>}
                  {task.due_date && (
                    <span
                      title={
                        "Deadline" +
                        (task.due_timezone ? " · " + task.due_timezone : "")
                      }
                    >
                      <CalendarDays size={12} />
                      {task.due_date.slice(5)}
                      {task.due_time ? " " + task.due_time.slice(0, 5) : ""}
                    </span>
                  )}
                  {task.planned_date && (
                    <span>Plan {task.planned_date.slice(5)}</span>
                  )}
                  {task.parent_task_id && (
                    <span
                      title={
                        p.allTasks.find((t) => t.id === task.parent_task_id)
                          ?.title
                      }
                    >
                      ↳ Subtask
                    </span>
                  )}
                  {!!task.tags.length && (
                    <span>
                      {task.tags
                        .slice(0, 3)
                        .map((t) => "#" + t)
                        .join(" ")}
                    </span>
                  )}
                </div>
                <select
                  aria-label={"Status for " + task.title}
                  value={task.status}
                  disabled={p.busy || task.is_template}
                  onChange={(e) => void p.onStatus(task, e.target.value)}
                >
                  {statuses.map((status) => (
                    <option key={status} value={status}>
                      {status.replaceAll("_", " ")}
                    </option>
                  ))}
                </select>
              </article>
            ))}
            {!group.tasks.length && <p className="board-empty">No tasks</p>}
          </section>
        ))}
      </div>
    </div>
  );
}
export type TimelineItem = {
  id: string;
  title: string;
  start: string | null;
  end: string | null;
  status?: string;
  subtitle?: string;
  range?: boolean;
};
export function Timeline({
  items,
  start,
  span,
  onStart,
  onSpan,
  onOpen,
  today,
  label = "Task timeline",
}: {
  items: TimelineItem[];
  start: string;
  span: TimelineSpan;
  onStart: (date: string) => void;
  onSpan: (span: TimelineSpan) => void;
  onOpen: (id: string) => void;
  today: string;
  label?: string;
}) {
  const days = timelineWindow(start, span),
    end = days.at(-1)!;
  const undated = items.filter((item) => !item.start && !item.end);
  const dated = items.filter((item) => item.start || item.end);
  const visible = dated.filter((item) => {
    if (!item.range)
      return [item.start, item.end].some(
        (date) => date && date >= start && date <= end,
      );
    const first =
      item.start && item.end
        ? item.start < item.end
          ? item.start
          : item.end
        : (item.start ?? item.end!);
    const last =
      item.start && item.end
        ? item.start > item.end
          ? item.start
          : item.end
        : (item.start ?? item.end!);
    return first <= end && last >= start;
  });
  const outside = dated.filter((item) => !visible.includes(item));
  const cell = span === 90 ? 22 : 34;
  return (
    <section className="timeline-view" aria-label={label}>
      <div className="timeline-controls">
        <div className="calendar-navigation">
          <button
            className="icon-button"
            aria-label="Previous timeline period"
            onClick={() => onStart(shiftDate(start, -span))}
          >
            <ChevronLeft size={17} />
          </button>
          <button className="secondary compact" onClick={() => onStart(today)}>
            Today
          </button>
          <button
            className="icon-button"
            aria-label="Next timeline period"
            onClick={() => onStart(shiftDate(start, span))}
          >
            <ChevronRight size={17} />
          </button>
        </div>
        <span>
          {start} – {end}
        </span>
        <select
          aria-label="Timeline range"
          value={span}
          onChange={(e) => onSpan(Number(e.target.value) as TimelineSpan)}
        >
          <option value={14}>2 weeks</option>
          <option value={30}>Month</option>
          <option value={90}>Quarter</option>
        </select>
      </div>
      <p className="timeline-legend">
        {items.some((item) => item.range)
          ? "Bars: project start to target. Single dates are milestones."
          : "● Planned work · ◆ Deadline. Dates do not reserve calendar time."}
      </p>
      <div
        className="timeline-scroll"
        tabIndex={0}
        role="region"
        aria-label={label + " dates"}
      >
        <div
          className="timeline-grid"
          style={
            {
              "--timeline-width": span * cell + "px",
              "--day-width": cell + "px",
            } as React.CSSProperties
          }
        >
          <div className="timeline-header">
            <strong>{label.startsWith("Project") ? "Project" : "Task"}</strong>
            <div className="timeline-days">
              {days.map((date, index) => (
                <span
                  key={date}
                  className={date === today ? "is-today" : ""}
                  title={date}
                >
                  {index === 0 || date.endsWith("-01")
                    ? date.slice(5)
                    : date.slice(-2)}
                </span>
              ))}
            </div>
          </div>
          {visible.map((item) => {
            const a = item.start ? dayOffset(start, item.start) : null,
              b = item.end ? dayOffset(start, item.end) : null;
            const barStart = Math.max(0, a ?? b ?? 0),
              barEnd = Math.min(span - 1, b ?? a ?? 0);
            return (
              <div className="timeline-row" key={item.id}>
                <button
                  className="timeline-name"
                  onClick={() => onOpen(item.id)}
                  title={item.title}
                >
                  <span>{item.title}</span>
                  {item.subtitle && <small>{item.subtitle}</small>}
                </button>
                <div className="timeline-track">
                  {item.range && a !== null && b !== null && b >= a && (
                    <button
                      className={
                        "timeline-bar " +
                        (item.status === "completed" ? "done" : "")
                      }
                      aria-label={
                        item.title + ": " + item.start + " to " + item.end
                      }
                      style={{
                        left: barStart * cell + 4,
                        width: Math.max(14, (barEnd - barStart + 1) * cell - 8),
                      }}
                      onClick={() => onOpen(item.id)}
                    />
                  )}
                  {(!item.range || a === null || b === null) && (
                    <>
                      {a !== null && a >= 0 && a < span && (
                        <button
                          className="timeline-mark planned-mark"
                          aria-label={
                            "Planned " + item.start + ": " + item.title
                          }
                          style={{ left: a * cell }}
                          onClick={() => onOpen(item.id)}
                        >
                          ●
                        </button>
                      )}
                      {b !== null && b >= 0 && b < span && (
                        <button
                          className="timeline-mark deadline-mark"
                          aria-label={
                            "Deadline " + item.end + ": " + item.title
                          }
                          style={{ left: b * cell }}
                          onClick={() => onOpen(item.id)}
                        >
                          ◆
                        </button>
                      )}
                    </>
                  )}
                </div>
              </div>
            );
          })}
          {!visible.length && (
            <p className="compact-empty">No dated work in this range.</p>
          )}
        </div>
      </div>
      {[
        { title: "Unscheduled", rows: undated },
        { title: "Outside this range", rows: outside },
      ].map(
        (group) =>
          !!group.rows.length && (
            <details
              className="timeline-other"
              key={group.title}
              open={group.title === "Unscheduled" || undefined}
            >
              <summary>
                {group.title}
                <span>{group.rows.length}</span>
              </summary>
              <div>
                {group.rows.map((item) => (
                  <button key={item.id} onClick={() => onOpen(item.id)}>
                    <span>{item.title}</span>
                    <small>
                      {[item.start, item.end].filter(Boolean).join(" → ") ||
                        "Set a planned date or deadline"}
                    </small>
                  </button>
                ))}
              </div>
            </details>
          ),
      )}
    </section>
  );
}

export function GroupedTaskList({
  tasks,
  group,
  today,
  busy,
  onOpen,
  onToggle,
}: {
  tasks: Task[];
  group: WorkGroup;
  today: string;
  busy: boolean;
  onOpen: (t: Task) => void;
  onToggle: (t: Task) => void;
}) {
  return (
    <>
      {groupedTasks(tasks, group)
        .filter((g) => g.tasks.length)
        .map((g) => (
          <section className="task-group" key={g.key}>
            <h3>
              {g.label}
              <span>{g.tasks.length}</span>
            </h3>
            {g.tasks.map((task) => (
              <TaskRow
                key={task.id}
                task={task}
                today={today}
                busy={busy}
                onOpen={() => onOpen(task)}
                onToggle={() => onToggle(task)}
              />
            ))}
          </section>
        ))}
    </>
  );
}
