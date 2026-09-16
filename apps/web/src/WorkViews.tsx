import { useEffect, useRef, useState, type RefObject } from "react";
import { humanLabel } from "./ux";
import { createPortal } from "react-dom";
import { useBoardDrag } from "./use-board-drag";
import {
  GripVertical,
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
export function BoardNavigator({root, groups, label, compact, onCompact}: {
  root: RefObject<HTMLDivElement | null>; groups: {key: string; label: string; count: number}[];
  label: string; compact: boolean; onCompact: (value: boolean) => void;
}) {
  const [active, setActive] = useState("");
  const signature = groups.map(g => g.key).join("|");
  const restored = useRef("");
  const go = (key: string) => {
    const columns = Array.from(root.current?.querySelectorAll<HTMLElement>("[data-board-key]") ?? []);
    const column = columns.find(c => c.dataset.boardKey === key);
    if (!column || !root.current) return;
    root.current.scrollTo({left: root.current.scrollLeft + column.getBoundingClientRect().left - root.current.getBoundingClientRect().left, behavior: "instant"});
    setActive(key); sessionStorage.setItem("eri-board:" + label, key);
  };
  useEffect(() => {
    if (!root.current || restored.current === signature) return;
    restored.current = signature;
    const saved = sessionStorage.getItem("eri-board:" + label);
    const initial = groups.find(g => g.key === saved) ?? groups.find(g => ["open", "active"].includes(g.key)) ?? groups.find(g => g.count) ?? groups[0];
    if (initial) go(initial.key);
  }, [signature, label]);
  useEffect(() => {
    const node = root.current;
    const update = () => {
      if (!node) return;
      const columns = Array.from(node.querySelectorAll<HTMLElement>("[data-board-key]"));
      const edge = node.getBoundingClientRect().left;
      const nearest = columns.sort((a,b) => Math.abs(a.getBoundingClientRect().left-edge) - Math.abs(b.getBoundingClientRect().left-edge))[0];
      if (nearest) {const key = nearest.dataset.boardKey!; setActive(key); sessionStorage.setItem("eri-board:" + label, key);}
    };
    node?.addEventListener("scroll", update, {passive:true});
    return () => node?.removeEventListener("scroll", update);
  }, [signature, label]);
  return <div className="board-navigation"><label>Column<select aria-label={label + " column"} value={groups.some(g => g.key === active) ? active : groups[0]?.key ?? ""} onChange={e => go(e.target.value)}>{groups.map(g => <option key={g.key} value={g.key}>{g.label} · {g.count}</option>)}</select></label>
    <label className="inline-check"><input type="checkbox" checked={compact} onChange={e => onCompact(e.target.checked)}/>Hide empty & finished</label>
  </div>;
}

type TaskProps = {
  tasks: Task[];
  allTasks: Task[];
  busy: boolean;
  onOpen: (t: Task) => void;
  onStatus: (t: Task, status: string) => Promise<unknown>;
  onMove: (t: Task, key: string) => Promise<unknown>;
  highlight?: string | null;
  group: WorkGroup;
};
export function TaskBoard(p: TaskProps) {
  const [compact, setCompact] = useState(false);
  const allGroups = groupedTasks(p.tasks, p.group);
  const visibleGroups = allGroups.filter(g => !compact || (g.tasks.length > 0 && !(p.group === "status" && ["completed", "cancelled"].includes(g.key))));
  const groups = visibleGroups.length ? visibleGroups : allGroups.filter(g => g.key === "open");
  const keyFor = (task: Task) =>
    p.group === "status"
      ? task.status
      : p.group === "project"
        ? (task.project_id ?? "")
        : (task.assignee_id ?? task.assignee);
  const move = (task: Task, key: string) =>
    keyFor(task) === key
      ? Promise.resolve({ unchanged: true })
      : p.onMove(task, key);
  const { root, drag, handle, message, saving } = useBoardDrag({
    items: p.tasks,
    keys: groups.map((g) => g.key),
    disabled: p.busy,
    move,
  });
  return (
    <>
      <BoardNavigator root={root} label={"Task board " + p.group} groups={groups.map(g => ({...g, count:g.tasks.length}))} compact={compact} onCompact={setCompact}/>
      <details className="board-help"><summary>Move cards</summary><p className="board-drag-help" id="board-drag-help">
        Drag the grip to change {p.group}. Keyboard: Space, left/right, Space.
        Your sort sets card order.
      </p></details>
      <p className="sr-only" role="status" aria-live="polite">
        {message}
      </p>
      <div
        ref={root}
        className="board-scroll"
        aria-label="Task board"
        tabIndex={0}
      >
        <div className="task-board">
          {groups.map((group) => (
            <section
              className={
                "board-column " +
                (drag?.over === group.key ? "board-drop-target" : "")
              }
              data-board-key={group.key}
              data-board-label={group.label}
              key={group.key}
              onDragOver={(e) => {
                if (!p.busy && !saving) e.preventDefault();
              }}
              onDrop={(e) => {
                e.preventDefault();
                if (p.busy || saving) return;
                const task = p.tasks.find(
                  (t) =>
                    t.id === e.dataTransfer.getData("application/eridani-task"),
                );
                if (task && !task.is_template && keyFor(task) !== group.key)
                  void move(task, group.key);
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
                    (p.highlight === task.id ? "record-highlight " : "") +
                    (drag?.id === task.id ? "board-card-dragging" : "")
                  }
                  id={"record-" + task.id}
                  key={task.id}
                  draggable={!p.busy && !saving && !task.is_template}
                  onDragStart={(e) =>
                    e.dataTransfer.setData("application/eridani-task", task.id)
                  }
                >
                  <div className="board-card-heading">
                    <button
                      className="board-title"
                      onClick={() => p.onOpen(task)}
                    >
                      {task.title}
                    </button>
                    {!task.is_template && (
                      <button
                        type="button"
                        className="board-drag-handle"
                        {...handle(task, group.key)}
                        onDragStart={(e) => {
                          e.preventDefault();
                          e.stopPropagation();
                        }}
                      >
                        <GripVertical size={17} />
                      </button>
                    )}
                  </div>
                  <div className="board-meta">
                    {task.project && <span>{task.project}</span>}
                    {task.priority > 0 && (
                      <span className={"priority p" + task.priority}>
                        {"!".repeat(task.priority)}
                      </span>
                    )}
                    {task.assignee !== "owner" && <span>{humanLabel(task.assignee)}</span>}
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
                    disabled={p.busy || saving || task.is_template}
                    onChange={(e) => void p.onStatus(task, e.target.value)}
                  >
                    {statuses.map((status) => (
                      <option key={status} value={status}>
                        {humanLabel(status)}
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
      {drag &&
        !drag.keyboard &&
        createPortal(
          <div
            className="board-drag-preview"
            style={{ left: drag.x + 12, top: drag.y + 12 }}
          >
            {drag.title}
          </div>,
          document.body,
        )}
    </>
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
                {group.title === "Unscheduled" ? "Unscheduled · no planned day or deadline" : group.title}
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
