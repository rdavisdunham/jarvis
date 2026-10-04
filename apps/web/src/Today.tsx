import { useEffect, useId, useRef, useState, type FormEvent } from "react";
import { Bell, Check, CircleCheck, Flag, FolderKanban, Plus, Repeat2 } from "lucide-react";
import { api } from "./api";
import type { CalendarEntry, Schedule, Task } from "./types";
import type { NoteRecord } from "./Notes";
import { notePreview } from "./note-preview";
import { scheduledBy } from "./productivity";
import { clockLabel, dueBucket, shortDate } from "./work-views";
import { shiftDate } from "./workspace";
import { plural, priorityLabels } from "./ux";
import { TodayReviews } from "./Reviews";
import "./today.css";

type Props = {
  tasks: Task[];
  schedules: Schedule[];
  today: string;
  zone: string;
  busy: boolean;
  canEdit?: boolean;
  /** Bumps when notes change so recent notes refresh. */
  noteRevision: number;
  quick: string;
  onQuick: (value: string) => void;
  onAdd: (event: FormEvent) => void;
  onTask: (task: Task) => void;
  onToggle: (task: Task) => void;
  onEntry: (entry: CalendarEntry) => void;
  onNote: (id: string) => void;
  onNavigate: (target: "tasks-today" | "inbox" | "calendar" | "notes") => void;
};

const isOpen = (t: Task) => !t.archived && !t.quick_list_parent_id && !t.is_template && !["completed", "cancelled"].includes(t.status);
const eventKinds = new Set(["event", "google", "block"]);

function localParts(zone: string, date = new Date()) {
  const parts = Object.fromEntries(new Intl.DateTimeFormat("en-CA", {
    timeZone: zone, year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hourCycle: "h23",
  }).formatToParts(date).map(p => [p.type, p.value]));
  return { day: `${parts.year}-${parts.month}-${parts.day}`, minutes: Number(parts.hour) * 60 + Number(parts.minute) };
}
function clockIn(value: string, zone: string) {
  return new Intl.DateTimeFormat("en-US", { timeZone: zone, hour: "numeric", minute: "2-digit" })
    .format(new Date(value)).replace(/\s?([AP])M$/, (_, m: string) => " " + m.toLowerCase() + "m");
}
function useMinute() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(new Date()), 60_000);
    return () => window.clearInterval(timer);
  }, []);
  return now;
}

/** Orbit ring: arc = done ÷ (done + still due), star at the current time of day. */
function DayRing({ done, total, minutes }: { done: number; total: number; minutes: number }) {
  const id = useId().replace(/:/g, "");
  const size = 96, stroke = 7, r = (size - stroke) / 2 - 3, c = 2 * Math.PI * r, center = size / 2;
  const share = total ? done / total : 0;
  const angle = (minutes / 1440) * 2 * Math.PI - Math.PI / 2;
  const star = { x: center + r * Math.cos(angle), y: center + r * Math.sin(angle) };
  const label = total ? `${done} of ${total} done today` : "Nothing due today";
  return <div className="day-ring" role="img" aria-label={label}>
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden="true">
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2={size} y2={size} gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="var(--iri-1)" /><stop offset=".4" stopColor="var(--iri-2)" />
          <stop offset=".75" stopColor="var(--iri-3)" /><stop offset="1" stopColor="var(--iri-4)" />
        </linearGradient>
      </defs>
      <circle className="day-ring-track" cx={center} cy={center} r={r} fill="none" strokeWidth={stroke} />
      {share > 0 && <circle cx={center} cy={center} r={r} fill="none" stroke={`url(#${id})`} strokeWidth={stroke} strokeLinecap="round"
        strokeDasharray={`${c * share} ${c}`} transform={`rotate(-90 ${center} ${center})`} />}
      <path className="day-ring-star" transform={`translate(${star.x} ${star.y})`}
        d="M0 -6.5 C0.9 -1.6 1.6 -0.9 6.5 0 C1.6 0.9 0.9 1.6 0 6.5 C-0.9 1.6 -1.6 0.9 -6.5 0 C-1.6 -0.9 -0.9 -1.6 0 -6.5Z" />
    </svg>
    <div className="day-ring-count" aria-hidden="true">
      {total ? <><strong className="tabular">{done}<span>/{total}</span></strong><small>done</small></> : <small>Clear</small>}
    </div>
  </div>;
}

function DueChip({ task, today }: { task: Task; today: string }) {
  if (task.due_date) {
    const bucket = dueBucket(task.due_date, today);
    const label = bucket === "today" && task.due_time ? clockLabel(task.due_time) : shortDate(task.due_date, today);
    return <span className={"chip" + (bucket === "overdue" ? " chip-due-overdue" : bucket === "today" ? " chip-due-today" : "")} title="Deadline">{label}</span>;
  }
  if (task.planned_date) {
    const slipped = task.planned_date < today;
    return <span className={"chip" + (slipped ? " chip-due-overdue" : "")} title="Planned day">
      {slipped ? "Planned " + shortDate(task.planned_date, today).replace(/^Yesterday$/, "yesterday") : "Planned"}</span>;
  }
  return null;
}

export function Today(p: Props) {
  const now = useMinute();
  const { minutes } = localParts(p.zone, now);
  const captureRef = useRef<HTMLInputElement>(null);
  const [entries, setEntries] = useState<CalendarEntry[] | null>(null);
  const [calendarError, setCalendarError] = useState("");
  const [notes, setNotes] = useState<NoteRecord[] | null>(null);
  useEffect(() => {
    let live = true;
    setCalendarError("");
    api<{ items: CalendarEntry[] }>(`/calendar?start=${p.today}&end=${shiftDate(p.today, 1)}&timezone=${encodeURIComponent(p.zone)}`)
      .then(data => { if (live) setEntries(data.items); })
      .catch(e => { if (live) setCalendarError(e.message); });
    return () => { live = false; };
  }, [p.today, p.zone, p.tasks, p.schedules]);
  useEffect(() => {
    let live = true;
    api<{ items: NoteRecord[] }>("/notes?limit=3")
      .then(data => { if (live) setNotes(data.items); })
      .catch(() => { if (live) setNotes([]); });
    return () => { live = false; };
  }, [p.noteRevision]);

  // ---- Derived lists -------------------------------------------------------------------
  const due = p.tasks.filter(t => isOpen(t) && scheduledBy(t, p.today)).sort((a, b) =>
    (a.due_date ?? a.planned_date ?? "9999").localeCompare(b.due_date ?? b.planned_date ?? "9999") ||
    (a.due_time ?? "99:99").localeCompare(b.due_time ?? "99:99") || b.priority - a.priority || a.title.localeCompare(b.title));
  const dueIds = new Set(due.map(t => t.id));
  const inbox = p.tasks.filter(t => isOpen(t) && !dueIds.has(t.id) && !t.project_id && !t.space_id && !t.area_id)
    .sort((a, b) => b.priority - a.priority || b.created_at.localeCompare(a.created_at));
  const doneToday = p.tasks.filter(t => t.status === "completed" && t.completed_at && localParts(p.zone, new Date(t.completed_at)).day === p.today).length;
  const dueToday = due.filter(t => t.due_date === p.today).length;
  const overdue = due.filter(t => t.due_date && t.due_date < p.today).length;
  const timed = (entries ?? []).filter(e => e.date === p.today && e.at && !(e.kind === "task" && e.timing === "planned"))
    .sort((a, b) => a.at!.localeCompare(b.at!));
  const allDay = (entries ?? []).filter(e => e.date === p.today && !e.at && eventKinds.has(e.kind));
  const events = [...timed, ...allDay].filter(e => eventKinds.has(e.kind)).length;
  const nowIndex = timed.findIndex(e => Date.parse(e.at!) > now.getTime());
  const summary = [dueToday && `${dueToday} due today`, overdue && `${overdue} overdue`, events && plural(events, "event")].filter(Boolean).join(", ");
  const heading = new Date(p.today + "T12:00:00Z").toLocaleDateString("en-US", { timeZone: "UTC", weekday: "long", month: "long", day: "numeric" });

  const taskRow = (task: Task, withDue: boolean) => <li key={task.id} className="row today-row">
    <button className="check" aria-label={"Complete " + task.title} disabled={p.busy} onClick={() => p.onToggle(task)}><Check aria-hidden="true" /></button>
    <div className="row-main">
      <button className="row-title" onClick={() => p.onTask(task)}>{task.title}</button>
      <div className="row-meta">
        {task.project && <span className="chip chip-home"><FolderKanban aria-hidden="true" />{task.project}</span>}
        {task.priority >= 2 && <span className={"chip chip-priority p" + task.priority} title="Priority"><Flag aria-hidden="true" />{priorityLabels[task.priority]}</span>}
        {withDue && <DueChip task={task} today={p.today} />}
      </div>
    </div>
  </li>;
  const glyph = (e: CalendarEntry) => e.kind === "task" ? <CircleCheck size={15} /> : e.kind === "reminder" ? <Bell size={15} />
    : e.kind === "routine" ? <Repeat2 size={15} /> : <span className={"sched-dot" + (e.kind === "block" ? " is-block" : "")} />;
  const scheduleRow = (e: CalendarEntry) => {
    const done = e.status === "completed";
    return <li key={e.id} className={"sched-row" + (done ? " done" : "") + (eventKinds.has(e.kind) ? " is-event" : "")}>
      <button className="sched-button" onClick={() => p.onEntry(e)}>
        <span className="sched-time tabular">{e.at ? clockIn(e.at, p.zone) : "All day"}</span>
        <span className="sched-glyph" aria-hidden="true">{glyph(e)}</span>
        <span className="sched-title">{e.title}</span>
        {e.end_at && e.at && <span className="sched-end tabular">until {clockIn(e.end_at, p.zone)}</span>}
      </button>
    </li>;
  };
  const nowLine = <li key="now" className="sched-now" aria-label={"Now, " + clockLabel(`${Math.floor(minutes / 60)}:${minutes % 60}`)}>
    <span className="sched-time tabular">{clockLabel(`${Math.floor(minutes / 60)}:${minutes % 60}`)}</span><span className="sched-now-line" aria-hidden="true" />
  </li>;

  return <section className="today" aria-label="Today">
    <header className="today-head">
      <div className="today-intro">
        <h1 className="today-date">{heading}</h1>
        <p className="today-summary">{summary ? summary + "." : "Nothing due today."}</p>
      </div>
      <DayRing done={doneToday} total={doneToday + due.length} minutes={minutes} />
    </header>
    <form className="today-capture" onSubmit={p.onAdd}>
      <Plus size={18} aria-hidden="true" />
      <input ref={captureRef} value={p.quick} onChange={e => p.onQuick(e.target.value)} aria-label="New task" placeholder="Add a task for today…" maxLength={500} />
      <button className="btn btn-primary btn-sm" type="submit" disabled={!p.quick.trim()}>Add</button>
    </form>
    <div className="today-grid">
      <section className="panel today-schedule" aria-labelledby="today-schedule-title">
        <header className="panel-header"><h2 id="today-schedule-title">Schedule</h2>{!!timed.length && <span className="panel-count">{timed.length}</span>}
          <button type="button" className="btn btn-ghost btn-sm panel-action" onClick={() => p.onNavigate("calendar")}>Calendar</button></header>
        {calendarError ? <p className="today-error" role="alert">{calendarError}</p>
          : entries === null ? <p className="today-loading" role="status">Loading your schedule…</p>
          : timed.length || allDay.length ? <ol className="sched-list">
            {allDay.map(scheduleRow)}
            {timed.map((e, i) => i === nowIndex ? [nowLine, scheduleRow(e)] : scheduleRow(e))}
            {nowIndex === -1 && !!timed.length && nowLine}
          </ol>
          : <div className="empty"><span>Nothing on your calendar today.</span><button type="button" className="btn btn-soft" onClick={() => p.onNavigate("calendar")}>Open calendar</button></div>}
      </section>
      <div className="today-side">
        <section className="panel" aria-labelledby="today-due-title">
          <header className="panel-header"><h2 id="today-due-title">Due and overdue</h2>{!!due.length && <span className="panel-count">{due.length}</span>}
            {due.length > 6 && <button type="button" className="btn btn-ghost btn-sm panel-action" onClick={() => p.onNavigate("tasks-today")}>See all</button>}</header>
          {due.length ? <ul className="today-rows">{due.slice(0, 6).map(t => taskRow(t, true))}</ul>
            : <div className="empty"><span>Nothing is due. Plan a task for today above.</span><button type="button" className="btn btn-soft" onClick={() => captureRef.current?.focus()}>Add a task</button></div>}
        </section>
        <TodayReviews today={p.today} zone={p.zone} canEdit={p.canEdit !== false} refresh={p.tasks}/>
        <section className="panel" aria-labelledby="today-inbox-title">
          <header className="panel-header"><h2 id="today-inbox-title">Inbox</h2>{!!inbox.length && <span className="panel-count">{inbox.length}</span>}
            <button type="button" className="btn btn-ghost btn-sm panel-action" onClick={() => p.onNavigate("inbox")}>Open inbox</button></header>
          {inbox.length ? <ul className="today-rows">{inbox.slice(0, 5).map(t => taskRow(t, false))}</ul>
            : <div className="empty"><span>Your inbox is clear. Unfiled tasks land here.</span><button type="button" className="btn btn-soft" onClick={() => p.onNavigate("inbox")}>Open inbox</button></div>}
        </section>
      </div>
    </div>
    {notes !== null && <section className="today-notes" aria-labelledby="today-notes-title">
      <header className="panel-header"><h2 id="today-notes-title">Recent notes</h2>
        <button type="button" className="btn btn-ghost btn-sm panel-action" onClick={() => p.onNavigate("notes")}>All notes</button></header>
      {notes.length ? <div className="today-note-grid">{notes.map(n => <button type="button" key={n.id} className="note-card today-note" onClick={() => p.onNote(n.id)}>
        <span className="note-card-title">{n.title || "Untitled note"}</span>
        {notePreview(n) && <span className="note-card-preview">{notePreview(n)}</span>}
        <span className="note-card-foot">
          {n.tags.slice(0, 3).map(tag => <span key={tag} className="chip">{tag}</span>)}
          <span className="note-card-date tabular">{shortDate(localParts(p.zone, new Date(n.updated_at)).day, p.today)}</span>
        </span>
      </button>)}</div>
        : <div className="empty"><span>Notes you write or save from Eri show up here.</span><button type="button" className="btn btn-soft" onClick={() => p.onNavigate("notes")}>Open notes</button></div>}
    </section>}
  </section>;
}
