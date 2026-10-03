import { SourceBadge } from "./SourceDetails";
import {
  calendarKind,
  clockLabel,
  layoutCascade,
  layoutLanes,
  minutesInZone,
  spansTime,
} from "./calendar-presentation";
import { useEffect, useRef, useState } from "react";
import {
  Bell,
  CalendarDays,
  Check,
  ChevronLeft,
  ChevronRight,
  Repeat2,
  Timer,
} from "lucide-react";
import { CalendarWriteActivity } from "./CalendarWriteActivity";
import { Availability } from "./Availability";
import type { CalendarEntry } from "./types";
import type { GoogleStatus } from "./GoogleSettings";
import {
  shiftDate,
  shiftMonth,
  weekDays,
  type CalendarMode,
} from "./workspace";
import "./calendar.css";

type Props = {
  day: string;
  today: string;
  zone: string;
  mode: CalendarMode;
  days: string[];
  events: CalendarEntry[];
  loaded: boolean;
  truncated: boolean;
  warnings?: { calendar: string; reason: string }[];
  error: string;
  google?: GoogleStatus;
  highlight?: string | null;
  filtered: boolean;
  onDay: (day: string) => void;
  onMode: (mode: CalendarMode) => void;
  onEntry: (event: CalendarEntry) => void;
  onRetry: () => void;
};

/** Task = check, reminder = bell, routine = repeat, work block = timer, event = calendar. */
export function KindIcon({
  kind,
  size = 14,
}: {
  kind: CalendarEntry["kind"];
  size?: number;
}) {
  if (kind === "task") return <Check size={size} aria-hidden="true" />;
  if (kind === "reminder") return <Bell size={size} aria-hidden="true" />;
  if (kind === "routine") return <Repeat2 size={size} aria-hidden="true" />;
  if (kind === "block") return <Timer size={size} aria-hidden="true" />;
  return <CalendarDays size={size} aria-hidden="true" />;
}

const HOUR = 48;
const WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const label = (day: string, options: Intl.DateTimeFormatOptions) =>
  new Date(day + "T12:00:00Z").toLocaleDateString(undefined, {
    ...options,
    timeZone: "UTC",
  });
const sortEntries = (a: CalendarEntry, b: CalendarEntry) =>
  (a.at ? 1 : 0) - (b.at ? 1 : 0) || (a.at ?? "").localeCompare(b.at ?? "");

function useNow() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const timer = setInterval(() => setNow(new Date()), 60_000);
    return () => clearInterval(timer);
  }, []);
  return now;
}

/** The page toolbar's "Filters & sort" and "Actions" disclosures behave as popovers here:
 *  an outside press or choosing a menu item closes them. */
function useToolbarPopovers() {
  useEffect(() => {
    const selector =
      ".view-control-row > .filter-panel[open], .work-space-panel .action-menu[open]";
    const outside = (event: PointerEvent) => {
      const target = event.target as Element | null;
      for (const open of document.querySelectorAll<HTMLDetailsElement>(selector))
        if (!target || !open.contains(target)) open.open = false;
    };
    const choose = (event: MouseEvent) => {
      const item = (event.target as Element | null)?.closest(
        ".work-space-panel .action-menu > div button",
      );
      const menu = item?.closest("details");
      if (menu) menu.open = false;
    };
    document.addEventListener("pointerdown", outside);
    document.addEventListener("click", choose);
    return () => {
      document.removeEventListener("pointerdown", outside);
      document.removeEventListener("click", choose);
    };
  }, []);
}

export function CalendarView(p: Props) {
  useToolbarPopovers();
  const lastTap = useRef({ day: "", time: 0 });
  const week = weekDays(p.day);
  const agendaDays = p.mode === "week" ? week : [p.day];
  const forDay = (day: string) =>
    p.events.filter((e) => e.date === day).sort(sortEntries);
  const move = (amount: number) =>
    p.onDay(
      p.mode === "month"
        ? shiftMonth(p.day, amount)
        : shiftDate(p.day, amount * (p.mode === "week" ? 7 : 1)),
    );
  const openDay = (day: string) => {
    p.onDay(day);
    p.onMode("day");
  };
  const selectDay = (day: string, detail: number) => {
    const time = performance.now();
    if (
      detail > 0 &&
      lastTap.current.day === day &&
      time - lastTap.current.time < 380
    ) {
      p.onMode("day");
      lastTap.current = { day: "", time: 0 };
    } else lastTap.current = { day, time };
    p.onDay(day);
  };
  const title =
    p.mode === "month"
      ? label(p.day, { month: "long", year: "numeric" })
      : p.mode === "week"
        ? label(week[0], { month: "short", day: "numeric" }) +
          " – " +
          label(week[6], { month: "short", day: "numeric", year: "numeric" })
        : label(p.day, { weekday: "long", month: "long", day: "numeric" });
  const sync = p.google?.calendar_enabled
    ? p.google.syncing
      ? "Google calendars are syncing…"
      : p.google.status === "needs_reconnect"
        ? "Reconnect Google Calendar in Settings."
        : p.google.stale || p.google.status === "error"
          ? "Google events may be out of date. Check the connection in Settings."
          : "Google synced " + new Date(p.google.last_sync_at!).toLocaleString()
    : "";

  const agenda = (day: string) => {
    const items = forDay(day);
    const quiet = p.mode === "week" && p.loaded && !items.length;
    return (
      <section
        className={"calendar-day-agenda" + (quiet ? " is-empty" : "")}
        key={day}
        aria-label={"Agenda for " + day}
      >
        <div className="cal-agenda-head">
          <h2>
            {label(day, {
              weekday: "long",
              month: p.mode === "week" ? "short" : "long",
              day: "numeric",
            })}
          </h2>
          {!!items.length && <span className="panel-count">{items.length}</span>}
          {quiet && <span className="cal-agenda-none">Nothing scheduled</span>}
          {p.mode !== "day" && (
            <button
              className="btn btn-ghost btn-sm cal-agenda-open"
              aria-label={"Open day view for " + day}
              onClick={() => openDay(day)}
            >
              Open day
            </button>
          )}
        </div>
        {p.mode !== "week" && (
          <Availability
            key={day + ":" + p.zone}
            day={day}
            timezone={p.zone}
            enabled={p.loaded}
          />
        )}
        {!!items.length && (
          <div className="calendar-agenda">
            {items.map((e) => {
              const kind = calendarKind(e);
              const siblings = e.task_id
                ? items.filter((item) => item.task_id === e.task_id)
                : [];
              return (
                <button
                  key={e.id}
                  className={
                    "agenda-row cal-kind-" +
                    e.kind +
                    (e.status === "completed" ? " done" : "") +
                    (p.highlight === e.entity_id ? " record-highlight" : "")
                  }
                  id={"record-" + e.entity_id}
                  onClick={() => p.onEntry(e)}
                >
                  <span className="agenda-time">
                    {e.at ? (
                      <>
                        <time>{clockLabel(e.at, p.zone)}</time>
                        {spansTime(e) && e.end_at && (
                          <time className="agenda-until">
                            {clockLabel(e.end_at, p.zone)}
                          </time>
                        )}
                      </>
                    ) : (
                      "All day"
                    )}
                  </span>
                  <span className="agenda-kind" aria-hidden="true">
                    <KindIcon kind={e.kind} size={15} />
                  </span>
                  <span className="agenda-main">
                    <strong>{e.title}</strong><SourceBadge source={e.source}/>
                    <span className="chip-row">
                      <span className="chip calendar-type-badge">
                        {kind.label}
                      </span>
                      {e.kind === "google" && e.calendar_title && (
                        <span className="chip">{e.calendar_title}</span>
                      )}
                      {e.projected && e.kind !== "google" && (
                        <span className="chip">Upcoming</span>
                      )}
                      {e.status === "completed" && (
                        <span className="chip chip-done">Completed</span>
                      )}
                      {!!e.conflicts?.length && (
                        <span className="chip chip-due-today">
                          Deadline during {e.conflicts.join(", ")}
                        </span>
                      )}
                      {siblings.length > 1 && (
                        <span className="chip linked-task-schedule">
                          Same task:{" "}
                          {siblings
                            .map((item) => calendarKind(item).label)
                            .filter((v, i, all) => all.indexOf(v) === i)
                            .join(", ")}
                        </span>
                      )}
                    </span>
                  </span>
                </button>
              );
            })}
          </div>
        )}
        {p.loaded && !items.length && p.mode !== "week" && (
          <p className="calendar-empty">
            Nothing {p.filtered ? "matching your filters " : ""}scheduled for
            this day.
          </p>
        )}
      </section>
    );
  };

  return (
    <div className={"cal cal-mode-" + p.mode}>
      <div className="calendar-heading">
        <div className="cal-title-group">
          <h2>{title}</h2>
          <div className="cal-nav">
            <button
              className="btn-icon"
              aria-label={"Previous " + p.mode}
              onClick={() => move(-1)}
            >
              <ChevronLeft size={18} />
            </button>
            <button
              className="btn btn-sm cal-today"
              onClick={() => p.onDay(p.today)}
            >
              Today
            </button>
            <button
              className="btn-icon"
              aria-label={"Next " + p.mode}
              onClick={() => move(1)}
            >
              <ChevronRight size={18} />
            </button>
          </div>
        </div>
        <div className="segmented" role="group" aria-label="Calendar view">
          {(["month", "week", "day"] as const).map((mode) => (
            <button
              key={mode}
              aria-pressed={p.mode === mode}
              onClick={() => p.onMode(mode)}
            >
              {mode[0].toUpperCase() + mode.slice(1)}
            </button>
          ))}
        </div>
      </div>
      <div className="cal-subbar">
        <div className="cal-legend" aria-label="Calendar item types">
          <span>
            <Check size={13} aria-hidden="true" /> Task
          </span>
          <span>
            <Bell size={13} aria-hidden="true" /> Reminder
          </span>
          <span>
            <i className="cal-legend-swatch" aria-hidden="true" /> Event or work
            block
          </span>
        </div>
        <div className="cal-meta">
          <span>{p.zone}</span>
          {p.mode === "month" && (
            <span className="cal-hint">Double-tap a date to open its day</span>
          )}
          {sync && (
            <span className="calendar-sync-row" role="status">
              {sync}
            </span>
          )}
        </div>
      </div>
      {p.error && (
        <p className="error-banner cal-banner" role="alert">
          {p.error} <button onClick={p.onRetry}>Retry calendar</button>
        </p>
      )}
      {!p.loaded && !p.error && (
        <p role="status" className="cal-loading">
          Loading calendar…
        </p>
      )}
      {p.truncated && (
        <p role="status" className="cal-banner cal-warning">
          {p.warnings?.length
            ? "A calendar source needs attention: " +
              p.warnings
                .map((w) => w.calendar + " (" + w.reason.replaceAll("_", " ") + ")")
                .join("; ")
            : "This range contains more items than can be displayed. Switch to Day or Week to see a smaller range."}
        </p>
      )}
      {p.mode === "month" && (
        <div className="calendar-grid" aria-label="Month dates">
          {WEEKDAYS.map((day) => (
            <div className="calendar-weekday" key={day}>
              {day}
            </div>
          ))}
          {p.days.map((day) => {
            const entries = forDay(day);
            const shown = entries.length > 3 ? entries.slice(0, 2) : entries;
            return (
              <div
                key={day}
                data-day={day}
                className={
                  "calendar-day" +
                  (day === p.day ? " selected" : "") +
                  (day === p.today ? " is-today" : "") +
                  (day.slice(0, 7) !== p.day.slice(0, 7) ? " outside" : "")
                }
                onClick={(event) => selectDay(day, event.detail)}
                onDoubleClick={() => openDay(day)}
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
                    const next = shiftDate(day, step);
                    p.onDay(next);
                    setTimeout(
                      () =>
                        document
                          .querySelector<HTMLButtonElement>(
                            '.calendar-number[data-day="' + next + '"]',
                          )
                          ?.focus(),
                      0,
                    );
                  }
                }}
              >
                <button
                  className="calendar-number"
                  data-day={day}
                  aria-label={day + ", " + entries.length + " items"}
                  aria-pressed={day === p.day}
                >
                  {Number(day.slice(-2))}
                </button>
                <span className="cal-pills">
                  {shown.map((e) => (
                    <button
                      key={e.id}
                      type="button"
                      aria-label={calendarKind(e).label + ": " + e.title}
                      title={
                        (e.at ? clockLabel(e.at, p.zone) + "  " : "") +
                        calendarKind(e).label +
                        ": " +
                        e.title
                      }
                      onClick={(click) => {
                        click.stopPropagation();
                        p.onEntry(e);
                      }}
                      className={
                        "cal-pill cal-kind-" +
                        e.kind +
                        (e.status === "completed" ? " completed" : "")
                      }
                    >
                      {!spansTime(e) && <KindIcon kind={e.kind} size={12} />}
                      <span className="cal-pill-name">{e.title}</span><SourceBadge source={e.source}/>
                    </button>
                  ))}
                  {entries.length > shown.length && (
                    <button
                      type="button"
                      className="cal-more"
                      aria-label={
                        "Show " +
                        (entries.length - shown.length) +
                        " more on " +
                        day
                      }
                      onClick={(click) => {
                        click.stopPropagation();
                        p.onDay(day);
                      }}
                    >
                      +{entries.length - shown.length} more
                    </button>
                  )}
                </span>
                <span className="cal-dots" aria-hidden="true">
                  {entries.slice(0, 3).map((e) => (
                    <i key={e.id} className={"cal-kind-" + e.kind} />
                  ))}
                </span>
              </div>
            );
          })}
        </div>
      )}
      {p.mode === "month" && (
        <div className="cal-agenda-panel panel">{agenda(p.day)}</div>
      )}
      {p.mode === "week" && (
        <>
          <HourGrid
            days={week}
            events={p.events}
            zone={p.zone}
            today={p.today}
            selected={p.day}
            loaded={p.loaded}
            onEntry={p.onEntry}
            onOpenDay={openDay}
          />
          <div className="cal-agenda-panel cal-week-agenda panel">
            {agendaDays.map(agenda)}
          </div>
        </>
      )}
      {p.mode === "day" && (
        <div className="cal-day-layout">
          <HourGrid
            days={[p.day]}
            events={p.events}
            zone={p.zone}
            today={p.today}
            selected={p.day}
            loaded={p.loaded}
            onEntry={p.onEntry}
            onOpenDay={openDay}
          />
          <div className="cal-agenda-panel panel">{agenda(p.day)}</div>
        </div>
      )}
      <CalendarWriteActivity />
    </div>
  );
}

function HourGrid({
  days,
  events,
  zone,
  today,
  selected,
  loaded,
  onEntry,
  onOpenDay,
}: {
  days: string[];
  events: CalendarEntry[];
  zone: string;
  today: string;
  selected: string;
  loaded: boolean;
  onEntry: (event: CalendarEntry) => void;
  onOpenDay: (day: string) => void;
}) {
  const scroller = useRef<HTMLDivElement>(null);
  const now = useNow();
  const isWeek = days.length > 1;
  const columns = days.map((day) => {
    const items = events.filter((e) => e.date === day);
    const untimed = items.filter((e) => !e.at || e.all_day).sort(sortEntries);
    const entries =
      items
        .filter((e) => e.at && !e.all_day)
        .map((e) => {
          const start = minutesInZone(e.at!, zone);
          const length =
            spansTime(e) && e.end_at
              ? (new Date(e.end_at).getTime() - new Date(e.at!).getTime()) /
                60000
              : 0;
          return {
            item: e,
            start,
            // Moments (deadlines, alerts) reserve a short slot so they stay legible.
            end: Math.min(1440, start + Math.max(length, 30)),
          };
        });
    // Week columns are narrow, so overlaps cascade; the wide day column keeps side-by-side lanes.
    const timed = isWeek
      ? layoutCascade(entries)
      : layoutLanes(entries).map((p) => ({ ...p, left: (p.lane / p.lanes) * 100, width: 100 / p.lanes, depth: 0, shared: p.lanes > 1, overlaps: p.lanes > 1 }));
    return { day, items, untimed, timed };
  });
  const first = Math.min(
    8 * 60,
    ...columns.flatMap((c) => c.timed.map((t) => t.start)),
  );
  const rangeKey = days[0] + ":" + days.length + ":" + loaded;
  useEffect(() => {
    if (scroller.current)
      scroller.current.scrollTop = Math.max(0, (first / 60 - 0.5) * HOUR);
    // Only re-anchor when the visible range changes, not on every refresh.
  }, [rangeKey]);
  const nowMinutes = minutesInZone(now, zone);
  const hasAllDay = columns.some((c) => c.untimed.length);
  return (
    <div
      className={"cal-hours" + (isWeek ? " is-week" : " is-day")}
      style={{ ["--cal-cols" as string]: days.length }}
    >
      {isWeek && (
        <div className="cal-hours-head" role="group" aria-label="Week dates">
          <span className="cal-gutter" />
          {columns.map(({ day, items }) => (
            <button
              key={day}
              className={
                "cal-col-head" +
                (day === today ? " is-today" : "") +
                (day === selected ? " selected" : "")
              }
              aria-label={"Open " + day}
              aria-pressed={day === selected}
              onClick={() => onOpenDay(day)}
            >
              <span className="cal-col-weekday">
                {label(day, { weekday: "short" })}
              </span>
              <span className="cal-col-date">{Number(day.slice(-2))}</span>
              <span className="cal-col-count">
                {items.length ? (
                  items
                    .slice(0, 3)
                    .map((e) => <i key={e.id} className={"cal-kind-" + e.kind} />)
                ) : (
                  <i className="cal-dot-none" />
                )}
              </span>
            </button>
          ))}
        </div>
      )}
      {(hasAllDay || !isWeek) && (
        <div className="cal-allday">
          <span className="cal-gutter">All day</span>
          {columns.map(({ day, untimed }) => (
            <div key={day} className="cal-allday-cell">
              {untimed.map((e) => (
                <button
                  key={e.id}
                  type="button"
                  className={
                    "cal-pill cal-kind-" +
                    e.kind +
                    (e.status === "completed" ? " completed" : "")
                  }
                  aria-label={calendarKind(e).label + ": " + e.title}
                  title={calendarKind(e).label + ": " + e.title}
                  onClick={() => onEntry(e)}
                >
                  {!spansTime(e) && <KindIcon kind={e.kind} size={12} />}
                  <span className="cal-pill-name">{e.title}</span><SourceBadge source={e.source}/>
                </button>
              ))}
            </div>
          ))}
        </div>
      )}
      <div className="cal-hours-scroll" ref={scroller}>
        <div className="cal-hours-body" style={{ height: 24 * HOUR }}>
          <div className="cal-gutter cal-hour-labels" aria-hidden="true">
            {Array.from({ length: 23 }, (_, i) => (
              <span key={i} style={{ top: (i + 1) * HOUR }}>
                {new Intl.DateTimeFormat(undefined, { hour: "numeric" }).format(
                  new Date(2000, 0, 1, i + 1),
                )}
              </span>
            ))}
          </div>
          {columns.map(({ day, timed }) => (
            <div
              key={day}
              className={"cal-hours-col" + (day === today ? " is-today" : "")}
            >
              {timed.map(({ item: e, start, end, left, width, depth, overlaps }) => {
                const height = ((end - start) / 60) * HOUR;
                // Overlapping items in a narrow week column drop the time (it stays in the tooltip
                // and accessible name) and let the title wrap instead of cutting it short.
                const narrow = isWeek && overlaps;
                const time =
                  clockLabel(e.at!, zone) +
                  (spansTime(e) && e.end_at
                    ? " – " + clockLabel(e.end_at, zone)
                    : "");
                return (
                  <button
                    key={e.id}
                    type="button"
                    className={
                      "cal-block cal-kind-" +
                      e.kind +
                      (spansTime(e) ? " is-span" : " is-moment") +
                      (height < 40 ? " is-short" : "") +
                      (narrow ? " is-narrow" + (height >= 32 ? " is-two-line" : "") : "") +
                      (depth ? " is-cascaded" : "") +
                      (e.status === "completed" ? " completed" : "")
                    }
                    style={{
                      top: (start / 60) * HOUR + 1,
                      height: height - 2,
                      left: `calc(${left}% + ${narrow ? 1 : 2}px)`,
                      width: `calc(${width}% - ${narrow ? 2 : 4}px)`,
                      zIndex: depth ? 1 + depth : undefined,
                    }}
                    aria-label={calendarKind(e).label + ": " + e.title + ", " + time}
                    title={time + "  " + e.title}
                    onClick={() => onEntry(e)}
                  >
                    <span className="cal-block-title">
                      {!spansTime(e) && <KindIcon kind={e.kind} size={12} />}
                      <span>{e.title}</span><SourceBadge source={e.source}/>
                    </span>
                    <time className="cal-block-time">{time}</time>
                  </button>
                );
              })}
              {day === today && (
                <span
                  className="cal-now"
                  style={{ top: (nowMinutes / 60) * HOUR }}
                  aria-hidden="true"
                />
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
