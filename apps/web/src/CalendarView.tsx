import { calendarKind } from "./calendar-presentation";
import { useRef } from "react";
import {
  CheckSquare,
  CalendarDays,
  ChevronLeft,
  ChevronRight,
  Clock3,
  Repeat2,
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
export function CalendarView(p: Props) {
  const lastTap = useRef({ day: "", time: 0 });
  const label = (day: string, options: Intl.DateTimeFormatOptions) =>
    new Date(day + "T12:00:00Z").toLocaleDateString(undefined, {
      ...options,
      timeZone: "UTC",
    });
  const week = weekDays(p.day);
  const agendaDays = p.mode === "week" ? week : [p.day];
  const move = (amount: number) =>
    p.onDay(
      p.mode === "month"
        ? shiftMonth(p.day, amount)
        : shiftDate(p.day, amount * (p.mode === "week" ? 7 : 1)),
    );
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
  return (
    <>
      <div
        className="calendar-view-tabs"
        role="group"
        aria-label="Calendar view"
      >
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
      <div className="calendar-heading">
        <h2>
          {p.mode === "month"
            ? label(p.day, { month: "long", year: "numeric" })
            : p.mode === "week"
              ? label(week[0], { month: "short", day: "numeric" }) +
                " – " +
                label(week[6], {
                  month: "short",
                  day: "numeric",
                  year: "numeric",
                })
              : label(p.day, {
                  weekday: "long",
                  month: "short",
                  day: "numeric",
                })}
        </h2>
        <div>
          <button
            className="icon-button"
            aria-label={"Previous " + p.mode}
            onClick={() => move(-1)}
          >
            <ChevronLeft size={18} />
          </button>
          <button className="text-button" onClick={() => p.onDay(p.today)}>
            Today
          </button>
          <button
            className="icon-button"
            aria-label={"Next " + p.mode}
            onClick={() => move(1)}
          >
            <ChevronRight size={18} />
          </button>
        </div>
      </div>
      <p className="footnote calendar-zone">
        {p.zone}
        {p.mode === "month" ? " · Double-tap a date to open its day." : ""}
      </p>
      {p.google?.calendar_enabled && (
        <p className="footnote calendar-sync-row" role="status">
          {p.google.syncing
            ? "Google calendars are syncing…"
            : p.google.status === "needs_reconnect"
              ? "Reconnect Google Calendar in Settings."
              : p.google.stale || p.google.status === "error"
                ? "Google events may be out of date. Check the connection in Settings."
                : "Google synced " +
                  new Date(p.google.last_sync_at!).toLocaleString()}
        </p>
      )}
      {p.error && (
        <p className="error-banner" role="alert">
          {p.error} <button onClick={p.onRetry}>Retry calendar</button>
        </p>
      )}
      {!p.loaded && !p.error && (
        <p role="status" className="footnote">
          Loading calendar…
        </p>
      )}
      <div className="calendar-type-legend" aria-label="Calendar item types">
        <span>
          <CheckSquare size={14} /> Task
        </span>
        <span>
          <Clock3 size={14} /> Task reminder
        </span>
        <span>
          <CalendarDays size={14} /> Event / work block
        </span>
      </div>
      {p.mode === "month" && (
        <div className="calendar-grid" aria-label="Month dates">
          {["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"].map((day) => (
            <div className="calendar-weekday" key={day}>
              {day}
            </div>
          ))}
          {p.days.map((day) => {
            const entries = p.events.filter((e) => e.date === day);
            return (
              <div
                key={day}
                data-day={day}
                className={
                  "calendar-day " +
                  (day === p.day ? "selected " : "") +
                  (day === p.today ? "is-today " : "") +
                  (day.slice(0, 7) !== p.day.slice(0, 7) ? "outside" : "")
                }
                onClick={(event) => selectDay(day, event.detail)}
                onDoubleClick={() => {
                  p.onDay(day);
                  p.onMode("day");
                }}
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
                <span className="calendar-labels">
                  {entries.slice(0, 2).map((e) => (
                    <button
                      key={e.id}
                      type="button"
                      aria-label={calendarKind(e).label + ": " + e.title}
                      title={calendarKind(e).label + ": " + e.title}
                      onClick={(click) => {
                        click.stopPropagation();
                        p.onEntry(e);
                      }}
                      className={
                        "calendar-chip " +
                        e.kind +
                        (e.status === "completed" ? " completed" : "")
                      }
                    >
                      {e.kind === "task" ? (
                        <CheckSquare size={12} />
                      ) : e.kind === "reminder" || e.kind === "routine" ? (
                        <Clock3 size={12} />
                      ) : (
                        <CalendarDays size={12} />
                      )}
                      <span className="calendar-chip-name">{e.title}</span>
                    </button>
                  ))}
                  {entries.length > 2 && (
                    <small>+{entries.length - 2} more</small>
                  )}
                </span>
                <span className="calendar-dots" aria-hidden="true">
                  {entries.slice(0, 3).map((e) => (
                    <i key={e.id} className={e.kind} />
                  ))}
                  {entries.length > 3 && <small>+{entries.length - 3}</small>}
                </span>
              </div>
            );
          })}
        </div>
      )}
      {p.mode === "week" && (
        <div className="calendar-week-strip" aria-label="Week dates">
          {week.map((day) => (
            <button
              key={day}
              aria-label={"Open " + day}
              aria-pressed={day === p.day}
              onClick={() => {
                p.onDay(day);
                p.onMode("day");
              }}
            >
              <small>{label(day, { weekday: "short" })}</small>
              <strong>{Number(day.slice(-2))}</strong>
              <small>
                {p.events.filter((e) => e.date === day).length || "–"}
              </small>
            </button>
          ))}
        </div>
      )}
      {p.truncated && (
        <p role="status">
          {p.warnings?.length
            ? "A calendar source needs attention: " +
              p.warnings
                .map((w) => w.calendar + " — " + w.reason.replaceAll("_", " "))
                .join("; ")
            : "This range contains more items than can be displayed. Switch to Day or Week to see a smaller range."}
        </p>
      )}
      <CalendarWriteActivity />
      {agendaDays.map((day) => {
        const agenda = p.events.filter((e) => e.date === day);
        return (
          <section
            className="calendar-day-agenda"
            key={day}
            aria-label={"Agenda for " + day}
          >
            <div className="section-head agenda-heading">
              <h2>
                {label(day, {
                  weekday: "long",
                  month: "short",
                  day: "numeric",
                })}
                <span>{agenda.length}</span>
              </h2>
              {p.mode !== "day" && (
                <button
                  className="text-button"
                  aria-label={"Open day view for " + day}
                  onClick={() => {
                    p.onDay(day);
                    p.onMode("day");
                  }}
                >
                  Open day <ChevronRight size={15} />
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
                  onClick={() => p.onEntry(e)}
                >
                  <span className={"agenda-kind " + e.kind}>
                    {e.kind === "task" ? (
                      <CheckSquare size={17} />
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
                    <span className={"calendar-type-badge " + e.kind}>
                      {calendarKind(e).label}
                    </span>
                    <small>
                      {e.kind === "google" ? e.calendar_title : ""}
                      {e.projected && e.kind !== "google" ? " · Upcoming" : ""}
                      {!!e.conflicts?.length &&
                        " · Deadline during " + e.conflicts.join(", ")}
                      {e.status === "completed" ? " · Completed" : ""}
                    </small>
                    {["google", "event", "block"].includes(e.kind) &&
                      e.at &&
                      e.end_at && (
                        <small>
                          Until{" "}
                          {new Intl.DateTimeFormat(undefined, {
                            timeZone: p.zone,
                            hour: "numeric",
                            minute: "2-digit",
                          }).format(new Date(e.end_at))}
                        </small>
                      )}
                  </span>
                  <ChevronRight size={15} />
                </button>
              ))}
              {p.loaded && !agenda.length && (
                <div className="calendar-empty">
                  Nothing {p.filtered ? "matching your filters " : ""}scheduled
                  for this day.
                </div>
              )}
            </div>
          </section>
        );
      })}
    </>
  );
}
