import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { Popover } from "./ux";
import { monthDays, shiftDate, shiftMonth } from "./workspace";
import { clockLabel } from "./work-views";
import { monthTitle, quickPicks, shiftMonthKeepDay } from "./date-picks";
import "./date-picker.css";

/** What a pick changes. `time`/`zone` are only sent by a field with a time. */
export type DateChange = { date: string | null; time?: string | null; zone?: string | null };
type TimeValue = { value: string | null; zone: string | null; homeZone: string };

/** "Tue, Nov 10" (with the year when it isn't this year). */
export function dateText(day: string) {
  const [y, m, d] = day.split("-").map(Number);
  if (!y || !m || !d) return day;
  return new Date(y, m - 1, d).toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric", year: y === new Date().getFullYear() ? undefined : "numeric" });
}
const longDate = (day: string) => new Date(day + "T12:00:00Z").toLocaleDateString("en-US", { timeZone: "UTC", weekday: "long", month: "long", day: "numeric", year: "numeric" });
const WEEKDAYS = ["S", "M", "T", "W", "T", "F", "S"];

/** A date field whose button opens quick picks, a Sunday-first month and Clear in a floating panel.
 * With `time`, the panel also sets a time ("Add time"); the zone defaults to `homeZone`. */
export function DateField({label, value, today, empty, onChange, time, disabled = false, buttonClassName = "prop-text"}: {
  label: string; value: string | null | undefined; today: string; empty: string; onChange: (change: DateChange) => void;
  time?: TimeValue; disabled?: boolean; buttonClassName?: string;
}) {
  const shown = value ? dateText(value) + (time?.value ? " · " + clockLabel(time.value) : "") : empty;
  return <Popover floating label={"Change " + label} title={label} disabled={disabled} buttonClassName={buttonClassName + (value ? "" : " empty")}
    panelClassName="date-panel" button={shown}>
    {close => <DatePanel label={label} value={value || null} today={today} time={time} onPick={change => { onChange(change); close(); }}/>}
  </Popover>;
}

function DatePanel({label, value, today, time, onPick}: {label: string; value: string | null; today: string; time?: TimeValue; onPick: (change: DateChange) => void}) {
  const [focus, setFocus] = useState(value ?? today);
  const [month, setMonth] = useState((value ?? today).slice(0, 7) + "-01");
  const [timeOpen, setTimeOpen] = useState(!!time?.value);
  const [clock, setClock] = useState(time?.value?.slice(0, 5) ?? "");
  const [zoneOpen, setZoneOpen] = useState(false);
  const [zone, setZone] = useState(time?.zone ?? time?.homeZone ?? "");
  const [typed, setTyped] = useState(value ?? "");
  const grid = useRef<HTMLDivElement>(null);
  const keyed = useRef(false);
  useEffect(() => {
    if (!keyed.current) return;
    keyed.current = false;
    grid.current?.querySelector<HTMLButtonElement>(`[data-day="${focus}"]`)?.focus();
  }, [focus, month]);
  // Picking a day keeps the time; a time typed but not yet set goes with it.
  const pick = (date: string) => {
    if (!time) return onPick({ date });
    const at = timeOpen && clock ? clock : time.value;
    onPick({ date, time: at || null, zone: at ? zone || time.homeZone : null });
  };
  const move = (day: string) => {
    keyed.current = true;
    setFocus(day);
    if (day.slice(0, 7) !== month.slice(0, 7)) setMonth(day.slice(0, 7) + "-01");
  };
  const keys = (e: KeyboardEvent<HTMLButtonElement>, day: string) => {
    const step = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -7, ArrowDown: 7 }[e.key];
    if (step) { e.preventDefault(); move(shiftDate(day, step)); }
    else if (e.key === "PageUp" || e.key === "PageDown") { e.preventDefault(); move(shiftMonthKeepDay(day, e.key === "PageUp" ? -1 : 1)); }
  };
  const days = monthDays(month);
  const weeks = Array.from({ length: 6 }, (_, w) => days.slice(w * 7, w * 7 + 7)).filter(week => week.some(d => d.startsWith(month.slice(0, 7))));
  const focusable = days.includes(focus) ? focus : month;
  const differentZone = !!time && zone !== time.homeZone;
  return <div className="date-picker">
    <div className="date-quick" role="group" aria-label="Quick picks">
      {quickPicks(today).map(q => <button key={q.label} type="button" className={"date-quick-pick" + (q.date === value ? " is-selected" : "")} aria-pressed={q.date === value} onClick={() => pick(q.date)}>
        <span>{q.label}</span><small>{q.hint}</small>
      </button>)}
    </div>
    <div className="date-month">
      <button type="button" className="btn-icon" aria-label="Previous month" onClick={() => setMonth(shiftMonth(month, -1))}><ChevronLeft size={16}/></button>
      <strong aria-live="polite">{monthTitle(month)}</strong>
      <button type="button" className="btn-icon" aria-label="Next month" onClick={() => setMonth(shiftMonth(month, 1))}><ChevronRight size={16}/></button>
    </div>
    <div ref={grid} className="date-grid" role="grid" aria-label={monthTitle(month)}>
      <div role="row" className="date-row">{WEEKDAYS.map((d, i) => <span key={i} role="columnheader" className="date-weekday">{d}</span>)}</div>
      {weeks.map(week => <div key={week[0]} role="row" className="date-row">
        {week.map(day => <div key={day} role="gridcell" aria-selected={day === value}>
          <button type="button" data-day={day} tabIndex={day === focusable ? 0 : -1} aria-label={longDate(day)} aria-current={day === today ? "date" : undefined}
            className={"date-day" + (day.startsWith(month.slice(0, 7)) ? "" : " is-outside") + (day === value ? " is-selected" : "") + (day === today ? " is-today" : "")}
            onKeyDown={e => keys(e, day)} onClick={() => pick(day)}>{Number(day.slice(8))}</button>
        </div>)}
      </div>)}
    </div>
    {time && value && <div className="date-time">
      {!timeOpen ? <button type="button" className="text-button" onClick={() => setTimeOpen(true)}>Add time</button> : <>
        <label className="date-time-row">Time
          <input type="time" aria-label={label.replace(/ date$/i, "") + " time"} value={clock} autoFocus={!time.value} onChange={e => setClock(e.target.value)}
            onKeyDown={e => { if (e.key === "Enter" && clock) { e.preventDefault(); pick(value); } }}/>
        </label>
        <div className="date-time-actions">
          <button type="button" className="btn btn-sm btn-primary" disabled={!clock || (clock === time.value?.slice(0, 5) && zone === (time.zone ?? time.homeZone))} onClick={() => pick(value)}>Set time</button>
          {time.value && <button type="button" className="text-button" onClick={() => onPick({ date: value, time: null, zone: null })}>Remove time</button>}
        </div>
        {zoneOpen ? <label className="date-time-row">Time zone<input aria-label={label.replace(/ date$/i, "") + " time zone"} value={zone} onChange={e => setZone(e.target.value)}/></label>
          : <small className="date-zone">{differentZone ? "Time zone: " + zone : "Your time zone"} · <button type="button" className="text-button" onClick={() => setZoneOpen(true)}>change</button></small>}
      </>}
    </div>}
    <div className="date-foot">
      <input type="date" className="date-typed" aria-label={label} value={typed} onChange={e => setTyped(e.target.value)}
        onKeyDown={e => { if (e.key === "Enter" && /^\d{4}-\d{2}-\d{2}$/.test(typed)) { e.preventDefault(); pick(typed); } }}/>
      {value && <button type="button" className="text-button" onClick={() => onPick(time ? { date: null, time: null, zone: null } : { date: null })}>Clear date</button>}
    </div>
  </div>;
}
