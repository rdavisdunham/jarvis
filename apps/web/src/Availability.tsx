import { useEffect, useRef, useState } from "react";
import { Clock3, Search } from "lucide-react";
import { post } from "./api";

type Result = {
  status: string;
  source?: string;
  note?: string;
  reason?: string;
  checked_at?: string;
  free: { start: string; end: string }[];
  busy: { start: string; end: string }[];
};
export function Availability({
  day,
  timezone,
  enabled,
}: {
  day: string;
  timezone: string;
  enabled: boolean;
}) {
  const [start, setStart] = useState("09:00"),
    [end, setEnd] = useState("17:00");
  const [minutes, setMinutes] = useState(30),
    [result, setResult] = useState<Result | null>(null);
  const [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  const key = [day, timezone, start, end, minutes, enabled].join("|");
  const currentKey = useRef(key);
  currentKey.current = key;
  useEffect(() => {
    setResult(null);
    setError("");
  }, [day, timezone, enabled]);
  const fmt = (v: string) =>
    new Intl.DateTimeFormat(undefined, {
      timeZone: timezone,
      hour: "numeric",
      minute: "2-digit",
    }).format(new Date(v));
  async function check() {
    const requestKey = key;
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const data = await post<Result>("/calendar/availability/day", {
        date: day,
        start_time: start,
        end_time: end,
        timezone,
        minutes,
      });
      if (currentKey.current === requestKey) setResult(data);
    } catch (e) {
      if (currentKey.current === requestKey) setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  if (!enabled) return null;
  return (
    <details className="availability-panel">
      <summary>
        <Clock3 size={15} aria-hidden="true" />
        Find an open time
      </summary>
      <p className="field-hint">
        Checks Eridani appointments, work blocks and connected Google calendars.
        Task deadlines and reminders don’t reserve time.
      </p>
      <div className="availability-controls">
        <label className="field">
          <span className="field-label-text">From</span>
          <input
            aria-label="Availability from"
            type="time"
            value={start}
            onChange={(e) => {
              setStart(e.target.value);
              setResult(null);
            }}
          />
        </label>
        <label className="field">
          <span className="field-label-text">Until</span>
          <input
            aria-label="Availability until"
            type="time"
            value={end}
            onChange={(e) => {
              setEnd(e.target.value);
              setResult(null);
            }}
          />
        </label>
        <label className="field">
          <span className="field-label-text">At least</span>
          <select
            aria-label="Minimum free time"
            value={minutes}
            onChange={(e) => {
              setMinutes(Number(e.target.value));
              setResult(null);
            }}
          >
            <option value={15}>15 minutes</option>
            <option value={30}>30 minutes</option>
            <option value={60}>1 hour</option>
            <option value={120}>2 hours</option>
          </select>
        </label>
        <button
          className="btn btn-soft"
          disabled={busy || !start || !end || end <= start}
          onClick={() => void check()}
        >
          <Search size={15} aria-hidden="true" />
          {busy ? "Checking…" : "Check availability"}
        </button>
      </div>
      {error && (
        <p className="error-banner availability-result" role="alert">
          {error}
        </p>
      )}
      {result && (
        <div role="status" className="availability-result">
          {result.status !== "fresh" ? (
            <p>{result.reason}</p>
          ) : (
            <>
              <p>
                {result.free.length
                  ? "Open times"
                  : "No opening of that length in this window."}{" "}
                <span className="chip">
                  {result.source === "eridani_only"
                    ? "Eridani only, Google is not connected"
                    : "Google and Eridani"}
                </span>{" "}
                <span className="chip">Checked {fmt(result.checked_at!)}</span>
              </p>
              {!!result.free.length && (
                <ul className="available-times">
                  {result.free.map((slot) => (
                    <li key={slot.start}>
                      {fmt(slot.start)}–{fmt(slot.end)}
                    </li>
                  ))}
                </ul>
              )}
            </>
          )}
        </div>
      )}
    </details>
  );
}
