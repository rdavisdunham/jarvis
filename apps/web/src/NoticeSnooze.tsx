import { useState } from "react";
import "./calendar.css";
export function NoticeSnooze({
  onSnooze,
}: {
  onSnooze: (args: { minutes?: number; until?: string }) => Promise<unknown>;
}) {
  const [until, setUntil] = useState(""),
    [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  const save = async (args: { minutes?: number; until?: string }) => {
    setBusy(true);
    setError("");
    try {
      await onSnooze(args);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <details className="notice-snooze">
      <summary>Snooze…</summary>
      <div className="notice-snooze-body">
        {(
          [
            [10, "10 min"],
            [60, "1 hour"],
            [180, "3 hours"],
          ] as const
        ).map(([n, label]) => (
          <button
            key={n}
            className="btn btn-sm"
            disabled={busy}
            onClick={() => void save({ minutes: n })}
          >
            {label}
          </button>
        ))}
        <label className="field">
          <span className="field-label-text">Local date and time</span>
          <input
            type="datetime-local"
            aria-label="Snooze until"
            value={until}
            onChange={(e) => setUntil(e.target.value)}
          />
        </label>
        <button
          className="btn btn-soft"
          disabled={!until || busy}
          onClick={() => void save({ until })}
        >
          Snooze until then
        </button>
      </div>
      {error && (
        <p role="alert" className="field-error">
          {error}
        </p>
      )}
    </details>
  );
}
