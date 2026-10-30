import { useState } from "react";
import "./calendar.css";

/** Quick choices; the server resolves words in the owner's time zone (apps/api/jarvis/snooze_words.py). */
export const SNOOZE_CHOICES: ReadonlyArray<readonly [string, { minutes?: number; until?: string }]> = [
  ["10 min", { minutes: 10 }],
  ["1 hour", { minutes: 60 }],
  ["After lunch", { until: "after lunch" }],
  ["This evening", { until: "this evening" }],
  ["Tomorrow morning", { until: "tomorrow morning" }],
  ["Next week", { until: "next week" }],
];

export function NoticeSnooze({
  onSnooze,
}: {
  onSnooze: (args: { minutes?: number; until?: string }) => Promise<unknown>;
}) {
  const [until, setUntil] = useState(""),
    [words, setWords] = useState(""),
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
        {SNOOZE_CHOICES.map(([label, args]) => (
          <button
            key={label}
            className="btn btn-sm"
            disabled={busy}
            onClick={() => void save(args)}
          >
            {label}
          </button>
        ))}
        <form
          className="notice-snooze-words"
          onSubmit={(e) => {
            e.preventDefault();
            if (words.trim()) void save({ until: words.trim() });
          }}
        >
          <label className="field">
            <span className="field-label-text">Or say when</span>
            <input
              aria-label="Snooze until, in words"
              placeholder="in 2 hours, friday 3pm…"
              maxLength={64}
              value={words}
              onChange={(e) => setWords(e.target.value)}
            />
          </label>
          <button className="btn btn-soft" disabled={!words.trim() || busy}>
            Snooze
          </button>
        </form>
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
