import { useEffect, useState } from "react";
import { Check, Clock3, Plus, Repeat2, X } from "lucide-react";
import type { Notice, Schedule } from "./types";
import { timeLabel, recurrenceLabel } from "./components";

export function ReminderList({
  schedules,
  notices,
  zone,
  busy,
  highlight,
  create,
  mutate,
}: {
  schedules: Schedule[];
  notices: Notice[];
  zone: string;
  busy: boolean;
  highlight: string | null;
  create: () => void;
  mutate: (tool: string, args: unknown, message: string) => Promise<unknown>;
}) {
  const [tab, setTab] = useState("pending");
  useEffect(() => {
    if (!highlight) return;
    const target = schedules.find((s) => s.id === highlight);
    setTab(
      target?.status === "completed"
        ? "completed"
        : target?.status === "cancelled"
          ? "cancelled"
          : "pending",
    );
  }, [highlight, schedules]);
  const rows = schedules.filter((s) =>
    tab === "pending"
      ? ["active", "finished"].includes(s.status)
      : s.status === tab,
  );
  const completedOccurrences =
    tab === "completed"
      ? notices.filter(
          (n) =>
            n.completed_at &&
            schedules.some((s) => s.id === n.schedule_id && s.recurrence),
        )
      : [];
  return (
    <>
      <div className="section-head">
        <h2>Reminders</h2>
        <button className="primary compact" onClick={create}>
          <Plus size={16} />
          New reminder
        </button>
      </div>
      <div className="record-tabs" aria-label="Reminder status">
        {[
          ["pending", "Upcoming & due"],
          ["completed", "Completed"],
          ["cancelled", "Cancelled"],
        ].map(([value, label]) => (
          <button
            key={value}
            className={tab === value ? "active" : ""}
            aria-pressed={tab === value}
            onClick={() => setTab(value)}
          >
            {label}
          </button>
        ))}
      </div>
      {rows.map((s) => {
        const occurrence = notices.find(
          (n) => n.schedule_id === s.id && !n.completed_at,
        );
        const complete = () =>
          s.recurrence && occurrence
            ? mutate(
                "notification.complete",
                { notification_id: occurrence.id },
                "Occurrence completed · routine continues",
              )
            : mutate(
                "schedule.complete",
                { schedule_id: s.id, expected_revision: s.revision },
                "Reminder completed",
              );
        return (
          <article
            id={"record-" + s.id}
            tabIndex={-1}
            className={
              "schedule-row " + (highlight === s.id ? "record-highlight" : "")
            }
            key={s.id}
          >
            <span className="item-icon">
              {s.status === "completed" ? (
                <Check size={19} />
              ) : s.recurrence ? (
                <Repeat2 size={19} />
              ) : (
                <Clock3 size={19} />
              )}
            </span>
            <div className="grow">
              <strong>{s.title}</strong>
              <span>
                {s.completed_at
                  ? "Completed " + timeLabel(s.completed_at, zone)
                  : s.status === "finished"
                    ? "Delivered · waiting for you"
                    : s.next_run_at
                      ? timeLabel(s.next_run_at, zone)
                      : "Cancelled"}
                {" · " + recurrenceLabel(s.recurrence)}
                {s.kind === "recurring_task" ? " · New task each time" : ""}
              </span>
            </div>
            {tab === "pending" && (
              <>
                {(!s.recurrence || occurrence) && (
                  <button
                    className="text-button complete-reminder"
                    disabled={busy}
                    aria-label={"Complete " + s.title}
                    onClick={() => void complete()}
                  >
                    <Check size={16} />
                    {s.recurrence ? "Done this time" : "Complete"}
                  </button>
                )}
                {s.status === "active" && (
                  <button
                    className="icon-button"
                    disabled={busy}
                    aria-label={"Cancel " + s.title}
                    title={
                      s.recurrence ? "Stop future reminders" : "Cancel reminder"
                    }
                    onClick={() =>
                      void mutate(
                        "schedule.cancel",
                        { schedule_id: s.id, expected_revision: s.revision },
                        "Reminder cancelled",
                      )
                    }
                  >
                    <X size={17} />
                  </button>
                )}
              </>
            )}
          </article>
        );
      })}
      {completedOccurrences.map((n) => (
        <article className="schedule-row" key={n.id}>
          <span className="item-icon">
            <Check size={19} />
          </span>
          <div className="grow">
            <strong>{n.title}</strong>
            <span>
              Completed {timeLabel(n.completed_at!, zone)} · Routine continues
            </span>
          </div>
        </article>
      ))}
      {!rows.length && !completedOccurrences.length && (
        <div className="empty-state">
          <Clock3 size={30} />
          <h3>
            {tab === "completed"
              ? "A little history of things done."
              : tab === "cancelled"
                ? "No cancelled reminders."
                : "Nothing waiting for a nudge."}
          </h3>
          <p>
            {tab === "completed"
              ? "Reminders you mark complete stay here."
              : tab === "pending"
                ? "Add a reminder and Eri will keep the time."
                : ""}
          </p>
        </div>
      )}
    </>
  );
}
