import { useEffect, useState } from "react";
import { ChevronRight } from "lucide-react";
import { api } from "./api";
type Row = {
  job_id: string;
  status: string;
  operation: string;
  created_at: string;
  result?: { message?: string; title?: string };
};
const statusChip: Record<string, [string, string]> = {
  succeeded: ["chip chip-done", "Confirmed"],
  failed: ["chip chip-due-overdue", "Failed"],
  cancelled: ["chip", "Cancelled"],
  unconfirmed: ["chip chip-due-today", "Not confirmed"],
};
export function CalendarWriteActivity() {
  const [rows, setRows] = useState<Row[]>([]);
  useEffect(() => {
    let active = true;
    const refresh = () =>
      api<{ items: Row[] }>("/calendar/writes")
        .then((data) => {
          if (active) setRows(data.items);
        })
        .catch(() => {});
    void refresh();
    const timer = setInterval(() => void refresh(), 5000);
    window.addEventListener("eri-calendar-write", refresh);
    return () => {
      active = false;
      clearInterval(timer);
      window.removeEventListener("eri-calendar-write", refresh);
    };
  }, []);
  return (
    <details className="calendar-write-activity">
      <summary>
        <ChevronRight size={15} aria-hidden="true" />
        Recent calendar changes
      </summary>
      {!rows.length && (
        <p className="field-hint">
          Calendar saves and their Google confirmation appear here.
        </p>
      )}
      {!!rows.length && (
        <div className="calendar-write-list">
          {rows.map((row) => {
            const [chipClass, chipText] = statusChip[row.status] ?? [
              "chip",
              "Waiting for Google",
            ];
            return (
              <div key={row.job_id} className="calendar-write-row">
                <strong>
                  {row.result?.title ||
                    (row.operation === "create"
                      ? "New event"
                      : row.operation === "delete"
                        ? "Delete event"
                        : "Edit event")}
                </strong>
                <span className={chipClass}>{chipText}</span>
                <time dateTime={row.created_at}>
                  {new Date(row.created_at).toLocaleString(undefined, {
                    dateStyle: "medium",
                    timeStyle: "short",
                  })}
                </time>
                {row.status !== "succeeded" && row.result?.message && (
                  <span className="write-message">{row.result.message}</span>
                )}
              </div>
            );
          })}
        </div>
      )}
    </details>
  );
}
