import { useEffect, useState } from "react";
import { api } from "./api";
type Row = {
  job_id: string;
  status: string;
  operation: string;
  created_at: string;
  result?: { message?: string; title?: string };
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
      <summary>Recent calendar changes</summary>
      {!rows.length && (
        <p className="footnote">
          Calendar saves and their Google confirmation appear here.
        </p>
      )}
      {rows.map((row) => (
        <div key={row.job_id} className="calendar-write-row">
          <strong>
            {row.result?.title ||
              (row.operation === "create"
                ? "New event"
                : row.operation === "delete"
                  ? "Delete event"
                  : "Edit event")}
          </strong>
          <span>
            {row.status === "succeeded"
              ? "Confirmed by Google"
              : row.result?.message || "Waiting for Google…"}
          </span>
          <small>
            {new Date(row.created_at).toLocaleString()} · {row.status}
          </small>
        </div>
      ))}
    </details>
  );
}
