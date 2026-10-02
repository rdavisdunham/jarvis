import { useEffect, useState } from "react";
import { api } from "./api";
type Hold = {
  id: string;
  model: string;
  state: string;
  created_at: string;
  recorded_usd: number;
  held_usd: number;
  reason: string;
};
export function BudgetHolds({ enforced = true }: { enforced?: boolean }) {
  const [holds, setHolds] = useState<Hold[]>([]),
    [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    api<{ items: Hold[] }>("/budget/holds")
      .then((r) => {
        if (active) setHolds(r.items);
      })
      .catch((e) => {
        if (active) setError(e.message);
      });
    return () => {
      active = false;
    };
  }, []);
  const uncertain = holds.filter((h) => h.state === "uncertain");
  if (error) return <p className="footnote">{error}</p>;
  if (!uncertain.length) return null;
  return (
    <details className="settings-block budget-holds">
      <summary>{uncertain.length} {uncertain.length === 1 ? "session" : "sessions"} awaiting final usage</summary>
      <p>
        {enforced ? "These estimates reserve room in your limit until provider billing is checked." : "These are estimates for sessions without a final usage report. They do not block Eri while spending limits are off."}
        {" "}They are not confirmed charges.
      </p>
      {uncertain.map((h) => (
        <div className="budget-hold" key={h.id}>
          <span>
            <strong>{h.model}</strong>
            <span className="budget-hold-meta">
              <span className="chip">{new Date(h.created_at).toLocaleString()}</span>
              <span className="chip">
                {h.reason === "activity_lease_expired"
                  ? "Connection ended without a final report"
                  : "Final report unavailable"}
              </span>
              <span className="chip">Recorded ${h.recorded_usd.toFixed(4)}</span>
            </span>
          </span>
          <span>${h.held_usd.toFixed(2)} held</span>
        </div>
      ))}
      <div className="settings-links">
        <a
          href="https://platform.openai.com/usage"
          target="_blank"
          rel="noreferrer"
        >
          Open provider usage
        </a>
      </div>
    </details>
  );
}
