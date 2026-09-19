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
    <details className="budget-holds">
      <summary>{uncertain.length} sessions awaiting final usage</summary>
      <p>
        {enforced ? "These estimates reserve room in your limit until provider billing is checked." : "These are estimates for sessions without a final usage report. They do not block Eri while spending limits are off."}
        {" "}They are not confirmed charges.
      </p>
      <a
        className="text-button"
        href="https://platform.openai.com/usage"
        target="_blank"
        rel="noreferrer"
      >
        Open provider usage
      </a>
      {uncertain.map((h) => (
        <div className="budget-hold" key={h.id}>
          <span>
            {h.model}
            <small>
              {new Date(h.created_at).toLocaleString()} ·{" "}
              {h.reason === "activity_lease_expired"
                ? "Connection ended without a final report"
                : "Final report unavailable"}
            </small>
            <small>Recorded: ${h.recorded_usd.toFixed(4)}</small>
          </span>
          <span>${h.held_usd.toFixed(2)} held</span>
        </div>
      ))}
    </details>
  );
}
