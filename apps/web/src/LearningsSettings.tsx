import { useEffect, useState } from "react";
import { api } from "./api";
import { SettingRow } from "./SettingsLayout";

export type Learning = {
  id: string; kind: "bug" | "knowledge" | string; title: string; applies_when: string; learning: string;
  paths: string[]; confirmations: number; run_id: string | null; repository?: string; created_at: string;
};

/** Entries grouped by repository, in the order the server returns them (most confirmed, then most recent). */
export function byRepository(items: Learning[]): [string, Learning[]][] {
  const groups = new Map<string, Learning[]>();
  for (const item of items) groups.set(item.repository || "Unnamed repository", [...(groups.get(item.repository || "Unnamed repository") ?? []), item]);
  return [...groups.entries()];
}

export const confirmedText = (n: number) => n > 0 ? `Came up again in ${n} later ${n === 1 ? "run" : "runs"}` : "";

/** Compound (run_learnings.py): what finished runs wrote down per repository. Every later run there is given these. */
export function LearningsSettings({ readOnly }: { readOnly: boolean }) {
  const [items, setItems] = useState<Learning[] | null>(null);
  const [busy, setBusy] = useState(""), [error, setError] = useState("");
  useEffect(() => { api<{ items: Learning[] }>("/remotea/learnings").then(d => setItems(d.items)).catch(() => {}); }, []);
  if (!items) return null;
  const remove = async (id: string) => {
    setBusy(id); setError("");
    try { await api(`/remotea/learnings/${id}/remove`, { method: "POST" }); setItems(items.filter(i => i.id !== id)); }
    catch (e) { setError(e instanceof Error ? e.message : "That was not removed."); }
    setBusy("");
  };
  return <>
    <SettingRow label="What runs have learned"
      hint="When a run hits something a later run would have to rediscover (a check that fails for a non-obvious reason, a convention it got wrong, something you corrected), it is written down here. Every later run in the same repository is given these at its start. Remove any that is wrong or out of date.">
      <span>{items.length ? `${items.length} ${items.length === 1 ? "entry" : "entries"}` : "Nothing yet"}</span>
    </SettingRow>
    {!!items.length && <div className="learnings-list">
      {byRepository(items).map(([repository, rows]) => <section key={repository} aria-label={repository}>
        <h4>{repository}</h4>
        <ul>{rows.map(row => <li key={row.id}>
          <div>
            <strong>{row.title}</strong>
            <p><span>When</span>{row.applies_when}</p>
            <p><span>Do</span>{row.learning}</p>
            {row.confirmations > 0 && <small>{confirmedText(row.confirmations)}</small>}
          </div>
          <button type="button" className="btn btn-ghost btn-sm" disabled={readOnly || busy === row.id} onClick={() => void remove(row.id)}>{busy === row.id ? "Removing…" : "Remove"}</button>
        </li>)}</ul>
      </section>)}
    </div>}
    {error && <p role="alert" className="work-error">{error}</p>}
  </>;
}
