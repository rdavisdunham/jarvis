import { useState } from "react";
import { Check, X } from "lucide-react";
import { Dialog } from "./ux";
import { useStructureActions } from "./structure-actions";
import type { MemoryCandidate, SetupSession } from "./setup-model";
import "./details.css";

const SOURCE: Record<MemoryCandidate["action"], string> = {
  add: "From this conversation", update: "From this conversation", keep: "Already known", retire: "No longer true",
};

/** Set up with Dani, step 3: what Dani should always keep in mind. Every row is on by default; unticked rows are
 * discarded (not rejected), and edits are saved as the owner wrote them. */
export function SetupMemoryReview({ candidates, knownCount, onClose, onKeepTalking, onSaved }: {
  candidates: MemoryCandidate[]; knownCount: number; onClose: () => void; onKeepTalking: () => void;
  onSaved: (result: SetupSession & { saved_ids: string[]; retired_ids: string[] }) => void;
}) {
  const [rows, setRows] = useState(candidates.map(candidate => ({ ...candidate, include: true })));
  const { run, busy, error } = useStructureActions();
  const edit = (id: string, patch: Partial<(typeof rows)[number]>) =>
    setRows(current => current.map(row => row.id === id ? { ...row, ...patch } : row));
  const blank = rows.some(row => row.include && !row.content.trim());
  const save = async () => onSaved(await run("setup.memories_save", {
    choices: rows.map(row => ({ id: row.id, content: row.content.trim() || row.content, include: row.include })),
  }));
  return <Dialog className="schema-dialog setup-memory-dialog" aria-label="About you" onEscape={busy ? undefined : onClose}>
    <header className="schema-dialog-head"><div><h2>About you</h2>
      <p>These are the things Dani will keep in mind. Edit or skip any of them.</p></div>
      <button className="btn-icon" aria-label="Close About you" disabled={busy} onClick={onClose}><X size={20}/></button></header>
    {error && <p role="alert" className="detail-alert schema-alert">{error}</p>}
    <div className="schema-dialog-body single"><section>
      <ul className="setup-memory-rows" aria-label="Things Dani will keep in mind">
        {rows.map(row => <li key={row.id} className={"setup-memory-row" + (row.include ? "" : " skipped")}>
          <label className="setup-memory-check">
            <input type="checkbox" checked={row.include} disabled={busy}
              aria-label={(row.action === "retire" ? "Stop keeping in mind: " : "Keep in mind: ") + row.content}
              onChange={e => edit(row.id, { include: e.target.checked })}/>
          </label>
          <div className="setup-memory-main">
            <textarea className="setup-memory-text" rows={2} maxLength={2000} value={row.content}
              aria-label="What Dani will keep in mind" disabled={busy || !row.include || row.action === "retire"}
              onChange={e => edit(row.id, { content: e.target.value })}/>
            <span className="chip" title={row.evidence && row.action !== "keep" ? "“" + row.evidence + "”" : undefined}>
              {SOURCE[row.action]}</span>
          </div>
        </li>)}
      </ul>
      {knownCount > 0 && <p className="setup-tree-impact">Also learned: {knownCount === 1 ? "1 fact" : knownCount + " facts"} from earlier chats will stay as {knownCount === 1 ? "it is" : "they are"}.</p>}
    </section></div>
    <footer className="schema-dialog-foot">
      <button className="btn" disabled={busy} onClick={onKeepTalking}>Keep talking</button>
      <button className="btn btn-primary" disabled={busy || blank} onClick={() => void save().catch(() => {})}>
        <Check size={17}/>Save about you</button>
    </footer>
  </Dialog>;
}
