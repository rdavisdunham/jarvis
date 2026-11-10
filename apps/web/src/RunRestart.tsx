import { useState } from "react";
import { RotateCcw } from "lucide-react";
import { postAction } from "./api";
import type { Run } from "./remotea-model";
import { RESTART_NOTE_LIMIT, RESTART_TITLES, restartBody, restartLinks, restartOffer, restartProblem, type RestartMode } from "./run-restart";

const message = (e: unknown) => (e as Error).message || "Something went wrong.";

/** "Restarted from" / "Restarted as": links between a finished run and the run that continues it. */
export function RunLineage({ run, onOpen }: { run: Pick<Run, "restarted_from" | "restarted_as">; onOpen: (id: string) => void }) {
  const links = restartLinks(run);
  const shown = [links.from, links.as].filter(link => link !== null);
  if (!shown.length) return null;
  return <p className="run-lineage">{shown.map(link => <span key={link.label}>{link.label}{" "}
    <a href={`?view=remotea&run=${link.id}`} onClick={e => { e.preventDefault(); onOpen(link.id); }}>{link.text}</a></span>)}</p>;
}

/** Restart run, where a finished run's drawer shows its end state: a note and two ways to continue, each a new run
 * linked to this one (the finished run stays finished). The server decides whether it is allowed; this shows why not. */
export function RunRestart({ run, disabled, onRestarted }: {
  run: Pick<Run, "id" | "state" | "state_revision" | "restart" | "restarted_as">; disabled: boolean; onRestarted: (id: string) => void;
}) {
  const [open, setOpen] = useState(false), [note, setNote] = useState("");
  const [busy, setBusy] = useState<RestartMode | null>(null), [error, setError] = useState("");
  const offer = restartOffer(run);
  if (!offer) return null;
  const send = async (mode: RestartMode) => {
    if (restartProblem(mode, note)) return;
    setBusy(mode); setError("");
    try {
      const started = await postAction<Run>(`/remotea/runs/${run.id}/restart`, restartBody(run, mode, note));
      setOpen(false); setNote("");
      onRestarted(started.id);
    } catch (e) { setError(message(e)); }
    finally { setBusy(null); }
  };
  if (!open) return <section className="run-restart" aria-label="Restart run">
    <button className="btn btn-sm" title={RESTART_TITLES.button} disabled={disabled || !offer.enabled} onClick={() => { setError(""); setOpen(true); }}>
      <RotateCcw size={14} aria-hidden="true"/>Restart run</button>
    {offer.why && <p className="footnote">{offer.why}</p>}
  </section>;
  const off = disabled || !!busy || !offer.enabled;
  const planProblem = restartProblem("plan", note), pilotProblem = restartProblem("pilot", note);
  return <section className="run-restart run-restart-open" aria-label="Restart run">
    <label className="field"><span className="field-label-text">What do you want to add or change?</span>
      <textarea rows={4} maxLength={RESTART_NOTE_LIMIT} value={note} disabled={!!busy} placeholder="Optional when you pilot it yourself."
        onChange={e => setNote(e.target.value)}/></label>
    <p className="footnote">Starts a new run on the same branch and worktree, linked to this one. Where remotea can, the agent resumes this run's conversation.</p>
    {error && <p role="alert" className="work-error">{error}</p>}
    <div className="remotea-control-row">
      <button className="btn btn-sm btn-primary" title={planProblem ?? RESTART_TITLES.plan} disabled={off || !!planProblem}
        onClick={() => void send("plan")}>{busy === "plan" ? "Starting…" : "Plan it"}</button>
      <button className="btn btn-sm" title={pilotProblem ?? RESTART_TITLES.pilot} disabled={off || !!pilotProblem}
        onClick={() => void send("pilot")}>{busy === "pilot" ? "Starting…" : "Pilot it myself"}</button>
      <button className="btn btn-sm btn-ghost" disabled={!!busy} onClick={() => { setOpen(false); setError(""); }}>Cancel</button>
    </div>
  </section>;
}
