import { useEffect, useId, useState } from "react";
import { FolderGit2, Play } from "lucide-react";
import { api, ApiError, post } from "./api";
import { Dialog } from "./ux";
import {
  afterTargetSaved, DEFAULT_DRAFT, HARNESSES, harnessChoice, needsTargetData, preflightLine, preflightNotes, preflightUrl, startBlock, startBody,
  type HarnessId, type NeedsTarget, type Preflight, type Run, type RunDraft,
} from "./remotea-model";
import { useRemotea } from "./remotea-context";
import { TargetPicker } from "./Remotea";
import "./runs.css";

const message = (e: unknown) => (e as Error).message || "Something went wrong.";

/** The compact Run dialog: where it runs (resolved, or the picker), which agent and model, whether to interview
 * first. One Start button; the run opens in the side panel. */
export function RunTaskDialog({ taskId, onClose, onStarted }: { taskId: string; onClose: () => void; onStarted: (runId: string) => void }) {
  const { status } = useRemotea();
  const hosts = status?.hosts ?? [];
  const [pre, setPre] = useState<Preflight | null | undefined>(undefined);
  const [task, setTask] = useState<{ id: string; title: string } | null>(null);
  const [draft, setDraft] = useState<RunDraft>(DEFAULT_DRAFT);
  const [need, setNeed] = useState<NeedsTarget | null>(null);
  const [busy, setBusy] = useState(false), [error, setError] = useState("");
  const heading = useId();
  useEffect(() => {
    let live = true;
    // Read again with the picked target once the picker saves one, so its harnesses, verify and MCP notes show
    // before anything starts.
    api<Preflight & { task?: { id: string; title: string } }>(preflightUrl(taskId, draft.harness, draft.target))
      .then(p => { if (!live) return; setPre(p); if (p.task) setTask(p.task); setNeed(p.needs_target && p.need ? p.need : null); })
      .catch(() => { if (live) setPre(null); });
    return () => { live = false; };
  }, [taskId, draft.harness, draft.target]);
  const start = async (target?: string | null) => {
    setBusy(true); setError("");
    try {
      const run = await post<Run>("/remotea/runs", startBody(taskId, { ...draft, target: target ?? draft.target }));
      onStarted(run.id);
    } catch (e) {
      const data = e instanceof ApiError ? e.data as { run_id?: string } | null : null;
      if (e instanceof ApiError && e.code === "RUN_ACTIVE" && data?.run_id) { onStarted(data.run_id); return; }
      const needed = e instanceof ApiError && e.code === "NEEDS_TARGET" ? needsTargetData(e.data) : null;
      if (needed) setNeed(needed); else setError(message(e));
    } finally { setBusy(false); }
  };
  const harness = HARNESSES.find(h => h.id === draft.harness) ?? HARNESSES[0];
  // The preflight says what the task's target can start; any connected host counts only when there is no preflight.
  const choice = (id: HarnessId) => harnessChoice(pre, hosts, id);
  const available = (id: HarnessId) => choice(id).ok;
  const unavailable = HARNESSES.map(h => choice(h.id).reason).filter(Boolean);
  const notes = preflightNotes(pre);
  // A blocking problem (the server offline, agents without Eridani tools) keeps Start off; a freestyle start lifts
  // the tools one. Warnings (a verify command not ready) only show as notes.
  const block = startBlock(pre, draft);
  const blocked = (!!pre && !pre.active_run && !pre.needs_target && !pre.target) || !!block.problem;
  const interviewHint = draft.interview === "skip" ? "Starts on the task as written; no questions first."
    : pre?.interview ? "The agent asks you a few questions first, then plans." : pre ? "This task is already planned; the agent starts on its plan." : "";
  return <Dialog className="dialog run-dialog" aria-labelledby={heading} onEscape={onClose} onBackdrop={onClose}>
    <h2 id={heading}>Run with an agent</h2>
    {task && <p className="run-dialog-task">{task.title}</p>}
    {pre?.active_run ? <>
      <p className="run-dialog-target">{preflightLine(pre)}</p>
      <div className="settings-form-actions">
        <button className="btn btn-primary btn-sm" onClick={() => onStarted(pre.active_run!.id)}>Open run</button>
        <button className="btn btn-ghost btn-sm" onClick={onClose}>Cancel</button>
      </div>
    </> : <>
      {need ? <TargetPicker key={need.task_id} need={need} saveLabel="Save"
        onSaved={id => { const next = afterTargetSaved(draft, id); setPre(undefined); setNeed(next.need); setDraft(next.draft); }} onCancel={onClose}/> : <>
        <p className="run-dialog-target"><FolderGit2 size={15} aria-hidden="true"/>{pre === undefined ? "Checking where it runs…" : preflightLine(pre)}</p>
        <div className="run-dialog-row">
          <span id={heading + "-agent"}>Agent</span>
          <div className="segmented" role="group" aria-labelledby={heading + "-agent"}>
            {HARNESSES.map(h => <button key={h.id} type="button" aria-pressed={draft.harness === h.id} disabled={!available(h.id)}
              title={choice(h.id).reason}
              onClick={() => setDraft(d => ({ ...d, harness: h.id, model: h.models[0].id }))}>{h.label}</button>)}
          </div>
          {unavailable.length > 0 && <p className="footnote">{unavailable.join(" ")}{pre?.harness_note ? ` ${pre.harness_note}` : ""}</p>}
          {notes.map(n => <p key={n} className="footnote run-dialog-note" role="note">{n}</p>)}
        </div>
        <label className="run-dialog-row"><span>Model</span>
          <select value={draft.model} onChange={e => setDraft(d => ({ ...d, model: e.target.value }))}>
            {harness.models.map(m => <option key={m.id} value={m.id}>{m.label}</option>)}
          </select>
        </label>
        <div className="run-dialog-row">
          <span id={heading + "-interview"}>Interview first</span>
          <div className="segmented" role="group" aria-labelledby={heading + "-interview"}>
            <button type="button" aria-pressed={draft.interview === "auto"} onClick={() => setDraft(d => ({ ...d, interview: "auto" }))}>Auto</button>
            <button type="button" aria-pressed={draft.interview === "skip"} onClick={() => setDraft(d => ({ ...d, interview: "skip" }))}>Skip</button>
          </div>
          {interviewHint && <p className="footnote">{interviewHint}</p>}
        </div>
        {block.freestyle && <label className="run-dialog-freestyle"
          title="Talk to the agent directly: no interview, no plan, turns are recorded but not verified or judged.">
          <input type="checkbox" checked={!!draft.freestyle} onChange={e => setDraft(d => ({ ...d, freestyle: e.target.checked }))}/>
          <span>Start in freestyle</span><small>the agent has no Eridani tools to plan with</small>
        </label>}
        {block.problem && !notes.includes(block.problem.message) && <p className="footnote run-dialog-note" role="note">{block.problem.message}</p>}
        <div className="settings-form-actions">
          <button className="btn btn-primary btn-sm" disabled={busy || blocked || !available(draft.harness)} onClick={() => void start()}>
            <Play size={14} aria-hidden="true"/>{busy ? "Starting…" : "Start"}</button>
          <button className="btn btn-ghost btn-sm" disabled={busy} onClick={onClose}>Cancel</button>
        </div>
      </>}
      {error && <p role="alert" className="work-error">{error}</p>}
    </>}
  </Dialog>;
}
