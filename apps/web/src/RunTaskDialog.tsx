import { type Dispatch, type SetStateAction, useEffect, useId, useState } from "react";
import { FolderGit2, Minus, Play, Plus, Wrench } from "lucide-react";
import { api, ApiError, post } from "./api";
import { Dialog } from "./ux";
import {
  afterTargetSaved, DRAFT_STORAGE_KEY, effectiveDraft, HARNESSES, harnessChoice, needsTargetData, preflightLine, preflightNotes, preflightUrl,
  restoreDraft, roleRun, ROLES_UNAVAILABLE, ROUNDS_MAX, ROUNDS_MIN, setupReason, splitWorker, startBlock, startBody, storedDraft,
  type HarnessId, type NeedsTarget, type Preflight, type PreflightProblem, type RoleChoice, type Run, type RunDraft,
} from "./remotea-model";
import { useRemotea } from "./remotea-context";
import { TargetPicker } from "./Remotea";
import "./runs.css";

const message = (e: unknown) => (e as Error).message || "Something went wrong.";

/** The remembered choices (DRAFT_STORAGE_KEY); storage may be missing or refuse (private mode), which is no error. */
function readDraft(): RunDraft {
  try { return restoreDraft(localStorage.getItem(DRAFT_STORAGE_KEY)); } catch { return restoreDraft(null); }
}
function saveDraft(draft: RunDraft) {
  try { localStorage.setItem(DRAFT_STORAGE_KEY, storedDraft(draft)); } catch { /* Remembering is a convenience only. */ }
}

type Choice = (id: HarnessId) => { ok: boolean; reason?: string };
const roleKey = (c: RoleChoice | null | undefined) => c ? `${c.harness}:${c.model}` : "";

/** One select over every harness and model, grouped by harness; a harness the target cannot start is greyed out
 * with its reason. `same` adds a first option that means "no role of its own" (the worker's "Same as planner"). */
function RoleSelect({ label, value, choice, onChange, same, disabled }: {
  label: string; value: RoleChoice | null | undefined; choice: Choice; onChange: (value: RoleChoice | null) => void; same?: string; disabled?: boolean;
}) {
  return <select aria-label={label} value={roleKey(value)} disabled={disabled} onChange={e => {
    const [harness, model] = e.target.value.split(/:(.*)/);
    onChange(harness ? { harness: harness as HarnessId, model } : null);
  }}>
    {same && <option value="">{same}</option>}
    {HARNESSES.map(h => <optgroup key={h.id} label={h.label + (choice(h.id).ok ? "" : " (cannot start here)")}>
      {h.models.map(m => <option key={m.id} value={`${h.id}:${m.id}`} disabled={!choice(h.id).ok}>{h.label} · {m.label}</option>)}
    </optgroup>)}
  </select>;
}

/** The Run dialog's roles: the Worker (default "Same as planner") and the Hyperplan toggle with its Reviewer, Rounds
 * and "Approve final plan before running". Off with the reason when the host cannot spawn roles (`roles` false, or a
 * HOST_NO_ROLES refusal) and in freestyle; a role the preflight or the start refused shows its problem under it. */
export function RoleControls({ draft, setDraft, choice, roles, rolesReason, problems, idPrefix }: {
  draft: RunDraft; setDraft: Dispatch<SetStateAction<RunDraft>>; choice: Choice; roles: boolean | null | undefined;
  /** Why roles are off (a HOST_NO_ROLES refusal at start); defaults to ROLES_UNAVAILABLE when `roles` is false. */
  rolesReason?: string;
  /** Problems about a role (`role` set): the preflight's, or the start's HARNESS_UNAVAILABLE refusal. */
  problems: PreflightProblem[]; idPrefix: string;
}) {
  const off = roles === false || !!rolesReason;
  const why = off ? rolesReason || ROLES_UNAVAILABLE : draft.freestyle ? "A freestyle run has no plan to review or hand over." : "";
  const locked = off || !!draft.freestyle;
  const about = (role: string) => problems.filter(p => p.role === role).map(p => <p key={p.code + p.message} className="footnote run-dialog-note" role="note">{p.message}</p>);
  const rounds = draft.rounds ?? 2;
  const setRounds = (n: number) => setDraft(d => ({ ...d, rounds: Math.min(ROUNDS_MAX, Math.max(ROUNDS_MIN, n)) }));
  return <>
    <label className="run-dialog-row"><span>Worker</span>
      <RoleSelect label="Worker" value={locked ? null : draft.worker} choice={choice} same="Same as planner" disabled={locked}
        onChange={worker => setDraft(d => ({ ...d, worker }))}/>
      {!locked && splitWorker(draft) && <small className="footnote">The planner only plans; this model gets the approved plan in a fresh session.</small>}
      {about("worker")}
    </label>
    <div className="run-dialog-row run-dialog-hyperplan">
      <label className="run-dialog-freestyle" title="A second model reviews each plan revision in a fresh read-only session before any work starts.">
        <input type="checkbox" checked={!locked && !!draft.hyperplan} disabled={locked} onChange={e => setDraft(d => ({ ...d, hyperplan: e.target.checked }))}/>
        <span>Hyperplan</span><small>a second model reviews the plan before any work starts</small>
      </label>
      {why && <p className="footnote run-dialog-note" role="note">{why}</p>}
      {!locked && draft.hyperplan && <div className="run-dialog-roles">
        <label className="run-dialog-row"><span>Reviewer</span>
          <RoleSelect label="Reviewer" value={draft.reviewer} choice={choice} onChange={reviewer => reviewer && setDraft(d => ({ ...d, reviewer }))}/>
          {about("reviewer")}
        </label>
        <div className="run-dialog-row">
          <span id={idPrefix + "-rounds"}>Rounds</span>
          <div className="run-dialog-stepper" role="group" aria-labelledby={idPrefix + "-rounds"}>
            <button type="button" className="btn btn-ghost btn-sm" aria-label="Fewer rounds" disabled={rounds <= ROUNDS_MIN} onClick={() => setRounds(rounds - 1)}><Minus size={14}/></button>
            <output aria-live="polite">{rounds}</output>
            <button type="button" className="btn btn-ghost btn-sm" aria-label="More rounds" disabled={rounds >= ROUNDS_MAX} onClick={() => setRounds(rounds + 1)}><Plus size={14}/></button>
          </div>
          <small className="footnote">At most {rounds} review{rounds === 1 ? "" : "s"}; a clean approve ends the loop early.</small>
        </div>
      </div>}
      {!locked && roleRun(draft) && <label className="run-dialog-freestyle">
        <input type="checkbox" checked={draft.approveFinal ?? true} onChange={e => setDraft(d => ({ ...d, approveFinal: e.target.checked }))}/>
        <span>Approve final plan before running</span>
      </label>}
    </div>
  </>;
}

/** The dialog's way to the guided setup view, with what the preflight found as its tooltip. */
export function SetupLink({ reason, onSetup }: { reason: string; onSetup: () => void }) {
  return <p className="footnote run-dialog-setup">
    <button type="button" className="text-button" title={reason} onClick={onSetup}>
      <Wrench size={13} aria-hidden="true"/>Open remotea setup</button> to see what is missing and how to fix it.</p>;
}

/** The compact Run dialog: where it runs (resolved, or the picker), which agent and model, whether to interview
 * first. One Start button; the run opens in the side panel. When the preflight reports a setup problem (no target,
 * agents without Eridani tools, the server or the target not ready), a link opens the guided setup view (`onSetup`). */
export function RunTaskDialog({ taskId, onClose, onStarted, onSetup }: {
  taskId: string; onClose: () => void; onStarted: (runId: string) => void; onSetup?: () => void;
}) {
  const { status } = useRemotea();
  const hosts = status?.hosts ?? [];
  const [pre, setPre] = useState<Preflight | null | undefined>(undefined);
  const [task, setTask] = useState<{ id: string; title: string } | null>(null);
  const [draft, setDraft] = useState<RunDraft>(readDraft);
  const [need, setNeed] = useState<NeedsTarget | null>(null);
  const [busy, setBusy] = useState(false), [error, setError] = useState("");
  // A start refused for its roles: HOST_NO_ROLES turns the role controls off, HARNESS_UNAVAILABLE names the role.
  const [noRoles, setNoRoles] = useState(""), [roleProblem, setRoleProblem] = useState<PreflightProblem | null>(null);
  const heading = useId();
  // What can start: a classic run while the host cannot spawn roles, or in freestyle.
  const effective = effectiveDraft(draft, noRoles ? false : pre?.roles);
  const reviewerHarness = effective.hyperplan ? effective.reviewer?.harness ?? null : null;
  const workerHarness = splitWorker(effective)?.harness ?? null;
  useEffect(() => {
    let live = true;
    // Read again with the picked target once the picker saves one, so its harnesses, verify and MCP notes show
    // before anything starts; and with the roles' harnesses, so a role the target cannot start says so.
    api<Preflight & { task?: { id: string; title: string } }>(preflightUrl(taskId, draft.harness, draft.target, { reviewer: reviewerHarness, worker: workerHarness }))
      .then(p => { if (!live) return; setPre(p); if (p.task) setTask(p.task); setNeed(p.needs_target && p.need ? p.need : null); })
      .catch(() => { if (live) setPre(null); });
    return () => { live = false; };
  }, [taskId, draft.harness, draft.target, reviewerHarness, workerHarness]);
  const start = async (target?: string | null) => {
    setBusy(true); setError(""); setRoleProblem(null);
    saveDraft(draft);
    try {
      const run = await post<Run>("/remotea/runs", startBody(taskId, { ...effective, target: target ?? draft.target }));
      onStarted(run.id);
    } catch (e) {
      const data = e instanceof ApiError ? e.data as { run_id?: string; role?: string } | null : null;
      if (e instanceof ApiError && e.code === "RUN_ACTIVE" && data?.run_id) { onStarted(data.run_id); return; }
      const needed = e instanceof ApiError && e.code === "NEEDS_TARGET" ? needsTargetData(e.data) : null;
      if (needed) setNeed(needed);
      else if (e instanceof ApiError && e.code === "HOST_NO_ROLES") setNoRoles(message(e));
      else if (e instanceof ApiError && e.code === "HARNESS_UNAVAILABLE" && data?.role) setRoleProblem({ code: e.code, message: message(e), role: data.role });
      else setError(message(e));
    } finally { setBusy(false); }
  };
  const harness = HARNESSES.find(h => h.id === draft.harness) ?? HARNESSES[0];
  // The preflight says what the task's target can start; any connected host counts only when there is no preflight.
  const choice = (id: HarnessId) => harnessChoice(pre, hosts, id);
  const available = (id: HarnessId) => choice(id).ok;
  // One line per distinct reason: a target that is not ready gives every harness the same one.
  const unavailable = [...new Set(HARNESSES.map(h => choice(h.id).reason).filter(Boolean))];
  const notes = preflightNotes(pre);
  // A blocking problem (the server offline, agents without Eridani tools) keeps Start off; a freestyle start lifts
  // the tools one. Warnings (a verify command not ready) only show as notes.
  const block = startBlock(pre, effective);
  const roleProblems = [...(pre?.problems ?? []).filter(p => p.role), ...(roleProblem ? [roleProblem] : [])];
  const blocked = (!!pre && !pre.active_run && !pre.needs_target && !pre.target) || !!block.problem;
  const setup = onSetup ? setupReason(pre, draft.harness) : "";
  const setupLink = setup && onSetup && <SetupLink reason={setup} onSetup={onSetup}/>;
  const interviewHint = roleRun(effective) && (draft.interview === "skip" || (pre && !pre.interview))
    ? "With a reviewer or a separate worker the planner always plans first, from the task as written."
    : draft.interview === "skip" ? "Starts on the task as written; no questions first."
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
          <span id={heading + "-agent"}>Planner</span>
          <div className="segmented" role="group" aria-labelledby={heading + "-agent"}>
            {HARNESSES.map(h => <button key={h.id} type="button" aria-pressed={draft.harness === h.id} disabled={!available(h.id)}
              title={choice(h.id).reason}
              onClick={() => setDraft(d => ({ ...d, harness: h.id, model: h.models[0].id }))}>{h.label}</button>)}
          </div>
          <select aria-label="Planner model" value={draft.model} onChange={e => setDraft(d => ({ ...d, model: e.target.value }))}>
            {harness.models.map(m => <option key={m.id} value={m.id}>{m.label}</option>)}
          </select>
          {unavailable.length > 0 && <p className="footnote">{unavailable.join(" ")}{pre?.harness_note ? ` ${pre.harness_note}` : ""}</p>}
          {notes.map(n => <p key={n} className="footnote run-dialog-note" role="note">{n}</p>)}
        </div>
        <RoleControls draft={draft} setDraft={change => { setRoleProblem(null); setDraft(change); }} choice={choice} roles={pre?.roles} rolesReason={noRoles}
          problems={roleProblems} idPrefix={heading}/>
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
        {block.problem && !block.problem.role && !notes.includes(block.problem.message) && !unavailable.includes(block.problem.message) && <p className="footnote run-dialog-note" role="note">{block.problem.message}</p>}
        {setupLink}
        <div className="settings-form-actions">
          <button className="btn btn-primary btn-sm" disabled={busy || blocked || !available(draft.harness)} onClick={() => void start()}>
            <Play size={14} aria-hidden="true"/>{busy ? "Starting…" : "Start"}</button>
          <button className="btn btn-ghost btn-sm" disabled={busy} onClick={onClose}>Cancel</button>
        </div>
      </>}
      {need && setupLink}
      {error && <p role="alert" className="work-error">{error}</p>}
    </>}
  </Dialog>;
}
