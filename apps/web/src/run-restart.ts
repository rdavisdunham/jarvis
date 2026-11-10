// Restart run (docs/AGENT_CONTROL.md, "Restart run"; run_restart.py): continue a finished run as a new run linked to
// it. "Plan it" sends the note to planning; "Pilot it myself" is a freestyle run the owner talks to directly.
import { ago, isSettled, runLabel, TERMINAL_RUN_STATES, type Run } from "./remotea-model";

export type RestartMode = "plan" | "pilot";

export const RESTART_TITLES: Record<RestartMode | "button", string> = {
  button: "Continue this finished run as a new run on the same task, branch and worktree. Plan it: the agent plans "
    + "what you add and waits for your approval. Pilot it myself: you talk to the agent directly, with no plan or judge.",
  plan: "The agent plans the work your note asks for and waits for your approval, like a normal run. Needs a note.",
  pilot: "A freestyle run: no plan, no judge. You talk to the agent from here or the remotea terminal. Your note, if "
    + "any, is its first message; without one it waits for you.",
};

export const RESTART_NOTE_LIMIT = 4000;
export const NOTE_REQUIRED = "Say what the plan should add or change.";

/** The drawer's Restart run offer, or null when it is not shown: the run has not finished, the server does not offer
 * restart (older Eridani), or a run that continues this one is still working (its own drawer has the controls). */
export function restartOffer(run: Pick<Run, "state" | "restart" | "restarted_as">): { enabled: boolean; why: string | null } | null {
  if (!isSettled(run) || !run.restart) return null;
  if (run.restarted_as && !TERMINAL_RUN_STATES.includes(run.restarted_as.state)) return null;
  return { enabled: run.restart.available, why: run.restart.why };
}

/** Why the form cannot be sent with this mode and note, or null. */
export function restartProblem(mode: RestartMode, note: string): string | null {
  if (note.length > RESTART_NOTE_LIMIT) return `Keep the note under ${RESTART_NOTE_LIMIT} characters.`;
  return mode === "plan" && !note.trim() ? NOTE_REQUIRED : null;
}

/** POST /remotea/runs/{id}/restart's body (postAction adds the action_id). */
export function restartBody(run: Pick<Run, "state_revision">, mode: RestartMode, note: string) {
  return { mode, note: note.trim(), expected_revision: run.state_revision };
}

export type RestartLink = { id: string; label: string; text: string };

/** "Restarted from" (the finished run this one continues) and "Restarted as" (the newest run continuing this one). */
export function restartLinks(run: Pick<Run, "restarted_from" | "restarted_as">): { from: RestartLink | null; as: RestartLink | null } {
  const from = run.restarted_from, as = run.restarted_as;
  const how = from?.mode === "pilot" ? "piloted by you" : from?.mode === "plan" ? "planned" : "";
  const resumed = from?.resumed === true ? "resumed its conversation" : from?.resumed === false ? "fresh conversation" : "";
  return {
    from: from ? { id: from.id, label: "Restarted from",
      text: "the earlier run (" + runLabel(from.state) + (from.finished_at ? " " + ago(from.finished_at) : "") + ")"
        + [how, resumed].filter(Boolean).map(part => " · " + part).join("") } : null,
    as: as ? { id: as.id, label: "Restarted as", text: "a new run (" + runLabel(as.state) + ")" } : null,
  };
}
