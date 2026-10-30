import { Play } from "lucide-react";
import { runChip } from "./remotea-model";
import { useRemotea, useRemoteaActions } from "./remotea-context";
import "./runs.css";

/** A task row's Run control: Play starts a run (the Run dialog asks what is needed); when the task already has a
 * run the chip says what it needs and opens it in the side panel. Nothing renders while remotea is off. */
export function RunButton({ taskId, title, className = "" }: { taskId: string; title: string; className?: string }) {
  const { enabled, status, runsByTask } = useRemotea();
  const { openRun, openStart } = useRemoteaActions();
  if (!enabled) return null;
  const chip = runChip(runsByTask.get(taskId), status?.hosts ?? []);
  if (chip.kind === "open") {
    return <button type="button" className={"run-chip tone-" + chip.tone + (className ? " " + className : "")}
      aria-label={chip.label + ": open the agent run for " + title} onClick={e => { e.stopPropagation(); openRun(chip.run); }}>
      <i className="run-dot" aria-hidden="true"/>{chip.label}</button>;
  }
  return <button type="button" className={"btn-icon run-start" + (className ? " " + className : "")} aria-label={"Run " + title + " with an agent"}
    title={chip.reason || "Run with an agent"} disabled={chip.disabled} onClick={e => { e.stopPropagation(); openStart(taskId); }}>
    <Play size={16} aria-hidden="true"/></button>;
}
