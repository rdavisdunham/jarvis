import { Play } from "lucide-react";
import { actorChip, runChip } from "./remotea-model";
import { useRemotea, useRemoteaActions } from "./remotea-context";
import "./runs.css";

/** A task row's Run control: Play starts a run (the Run dialog asks what is needed); when the task already has a
 * run the chip says what it needs and opens it in the side panel. On a project (`project`), Play opens a swarm plan
 * of its open tasks instead (Swarm.tsx); nothing starts until Start. Nothing renders while remotea is off. */
export function RunButton({ taskId, title, className = "", project = false }: { taskId: string; title: string; className?: string; project?: boolean }) {
  const { enabled, status, runsByTask } = useRemotea();
  const { openRun, openStart, openSwarm } = useRemoteaActions();
  if (!enabled) return null;
  const run = runsByTask.get(taskId);
  const chip = runChip(run, status?.hosts ?? []);
  if (chip.kind === "open") {
    // The tooltip says who acts on a hyperplan run and what the agent is doing now (the run summary), when known.
    const tip = [run ? actorChip(run) : null, run?.summary].filter(Boolean).join(" — ");
    return <button type="button" className={"run-chip tone-" + chip.tone + (className ? " " + className : "")}
      title={tip || undefined} aria-label={chip.label + ": open the agent run for " + title} onClick={e => { e.stopPropagation(); openRun(chip.run); }}>
      <i className="run-dot" aria-hidden="true"/>{chip.label}</button>;
  }
  return <button type="button" className={"btn-icon run-start" + (className ? " " + className : "")}
    aria-label={project ? "Plan a swarm of agents for " + title : "Run " + title + " with an agent"}
    title={chip.reason || (project ? "Plan a swarm: run this project's tasks with agents, in batches" : "Run with an agent")} disabled={chip.disabled}
    onClick={e => { e.stopPropagation(); if (project) openSwarm(taskId); else openStart(taskId); }}>
    <Play size={16} aria-hidden="true"/></button>;
}
