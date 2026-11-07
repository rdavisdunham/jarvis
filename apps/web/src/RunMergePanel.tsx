import { ExternalLink, GitMerge } from "lucide-react";
import { mergePanel, type Run } from "./remotea-model";

/** Merge from Eridani (run_merge.py), beside the CI card in Verify: Merge into main once CI is green (or the repository
 * has no CI), Update branch when main moved on, Send back for conflicts or failing CI. The server decides what is
 * allowed; this shows it. */
export function RunMergePanel({ run, disabled, onMerge, onUpdate, onSendBack }: {
  run: Pick<Run, "merge" | "state">; disabled: boolean;
  onMerge: (head: string) => void; onUpdate: (head: string) => void; onSendBack: (note: string) => void;
}) {
  const panel = mergePanel(run.merge);
  if (!panel || !run.merge) return null;
  const { pr } = run.merge;
  const head = pr?.head_sha ?? "";
  const settled = ["completed", "stopped", "capped", "failed"].includes(run.state);
  return <section className={"run-merge tone-" + panel.tone} aria-label="Merge">
    <div className="run-merge-row">
      {panel.action !== "none" && <button className={"btn btn-sm" + (panel.enabled ? " btn-primary" : "")} disabled={disabled || !panel.enabled || !head}
        onClick={() => panel.action === "update" ? onUpdate(head) : onMerge(head)}><GitMerge size={14} aria-hidden="true"/>{panel.label}</button>}
      <p className="run-merge-line" role="status">{panel.line}{pr?.url && <> <a href={pr.url} target="_blank" rel="noreferrer">PR #{pr.number}<ExternalLink size={12} aria-hidden="true"/></a></>}</p>
    </div>
    {panel.refused && <p className="footnote run-merge-refused">{panel.refused}</p>}
    {panel.sendBack && !settled && <button className="btn btn-sm" disabled={disabled} onClick={() => onSendBack(panel.sendBack!)}>
      {run.merge.status === "conflicts" ? "Send back to fix conflicts" : "Send back"}</button>}
  </section>;
}
