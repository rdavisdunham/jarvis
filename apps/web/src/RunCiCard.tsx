import { ExternalLink } from "lucide-react";
import { ago, ciCard, ciChip, type Run, type RunCi } from "./remotea-model";

/** GitHub CI's state on the PR head as a chip (pending, passing, failing), linked to the checks when there is a link. */
export function RunCiChip({ ci }: { ci: RunCi | null | undefined }) {
  const chip = ciChip(ci);
  if (!chip) return null;
  const className = "run-ci-chip tone-" + chip.tone;
  return chip.href
    ? <a className={className} href={chip.href} target="_blank" rel="noreferrer" title={chip.title}>{chip.label}<ExternalLink size={12} aria-hidden="true"/></a>
    : <span className={className} title={chip.title}>{chip.label}</span>;
}

/** The drawer's CI card (run_ci.py): while the run awaits CI, how long it has waited and when it gives up; when CI
 * failed, the failed jobs and the log lines the agent was sent. Otherwise just the chip, when the run has CI. */
export function RunCiCard({ run, now = Date.now() }: { run: Pick<Run, "state" | "ci">; now?: number }) {
  const card = ciCard(run, now);
  if (!run.ci || !ciChip(run.ci)) return null;
  if (!card) return <p className="run-ci-line">GitHub CI <RunCiChip ci={run.ci}/>{run.ci.checked_at ? <small> · checked {ago(run.ci.checked_at, now)}</small> : null}</p>;
  return <section className={"remotea-question run-ci-card" + (card.failed.length ? " run-ci-failed" : "")} aria-label={card.title}>
    <p className="remotea-card-kicker">{card.title}</p>
    <p className="run-ci-wait">{card.line} <RunCiChip ci={run.ci}/></p>
    {card.error && <p className="footnote">The host could not read CI: {card.error}</p>}
    {!!card.failed.length && <ul className="run-ci-jobs" aria-label="Failed jobs">{card.failed.map(f =>
      <li key={f.name}>{f.link ? <a href={f.link} target="_blank" rel="noreferrer">{f.name}</a> : f.name}</li>)}</ul>}
    {!!card.log.length && <pre className="run-stop-output" aria-label="Failing log">{card.log.join("\n")}</pre>}
  </section>;
}
