/* The run drawer's sectioned body (run-stages.ts holds the logic): the progress header with its orbit ring and stage
   track, the collapsible Plan / Run / Verify sections, the phased step list and the success criteria. */
import { useEffect, useId, useRef, type ReactNode, type RefObject } from "react";
import { Check, ChevronRight, X } from "lucide-react";
import { ago, attemptTitle, criteriaEvidence, evidenceAge, isSettled, latestAttempt, reviewPacket, runLabel, verifyLine, type RunDetail } from "./remotea-model";
import { activeStep, groupSteps, RUN_STAGES, runStage, STAGE_TITLES, stepProgress, stepWord, finished, type PlanStep, type RunStage } from "./run-stages";
import "./run-stages.css";

/** Percent of steps settled, as an orbit-stroked ring (the Today day ring's smaller sibling). */
export function ProgressRing({ done, total }: { done: number; total: number }) {
  const id = useId().replace(/:/g, "");
  const size = 44, stroke = 4, r = (size - stroke) / 2, c = 2 * Math.PI * r, center = size / 2;
  const share = total ? done / total : 0, percent = Math.round(share * 100);
  return <div className="run-ring" role="img" aria-label={total ? `${percent}% of steps done` : "No steps yet"}>
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden="true">
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2={size} y2={size} gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="var(--iri-1)"/><stop offset=".4" stopColor="var(--iri-2)"/>
          <stop offset=".75" stopColor="var(--iri-3)"/><stop offset="1" stopColor="var(--iri-4)"/>
        </linearGradient>
      </defs>
      <circle className="run-ring-track" cx={center} cy={center} r={r} fill="none" strokeWidth={stroke}/>
      {share > 0 && <circle className="run-ring-arc" cx={center} cy={center} r={r} fill="none" stroke={`url(#${id})`} strokeWidth={stroke}
        strokeLinecap="round" strokeDasharray={`${c * share} ${c}`} transform={`rotate(-90 ${center} ${center})`}/>}
    </svg>
    <span className="run-ring-value tabular" aria-hidden="true">{total ? <>{percent}<small>%</small></> : "–"}</span>
  </div>;
}

/** Above every view: the ring, the one status line, and where the run is in Plan, Run, Verify. */
export function StageHeader({ run, status }: { run: RunDetail; status: ReactNode }) {
  const steps = run.plan?.steps ?? [], { done, total } = stepProgress(steps);
  const stage = runStage(run), at = RUN_STAGES.indexOf(stage), complete = run.state === "completed";
  return <div className="run-stage-head">
    <ProgressRing done={done} total={total}/>
    <div className="run-stage-head-text">
      {status}
      <ol className="run-stage-track" aria-label="Stages">
        {RUN_STAGES.map((name, i) => {
          const state = complete || i < at ? "is-done" : i === at ? "is-current" : "";
          return <li key={name} className={state} aria-current={i === at ? "step" : undefined}>
            <span className="run-stage-dot" aria-hidden="true">{state === "is-done" ? <Check size={10} strokeWidth={3}/> : null}</span>{STAGE_TITLES[name]}
          </li>;
        })}
      </ol>
    </div>
  </div>;
}

/** One stage: a toggle row with its one-line summary (always shown), and its body when open (unmounted when closed). */
export function StageSection({ name, summary, open, attention = false, onToggle, children }: {
  name: RunStage; summary: string; open: boolean; attention?: boolean; onToggle: () => void; children: ReactNode;
}) {
  const id = useId();
  return <section className={`run-stage run-stage-${name}${open ? " is-open" : ""}${attention ? " needs-you" : ""}`} aria-labelledby={id + "-title"}>
    <h3 className="run-stage-title">
      <button type="button" id={id + "-title"} className="run-stage-toggle" aria-expanded={open} aria-controls={id + "-body"} onClick={onToggle}>
        <ChevronRight className="run-stage-chevron" size={16} aria-hidden="true"/>
        <span className="run-stage-name">{STAGE_TITLES[name]}</span>
        {attention && <span className="run-stage-flag">Needs you</span>}
        <span className="run-stage-summary" title={summary}>{summary}</span>
      </button>
    </h3>
    {open && <div id={id + "-body"} className="run-stage-body">{children}</div>}
  </section>;
}

function StepMarker({ status }: { status: string }) {
  return <span className="run-step-marker" aria-hidden="true">
    {status === "done" ? <Check size={12} strokeWidth={3}/> : status === "disputed" ? <X size={12} strokeWidth={3}/> : status === "skipped" ? "–" : null}
  </span>;
}

/** Brings the active step into view when it changes, while the Run section is open and chat is hidden. */
function useScrollActiveStep(list: RefObject<HTMLOListElement | null>, activeId: string | null, enabled: boolean) {
  const last = useRef<string | null>(null);
  useEffect(() => {
    if (!enabled) { last.current = null; return; }
    if (!activeId || activeId === last.current) return;
    last.current = activeId;
    const row = [...(list.current?.querySelectorAll<HTMLElement>("[data-step]") ?? [])].find(node => node.dataset.step === activeId);
    if (typeof row?.scrollIntoView !== "function") return;
    const reduce = typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
    row.scrollIntoView({ block: "nearest", behavior: reduce ? "auto" : "smooth" });
  }, [activeId, enabled]);
}

/** The plan as a vertical stepper: phases numbered 1..K, steps 1a, 1b …; done steps stay in place, dimmed. */
export function StepList({ steps, settled = false, scroll = false, compact = false }: {
  steps: PlanStep[]; settled?: boolean; scroll?: boolean; compact?: boolean;
}) {
  const list = useRef<HTMLOListElement>(null);
  const { flat, groups } = groupSteps(steps), active = activeStep(steps);
  useScrollActiveStep(list, active?.step.id ?? null, scroll && !compact);
  if (!steps.length) return <p className="footnote">This plan has no steps.</p>;
  return <ol ref={list} className={"run-steps" + (flat ? " is-flat" : "") + (compact ? " is-compact" : "")} aria-label="Steps">
    {groups.map(group => {
      const done = group.steps.filter(s => finished(s.step)).length;
      const current = !!active && group.steps.some(s => s.step.id === active.step.id);
      return <li key={group.index} className={"run-phase" + (done === group.steps.length ? " is-done" : current ? " is-current" : "")}>
        {!flat && <h4 className="run-phase-title">
          <span className="run-phase-index tabular" aria-hidden="true">{group.index}</span>
          <span className="run-phase-name">{group.label || `Phase ${group.index}`}</span>
          <span className="run-phase-count tabular">{done} of {group.steps.length}</span>
        </h4>}
        <ol className="run-phase-steps">
          {group.steps.map(({ label, step, prevDone }) => {
            const doing = !compact && active?.kind === "doing" && active.step.id === step.id;
            const next = !compact && !settled && active?.kind === "next" && active.step.id === step.id;
            const showNote = !compact && !!step.note && (step.status === "doing" || step.status === "disputed" || next);
            return <li key={step.id} data-step={step.id} data-prev-done={prevDone || undefined} aria-current={doing ? "step" : undefined}
              className={`run-step status-${["done", "doing", "skipped", "disputed"].includes(step.status) ? step.status : "todo"}${doing ? " is-active" : ""}${next ? " is-next" : ""}`}>
              <StepMarker status={step.status}/>
              <span className="run-step-label tabular">{label}</span>
              <span className="run-step-title">{step.title}<span className="sr-only">, {stepWord(step.status)}</span></span>
              {doing && <span className="run-step-chip">{settled ? "Last reported" : "Working"}</span>}
              {next && <span className="run-step-chip is-next">Next</span>}
              {showNote && <p className="run-step-note">{step.note}</p>}
              {(doing || step.status === "disputed") && step.updated_at && <small className="run-step-when">Reported {ago(step.updated_at)}</small>}
            </li>;
          })}
        </ol>
      </li>;
    })}
  </ol>;
}

const CRITERION_WORD = { met: "Met", not_met: "Not met", unclear: "Unclear", pending: "Not judged" } as const;
function CriterionMarker({ status }: { status: string }) {
  return <span className="run-criterion-marker" aria-hidden="true">
    {status === "met" ? <Check size={11} strokeWidth={3}/> : status === "not_met" ? <X size={11} strokeWidth={3}/> : status === "unclear" ? "?" : null}
  </span>;
}

/** Verify: each success criterion with the latest judge's verdict, the verify command, the latest results and the
 * outcome once the run settles. */
export function VerifyCriteria({ run }: { run: RunDetail }) {
  const rows = criteriaEvidence(run), met = rows.filter(r => r.status === "met").length;
  const judge = latestAttempt(run, "judge"), verify = latestAttempt(run, "verify");
  const judged = rows.some(r => r.status !== "pending");
  const line = verifyLine(run).text, packet = reviewPacket(run);
  const earlier = !!judge && judge.status === "done" && judge.plan_revision !== run.plan_revision;
  return <div className="run-verify">
    {isSettled(run) && <p className="run-outcome"><strong>{runLabel(run.state)}</strong>{run.finished_at && <small>{ago(run.finished_at)}</small>}</p>}
    {packet?.decision && run.state !== "awaiting_approval" && <p className={"run-outcome-note kind-" + packet.decision.kind}>{packet.decision.text}</p>}
    {rows.length ? <>
      <p className="run-criteria-count"><strong className="tabular">{judged ? `${met} of ${rows.length} met` : `${rows.length} to meet`}</strong>
        {judge?.turn_seq ? <small>judged after turn {judge.turn_seq}</small> : !judged ? <small>not judged yet</small> : null}</p>
      {earlier && <p className="footnote attention">Judged against an earlier plan: the criteria changed since. Judge again to check them.</p>}
      <ul className="run-criteria" aria-label="Success criteria">
        {rows.map(r => <li key={r.id} className={"run-criterion criterion-" + r.status}>
          <CriterionMarker status={r.status}/>
          <span className="run-criterion-text">{r.text}<span className="sr-only">, {CRITERION_WORD[r.status as keyof typeof CRITERION_WORD] ?? r.status}</span>
            {r.kind === "verify" && <small>needs {r.verify_id} to pass</small>}</span>
          {r.evidence && <p>{r.evidence}</p>}
        </li>)}
      </ul>
    </> : run.plan && <p className="footnote">No success criteria: the judge cannot complete this run; you decide.</p>}
    {line && <p className="run-verify-line">{line}</p>}
    {verify && <p className="run-verify-last">{attemptTitle(verify)}{evidenceAge(verify.updated_at || verify.created_at) && <small>{evidenceAge(verify.updated_at || verify.created_at)}</small>}</p>}
    {judge && (judge.summary || (judge.gap && judge.verdict !== "met")) && <div className="run-judge">
      <strong>{attemptTitle(judge)}</strong>
      {judge.summary && <p>{judge.summary}</p>}
      {judge.gap && judge.verdict !== "met" && <p>{judge.gap}</p>}
    </div>}
  </div>;
}
