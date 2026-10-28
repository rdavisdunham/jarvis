import { useCallback, useEffect, useRef, useState } from "react";
import { z } from "zod";
import { useEditor } from "./editor-control";
import { api } from "./api";
import { useStructureActions } from "./structure-actions";
import { setupTypeRows, type SetupProgress } from "./setup-model";
import type { Proposal } from "./structure-types";
import "./capture.css";

function SetupGuard({ dirty, busy, save, close }: {
  dirty: boolean; busy: boolean; save: () => Promise<unknown>; close: () => void;
}) {
  useEditor({ kind: "record", mode: "edit", dirty, busy, schema: z.object({}),
    values: { form: "Personal setup" },
    patch: async () => { throw Error("Use setup tools to save the user’s answers."); },
    save, close, discard: close });
  return null;
}

export function SetupProposalDetails({ proposal }: { proposal: Proposal }) {
  return <div className="setup-example">
    <h3>{proposal.applied_at ? "Applied organization" : "Preview — nothing added yet"}</h3>
    <p>These are types and allowed homes, not actual clients, projects or tasks. Existing records are retained.</p>
    <p>{proposal.impact.affected_count} existing records use changed definitions.
      {" "}{proposal.impact.blocking_count} blocking issues.</p>
    <details open><summary>Proposed types and homes</summary>
      <p className="footnote">Indentation is the type layout. Allowed homes can include other branches.</p>
      <ul className="setup-type-tree">{setupTypeRows(proposal.definition).map(({ type, depth, homes }) =>
        <li key={type.id} style={{ marginInlineStart: Math.min(depth, 6) * 16 }}>
          <strong>{type.name}</strong>{type.archived && <span className="chip">Archived</span>}
          <p>{type.description}</p>
          <p className="footnote">Allowed homes: {homes.length ? homes.join(", ") : "Top level only"}</p>
          <details><summary>Fields and behaviors</summary>
            <p>{type.capabilities.length ? type.capabilities.join(", ") : "Organizing record"}
              {type.review?.enabled ? " · Review every " + type.review.every : ""}</p>
            {!!type.statuses.length && <p>Statuses: {type.statuses.map(status => status.name).join(", ")}</p>}
            <ul>{type.fields.map(field => <li key={field.id}>
              <strong>{field.name}</strong> — {field.description}
              <span className="footnote"> ({field.kind}{field.archived ? ", archived" : ""})</span>
            </li>)}</ul>
          </details>
        </li>)}</ul>
    </details>
    {!!proposal.definition.relationships.length && <details><summary>Extra relationships</summary>
      <ul>{proposal.definition.relationships.map(link => <li key={link.id}>
        <strong>{link.name}</strong> — {link.description} ({link.cardinality}{link.archived ? ", archived" : ""})
      </li>)}</ul>
    </details>}
    {!!proposal.impact.issues.length && <ul role="status">{proposal.impact.issues.map((issue, index) =>
      <li key={index}>{issue.title ? issue.title + ": " : ""}{issue.message}</li>)}</ul>}
  </div>;
}

export function Onboarding({ compact = false, onChanged, onTalk }: {
  compact?: boolean; onChanged: () => void; onTalk: () => Promise<void>;
}) {
  const [progress, setProgress] = useState<SetupProgress | null>(null);
  const [draft, setDraft] = useState<SetupProgress | null>(null);
  const [open, setOpen] = useState(false), [dirty, setDirty] = useState(false);
  const [starting, setStarting] = useState(false), [proposal, setProposal] = useState<Proposal | null>(null);
  const { run, busy, error, setError } = useStructureActions();
  const latest = useRef<SetupProgress | null>(null), editVersion = useRef(0), loadVersion = useRef(0);
  const dirtyRef = useRef(false), startingRef = useRef(false);
  const load = useCallback(async () => {
    const version = ++loadVersion.current, edits = editVersion.current;
    const next = await api<SetupProgress>("/onboarding");
    const nextProposal = next.proposal_id ? await api<Proposal>("/structure/proposals/" + next.proposal_id) : null;
    if (version !== loadVersion.current || edits !== editVersion.current || dirtyRef.current ||
        next.revision < (latest.current?.revision ?? 0)) return;
    latest.current = next; setProgress(next); setDraft(next); setProposal(nextProposal);
  }, []);
  useEffect(() => { void load().catch(e => setError(e.message)); return () => { loadVersion.current += 1; }; }, [load]);
  useEffect(() => {
    if (!open || dirty || busy || starting) return;
    const timer = setInterval(() => void load().catch(e => setError(e.message)), 5000);
    return () => clearInterval(timer);
  }, [open, dirty, busy, starting, load]);

  if (!progress || !draft) return compact ? null : <p role="status">{error || "Loading setup…"}</p>;
  if (compact && !open && ["completed", "skipped"].includes(progress.status)) return null;
  const blocked = busy || starting;
  const field = (key: keyof SetupProgress, value: string) => {
    editVersion.current += 1; dirtyRef.current = true;
    setDraft({ ...draft, [key]: value }); setDirty(true); setProposal(null);
  };
  const save = async (status = "in_progress", step = draft.step) => {
    const edits = ++editVersion.current;
    const next = await run<SetupProgress>("onboarding.save", {
      expected_revision: progress.revision, status, step,
      preferred_name: draft.preferred_name, timezone: draft.timezone, purpose: draft.purpose,
      classification_name: draft.classification_name, description: draft.description,
      examples: draft.examples, design_notes: draft.design_notes,
    });
    latest.current = next; setProgress(next);
    if (edits === editVersion.current) {
      setDraft(next); setDirty(false); dirtyRef.current = false;
    }
    onChanged(); return next;
  };
  const preview = async () => {
    const saved = await save("in_progress", "organization");
    const result = await run<{ progress: SetupProgress; proposal: Proposal }>("onboarding.preview", { expected_revision: saved.revision });
    latest.current = result.progress; setProgress(result.progress); setDraft(result.progress); setProposal(result.proposal);
  };
  const complete = async () => {
    if (!proposal) return;
    if (!proposal.applied_at) {
      await run("structure.apply", { proposal_id: proposal.id, expected_revision: proposal.schema_revision });
      setProposal({ ...proposal, applied_at: new Date().toISOString() });
    }
    await save("completed", "done"); setOpen(false); onChanged();
  };
  const talk = async () => {
    if (startingRef.current || busy) return;
    startingRef.current = true; setStarting(true); setError("");
    try {
      if (dirty || progress.status !== "in_progress")
        await save("in_progress", progress.step === "done" ? "organization" : draft.step);
      // Publish the saved editor state before syncing screen context to the backend.
      await new Promise<void>(resolve => requestAnimationFrame(() => resolve()));
      await onTalk(); setOpen(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not start setup. Your saved answers are kept; retry when connected.");
    } finally {
      startingRef.current = false; setStarting(false);
    }
  };
  const reloadSaved = async () => {
    editVersion.current += 1; dirtyRef.current = false; setDirty(false); setError("");
    await load();
  };
  return <section className="setup-panel" aria-label="Set up with Eri">
    <h2>{progress.status === "completed" ? "Your setup" : "Make Eridani yours"}</h2>
    <p>Eri can help design your types, fields and homes, one question at a time. You review the structure before it changes.</p>
    {error && <div role="alert"><p>{error}</p>
      {open && <button className="btn" disabled={blocked} onClick={() => void reloadSaved().catch(e => setError(e.message))}>
        {dirty ? "Discard local changes and reload" : "Reload saved progress"}
      </button>}
    </div>}
    {!open ? <div className="setup-actions">
      <button className="btn btn-primary" disabled={blocked} onClick={() => void talk()}>
        {starting ? "Starting setup…" : progress.status === "not_started" ? "Set up with Eri" :
          progress.status === "completed" ? "Refine with Eri" : "Resume with Eri"}
      </button>
      <button className="btn" disabled={blocked} onClick={() => setOpen(true)}>Use setup form</button>
      {progress.status === "not_started" && <button className="btn" disabled={blocked}
        onClick={() => void save("skipped").catch(() => {})}>Skip for now</button>}
    </div> : <>
      <SetupGuard dirty={dirty} busy={blocked} save={() => save()} close={() => setOpen(false)}/>
      <p>Fill this in yourself, or save these answers and continue with Eri. Example names do not create real work.</p>
      <button className="btn" disabled={blocked} onClick={() => void talk()}>{starting ? "Starting setup…" : "Talk to Eri"}</button>
      <fieldset disabled={blocked}><legend>What should Eri know first?</legend>
        <label className="field">Preferred name<input aria-label="Setup name" value={draft.preferred_name} maxLength={100} onChange={e => field("preferred_name", e.target.value)}/></label>
        <label className="field">Time zone<input aria-label="Setup timezone" value={draft.timezone} onChange={e => field("timezone", e.target.value)}/></label>
        <label className="field">What do you want to track?<textarea aria-label="Setup purpose" value={draft.purpose} placeholder="Personal plans, work for clients, courses…" onChange={e => field("purpose", e.target.value)}/></label>
        <details open={draft.step !== "profile" || !!draft.classification_name}><summary>How would you like to organize it?</summary>
          <p>The form keeps your existing structure and can add one grouping. Talk to Eri to design several types, reusable fields and relationships.</p>
          <label className="field">Optional grouping name<input aria-label="Grouping name" placeholder="Course, Client, Property…" maxLength={80} value={draft.classification_name} onChange={e => field("classification_name", e.target.value)}/></label>
          {draft.classification_name && <>
            <label className="field">What belongs in this group?<textarea aria-label="Grouping description" minLength={20} maxLength={4500} value={draft.description} onChange={e => field("description", e.target.value)}/></label>
            <label className="field">Give one example and how a project relates to it<input aria-label="Grouping example" maxLength={450} value={draft.examples} onChange={e => field("examples", e.target.value)}/></label>
          </>}
          <label className="field">Additional structure notes<textarea aria-label="Setup design notes" maxLength={12000}
            value={draft.design_notes ?? ""} onChange={e => field("design_notes", e.target.value)}/></label>
          <p className="footnote">These answers guide organization separately from personal memory. Names such as ABC are records; Client is a type.</p>
        </details>
      </fieldset>
      <div className="setup-actions">
        <button className="btn" disabled={blocked || !draft.preferred_name.trim()} onClick={() => void save().catch(() => {})}>Save progress</button>
        <button className="btn btn-primary" disabled={blocked || !draft.preferred_name.trim()} onClick={() => void preview().catch(() => {})}>Preview my organization</button>
        <button className="btn" disabled={blocked} onClick={() => void save("skipped").then(() => setOpen(false)).catch(() => {})}>Save & finish later</button>
      </div>
      {proposal && !dirty && <>
        <SetupProposalDetails proposal={proposal}/>
        <button className="btn btn-primary" disabled={blocked || !!proposal.impact.blocking_count}
          onClick={() => void complete().catch(() => {})}>{proposal.applied_at ? "Finish setup" : "Confirm this setup"}</button>
      </>}
    </>}
  </section>;
}
