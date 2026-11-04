import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowLeft, BellRing, Check, RefreshCw } from "lucide-react";
import * as context from "./context-api";
import {
  chosen, DESTINATION_LABELS, effectiveAt, groupItems, meetingDate, meetingStatus, nextSelection, proposalGroup, PUBLICATION_LABELS,
  SECTION_LABELS, selectable, timecode,
  type ContextItem, type Destination, type Evidence, type Proposal, type ProposalGroup, type Review, type SourcePage, type Speaker,
  type TranscriptSegment,
} from "./context-model";

const sameSet = (a: Set<string>, b: Set<string>) => a.size === b.size && [...a].every(x => b.has(x));
type Props = { sourceId: string; zone?: string; onBack: () => void; onPerson: (actorId: string) => void };
const GROUPS: { id: ProposalGroup; title: string; hint: string }[] = [
  { id: "tasks", title: "Work to add", hint: "New tasks, or links to tasks you already have." },
  { id: "updates", title: "Changes to existing work", hint: "Applied separately, only when you choose to." },
  { id: "suggestions", title: "Suggested next steps", hint: "Nobody promised these; add them only if you want to." },
];

/** One meeting's review: brief, proposed work, extracted context with corrections, speakers and the transcript. */
export function MeetingReview({ sourceId, zone, onBack, onPerson }: Props) {
  const [review, setReview] = useState<Review | null>(null), [page, setPage] = useState<SourcePage | null>(null);
  const [segments, setSegments] = useState<TranscriptSegment[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set()), [destination, setDestination] = useState<Destination | null>(null);
  const [error, setError] = useState(""), [stale, setStale] = useState(false), [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false), [focus, setFocus] = useState<{ assertion?: string; segment?: number } | null>(null);
  const [finishing, setFinishing] = useState(false);
  const current = useRef<Review | null>(null);

  const apply = useCallback((next: Review) => {
    setSelected(previous => nextSelection(previous, current.current?.proposals ?? null, next.proposals));
    current.current = next; setReview(next); setStale(false); setUpdated(false);
  }, []);
  const load = useCallback(async () => {
    try { apply(await context.getReview(sourceId)); setError(""); }
    catch (e) { setError((e as Error).message); }
  }, [sourceId, apply]);
  useEffect(() => {
    void load();
    context.transcript(sourceId).then(p => { setPage(p); setSegments(p.segments); }).catch(e => setError((e as Error).message));
  }, [sourceId, load]);

  // A context change elsewhere (processing finished, another device, publication progress): reload when the reader has
  // nothing unsaved here; otherwise ask, so their selections are not replaced under them.
  const dirty = !!review && (destination !== null || !sameSet(selected, nextSelection(new Set(), null, review.proposals)));
  const [updated, setUpdated] = useState(false);
  context.useContextChanged(() => {
    if (busy) return;  // the write in flight returns the latest review
    if (dirty) setUpdated(true); else void load();
  });

  /** Every write: apply the review the server returns (with its new revisions), or reload; never guess. */
  async function act(run: () => Promise<context.WriteResult>, done: string) {
    setBusy(true); setNotice(""); setError("");
    try {
      const result = await run();
      if (result.review) apply(result.review); else await load();
      setNotice(done);
      return true;
    } catch (e) {
      if (context.isStale(e)) setStale(true); else setError((e as Error).message);
      return false;
    } finally { setBusy(false); }
  }

  async function reveal(seq: number, assertion?: string) {
    setFocus({ assertion, segment: seq });
    let loaded = segments, next = page?.next_segment ?? null;
    while (next !== null && !loaded.some(s => s.seq === seq) && next <= seq) {
      try {
        const more = await context.transcript(sourceId, next, Math.min(seq + 10, next + 49));
        loaded = [...loaded, ...more.segments]; next = more.next_segment;
        setSegments(loaded); setPage(p => p && { ...p, next_segment: more.next_segment });
      } catch (e) { setError((e as Error).message); break; }
    }
    requestAnimationFrame(() => document.getElementById(`segment-${sourceId}-${seq}`)?.scrollIntoView?.({ block: "center", behavior: "smooth" }));
  }
  async function more() {
    if (page?.next_segment == null) return;
    try {
      const next = await context.transcript(sourceId, page.next_segment);
      setSegments(rows => [...rows, ...next.segments]); setPage(p => p && { ...p, next_segment: next.next_segment });
    } catch (e) { setError((e as Error).message); }
  }

  const title = review?.source?.title || page?.source.title || "Meeting";
  const occurred = review?.source?.occurred_at ?? page?.source.occurred_at ?? null;
  const state = review?.source ? meetingStatus(review.source) : null;
  const target = destination ?? review?.destination ?? "eridani";
  const toAdd = review ? chosen(review.proposals, selected, "approvable") : [];
  const toUpdate = review ? chosen(review.proposals, selected, "updates") : [];
  const open = review ? review.proposals.filter(selectable) : [];
  const toggle = (id: string, on: boolean) => setSelected(s => { const next = new Set(s); if (on) next.add(id); else next.delete(id); return next; });

  return <section className="meetings-page meeting-review" aria-label="Meeting review">
    <header className="meeting-review-head">
      <button className="btn btn-ghost btn-sm" onClick={onBack}><ArrowLeft size={16}/>Meetings</button>
      <div className="meeting-review-title">
        <h2>{title}</h2>
        <span className="meeting-row-meta"><time dateTime={occurred ?? undefined}>{meetingDate(occurred, zone)}</time>
          {state && <span className={"chip meeting-status tone-" + state.tone}>{state.label}</span>}
          {review?.status === "done" && <span className="chip">Reviewed</span>}</span>
        {state && state.tone !== "ok" && state.detail && <p className="meeting-row-detail">{state.detail}</p>}
      </div>
      {review && <div className="meeting-review-actions">
        <button className="btn btn-sm" disabled={busy || stale} onClick={() => void act(() => context.later(sourceId), "Saved for later. It will be in today's evening digest.").then(ok => ok && onBack())}><BellRing size={15}/>Remind me later</button>
        <button className="btn btn-sm btn-soft" disabled={busy || stale} onClick={() => toAdd.length || toUpdate.length ? setFinishing(true)
          : void act(() => context.done(sourceId, review.revision, open.map(p => ({ proposal_id: p.id, revision: p.revision }))), "Review finished.").then(ok => ok && onBack())}><Check size={15}/>Done</button>
      </div>}
    </header>
    {finishing && review && <div className="meeting-confirm" role="alertdialog" aria-label="Finish without adding">
      <p>{toAdd.length + toUpdate.length} selected {toAdd.length + toUpdate.length === 1 ? "item has" : "items have"} not been added. Finishing dismisses every proposal still open; the meeting's context stays.</p>
      <button className="btn btn-sm btn-primary" disabled={busy} onClick={() => { setFinishing(false); void act(() => context.done(sourceId, review.revision, open.map(p => ({ proposal_id: p.id, revision: p.revision }))), "Review finished.").then(ok => ok && onBack()); }}>Finish anyway</button>
      <button className="btn btn-sm" onClick={() => setFinishing(false)}>Keep reviewing</button>
    </div>}
    {stale && <div className="error-banner meeting-stale" role="alert"><span>This meeting changed since you opened it, so nothing was saved. Refresh to review the latest version; your selections are kept where nothing changed.</span>
      <button className="btn btn-sm" onClick={() => void load()}><RefreshCw size={14}/>Refresh</button></div>}
    {updated && !stale && <div className="meeting-updated" role="status"><span>This meeting has new changes. Refresh to see them; selections on unchanged proposals are kept.</span>
      <button className="btn btn-sm" onClick={() => void load()}><RefreshCw size={14}/>Refresh</button></div>}
    {error && <p role="alert" className="error-banner">{error}</p>}
    {notice && <p role="status" className="meeting-notice">{notice}</p>}
    {!review ? !error && <p role="status" className="meetings-hint">Loading the review…</p> : <div className="meeting-review-body">
      <div className="meeting-review-main">
        <BriefPanel review={review} onCite={reveal}/>
        <section className="meeting-section" aria-labelledby={`proposals-${sourceId}`}>
          <h3 id={`proposals-${sourceId}`}>Proposed work</h3>
          {!review.proposals.length && <p className="meetings-hint">Nothing to add from this meeting.</p>}
          {GROUPS.map(group => {
            const rows = review.proposals.filter(p => proposalGroup(p) === group.id);
            return !!rows.length && <div className="proposal-group" key={group.id} role="group" aria-label={group.title}>
              <h4>{group.title}</h4><p className="meetings-hint">{group.hint}</p>
              {rows.map(p => <ProposalCard key={p.id} proposal={p} checked={selected.has(p.id)} disabled={busy || stale}
                onToggle={on => toggle(p.id, on)} onCite={seq => void reveal(seq)} zone={zone}/>)}
            </div>;
          })}
          {!!review.proposals.length && <div className="proposal-footer">
            <label className="meeting-destination">Add to<select aria-label="Destination" value={target} disabled={busy || stale}
              onChange={e => setDestination(e.target.value as Destination)}>
              {(review.destinations?.length ? review.destinations : ["eridani" as Destination]).map(d => <option key={d} value={d}>{DESTINATION_LABELS[d] ?? d}</option>)}
            </select></label>
            <button className="btn btn-primary" disabled={busy || stale || !toAdd.length}
              onClick={() => void act(() => context.approve(sourceId, target, toAdd), `Adding ${toAdd.length} to ${DESTINATION_LABELS[target]}.`).then(ok => ok && setDestination(null))}>
              Add selected tasks{toAdd.length ? ` (${toAdd.length})` : ""}</button>
            <button className="btn" disabled={busy || stale || !toUpdate.length}
              onClick={() => void act(() => context.applyUpdates(sourceId, toUpdate), `Applying ${toUpdate.length} ${toUpdate.length === 1 ? "update" : "updates"}.`)}>
              Apply selected updates{toUpdate.length ? ` (${toUpdate.length})` : ""}</button>
          </div>}
        </section>
        <section className="meeting-section" aria-labelledby={`items-${sourceId}`}>
          <h3 id={`items-${sourceId}`}>What Eridani found</h3>
          {!review.items.length && <p className="meetings-hint">No decisions, commitments or other context were found.</p>}
          {groupItems(review.items).map(group => <div key={group.kind} className="context-group" role="group" aria-label={group.label}>
            <h4>{group.label}</h4>
            {group.items.map(item => <ContextRow key={item.id} item={item} names={review.names ?? {}} disabled={busy || stale} focused={focus?.assertion === item.id}
              onPerson={onPerson} onReveal={seq => void reveal(seq, item.id)}
              onWrong={reason => act(() => context.wrong(sourceId, item.id, item.revision, reason), "Marked as wrong. Proposals based on it were withdrawn.")}
              onChanged={(statement, since) => act(() => context.changed(sourceId, item.id, item.revision, statement, since), "Recorded the change.")}/>)}
          </div>)}
        </section>
        {!!review.speakers.length && <SpeakerPanel speakers={review.speakers} disabled={busy || stale} onPerson={onPerson}
          onAction={(s, action) => act(() => context.speaker(sourceId, s.link_id!, s.link_revision ?? 0, action, s.decision_id),
            action === "undo" ? "Undid that speaker change." : action === "confirm" ? `Confirmed ${s.label}.` : `${s.label} is no longer matched.`)}/>}
      </div>
      <TranscriptPanel sourceId={sourceId} page={page} segments={segments} focus={focus} onMore={() => void more()}/>
    </div>}
  </section>;
}

function BriefPanel({ review, onCite }: { review: Review; onCite: (seq: number, assertion?: string) => void }) {
  const sections = (review.brief?.sections ?? []).filter(s => s.claims.length);
  if (!sections.length && !review.brief?.gaps.length) return null;
  return <section className="meeting-section meeting-brief" aria-label="Brief">
    <h3>Brief</h3>
    {sections.map(section => <div key={section.name}><h4>{SECTION_LABELS[section.name] ?? section.name.replaceAll("_", " ")}</h4>
      <ul>{section.claims.map((claim, i) => { const cite = claim.citations[0]; return <li key={i}>{claim.text}
        {cite && <button className="text-button" onClick={() => onCite(cite.segment, cite.assertion_id)}>Source</button>}</li>; })}</ul></div>)}
    {review.brief?.gaps.map(gap => <p key={gap} className="meetings-hint">{gap}</p>)}
  </section>;
}

function ProposalCard({ proposal: p, checked, disabled, onToggle, onCite, zone }: {
  proposal: Proposal; checked: boolean; disabled: boolean; onToggle: (on: boolean) => void; onCite: (seq: number) => void; zone?: string;
}) {
  const open = selectable(p), verb = p.kind === "update" ? "Apply" : p.kind === "link" ? "Link" : "Add";
  return <article className={"proposal-card" + (open ? "" : " is-settled")}>
    <label className="check-label">
      <input type="checkbox" checked={open && checked} disabled={disabled || !open} onChange={e => onToggle(e.target.checked)}
        aria-label={`${verb}: ${p.title}`}/>
      <strong>{p.title}</strong>
    </label>
    <div className="chip-row">
      {p.kind === "link" && p.target && <span className="chip">Links to “{p.target.title}”</span>}
      {p.assignee && <span className="chip">{p.assignee.name}</span>}
      {p.project && <span className="chip">{p.project.name}</span>}
      {p.due_at && <span className="chip">Due {new Date(p.due_at).toLocaleDateString([], { month: "short", day: "numeric", ...(zone ? { timeZone: zone } : {}) })}</span>}
      {p.publication ? <span className={"chip publication-" + p.publication.status}>{PUBLICATION_LABELS[p.publication.status] ?? p.publication.status}{p.publication.status === "published" ? ` to ${DESTINATION_LABELS[p.destination]}` : ""}</span>
        : !open && <span className="chip">{p.disposition === "dismissed" ? "Dismissed" : p.disposition === "superseded" ? "Replaced" : p.disposition}</span>}
      {p.disposition === "later" && <span className="chip">For later</span>}
    </div>
    {p.reason && <p className="proposal-reason">{p.reason}</p>}
    {p.publication?.error && <p className="proposal-reason" role="note">{p.publication.error}</p>}
    {p.kind === "update" && p.before && p.after && <dl className="proposal-diff" aria-label="Before and after">
      {Object.keys({ ...p.before, ...p.after }).map(key => <div key={key}><dt>{key.replaceAll("_", " ")}</dt>
        <dd><del>{String(p.before?.[key] ?? "none")}</del> → <ins>{String(p.after?.[key] ?? "none")}</ins></dd></div>)}
    </dl>}
    {p.evidence.map((e, i) => <blockquote key={i}>{e.speaker && <cite>{e.speaker}: </cite>}{e.quote}
      {e.segment !== null && <button className="text-button" onClick={() => onCite(e.segment!)}>In transcript</button>}</blockquote>)}
  </article>;
}

function ContextRow({ item, names, disabled, focused, onPerson, onReveal, onWrong, onChanged }: {
  item: ContextItem; names: Record<string, string>; disabled: boolean; focused: boolean; onPerson: (actorId: string) => void;
  onReveal: (seq: number) => void; onWrong: (reason: string) => Promise<boolean>; onChanged: (statement: string, since: string) => Promise<boolean>;
}) {
  const [mode, setMode] = useState<"" | "wrong" | "changed">(""), [statement, setStatement] = useState(item.statement), [since, setSince] = useState("");
  const [proof, setProof] = useState<Evidence | null>(null), [proofError, setProofError] = useState(""), [showProof, setShowProof] = useState(false);
  const ended = item.status === "retracted" || item.status === "historical";
  async function toggleProof() {
    setShowProof(v => !v);
    if (proof || showProof) return;
    try { setProof(await context.evidence(item.id)); } catch (e) { setProofError((e as Error).message); }
  }
  return <article className={"context-row" + (focused ? " is-focused" : "") + (ended ? " is-ended" : "")}>
    <p>{item.statement}</p>
    <div className="chip-row">
      {item.basis === "inferred" && <span className="chip badge-inferred" title="Eridani inferred this; nobody said it outright.">Inferred</span>}
      {item.status === "contested" && <span className="chip badge-contested" title="Meetings disagree about this. Questions asks you to settle it.">Contested</span>}
      {item.modality && item.modality !== "asserted" && <span className="chip">{item.modality}</span>}
      {ended && <span className="chip">{item.status === "retracted" ? "Marked wrong" : "No longer true"}</span>}
      {item.entities.map((e, i) => e.entity_type === "actor" && e.entity_id
        ? <button key={i} className="chip" onClick={() => onPerson(e.entity_id!)}>{names[`actor:${e.entity_id}`] ?? "Person"}</button>
        : <span key={i} className="chip">{e.entity_id ? names[`${e.entity_type}:${e.entity_id}`] ?? e.entity_type : e.mention}</span>)}
    </div>
    <div className="context-row-actions">
      <button className="text-button" aria-expanded={showProof} onClick={() => void toggleProof()}>Evidence</button>
      {item.citation && <button className="text-button" onClick={() => onReveal(item.citation!.segment)}>In transcript</button>}
      {!ended && <><button className="text-button" disabled={disabled} onClick={() => setMode(mode === "wrong" ? "" : "wrong")}>Wrong</button>
        <button className="text-button" disabled={disabled} onClick={() => setMode(mode === "changed" ? "" : "changed")}>Changed</button></>}
    </div>
    {showProof && <div className="context-evidence" aria-label="Evidence">
      {proofError ? <p role="alert">{proofError}</p> : !proof ? <p role="status" className="meetings-hint">Loading evidence…</p> : <>
        {proof.evidence.map(span => <blockquote key={span.evidence_id} className={span.role === "contradiction" ? "is-contradiction" : ""}>
          {span.role === "contradiction" && <strong>Contradicts: </strong>}{span.speaker && <cite>{span.speaker}: </cite>}{span.quote}
          <small> · {span.source_title}{span.occurred_at ? ", " + new Date(span.occurred_at).toLocaleDateString() : ""}</small></blockquote>)}
        {proof.gaps.map(gap => <p key={gap} className="meetings-hint">{gap}</p>)}
      </>}
    </div>}
    {mode === "wrong" && <form className="context-correction" onSubmit={e => { e.preventDefault(); void onWrong("").then(ok => ok && setMode("")); }}>
      <p>Mark this as wrong? It leaves your context, Eridani will not propose it again from this meeting, and work proposed from it is withdrawn.</p>
      <div><button className="btn btn-sm btn-danger" disabled={disabled}>Mark wrong</button><button type="button" className="btn btn-sm" onClick={() => setMode("")}>Cancel</button></div>
    </form>}
    {mode === "changed" && <form className="context-correction" onSubmit={e => { e.preventDefault(); const at = effectiveAt(since); if (at) void onChanged(statement.trim(), at).then(ok => ok && setMode("")); }}>
      <p>It was right then but is different now. The old version stays in history.</p>
      <label className="field">What is it now?<textarea rows={2} value={statement} maxLength={2000} onChange={e => setStatement(e.target.value)}/></label>
      <label className="field">Since<input type="date" required value={since} onChange={e => setSince(e.target.value)}/></label>
      <div><button className="btn btn-sm btn-primary" disabled={disabled || !statement.trim() || statement.trim() === item.statement || !since}>Save change</button>
        <button type="button" className="btn btn-sm" onClick={() => setMode("")}>Cancel</button></div>
    </form>}
  </article>;
}

function SpeakerPanel({ speakers, disabled, onPerson, onAction }: {
  speakers: Speaker[]; disabled: boolean; onPerson: (actorId: string) => void; onAction: (s: Speaker, action: "confirm" | "reject" | "undo") => void;
}) {
  return <section className="meeting-section" aria-label="Speakers">
    <h3>Speakers</h3>
    <ul className="speaker-list">{speakers.map(s => <li key={s.label}>
      <span className="speaker-label">{s.label}</span>
      {s.actor_id && s.status !== "rejected" ? <button className="chip" onClick={() => onPerson(s.actor_id!)}>{s.actor_name ?? "Person"}</button>
        : <span className="meetings-hint">{s.status === "rejected" ? "Not matched (you said no)" : "Not matched"}</span>}
      {s.status === "inferred" && <span className="chip badge-inferred" title="Matched by Eridani, not confirmed by you.">Inferred</span>}
      {s.status === "confirmed" && <span className="chip">Confirmed</span>}
      {s.link_id && <span className="speaker-actions">
        {s.actor_id && (s.status === "inferred" || s.status === "exact") && <>
          <button className="text-button" disabled={disabled} onClick={() => onAction(s, "confirm")}>Confirm</button>
          <button className="text-button" disabled={disabled} onClick={() => onAction(s, "reject")}>Not them</button></>}
        {s.decision_id && (s.status === "confirmed" || s.status === "rejected") && <button className="text-button" disabled={disabled} onClick={() => onAction(s, "undo")}>Undo</button>}
      </span>}
    </li>)}</ul>
  </section>;
}

function TranscriptPanel({ sourceId, page, segments, focus, onMore }: {
  sourceId: string; page: SourcePage | null; segments: TranscriptSegment[]; focus: { assertion?: string; segment?: number } | null; onMore: () => void;
}) {
  return <section className="meeting-section meeting-transcript" aria-label="Transcript">
    <h3>Transcript</h3>
    {!page ? <p role="status" className="meetings-hint">Loading the transcript…</p> : <>
      {page.gaps.map(gap => <p key={gap} className="meetings-hint">{gap}</p>)}
      <ol className="transcript-list">{segments.map(s => {
        const supported = s.supports.some(x => x.status !== "retracted");
        const focused = focus?.segment === s.seq || (!!focus?.assertion && s.supports.some(x => x.assertion_id === focus.assertion));
        return <li key={s.seq} id={`segment-${sourceId}-${s.seq}`} className={"transcript-segment" + (supported ? " is-evidence" : "") + (focused ? " is-focused" : "")}
          aria-current={focused ? "true" : undefined}>
          <span className="transcript-who">{s.kind === "private_note" ? "Your note" : s.speaker ?? "Unknown speaker"}{s.start_ms !== null && <small>{timecode(s.start_ms)}</small>}</span>
          <span className="transcript-text">{s.text}</span>
          {supported && <span className="sr-only"> (evidence for {s.supports.length} {s.supports.length === 1 ? "item" : "items"})</span>}
        </li>;
      })}</ol>
      {page.next_segment !== null && <button className="btn btn-sm" onClick={onMore}>Load more of the transcript</button>}
    </>}
  </section>;
}
