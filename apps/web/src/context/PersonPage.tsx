import { useCallback, useEffect, useState } from "react";
import { ArrowLeft, Check, RefreshCw, UserRound, X } from "lucide-react";
import * as context from "./context-api";
import type { OpenLoop, PersonBrief, PersonChoice, PersonCitation, PersonView } from "./context-api";
import { KIND_LABELS, SECTION_LABELS, type Evidence } from "./context-model";
import "./meetings.css";

type Props = { actorId: string | null; onMeetings: () => void; onMeeting: (sourceId: string) => void; onPerson?: (actorId: string) => void };
const LOOP_GROUPS: { id: string; title: string; hint: string; buckets: OpenLoop["bucket"][] }[] = [
  { id: "own", title: "You promised", hint: "Work you took on.", buckets: ["own"] },
  { id: "waiting", title: "Waiting for", hint: "Work someone promised you.", buckets: ["waiting_for"] },
  { id: "others", title: "Shared or between others", hint: "Work promised by a group, or to someone else.", buckets: ["collective", "between_others"] },
];
const NAMESPACES: Record<string, string> = { email: "Email", alias: "Also called", calendar_name: "Calendar name",
  granola_speaker: "Speaker label", linear_user: "Linear user", organization: "Organization" };
const TASK_STATES: Record<string, string> = { none: "No task", open: "Task open", covered: "Task done", cancelled: "Task cancelled", withheld: "Task hidden" };
const day = (value: string | null) => value ? new Date(value).toLocaleDateString([], { month: "short", day: "numeric", year: "numeric" }) : "Undated";
type Merging = { other: string; otherName: string; keep: "this" | "other" };

/** A person page: what meetings say about someone, their open work, meetings, and who they are (identity corrections).
 * Reads only the context API; every correction is a human decision sent with its own command ID. */
export function PersonPage({ actorId, onMeetings, onMeeting, onPerson }: Props) {
  const [shown, setShown] = useState(actorId);
  const [view, setView] = useState<PersonView | null>(null), [brief, setBrief] = useState<PersonBrief | null>(null);
  const [loops, setLoops] = useState<{ items: OpenLoop[]; gaps: string[] } | null>(null);
  const [error, setError] = useState(""), [partial, setPartial] = useState<string[]>([]), [off, setOff] = useState(false);
  const [stale, setStale] = useState(false), [notice, setNotice] = useState(""), [busy, setBusy] = useState(false);
  useEffect(() => setShown(actorId), [actorId]);

  const loadRest = useCallback(async (actor: string) => {
    const [b, l] = await Promise.allSettled([context.personBrief(actor), context.personLoops(actor)]);
    setBrief(b.status === "fulfilled" ? b.value : null); setLoops(l.status === "fulfilled" ? l.value : null);
    setPartial([b.status === "rejected" && "The brief could not be loaded.", l.status === "rejected" && "Open work could not be loaded."].filter(Boolean) as string[]);
  }, []);
  const load = useCallback(async () => {
    if (!shown) return;
    setError(""); setStale(false);
    try { setView(await context.person(shown)); }
    catch (e) { if (context.isUnavailable(e)) setOff(true); else setError((e as Error).message); return; }
    await loadRest(shown);
  }, [shown, loadRest]);
  useEffect(() => { setView(null); setBrief(null); setLoops(null); setNotice(""); void load(); }, [load]);

  function go(actor: string) { if (onPerson) onPerson(actor); else setShown(actor); }
  /** Every correction: show the page the server returns (the person kept, after a merge), then refresh brief and loops. */
  async function act(run: () => Promise<context.PersonResult>, done: string) {
    setBusy(true); setNotice(""); setError("");
    try {
      const { person } = await run();
      if (person.actor.id !== shown) { if (onPerson) onPerson(person.actor.id); else setShown(person.actor.id); }
      else { setView(person); await loadRest(person.actor.id); }
      setNotice(done);
      return true;
    } catch (e) {
      if (context.isStale(e)) setStale(true); else setError((e as Error).message);
      return false;
    } finally { setBusy(false); }
  }

  if (off) return <section className="meetings-page" aria-label="Person"><p className="empty">Meeting memory is not on for this workspace. It works in your personal workspace once it is turned on.</p></section>;
  if (!shown) return <section className="meetings-page" aria-label="Person">
    <p className="empty"><UserRound size={20} aria-hidden/>Choose a person from a meeting to see what you know about them.
      <button className="btn btn-sm" onClick={onMeetings}>Open Meetings</button></p></section>;

  const name = view?.actor.name ?? "Person";
  const mergedInto = view?.actor.archived ? view.merges.find(m => m.loser_actor_id === view.actor.id && !m.undone_at) : undefined;
  const relationships = view?.context.filter(c => c.kind === "relationship") ?? [];
  const buckets = view?.buckets ?? [];
  /** Confirm or remove a person's link to a client or project, then reload the page. */
  async function linkBucket(recordId: string, status: "confirmed" | "rejected", done: string) {
    if (!view) return;
    setBusy(true); setNotice(""); setError("");
    try {
      await context.linkBucket(recordId, view.actor.id, status);
      setView(await context.person(view.actor.id)); setNotice(done);
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  }
  const observations = view?.context.filter(c => c.kind !== "relationship") ?? [];
  return <section className="meetings-page meeting-review person-page" aria-label="Person">
    <header className="meeting-review-head">
      <button className="btn btn-ghost btn-sm" onClick={onMeetings}><ArrowLeft size={16}/>Meetings</button>
      <div className="meeting-review-title">
        <h2>{view?.actor.is_me ? "You" : name}</h2>
        <span className="meeting-row-meta">
          {view?.actor.is_me && <span className="chip">This is you</span>}
          {view?.actor.archived && <span className="chip">{mergedInto ? "Merged" : "Archived"}</span>}
        </span>
      </div>
      <div className="meeting-review-actions"><button className="btn btn-sm" disabled={busy} onClick={() => void load()}><RefreshCw size={14}/>Refresh</button></div>
    </header>
    {mergedInto && <p className="meeting-notice">This record was merged into{" "}
      <button className="text-button" onClick={() => go(mergedInto.winner_actor_id)}>{mergedInto.winner_name ?? "another person"}</button>. Undo the merge below to separate them again.</p>}
    {stale && <div className="error-banner meeting-stale" role="alert"><span>This changed since you opened it, so nothing was saved. Refresh and try again.</span>
      <button className="btn btn-sm" onClick={() => void load()}><RefreshCw size={14}/>Refresh</button></div>}
    {error && <p role="alert" className="error-banner">{error}</p>}
    {notice && <p role="status" className="meeting-notice">{notice}</p>}
    {!view ? !error && <p role="status" className="meetings-hint">Loading…</p> : <div className="meeting-review-main">
      {partial.map(p => <p key={p} className="meetings-hint" role="note">{p}</p>)}
      <BriefSection brief={brief} onMeeting={onMeeting}/>
      <LoopsSection loops={loops} onMeeting={onMeeting}/>

      <section className="meeting-section" aria-label="Projects and relationships">
        <h3>Projects and relationships</h3>
        {!buckets.length && !view.projects.length && !relationships.length && <p className="meetings-hint">No projects or relationships recorded yet.</p>}
        {!!buckets.length && <div className="chip-row" aria-label="Clients and projects">{buckets.map(b => <span key={b.record_id}
          className={"chip" + (b.status === "confirmed" ? " chip-done" : "")}
          title={[`${b.tasks} task(s), ${b.meeting_facts} meeting fact(s)`, ...(b.evidence.tasks ?? []), ...(b.evidence.meeting_facts ?? [])].join("\n")}>
          {b.title}
          {b.status !== "confirmed" && <button className="chip-action" aria-label={`Confirm ${b.title}`} title="Confirm this link" disabled={busy}
            onClick={() => void linkBucket(b.record_id, "confirmed", `Linked to ${b.title}.`)}><Check size={12}/></button>}
          <button className="chip-action" aria-label={`Remove ${b.title}`} title="Not connected: remove this link" disabled={busy}
            onClick={() => void linkBucket(b.record_id, "rejected", `Removed ${b.title}.`)}><X size={12}/></button>
        </span>)}</div>}
        {!buckets.length && !!view.projects.length && <div className="chip-row">{view.projects.map(p => <span key={p.project_id} className="chip"
          title={`${p.context_items} meeting item(s), ${p.tasks} task(s)`}>{p.name}</span>)}</div>}
        {relationships.map(c => <Observation key={c.assertion_id} item={c} onMeeting={onMeeting}/>)}
      </section>

      <section className="meeting-section" aria-label="Work and observations">
        <h3>Work and observations</h3>
        {!observations.length && <p className="meetings-hint">Meetings have not said anything else about {view.actor.is_me ? "you" : name} yet.</p>}
        {observations.map(c => <Observation key={c.assertion_id} item={c} onMeeting={onMeeting}/>)}
        {!!view.tasks.length && <><h4>{view.actor.is_me ? "Your tasks" : `Tasks assigned to ${name}`}</h4>
          <ul className="person-tasks">{view.tasks.map(t => <li key={t.id}>{t.status === "completed" ? <del>{t.title}</del> : t.title}
            {t.due_date && <span className="chip">Due {day(t.due_date)}</span>}{t.status !== "open" && <span className="chip">{t.status}</span>}</li>)}</ul></>}
      </section>

      <section className="meeting-section" aria-label="Meetings">
        <h3>Meetings</h3>
        {!view.meetings.length ? <p className="meetings-hint">No meetings mention {view.actor.is_me ? "you" : name} yet.</p>
          : <ul className="meeting-list">{view.meetings.map(m => <li key={m.source_id}>
            <button type="button" className="meeting-row" onClick={() => onMeeting(m.source_id)}>
              <span className="meeting-row-main"><strong>{m.title || "Untitled meeting"}</strong>
                <span className="meeting-row-meta"><time dateTime={m.occurred_at ?? undefined}>{day(m.occurred_at)}</time></span></span>
            </button></li>)}</ul>}
      </section>

      <IdentitySection view={view} disabled={busy || stale} onPerson={go}
        onLink={(link, action) => act(() => context.settleLink(view.actor.id, link.link_id, link.revision, action),
          action === "confirm" ? `Confirmed that this is ${name}.` : `That mention is no longer matched to ${name}.`)}
        onNotSame={(other, otherName) => act(() => context.notSamePerson(view.actor.id, other), `${otherName} and ${name} stay separate.`)}
        onMerge={m => act(() => context.mergePeople(view.actor.id, m.keep === "this" ? view.actor.id : m.other, m.keep === "this" ? m.other : view.actor.id),
          `Merged ${m.keep === "this" ? m.otherName : name} into ${m.keep === "this" ? name : m.otherName}. You can undo this below.`)}
        onUnmerge={mergeId => act(() => context.unmergePeople(view.actor.id, mergeId), "Merge undone. Both people are separate again.")}/>
      {view.gaps.map(g => <p key={g} className="meetings-hint" role="note">{g}</p>)}
    </div>}
  </section>;
}

function BriefSection({ brief, onMeeting }: { brief: PersonBrief | null; onMeeting: (sourceId: string) => void }) {
  const sections = (brief?.sections ?? []).filter(s => s.claims.length);
  return <section className="meeting-section meeting-brief" aria-label="Current context">
    <h3>Current context</h3>
    {brief?.stale && <p className="meetings-hint">This summary is being refreshed with the latest meetings.</p>}
    {!sections.length && <p className="meetings-hint">Nothing current from your meetings yet.</p>}
    {sections.map(section => <div key={section.name}><h4>{SECTION_LABELS[section.name] ?? section.name.replaceAll("_", " ")}</h4>
      <ul>{section.claims.map((claim, i) => <li key={i}>{claim.text}
        {claim.assertion_ids[0] && <Proof assertionId={claim.assertion_ids[0]} onMeeting={onMeeting}/>}</li>)}</ul></div>)}
    {brief?.gaps.map(gap => <p key={gap} className="meetings-hint">{gap}</p>)}
  </section>;
}

function LoopsSection({ loops, onMeeting }: { loops: { items: OpenLoop[]; gaps: string[] } | null; onMeeting: (sourceId: string) => void }) {
  return <section className="meeting-section" aria-label="Open work">
    <h3>Open work</h3>
    {loops && !loops.items.length && <p className="meetings-hint">No open promises involve this person.</p>}
    {loops && LOOP_GROUPS.map(group => {
      const rows = loops.items.filter(l => group.buckets.includes(l.bucket));
      return !!rows.length && <div key={group.id} className="context-group" role="group" aria-label={group.title}>
        <h4>{group.title}</h4><p className="meetings-hint">{group.hint}</p>
        {rows.map(loop => <article key={loop.assertion_id} className="context-row">
          <p>{loop.statement}</p>
          <div className="chip-row">
            {loop.basis === "inferred" && <span className="chip badge-inferred" title="Eridani inferred this; nobody said it outright.">Inferred</span>}
            {loop.due_at && <span className="chip">Due {day(loop.due_at)}</span>}
            <span className="chip">{TASK_STATES[loop.task_state] ?? loop.task_state}</span>
            {loop.responsible.filter(p => !p.is_me && p.name).map((p, i) => <span key={i} className="chip">{p.name}</span>)}
          </div>
          {!!loop.tasks?.length && <ul className="person-tasks">{loop.tasks.map(t => <li key={t.id}>{t.status === "completed" ? <del>{t.title}</del> : t.title}</li>)}</ul>}
          <Proof assertionId={loop.assertion_id} citation={loop.citation} onMeeting={onMeeting}/>
        </article>)}
      </div>;
    })}
    {loops?.gaps.map(gap => <p key={gap} className="meetings-hint">{gap}</p>)}
  </section>;
}

function Observation({ item, onMeeting }: { item: PersonView["context"][number]; onMeeting: (sourceId: string) => void }) {
  return <article className="context-row">
    <p>{item.statement}</p>
    <div className="chip-row">
      <span className="chip">{KIND_LABELS[item.kind]?.replace(/s$/, "") ?? item.kind.replaceAll("_", " ")}</span>
      {item.inferred && <span className="chip badge-inferred" title="Eridani inferred this; nobody said it outright.">Inferred</span>}
      {item.lifecycle === "contested" && <span className="chip badge-contested" title="Meetings disagree about this. Questions asks you to settle it.">Contested</span>}
      {item.modality && item.modality !== "asserted" && <span className="chip">{item.modality}</span>}
    </div>
    <Proof assertionId={item.assertion_id} citation={item.citation} onMeeting={onMeeting}/>
  </article>;
}

/** Evidence drill-down: every supporting and contradicting quote, each opening its meeting. */
function Proof({ assertionId, citation, onMeeting }: { assertionId: string; citation?: PersonCitation; onMeeting: (sourceId: string) => void }) {
  const [open, setOpen] = useState(false), [proof, setProof] = useState<Evidence | null>(null), [error, setError] = useState("");
  async function toggle() {
    setOpen(v => !v);
    if (proof || open) return;
    try { setProof(await context.evidence(assertionId)); } catch (e) { setError((e as Error).message); }
  }
  return <>
    <div className="context-row-actions">
      <button className="text-button" aria-expanded={open} onClick={() => void toggle()}>Evidence</button>
      {citation && <button className="text-button" onClick={() => onMeeting(citation.source_id)}>Open meeting</button>}
    </div>
    {open && <div className="context-evidence" aria-label="Evidence">
      {error ? <p role="alert">{error}</p> : !proof ? <p role="status" className="meetings-hint">Loading evidence…</p> : <>
        {proof.evidence.map(span => <blockquote key={span.evidence_id} className={span.role === "contradiction" ? "is-contradiction" : ""}>
          {span.role === "contradiction" && <strong>Contradicts: </strong>}{span.speaker && <cite>{span.speaker}: </cite>}{span.quote}
          <small> · <button className="text-button" onClick={() => onMeeting(span.source_id)}>{span.source_title || "Meeting"}</button>
            {span.occurred_at ? ", " + day(span.occurred_at) : ""}</small></blockquote>)}
        {proof.gaps.map(gap => <p key={gap} className="meetings-hint">{gap}</p>)}
      </>}
    </div>}
  </>;
}

function IdentitySection({ view, disabled, onPerson, onLink, onNotSame, onMerge, onUnmerge }: {
  view: PersonView; disabled: boolean; onPerson: (actorId: string) => void;
  onLink: (link: PersonView["links_to_review"][number], action: "confirm" | "reject") => Promise<boolean>;
  onNotSame: (other: string, otherName: string) => Promise<boolean>; onMerge: (m: Merging) => Promise<boolean>; onUnmerge: (mergeId: string) => Promise<boolean>;
}) {
  const [merging, setMerging] = useState<Merging | null>(null);
  const [query, setQuery] = useState(""), [found, setFound] = useState<PersonChoice[] | null>(null), [findError, setFindError] = useState("");
  const name = view.actor.name, active = !view.actor.archived;
  async function search(e: React.FormEvent) {
    e.preventDefault(); setFindError("");
    try { setFound((await context.findPeople(query.trim())).items.filter(p => p.id !== view.actor.id)); }
    catch (err) { setFindError((err as Error).message); }
  }
  const choose = (other: string, otherName: string) => { setMerging({ other, otherName, keep: "this" }); setFound(null); setQuery(""); };
  return <section className="meeting-section" aria-label="Identity">
    <h3>Who this is</h3>
    {!view.identities.length ? <p className="meetings-hint">No emails or other identifiers recorded.</p>
      : <ul className="speaker-list">{view.identities.map(i => <li key={i.id}>
        <span className="speaker-label">{NAMESPACES[i.namespace] ?? i.namespace}</span><span>{i.value}</span>
        {i.status === "inferred" && <span className="chip badge-inferred" title="Matched by Eridani, not confirmed by you.">Inferred</span>}
      </li>)}</ul>}

    {!!view.links_to_review.length && <div className="context-group" role="group" aria-label="Mentions to check">
      <h4>Mentions to check</h4><p className="meetings-hint">Eridani matched these to {name} without being sure.</p>
      <ul className="speaker-list">{view.links_to_review.map(link => <li key={link.link_id}>
        <span className="speaker-label">“{link.mention || name}”</span>
        <span className="chip badge-inferred">Inferred</span>
        {link.statement && <span className="meetings-hint">{link.statement}</span>}
        <span className="speaker-actions">
          <button className="text-button" disabled={disabled} onClick={() => void onLink(link, "confirm")}>Confirm</button>
          <button className="text-button" disabled={disabled} onClick={() => void onLink(link, "reject")}>Not {name}</button>
        </span>
      </li>)}</ul>
    </div>}

    {active && <div className="context-group" role="group" aria-label="Possible duplicates">
      <h4>Possible duplicates</h4>
      {!view.merge_candidates.length && <p className="meetings-hint">No other records look like the same person.</p>}
      {view.merge_candidates.map(c => <article key={c.actor_id} className="context-row">
        <p><button className="text-button" onClick={() => onPerson(c.actor_id)}>{c.name ?? "Someone"}</button>{" "}
          {c.identities.filter(i => i.namespace === "email").map(i => <span key={i.value} className="chip">{i.value}</span>)}</p>
        <p className="meetings-hint">{c.reason}</p>
        <div className="context-row-actions">
          <button className="text-button" disabled={disabled} onClick={() => choose(c.actor_id, c.name ?? "Someone")}>Same person…</button>
          <button className="text-button" disabled={disabled} onClick={() => void onNotSame(c.actor_id, c.name ?? "Someone")}>Not the same person</button>
        </div>
      </article>)}
      {!merging && <form className="context-correction person-find" onSubmit={e => void search(e)}>
        <label className="field">Merge with someone else<input type="search" value={query} maxLength={100} placeholder="Name or email"
          onChange={e => setQuery(e.target.value)}/></label>
        <div><button className="btn btn-sm" disabled={disabled || !query.trim()}>Find</button></div>
        {findError && <p role="alert">{findError}</p>}
        {found && (!found.length ? <p className="meetings-hint">Nobody else matches.</p>
          : <ul className="speaker-list" aria-label="Matching people">{found.map(p => <li key={p.id}>
            <span>{p.is_me ? `${p.name} (you)` : p.name}</span>
            {p.identities.filter(i => i.namespace === "email").map(i => <span key={i.value} className="chip">{i.value}</span>)}
            <button type="button" className="text-button" disabled={disabled} onClick={() => choose(p.id, p.name)}>Choose</button></li>)}</ul>)}
      </form>}
      {merging && <form className="context-correction" aria-label="Confirm merge" onSubmit={e => { e.preventDefault(); void onMerge(merging).then(ok => ok && setMerging(null)); }}>
        <p>Merge {name} and {merging.otherName} into one person? Emails, aliases, assigned tasks and meeting references move to the record you keep;
          the other is archived. You can undo this.</p>
        <fieldset><legend>Keep the record of</legend>
          <label><input type="radio" name="keep" checked={merging.keep === "this"} onChange={() => setMerging({ ...merging, keep: "this" })}/>{name}</label>
          <label><input type="radio" name="keep" checked={merging.keep === "other"} onChange={() => setMerging({ ...merging, keep: "other" })}/>{merging.otherName}</label>
        </fieldset>
        <div><button className="btn btn-sm btn-primary" disabled={disabled}>Merge</button>
          <button type="button" className="btn btn-sm" onClick={() => setMerging(null)}>Cancel</button></div>
      </form>}
    </div>}

    {!!view.merges.length && <div className="context-group" role="group" aria-label="Merges">
      <h4>Merges</h4>
      <ul className="speaker-list">{view.merges.map(m => <li key={m.merge_id}>
        <span>Merged {m.loser_name ?? "someone"} into {m.winner_name ?? "someone"}{m.created_at ? ` on ${day(m.created_at)}` : ""}</span>
        {m.undone_at ? <span className="chip">Undone</span>
          : <span className="speaker-actions"><button className="text-button" disabled={disabled} onClick={() => void onUnmerge(m.merge_id)}>Unmerge</button></span>}
      </li>)}</ul>
    </div>}
  </section>;
}
