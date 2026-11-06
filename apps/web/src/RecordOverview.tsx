import { useCallback, useEffect, useState } from "react";
import { RefreshCw } from "lucide-react";
import { api } from "./api";

/** A client or project's living overview (record_overviews.py): distilled from the owner's description, its tasks
 * (Linear included), meeting facts and briefs; rewritten daily when something changed; every version kept. */

export const OVERVIEW_TYPES = ["client", "project"];

type Sections = {
  what_it_is: string; scope_and_themes: string[]; vocabulary: string[]; people: string[];
  recent_changes: string[]; open_questions: string[];
};
type Stats = { tasks?: number; tasks_done?: number; meeting_facts?: number; new_tasks?: number; new_meeting_facts?: number };
type Version = { version: number; created_at: string; mode: string; origin: string; stats: Stats };
type Full = Version & { sections: Sections; seed: string };
export type OverviewData = {
  eligible: boolean; enabled: boolean; labels: Record<string, string>; sections: (keyof Sections)[];
  overview: Full | null; versions: Version[]; pending: boolean; next_check_at: string | null;
  last_error: { kind: string } | null; has_seed: boolean;
  /** People linked to this client or project from tasks and meetings (people_links.py). */
  linked_people?: { actor_id: string; name: string; status: string; tasks: number; meeting_facts: number }[];
};

type ListName = Exclude<keyof Sections, "what_it_is">;
const LISTS: ListName[] = ["scope_and_themes", "vocabulary", "people", "recent_changes", "open_questions"];
const POLL_MS = 5000, POLL_FOR_MS = 3 * 60 * 1000;

const day = (iso: string) => new Date(iso).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });

export function basis(v: Version) {
  const s = v.stats ?? {};
  const parts = [s.tasks ? `${s.tasks} task${s.tasks === 1 ? "" : "s"}` : "", s.meeting_facts ? `${s.meeting_facts} meeting fact${s.meeting_facts === 1 ? "" : "s"}` : ""].filter(Boolean);
  return parts.length ? "from " + parts.join(" and ") : "from your description";
}

export function RecordOverview({ recordId, canEdit, revision }: { recordId: string; canEdit: boolean; revision: number }) {
  const [data, setData] = useState<OverviewData | null>(null);
  const [shown, setShown] = useState<Full | null>(null);
  const [busy, setBusy] = useState(false), [message, setMessage] = useState("");
  const [pollUntil, setPollUntil] = useState(0);

  const load = useCallback(async () => {
    const next = await api<OverviewData>(`/structure/records/${recordId}/overview`);
    setData(next); setShown(next.overview);
    return next;
  }, [recordId]);

  useEffect(() => { void load().catch(() => {}); }, [load, revision]);

  // After a refresh, check back until the new version lands (or a few minutes pass).
  useEffect(() => {
    if (!pollUntil) return;
    const before = data?.overview?.version ?? 0;
    const timer = setInterval(() => {
      if (Date.now() > pollUntil) { setPollUntil(0); return; }
      void load().then(next => { if (!next.pending || (next.overview?.version ?? 0) > before) setPollUntil(0); }).catch(() => {});
    }, POLL_MS);
    return () => clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pollUntil, load]);

  if (!data || !data.eligible) return null;
  const latest = data.overview, current = shown ?? latest;
  const refresh = async (rebuild: boolean) => {
    setBusy(true); setMessage("");
    try {
      const result = await api<OverviewData & { queued: boolean; reason?: string }>(`/structure/records/${recordId}/overview/refresh`,
        { method: "POST", body: JSON.stringify({ rebuild }) });
      setData(result);
      if (result.queued) { setMessage("Updating the overview…"); setPollUntil(Date.now() + POLL_FOR_MS); }
      else setMessage("Nothing new since the last version.");
    } catch (e) { setMessage(e instanceof Error ? e.message : "Could not refresh the overview."); }
    finally { setBusy(false); }
  };
  const pick = async (version: number) => {
    if (version === latest?.version) { setShown(latest); return; }
    try { setShown((await api<OverviewData>(`/structure/records/${recordId}/overview?version=${version}`)).overview); }
    catch (e) { setMessage(e instanceof Error ? e.message : "Could not load that version."); }
  };
  const pending = data.pending || !!pollUntil;
  return <details className="detail-section record-overview" open aria-label="Overview">
    <summary>Overview{latest ? ` · v${latest.version}` : ""}{pending ? " · updating…" : ""}</summary>
    {!current && <p className="detail-empty">{!data.enabled ? "Overviews need the OpenAI key and external services turned on."
      : data.has_seed ? "Eridani builds an overview from your description, this record's tasks (Linear included) and meeting notes. The first one arrives within a day, or refresh now."
      : "Add a description above (what this is and what your work on it covers). Eridani builds an overview from it, the tasks filed here and meeting notes."}</p>}
    {current && <div className="overview-body">
      {current.version !== latest?.version && <p className="overview-old" role="status">Showing version {current.version} of {latest?.version}. <button className="text-button" onClick={() => setShown(latest)}>Show latest</button></p>}
      {current.sections.what_it_is && <p className="overview-what">{current.sections.what_it_is}</p>}
      {LISTS.filter(name => current.sections[name]?.length).map(name => <div key={name} className="overview-part">
        <h4>{data.labels[name] ?? name}</h4>
        <ul>{current.sections[name].map((item, i) => <li key={i}>{item}</li>)}</ul>
      </div>)}
    </div>}
    {!!data.linked_people?.length && <p className="overview-people"><span className="footnote">People here: </span>
      {data.linked_people.map((p, i) => <span key={p.actor_id} title={`${p.tasks} task(s), ${p.meeting_facts} meeting fact(s)`}>
        {i ? ", " : ""}{p.name}</span>)}</p>}
    <div className="overview-foot">
      {current && <span className="footnote">Updated {day(current.created_at)} {basis(current)}{current.mode === "rebuild" && current.version > 1 ? " · rebuilt" : ""}</span>}
      {data.versions.length > 1 && <select aria-label="Overview version" value={current?.version ?? ""} onChange={e => void pick(Number(e.target.value))}>
        {data.versions.map(v => <option key={v.version} value={v.version}>v{v.version} · {day(v.created_at)}</option>)}
      </select>}
      {canEdit && data.enabled && <>
        <button className="btn btn-sm btn-soft" disabled={busy || pending} onClick={() => void refresh(false)}><RefreshCw size={13} aria-hidden="true"/> Refresh</button>
        {latest && <button className="text-button" disabled={busy || pending} title="Write it again from all the material, not just what changed" onClick={() => void refresh(true)}>Rebuild</button>}
      </>}
    </div>
    {data.last_error && !pending && <p className="footnote">The last update did not finish ({data.last_error.kind}). It tries again in a few hours.</p>}
    {message && <p className="footnote" role="status">{message}</p>}
  </details>;
}
