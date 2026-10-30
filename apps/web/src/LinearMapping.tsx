import { useEffect, useState } from "react";
import { api, post } from "./api";
import { SettingRow } from "./SettingsLayout";

/** Optional Linear structure onboarding: map Linear teams and projects onto Homes
 * (backend `linear_structure.py`). Nothing is assumed: the owner says what a team or
 * project is, previews the result, then applies it. */

export type MappingSummary = {
  status: "" | "applied" | "skipped";
  offer: boolean;
  homes: number;
  applied_at: string | null;
  discovered_at: string | null;
};
type Source = { id: string; name: string; issues: number; archived?: boolean; team_ids?: string[] };
type Home = { id: string; title: string; type_id: string; type_name: string; path: string };
type SavedRule = {
  key: string;
  source: "team" | "project";
  source_id: string;
  action: "home" | "skip";
  record_id: string | null;
  created: boolean;
  home: Home | null;
};
export type MappingView = {
  status: string;
  catalog: { teams: Source[]; projects: Source[]; unassigned: Record<string, number>; discovered_at: string } | null;
  meanings: { team?: string | null; project?: string | null };
  rules: SavedRule[];
  types: { id: string; name: string; plural: string; capabilities: string[] }[];
  homes: Home[];
};
export type Kind = "team" | "project";
/** "" = not answered yet; "skip" = not a Home; "new" = a new type named `name`; else a type id. */
export type MeaningChoice = { choice: string; name: string };
export type Choice = { action: "create" | "reuse" | "skip"; record_id: string; title: string; parent: "team" | "top" };
type TreeNode = { ref: string; title: string; type_name: string; new: boolean; issues: number; sources: { kind: Kind; name: string; archived: boolean }[]; children: TreeNode[] };
export type Plan = {
  fingerprint: string;
  schema_changes: { kind: string; text: string }[];
  tree: { homes: TreeNode[]; unfiled: number };
  placement: {
    moved: number; kept: number; unchanged: number; subtasks: number; unmapped: number; blocked: number;
    changes: { task_id: string; title: string; outcome: string; from: string; to: string }[];
  };
  proposal: { id: string; blocking_count: number; issues: { message: string }[] } | null;
};

const key = (kind: Kind, id: string) => kind + ":" + id;

/** Start from what was applied before; anything never decided is skipped, never guessed. */
export function initialChoices(view: MappingView): Record<string, Choice> {
  const saved = new Map(view.rules.map((r) => [r.key, r]));
  const choices: Record<string, Choice> = {};
  const sources: [Kind, Source][] = [
    ...(view.catalog?.teams ?? []).map((t) => ["team", t] as [Kind, Source]),
    ...(view.catalog?.projects ?? []).map((p) => ["project", p] as [Kind, Source]),
  ];
  for (const [kind, source] of sources) {
    const rule = saved.get(key(kind, source.id));
    choices[key(kind, source.id)] =
      rule?.action === "home" && rule.record_id
        ? rule.created
          ? { action: "create", record_id: "", title: source.name, parent: "team" }
          : { action: "reuse", record_id: rule.record_id, title: source.name, parent: "team" }
        : { action: "skip", record_id: "", title: source.name, parent: "team" };
  }
  return choices;
}

export function initialMeaning(view: MappingView, kind: Kind): MeaningChoice {
  return { choice: view.meanings[kind] ?? "", name: "" };
}

/** The request body for preview/apply; throws a readable message for an unanswered question. */
export function mappingBody(meanings: Record<Kind, MeaningChoice>, choices: Record<string, Choice>) {
  const rules = Object.entries(choices).map(([id, c]) => {
    const [source, source_id] = [id.slice(0, id.indexOf(":")) as Kind, id.slice(id.indexOf(":") + 1)];
    const skipped = meanings[source].choice === "skip";
    if (skipped || c.action === "skip") return { source, source_id, action: "skip" as const };
    if (c.action === "reuse") {
      if (!c.record_id) throw new Error("Choose an existing Home for every source you reuse.");
      return { source, source_id, action: "reuse" as const, record_id: c.record_id };
    }
    return { source, source_id, action: "create" as const, title: c.title.trim() || undefined, parent: c.parent };
  });
  const meaning = (kind: Kind) => {
    const m = meanings[kind];
    if (!rules.some((r) => r.source === kind && r.action !== "skip")) return null;
    if (!m.choice) throw new Error(`What is a Linear ${kind} in Eridani? Choose a type first.`);
    if (m.choice === "new") {
      if (!m.name.trim()) throw new Error(`Name the new type for Linear ${kind}s.`);
      return { new_type: m.name.trim() };
    }
    return { type_id: m.choice };
  };
  return { team: meaning("team"), project: meaning("project"), rules };
}

export function MappingOffer({ busy, onReview, onSkip }: { busy: boolean; onReview: () => void; onSkip: () => void }) {
  return (
    <div className="settings-block linear-mapping-offer" role="region" aria-label="Organize Linear in your Homes">
      <p>
        <strong>Organize Linear in your Homes?</strong> Map Linear teams and projects onto your own
        types and Homes, preview the result, then apply it. Or skip and import issues as usual.
      </p>
      <div className="setting-actions">
        <button type="button" className="btn btn-primary" disabled={busy} onClick={onReview}>Review mapping</button>
        <button type="button" className="btn" disabled={busy} onClick={onSkip}>Skip, import normally</button>
      </div>
    </div>
  );
}

function Tree({ nodes }: { nodes: TreeNode[] }) {
  return (
    <ul className="linear-mapping-tree">
      {nodes.map((n) => (
        <li key={n.ref}>
          <span className="linear-mapping-node">
            <strong>{n.title}</strong> <span className="chip">{n.type_name}</span>
            {n.new && <span className="chip">New</span>}
            {n.sources.some((s) => s.archived) && <span className="chip">Archived in Linear</span>}
            <span className="footnote">{n.issues} {n.issues === 1 ? "issue" : "issues"}</span>
          </span>
          {n.children.length > 0 && <Tree nodes={n.children} />}
        </li>
      ))}
    </ul>
  );
}

export function MappingPreview({ plan }: { plan: Plan }) {
  const p = plan.placement;
  return (
    <div className="linear-mapping-preview" aria-label="Mapping preview">
      <h4>Structure changes</h4>
      {plan.schema_changes.length ? (
        <ul>{plan.schema_changes.map((c, i) => <li key={i}>{c.text}</li>)}</ul>
      ) : (
        <p className="footnote">No type changes.</p>
      )}
      {plan.proposal && plan.proposal.blocking_count > 0 && (
        <p role="alert" className="error-banner">
          Resolve these structure issues first: {plan.proposal.issues.map((i) => i.message).join(" ")}
        </p>
      )}
      <h4>Resulting Homes</h4>
      <Tree nodes={plan.tree.homes} />
      {plan.tree.unfiled > 0 && <p className="footnote">{plan.tree.unfiled} issues stay unfiled.</p>}
      <h4>Issues already imported</h4>
      <p className="footnote">
        {p.moved} filed in their new Home · {p.unchanged} already there · {p.kept} kept where you put them
        {p.subtasks > 0 && <> · {p.subtasks} follow their parent issue</>}
        {p.blocked > 0 && <> · {p.blocked} cannot move</>}
      </p>
      {p.changes.length > 0 && (
        <details>
          <summary>Affected tasks</summary>
          <ul className="linear-mapping-changes">
            {p.changes.map((c) => (
              <li key={c.task_id}>
                {c.title}: {c.outcome === "kept" ? `stays in ${c.from} (your choice)` : `${c.from} → ${c.to}`}
              </li>
            ))}
          </ul>
        </details>
      )}
      <p className="footnote">
        Filing only changes Homes in Eridani. A task's Linear project changes only when you edit it in the Linear
        section of the task.
      </p>
    </div>
  );
}

export function LinearMapping({ summary, onChanged }: { summary: MappingSummary; onChanged: () => void }) {
  const [open, setOpen] = useState(false);
  const [view, setView] = useState<MappingView | null>(null);
  const [meanings, setMeanings] = useState<Record<Kind, MeaningChoice>>({
    team: { choice: "", name: "" },
    project: { choice: "", name: "" },
  });
  const [choices, setChoices] = useState<Record<string, Choice>>({});
  const [plan, setPlan] = useState<Plan | null>(null);
  const [busy, setBusy] = useState(false), [error, setError] = useState(""), [done, setDone] = useState("");

  function load(next: MappingView) {
    setView(next);
    setChoices(initialChoices(next));
    setMeanings({ team: initialMeaning(next, "team"), project: initialMeaning(next, "project") });
    setPlan(null);
  }
  async function act(fn: () => Promise<unknown>) {
    setBusy(true);
    setError("");
    try {
      await fn();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  useEffect(() => {
    if (!open) return;
    let live = true;
    void act(async () => {
      let next = await api<MappingView>("/integrations/linear/mapping");
      if (!next.catalog) next = await post<MappingView>("/integrations/linear/mapping/discover");
      if (live) load(next);
    });
    return () => {
      live = false;
    };
  }, [open]);
  const edit = (id: string, change: Partial<Choice>) => {
    setPlan(null);
    setChoices((all) => ({ ...all, [id]: { ...all[id], ...change } }));
  };
  const typeName = (kind: Kind) => {
    const m = meanings[kind];
    return m.choice === "new" ? m.name || "new type" : view?.types.find((t) => t.id === m.choice)?.name ?? "Home";
  };

  if (!open)
    return (
      <>
        {error && <p role="alert" className="error-banner">{error}</p>}
        {done && <p role="status">{done}</p>}
        {summary.offer ? (
          <MappingOffer
            busy={busy}
            onReview={() => { setDone(""); setOpen(true); }}
            onSkip={() => void act(async () => { await post("/integrations/linear/mapping/skip"); onChanged(); })}
          />
        ) : (
          <SettingRow
            label="Homes for Linear"
            hint={summary.status === "applied"
              ? `Teams and projects file into ${summary.homes} Homes. New issues follow the mapping.`
              : "Issues import without a Home mapping."}
          >
            <button type="button" className="btn" onClick={() => { setDone(""); setOpen(true); }}>Review mapping</button>
          </SettingRow>
        )}
      </>
    );

  // Plain render helpers, not components: a component defined here would remount and drop input focus.
  const meaningRow = (kind: Kind) => {
    const m = meanings[kind];
    const set = (change: Partial<MeaningChoice>) => {
      setPlan(null);
      setMeanings((all) => ({ ...all, [kind]: { ...all[kind], ...change } }));
    };
    return (
      <SettingRow key={kind} label={`A Linear ${kind} is…`} hint="Eridani does not assume a meaning such as Client.">
        <select aria-label={`Meaning of a Linear ${kind}`} value={m.choice} onChange={(e) => set({ choice: e.target.value })}>
          <option value="">Choose…</option>
          <option value="skip">Not a Home (don't map {kind}s)</option>
          {view?.types.filter((t) => !t.capabilities.includes("content") || t.capabilities.includes("work")).map((t) => (
            <option key={t.id} value={t.id}>{t.name}</option>
          ))}
          <option value="new">A new type…</option>
        </select>
        {m.choice === "new" && (
          <input aria-label={`New type for Linear ${kind}s`} placeholder="Type name" value={m.name}
            onChange={(e) => set({ name: e.target.value })} />
        )}
      </SettingRow>
    );
  };

  const sourceRow = (kind: Kind, source: Source) => {
    const id = key(kind, source.id), c = choices[id];
    if (!c) return null;
    const off = meanings[kind].choice === "skip";
    return (
      <div key={kind + source.id} className={"settings-item linear-mapping-row" + (kind === "project" ? " nested" : "")}>
        <span className="settings-item-main">
          {source.name} <span className="footnote">{source.issues} issues</span>
          {source.archived && <span className="chip">Archived in Linear</span>}
        </span>
        <select aria-label={`Destination for ${source.name}`} disabled={off} value={off ? "skip" : c.action}
          onChange={(e) => edit(id, { action: e.target.value as Choice["action"] })}>
          <option value="create">Create a new {typeName(kind)}</option>
          <option value="reuse">Use an existing Home</option>
          <option value="skip">Skip</option>
        </select>
        {!off && c.action === "reuse" && (
          <select aria-label={`Existing Home for ${source.name}`} value={c.record_id}
            onChange={(e) => edit(id, { record_id: e.target.value })}>
            <option value="">Choose a Home…</option>
            {view?.homes.map((h) => <option key={h.id} value={h.id}>{h.path} ({h.type_name})</option>)}
          </select>
        )}
        {!off && c.action === "create" && (
          <>
            <input aria-label={`Home title for ${source.name}`} value={c.title} onChange={(e) => edit(id, { title: e.target.value })} />
            {kind === "project" && (
              <select aria-label={`Place for ${source.name}`} value={c.parent}
                onChange={(e) => edit(id, { parent: e.target.value as Choice["parent"] })}>
                <option value="team">Inside its team's Home</option>
                <option value="top">At the top level</option>
              </select>
            )}
          </>
        )}
      </div>
    );
  };

  const body = () => mappingBody(meanings, choices);
  return (
    <div className="settings-block linear-mapping" role="region" aria-label="Linear Homes mapping">
      {error && <p role="alert" className="error-banner">{error}</p>}
      {done && <p role="status">{done}</p>}
      {!view?.catalog ? (
        <p className="settings-empty">{busy ? "Reading your Linear teams and projects…" : "No Linear teams to map yet."}</p>
      ) : (
        <fieldset disabled={busy}>
          <legend className="sr-only">Map Linear teams and projects</legend>
          {meaningRow("team")}
          {meaningRow("project")}
          <div className="settings-list">
            {view.catalog.teams.map((team) => (
              <div key={team.id}>
                {sourceRow("team", team)}
                {view.catalog!.projects.filter((p) => p.team_ids?.includes(team.id)).map((p) => sourceRow("project", p))}
              </div>
            ))}
            {view.catalog.projects
              .filter((p) => !p.team_ids?.some((t) => view.catalog!.teams.some((team) => team.id === t)))
              .map((p) => sourceRow("project", p))}
          </div>
          <div className="setting-actions">
            <button type="button" className="btn btn-primary"
              onClick={() => void act(async () => setPlan(await post<Plan>("/integrations/linear/mapping/preview", body())))}>
              Preview
            </button>
            <button type="button" className="btn"
              onClick={() => void act(async () => load(await post<MappingView>("/integrations/linear/mapping/discover")))}>
              Refresh from Linear
            </button>
            <button type="button" className="btn btn-ghost" onClick={() => { setOpen(false); setPlan(null); }}>Close</button>
          </div>
          {plan && (
            <>
              <MappingPreview plan={plan} />
              <div className="setting-actions">
                <button type="button" className="btn btn-primary" disabled={(plan.proposal?.blocking_count ?? 0) > 0}
                  onClick={() => void act(async () => {
                    await post("/integrations/linear/mapping/apply", { ...body(), fingerprint: plan.fingerprint });
                    setDone("Mapping applied. New Linear issues will file into these Homes.");
                    setOpen(false);
                    onChanged();
                  })}>
                  Apply mapping
                </button>
              </div>
            </>
          )}
        </fieldset>
      )}
    </div>
  );
}
