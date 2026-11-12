import { useEffect, useState } from "react";
import { api } from "./api";
import { SettingRow } from "./SettingsLayout";

export type AutoStart = {
  enabled: boolean; targets: string[]; daily_cap: number; host_cap: number; harness: string;
  choices: { id: string; name: string; repo: string }[]; started_today: number;
};

/** What the switch will do right now, in one line. */
export function autoStartNote(value: AutoStart): string {
  if (!value.enabled) return "Off: every run is started by you.";
  if (!value.targets.length) return "On, but no repository is chosen, so nothing starts.";
  return `On for ${value.targets.length} ${value.targets.length === 1 ? "repository" : "repositories"}. `
    + `${value.started_today} of ${value.daily_cap} started in the last 24 hours.`;
}

/** Auto-start (auto_start.py): tasks that need nothing from the owner start their own run in the chosen repositories. */
export function AutoStartSettings({ readOnly }: { readOnly: boolean }) {
  const [value, setValue] = useState<AutoStart | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => { api<AutoStart>("/remotea/auto-start").then(setValue).catch(() => {}); }, []);
  if (!value) return null;
  const save = async (patch: Partial<AutoStart>) => {
    const next = { ...value, ...patch };
    setBusy(true); setError("");
    try {
      setValue(await api<AutoStart>("/remotea/auto-start", { method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: next.enabled, targets: next.targets, daily_cap: next.daily_cap, host_cap: next.host_cap, harness: next.harness }) }));
    } catch (e) { setError(e instanceof Error ? e.message : "That did not save."); }
    setBusy(false);
  };
  const off = busy || readOnly;
  return <>
    <SettingRow as="label" label="Start agent-ready tasks by themselves"
      hint={"A task marked \"Nothing: an agent can finish it\" starts its own run in the repositories you choose below. The agent plans and works without asking you, unless its plan adds a migration. You still approve the result unless automatic completion or auto-merge is on. " + autoStartNote(value)}>
      <input type="checkbox" checked={value.enabled} disabled={off} onChange={e => void save({ enabled: e.target.checked })}/>
    </SettingRow>
    {value.enabled && <div className="auto-start-options">
      <fieldset disabled={off}><legend>Repositories it may start runs in</legend>
        {value.choices.length === 0 && <small>No repository is saved yet. Start one run by hand to save one.</small>}
        {value.choices.map(c => <label key={c.id}><input type="checkbox" checked={value.targets.includes(c.id)}
          onChange={e => void save({ targets: e.target.checked ? [...value.targets, c.id] : value.targets.filter(id => id !== c.id) })}/>
          {c.name} <small>{c.repo}</small></label>)}
      </fieldset>
      <label>Most runs started per day
        <input type="number" min={1} max={50} value={value.daily_cap} disabled={off}
          onChange={e => { const n = Number(e.target.value); if (n >= 1 && n <= 50) void save({ daily_cap: n }); }}/></label>
      <label>Most working at once per computer
        <input type="number" min={1} max={10} value={value.host_cap} disabled={off}
          onChange={e => { const n = Number(e.target.value); if (n >= 1 && n <= 10) void save({ host_cap: n }); }}/></label>
    </div>}
    {error && <p role="alert" className="work-error">{error}</p>}
  </>;
}
