import { useEffect, useState } from "react";
import { Brain, Pause, Play } from "lucide-react";
import { api, post } from "../api";
import { SettingRow, SettingsGroup } from "../SettingsLayout";

export type MeetingPreferences = { synthesize_briefs: boolean; semantic_supersede: boolean; lint_budget_usd: number;
  review_destination: "eridani" | "linear" };
export type PreferenceLimits = { lint_budget_usd: { min: number; max: number; default: number; per_call: number };
  review_destination: string[] };
export type ProcessingStatus = { epoch: number; paused: boolean; paused_at: string | null;
  queue: { jobs: Record<string, number>; runs: Record<string, number> } };

const BASE = "/context/settings", ADMIN = "/context/admin";
const OUTSTANDING = ["queued", "running"];

export function outstandingRuns(status: ProcessingStatus) {
  return OUTSTANDING.reduce((sum, key) => sum + (status.queue.runs[key] ?? 0), 0);
}
/** "Paused since …" or "Running", then the epoch and the runs still queued or running. */
export function processingLine(status: ProcessingStatus) {
  const runs = outstandingRuns(status);
  const state = status.paused ? `Paused since ${status.paused_at ? new Date(status.paused_at).toLocaleString() : "an unknown time"}` : "Running";
  return `${state} · epoch ${status.epoch} · ${runs} ${runs === 1 ? "run" : "runs"} outstanding`;
}

export function MeetingMemorySettings({ preferences, limits, initialStatus = null }: {
  preferences: MeetingPreferences; limits: PreferenceLimits; initialStatus?: ProcessingStatus | null;
}) {
  const [values, setValues] = useState(preferences), [budget, setBudget] = useState(String(preferences.lint_budget_usd));
  const [status, setStatus] = useState<ProcessingStatus | null>(initialStatus);
  const [busy, setBusy] = useState(false), [error, setError] = useState(""), [message, setMessage] = useState("");
  useEffect(() => { setValues(preferences); setBudget(String(preferences.lint_budget_usd)); }, [preferences]);
  useEffect(() => {
    if (initialStatus) return;
    let live = true;
    api<ProcessingStatus>(`${ADMIN}/status`).then(value => { if (live) setStatus(value); })
      .catch(e => { if (live) setError((e as Error).message); });
    return () => { live = false; };
  }, [initialStatus]);
  async function save(change: Partial<MeetingPreferences>) {
    setBusy(true); setError(""); setMessage("");
    try {
      const result = await api<{ preferences: MeetingPreferences }>(`${BASE}/preferences`, { method: "PATCH", body: JSON.stringify(change) });
      setValues(result.preferences); setBudget(String(result.preferences.lint_budget_usd)); setMessage("Saved.");
    } catch (e) { setError((e as Error).message); setBudget(String(values.lint_budget_usd)); }
    finally { setBusy(false); }
  }
  async function toggle() {
    if (!status) return;
    setBusy(true); setError(""); setMessage("");
    try { setStatus(await post<ProcessingStatus>(`${ADMIN}/${status.paused ? "resume" : "pause"}`)); }
    catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  }
  const { min, max, per_call } = limits.lint_budget_usd;
  const amount = Number(budget), budgetValid = budget.trim() !== "" && Number.isFinite(amount) && amount >= min && amount <= max;
  return <SettingsGroup className="meeting-memory-settings" icon={<Brain size={18}/>} title="Meeting memory"
    description="How meetings become context: processing, briefs and where approved tasks go.">
    {error && <p role="alert" className="error-banner">{error}</p>}
    {message && <p role="status" className="footnote">{message}</p>}
    <SettingRow label="Processing" hint={status ? processingLine(status) : "Loading status…"}>
      <button type="button" className="btn" disabled={busy || !status} onClick={() => void toggle()}>
        {status?.paused ? <><Play size={15}/> Resume</> : <><Pause size={15}/> Pause</>}</button>
    </SettingRow>
    <p className="footnote">Pausing stops automatic extraction and writes. Reading, your corrections and approved tasks continue.</p>
    <SettingRow as="label" className="switch-row" label="Model-written briefs"
      hint="Off: briefs are built from cited statements, with no model. On: one model call per changed person or project in each meeting, which adds cost.">
      <input type="checkbox" className="switch" role="switch" disabled={busy} checked={values.synthesize_briefs}
        onChange={e => void save({ synthesize_briefs: e.target.checked })}/>
    </SettingRow>
    <SettingRow as="label" className="switch-row" label="Apply reworded replacements"
      hint="Off (recommended for the pilot): when a meeting says something replaces an earlier statement in different words, the newer one is kept aside and the current value stays. On: it replaces the earlier statement.">
      <input type="checkbox" className="switch" role="switch" disabled={busy} checked={values.semantic_supersede}
        onChange={e => void save({ semantic_supersede: e.target.checked })}/>
    </SettingRow>
    <form className="settings-form" onSubmit={e => { e.preventDefault(); if (budgetValid) void save({ lint_budget_usd: amount }); }}>
      <label className="field"><span className="field-label-text">Weekly cleanup budget (USD)</span>
        <input type="number" inputMode="decimal" min={min} max={max} step={0.05} value={budget} onChange={e => setBudget(e.target.value)}/>
        <span className="field-hint">The weekly check of open loops and duplicates. Each model check reserves ${per_call.toFixed(2)}; 0 turns them off. Up to ${max.toFixed(2)}.</span></label>
      <div className="settings-form-actions"><button className="btn" disabled={busy || !budgetValid || amount === values.lint_budget_usd}>Save budget</button></div>
    </form>
    <SettingRow as="label" label="Approved tasks go to" hint="The default destination on the meeting review. Choosing one there also changes this.">
      <select value={values.review_destination} disabled={busy}
        onChange={e => void save({ review_destination: e.target.value as MeetingPreferences["review_destination"] })}>
        {limits.review_destination.map(key => <option key={key} value={key}>{key === "linear" ? "Linear" : "Eridani"}</option>)}
      </select>
    </SettingRow>
  </SettingsGroup>;
}
