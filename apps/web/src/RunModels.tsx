import { createContext, useContext } from "react";
import { api, durableScope, durableStorageKey } from "./api";
import { HARNESSES, type RoleChoice, type HarnessId } from "./remotea-model";
export type ModelOption = { host_id: string; harness: string; model: string; label?: string; efforts: string[]; permission_profiles: string[]; default?: boolean };
/** null means a legacy server; [] means this host advertises no supported models. */
export const RunModelsContext = createContext<ModelOption[] | null>(null);
export function modelOptions(catalog: ModelOption[] | null, harness: string) {
  if (catalog === null) return (HARNESSES.find(h => h.id === harness)?.models ?? []).map(m => ({ harness, model: m.id, label: m.label, efforts: [], permission_profiles: [], host_id: "" } as ModelOption));
  return catalog.filter(m => m.harness === harness).filter((m, i, items) => items.findIndex(x => x.model === m.model) === i);
}
export function unsupportedRole(catalog: ModelOption[] | null, role: RoleChoice | null | undefined): string {
  if (!role || catalog === null) return "";
  const choices = modelOptions(catalog, role.harness);
  const model = role.model ? choices.find(m => m.model === role.model) : choices.find(m => m.default);
  if (role.model && !model) return `The host does not support ${role.model}. Choose a supported model.`;
  if (role.effort && (!model || !model.efforts.includes(role.effort))) return `The host does not support ${role.effort} effort for this model. Choose an available effort.`;
  return "";
}
export function EffortSelect({ label, value, onChange, disabled }: { label: string; value: RoleChoice; onChange: (value: RoleChoice) => void; disabled?: boolean }) {
  const catalog = useContext(RunModelsContext);
  const options = modelOptions(catalog, value.harness);
  const model = value.model ? options.find(m => m.model === value.model) : options.find(m => m.default);
  const efforts = model?.efforts ?? [];
  return <span className="run-role-effort"><span>{label} effort</span><select aria-label={label + " effort"} value={value.effort ?? ""} disabled={disabled}
    onChange={e => onChange({ ...value, effort: e.target.value || undefined })}>
    <option value="">Default</option>
    {!!value.effort && !efforts.includes(value.effort) && <option value={value.effort} disabled>{value.effort} (unsupported)</option>}
    {efforts.map(e => <option key={e} value={e}>{e}</option>)}
  </select></span>;
}
export function RoleModelSelect({ label, value, choice, onChange, same, disabled }: {
  label: string; value: RoleChoice | null | undefined; choice: (id: HarnessId) => { ok: boolean; reason?: string }; onChange: (value: RoleChoice | null) => void; same?: string; disabled?: boolean;
}) {
  const catalog = useContext(RunModelsContext);
  const key = value ? `${value.harness}:${value.model}` : "";
  return <><select aria-label={label} value={key} disabled={disabled} onChange={e => {
    const [harness, model] = e.target.value.split(/:(.*)/);
    onChange(harness ? { harness: harness as HarnessId, model } : null);
  }}>
    {same && <option value="">{same}</option>}
    {value?.model && !modelOptions(catalog, value.harness).some(m => m.model === value.model) && <option value={key} disabled>{value.model} (unsupported)</option>}
    {HARNESSES.map(h => <optgroup key={h.id} label={h.label + (choice(h.id).ok ? "" : " (cannot start here)")}>
      <option value={`${h.id}:`} disabled={!choice(h.id).ok}>{h.label} · Model default{modelOptions(catalog, h.id).find(m => m.default)?.label ? ` (${modelOptions(catalog, h.id).find(m => m.default)!.label})` : ""}</option>
      {modelOptions(catalog, h.id).map(m => <option key={m.model} value={`${h.id}:${m.model}`} disabled={!choice(h.id).ok}>{h.label} · {m.label ?? m.model}</option>)}
    </optgroup>)}
  </select>{value && <EffortSelect label={label} value={value} onChange={onChange} disabled={disabled}/>}</>;
}

export type RolePreferences = { planner: RoleChoice | null; reviewer: RoleChoice | null; worker: RoleChoice | null };
const ROLE_PREFS_PENDING = "eri-role-preferences-pending";
let roleTimer: ReturnType<typeof setTimeout> | null = null;
export function pendingRolePreferences(): RolePreferences | null {
  try { const key = durableStorageKey(ROLE_PREFS_PENDING); return key ? JSON.parse(localStorage.getItem(key) || "null") : null; } catch { return null; }
}
export async function flushRolePreferences() {
  const scope = durableScope();
  const pending = pendingRolePreferences();
  if (!pending) return;
  try {
    await api("/remotea/role-preferences", { method: "PUT", body: JSON.stringify(pending) });
    const key = durableStorageKey(ROLE_PREFS_PENDING);
    if (durableScope() === scope && key && JSON.stringify(pendingRolePreferences()) === JSON.stringify(pending)) localStorage.removeItem(key);
  } catch { /* Keep the local choices for reconnection. */ }
}
export function queueRolePreferences(preferences: RolePreferences) {
  try { const key = durableStorageKey(ROLE_PREFS_PENDING); if (key) localStorage.setItem(key, JSON.stringify(preferences)); } catch { /* The current draft remains available. */ }
  if (roleTimer) clearTimeout(roleTimer);
  roleTimer = setTimeout(() => { roleTimer = null; void flushRolePreferences(); }, 500);
}
