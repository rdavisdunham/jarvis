import { actionReceipts, api, durableScope, durableStorageKey, subscribeActions, type SubmittedDraft } from "./api";

export type DraftRecord<T> = { value: T; dirty: boolean; revision: number; edit_id: string };
const PREFIX = "eri-draft:";
const timers = new Map<string, ReturnType<typeof setTimeout>>();
const memory = new Map<string, DraftRecord<unknown>>();
const saving = new Set<string>();
const listeners = new Set<(key: string, scope: string | null) => void>();
const localIdentity = (key: string) => (durableScope() ?? "unbound") + ":" + key;
export function subscribeDrafts(listener: (key: string, scope: string | null) => void) { listeners.add(listener); return () => { listeners.delete(listener); }; }
export function readDraft<T>(key: string, initial: T): DraftRecord<T> {
  try {
    const prefix = durableStorageKey(PREFIX), stored = prefix ? localStorage.getItem(prefix + ":" + key) : null;
    if (stored !== null) {
      const parsed = JSON.parse(stored);
      const record = parsed && typeof parsed === "object" && "dirty" in parsed && "value" in parsed ? parsed : { value: parsed, dirty: true, revision: 0 };
      if (!record.edit_id) {
        record.edit_id = crypto.randomUUID();
        memory.set(localIdentity(key), record);
        if (prefix) localStorage.setItem(prefix + ":" + key, JSON.stringify(record));
      }
      return record;
    }
  } catch { /* In-memory editing remains available when browser storage refuses writes. */ }
  return (memory.get(localIdentity(key)) as DraftRecord<T> | undefined) ?? { value: initial, dirty: false, revision: 0, edit_id: "" };
}
export function storeDraft<T>(key: string, record: DraftRecord<T>) {
  const stored = { ...record, edit_id: record.edit_id || crypto.randomUUID() };
  memory.set(localIdentity(key), stored);
  try { const prefix = durableStorageKey(PREFIX); if (prefix) localStorage.setItem(prefix + ":" + key, JSON.stringify(stored)); } catch { /* Retain the current tab's copy. */ }
  listeners.forEach(listener => listener(key, durableScope()));
}
function schedule(key: string) {
  const identity = localIdentity(key), scope = durableScope();
  clearTimeout(timers.get(identity));
  timers.set(identity, setTimeout(() => {
    timers.delete(identity);
    if (durableScope() === scope) void saveKey(key);
  }, 600));
}
/** Each edit receives a new identity, even when a user deliberately types the same text again. */
export function writeDraft<T>(key: string, value: T) {
  storeDraft(key, { ...readDraft(key, value), value, dirty: true, edit_id: crypto.randomUUID() });
  schedule(key);
}
export function submittedDraft(key: string): SubmittedDraft | undefined {
  const scope = durableScope(), record = readDraft<unknown>(key, null);
  return scope && record.edit_id ? { scope, key, edit_id: record.edit_id } : undefined;
}
async function saveKey(key: string) {
  const scope = durableScope(), identity = localIdentity(key), record = readDraft<unknown>(key, null);
  if (!scope || !record.dirty || saving.has(identity)) return;
  saving.add(identity);
  try {
    const saved = await api<{ revision: number }>(`/remotea/drafts/${encodeURIComponent(key)}`, { method: "PUT", body: JSON.stringify({ value: record.value }) });
    if (durableScope() !== scope) return;
    const latest = readDraft<unknown>(key, null);
    if (latest.edit_id === record.edit_id) storeDraft(key, { ...latest, dirty: false, revision: saved.revision });
  } catch { /* The existing provider retries dirty drafts on reconnection. */ }
  finally {
    saving.delete(identity);
    if (durableScope() === scope) {
      const latest = readDraft<unknown>(key, null);
      if (latest.dirty && latest.edit_id !== record.edit_id) schedule(key);
    }
  }
}
export async function flushDrafts() {
  const prefix = durableStorageKey(PREFIX), scope = durableScope();
  if (!prefix || !scope) return;
  const keys = new Set([...memory.keys()].filter(key => key.startsWith(scope + ":")).map(key => key.slice(scope.length + 1)));
  try { Object.keys(localStorage).filter(key => key.startsWith(prefix + ":")).forEach(key => keys.add(key.slice(prefix.length + 1))); } catch { /* In-memory drafts can still sync. */ }
  await Promise.all([...keys].map(saveKey));
}
function clearAcknowledgedDrafts() {
  const scope = durableScope();
  if (!scope) return;
  for (const receipt of actionReceipts()) {
    const reference = receipt.draft;
    if (!reference || reference.scope !== scope || !["saved", "delivered", "applied"].includes(receipt.status)) continue;
    const draft = readDraft<unknown>(reference.key, null);
    if (draft.edit_id !== reference.edit_id) continue;
    // Acknowledgement consumes only this submitted version. Newer local edits are independent.
    writeDraft(reference.key, "");
  }
}
subscribeActions(clearAcknowledgedDrafts);
clearAcknowledgedDrafts();
