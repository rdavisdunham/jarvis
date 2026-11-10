import { command, post } from "./api";
import type { CustomRecord, Schema } from "./structure-types";

/** A home as the picker shows it: enough to list it again later without a fetch. */
export type HomeRef = { id: string; title: string; type_id: string; path: string };

/** "Business › Brainforge": the home's ancestors, for a row's second line. */
export function pathOf(record: Pick<CustomRecord, "home">) {
  return record.home.map(h => h.title).join(" › ");
}
export function homeRef(record: Pick<CustomRecord, "id" | "title" | "type_id" | "home">): HomeRef {
  return { id: record.id, title: record.title, type_id: record.type_id, path: pathOf(record) };
}

/** Why `candidate` can't be `row`'s home, or "" when it can. The row itself and anything inside it never can. */
export function refusal(schema: Schema, row: Pick<CustomRecord, "id" | "type_id">, candidate: Pick<CustomRecord, "id" | "type_id" | "home">) {
  if (candidate.id === row.id || candidate.home.some(h => h.id === row.id)) return "It's inside this record.";
  const allowed = schema.types.find(t => t.id === row.type_id)?.parent_types ?? [];
  if (allowed.includes(candidate.type_id)) return "";
  const name = (id: string) => schema.types.find(t => t.id === id)?.name ?? id;
  return `A ${name(candidate.type_id).toLowerCase()} can't hold a ${name(row.type_id).toLowerCase()}.`;
}
/** Whether any type with `capability` may hold `row` (so the Tasks or Notes section is worth offering). */
export function offersSection(schema: Schema, row: Pick<CustomRecord, "type_id">, capability: "work" | "content") {
  const allowed = schema.types.find(t => t.id === row.type_id)?.parent_types ?? [];
  return schema.types.some(t => !t.archived && allowed.includes(t.id) && t.capabilities.includes(capability));
}

/** Search results grouped by type name, groups in order of first appearance. */
export function groupByType<T extends Pick<CustomRecord, "type_name">>(items: T[]) {
  const groups = new Map<string, T[]>();
  for (const item of items) groups.set(item.type_name, [...(groups.get(item.type_name) ?? []), item]);
  return [...groups].map(([type, rows]) => ({ type, rows }));
}

// ---- Recent homes (this device) ------------------------------------------------------------
const RECENT_KEY = "eri-recent-homes", RECENT_MAX = 5;
/** Newest first, no repeats, at most five. */
export function rememberHome(list: HomeRef[], home: HomeRef) {
  return [home, ...list.filter(h => h.id !== home.id)].slice(0, RECENT_MAX);
}
export function recentHomes(): HomeRef[] {
  try {
    const value = JSON.parse(localStorage.getItem(RECENT_KEY) ?? "[]");
    return Array.isArray(value) ? value.filter(h => h && typeof h.id === "string" && typeof h.title === "string").slice(0, RECENT_MAX) : [];
  } catch { return []; }
}
export function saveRecentHome(home: HomeRef) {
  try { localStorage.setItem(RECENT_KEY, JSON.stringify(rememberHome(recentHomes(), home))); } catch { /* storage off: recents are a convenience */ }
}

// ---- Moving ---------------------------------------------------------------------------------
/** Move a record (with everything inside it) under `parentId`, or out of any home with null. Resolves to the
 * command to pass to `undoMove`, or throws with the preview's first issue. */
export async function moveRecord(row: Pick<CustomRecord, "id" | "revision">, schemaRevision: number, parentId: string | null) {
  const args = { record_id: row.id, expected_revision: row.revision, schema_revision: schemaRevision, operation: "move" as const, mode: "subtree" as const, parent_id: parentId };
  const preview = await post<{ preview_hash: string; issues: string[] }>("/structure/contents/preview", args);
  if (preview.issues.length) throw new Error(preview.issues[0]);
  const run = command("record.contents", { ...args, preview_hash: preview.preview_hash });
  await run.send();
  return run.id;
}
export function undoMove(commandId: string) {
  return command("record.restore_contents", { source_command_id: commandId }).send();
}
