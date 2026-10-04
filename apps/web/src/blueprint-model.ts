/** Blueprint rules shared by the Atlas Blueprint and the Types & fields visual tree.
 * Every change returns a new draft; nothing here saves. Drafts go through structure preview → apply. */
import type { Schema, SchemaField, SchemaType } from "./structure-types";
import { placeType, presentation, updateType } from "./field-library";

export const CAPABILITY_GLYPHS: [string, string, string][] = [
  ["work", "✓", "Actionable work"], ["content", "¶", "Authored content"], ["timeline", "⟷", "Timeline dates"], ["metric", "◎", "Outcome metric"],
];
export const CONTAINER_GLYPH: [string, string] = ["▢", "Organizing container"];
export { REVIEW_GLYPH } from "./review-model";

export const typeById = (schema: Pick<Schema, "types">, id: string) => schema.types.find(t => t.id === id);
export function diagramParent(schema: Schema, typeId: string): string | null {
  return presentation(schema).find(p => p.type_id === typeId)?.parent_type_id ?? null;
}
export const nestsItself = (t: Pick<SchemaType, "id" | "parent_types">) => t.parent_types.includes(t.id);
/** Other allowed homes beyond the diagram parent and itself, so the drawing never hides a rule. */
export function alsoAllowedIn(schema: Schema, typeId: string): SchemaType[] {
  const t = typeById(schema, typeId);
  if (!t) return [];
  const parent = diagramParent(schema, typeId);
  return t.parent_types.filter(h => h !== parent && h !== typeId).map(h => typeById(schema, h)).filter((h): h is SchemaType => !!h && !h.archived);
}
/** True when `candidate` sits somewhere below `ancestor` in the diagram. */
export function isDiagramDescendant(schema: Schema, candidate: string, ancestor: string): boolean {
  const parents = new Map(presentation(schema).map(p => [p.type_id, p.parent_type_id]));
  const seen = new Set<string>();
  let current = parents.get(candidate) ?? null;
  while (current && !seen.has(current)) {
    if (current === ancestor) return true;
    seen.add(current);
    current = parents.get(current) ?? null;
  }
  return false;
}

export type TypeMoveChoice = { cycle: boolean; allowed: boolean; title: string; rearrange: { enabled: boolean; note: string }; allow: { enabled: boolean; label: string; note: string } };
/** Dragging a type onto another: rearrange the drawing only if it's already allowed, or explicitly allow it. Loops are refused. */
export function typeMoveChoice(schema: Schema, typeId: string, targetId: string): TypeMoveChoice {
  const t = typeById(schema, typeId)!, target = typeById(schema, targetId)!;
  const cycle = typeId === targetId || isDiagramDescendant(schema, targetId, typeId);
  const allowed = t.parent_types.includes(targetId);
  const many = (x: SchemaType) => (x.plural || x.name).toLowerCase();
  return {
    cycle, allowed,
    title: `Place ${t.name} under ${target.name}`,
    rearrange: {
      enabled: allowed && !cycle,
      note: cycle ? `${target.name} sits inside ${t.name} in the diagram, so this would make a loop` : allowed ? "Existing records stay where they are" : `${target.name} isn't an allowed home for ${many(t)} yet`,
    },
    allow: {
      enabled: !allowed && !cycle,
      label: `Also allow ${many(t)} inside ${many(target)}`,
      note: cycle ? "Loops can't be drawn" : allowed ? "Already allowed" : "Adds an allowed home, then rearranges the diagram. No records move",
    },
  };
}
export function allowAndPlace(schema: Schema, typeId: string, targetId: string): Schema {
  const t = typeById(schema, typeId)!;
  const next = t.parent_types.includes(targetId) ? schema : updateType(schema, typeId, { parent_types: [...t.parent_types, targetId] });
  return placeType(next, typeId, targetId);
}
export function removeAllowedHome(schema: Schema, typeId: string, homeId: string): Schema {
  const t = typeById(schema, typeId)!;
  return updateType(schema, typeId, { parent_types: t.parent_types.filter(h => h !== homeId) });
}
export function addAllowedHome(schema: Schema, typeId: string, homeId: string): Schema {
  const t = typeById(schema, typeId)!;
  return t.parent_types.includes(homeId) ? schema : updateType(schema, typeId, { parent_types: [...t.parent_types, homeId] });
}

/** Records of `typeId` whose current home is of `homeTypeId`: what removing that allowed home would strand. */
export function recordsLivingIn<R extends { id: string; type_id: string; parent_id: string | null }>(records: R[], typeId: string, homeTypeId: string): R[] {
  const byId = new Map(records.map(r => [r.id, r]));
  return records.filter(r => r.type_id === typeId && r.parent_id && byId.get(r.parent_id)?.type_id === homeTypeId);
}

export type FieldRole = "operational" | "inherits" | "can_inherit";
export function fieldRole(f: Pick<SchemaField, "binding" | "inherit">): FieldRole {
  return f.binding ? "operational" : f.inherit ? "inherits" : "can_inherit";
}
export const fieldRoleLabel: Record<FieldRole, string> = { operational: "Operational, never inherited", inherits: "Inherits from home", can_inherit: "Can inherit from home" };

/** Changed type ids between the saved schema and a draft, for the "unsaved structure" bar. */
export function draftChanges(saved: Schema, draft: Schema): number {
  const before = new Map(saved.types.map(t => [t.id, JSON.stringify(t)]));
  let n = draft.types.filter(t => before.get(t.id) !== JSON.stringify(t)).length;
  if (JSON.stringify(presentation(saved)) !== JSON.stringify(presentation(draft))) n = Math.max(n, 1);
  return n;
}
