import type { Proposal, SchemaType } from "./structure-types";

export type SetupProgress = {
  revision: number; status: string; step: string; preferred_name: string; timezone: string;
  purpose: string; classification_name: string; description: string; examples: string;
  design_notes: string; proposal_id?: string;
};

export const SETUP_REQUEST = "Help me design or refine my organization in Eridani. Resume my saved setup, ask me one useful question at a time, and preview the types, fields and homes before applying changes. Treat examples as examples unless I ask you to create real items.";

/** Display placement is separate from all of a type's permitted homes. */
export function setupTypeRows(definition: Proposal["definition"]) {
  const types = new Map(definition.types.map(type => [type.id, type]));
  const parents = new Map((definition.type_layout ?? []).map(row => [row.type_id, row.parent_type_id]));
  const result: { type: SchemaType; depth: number; homes: string[] }[] = [], seen = new Set<string>();
  function visit(type: SchemaType, depth: number) {
    if (seen.has(type.id)) return;
    seen.add(type.id);
    result.push({ type, depth, homes: type.parent_types.map(id => types.get(id)?.name ?? id) });
    for (const child of definition.types) if (parents.get(child.id) === type.id) visit(child, depth + 1);
  }
  for (const type of definition.types)
    if (!parents.get(type.id) || !types.has(parents.get(type.id)!)) visit(type, 0);
  // Keep a malformed/old layout inspectable without hiding any definition.
  for (const type of definition.types) visit(type, 0);
  return result;
}
