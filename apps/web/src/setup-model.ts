import type { Proposal, SchemaType } from "./structure-types";
import type { Organization } from "./productivity";

export type SetupProgress = {
  revision: number; status: string; step: string; preferred_name: string; timezone: string;
  purpose: string; classification_name: string; description: string; examples: string;
  design_notes: string; proposal_id?: string;
};

/** The account already has real structure (a space and a project), so Today's setup banner retires itself. */
export const setupEstablished = (organization: Pick<Organization, "spaces" | "projects">) =>
  organization.spaces.some(space => !space.archived) && organization.projects.some(project => !project.archived);

/** Set up with Dani: one interview session per owner (docs/SETUP_WITH_DANI_PLAN.md). */
export type SetupSession = {
  status: "not_started" | "interviewing" | "proposing" | "reviewing" | "completed" | "abandoned";
  revision: number; conversation_id: string | null; mode: "voice" | "typed" | null;
  focus: "both" | "organization" | "about_you"; notes: { covered?: string[]; rerun?: boolean };
  tree_proposal_id: string | null; history_off: boolean;
  /** The backend can build a proposal from the interview. */
  can_propose?: boolean;
};
export type SetupMode = "voice" | "typed";

export const setupActive = (session?: SetupSession | null) =>
  !!session && ["interviewing", "proposing", "reviewing"].includes(session.status);

export const SETUP_MODE_QUESTION = "Want to do this by voice? We recommend a conversation for this. If you can’t talk right now, feel free to type.";

/** Topics the interview covers. The synthesizer's notes tick them, not the live model. */
export const SETUP_TOPICS = [
  { id: "you", label: "Getting to know you" },
  { id: "week", label: "Your week" },
  { id: "areas_goals", label: "Areas and goals" },
  { id: "in_flight", label: "What’s in flight" },
  { id: "help", label: "How to help" },
] as const;

/** The launcher's primary action for the saved setup state. */
export function setupLaunchLabel(onboardingStatus: string, session?: SetupSession | null) {
  if (setupActive(session)) return "Resume with Dani";
  if (onboardingStatus === "completed" || session?.status === "completed") return "Update with Dani";
  return "Set up with Dani";
}

export const SETUP_REQUEST ="Help me design or refine my organization in Eridani. Resume my saved setup, ask me one useful question at a time, and preview the types, fields and homes before applying changes. Treat examples as examples unless I ask you to create real items.";

/** Types & fields enters the same guided setup, narrowed to type and field changes. */
export function typesFieldsRequest(typeName?: string) {
  const start = typeName?.trim() ? " Start with my " + typeName.trim() + " type." : "";
  return "Help me refine the types and fields in my Eridani organization. Resume my saved setup and ask me one useful question at a time about record types, fields, field options and which types each field applies to." + start + " Keep my other types, homes and statuses unless I ask, and preview the changes before applying them.";
}

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
