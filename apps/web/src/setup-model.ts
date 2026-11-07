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
  focus: "both" | "organization" | "about_you";
  notes: { covered?: string[]; missing?: string[]; rerun?: boolean; propose_error?: string | null };
  tree_proposal_id: string | null; history_off: boolean;
  /** Suggested core facts from the last proposal, waiting for the owner (SetupMemoryReview). */
  memory_candidates?: MemoryCandidate[]; confirmed_memory_ids?: string[];
  /** Saved facts that are not suggestions; they stay as they are. */
  known_memory_count?: number;
  /** The backend can build a proposal from the interview. */
  can_propose?: boolean;
};
export type SetupMode = "voice" | "typed";
export type MemoryCandidate = {
  id: string; content: string; fact_key: string; evidence: string; evidence_source_id: string | null;
  existing_memory_id: string | null; action: "add" | "keep" | "update" | "retire";
};

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

/** A proposed organization from Set up with Dani: spaces → areas → goals → projects. */
export type TreeKind = "space" | "area" | "goal" | "project";
export type TreeNode = {
  id: string; kind: TreeKind; title: string; description: string;
  existing_id?: string | null; action: "create" | "keep" | "rename" | "archive";
  archive_confirmed?: boolean; children: TreeNode[];
};
export type OrganizationProposal = {
  id: string; tree: TreeNode[]; preview_hash: string; applied_at: string | null; expires_at: string;
  impact: { create: Record<TreeKind, number>; rename: number; archive: number; keep: number;
    issues: { node_id: string | null; message: string }[]; blocking_count: number };
};

export const TREE_KIND_LABEL: Record<TreeKind, string> = { space: "Space", area: "Area", goal: "Goal", project: "Project" };
/** What a row may hold (mirrors setup_tree.PARENTS). */
export const childKinds = (kind: TreeKind | null): TreeKind[] =>
  kind === null ? ["space"] : kind === "space" ? ["area", "goal", "project"] : kind === "area" ? ["goal", "project"] :
    kind === "goal" ? ["project"] : [];

export function mapTree(nodes: TreeNode[], fn: (node: TreeNode) => TreeNode | null): TreeNode[] {
  return nodes.flatMap(node => {
    const next = fn(node);
    return next ? [{ ...next, children: mapTree(next.children, fn) }] : [];
  });
}

export const editNode = (tree: TreeNode[], id: string, patch: Partial<TreeNode>) =>
  mapTree(tree, node => node.id !== id ? node : {
    ...node, ...patch,
    // Retitling an existing record renames it; the backend turns an unchanged title back into "keep".
    ...(patch.title !== undefined && node.existing_id && node.action === "keep" ? { action: "rename" as const } : {}),
  });

export const removeNode = (tree: TreeNode[], id: string) => mapTree(tree, node => node.id === id ? null : node);

export function addNode(tree: TreeNode[], parentId: string | null, kind: TreeKind): TreeNode[] {
  const node: TreeNode = { id: crypto.randomUUID(), kind, title: "New " + TREE_KIND_LABEL[kind].toLowerCase(),
    description: "", action: "create", children: [] };
  if (parentId === null) return [...tree, node];
  return mapTree(tree, row => row.id === parentId ? { ...row, children: [...row.children, node] } : row);
}

export function walkTree(nodes: TreeNode[], depth = 0, parent: TreeNode | null = null): { node: TreeNode; depth: number; parent: TreeNode | null }[] {
  return nodes.flatMap(node => [{ node, depth, parent }, ...walkTree(node.children, depth + 1, node)]);
}

const plural = (count: number, kind: TreeKind) =>
  count + " " + (count === 1 ? TREE_KIND_LABEL[kind].toLowerCase() : TREE_KIND_LABEL[kind].toLowerCase() + "s");
const list = (parts: string[]) => parts.length < 2 ? parts.join("") : parts.slice(0, -1).join(", ") + " and " + parts.at(-1);

/** "4 areas, 6 goals and 9 projects will be created. Nothing is created until you confirm." */
export function treeImpact(tree: TreeNode[]) {
  const rows = walkTree(tree).map(row => row.node);
  const counts = (["space", "area", "goal", "project"] as TreeKind[])
    .map(kind => [kind, rows.filter(n => n.kind === kind && n.action === "create").length] as const)
    .filter(([, count]) => count > 0).map(([kind, count]) => plural(count, kind));
  const renamed = rows.filter(n => n.action === "rename").length;
  const archived = rows.filter(n => n.action === "archive" && n.archive_confirmed).length;
  const parts = [
    counts.length ? list(counts) + " will be created." : "",
    renamed ? (renamed === 1 ? "1 record will be renamed." : renamed + " records will be renamed.") : "",
    archived ? (archived === 1 ? "1 record will be archived." : archived + " records will be archived.") : "",
  ].filter(Boolean);
  return (parts.length ? parts.join(" ") : "Nothing new will be created.") + " Nothing is created until you confirm.";
}
