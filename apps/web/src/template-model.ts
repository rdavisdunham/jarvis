import type { Schema, SchemaField, SchemaType } from "./structure-types";

/** Reusable templates: a saved starting structure for one type (default values, outline, child records). */
export type TemplateNode = { type_id: string; title: string; body?: string; values?: Record<string, unknown>; children?: TemplateNode[] };
export type TemplatePayload = { values: Record<string, unknown>; body: string; children: TemplateNode[] };
export type RecordTemplate = {
  id: string; type_id: string; type_name: string; name: string; description: string; payload: TemplatePayload;
  revision: number; archived: boolean; summary: string; record_count: number;
};
export type PreviewNode = { type_id: string; type_name: string; title: string; children: PreviewNode[] };
export type InstantiatePreview = {
  preview_hash: string; template_id: string; name: string; title: string; parent_id: string | null;
  home: { id: string; title: string; type_id: string }[]; tree: PreviewNode; record_count: number;
  types: Record<string, number>; summary: string; source_effect: string; schema_revision: number;
};
export type InstantiateResult = InstantiatePreview & { id: string; changed_ids: string[] };

/** Matches the server bounds in structure_schema.py. */
export const TEMPLATE_DEPTH = 4;
export const TEMPLATE_NODES = 100;
const EXCLUDED_BINDINGS = new Set(["due_date", "due_time", "due_timezone", "planned_date", "start_date", "target_date", "assignee"]);

/** Fields a template may preset: no dates, assignees or record links (templates never carry deadlines). */
export const defaultableFields = (t: SchemaType | undefined): SchemaField[] =>
  (t?.fields ?? []).filter(f => !f.archived && !EXCLUDED_BINDINGS.has(f.binding ?? "") && !["date", "datetime", "relation"].includes(f.kind));

/** Types that may live directly inside the given type. */
export const childTypes = (schema: Pick<Schema, "types">, parentType: string) =>
  schema.types.filter(t => !t.archived && t.parent_types.includes(parentType));

/** The editor works on a flat list of rows with a depth; the payload is a tree. */
export type Row = { key: string; depth: number; type_id: string; title: string; body: string; values: Record<string, unknown> };
let counter = 0;
const nextKey = () => "row-" + (++counter);

export function flatten(nodes: TemplateNode[], depth = 1): Row[] {
  return nodes.flatMap(n => [
    { key: nextKey(), depth, type_id: n.type_id, title: n.title, body: n.body ?? "", values: n.values ?? {} },
    ...flatten(n.children ?? [], depth + 1),
  ]);
}

export function build(rows: Row[]): TemplateNode[] {
  const root: TemplateNode[] = [], stack: { depth: number; node: TemplateNode }[] = [];
  for (const r of rows) {
    const node: TemplateNode = { type_id: r.type_id, title: r.title.trim(), body: r.body, values: r.values, children: [] };
    while (stack.length && stack[stack.length - 1].depth >= r.depth) stack.pop();
    (stack.length ? stack[stack.length - 1].node.children! : root).push(node);
    stack.push({ depth: r.depth, node });
  }
  return root;
}

/** Move a row (with the rows nested under it) one level in or out. */
export function shift(rows: Row[], index: number, by: 1 | -1): Row[] {
  const row = rows[index];
  if (!row) return rows;
  const target = row.depth + by;
  const previous = rows[index - 1];
  if (target < 1 || target > TEMPLATE_DEPTH || (by === 1 && (!previous || previous.depth < row.depth))) return rows;
  let end = index + 1;
  while (end < rows.length && rows[end].depth > row.depth) end++;
  return rows.map((r, i) => (i >= index && i < end ? { ...r, depth: Math.min(TEMPLATE_DEPTH, r.depth + by) } : r));
}

/** Remove a row; rows nested under it move up one level so nothing is lost silently. */
export function removeRow(rows: Row[], index: number): Row[] {
  const row = rows[index];
  let end = index + 1;
  while (end < rows.length && rows[end].depth > row.depth) end++;
  return rows.flatMap((r, i) => (i === index ? [] : i > index && i < end ? [{ ...r, depth: r.depth - 1 }] : [r]));
}

/** The home type each row would live in, so the editor offers only allowed types. */
export function parentTypeOf(rows: Row[], index: number, rootType: string): string {
  for (let i = index - 1; i >= 0; i--) if (rows[i].depth < rows[index].depth) return rows[i].type_id;
  return rootType;
}

export function rowIssues(rows: Row[], rootType: string, schema: Pick<Schema, "types">): string[] {
  const issues: string[] = [];
  const name = (id: string) => schema.types.find(t => t.id === id);
  if (rows.length > TEMPLATE_NODES) issues.push(`Templates hold at most ${TEMPLATE_NODES} records inside.`);
  rows.forEach((r, i) => {
    if (!r.title.trim()) issues.push(`Row ${i + 1} needs a title.`);
    if (r.depth > TEMPLATE_DEPTH) issues.push(`“${r.title || "Row " + (i + 1)}” is nested too deeply.`);
    const parent = parentTypeOf(rows, i, rootType), t = name(r.type_id);
    if (!t || t.archived) issues.push(`“${r.title || "Row " + (i + 1)}” uses a type that is not active.`);
    else if (!t.parent_types.includes(parent)) issues.push(`${t.plural} can't live inside ${name(parent)?.plural ?? "that type"} (“${r.title}”).`);
  });
  return issues;
}

/** Flattened preview lines for "what will be created". */
export function previewLines(node: PreviewNode, depth = 0): { depth: number; title: string; type_name: string }[] {
  return [{ depth, title: node.title, type_name: node.type_name }, ...node.children.flatMap(c => previewLines(c, depth + 1))];
}

export const createdLabel = (p: Pick<InstantiatePreview, "record_count">) =>
  p.record_count === 1 ? "1 record will be created." : `${p.record_count} records will be created.`;

/** Templates offered when adding records of these types. */
export const templatesFor = (templates: RecordTemplate[], typeIds: string[]) =>
  templates.filter(t => !t.archived && typeIds.includes(t.type_id));

/** A suggestion only: the "Client onboarding" example fills the editor, never the workspace. */
export function onboardingExample(schema: Pick<Schema, "types">, typeId: string): { name: string; description: string; payload: TemplatePayload } | null {
  const work = childTypes(schema, typeId).find(t => t.id === "task") ?? childTypes(schema, typeId).find(t => t.capabilities.includes("work"));
  if (!work) return null;
  return {
    name: "Client onboarding",
    description: "Everything a new client engagement needs before work starts.",
    payload: {
      values: {},
      body: "## Goals\n- \n\n## Contacts\n- \n\n## Links\n- ",
      children: ["Contract", "Access", "Kickoff", "Discovery"].map(title => ({ type_id: work.id, title, body: "", values: {}, children: [] })),
    },
  };
}
