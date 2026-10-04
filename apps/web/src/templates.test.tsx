import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { PreviewTree, TemplateEditor, TemplateStamps } from "./Templates";
import {
  build, defaultableFields, flatten, onboardingExample, parentTypeOf, previewLines, removeRow, rowIssues, shift, templatesFor,
  type InstantiatePreview, type RecordTemplate, type TemplateNode,
} from "./templates";
import type { Schema, SchemaField, SchemaType } from "./structure-types";

const field = (id: string, kind: string, binding: string | null = null): SchemaField =>
  ({ id, name: id.replace("_", " "), description: "", kind, options: [], target_types: [], multiple: false, inherit: false, visible: true, archived: false, binding });
const type = (id: string, parents: string[], extra: Partial<SchemaType> = {}): SchemaType =>
  ({ id, name: id[0].toUpperCase() + id.slice(1), plural: id[0].toUpperCase() + id.slice(1) + "s", description: "", capabilities: [], parent_types: parents, fields: [], statuses: [], archived: false, ...extra });
const schema: Schema = {
  revision: 3, relationships: [], understandings: [],
  types: [
    type("client", []),
    type("project", ["client"], { capabilities: ["work", "timeline"], fields: [field("priority", "number", "priority"), field("due_date", "date", "due_date"), field("assignee", "text", "assignee"), field("start_date", "date", "start_date"), field("budget", "number"), field("kickoff", "date"), field("client_ref", "relation")] }),
    type("task", ["project", "task"], { capabilities: ["work"] }),
    type("note", ["project"], { capabilities: ["content"] }),
  ],
};
const nodes: TemplateNode[] = [
  { type_id: "task", title: "Contract" },
  { type_id: "task", title: "Access", children: [{ type_id: "task", title: "Shared drive" }] },
  { type_id: "note", title: "Kickoff notes", body: "Agenda" },
];

describe("template outline rows", () => {
  it("round-trips a tree through flat rows", () => {
    const rows = flatten(nodes);
    expect(rows.map(r => [r.depth, r.title])).toEqual([[1, "Contract"], [1, "Access"], [2, "Shared drive"], [1, "Kickoff notes"]]);
    expect(build(rows).map(n => [n.title, n.children!.map(c => c.title)])).toEqual([["Contract", []], ["Access", ["Shared drive"]], ["Kickoff notes", []]]);
  });
  it("nests and un-nests a row with its children, within bounds", () => {
    const rows = flatten(nodes);
    expect(shift(rows, 0, 1)).toBe(rows); // nothing above to nest into
    const nested = shift(rows, 1, 1);
    expect(nested.map(r => r.depth)).toEqual([1, 2, 3, 1]);
    expect(build(nested)[0].children![0].children![0].title).toBe("Shared drive");
    expect(shift(nested, 1, -1).map(r => r.depth)).toEqual([1, 1, 2, 1]);
    const deep = flatten([{ type_id: "task", title: "a", children: [{ type_id: "task", title: "b", children: [{ type_id: "task", title: "c", children: [{ type_id: "task", title: "d" }] }] }] }, { type_id: "task", title: "e" }]);
    expect(shift(deep, 3, 1)).toBe(deep); // already at the deepest level
  });
  it("removing a row keeps its children one level up", () => {
    const rows = removeRow(flatten(nodes), 1);
    expect(rows.map(r => [r.depth, r.title])).toEqual([[1, "Contract"], [1, "Shared drive"], [1, "Kickoff notes"]]);
  });
  it("checks titles, allowed homes and bounds against the schema", () => {
    const rows = flatten(nodes);
    expect(rowIssues(rows, "project", schema)).toEqual([]);
    expect(parentTypeOf(rows, 2, "project")).toBe("task");
    const bad = flatten([{ type_id: "task", title: "Access", children: [{ type_id: "note", title: "Inside a task" }] }, { type_id: "task", title: " " }]);
    expect(rowIssues(bad, "project", schema)).toEqual(["Notes can't live inside Tasks (“Inside a task”).", "Row 3 needs a title."]);
    expect(rowIssues(flatten(Array.from({ length: 101 }, (_, i) => ({ type_id: "task", title: "T" + i }))), "project", schema)[0]).toMatch(/at most 100/);
  });
});

describe("template defaults and suggestions", () => {
  it("never offers dates, assignees or record links as defaults", () => {
    expect(defaultableFields(schema.types[1]).map(f => f.id)).toEqual(["priority", "budget"]);
  });
  it("offers the Client onboarding example only where work can live inside", () => {
    const example = onboardingExample(schema, "project")!;
    expect(example.payload.children.map(c => [c.type_id, c.title])).toEqual([["task", "Contract"], ["task", "Access"], ["task", "Kickoff"], ["task", "Discovery"]]);
    expect(onboardingExample(schema, "note")).toBeNull();
  });
});

const template: RecordTemplate = {
  id: "t1", type_id: "project", type_name: "Project", name: "Client onboarding", description: "", revision: 2, archived: false, record_count: 5,
  summary: "A project with 4 tasks: Contract, Access, Kickoff, Discovery",
  payload: { values: { priority: 2 }, body: "## Goals", children: nodes },
};

describe("template rendering", () => {
  it("renders dashed stamps with the summary and filters by type", () => {
    const html = renderToStaticMarkup(<TemplateStamps templates={[template]} onPick={() => {}}/>);
    expect(html).toContain("Start from template: Client onboarding");
    expect(html).toContain("A project with 4 tasks: Contract, Access, Kickoff, Discovery");
    expect(templatesFor([template, { ...template, id: "t2", type_id: "task" }, { ...template, id: "t3", archived: true }], ["project"]).map(t => t.id)).toEqual(["t1"]);
    expect(renderToStaticMarkup(<TemplateStamps templates={[]} onPick={() => {}}/>)).toBe("");
  });
  it("previews the tree that will be created", () => {
    const preview = { tree: { type_id: "project", type_name: "Project", title: "ABC onboarding", children: [
      { type_id: "task", type_name: "Task", title: "Access", children: [{ type_id: "task", type_name: "Task", title: "Shared drive", children: [] }] }] } } as unknown as InstantiatePreview;
    expect(previewLines(preview.tree).map(l => [l.depth, l.title])).toEqual([[0, "ABC onboarding"], [1, "Access"], [2, "Shared drive"]]);
    const html = renderToStaticMarkup(<PreviewTree preview={preview}/>);
    expect(html).toContain('aria-label="Records to create"');
    expect(html).toContain("Shared drive");
  });
  it("renders the editor with name, outline, allowed defaults and nested rows", () => {
    const html = renderToStaticMarkup(<TemplateEditor schema={schema} typeId="project" template={template} onClose={() => {}} onSaved={() => {}}/>);
    expect(html).toContain('aria-label="Template editor"');
    expect(html).toContain("Edit template");
    expect(html).toContain('value="Client onboarding"');
    expect(html).toContain("## Goals");
    expect(html).toContain('aria-label="Default budget"');
    expect(html).not.toContain('aria-label="Default due date"');
    expect(html).toContain('aria-label="Title of Shared drive"');
    expect(html).toContain('aria-label="Nest Contract inside the record above"');
  });
});
