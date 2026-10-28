import { describe, it, expect } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { SetupProposalDetails } from "./Onboarding";
import { setupTypeRows } from "./setup-model";
import type { Proposal, SchemaType } from "./structure-types";

function type(id: string, name: string, parent_types: string[] = []): SchemaType {
  return { id, name, plural: name, description: name + " definition", parent_types,
    capabilities: [], fields: [], statuses: [], archived: false };
}
const proposal: Proposal = {
  id: "preview", schema_revision: 4, expires_at: "2026-11-01T12:00:00Z",
  definition: {
    types: [type("client", "Customer"), type("course", "Course"),
      type("project", "Project", ["client", "course"])],
    type_layout: [{ type_id: "project", parent_type_id: "client" }],
    relationships: [],
  },
  impact: { affected_count: 3, affected_records: [], issues: [], blocking_count: 0 },
};

describe("guided setup schema preview", () => {
  it("shows the proposed type layout without hiding other permitted homes", () => {
    const rows = setupTypeRows(proposal.definition);
    expect(rows.map(r => [r.type.id, r.depth])).toEqual([["client", 0], ["project", 1], ["course", 0]]);
    expect(rows.find(r => r.type.id === "project")?.homes).toEqual(["Customer", "Course"]);
    const html = renderToStaticMarkup(<SetupProposalDetails proposal={proposal}/>);
    expect(html).toContain("Allowed homes: Customer, Course");
    expect(html).toContain("3 existing records use changed definitions");
    expect(html).toContain("not actual clients, projects or tasks");
    expect(html).not.toContain("Example project");
  });
  it("does not hide archived definitions or actual blocking issues", () => {
    const copy = structuredClone(proposal);
    copy.definition.types[0].archived = true;
    copy.impact.issues = [{ title: "Existing project", message: "Move this record before disallowing its parent type." }];
    copy.impact.blocking_count = 1;
    const html = renderToStaticMarkup(<SetupProposalDetails proposal={copy}/>);
    expect(html).toContain("Archived");
    expect(html).toContain("Existing project: Move this record");
  });
  it("renders old or incomplete layout metadata without omitting a type", () => {
    const copy = structuredClone(proposal.definition);
    copy.type_layout = [{ type_id: "client", parent_type_id: "project" }, { type_id: "project", parent_type_id: "client" }];
    const ids = setupTypeRows(copy).map(r => r.type.id);
    expect(new Set(ids).size).toBe(3);
    expect(ids).toHaveLength(3);
  });
});
