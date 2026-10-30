import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import {
  initialChoices, initialMeaning, LinearMapping, MappingOffer, mappingBody, MappingPreview,
  type MappingView, type Plan,
} from "./LinearMapping";

const view: MappingView = {
  status: "applied",
  catalog: {
    teams: [{ id: "t1", name: "Platform", issues: 4 }],
    projects: [
      { id: "p1", name: "Acme Web", issues: 2, team_ids: ["t1"] },
      { id: "p2", name: "Beta", issues: 1, team_ids: ["t1"], archived: true },
    ],
    unassigned: { t1: 1 },
    discovered_at: "2026-10-30T10:00:00Z",
  },
  meanings: { team: "area", project: null },
  rules: [
    { key: "team:t1", source: "team", source_id: "t1", action: "home", record_id: "r-team", created: true, home: null },
    { key: "project:p1", source: "project", source_id: "p1", action: "home", record_id: "r-acme", created: false, home: null },
  ],
  types: [{ id: "area", name: "Area", plural: "Areas", capabilities: [] }],
  homes: [],
};

describe("Linear Homes mapping", () => {
  it("starts from the saved mapping and never guesses an undecided source", () => {
    const choices = initialChoices(view);
    expect(choices["team:t1"].action).toBe("create"); // Server reuses the Home it created.
    expect(choices["project:p1"]).toMatchObject({ action: "reuse", record_id: "r-acme" });
    expect(choices["project:p2"].action).toBe("skip");
    expect(initialMeaning(view, "project").choice).toBe("");
  });
  it("asks for a meaning before mapping and sends a new type by name", () => {
    const choices = initialChoices(view);
    const meanings = { team: initialMeaning(view, "team"), project: initialMeaning(view, "project") };
    expect(() => mappingBody(meanings, choices)).toThrow("What is a Linear project");
    const body = mappingBody({ ...meanings, project: { choice: "new", name: " Engagement " } }, choices);
    expect(body.team).toEqual({ type_id: "area" });
    expect(body.project).toEqual({ new_type: "Engagement" });
    expect(body.rules).toContainEqual({ source: "project", source_id: "p1", action: "reuse", record_id: "r-acme" });
    expect(body.rules).toContainEqual({ source: "team", source_id: "t1", action: "create", title: "Platform", parent: "team" });
  });
  it("skips every source of a kind that is not a Home, without needing its meaning", () => {
    const body = mappingBody({ team: { choice: "area", name: "" }, project: { choice: "skip", name: "" } }, initialChoices(view));
    expect(body.project).toBeNull();
    expect(body.rules.filter((r) => r.source === "project").every((r) => r.action === "skip")).toBe(true);
  });
  it("offers review or a normal import", () => {
    const html = renderToStaticMarkup(<MappingOffer busy={false} onReview={() => {}} onSkip={() => {}} />);
    expect(html).toContain("Review mapping");
    expect(html).toContain("Skip, import normally");
    const closed = renderToStaticMarkup(
      <LinearMapping summary={{ status: "applied", offer: false, homes: 3, applied_at: null, discovered_at: null }} onChanged={() => {}} />,
    );
    expect(closed).toContain("3 Homes");
  });
  it("previews the tree, counts, schema changes and kept exceptions", () => {
    const plan: Plan = {
      fingerprint: "f".repeat(64),
      schema_changes: [{ kind: "new_type", text: "New type “Engagement” for Linear projects" }],
      tree: {
        homes: [{
          ref: "r-team", title: "Platform", type_name: "Area", new: false, issues: 1, sources: [{ kind: "team", name: "Platform", archived: false }],
          children: [{ ref: "new:project:p2", title: "Beta", type_name: "Engagement", new: true, issues: 1, sources: [{ kind: "project", name: "Beta", archived: true }], children: [] }],
        }],
        unfiled: 2,
      },
      placement: {
        moved: 2, kept: 1, unchanged: 0, subtasks: 1, unmapped: 0, blocked: 0,
        changes: [{ task_id: "a", title: "Fix login", outcome: "kept", from: "Inbox", to: "Beta" }],
      },
      proposal: null,
    };
    const html = renderToStaticMarkup(<MappingPreview plan={plan} />);
    for (const text of ["New type “Engagement”", "Platform", "Archived in Linear", "2 issues stay unfiled",
      "2 filed in their new Home", "1 kept where you put them", "Fix login: stays in Inbox (your choice)"])
      expect(html).toContain(text);
  });
});
