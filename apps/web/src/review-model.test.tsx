import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { ReviewCadenceControl } from "./Reviews";
import { indexAtlas, markName, type AtlasRecord, type AtlasType } from "./atlas-model";
import { REVIEW_INTERVALS, intervalLabel, localDay, reviewLabels, reviewLensDim, reviewOf, reviewPatch, reviews } from "./review-model";
import { draftChanges } from "./blueprint-model";
import { updateType } from "./field-library";
import type { Schema, SchemaType } from "./structure-types";

const type = (id: string, extra: Partial<SchemaType> = {}): SchemaType =>
  ({ id, name: id[0].toUpperCase() + id.slice(1), plural: id[0].toUpperCase() + id.slice(1) + "s", description: "", capabilities: [], parent_types: [], fields: [], statuses: [], archived: false, ...extra });

describe("review cadence setting", () => {
  it("defaults to off monthly and patches the draft without losing the other half", () => {
    const client = type("client");
    expect(reviewOf(client)).toEqual({ enabled: false, every: "1m" });
    expect(reviews(client)).toBe(false);
    const on = { ...client, ...reviewPatch(client, { enabled: true }) };
    expect(on.review).toEqual({ enabled: true, every: "1m" });
    const quarterly = { ...on, ...reviewPatch(on, { every: "3m" }) };
    expect(quarterly.review).toEqual({ enabled: true, every: "3m" });
    expect(reviewPatch(quarterly, { enabled: false }).review).toEqual({ enabled: false, every: "3m" });
    expect(reviews({ ...quarterly, archived: true })).toBe(false);
    expect(intervalLabel("2w")).toBe("Every 2 weeks");
  });

  it("is a schema draft change that goes through preview and apply", () => {
    const schema: Schema = { revision: 1, relationships: [], understandings: [], types: [type("client"), type("task")] };
    const draft = updateType(schema, "client", reviewPatch(schema.types[0], { enabled: true, every: "1w" }));
    expect(draftChanges(schema, draft)).toBe(1);
    expect(draft.types[0].review).toEqual({ enabled: true, every: "1w" });
    expect(schema.types[0].review).toBeUndefined();
  });

  it("renders a labelled switch and an interval select that is disabled while off", () => {
    const off = renderToStaticMarkup(<ReviewCadenceControl type={type("client")} onChange={() => {}}/>);
    expect(off).toContain('aria-label="Review cadence"');
    expect(off).toContain('aria-label="Review interval"');
    expect(off).toMatch(/<select[^>]*disabled/);
    expect(off).toContain("Resurface clients for review");
    const on = renderToStaticMarkup(<ReviewCadenceControl type={type("client", { review: { enabled: true, every: "6m" } })} onChange={() => {}}/>);
    expect(on).not.toMatch(/<select[^>]*disabled/);
    expect(on).toMatch(/<option value="6m" selected="">Every 6 months<\/option>/);
    expect(REVIEW_INTERVALS.map(([id]) => id)).toEqual(["1w", "2w", "1m", "3m", "6m", "1y"]);
  });
});

describe("due labels", () => {
  const today = "2026-10-04";
  it("keeps last and next review as separate phrases", () => {
    expect(reviewLabels({ next_review_at: "2026-11-04T15:00:00Z", review_due: false }, today, "UTC")).toEqual({ last: "Not yet", next: "Nov 4", tone: "scheduled" });
    expect(reviewLabels({ last_reviewed_at: "2026-10-03T09:00:00Z", next_review_at: "2026-11-03T09:00:00Z" }, today, "UTC").last).toBe("Yesterday");
  });
  it("says when a review is due, overdue or paused", () => {
    expect(reviewLabels({ next_review_at: "2026-10-04T01:00:00Z", review_due: true }, today, "UTC")).toMatchObject({ next: "Due now", tone: "due" });
    expect(reviewLabels({ next_review_at: "2026-10-03T01:00:00Z", review_due: true }, today, "UTC").next).toBe("Due since yesterday");
    expect(reviewLabels({ next_review_at: "2026-09-20T01:00:00Z", review_due: true }, today, "UTC").next).toBe("Due since Sep 20");
    expect(reviewLabels({ next_review_at: "2026-09-20T01:00:00Z", review_paused: true }, today, "UTC")).toMatchObject({ next: "Paused for this record", tone: "paused" });
    expect(reviewLabels({}, today, "UTC").next).toBe("Not scheduled");
  });
  it("uses the local day of an instant", () => {
    expect(localDay("2026-10-04T02:00:00Z", "America/Chicago")).toBe("2026-10-03");
    expect(localDay("2026-10-04T02:00:00Z", "UTC")).toBe("2026-10-04");
  });
});

describe("Atlas Reviews lens", () => {
  const rec = (id: string, parent: string | null, extra: Partial<AtlasRecord> = {}): AtlasRecord =>
    ({ id, title: id, type_id: "task", parent_id: parent, depth: 0, revision: 1, sort_order: 0, opens_as: "item", status_meaning: "open", work: true, due_date: null, planned_date: null, ...extra });
  const types: AtlasType[] = [{ id: "client", name: "Client", plural: "Clients", capabilities: [], parent_types: [], review: { enabled: true, every: "1m" } }, { id: "task", name: "Task", plural: "Tasks", capabilities: ["work"], parent_types: ["client"] }];
  const records = [
    rec("work", null, { type_id: "client", opens_as: "container", work: false, status_meaning: null }),
    rec("abc", "work", { type_id: "client", opens_as: "container", work: false, status_meaning: null, review_due: true }),
    rec("xyz", "work", { type_id: "client", opens_as: "container", work: false, status_meaning: null }),
    rec("t1", "xyz"),
  ];
  const index = indexAtlas({ records, types });
  it("counts due reviews inside each region", () => {
    expect(index.reviews.get("work")).toBe(1);
    expect(index.reviews.get("abc")).toBeUndefined();
    expect(index.reviews.get("xyz")).toBeUndefined();
  });
  it("dims everything that is neither due nor holding due reviews", () => {
    const dim = (id: string) => reviewLensDim(index.byId.get(id)!.review_due, index.reviews.get(id) ?? 0);
    expect(dim("abc")).toBe(false);
    expect(dim("work")).toBe(false);
    expect(dim("xyz")).toBe(true);
    expect(dim("t1")).toBe(true);
  });
  it("names review state in accessible labels", () => {
    expect(markName(index.byId.get("abc")!, index, "2026-10-04")).toContain("review due");
    expect(markName(index.byId.get("work")!, index, "2026-10-04")).toContain("1 review due inside");
  });
});
