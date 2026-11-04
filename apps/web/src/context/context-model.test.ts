import { describe, expect, it } from "vitest";
import { chosen, effectiveAt, groupItems, linkedId, meetingStatus, nextSelection } from "./context-model";
import { proposal, review, source } from "./test-fixtures";

describe("meeting status", () => {
  it("explains partial coverage, budget pauses and the failure kind", () => {
    const partial = meetingStatus(source({ status: "partial", processing: { stage: "extract", status: "partial", attempts: 1, updated_at: "",
      coverage: { "0": "extracted", "1": "skipped", "2": "failed", "3": "no_content" } } }));
    expect(partial).toMatchObject({ label: "Partly read", tone: "warn" });
    expect(partial.detail).toBe("2 of 4 passages read; 1 skipped, 1 failed.");
    expect(meetingStatus(source({ status: "awaiting_input" })).label).toBe("Paused");
    const failed = meetingStatus(source({ status: "failed", last_error: { stage: "extract", status: "failed", message: "Model refused.", retryable: false, at: "" } }));
    expect(failed).toMatchObject({ label: "Extraction failed", tone: "error" });
    expect(failed.detail).toContain("stopped trying");
    expect(meetingStatus(source({ status: "processing", processing: { stage: "embed", status: "running", coverage: {}, attempts: 1, updated_at: "" } })).detail)
      .toBe("Now indexing for search.");
  });
});

describe("deep links", () => {
  it("accepts IDs and ignores anything else", () => {
    expect(linkedId("?view=meetings&source=0b6f-12_x", "source")).toBe("0b6f-12_x");
    expect(linkedId("?view=person&actor=<script>", "actor")).toBeNull();
    expect(linkedId("?view=meetings", "source")).toBeNull();
  });
});

describe("selection", () => {
  it("starts from the server's preselection and keeps choices on unchanged proposals", () => {
    const first = review().proposals;
    const initial = nextSelection(new Set(), null, first);
    expect([...initial]).toEqual(["p-own"]);
    const mine = new Set(["p-other"]);
    const second = [proposal({ revision: 2, preselected: true }), first[1], first[2]];
    expect([...nextSelection(mine, first, second)].sort()).toEqual(["p-other", "p-own"]);
    const settled = [proposal({ disposition: "approved", publication: { id: "pub-q", status: "queued", task_id: null } }), first[1]];
    expect([...nextSelection(new Set(["p-own", "p-other"]), first, settled)]).toEqual(["p-other"]);
  });

  it("keeps task approval and update application apart", () => {
    const all = new Set(["p-own", "p-other", "p-upd"]), proposals = review().proposals;
    expect(chosen(proposals, all, "approvable")).toEqual([{ proposal_id: "p-own", revision: 1 }, { proposal_id: "p-other", revision: 1 }]);
    expect(chosen(proposals, all, "updates")).toEqual([{ proposal_id: "p-upd", revision: 1 }]);
  });
});

describe("helpers", () => {
  it("groups extracted items by kind in reading order", () => {
    expect(groupItems(review().items).map(g => g.label)).toEqual(["Decisions", "Risks"]);
  });
  it("turns a chosen day into an instant", () => {
    expect(effectiveAt("2026-10-01")).toBe(new Date("2026-10-01T00:00:00").toISOString());
    expect(effectiveAt("")).toBeNull();
  });
});
