import { describe, expect, it } from "vitest";
import { byRepository, confirmedText, type Learning } from "./LearningsSettings";

const entry = (id: string, repository: string, confirmations = 0): Learning => ({ id, kind: "bug", title: id, applies_when: "", learning: "",
  paths: [], confirmations, run_id: null, repository, created_at: "" });

describe("compound: what runs have learned", () => {
  it("groups entries by repository and keeps the server's order", () => {
    expect(byRepository([entry("a", "eridani", 2), entry("b", "remotea"), entry("c", "eridani"), entry("d", "")]).map(([name, rows]) => [name, rows.map(r => r.id)]))
      .toEqual([["eridani", ["a", "c"]], ["remotea", ["b"]], ["Unnamed repository", ["d"]]]);
  });
  it("says how often an entry came up again", () => {
    expect(confirmedText(0)).toBe("");
    expect(confirmedText(1)).toBe("Came up again in 1 later run");
    expect(confirmedText(3)).toBe("Came up again in 3 later runs");
  });
});
