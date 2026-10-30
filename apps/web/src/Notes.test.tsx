import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { FrozenPrdNotice } from "./Notes";

describe("a PRD frozen with a live run", () => {
  it("names the run and says the human's edit does not change the frozen plan", () => {
    const html = renderToStaticMarkup(<FrozenPrdNotice runId="0123456789abcdef" />);
    expect(html).toContain("Frozen with run 01234567");
    expect(html).toContain('title="Run 0123456789abcdef"');
    expect(html).toContain("do not change the plan");
  });
});
