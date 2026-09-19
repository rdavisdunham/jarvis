import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { UsageReport, costLabel, type CostReport } from "./UsageReport";

const report: CostReport = { as_of: "2026-09-19T00:00:00Z", tracking_since: "2026-09-19T00:00:00Z", last_7_days_usd: 0.0031, last_30_days_usd: 1.02, partial_7_days: true, partial_30_days: true, features: [{ id: "note_organization", label: "Note organization", last_7_days_usd: 0.0031, last_30_days_usd: 1.02, usage_records: 4 }] };
describe("usage reporting", () => {
  it("shows tiny charges without rounding them to zero", () => {
    expect(costLabel(.0003)).toBe("$0.0003");
    expect(costLabel(12)).toBe("$12.00");
  });
  it("labels missing history and isolated eval exclusions", () => {
    const html = renderToStaticMarkup(<UsageReport report={report} />);
    expect(html).toContain("Note organization");
    expect(html).toContain("$0.0031");
    expect(html).toContain("Partial history");
    expect(html).toContain("isolated eval runs are excluded");
  });
  it("does not present an empty period as a predicted bill", () => {
    const html = renderToStaticMarkup(<UsageReport report={{ ...report, features: [] }} />);
    expect(html).toContain("No model usage recorded yet");
    expect(html).not.toContain("projected");
  });
});
