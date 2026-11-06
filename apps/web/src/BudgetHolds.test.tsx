import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { BudgetHolds, MonthSpend } from "./BudgetHolds";

describe("monthly spend", () => {
  it("shows recorded spend, the projection and one line for work in progress", () => {
    const html = renderToStaticMarkup(<MonthSpend spent={3.2} limit={20} projected={9.6} inFlight={0.28} />);
    expect(html).toContain("$3.20");
    expect(html).toContain(" / $20 spent this month");
    expect(html).toContain("about $9.60 by month end");
    expect(html).toContain("Plus $0.28 set aside for work that is still finishing.");
    expect(html).not.toMatch(/unconfirmed|never assumed|possible usage/i);
  });
  it("leaves out the limit when limits are off and the projection when history is partial", () => {
    const html = renderToStaticMarkup(<MonthSpend spent={1} projected={null} inFlight={0} />);
    expect(html).toContain("spent this month");
    expect(html).not.toContain(" / $");
    expect(html).not.toContain("pace");
    expect(html).not.toContain("set aside");
  });
  it("stays quiet when less than a cent is in progress", () => {
    expect(renderToStaticMarkup(<BudgetHolds usd={0.004} />)).toBe("");
    expect(renderToStaticMarkup(<BudgetHolds usd={0.01} />)).toContain("$0.01");
  });
});
