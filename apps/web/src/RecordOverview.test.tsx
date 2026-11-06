// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { RecordOverview, type OverviewData } from "./RecordOverview";
import { api } from "./api";

vi.mock("./api", async importOriginal => ({
  ...await importOriginal<typeof import("./api")>(), api: vi.fn(),
}));

const sections = {
  what_it_is: "ABC is a home services client; we work with its Customer Service team.",
  scope_and_themes: ["QA disputes and score corrections"], vocabulary: ["TI: Transcript Intelligence"],
  people: ["Janiece: customer service lead"], recent_changes: [], open_questions: [],
};
const labels = { what_it_is: "What it is", scope_and_themes: "Scope and themes", vocabulary: "Vocabulary",
  people: "People", recent_changes: "Recent changes", open_questions: "Open questions" };
const version = (n: number, what = sections.what_it_is) => ({ version: n, created_at: "2026-11-06T12:00:00Z", mode: "update",
  origin: "scheduled", stats: { tasks: 12, meeting_facts: 3 }, seed: "Seed", sections: { ...sections, what_it_is: what } });
const data = (over: Partial<OverviewData> = {}): OverviewData => ({
  eligible: true, enabled: true, labels, sections: Object.keys(labels) as OverviewData["sections"],
  overview: version(2), versions: [version(2), version(1)], pending: false, next_check_at: null, last_error: null,
  has_seed: true, ...over,
});

let element: HTMLDivElement;
let root: Root;
beforeEach(() => {
  (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
  element = document.createElement("div");
  document.body.append(element);
  root = createRoot(element);
});
afterEach(async () => {
  await act(async () => root.unmount());
  element.remove();
  vi.resetAllMocks();
});

describe("record overview", () => {
  it("shows the distilled sections, where they came from, and older versions on request", async () => {
    vi.mocked(api).mockImplementation(async (url: string) =>
      url.endsWith("?version=1") ? data({ overview: version(1, "The first take.") }) : data());
    await act(async () => root.render(<RecordOverview recordId="r1" canEdit revision={1} />));
    expect(element.textContent).toContain("Overview · v2");
    expect(element.textContent).toContain("QA disputes and score corrections");
    expect(element.textContent).toContain("from 12 tasks and 3 meeting facts");
    const select = element.querySelector<HTMLSelectElement>('select[aria-label="Overview version"]')!;
    await act(async () => { select.value = "1"; select.dispatchEvent(new Event("change", { bubbles: true })); });
    expect(element.textContent).toContain("The first take.");
    expect(element.textContent).toContain("Showing version 1 of 2");
  });

  it("refreshes on request and says when nothing changed", async () => {
    vi.mocked(api).mockResolvedValueOnce(data()).mockResolvedValueOnce({ ...data(), queued: false, reason: "no_change" });
    await act(async () => root.render(<RecordOverview recordId="r1" canEdit revision={1} />));
    const refresh = [...element.querySelectorAll("button")].find(b => b.textContent?.includes("Refresh"))!;
    await act(async () => refresh.click());
    expect(api).toHaveBeenLastCalledWith("/structure/records/r1/overview/refresh", { method: "POST", body: JSON.stringify({ rebuild: false }) });
    expect(element.textContent).toContain("Nothing new since the last version.");
  });

  it("explains how to start when there is no overview yet, and offers no refresh to viewers", async () => {
    vi.mocked(api).mockResolvedValue(data({ overview: null, versions: [], has_seed: false }));
    await act(async () => root.render(<RecordOverview recordId="r1" canEdit={false} revision={1} />));
    expect(element.textContent).toContain("Add a description above");
    expect([...element.querySelectorAll("button")].some(b => b.textContent?.includes("Refresh"))).toBe(false);
  });

  it("renders nothing for records that are not clients or projects", async () => {
    vi.mocked(api).mockResolvedValue(data({ eligible: false }));
    await act(async () => root.render(<RecordOverview recordId="t1" canEdit revision={1} />));
    expect(element.innerHTML).toBe("");
  });
});
