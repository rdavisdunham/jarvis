import { expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { FinishedToday } from "./FinishedToday";
import type { Task } from "./types";
it("renders separate visible completion and cancellation checklists with reopen controls", () => {
  const tasks = ["completed", "cancelled"].map(status => ({ id: status, title: status + " task", status, finished_at: "2026-11-02T15:00:00Z", archived: false })) as Task[];
  const html = renderToStaticMarkup(<FinishedToday tasks={tasks} today="2026-11-02" zone="America/Chicago" busy={false} onTask={() => {}} onReopen={() => {}}/>);
  expect(html).toContain("Completed today");
  expect(html).toContain("Canceled today");
  expect(html).toContain('aria-pressed="true"');
  expect(html).toContain('aria-label="Reopen completed task"');
  expect(html).toContain("cancelled task");
});
