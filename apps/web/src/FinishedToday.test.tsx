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
it("counts only what was finished in Eridani and keeps Linear closures apart", () => {
  const at = "2026-11-02T15:00:00Z";
  const tasks = [
    { id: "mine", title: "Mine", status: "completed", finished_at: at, archived: false },
    { id: "synced", title: "Closed in Linear", status: "completed", finished_at: at, archived: false, external: { provider: "linear", finished_via: "linear" } },
  ] as Task[];
  const html = renderToStaticMarkup(<FinishedToday tasks={tasks} today="2026-11-02" zone="America/Chicago" busy={false} onTask={() => {}} onReopen={() => {}}/>);
  expect(html).toMatch(/<h2>Completed today<\/h2><span class="panel-count">1<\/span>/);
  expect(html).toContain("<summary>1 closed in Linear</summary>");
  expect(html).toContain('aria-label="Reopen Closed in Linear"');
});
