import { describe, expect, it, vi } from "vitest";
import type { CalendarEntry, Task } from "./types";
import { activeTodayEntry, finishedToday, mergeFinishedTasks, readFinishedTasks } from "./task-finished";
vi.mock("./api", () => ({ api: vi.fn() }));
import { api } from "./api";
const task = (id: string, extra: Partial<Task> = {}) => ({ id, title: id, status: "completed", completed_at: "2026-11-02T07:30:00Z", archived: false, ...extra }) as Task;
describe("daily finished work", () => {
  it("uses configured local calendar day across fall DST, regardless of due date", () => {
    const rows = [task("undated"), task("future", { due_date: "2027-01-01" }), task("yesterday", { finished_at: "2026-11-02T05:59:59Z" }), task("cancel", { status: "cancelled", completed_at: null, finished_at: "2026-11-02T06:00:00Z" })];
    expect(finishedToday(rows, "2026-11-02", "America/Chicago").map(t => t.id).sort()).toEqual(["cancel", "future", "undated"]);
  });
  it("retains automatic history but excludes manual/external archives, templates and quick children", () => {
    const rows = [task("automatic", { archived: true, archive_reason: "automatic" }), task("manual", { archived: true, archive_reason: "manual" }), task("external", { archived: true, archive_reason: "external" }), task("template", { is_template: true }), task("child", { quick_list_parent_id: "parent" }), task("unknown cancel", { status: "cancelled", finished_at: null })];
    expect(finishedToday(rows, "2026-11-02", "America/Chicago").map(t => t.id)).toEqual(["automatic"]);
  });
  it("reopening overrides stale history without duplicates", () => {
    const merged = mergeFinishedTasks([task("done")], [task("done", { status: "open", finished_at: null })]);
    expect(merged).toHaveLength(1);
    expect(finishedToday(merged, "2026-11-02", "America/Chicago")).toEqual([]);
  });
  it("removes finished tasks from agenda even with stale calendar projections", () => {
    const entry = { kind: "task", task_id: "done", status: "open" } as CalendarEntry;
    expect(activeTodayEntry(entry, [task("done")])).toBe(false);
    expect(activeTodayEntry({ ...entry, status: "cancelled" }, [])).toBe(false);
    expect(activeTodayEntry({ ...entry, kind: "event", status: "completed" }, [])).toBe(true);
    expect(activeTodayEntry(entry, [task("done", { status: "open" })])).toBe(true);
  });
});

it("pages the configured daily query without widening ordinary loads", async () => {
  vi.mocked(api).mockResolvedValueOnce({ items: [task("one")], next_cursor: "cursor" }).mockResolvedValueOnce({ items: [task("two")], next_cursor: null });
  expect((await readFinishedTasks("2026-11-02", "America/Chicago")).map(t => t.id)).toEqual(["one", "two"]);
  const urls = vi.mocked(api).mock.calls.map(([url]) => new URL(url, "https://example.test"));
  expect(urls).toHaveLength(2);
  for (const url of urls) {
    expect(url.pathname).toBe("/tasks");
    expect(url.searchParams.get("finished_on")).toBe("2026-11-02");
    expect(url.searchParams.get("timezone")).toBe("America/Chicago");
    expect(url.searchParams.has("include_archived")).toBe(false);
  }
  expect(urls[1].searchParams.get("before")).toBe("cursor");
});
