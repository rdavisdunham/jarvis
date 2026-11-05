// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LinearSettings, type LinearStatus } from "./Linear";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

type Change = LinearStatus["recent_changes"][number];
const change = (overrides: Partial<Change> = {}): Change => ({
  job_id: "job-one", status: "queued", message: "Waiting to save", created_at: "2026-11-04T15:04:05+00:00",
  finished_at: null, task_id: "task-outside-list", issue_identifier: "ERI-42", issue_title: "Requested title",
  payload: { base: { title: "Before" }, patch: { title: "Requested title" } }, result: null, ...overrides,
});
const status = (changes: Change[] = [change(), change({ job_id: "job-two", status: "succeeded", result: {} })]): LinearStatus => ({
  connected: true, workspace: "Workspace", revision: 1, team_ids: [], only_mine: true, status: "ready", error: "",
  last_sync_at: null, teams: [], states: [], users: [], projects: [], recent_changes: changes,
});
let host: HTMLDivElement, root: Root, current: LinearStatus;
const onOpenTask = vi.fn();
const mutate = vi.fn(async () => true);
beforeEach(() => {
  current = status(); onOpenTask.mockClear(); mutate.mockClear();
  host = document.createElement("div"); document.body.append(host); root = createRoot(host);
  vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(current), { status: 200 })));
});
afterEach(() => { act(() => root.unmount()); host.remove(); vi.unstubAllGlobals(); });
async function render(revision = 1, key = "settings") {
  await act(async () => { root.render(<LinearSettings key={key} revision={revision} mutate={mutate} onOpenTask={onOpenTask} />); });
}
const outer = () => host.querySelector<HTMLDetailsElement>(".linear-log")!;
const rows = () => [...host.querySelectorAll<HTMLDetailsElement>(".linear-log-details")];
async function toggle(details: HTMLDetailsElement) {
  await act(async () => { details.querySelector("summary")!.click(); });
}
// jsdom has no layout; check the native disclosure visibility contract via ancestors.
function disclosed(node: Element) {
  for (let parent = node.parentElement; parent; parent = parent.parentElement) {
    if (parent instanceof HTMLDetailsElement && !parent.open) return false;
  }
  return true;
}

describe("Recent Linear changes", () => {
  it("starts closed, hides the entire log again on close, and resets when Settings mounts", async () => {
    await render();
    expect(outer().open).toBe(false);
    const title = host.querySelector(".linear-log-entry .settings-item-title")!;
    expect(disclosed(title)).toBe(false);
    expect(rows().every(row => !row.open)).toBe(true);
    await toggle(outer()); expect(disclosed(title)).toBe(true);
    await toggle(rows()[0]);
    expect(disclosed(host.querySelector(".linear-log-json")!)).toBe(true);
    await toggle(outer());
    expect(disclosed(title)).toBe(false);
    expect(disclosed(host.querySelector(".linear-log-json")!)).toBe(false);
    await render(2, "remounted"); expect(outer().open).toBe(false);
  });

  it("expands rows independently and preserves disclosure state by job id through refreshes", async () => {
    await render(); await toggle(outer()); await toggle(rows()[0]);
    expect(rows().map(row => row.open)).toEqual([true, false]);
    current = status([change({ job_id: "newest" }), ...current.recent_changes]);
    await render(2);
    expect(outer().open).toBe(true);
    expect(rows().map(row => row.open)).toEqual([false, true, false]);
    await toggle(rows()[2]);
    expect(rows().map(row => row.open)).toEqual([false, true, true]);
    await toggle(outer());
    current = status([change({ job_id: "another" }), ...current.recent_changes]);
    await render(3); expect(outer().open).toBe(false);
    expect(vi.mocked(fetch)).toHaveBeenCalledTimes(3);
  });

  it("labels local queued and finished times with a timezone and preserves exact timestamps", async () => {
    const created = "2026-11-04T15:04:05+00:00", finished = "2026-11-04T15:05:05Z";
    current = status([change({ finished_at: finished, status: "succeeded" })]);
    await render(); await toggle(outer()); await toggle(rows()[0]);
    const format = new Intl.DateTimeFormat(undefined, { year: "numeric", month: "short", day: "numeric",
      hour: "numeric", minute: "2-digit", second: "2-digit", timeZoneName: "short" });
    const times = [...host.querySelectorAll(".linear-log time")];
    expect(times.map(t => t.getAttribute("datetime"))).toEqual([created, created, finished]);
    expect(times[0].textContent).toBe(format.format(new Date(created)));
    const zone = format.formatToParts(new Date(created)).find(p => p.type === "timeZoneName")!.value;
    expect(times.every(t => t.textContent!.includes(zone))).toBe(true);
    expect(times[0].parentElement!.textContent).toContain("Queued:");
    expect(rows()[0].textContent).toContain("Finished");
  });

  it("preserves nested stored JSON as escaped formatted text, including nulls, empty objects and multiline notes", async () => {
    const unsafe = '<img src=x onerror="alert(1)">';
    const payload = { base: { title: "雪", description: "First line\nSecond line" }, patch: { title: unsafe }, list: [null, {}, [1, true]] };
    const result = { message: "Saved", metadata: { value: null, html: "<script>bad()</script>" } };
    current = status([change({ issue_title: unsafe, payload, result }), change({ job_id: "empty", payload: {}, result: {} })]);
    await render(); await toggle(outer()); await toggle(rows()[0]);
    const code = rows()[0].querySelectorAll("pre code");
    expect(code[0].textContent).toBe(JSON.stringify(payload, null, 2));
    expect(code[1].textContent).toBe(JSON.stringify(result, null, 2));
    expect(host.querySelector(".linear-log img, .linear-log script")).toBeNull();
    expect(host.querySelector(".settings-item-title")!.textContent).toContain(unsafe);
    await toggle(rows()[1]);
    expect([...rows()[1].querySelectorAll("pre code")].map(c => c.textContent)).toEqual(["{}", "{}"]);
    expect(rows()[1].textContent).not.toContain("No outcome recorded yet");
  });

  it("keeps missing context, null outcomes and invalid or absent timestamps readable without misleading task actions", async () => {
    current = status([change({ task_id: null, issue_identifier: null, issue_title: null, created_at: null }),
      change({ job_id: "invalid", task_id: null, created_at: "bad timestamp", finished_at: "also bad", status: "failed" })]);
    await render(); await toggle(outer());
    for (const row of rows()) await toggle(row);
    expect(host.querySelector(".linear-log")!.textContent).toContain("Time unavailable");
    expect(host.textContent).not.toContain("Invalid Date");
    expect(rows()[0].textContent).toContain("No completion time recorded");
    expect(rows()[0].textContent).toContain("No outcome recorded yet");
    expect(rows()[0].textContent).toContain("Local task unavailable");
    expect(rows()[0].textContent).toContain("Issue identifier unavailable");
    expect(rows()[0].textContent).toContain("Issue title unavailable");
    expect(host.querySelector(".linear-log button")).toBeNull();
    expect(vi.mocked(fetch)).toHaveBeenCalledTimes(1);
    expect(mutate).not.toHaveBeenCalled();
  });

  it("opens the exact task through its callback without expanding the row or requesting Linear", async () => {
    await render(); await toggle(outer());
    await act(async () => { host.querySelector<HTMLButtonElement>(".linear-log button")!.click(); });
    expect(onOpenTask).toHaveBeenCalledWith("task-outside-list");
    expect(rows()[0].open).toBe(false);
    expect(outer().open).toBe(true);
    expect(vi.mocked(fetch)).toHaveBeenCalledTimes(1);
    expect(mutate).not.toHaveBeenCalled();
  });

  it("shows the empty state only inside the selected dropdown", async () => {
    current = status([]); await render();
    const empty = outer().querySelector(".settings-empty")!;
    expect(empty.textContent).toBe("No changes yet.");
    expect(disclosed(empty)).toBe(false);
    await toggle(outer()); expect(disclosed(empty)).toBe(true);
    await toggle(outer()); expect(disclosed(empty)).toBe(false);
  });
});
