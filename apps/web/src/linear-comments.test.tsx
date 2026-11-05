// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { commentThreads, LinearComments, type LinearComment } from "./LinearComments";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
const c = (id: string, over: Partial<LinearComment> = {}): LinearComment => ({ id, issue_id: "i", parent_id: null, author: "Uttam",
  body: "Body " + id, url: `https://linear.app/c/${id}`, created_at: "2026-11-03T20:48:54Z", updated_at: null, edited_at: null, resolved_at: null, ...over });
let host: HTMLDivElement, root: Root;
beforeEach(() => { host = document.createElement("div"); document.body.append(host); root = createRoot(host); });
afterEach(() => { act(() => root.unmount()); host.remove(); vi.unstubAllGlobals(); });

describe("Linear comments on a task", () => {
  it("threads replies under their parent, oldest first, and keeps an orphan reply visible", () => {
    const threads = commentThreads([c("a"), c("b", { parent_id: "a" }), c("d"), c("e", { parent_id: "gone" })]);
    expect(threads.map(t => [t.comment.id, t.replies.map(r => r.id)])).toEqual([["a", ["b"]], ["d", []], ["e", []]]);
  });

  it("shows the copied comments with authors, plain-text bodies and a way to reply in Linear", async () => {
    const urls: string[] = [];
    vi.stubGlobal("fetch", async (url: string) => {
      urls.push(String(url));
      return new Response(JSON.stringify({ identifier: "ABCD-434", url: "https://linear.app/x/issue/ABCD-434", synced_at: "2026-11-05T22:58:37Z",
        comments: [c("a", { body: "**Nov 3 SSH check:** confirmed\nsecond line", author: "Davis Dunham" }), c("b", { parent_id: "a", edited_at: "2026-11-04T00:00:00Z" })] }));
    });
    await act(async () => { root.render(<LinearComments taskId="task-1"/>); });
    expect(urls).toEqual(["/api/v1/integrations/linear/tasks/task-1/comments"]);
    const section = host.querySelector('[aria-label="Linear comments"]')!;
    expect(section.querySelector(".detail-count")!.textContent).toBe("2");
    expect(section.querySelector(".linear-comment-body")!.textContent).toBe("**Nov 3 SSH check:** confirmed\nsecond line");
    expect(section.querySelector(".linear-comment.is-reply")!.textContent).toContain("edited");
    expect(section.querySelector<HTMLAnchorElement>('a[href="https://linear.app/x/issue/ABCD-434"]')!.textContent).toBe("Reply in Linear");
  });

  it("says so when the issue has no comments, and when they cannot load", async () => {
    vi.stubGlobal("fetch", async () => new Response(JSON.stringify({ identifier: null, url: null, synced_at: null, comments: [] })));
    await act(async () => { root.render(<LinearComments taskId="empty"/>); });
    expect(host.textContent).toContain("No comments on this issue in Linear yet.");
    vi.stubGlobal("fetch", async () => new Response(JSON.stringify({ error: { code: "NOT_FOUND", message: "This task has no Linear issue." } }), { status: 404 }));
    await act(async () => { root.render(<LinearComments taskId="local"/>); });
    expect(host.textContent).toContain("Comments could not load");
  });
});
