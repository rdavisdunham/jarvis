// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { durableStorageKey, setDurableScope } from "./api";
import { InterviewForm } from "./Remotea";
import type { Interview, RunDetail } from "./remotea-model";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
const interview: Interview = {
  id: "notes-interview", run_id: "notes-run", generation: 1, status: "open", answers: {}, answered_at: null,
  created_at: "2026-11-02T23:00:00Z",
  questions: [{ id: "scope", question: "Which widgets?", options: [{ label: "All widgets" }, { label: "Charts" }], multi: false, free_text: false }],
};
const run = { id: interview.run_id, state: "interviewing", generation: 1,
  decisions: [{ id: interview.id, kind: "interview", run_id: interview.run_id, revision: 1, status: "open", payload: {}, actions: ["answer"], generation: 1, plan_revision: 0 }],
} as unknown as RunDetail;
let host: HTMLDivElement, root: Root;
let posts: { url: string; body: unknown }[];
const notes = "Include the charts first.\nKeep admin widgets unchanged.";
function seed(picked: Record<string, string[]>, typed: Record<string, string>) {
  const prefix = durableStorageKey("eri-draft:")!;
  // Original draft keys and plain-value format must continue to restore correctly.
  localStorage.setItem(prefix + ":interview-picks:" + interview.id, JSON.stringify(picked));
  localStorage.setItem(prefix + ":interview-text:" + interview.id, JSON.stringify(typed));
}
const button = (label: string) => [...host.querySelectorAll<HTMLButtonElement>("button")].find(element => element.textContent === label)!;
const textarea = () => host.querySelector<HTMLTextAreaElement>('textarea[aria-label="Answer: Which widgets?"]')!;
async function render() {
  await act(async () => { root.render(<InterviewForm interview={interview} run={run} onDone={() => {}}/>); await Promise.resolve(); });
}
function type(value: string) {
  Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")!.set!.call(textarea(), value);
  textarea().dispatchEvent(new Event("input", { bubbles: true }));
}
beforeEach(() => {
  localStorage.clear(); setDurableScope("interview-owner:personal");
  host = document.createElement("div"); document.body.append(host); root = createRoot(host); posts = [];
  vi.stubGlobal("fetch", async (url: string, init: RequestInit = {}) => {
    if (init.method === "POST") posts.push({ url, body: JSON.parse(String(init.body)) });
    return new Response(JSON.stringify(init.method === "POST" ? { receipt: { status: "saved" } } : { value: null, revision: 0 }));
  });
});
afterEach(() => { act(() => root.unmount()); host.remove(); vi.unstubAllGlobals(); });

it("restores an existing selected option plus multiline notes and submits both through the shared decision", async () => {
  seed({ scope: ["All widgets"] }, { scope: notes });
  await render();
  expect(button("All widgets").getAttribute("aria-pressed")).toBe("true");
  expect(textarea().value).toBe(notes);
  expect(textarea().placeholder).toBe("Add notes");
  expect(button("Send answers").disabled).toBe(false);
  await act(async () => { button("Send answers").click(); await Promise.resolve(); });
  expect(posts).toEqual([{ url: `/api/v1/remotea/decisions/${interview.id}/respond`, body: {
    action_id: expect.any(String), expected_revision: 1, action: "answer", answers: { scope: `All widgets\n\nNotes: ${notes}` },
  } }]);
});

it("keeps notes while toggling a single-choice option and allows notes alone from an old restricted question", async () => {
  seed({ scope: ["All widgets"] }, { scope: notes });
  await render();
  act(() => button("All widgets").click());
  expect(textarea().placeholder).toBe("Or type anything here");
  expect(textarea().value).toBe(notes);
  expect(button("Send answers").disabled).toBe(false);
  act(() => button("Charts").click());
  expect(textarea().placeholder).toBe("Add notes");
  expect(textarea().value).toBe(notes);
  await act(async () => { button("Send answers").click(); await Promise.resolve(); });
  expect(posts[0].body).toMatchObject({ answers: { scope: `Charts\n\nNotes: ${notes}` } });
});

it("submits text alone without selecting an option even when the persisted free_text flag is false", async () => {
  seed({}, {});
  await render();
  expect(textarea().placeholder).toBe("Or type anything here");
  act(() => type("My own scope\nwith more detail"));
  expect(button("Send answers").disabled).toBe(false);
  await act(async () => { button("Send answers").click(); await Promise.resolve(); });
  expect(posts[0].body).toMatchObject({ answers: { scope: "My own scope\nwith more detail" } });
});
