// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Questions } from "../Questions";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const contested = { key: "context:contested:project:p1:due_date", kind: "context", revision: 3, status: "pending", source_id: "k1",
  question: "These meeting notes disagree about the due date. Which is right?", answer_tool: "context.resolve",
  candidates: [{ assertion_id: "a1", revision: 1, statement: "Launch is Dec 1." }, { assertion_id: "a2", revision: 1, statement: "Launch is Dec 8." }],
  evidence: {}, actions: ["keep", "both"], answer_args: { question_key: "context:contested:project:p1:due_date", expected_revision: 3 } };
// The pre-service shape keeps actions and ids inside answer_args; they must not be sent back.
const link = { key: "context:link:l1", kind: "context", revision: 1, status: "pending", source_id: "k2", question: "Who is “Sam” in “Sam sends the deck”?",
  answer_tool: "context.resolve", candidates: [], evidence: {},
  answer_args: { question_key: "context:link:l1", expected_revision: 1, link_id: "l1", actions: ["link", "dismiss"] } };

let host: HTMLDivElement, root: Root, commands: { tool: string; arguments: Record<string, unknown> }[];
beforeEach(() => {
  host = document.createElement("div"); document.body.append(host); root = createRoot(host); commands = [];
  vi.stubGlobal("fetch", vi.fn(async (url: string, init: RequestInit = {}) => {
    if (url.startsWith("/api/v1/commands")) { const body = JSON.parse(String(init.body)); commands.push(body); return new Response(JSON.stringify({ data: {}, command_id: body.command_id })); }
    return new Response(JSON.stringify({ items: [contested, link], total: 2, next_offset: null, counts: {}, learning: { runs: [] } }));
  }));
});
afterEach(() => { act(() => root.unmount()); host.remove(); vi.unstubAllGlobals(); });
const settle = async () => { for (let i = 0; i < 4; i++) await act(async () => { await Promise.resolve(); }); };
const button = (label: string, scope: ParentNode = host) => [...scope.querySelectorAll("button")].find(b => b.textContent?.trim() === label)!;

describe("context questions", () => {
  it("keeps one contested statement or links a mention to a person", async () => {
    await act(async () => { root.render(<Questions people={[{ id: "act-sam", name: "Sam Lee" }]}/>); });
    await settle();
    const [first, second] = host.querySelectorAll<HTMLElement>(".question-card");
    expect(first.textContent).toContain("Launch is Dec 8.");
    expect(first.textContent).toContain("Meeting context");
    await act(async () => { button("Keep this one", first.querySelectorAll("li")[1]).click(); });
    await settle();
    expect(commands[0]).toMatchObject({ tool: "context.resolve", arguments: { question_key: contested.key, expected_revision: 3, action: "keep", assertion_id: "a2" } });
    expect(button("Both are right", first)).toBeTruthy();
    const select = second.querySelector<HTMLSelectElement>("select[aria-label='Who this is']")!;
    await act(async () => { select.value = "act-sam"; select.dispatchEvent(new Event("change", { bubbles: true })); });
    await act(async () => { button("Link", second).click(); });
    await settle();
    expect(commands[1].arguments).toEqual({ question_key: link.key, expected_revision: 1, action: "link", entity_type: "actor", entity_id: "act-sam" });
  });
});
