// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { MemoryRow } from "./MemoryActions";
import type { Memory } from "./types";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement, root: Root;
beforeEach(() => {
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
});
afterEach(() => {
  act(() => root.unmount());
  host.remove();
});

const memory = (overrides: Partial<Memory> = {}): Memory => ({
  id: "m1", content: "The user is not named Davis.", evidence: "", tags: [], embedding_model: null,
  source_id: "s1", attribution: "automatic", created_at: "2026-10-03T15:00:00Z", revision: 1,
  source: { origin: "conversation", conversation_id: "c1", created_at: "2026-10-02T15:00:00Z" },
  ...overrides,
});

const button = (text: string) =>
  [...host.querySelectorAll("button")].find(b => b.textContent?.includes(text)) as HTMLButtonElement;

it("links a learned fact to the conversation it came from", () => {
  const open = vi.fn();
  act(() => root.render(<MemoryRow memory={memory()} onCorrect={() => {}} onForget={async () => true} onWrong={async () => true} onOpenConversation={open}/>));
  expect(host.textContent).toContain("Learned from a conversation on ");
  act(() => button("a conversation on").click());
  expect(open).toHaveBeenCalledWith("c1");
});

it("names stated facts and sources without a conversation plainly", () => {
  act(() => root.render(<>
    <MemoryRow memory={memory({ id: "a", attribution: "owner_statement" })} onCorrect={() => {}} onForget={async () => true} onWrong={async () => true}/>
    <MemoryRow memory={memory({ id: "b", source: null })} onCorrect={() => {}} onForget={async () => true} onWrong={async () => true}/>
  </>));
  expect(host.textContent).toContain("You told Dani");
  expect(host.textContent).toContain("Learned from a saved source");
  expect(host.querySelector(".memory-source-link")).toBeNull();
});

it("rejects a fact in one click, with no confirmation dialog", async () => {
  let finish = () => {};
  const wrong = vi.fn(() => new Promise<void>(resolve => { finish = resolve; }));
  act(() => root.render(<MemoryRow memory={memory()} onCorrect={() => {}} onForget={async () => true} onWrong={wrong}/>));
  act(() => button("This is wrong").click());
  expect(wrong).toHaveBeenCalledTimes(1);
  expect(button("This is wrong").disabled).toBe(true);
  expect(document.querySelector("[role=dialog]")).toBeNull();
  await act(async () => finish());
  expect(button("This is wrong").disabled).toBe(false);
});
