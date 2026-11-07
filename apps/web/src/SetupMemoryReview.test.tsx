// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SetupMemoryReview } from "./SetupMemoryReview";
import type { MemoryCandidate } from "./setup-model";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const candidates: MemoryCandidate[] = [
  { id: "a", content: "Davis teaches chemistry.", fact_key: "job", evidence: "I teach chemistry", evidence_source_id: "s",
    existing_memory_id: null, action: "add" },
  { id: "b", content: "Davis coaches robotics.", fact_key: "coach", evidence: "", evidence_source_id: null,
    existing_memory_id: "m", action: "keep" },
];

describe("the About you review", () => {
  let host: HTMLDivElement, root: Root;
  beforeEach(() => { host = document.createElement("div"); document.body.append(host); root = createRoot(host); });
  afterEach(() => { act(() => root.unmount()); host.remove(); vi.unstubAllGlobals(); });

  it("saves edits and skipped rows in one command", async () => {
    const sent: { tool: string; arguments: { choices: unknown[] } }[] = [];
    vi.stubGlobal("fetch", vi.fn(async (_url: string, init: RequestInit) => {
      sent.push(JSON.parse(String(init.body)));
      return new Response(JSON.stringify({ data: { status: "reviewing", saved_ids: ["x"], retired_ids: [] } }),
        { status: 200, headers: { "Content-Type": "application/json" } });
    }));
    let saved = 0;
    act(() => root.render(<SetupMemoryReview candidates={candidates} knownCount={3} onClose={() => {}}
      onKeepTalking={() => {}} onSaved={() => { saved += 1; }}/>));
    expect(host.textContent).toContain("These are the things Dani will keep in mind.");
    expect(host.textContent).toContain("Also learned: 3 facts from earlier chats will stay as they are.");
    expect([...host.querySelectorAll(".setup-memory-row .chip")].map(c => c.textContent)).toEqual(["From this conversation", "Already known"]);
    const [first] = [...host.querySelectorAll("textarea")];
    act(() => {
      const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")!.set!;
      setter.call(first, "Davis teaches AP chemistry.");
      first.dispatchEvent(new Event("input", { bubbles: true }));
    });
    act(() => (host.querySelectorAll("input[type=checkbox]")[1] as HTMLInputElement).click());
    const save = [...host.querySelectorAll("button")].find(b => b.textContent === "Save about you")!;
    await act(async () => { save.click(); });
    expect(sent[0].tool).toBe("setup.memories_save");
    expect(sent[0].arguments.choices).toEqual([
      { id: "a", content: "Davis teaches AP chemistry.", include: true },
      { id: "b", content: "Davis coaches robotics.", include: false },
    ]);
    expect(saved).toBe(1);
  });
});
