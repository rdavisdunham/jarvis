// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { SetupConversation } from "./SetupConversation";
import type { SetupSession } from "./setup-model";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const session: SetupSession = { status: "interviewing", revision: 1, conversation_id: "c1", mode: "typed", focus: "both",
  notes: { covered: ["you", "week"] }, tree_proposal_id: null, history_off: true };

describe("the setup conversation strip", () => {
  let host: HTMLDivElement, root: Root;
  beforeEach(() => { host = document.createElement("div"); document.body.append(host); root = createRoot(host); });
  afterEach(() => { act(() => root.unmount()); host.remove(); });
  const names = () => [...host.querySelectorAll("button")].map(b => b.textContent);

  it("ticks covered topics and keeps Switch to voice visible in typed mode", () => {
    let switched = 0;
    act(() => root.render(<SetupConversation session={session} voiceActive={false} voiceAvailable busy={false}
      onSwitchToVoice={() => { switched += 1; }}/>));
    const topics = [...host.querySelectorAll(".setup-topics li")];
    expect(topics.map(t => t.textContent)).toEqual(["Getting to know you (covered)", "Your week (covered)", "Areas and goals",
      "What’s in flight", "How to help"]);
    expect(topics.filter(t => t.classList.contains("chip-done"))).toHaveLength(2);
    expect(names()).toEqual(["Switch to voice"]);
    act(() => host.querySelector("button")!.click());
    expect(switched).toBe(1);
    expect(host.textContent).toContain("Dani keeps this setup conversation");
  });

  it("hides Switch to voice while voice runs, and shows Propose when the backend can build one", () => {
    act(() => root.render(<SetupConversation session={session} voiceActive voiceAvailable busy={false}
      onSwitchToVoice={() => {}} onPropose={() => {}}/>));
    expect(names()).toEqual(["Propose my organization"]);
  });
});
