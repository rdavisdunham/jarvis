import { describe, expect, it } from "vitest";
import { attentionKey, brainstormSaid, DEFAULT_DRAFT, startBody, stopCard, type Run, type RunDetail } from "./remotea-model";

describe("brainstorm: talk it through before planning", () => {
  it("is a start option beside Auto and Skip", () => {
    expect(startBody("t1", { ...DEFAULT_DRAFT, interview: "talk" })).toMatchObject({ task_id: "t1", brainstorm: true });
    expect(startBody("t1", { ...DEFAULT_DRAFT, interview: "talk" })).not.toHaveProperty("interview");
    expect(startBody("t1", DEFAULT_DRAFT)).not.toHaveProperty("brainstorm");
    expect(startBody("t1", { ...DEFAULT_DRAFT, interview: "skip" })).toMatchObject({ interview: false });
    // Freestyle has no plan to talk towards.
    expect(startBody("t1", { ...DEFAULT_DRAFT, interview: "talk", freestyle: true })).not.toHaveProperty("brainstorm");
  });

  it("shows the agent's latest message of this generation", () => {
    const turn = (seq: number, text: string, extra = {}) => ({ id: "t" + seq, seq, text, clipped: false, truncated: false, generation: 1, at: "", prompt: "", ...extra });
    const run = { generation: 1, turns: [turn(2, "Second question?"), turn(1, "First question?"), turn(3, "my reply", { role: "user" as const }),
      turn(9, "from an older generation", { generation: 0 })] };
    expect(brainstormSaid(run)).toBe("Second question?");
    expect(brainstormSaid({ generation: 1, turns: [] })).toBe("");
  });

  it("waits for a reply: an answer, not a result to approve", () => {
    const run = { id: "r", state: "needs_you", state_reason: "brainstorm", attention_detail: { headline: "h", action: "a" } } as unknown as RunDetail;
    expect(attentionKey(run as unknown as Run)).toBe("answer");
    expect(stopCard(run)).toBeNull();
  });
});
