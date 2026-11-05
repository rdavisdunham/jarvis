// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { RunStopCard } from "./RunStopCard";
import { RunRow } from "./RunsPanel";
import { stopCard, stopLine, type RunDetail } from "./remotea-model";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const HEADLINE = "Verification timed out after 30 min at 94% of the suite, no failures.";
const ACTION = "Raise the verify timeout or verify again.";
const OUTPUT = ["....... [ 86%]", "....... [ 88%]", "....... [ 90%]", "....... [ 92%]", "....... [ 94%]"];

const parked = (values: Partial<RunDetail> = {}): RunDetail => ({
  id: "run-1", task: { id: "t1", title: "fix linear log in settings", status: "in_progress" }, state: "needs_you", state_revision: 9,
  state_reason: "verify_infra", reason_text: "Verification could not run on the host.", state_since: "2026-11-04T10:00:00Z",
  generation: 1, plan_revision: 1, tool: "claude", model: "", branch: "task/x", base_branch: "main", worktree: "",
  effective_permission_mode: "", turn_count: 4, turn_cap: 15, turn_seq: 4, started_at: null, finished_at: null,
  stop_requested_at: null, stop_mode: "", surf: "surf-1", host: { id: "h", name: "desk", connected: true },
  structured_answers: true, session: null, created_at: "2026-11-04T09:00:00Z", history: [], turns: [], commands: [], questions: [],
  verification: { verify_id: "full", status: "timed_out", at: "2026-11-04T10:00:00Z", evidence: "fresh" },
  attention_detail: { reason: "verify_infra", headline: HEADLINE, action: ACTION, detail: "", last_output: OUTPUT },
  plan: { revision: 1, title: "t", prd_text: "", steps: [], verify: { target_id: "jarvis", verify_id: "full" },
    criteria: [{ id: "c1", text: "tests pass", kind: "verify", verify_id: "full" }] },
  ...values,
} as RunDetail);

describe("the stop reason leads, without opening a dropdown", () => {
  it("puts the headline, action and last output in the card and the headline on the row", () => {
    const html = renderToStaticMarkup(<RunStopCard run={parked()} disabled={false} onApprove={() => {}} onVerifyAgain={() => {}} onSendBack={() => {}}/>);
    expect(html).toContain(HEADLINE);
    expect(html).toContain(ACTION);
    expect(html).toContain("....... [ 94%]");
    expect(html).not.toContain("<details");
    const row = renderToStaticMarkup(<RunRow run={parked()} onOpen={() => {}}/>);
    expect(row).toContain(HEADLINE);
    expect(stopLine(parked())).toBe(`${HEADLINE} ${ACTION}`);
    expect(stopLine(parked({ attention_detail: null }))).toBe("Verification could not run on the host.");
  });

  it("offers Approve, Verify again and Send back on every needs_you stop, but leaves plan stops to their cards", () => {
    const card = stopCard(parked())!;
    expect([card.approve.enabled, card.verifyAgain.enabled, card.sendBack.enabled]).toEqual([true, true, true]);
    expect(card.verifyAgain.label).toBe("Verify again");
    expect(card.approve.confirm).toBe("Verification has no passing run; approve anyway?");
    const turnCap = stopCard(parked({ state_reason: "turn_cap", attention_detail: { reason: "turn_cap", headline: "The run reached its turn cap.",
      action: "Approve the work as it is, or send the agent back with a note.", detail: "", last_output: [] } }))!;
    expect(turnCap.approve.enabled).toBe(true);
    expect(stopCard(parked({ state_reason: "plan_approval" }))).toBeNull();
    expect(stopCard(parked({ state: "awaiting_approval" }))).toBeNull();
    expect(stopCard(parked({ freestyle: true }))!.verifyAgain).toMatchObject({ enabled: false, reason: "Freestyle is on, so the judge is off." });
  });
});

describe("the three actions", () => {
  let host: HTMLDivElement, root: Root;
  beforeEach(() => { host = document.createElement("div"); document.body.append(host); root = createRoot(host); });
  afterEach(() => { act(() => root.unmount()); host.remove(); });
  const click = (text: string) => act(() => { [...host.querySelectorAll("button")].find(b => b.textContent?.includes(text))!.click(); });
  const type = (label: string, value: string) => act(() => {
    const area = host.querySelector<HTMLTextAreaElement>(`textarea[aria-label="${label}"]`)!;
    Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")!.set!.call(area, value);
    area.dispatchEvent(new Event("input", { bubbles: true }));
  });

  it("asks before approving and passes the note; verifies again; sends a note back", () => {
    const onApprove = vi.fn(), onVerifyAgain = vi.fn(), onSendBack = vi.fn();
    act(() => root.render(<RunStopCard run={parked()} disabled={false} onApprove={onApprove} onVerifyAgain={onVerifyAgain} onSendBack={onSendBack}/>));
    click("Approve");
    expect(onApprove).not.toHaveBeenCalled();
    expect(host.textContent).toContain("Verification has no passing run; approve anyway?");
    type("Why you approve", "CI is green");
    click("Approve anyway");
    expect(onApprove).toHaveBeenCalledWith("CI is green");
    click("Verify again");
    expect(onVerifyAgain).toHaveBeenCalledOnce();
    click("Send back to the agent");
    const send = [...host.querySelectorAll("button")].find(b => b.textContent === "Send back")!;
    expect(send.disabled).toBe(true);  // A note is required.
    type("Note for the agent", "Split the slow tests out");
    click("Send back");
    expect(onSendBack).toHaveBeenCalledWith("Split the slow tests out");
  });

  it("disables every action while another one is in flight", () => {
    act(() => root.render(<RunStopCard run={parked()} disabled onApprove={() => {}} onVerifyAgain={() => {}} onSendBack={() => {}}/>));
    expect([...host.querySelectorAll("button")].every(b => b.disabled)).toBe(true);
  });
});

describe("the timeline records the override", () => {
  it("shows an approve_override entry with its reason and note", async () => {
    const { timeline } = await import("./remotea-model");
    const run = parked({ history: [{ at: "2026-11-04T10:05:00Z", event: "approve_override", reason: "verify_infra", note: "CI is green" }] });
    const entry = timeline(run).find(e => e.title.startsWith("Approved by you"))!;
    expect(entry.detail).toBe("verification could not run · CI is green");
  });
});
