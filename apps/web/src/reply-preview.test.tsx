// @vitest-environment jsdom
// Dani at work: thinking dots, then her reply streaming in, round by round (ReplyPreview.tsx).
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { PendingReplies, WORK_DELTA, useReplyStreams } from "./ReplyPreview";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
let host: HTMLDivElement, root: Root;
beforeEach(() => { host = document.createElement("div"); document.body.append(host); root = createRoot(host); });
afterEach(() => { act(() => root.unmount()); host.remove(); });

function Chat({ ids, sending = false }: { ids: string[]; sending?: boolean }) {
  const streams = useReplyStreams(ids);
  return <PendingReplies ids={ids} streams={streams} sending={sending} />;
}
const delta = (w: string, r: number, t: string) =>
  act(() => { window.dispatchEvent(new CustomEvent(WORK_DELTA, { detail: { w, r, s: 1, t } })); });

describe("pending replies", () => {
  it("shows thinking dots while sending and while Dani works, then streams her reply", () => {
    act(() => root.render(<Chat ids={[]} sending />));
    expect(host.querySelector("[aria-label='Dani is thinking']")).not.toBeNull();
    act(() => root.render(<Chat ids={["w1"]} />));
    expect(host.querySelector(".thinking-dots")).not.toBeNull();
    delta("w1", 0, "Let me ");
    delta("w1", 0, "check.");
    expect(host.querySelector(".pending-reply p")?.textContent).toBe("Let me check.");
    expect(host.querySelector(".stream-caret")).not.toBeNull();
  });

  it("starts a fresh preview each model round and ignores late batches from an earlier one", () => {
    act(() => root.render(<Chat ids={["w1"]} />));
    delta("w1", 0, "Let me check.");
    delta("w1", 10, "Added ");
    delta("w1", 0, "stale");
    delta("w1", 10, "Alpha.");
    expect(host.querySelector(".pending-reply p")?.textContent).toBe("Added Alpha.");
  });

  it("drops the preview once the request is no longer active and ignores other requests", () => {
    act(() => root.render(<Chat ids={["w1"]} />));
    delta("w1", 0, "Done soon");
    delta("w2", 0, "not mine");
    expect(host.querySelectorAll(".pending-reply")).toHaveLength(1);
    act(() => root.render(<Chat ids={[]} />));
    expect(host.querySelector(".pending-reply")).toBeNull();
  });
});
