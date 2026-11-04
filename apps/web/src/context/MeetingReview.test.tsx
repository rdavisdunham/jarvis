// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MeetingReview } from "./MeetingReview";
import type { Review } from "./context-model";
import { page, proposal, review } from "./test-fixtures";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

type Call = { url: string; method: string; body: Record<string, unknown> };
type Reply = [number, unknown] | "network";
let host: HTMLDivElement, root: Root, calls: Call[], current: Review, queued: Record<string, Reply[]>;
const R = "/api/v1/context/reviews/src-1";
beforeEach(() => {
  host = document.createElement("div"); document.body.append(host); root = createRoot(host);
  calls = []; current = review(); queued = {};
  vi.stubGlobal("requestAnimationFrame", (f: FrameRequestCallback) => { f(0); return 0; });
  vi.stubGlobal("fetch", vi.fn(async (url: string, init: RequestInit = {}) => {
    const method = (init.method ?? "GET").toUpperCase(), path = url.split("?")[0];
    const call = { url, method, body: init.body ? JSON.parse(String(init.body)) : {} }; calls.push(call);
    const reply = queued[method + " " + path]?.shift()
      ?? (method === "GET" && path === R ? [200, current] as Reply
        : method === "GET" && path === "/api/v1/context/sources/src-1/segments" ? [200, page()] as Reply
        : method === "GET" && path.startsWith("/api/v1/context/evidence/") ? [200, { ...current.items[0], gaps: [], evidence: [
          { evidence_id: "e1", role: "support", origin: "independent", source_id: "src-1", source_title: "Pricing sync", occurred_at: null,
            segment: 0, speaker: "Speaker A", quote: "We move pricing to annual plans.", current_revision: true }] }] as Reply
        : method === "POST" ? [200, {}] as Reply : [404, { error: { code: "NOT_FOUND", message: "Not found." } }] as Reply);
    if (reply === "network") throw new TypeError("offline");
    return new Response(JSON.stringify(reply[1]), { status: reply[0] });
  }));
});
afterEach(() => { act(() => root.unmount()); host.remove(); vi.unstubAllGlobals(); });

async function settle() { for (let i = 0; i < 4; i++) await act(async () => { await Promise.resolve(); }); }
async function render(props: Partial<React.ComponentProps<typeof MeetingReview>> = {}) {
  await act(async () => { root.render(<MeetingReview sourceId="src-1" onBack={() => {}} onPerson={() => {}} {...props}/>); });
  await settle();
}
const text = () => host.textContent ?? "";
const button = (label: string, scope: ParentNode = host) => [...scope.querySelectorAll("button")].find(b => b.textContent?.trim().startsWith(label))!;
const box = (title: string) => host.querySelector<HTMLInputElement>(`input[type=checkbox][aria-label$="${title}"]`)!;
const click = async (el: HTMLElement) => { await act(async () => { el.click(); }); await settle(); };
const posts = (action: string) => calls.filter(c => c.method === "POST" && c.url === `${R}/${action}`);
function type(el: HTMLInputElement | HTMLTextAreaElement, value: string) {
  const setter = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(el), "value")!.set!;
  act(() => { setter.call(el, value); el.dispatchEvent(new Event("input", { bubbles: true })); });
}

describe("MeetingReview", () => {
  it("shows the brief, grouped proposals, badges, speakers and highlighted transcript evidence", async () => {
    await render();
    expect(text()).toContain("Pricing sync");
    expect(text()).toContain("Pricing moves to annual plans.");
    for (const heading of ["Work to add", "Changes to existing work", "Decisions", "Risks", "Speakers", "Transcript"]) expect(text()).toContain(heading);
    expect(host.querySelectorAll(".badge-inferred").length).toBe(2); // the risk and Speaker A
    expect(host.querySelector(".badge-contested")?.textContent).toBe("Contested");
    expect(box("Send the pricing deck").checked).toBe(true);
    expect(box("Kim drafts the FAQ").checked).toBe(false);
    expect(box("Move launch to Dec 1").checked).toBe(false);
    expect(host.querySelector(".proposal-diff")?.textContent).toContain("2026-11-20");
    expect(host.querySelectorAll(".transcript-segment.is-evidence").length).toBe(1);
    expect(host.querySelector<HTMLSelectElement>("select[aria-label=Destination]")!.value).toBe("eridani");
    expect(calls.every(c => c.url.startsWith("/api/v1/context/"))).toBe(true);
  });

  it("adds selected tasks and applies updates as separate approvals, using the server's new revisions", async () => {
    await render();
    await click(box("Kim drafts the FAQ"));
    await click(box("Move launch to Dec 1"));
    const select = host.querySelector<HTMLSelectElement>("select[aria-label=Destination]")!;
    await act(async () => { select.value = "linear"; select.dispatchEvent(new Event("change", { bubbles: true })); });
    queued[`POST ${R}/approve`] = [[200, { review: review({ revision: 5, destination: "linear", proposals: [
      proposal({ revision: 2, disposition: "approved", publication: { id: "pub-q", status: "queued", task_id: null } }),
      proposal({ id: "p-other", title: "Kim drafts the FAQ", revision: 2, disposition: "approved", preselected: false, publication: { id: "pub-q", status: "queued", task_id: null } }),
      review().proposals[2]] }) }]];
    await click(button("Add selected tasks (2)"));
    const [approve] = posts("approve");
    expect(approve.body).toMatchObject({ destination: "linear", items: [{ proposal_id: "p-own", revision: 1 }, { proposal_id: "p-other", revision: 1 }] });
    expect(typeof approve.body.command_id).toBe("string");
    expect(text()).toContain("Queued");
    expect(box("Send the pricing deck").disabled).toBe(true);
    expect(box("Move launch to Dec 1").checked).toBe(true); // unchanged revision keeps the reader's choice
    await click(button("Apply selected updates (1)"));
    expect(posts("apply-updates")[0].body.items).toEqual([{ proposal_id: "p-upd", revision: 1 }]);
  });

  it("shows a refresh prompt on a stale revision and saves nothing", async () => {
    await render();
    queued[`POST ${R}/approve`] = [[409, { error: { code: "REVISION_CONFLICT", message: "This proposal changed." } }]];
    await click(button("Add selected tasks"));
    expect(host.querySelector(".meeting-stale")?.textContent).toContain("changed since you opened it");
    expect(button("Add selected tasks").disabled).toBe(true);
    current = review({ revision: 6, proposals: [proposal({ revision: 2, title: "Send the revised deck" }), ...review().proposals.slice(1)] });
    await click(button("Refresh"));
    expect(host.querySelector(".meeting-stale")).toBeNull();
    expect(box("Send the revised deck").checked).toBe(true);
  });

  it("retries after a lost connection with the same command ID", async () => {
    await render();
    queued[`POST ${R}/approve`] = ["network"];
    await click(button("Add selected tasks"));
    expect(host.querySelector("[role=alert]")?.textContent).toContain("Connection lost");
    await click(button("Add selected tasks"));
    const [first, second] = posts("approve");
    expect(second.body.command_id).toBe(first.body.command_id);
  });

  it("asks for the effective date before recording Changed, and confirms Wrong", async () => {
    await render();
    const row = [...host.querySelectorAll<HTMLElement>(".context-row")].find(r => r.textContent?.includes("annual plans"))!;
    await click(button("Changed", row));
    const save = button("Save change", row);
    type(row.querySelector("textarea")!, "Pricing moves to monthly plans.");
    expect(save.disabled).toBe(true);
    type(row.querySelector<HTMLInputElement>("input[type=date]")!, "2026-10-01");
    expect(save.disabled).toBe(false);
    await click(save);
    expect(posts("changed")[0].body).toMatchObject({ assertion_id: "a-dec", expected_revision: 2, statement: "Pricing moves to monthly plans.",
      effective_at: new Date("2026-10-01T00:00:00").toISOString() });
    await click(button("Wrong", row));
    await click(button("Mark wrong", row));
    expect(posts("wrong")[0].body).toMatchObject({ assertion_id: "a-dec", expected_revision: 2 });
  });

  it("drills into evidence and reveals the cited transcript passage", async () => {
    await render();
    const row = [...host.querySelectorAll<HTMLElement>(".context-row")].find(r => r.textContent?.includes("annual plans"))!;
    await click(button("Evidence", row));
    expect(row.querySelector(".context-evidence")?.textContent).toContain("We move pricing to annual plans.");
    await click(button("In transcript", row));
    expect(host.querySelector(".transcript-segment.is-focused")?.textContent).toContain("annual plans");
  });

  it("confirms, rejects and undoes speaker matches; people are navigable", async () => {
    const onPerson = vi.fn();
    await render({ onPerson });
    const speakers = host.querySelector<HTMLElement>("[aria-label=Speakers]")!;
    await click(button("Not them", speakers));
    expect(posts("speakers")[0].body).toMatchObject({ link_id: "l-1", expected_revision: 3, action: "reject" });
    await click(button("Undo", speakers));
    expect(posts("speakers")[1].body).toMatchObject({ link_id: "l-2", action: "undo", decision_id: "d-9" });
    await click(button("Sam", speakers));
    expect(onPerson).toHaveBeenCalledWith("act-sam");
  });

  it("defers the review, and Done asks before dropping selected work", async () => {
    const onBack = vi.fn();
    await render({ onBack });
    await click(button("Remind me later"));
    expect(posts("later")[0].body.items).toEqual([]);
    expect(onBack).toHaveBeenCalledTimes(1);
    await click(button("Done"));
    expect(posts("done")).toHaveLength(0);
    expect(text()).toContain("1 selected item has not been added");
    await click(button("Finish anyway"));
    expect(posts("done")[0].body).toMatchObject({ review_revision: 4, items: [
      { proposal_id: "p-own", revision: 1 }, { proposal_id: "p-other", revision: 1 }, { proposal_id: "p-upd", revision: 1 }] });
    expect(onBack).toHaveBeenCalledTimes(2);
  });

  it("reloads after a burst of context changes, but asks first when the reader has unsaved selections", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    try {
      await render();
      const reads = () => calls.filter(c => c.method === "GET" && c.url === R).length;
      const before = reads();
      current = review({ revision: 5, proposals: [proposal({ revision: 2, title: "Send the final deck" }), ...review().proposals.slice(1)] });
      for (let i = 0; i < 3; i++) window.dispatchEvent(new Event("eri-context-changed"));
      await act(async () => { vi.advanceTimersByTime(1000); }); await settle();
      expect(reads()).toBe(before + 1);
      expect(box("Send the final deck").checked).toBe(true);
      await click(box("Kim drafts the FAQ"));
      window.dispatchEvent(new Event("eri-context-changed"));
      await act(async () => { vi.advanceTimersByTime(1000); }); await settle();
      expect(reads()).toBe(before + 1);
      expect(host.querySelector(".meeting-updated")?.textContent).toContain("new changes");
      expect(box("Kim drafts the FAQ").checked).toBe(true);
      await click(button("Refresh", host.querySelector(".meeting-updated")!));
      expect(reads()).toBe(before + 2);
      expect(host.querySelector(".meeting-updated")).toBeNull();
      expect(box("Kim drafts the FAQ").checked).toBe(true);
    } finally { vi.useRealTimers(); }
  });

  it("lists work others owe under Waiting on others, unchecked even when the server preselects it", async () => {
    current = review({ proposals: [proposal(), proposal({ id: "p-wait", title: "Kim sends the 50 calls", assignee: { id: "act-kim", name: "Kim" },
      preselected: true, reason: "Waiting on Kim to send the 50 calls." })] });
    await render();
    const waiting = host.querySelector<HTMLElement>("[role=group][aria-label='Waiting on others']")!;
    expect(waiting.textContent).toContain("Kim sends the 50 calls");
    expect(host.querySelector<HTMLElement>("[role=group][aria-label='Work to add']")!.textContent).not.toContain("Kim sends");
    expect(box("Kim sends the 50 calls").checked).toBe(false);
    expect(box("Send the pricing deck").checked).toBe(true);
    await click(box("Kim sends the 50 calls"));
    await click(button("Add selected tasks (2)"));
    expect(posts("approve")[0].body.items).toEqual([{ proposal_id: "p-own", revision: 1 }, { proposal_id: "p-wait", revision: 1 }]);
  });

  it("reprocesses after a confirm, shows the pending state and checks back until processing settles", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    try {
      await render();
      await click(button("Reprocess"));
      expect(posts("reprocess")).toHaveLength(0);
      expect(host.querySelector("[role=alertdialog]")?.textContent).toContain("Work you added or dismissed stays");
      await click(button("Cancel"));
      expect(host.querySelector("[role=alertdialog]")).toBeNull();
      await click(button("Reprocess"));
      queued[`POST ${R}/reprocess`] = [[200, { reprocess: { job_id: "j-1", replayed: false }, review: review({ busy: true }) }]];
      await click(button("Reprocess now"));
      const [sent] = posts("reprocess");
      expect(Object.keys(sent.body)).toEqual(["command_id"]);
      expect(button("Reprocessing").disabled).toBe(true);
      expect(text()).toContain("Reprocessing. Updated proposals appear here");
      const reads = () => calls.filter(c => c.method === "GET" && c.url === R).length, before = reads();
      current = review({ busy: true });
      await act(async () => { vi.advanceTimersByTime(5000); }); await settle();
      expect(reads()).toBe(before + 1);
      current = review({ busy: false, proposals: [proposal({ revision: 2, title: "Send the final deck" })] });
      await act(async () => { vi.advanceTimersByTime(5000); }); await settle();
      expect(reads()).toBe(before + 2);
      expect(button("Reprocess").disabled).toBe(false);
      expect(box("Send the final deck").checked).toBe(true);
      await act(async () => { vi.advanceTimersByTime(20000); }); await settle();
      expect(reads()).toBe(before + 2);  // settled: no more polling
    } finally { vi.useRealTimers(); }
  });

  it("explains when the meeting is still processing", async () => {
    await render();
    queued[`POST ${R}/reprocess`] = [[409, { error: { code: "CONTEXT_BUSY", message: "This meeting is still being processed. Try again when it finishes." } }]];
    await click(button("Reprocess"));
    await click(button("Reprocess now"));
    expect(host.querySelector("[role=alert]")?.textContent).toContain("still being processed");
    expect(host.querySelector(".meeting-stale")).toBeNull();
  });

  it("retries failed, conflicting and unconfirmed publications, not finished ones", async () => {
    const sent = (id: string, status: "failed" | "conflict" | "unconfirmed" | "published", error: string | null = null) =>
      proposal({ id: "p-" + id, title: "Task " + id, disposition: "approved", publication: { id: "pub-" + id, status, task_id: null, error } });
    current = review({ proposals: [sent("f", "failed", "Linear timed out."), sent("c", "conflict"), sent("u", "unconfirmed"), sent("ok", "published")] });
    await render();
    const retries = [...host.querySelectorAll<HTMLButtonElement>("button[aria-label^='Retry:']")].map(b => b.getAttribute("aria-label"));
    expect(retries).toEqual(["Retry: Task f", "Retry: Task c", "Retry: Task u"]);
    expect(text()).toContain("Linear timed out.");
    queued["POST /api/v1/context/reviews/publications/pub-c/retry"] = [[409, { error: { code: "PUBLICATION_CONFLICT", message: "The task changed since you approved this. Review it there." } }]];
    await click(host.querySelector<HTMLButtonElement>("button[aria-label='Retry: Task c']")!);
    expect(host.querySelector("[role=alert]")?.textContent).toContain("The task changed since you approved this");
    expect(host.querySelector(".meeting-stale")).toBeNull();
    const path = "/api/v1/context/reviews/publications/pub-f/retry";
    queued[`POST ${path}`] = [[200, { publication: { id: "pub-f", status: "retrying" }, review: review({ proposals: [
      proposal({ id: "p-f", title: "Task f", disposition: "approved", publication: { id: "pub-f", status: "retrying", task_id: null } })] }) }]];
    await click(host.querySelector<HTMLButtonElement>("button[aria-label='Retry: Task f']")!);
    const [call] = calls.filter(c => c.method === "POST" && c.url === path);
    expect(Object.keys(call.body)).toEqual(["command_id"]);
    expect(text()).toContain("Retrying");
    expect(host.querySelector("button[aria-label^='Retry:']")).toBeNull();
  });
});
