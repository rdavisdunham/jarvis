// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { setDurableScope } from "./api";
import { RunDrawer } from "./Remotea";
import { timeline, type RunDetail } from "./remotea-model";
import { RunLineage, RunRestart } from "./RunRestart";
import { NOTE_REQUIRED, RESTART_TITLES, restartBody, restartLinks, restartOffer, restartProblem } from "./run-restart";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const OLD = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", NEW = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb";
const detail = (values: Partial<RunDetail> = {}): RunDetail => ({
  id: OLD, task: { id: "t-1", title: "Task", status: "completed" }, state: "completed", state_revision: 9,
  state_reason: "approved", reason_text: "", state_since: "2026-11-10T10:00:00Z", generation: 1, plan_revision: 1, tool: "claude",
  model: "", branch: "eri/task-1", base_branch: "main", worktree: "/wt/task-1", effective_permission_mode: "",
  turn_count: 3, turn_cap: 15, started_at: null, finished_at: "2026-11-10T10:00:00Z", stop_requested_at: null, stop_mode: "",
  surf: "surf-1", host: { id: "h", name: "desk", connected: true }, structured_answers: true, session: null,
  created_at: "2026-11-10T09:00:00Z", history: [], turns: [], commands: [], questions: [], plan: null,
  restart: { available: true, why: null }, restarted_from: null, restarted_as: null,
  ...values,
});

describe("the Restart run offer", () => {
  it("is shown only on a finished run the server offers it for", () => {
    expect(restartOffer(detail())).toEqual({ enabled: true, why: null });
    expect(restartOffer(detail({ state: "executing", restart: { available: false, why: null } }))).toBeNull();
    expect(restartOffer(detail({ restart: undefined }))).toBeNull();  // Older Eridani.
    expect(restartOffer(detail({ restart: { available: false, why: "The remotea server is offline." } })))
      .toEqual({ enabled: false, why: "The remotea server is offline." });
  });

  it("hides while the run that continues this one is still working, and comes back once it finished", () => {
    expect(restartOffer(detail({ restarted_as: { id: NEW, state: "planning" }, restart: { available: false, why: "busy" } }))).toBeNull();
    expect(restartOffer(detail({ restarted_as: { id: NEW, state: "stopped" } }))).toEqual({ enabled: true, why: null });
  });

  it("needs a note to plan, not to pilot", () => {
    expect(restartProblem("plan", "  ")).toBe(NOTE_REQUIRED);
    expect(restartProblem("plan", "Add a toggle")).toBeNull();
    expect(restartProblem("pilot", "")).toBeNull();
    expect(restartProblem("pilot", "x".repeat(4001))).toMatch(/4000/);
    expect(restartBody(detail(), "plan", "  Add a toggle \n")).toEqual({ mode: "plan", note: "Add a toggle", expected_revision: 9 });
  });

  it("says what each mode does in the button titles", () => {
    expect(RESTART_TITLES.plan).toMatch(/plans.*approval/);
    expect(RESTART_TITLES.pilot).toMatch(/freestyle.*no judge/i);
    expect(RESTART_TITLES.button).toMatch(/Plan it.*Pilot it myself/);
  });

  it("links the two runs both ways, with whether the conversation was resumed", () => {
    const links = restartLinks({ restarted_from: { id: OLD, state: "completed", finished_at: null, mode: "pilot", resumed: true },
      restarted_as: { id: NEW, state: "executing" } });
    expect(links.from).toMatchObject({ id: OLD, label: "Restarted from" });
    expect(links.from!.text).toContain("Completed");
    expect(links.from!.text).toContain("piloted by you · resumed its conversation");
    expect(links.as).toEqual({ id: NEW, label: "Restarted as", text: "a new run (Working)" });
    expect(restartLinks({ restarted_from: { id: OLD, state: "failed", finished_at: null, mode: "plan", resumed: false } }).from!.text)
      .toContain("planned · fresh conversation");
    expect(restartLinks({})).toEqual({ from: null, as: null });
  });

  it("puts the restarted notes on the timeline", () => {
    const entries = timeline(detail({ history: [
      { at: "2026-11-10T11:00:00Z", event: "restarted", mode: "plan", restarted_from: OLD, text: "Add a toggle", resumed: true, actor: { kind: "human" } },
      { at: "2026-11-10T11:00:01Z", event: "restarted", mode: "pilot", restarted_as: NEW, actor: { kind: "human" } },
    ] }));
    expect(entries.map(e => e.title)).toEqual(["Restarted from an earlier run by you · Plan it", "Restarted as a new run by you · Pilot it myself"]);
    expect(entries[0].detail).toBe("Add a toggle · Resumed the earlier conversation");
  });

  it("appears in a finished run's drawer and not in a working one", () => {
    expect(renderToStaticMarkup(<RunDrawer id={OLD} preloaded={detail()} onClose={() => {}}/>)).toContain("Restart run");
    expect(renderToStaticMarkup(<RunDrawer id={OLD} preloaded={detail({ state: "executing", finished_at: null })} onClose={() => {}}/>))
      .not.toContain("Restart run");
    const linked = renderToStaticMarkup(<RunDrawer id={NEW} preloaded={detail({ id: NEW, state: "planning", finished_at: null,
      restarted_from: { id: OLD, state: "completed", finished_at: null, mode: "plan", resumed: null } })} onClose={() => {}}/>);
    expect(linked).toContain("Restarted from");
    expect(linked).toContain(`?view=remotea&amp;run=${OLD}`);
  });
});

describe("the Restart run form", () => {
  let host: HTMLDivElement, root: Root;
  beforeEach(() => { host = document.createElement("div"); document.body.append(host); root = createRoot(host); localStorage.clear(); setDurableScope("test-owner:personal"); });
  afterEach(() => { act(() => root.unmount()); host.remove(); vi.unstubAllGlobals(); });

  function stubFetch(answer: (url: string, body: Record<string, unknown>) => { status?: number; body: unknown }) {
    const calls: { url: string; body: Record<string, unknown> }[] = [];
    vi.stubGlobal("fetch", async (url: string, init: RequestInit = {}) => {
      const body = init.body ? JSON.parse(String(init.body)) : {};
      calls.push({ url, body });
      const out = answer(url, body);
      return new Response(JSON.stringify(out.body), { status: out.status ?? 200 });
    });
    return calls;
  }
  const settle = () => act(async () => { await new Promise(r => setTimeout(r, 0)); });
  const byText = (text: string) => [...host.querySelectorAll<HTMLButtonElement>("button")].find(b => b.textContent?.trim() === text)!;
  function type(value: string) {
    const el = host.querySelector("textarea")!;
    act(() => {
      Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")!.set!.call(el, value);
      el.dispatchEvent(new Event("input", { bubbles: true }));
    });
  }

  it("plans the owner's note and switches to the new run", async () => {
    const calls = stubFetch(() => ({ body: { id: NEW } }));
    const opened: string[] = [];
    act(() => root.render(<RunRestart run={detail()} disabled={false} onRestarted={id => opened.push(id)}/>));
    expect(byText("Restart run").title).toBe(RESTART_TITLES.button);
    act(() => byText("Restart run").click());
    expect(host.textContent).toContain("What do you want to add or change?");
    expect(byText("Plan it").disabled).toBe(true);
    expect(byText("Plan it").title).toBe(NOTE_REQUIRED);
    expect(byText("Pilot it myself").disabled).toBe(false);
    type("Add a dark mode toggle");
    expect(byText("Plan it").disabled).toBe(false);
    expect(byText("Plan it").title).toBe(RESTART_TITLES.plan);
    act(() => byText("Plan it").click());
    await settle();
    expect(calls).toHaveLength(1);
    expect(calls[0].url).toBe(`/api/v1/remotea/runs/${OLD}/restart`);
    expect(calls[0].body).toMatchObject({ mode: "plan", note: "Add a dark mode toggle", expected_revision: 9, action_id: expect.any(String) });
    expect(opened).toEqual([NEW]);
  });

  it("pilots without a note and shows a refusal inline", async () => {
    const calls = stubFetch(() => ({ status: 409, body: { error: { code: "RUN_ACTIVE", message: "An agent run is already working on this task.", data: {} } } }));
    const opened: string[] = [];
    act(() => root.render(<RunRestart run={detail()} disabled={false} onRestarted={id => opened.push(id)}/>));
    act(() => byText("Restart run").click());
    act(() => byText("Pilot it myself").click());
    await settle();
    expect(calls[0].body).toMatchObject({ mode: "pilot", note: "" });
    expect(host.querySelector('[role="alert"]')!.textContent).toBe("An agent run is already working on this task.");
    expect(opened).toEqual([]);
    act(() => byText("Cancel").click());
    expect(byText("Restart run")).toBeTruthy();
  });

  it("is off with the server's reason when the run cannot be restarted", () => {
    act(() => root.render(<RunRestart run={detail({ restart: { available: false, why: "The remotea server is offline." } })} disabled={false} onRestarted={() => {}}/>));
    expect(byText("Restart run").disabled).toBe(true);
    expect(host.textContent).toContain("The remotea server is offline.");
  });

  it("opens the linked run from Restarted from", () => {
    const opened: string[] = [];
    act(() => root.render(<RunLineage run={{ restarted_from: { id: OLD, state: "completed", finished_at: null } }} onOpen={id => opened.push(id)}/>));
    act(() => host.querySelector("a")!.click());
    expect(opened).toEqual([OLD]);
  });
});
