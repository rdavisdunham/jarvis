import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { RemoteaDataContext, type RemoteaData } from "./remotea-context";
import { RunButton } from "./RunButton";
import { RunPanel } from "./RunPanel";
import { PendingControlCard } from "./PendingControls";
import {
  composerAction, HARNESSES, harnessAvailable, preflightLine, railDot, restorePanel, runChip, runsBadge, runsByTask, startBody, storedPanel,
  type Host, type PendingControl, type Run,
} from "./remotea-model";

const run = (over: Partial<Run>): Run => ({
  id: "11111111-1111-4111-8111-111111111111", task: { id: "t1", title: "Ship it", status: "open" }, state: "executing", state_revision: 1,
  state_reason: "turn", reason_text: "", state_since: "2026-10-29T10:00:00Z", generation: 1, plan_revision: 1, tool: "claude", model: "opus",
  branch: "task/t1", base_branch: "main", worktree: "", effective_permission_mode: "", turn_count: 2, turn_cap: 15, started_at: null,
  finished_at: null, stop_requested_at: null, stop_mode: "", surf: "s1", host: { id: "h", name: "desk", connected: true }, structured_answers: true,
  session: null, created_at: "2026-10-29T09:00:00Z", ...over,
});
const host = (over: Partial<Host> = {}): Host => ({ id: "h", name: "desk", machine: "wsl", connected: true, last_seen_at: null, conn_epoch: 1,
  caps: ["spawn"], harnesses: { claude: { spawn: true } }, proto: 2, ...over });
const data = (over: Partial<RemoteaData>): RemoteaData => ({ enabled: true, status: { enabled: true, key: null, hosts: [host()], link_url: "", connect_command: "" },
  activeRuns: [], runsByTask: new Map(), pending: [], cards: {}, error: "", panel: { run: null, collapsed: false }, start: null, ...over });
const render = (ui: React.ReactElement, value: RemoteaData) => renderToStaticMarkup(<RemoteaDataContext.Provider value={value}>{ui}</RemoteaDataContext.Provider>);

describe("runChip", () => {
  it("offers Run when the task has no run and a server is connected", () => {
    expect(runChip(undefined, [host()])).toEqual({ kind: "start", label: "Run", disabled: false, reason: "" });
  });
  it("is disabled with a reason when no server can take it", () => {
    expect(runChip(undefined, [])).toMatchObject({ kind: "start", disabled: true });
    expect(runChip(undefined, [host({ connected: false })]).kind === "start" && runChip(undefined, [host({ connected: false })])).toMatchObject({ reason: "The remotea server is offline." });
  });
  it("names what an existing run needs", () => {
    expect(runChip(run({ state: "awaiting_approval" }), [])).toMatchObject({ kind: "open", label: "Approve", tone: "attention" });
    expect(runChip(run({ state: "needs_you" }), [])).toMatchObject({ label: "Needs you", tone: "attention" });
    expect(runChip(run({ state: "spawning" }), [])).toMatchObject({ label: "Starting", tone: "working" });
    expect(runChip(run({ state: "executing" }), [])).toMatchObject({ label: "Running", tone: "working" });
    expect(runChip(run({ state: "completed" }), [host()])).toMatchObject({ kind: "start" });
  });
});

describe("panel helpers", () => {
  it("restores the run from ?run= and the collapsed rail from storage", () => {
    expect(restorePanel("?view=today&run=11111111-1111-4111-8111-111111111111", "rail")).toEqual({ run: "11111111-1111-4111-8111-111111111111", collapsed: true });
    expect(restorePanel("?run=nope", null)).toEqual({ run: null, collapsed: false });
    expect(storedPanel({ run: "x", collapsed: true })).toBe("rail");
    expect(storedPanel({ run: "x", collapsed: false })).toBeNull();
  });
  it("rail dot follows the run's state", () => {
    expect(railDot(undefined)).toEqual({ tone: "unknown", label: "Run" });
    expect(railDot(run({ state: "needs_you" }))).toEqual({ tone: "attention", label: "Needs you" });
    expect(railDot(run({ state: "executing" }))).toEqual({ tone: "working", label: "Working" });
    expect(railDot(run({ state: "completed" }))).toEqual({ tone: "settled", label: "Completed" });
  });
  it("maps the most urgent active run to each task", () => {
    const a = run({ id: "a", state: "executing" }), b = run({ id: "b", state: "needs_you", state_since: "2026-10-29T09:00:00Z" });
    expect(runsByTask([a, b]).get("t1")?.id).toBe("b");
  });
  it("counts the badge and flags attention", () => {
    const pending: PendingControl[] = [{ id: "p", tool: "control.start", summary: "Start", run_id: null, expires_at: "" }];
    expect(runsBadge([run({})], [])).toEqual({ count: 1, attention: false, aria: "Runs, 1 active" });
    expect(runsBadge([run({ state: "awaiting_approval" })], pending)).toMatchObject({ count: 2, attention: true });
    expect(runsBadge([run({ state: "completed" })], [])).toEqual({ count: 0, attention: false, aria: "Runs" });
  });
});

describe("Run dialog helpers", () => {
  it("builds the start body with defaults left out", () => {
    expect(startBody("t1", { harness: "claude", model: "opus", interview: "auto", target: null })).toEqual({ task_id: "t1", harness: "claude" });
    expect(startBody("t1", { harness: "claude", model: "sonnet", interview: "skip", target: "x" }))
      .toEqual({ task_id: "t1", harness: "claude", model: "sonnet", execution_target_id: "x", interview: false });
    expect(startBody("t1", { harness: "codex", model: HARNESSES[1].models[0].id, interview: "auto", target: null })).toEqual({ task_id: "t1", harness: "codex" });
  });
  it("reads harness availability from connected hosts", () => {
    expect(harnessAvailable([host()], "claude")).toBe(true);
    expect(harnessAvailable([host()], "codex")).toBe(false);
    expect(harnessAvailable([host({ connected: false })], "claude")).toBe(false);
  });
  it("describes the preflight", () => {
    const base = { needs_target: false, need: null, problems: [], active_run: null, interview: true, harnesses: {} };
    expect(preflightLine(null)).toMatch(/chosen when the run starts/);
    expect(preflightLine({ ...base, target: { id: "x", name: "jarvis", repo: "/home/d/jarvis" } })).toBe("Runs in jarvis · /home/d/jarvis.");
    expect(preflightLine({ ...base, target: null, needs_target: true })).toMatch(/Choose where/);
    expect(preflightLine({ ...base, target: null, active_run: { id: "r", state: "executing" } })).toMatch(/already working/);
    expect(preflightLine({ ...base, target: null, problems: [{ code: "HOST_OFFLINE", message: "Offline." }] })).toBe("Offline.");
  });
});

describe("composerAction", () => {
  const key = (over: Partial<Parameters<typeof composerAction>[0]>) => ({ key: "Enter", shiftKey: false, ctrlKey: false, metaKey: false, ...over });
  it("Enter queues, Ctrl/Cmd+Enter steers, Shift+Enter is a new line", () => {
    expect(composerAction(key({}))).toBe("queue");
    expect(composerAction(key({ ctrlKey: true }))).toBe("steer");
    expect(composerAction(key({ metaKey: true }))).toBe("steer");
    expect(composerAction(key({ shiftKey: true }))).toBeNull();
    expect(composerAction(key({ key: "a" }))).toBeNull();
  });
});

describe("RunButton", () => {
  it("renders nothing while remotea is off", () => {
    expect(render(<RunButton taskId="t1" title="Ship it"/>, data({ enabled: false }))).toBe("");
  });
  it("offers Run, or the run's state when one exists", () => {
    expect(render(<RunButton taskId="t1" title="Ship it"/>, data({}))).toContain('aria-label="Run Ship it with an agent"');
    const html = render(<RunButton taskId="t1" title="Ship it"/>, data({ runsByTask: new Map([["t1", run({ state: "awaiting_approval" })]]) }));
    expect(html).toContain("run-chip tone-attention");
    expect(html).toContain("Approve");
  });
  it("is disabled with the reason when the server is offline", () => {
    const html = render(<RunButton taskId="t1" title="Ship it"/>, data({ status: { enabled: true, key: null, hosts: [host({ connected: false })], link_url: "", connect_command: "" } }));
    expect(html).toContain("disabled");
    expect(html).toContain("offline");
  });
});

describe("RunPanel", () => {
  it("renders nothing without an open run", () => {
    expect(render(<RunPanel phone={false}/>, data({}))).toBe("");
  });
  it("collapsed: a rail with the run's state", () => {
    const r = run({ state: "needs_you" });
    const html = render(<RunPanel phone={false}/>, data({ activeRuns: [r], panel: { run: r.id, collapsed: true } }));
    expect(html).toContain('data-state="rail"');
    expect(html).toContain("tone-attention");
    expect(html).toContain("Expand run panel, Needs you");
    expect(html).toContain("hidden");  // the body stays mounted on wide screens
  });
  it("open on a phone: a sheet dialog", () => {
    const r = run({});
    const html = render(<RunPanel phone={true}/>, data({ activeRuns: [r], panel: { run: r.id, collapsed: false } }));
    expect(html).toContain('role="dialog"');
    expect(html).toContain('data-state="open"');
  });
});

describe("PendingControlCard", () => {
  const item: PendingControl = { id: "p1", tool: "control.start", summary: "Start Claude Code on “Ship it”", run_id: null, expires_at: "", task_id: "t1" };
  it("shows Confirm and Decline", () => {
    const html = renderToStaticMarkup(<PendingControlCard item={item} onDecide={() => {}} onCancelPicker={() => {}}/>);
    expect(html).toContain("Confirm");
    expect(html).toContain("Decline");
    expect(html).toContain("Start Claude Code on");
  });
  it("shows the refusal and a problem", () => {
    const html = renderToStaticMarkup(<PendingControlCard item={{ ...item, problems: [{ code: "HOST_OFFLINE", message: "Offline." }] }}
      card={{ busy: false, error: "Nope.", need: undefined }} onDecide={() => {}} onCancelPicker={() => {}}/>);
    expect(html).toContain("Offline.");
    expect(html).toContain("Nope.");
  });
});
