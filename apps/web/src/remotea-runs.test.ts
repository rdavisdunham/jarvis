import { describe, expect, it } from "vitest";
import {
  buildAnswers, defaultScope, harnessChoice, linkedRun, needsTargetData, orderRuns, questionMode, runButton, runControls, targetOptions, timeline,
  unmergedSummary, verifyLine, type Host, type NeedsTarget, type Preflight, type Run, type RunDetail, type RunQuestion,
} from "./remotea-model";

const host = (connected: boolean): Host => ({
  id: "h", name: "desk", machine: "WSL", connected, last_seen_at: null, conn_epoch: 1, caps: [], harnesses: {}, proto: 1,
});
const run = (values: Partial<Run> = {}): Run => ({
  id: "r1", task: { id: "t1", title: "Task", status: "in_progress" }, state: "executing", state_revision: 2, state_reason: "",
  reason_text: "", state_since: "2026-10-28T10:00:00Z", generation: 0, plan_revision: 1, tool: "claude", model: "",
  branch: "davin/eri-1-task", base_branch: "main", worktree: "", effective_permission_mode: "", turn_count: 1, turn_cap: 40,
  started_at: null, finished_at: null, stop_requested_at: null, stop_mode: "", surf: "surf-1",
  host: { id: "h", name: "desk", connected: true }, structured_answers: true, session: null, created_at: "2026-10-28T09:59:00Z",
  ...values,
});
const question = (values: Partial<RunQuestion> = {}): RunQuestion => ({
  id: "q1", question_id: "pq", run_id: "r1", generation: 0, status: "open", answers: null, answered_via: "",
  remote_answerable: true, hold_until: null, detail: "", created_at: "2026-10-28T10:01:00Z", resolved_at: null,
  items: [{ id: "a", text: "Which db?", choices: [{ id: "pg", label: "Postgres" }, { id: "sq", label: "SQLite" }] },
    { id: "b", text: "Anything else?" }],
  ...values,
});

describe("run drawer model", () => {
  it("hides the Run button unless remotea is on, and explains a disabled one", () => {
    expect(runButton(false, [host(true)], null).visible).toBe(false);
    expect(runButton(true, [], null)).toMatchObject({ visible: true, disabled: true });
    expect(runButton(true, [host(false)], null).reason).toMatch(/offline/);
    expect(runButton(true, [host(true)], null)).toMatchObject({ disabled: false, label: "Run" });
    expect(runButton(true, [], run())).toMatchObject({ disabled: false, label: "Open run", run: "r1" });
  });

  it("offers Message, Steer, Stop and Approve by state, and shows stopping", () => {
    const working = runControls(run());
    expect([working.message.enabled, working.steer.enabled, working.stop.enabled, working.approve.enabled]).toEqual([true, true, true, true]);
    expect(working.stop).toMatchObject({ label: "Stop", mode: "interrupt" });
    const stopping = runControls(run({ state: "stopping", stop_mode: "interrupt" }));
    expect(stopping.message.enabled).toBe(false);
    expect(stopping.stop).toMatchObject({ enabled: true, label: "Force stop", mode: "kill" });
    expect(runControls(run({ state: "stopping", stop_mode: "kill" })).stop.label).toBe("Stopping…");
    const done = runControls(run({ state: "completed" }));
    expect([done.message.enabled, done.stop.enabled, done.approve.enabled]).toEqual([false, false, false]);
    expect(runControls(run({ state: "spawning", surf: "" })).message.reason).toMatch(/not started/);
    expect(runControls(run({ host: { id: "h", name: "desk", connected: false } })).offline).toBe(true);
  });

  it("answers here only when the host advertised structured answers", () => {
    expect(questionMode(question(), run())).toBe("answer");
    expect(questionMode(question(), run({ structured_answers: false }))).toBe("terminal");
    expect(questionMode(question({ remote_answerable: false }), run())).toBe("terminal");
    expect(questionMode(question({ status: "timed_out" }), run())).toBe("terminal");
    expect(questionMode(question({ status: "submitted" }), run())).toBe("pending");
    expect(questionMode(question({ status: "accepted" }), run())).toBe("closed");
    expect(questionMode(question({ generation: 0 }), run({ generation: 1 }))).toBe("closed");
  });

  it("builds answers only once every item has a choice or text", () => {
    const q = question();
    expect(buildAnswers(q, { a: ["pg"] }, {})).toBeNull();
    expect(buildAnswers(q, { a: ["pg", "bogus"] }, { b: "  no  " })).toEqual([
      { item_id: "a", choice_ids: ["pg"] }, { item_id: "b", text: "no" }]);
  });

  it("merges turns, commands with outcomes, questions and states into one timeline", () => {
    const detail: RunDetail = {
      ...run(), plan: null,
      history: [{ at: "2026-10-28T09:59:00Z", from: "queued", to: "spawning", actor: { kind: "human" } },
        { at: "2026-10-28T10:00:30Z", from: "spawning", to: "executing" }],
      turns: [{ id: "t", seq: 1, text: "Done step one", clipped: false, truncated: true, generation: 0, at: "2026-10-28T10:02:00Z", prompt: "" }],
      commands: [{ id: "c", op_id: "o", kind: "spawn", status: "acked", generation: 0, mode: null, text: null,
        outcome: { state: "executed" }, created_at: "2026-10-28T09:59:30Z", updated_at: "2026-10-28T10:00:00Z" },
        { id: "c2", op_id: "o2", kind: "input", status: "rejected", generation: 0, mode: "steer", text: "Use pg",
          outcome: null, created_at: "2026-10-28T10:03:00Z", updated_at: "2026-10-28T10:03:00Z" }],
      questions: [question()],
    };
    const entries = timeline(detail);
    expect(entries.map(e => e.title)).toEqual([
      "Starting", "Start · done", "Working", "Question · waiting for you", "Turn 1", "Steer · refused by remotea"]);
    expect(entries[0].detail).toBe("by you");
    expect(entries[4].detail).toMatch(/\[truncated\]$/);
    expect(entries[5]).toMatchObject({ tone: "warn", detail: "Use pg" });
  });

  it("puts runs needing you first and reads the run from a push deep link", () => {
    const ordered = orderRuns([run({ id: "done", state: "completed" }), run({ id: "work" }), run({ id: "ask", state: "waiting_answer" })]);
    expect(ordered.map(r => r.id)).toEqual(["ask", "work", "done"]);
    expect(linkedRun("?view=remotea&run=0b9c1d2e-3f40-4a5b-8c6d-7e8f90a1b2c3")).toBe("0b9c1d2e-3f40-4a5b-8c6d-7e8f90a1b2c3");
    expect(linkedRun("?view=remotea&run=../x")).toBeNull();
  });

  it("lists saved then advertised targets and summarizes unmerged work", () => {
    const options = targetOptions({ task_id: "t1", records: [],
      candidates: [{ id: "s", host_id: "h", host_target: "jarvis", name: "Jarvis", repo: "/r/jarvis", worktree_root: "", record_id: null }],
      advertised: [{ host_id: "h", host_name: "desk", host_target: "jarvis", repo: "/r/jarvis", worktree_root: "" },
        { host_id: "h", host_name: "desk", host_target: "remotea", repo: "/r/remotea", worktree_root: "" }] });
    expect(options.map(o => o.key)).toEqual(["saved:s", "host:h:remotea"]);
    expect(options[0]).toMatchObject({ label: "Jarvis", detail: "/r/jarvis" });
    expect(options[1]).toMatchObject({ label: "remotea", detail: "/r/remotea" });
    expect(unmergedSummary({ host_id: "h", host_name: "desk", target: "jarvis", path: "/w", branch: "b", base: "main", ahead: 2,
      behind: 0, unpushed: 1, uncommitted: 0, untracked: 0, merged: false, active: false, reasons: null })).toBe("2 ahead of main · 1 unpushed");
  });
});

describe("execution target picker", () => {
  const need = (records: NeedsTarget["records"]): NeedsTarget => ({ task_id: "t1", candidates: [], advertised: [], records });

  it("remembers the choice for the nearest record that is not a task, so sibling tasks share it", () => {
    expect(defaultScope(need([{ id: "task-rec", title: "Fix login", type_id: "task" },
      { id: "proj", title: "Eridani roadmap", type_id: "project" }, { id: "client", title: "Acme", type_id: "client" }]))).toBe("proj");
    // A sub-task skips its parent task too.
    expect(defaultScope(need([{ id: "sub", title: "Sub", type_id: "task" }, { id: "parent", title: "Parent", type_id: "task" },
      { id: "proj", title: "Roadmap", type_id: "project" }]))).toBe("proj");
  });

  it("falls back to the task's own record, then to only this task", () => {
    expect(defaultScope(need([{ id: "task-rec", title: "Loose task", type_id: "task" }]))).toBe("task-rec");
    expect(defaultScope(need([{ id: "old", title: "Older server" }]))).toBe("old");
    expect(defaultScope(need([]))).toBe("task");
  });

  it("opens the picker only for a NEEDS_TARGET that carries choices", () => {
    expect(needsTargetData({ task_id: "t1", candidates: [], advertised: [], records: [] })).not.toBeNull();
    expect(needsTargetData({ task_id: "t1" })).toBeNull();  // A target without a server: shown as an error instead.
    expect(needsTargetData(null)).toBeNull();
  });
});

describe("harnessChoice", () => {
  const spawning = (): Host => ({ ...host(true), harnesses: { claude: { spawn: true }, codex: { spawn: true } } });
  const pre = (values: Partial<Preflight> = {}): Preflight => ({
    target: { id: "t", name: "jarvis", repo: "/r/jarvis" }, needs_target: false, need: null, problems: [], active_run: null,
    interview: true, harnesses: { claude: true, codex: false },
    harness_reasons: { codex: "The jarvis target on desk does not enable Codex." }, harness_note: null, ...values,
  });

  it("follows the preflight for the task's target, even when a connected host spawns the harness", () => {
    expect(harnessChoice(pre(), [spawning()], "claude")).toEqual({ ok: true });
    expect(harnessChoice(pre(), [spawning()], "codex")).toEqual({ ok: false, reason: "The jarvis target on desk does not enable Codex." });
    // An older Eridani that sends no reasons still disables it, with a generic reason.
    expect(harnessChoice(pre({ harness_reasons: undefined }), [spawning()], "codex")).toMatchObject({ ok: false, reason: /cannot start Codex/ });
  });

  it("falls back to any connected host only when there is no preflight", () => {
    expect(harnessChoice(null, [spawning()], "codex")).toEqual({ ok: true });
    expect(harnessChoice(undefined, [host(true)], "codex")).toMatchObject({ ok: false, reason: /cannot start Codex/ });
    expect(harnessChoice(pre({ harnesses: {} }), [spawning()], "codex")).toEqual({ ok: true });
  });
});

describe("verifyLine", () => {
  const plan = { revision: 1, title: "t", prd_text: "", steps: [], criteria: [{ id: "c1", text: "tests pass" }], verify: null };
  it("names the command that runs after each turn", () => {
    expect(verifyLine({ plan: { ...plan, verify: { target_id: "x", verify_id: "tests" } }, verify_options: ["tests"] }))
      .toEqual({ current: "tests", options: ["tests"], text: "Verified by tests after each turn", canSet: false });
  });
  it("flags a plan with criteria but no verify, offering the target's ids", () => {
    const line = verifyLine({ plan, verify_options: ["lint", "tests"] });
    expect(line.current).toBeNull();
    expect(line.text).toMatch(/No verify command/);
    expect(line.canSet).toBe(true);
    expect(verifyLine({ plan, verify_options: [] }).canSet).toBe(false);
    expect(verifyLine({ plan: { ...plan, criteria: [] }, verify_options: ["tests"] }).text).toBe("");
  });
});
