import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { RemoteaActionsContext, RemoteaDataContext, type RemoteaActions, type RemoteaData } from "./remotea-context";
import { RunButton } from "./RunButton";
import { SwarmPlan } from "./Swarm";
import {
  batchLine, costLine, editBody, integrateLines, itemLabel, itemTone, moveTargets, startBody, startState,
  type Swarm, type SwarmBatch, type SwarmItem,
} from "./swarm-model";

const item = (over: Partial<SwarmItem> = {}): SwarmItem => ({ task_id: "t1", title: "Alpha", repo: "/repo/jarvis",
  target: { id: "x", name: "jarvis", repo: "/repo/jarvis" }, blocked_by: [], areas: [], why: "", error: null, retries: 0,
  status: "planned", run: null, ...over });
const cost = { runs: 1, basis: "no finished runs yet", eridani_cost: null, turns: null, minutes: null, judge_model: "Luna", note: "" };
const batch = (over: Partial<SwarmBatch> = {}): SwarmBatch => ({ id: "b1", number: 1, reason: "Independent tasks.", state: "pending",
  started_at: null, settled_at: null, items: [item()], cost, integrate: null, ...over });
const swarm = (over: Partial<Swarm> = {}): Swarm => ({ id: "s1", record_id: "r1", title: "Website", state: "draft", revision: 3, cap: 3,
  auto_continue: false, harness: "claude", planner: { source: "fallback", note: "No planning model is configured; grouped by repository and Blocks links." },
  excluded: [], warnings: [], batches: [batch(), batch({ id: "b2", number: 2, items: [item({ task_id: "t2", title: "Beta" })] })],
  next_batch: "b1", created_at: "", updated_at: "", ...over });

describe("swarm-model", () => {
  it("labels items by run state, waiting slot or refusal", () => {
    expect(itemLabel(item())).toBe("Planned");
    expect(itemLabel(item({ status: "queued" }))).toBe("Waiting for a free slot");
    expect(itemLabel(item({ status: "not_started", error: { code: "NEEDS_TARGET", message: "Choose" } }))).toBe("Could not start");
    expect(itemTone(item({ status: "awaiting_approval", run: { id: "r", state: "awaiting_approval", state_reason: "", branch: "b", summary: "", pr_urls: [] } }))).toBe("attention");
    expect(itemTone(item({ status: "not_started" }))).toBe("bad");
  });
  it("starts only the next batch, and only when none is running", () => {
    expect(startState(swarm()).batch?.id).toBe("b1");
    const running = swarm({ state: "running", next_batch: null, batches: [batch({ state: "running" }), batch({ id: "b2", number: 2 })] });
    expect(startState(running)).toEqual({ batch: null, reason: expect.stringContaining("Batch 1 is running") });
    expect(startState(swarm({ state: "canceled" })).batch).toBeNull();
  });
  it("builds edit and start bodies with the plan's revision", () => {
    expect(editBody(swarm(), { op: "cap", value: 2 })).toEqual({ expected_revision: 3, ops: [{ op: "cap", value: 2 }] });
    expect(startBody(swarm(), true)).toEqual({ expected_revision: 3, all: true });
  });
  it("offers only other pending batches as move targets", () => {
    expect(moveTargets(swarm(), "t1").map(b => b.id)).toEqual(["b2"]);
  });
  it("words the batch, the estimate and the integrate step", () => {
    expect(batchLine(batch())).toBe("Batch 1 · not started · 1 task");
    expect(costLine(cost)).toContain("no finished runs yet");
    expect(costLine({ ...cost, runs: 2, eridani_cost: 0.12, turns: 18, minutes: 40, basis: "average of your last 5 finished runs" }))
      .toBe("2 runs: about $0.12 in Eridani checks, about 18 agent turns, about 40 min each (average of your last 5 finished runs)");
    const s = swarm();
    expect(integrateLines(s, { blocking: true, repos: [{ repo: "/repo/jarvis", merge_order: ["t2", "t1"], runs: [],
      conflicts: [{ tasks: ["t1", "t2"], shared: ["apps/api"], basis: "planner areas", unmerged: true }] }] }))
      .toEqual(["/repo/jarvis: merge Beta → Alpha; Alpha and Beta both change apps/api (planner areas)."]);
  });
});

const data = (over: Partial<RemoteaData> = {}): RemoteaData => ({ enabled: true, status: { enabled: true, key: null, hosts: [{ id: "h", name: "desk",
  machine: "wsl", connected: true, last_seen_at: null, conn_epoch: 1, caps: ["spawn"], harnesses: { claude: { spawn: true } }, proto: 2 }], link_url: "",
  connect_command: "" }, activeRuns: [], runsByTask: new Map(), pending: [], cards: {}, error: "", panel: { run: null, collapsed: false }, start: null, ...over });

describe("Swarm view", () => {
  it("renders batches, edits for pending ones and Start for the next batch", () => {
    const html = renderToStaticMarkup(<SwarmPlan swarm={swarm()} onChange={() => {}} onOpenRun={() => {}}/>);
    expect(html).toContain("Batch 1 · not started · 1 task");
    expect(html).toContain("Leave out");
    expect(html).toContain("Merge into batch 1");
    expect(html).toContain("Start batch 1");
    expect(html).toContain("Start all");
    expect(html).toContain("No planning model is configured");
  });
  it("shows a running batch's runs and holds Start", () => {
    const running = swarm({ state: "running", next_batch: null, batches: [batch({ state: "running", items: [item({ status: "executing",
      run: { id: "r1", state: "executing", state_reason: "", branch: "task/t1", summary: "Adding tests", pr_urls: [] } })] })] });
    const html = renderToStaticMarkup(<SwarmPlan swarm={running} onChange={() => {}} onOpenRun={() => {}}/>);
    expect(html).toContain("Open run");
    expect(html).toContain("Adding tests");
    expect(html).not.toContain("Leave out");
    expect(html).toContain("Batch 1 is running");
  });
  it("Play on a project plans a swarm instead of starting a run", () => {
    const actions = { openSwarm: () => {}, openStart: () => {}, openRun: () => {} } as unknown as RemoteaActions;
    const html = renderToStaticMarkup(<RemoteaActionsContext.Provider value={actions}><RemoteaDataContext.Provider value={data()}>
      <RunButton taskId="p1" title="Website" project/></RemoteaDataContext.Provider></RemoteaActionsContext.Provider>);
    expect(html).toContain('aria-label="Plan a swarm of agents for Website"');
  });
});
