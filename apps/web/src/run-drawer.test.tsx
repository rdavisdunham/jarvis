import { afterEach, describe, expect, it, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { api, post } from "./api";
import { RunDrawer } from "./Remotea";
import { isRevisionConflict, latestOnly, latestRun, RUN_CHANGED, runAction, type RunAction, type RunDetail } from "./remotea-model";

const A = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", B = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb";
const detail = (id: string, title: string, values: Partial<RunDetail> = {}): RunDetail => ({
  id, task: { id: "t-" + title, title, status: "in_progress" }, state: "awaiting_approval", state_revision: id === A ? 3 : 7,
  state_reason: "judge_met", reason_text: "", state_since: "2026-10-30T10:00:00Z", generation: 1, plan_revision: 1, tool: "claude",
  model: "", branch: "task/x", base_branch: "main", worktree: "", effective_permission_mode: "", turn_count: 1, turn_cap: 15,
  started_at: null, finished_at: null, stop_requested_at: null, stop_mode: "", surf: "surf-1",
  host: { id: "h", name: "desk", connected: true }, structured_answers: true, session: null, created_at: "2026-10-30T09:00:00Z",
  history: [], turns: [], commands: [], questions: [],
  plan: { revision: 1, title, prd_text: "", steps: [], verify: { target_id: "jarvis", verify_id: "tests" }, criteria: [] },
  ...values,
});

/** A fake fetch whose responses the test releases by URL, in any order. */
function fakeFetch(bodies: Record<string, unknown>) {
  const waiting = new Map<string, () => void>(), signals = new Map<string, AbortSignal | undefined>();
  const calls: { url: string; body: unknown }[] = [];
  vi.stubGlobal("fetch", (url: string, init: RequestInit = {}) => {
    calls.push({ url, body: init.body ? JSON.parse(String(init.body)) : undefined });
    signals.set(url, init.signal ?? undefined);
    return new Promise<Response>(resolve => waiting.set(url, () => resolve(new Response(JSON.stringify(bodies[url]), { status: 200 }))));
  });
  const release = async (url: string) => { waiting.get(url)!(); await new Promise(r => setTimeout(r, 0)); };
  return { calls, signals, release };
}

afterEach(() => vi.unstubAllGlobals());

describe("the run drawer's latest-only loading", () => {
  it("keeps run B when the responses for A and B arrive out of order, and binds the next action to B", async () => {
    const net = fakeFetch({ [`/api/v1/remotea/runs/${A}`]: detail(A, "Task A"), [`/api/v1/remotea/runs/${B}`]: detail(B, "Task B") });
    const requests = latestOnly();
    const installed: RunDetail[] = [];
    // The drawer's refresh, exactly as RunDrawer calls it: A is opened, then the panel switches to B.
    const load = (id: string) => latestRun(requests, id, signal => api<RunDetail>(`/remotea/runs/${id}`, { signal }))
      .then(result => { if (result && "run" in result) installed.push(result.run); return result; });
    const forA = load(A), forB = load(B);
    expect(net.signals.get(`/api/v1/remotea/runs/${A}`)?.aborted).toBe(true);  // Switching runs aborts the old request.
    await net.release(`/api/v1/remotea/runs/${B}`);
    await net.release(`/api/v1/remotea/runs/${A}`);  // A's late reply must not replace B.
    expect(await forB).toEqual({ run: expect.objectContaining({ id: B }) });
    expect(await forA).toBeNull();
    const shown = installed.at(-1)!;
    expect(installed.map(r => r.id)).toEqual([B]);
    expect(shown.task?.title).toBe("Task B");
    // Title, controls and the next action's URL all refer to B.
    const html = renderToStaticMarkup(<RunDrawer id={B} preloaded={shown} onClose={() => {}}/>);
    expect(html).toContain("Task B");
    expect(button(html, "Approve")).not.toContain("disabled");
    const next = runAction(shown, B, "approve")!;
    expect(next).toEqual({ url: `/remotea/runs/${B}/approve`, body: { expected_revision: 7 } });
  });

  it("drops an older response for the same run (a slow poll cannot roll the view back)", async () => {
    const requests = latestOnly();
    let first!: (r: RunDetail) => void;
    const slow = latestRun(requests, A, () => new Promise<RunDetail>(r => { first = r; }));
    const fresh = latestRun(requests, A, async () => detail(A, "Task A", { state_revision: 4 }));
    expect(await fresh).toEqual({ run: expect.objectContaining({ state_revision: 4 }) });
    first(detail(A, "Task A"));
    expect(await slow).toBeNull();
  });

  it("ignores a response for another run, and errors from superseded requests", async () => {
    const requests = latestOnly();
    expect(await latestRun(requests, B, async () => detail(A, "Task A"))).toBeNull();
    const old = latestRun(requests, A, () => Promise.reject(new Error("gone")));
    requests.cancel();  // Unmount, or the panel switched runs.
    expect(await old).toBeNull();
    expect(await latestRun(requests, A, () => Promise.reject(new Error("down")))).toEqual({ error: new Error("down") });
  });
});

function button(html: string, label: string) {
  const found = [...html.matchAll(/<button[^>]*>(?:(?!<\/button>).)*<\/button>/g)].map(m => m[0]).find(b => b.replace(/<[^>]+>/g, "").trim() === label);
  if (!found) throw new Error("no button " + label);
  return found.slice(0, found.indexOf(">") + 1);
}

describe("the run drawer's controls", () => {
  const labels = ["Message", "Steer", "Re-plan", "Verify now", "Stop", "Approve"];

  it("are all disabled while the loaded run is not the run the panel asked for", () => {
    const html = renderToStaticMarkup(<RunDrawer id={B} preloaded={detail(A, "Task A", { state: "executing" })} onClose={() => {}}/>);
    expect(html).toContain("Loading the selected run");
    for (const label of labels) expect(button(html, label), label).toContain("disabled");
    expect(html).toMatch(/<input type="checkbox"[^>]*disabled/);  // Freestyle
    expect(html).toMatch(/<textarea[^>]*disabled/);
  });

  it("are enabled once the loaded run is the panel's run", () => {
    const html = renderToStaticMarkup(<RunDrawer id={A} preloaded={detail(A, "Task A", { state: "executing" })} onClose={() => {}}/>);
    expect(html).not.toContain("Loading the selected run");
    for (const label of ["Re-plan", "Verify now", "Stop", "Approve"]) expect(button(html, label), label).not.toContain("disabled");
  });

  it("post to the loaded run with the revision the reader saw, except Stop", () => {
    const run = detail(A, "Task A");
    const actions: [RunAction, Record<string, unknown>, string, object][] = [
      ["approve", {}, "approve", { expected_revision: 3 }],
      ["replan", { reason: "x" }, "replan", { reason: "x", expected_revision: 3 }],
      ["verify", {}, "verify", { expected_revision: 3 }],
      ["verify-config", { verify_id: "tests" }, "verify-config", { verify_id: "tests", expected_revision: 3 }],
      ["freestyle", { enabled: true }, "freestyle", { enabled: true, expected_revision: 3 }],
      ["message", { text: "hi", mode: "queue" }, "message", { text: "hi", mode: "queue", expected_revision: 3 }],
      ["steer", { text: "hi", mode: "steer" }, "message", { text: "hi", mode: "steer", expected_revision: 3 }],
      ["stop", { mode: "interrupt" }, "stop", { mode: "interrupt" }],
    ];
    for (const [action, values, path, body] of actions) {
      expect(runAction(run, A, action, values)).toEqual({ url: `/remotea/runs/${A}/${path}`, body });
      expect(runAction(run, B, action, values)).toBeNull();  // Never to the prop id while another run is shown.
    }
    expect(runAction(null, A, "stop")).toBeNull();
  });

  it("reads a stale approval's REVISION_CONFLICT as 'the run changed' so the drawer reloads", async () => {
    vi.stubGlobal("fetch", async () => new Response(JSON.stringify({ error: {
      code: "REVISION_CONFLICT", message: "This run changed. Reload it and try again.", data: { run_id: A, state_revision: 4 } } }), { status: 409 }));
    const next = runAction(detail(A, "Task A"), A, "approve")!;
    const error = await post(next.url, next.body).then(() => null, e => e);
    expect(isRevisionConflict(error)).toBe(true);
    expect(RUN_CHANGED).toBe("This run changed; reloaded");
    expect(isRevisionConflict(new Error("Something went wrong."))).toBe(false);
  });
});
