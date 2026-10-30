import { describe, expect, it } from "vitest";
import {
  APPROVAL_FACTS, attemptFacts, DEFAULT_DRAFT, freshnessFacts, sameText, startBlock, startBody, timeline, TYPED_TITLE,
  type Attempt, type Preflight, type Run, type RunDetail,
} from "./remotea-model";

const NOW = Date.parse("2026-10-30T12:00:00Z");
const at = (seconds: number) => new Date(NOW - seconds * 1000).toISOString();
const run = (values: Partial<Run> = {}): Run => ({
  id: "r1", task: { id: "t1", title: "Task", status: "in_progress" }, state: "awaiting_approval", state_revision: 2, state_reason: "",
  reason_text: "", state_since: at(240), generation: 1, plan_revision: 1, tool: "claude", model: "",
  branch: "b", base_branch: "main", worktree: "", effective_permission_mode: "", turn_count: 1, turn_cap: 40,
  started_at: at(3600), finished_at: null, stop_requested_at: null, stop_mode: "", surf: "surf-1",
  host: { id: "h", name: "desk", connected: true, last_seen_at: at(5) }, structured_answers: true, session: null,
  created_at: at(3600), ...values,
});
const attempt = (values: Partial<Attempt> = {}): Attempt => ({
  id: "a1", kind: "verify", status: "passed", generation: 1, turn_seq: 3, plan_revision: 1, verify_id: "tests", initiator: "eridani",
  verdict: null, gap: null, summary: null, criteria: null, rules: null, non_verdict: null, exit_code: 0, tail: null, on_demand: null,
  created_at: at(200), updated_at: at(180), ...values,
});

describe("freshness facts (E-S6)", () => {
  it("keeps each fact apart, with its own age and tone", () => {
    const facts = freshnessFacts(run({
      last_turn_at: at(120),
      verification: { verify_id: "tests", status: "passed", at: at(180), evidence: "fresh", isolated: true,
        workspace_complete: true, workspace_reasons: [] },
      pending_controls: [{ kind: "input", mode: "steer", status: "uncertain", at: at(40) },
        { kind: "verify", status: "pending", at: at(90) }],
    }), null, NOW);
    expect(facts.map(f => [f.key, f.label, f.tone])).toEqual([
      ["host", "host seen just now", "ok"],
      ["turn", "last turn 2m ago", "plain"],
      ["state", "awaiting approval for 4m", "plain"],
      ["verify", "verify tests passed 3m ago · fresh", "ok"],
      ["workspace", "workspace fingerprint 3m ago", "plain"],
      ["sandbox", "sandboxed", "ok"],
      ["control:0", "steer 40s ago: delivery uncertain", "warn"],
      ["control:1", "verify sent 2m ago, not taken by remotea yet", "warn"],
    ]);
    expect(facts.filter(APPROVAL_FACTS).map(f => f.key)).toEqual(["host", "verify", "workspace", "sandbox", "control:0", "control:1"]);
  });

  it("warns on a silent host, stale or incomplete evidence, an unsandboxed check and a verification past its deadline", () => {
    const silent = freshnessFacts(run({ host: { id: "h", name: "desk", connected: false, last_seen_at: at(240) } }), null, NOW);
    expect(silent[0]).toMatchObject({ label: "host silent for 4m", tone: "warn" });
    // The page's own host row wins over the run's copy.
    expect(freshnessFacts(run(), { connected: true, last_seen_at: at(12) }, NOW)[0].label).toBe("host seen 12s ago");
    const incomplete = freshnessFacts(run({ verification: { verify_id: "tests", status: "passed", at: at(60), evidence: "incomplete",
      isolated: false, isolation_reason: "bwrap (bubblewrap) is not installed on the service PATH", workspace_complete: false,
      workspace_reasons: ["untracked files capped at 2000 (2400 found)"] } }), null, NOW);
    expect(incomplete.filter(f => ["verify", "workspace", "sandbox"].includes(f.key)).map(f => [f.label, f.tone])).toEqual([
      ["verify tests passed 1m ago · incomplete workspace", "warn"],
      ["workspace fingerprint 1m ago · incomplete (untracked files capped at 2000 (2400 found))", "warn"],
      ["not sandboxed (bwrap (bubblewrap) is not installed on the service PATH)", "warn"],
    ]);
    const stale = freshnessFacts(run({ verification: { verify_id: "tests", status: "passed", at: at(60), evidence: "stale" } }), null, NOW);
    expect(stale.find(f => f.key === "verify")).toMatchObject({ tone: "warn", label: "verify tests passed 1m ago · stale: an older turn, or the checkout changed" });
    const late = freshnessFacts(run({ verification: { verify_id: "tests", status: "running", at: at(900), evidence: "pending",
      deadline_at: at(60) } }), null, NOW);
    expect(late.find(f => f.key === "verify")).toMatchObject({ tone: "warn", label: "verify tests running for 15m (past its deadline)" });
  });

  it("leaves out what an older server did not send", () => {
    const old = run({ host: { id: "h", name: "desk", connected: true } });
    expect(freshnessFacts(old, null, NOW).map(f => f.label)).toEqual(["host connected", "awaiting approval for 4m"]);
    expect(freshnessFacts(run({ last_turn_at: null }), null, NOW).map(f => f.key)).toContain("turn");
  });
});

describe("a verification's facts on its card", () => {
  it("shows the exit code, the sandbox and the workspace coverage as separate labels", () => {
    expect(attemptFacts(attempt({ isolated: false, isolation_reason: "the entry sets isolation \"none\"", workspace_complete: false,
      workspace_reasons: ["git status failed"] })).map(f => [f.label, f.tone])).toEqual([
      ["exit 0", "ok"], ["not sandboxed (the entry sets isolation \"none\")", "warn"], ["workspace incomplete (git status failed)", "warn"]]);
    expect(attemptFacts(attempt({ isolated: true, workspace_complete: true, exit_code: 1, status: "failed" })).map(f => f.label))
      .toEqual(["exit 1", "isolated", "complete workspace"]);
    expect(attemptFacts(attempt())).toEqual([{ key: "exit", label: "exit 0", tone: "ok" }]);  // An older host: the exit code only.
    expect(attemptFacts(attempt({ status: "running", exit_code: null }))).toEqual([]);
    expect(attemptFacts(attempt({ kind: "judge" }))).toEqual([]);
  });

  it("puts them on the timeline entry", () => {
    const detail: RunDetail = { ...run(), plan: null, history: [], turns: [], commands: [], questions: [],
      attempts: [attempt({ isolated: false, isolation_reason: "no bwrap" })] };
    expect(timeline(detail)[0].facts?.map(f => f.label)).toEqual(["exit 0", "not sandboxed (no bwrap)"]);
  });
});

describe("long Eridani texts are not shown twice", () => {
  const long = "Implement the plan. ".repeat(300);  // 6,000 characters, past every clip.
  const detail = (turns: RunDetail["turns"], history: RunDetail["history"]): RunDetail =>
    ({ ...run({ state: "executing" }), plan: null, history, turns, commands: [], questions: [] });
  const turn = (values: Partial<RunDetail["turns"][number]>): RunDetail["turns"][number] => ({
    id: "t", seq: 1, text: "", clipped: false, truncated: false, generation: 1, at: at(10), prompt: "", ...values });

  it("matches a clipped copy by prefix, and short texts only whole", () => {
    expect(sameText(long.slice(0, 2000), long.slice(0, 4000))).toBe(true);  // The "sent" entry vs the host's prompt.
    expect(sameText(long, long.slice(0, 500) + "…")).toBe(true);  // A turn's 500-character prompt, with an ellipsis.
    expect(sameText("Use  pg\n", "Use pg")).toBe(true);
    expect(sameText("yes", "yes, and also drop the table")).toBe(false);
    expect(sameText("", "")).toBe(false);
  });

  it("dedupes the host's user prompt and an older host's turn prompt against the sent text", () => {
    const history = [{ at: at(30), event: "sent", text: long.slice(0, 2000), kind: "execute", actor: { kind: "eridani" } }];
    const entries = timeline(detail([
      turn({ id: "u1", role: "user", text: long.slice(0, 4000), at: at(20) }),
      turn({ id: "a1", role: "assistant", text: "Done.", prompt: long.slice(0, 500) }),
    ], history));
    expect(entries.map(e => e.title)).toEqual(["Sent by Eridani · plan frozen, start work", "Turn 1"]);
    // Something actually typed in the terminal still shows.
    const typed = timeline(detail([turn({ id: "u2", role: "user", text: "Also fix the README", at: at(20) })], history));
    expect(typed.map(e => e.title)).toContain(TYPED_TITLE);
  });
});

describe("starting where agents have no Eridani tools", () => {
  const pre = (values: Partial<Preflight> = {}): Preflight => ({
    target: { id: "t", name: "jarvis", repo: "/r" }, needs_target: false, need: null, problems: [], active_run: null,
    interview: true, harnesses: { claude: true }, ...values });
  const tools = { code: "MCP_UNCONFIGURED", message: "Agents on this target have no Eridani tools: no key.", blocking: true, freestyle_ok: true };

  it("blocks Start unless freestyle is on; warnings never block", () => {
    expect(startBlock(pre({ problems: [tools] }), DEFAULT_DRAFT)).toEqual({ problem: tools, freestyle: true });
    expect(startBlock(pre({ problems: [tools] }), { freestyle: true })).toEqual({ problem: null, freestyle: true });
    const warning = { code: "VERIFY_NOT_READY", message: "not ready", blocking: false };
    expect(startBlock(pre({ problems: [warning] }), DEFAULT_DRAFT)).toEqual({ problem: null, freestyle: false });
    const offline = { code: "HOST_OFFLINE", message: "The remotea server is not connected." };  // Older server: no flag.
    expect(startBlock(pre({ problems: [offline, tools] }), { freestyle: true })).toEqual({ problem: offline, freestyle: false });
    expect(startBlock(null, DEFAULT_DRAFT)).toEqual({ problem: null, freestyle: false });
  });

  it("sends freestyle only when chosen", () => {
    expect(startBody("t1", { ...DEFAULT_DRAFT, freestyle: true })).toEqual({ task_id: "t1", harness: "claude", freestyle: true });
    expect(startBody("t1", DEFAULT_DRAFT)).toEqual({ task_id: "t1", harness: "claude" });
  });
});
