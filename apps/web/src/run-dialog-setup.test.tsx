import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { SetupLink } from "./RunTaskDialog";
import { TargetPicker } from "./Remotea";
import { setupReason, type NeedsTarget, type Preflight } from "./remotea-model";

const pre = (values: Partial<Preflight> = {}): Preflight => ({
  target: { id: "t", name: "jarvis", repo: "/r" }, needs_target: false, need: null, problems: [], active_run: null,
  interview: true, harnesses: { claude: true, codex: true }, ...values });

describe("the Run dialog's link to the guided setup view", () => {
  it("shows for a target to choose, a setup problem, or an agent the target cannot start", () => {
    expect(setupReason(pre({ needs_target: true, target: null }), "claude")).toBe("No repository is set up for this task yet.");
    const tools = { code: "MCP_UNCONFIGURED", message: "Agents on this target have no Eridani tools: no key.", blocking: true, freestyle_ok: true };
    expect(setupReason(pre({ problems: [tools] }), "claude")).toBe(tools.message);
    const offline = { code: "HOST_OFFLINE", message: "The remotea server is not connected." };
    expect(setupReason(pre({ problems: [offline] }), "claude")).toBe(offline.message);
    const verify = { code: "VERIFY_NOT_READY", message: "This target's verify command is not ready (tests: not on PATH).", blocking: false };
    expect(setupReason(pre({ problems: [verify] }), "claude")).toBe(verify.message);
    const template = { code: "TARGET_NOT_READY", message: "The jarvis template is not ready: node_modules: missing.", blocking: true };
    expect(setupReason(pre({ problems: [template] }), "claude")).toBe(template.message);
    expect(setupReason(pre({ harnesses: { claude: true, codex: false }, harness_reasons: { codex: "Codex is not enabled on jarvis." } }), "codex"))
      .toBe("Codex is not enabled on jarvis.");
    expect(setupReason(pre({ harnesses: { claude: true, codex: false } }), "codex")).toBe("This agent cannot start on this task's target.");
  });

  it("stays away when nothing the setup view fixes is in the way", () => {
    expect(setupReason(pre(), "claude")).toBe("");
    expect(setupReason(undefined, "claude")).toBe("");
    expect(setupReason(null, "claude")).toBe("");
    expect(setupReason(pre({ problems: [{ code: "NOT_FOUND", message: "That task is no longer available." }] }), "claude")).toBe("");
    // A run already working on the task: the dialog offers Open run, not setup.
    expect(setupReason(pre({ active_run: { id: "r", state: "executing" }, problems: [{ code: "HOST_OFFLINE", message: "x" }] }), "claude")).toBe("");
  });

  it("renders a button that names what is missing", () => {
    const html = renderToStaticMarkup(<SetupLink reason="The remotea server is not connected." onSetup={() => {}}/>);
    expect(html).toContain('title="The remotea server is not connected."');
    expect(html).toContain("Open remotea setup");
    expect(html).toContain('type="button"');
  });
});

describe("changing where a task runs from the Run dialog", () => {
  const need = { task_id: "task-1", candidates: [], records: [{ id: "p1", title: "Eridani roadmap", type_id: "project" }],
    advertised: [{ host_id: "h", host_name: "davispc", host_target: "remotea", repo: "/home/davin/remotea", worktree_root: "/wt",
      configured: false, source: "host" as const, host_connected: true }] } as unknown as NeedsTarget;

  it("defaults to the task's project when a target is first chosen", () => {
    const html = renderToStaticMarkup(<TargetPicker need={need} onSaved={() => {}} onCancel={() => {}}/>);
    expect(html).toMatch(/<option value="p1" selected="">Remember for Eridani roadmap<\/option>/);
  });

  it("defaults to only this task when changing a target that already resolves", () => {
    const html = renderToStaticMarkup(<TargetPicker need={need} initialScope="task" onSaved={() => {}} onCancel={() => {}}/>);
    expect(html).toMatch(/<option value="task" selected="">Only this task<\/option>/);
    expect(html).not.toMatch(/<option value="p1" selected=""/);
  });
});
