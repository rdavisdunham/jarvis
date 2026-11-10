// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { RunDrawer } from "./Remotea";
import { setDurableScope } from "./api";
import type { FocusResult, RunDetail } from "./remotea-model";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
const A = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa";
const run: RunDetail = {
  id: A, task: { id: "t-1", title: "Task", status: "in_progress" }, state: "executing", state_revision: 3,
  state_reason: "", reason_text: "", state_since: "2026-10-30T10:00:00Z", generation: 1, plan_revision: 1, tool: "claude",
  model: "", branch: "eri/task-1", base_branch: "main", worktree: "/home/davin/wt/eri--task-1", effective_permission_mode: "",
  turn_count: 1, turn_cap: 15, started_at: null, finished_at: null, stop_requested_at: null, stop_mode: "", surf: "surf-eri-abc",
  host: { id: "h", name: "DavisPC", connected: true }, structured_answers: true, session: null, created_at: "2026-10-30T09:00:00Z",
  history: [], turns: [], commands: [], questions: [], plan: null,
  worker_resume_command: "remotea open --group Inbox --cwd /home/davin/wt/eri--task-1 -- claude --resume abc",
};
let host: HTMLDivElement, root: Root;
beforeEach(() => {
  sessionStorage.clear(); localStorage.clear(); setDurableScope("terminal-focus:personal");
  host = document.createElement("div"); document.body.append(host); root = createRoot(host);
});
afterEach(() => { act(() => root.unmount()); host.remove(); vi.unstubAllGlobals(); });

async function open(answer: FocusResult, shown: RunDetail = run, posted: { url: string; body: Record<string, unknown> }[] = []) {
  let current = shown;
  vi.stubGlobal("fetch", async (url: string, init: RequestInit = {}) => {
    if (String(url).endsWith("/focus") && init.method === "POST") return new Response(JSON.stringify(answer));
    if (String(url).endsWith("/reopen") && init.method === "POST") {
      posted.push({ url: String(url), body: JSON.parse(String(init.body)) });
      current = { ...current, state_revision: current.state_revision + 1, reopen: { available: false, pending: true, why: "A reopen is already on its way to the host." } };
      return new Response(JSON.stringify(current));
    }
    if (String(url).includes("/drafts/")) return new Response(JSON.stringify({ value: null, revision: 0 }));
    return new Response(JSON.stringify(current));
  });
  await act(async () => root.render(<RunDrawer id={A} preloaded={shown} onClose={() => {}}/>));
  const button = host.querySelector<HTMLButtonElement>('[aria-label="Open terminal in Remotea"]')!;
  expect(button.title).toBe("Open this run's terminal in the remotea app on DavisPC");
  await act(async () => { button.click(); await new Promise(r => setTimeout(r, 0)); });
  return host.querySelector("header .run-focus-result");
}

it("says right under the button that the terminal was brought forward", async () => {
  const result = await open({ sent: true, desktops: 1 });
  expect(result?.textContent).toContain("Brought the terminal forward in remotea on DavisPC.");
  await act(async () => { result!.querySelector<HTMLButtonElement>('[aria-label="Dismiss"]')!.click(); });
  expect(host.querySelector(".run-focus-result")).toBeNull();
});

it("explains a closed terminal under the button and offers the resume command last", async () => {
  const result = await open({ sent: false, reason: "no_surface" });
  expect(result?.textContent).toContain("This run's terminal is not open on the host any more.");
  const commands = [...result!.querySelectorAll(".remotea-copy-row code")].map(c => c.textContent);
  expect(commands.at(-1)).toBe(run.worker_resume_command);
  expect(host.querySelector(".run-drawer-details .remotea-terminal-fallback")).toBeNull();  // Not hidden in Details.
});

it("offers Reopen terminal under the button when the agent exited, and sends it", async () => {
  const posted: { url: string; body: Record<string, unknown> }[] = [];
  const exited: RunDetail = { ...run, state: "needs_you", state_reason: "exited", reopen: { available: true, pending: false, why: null } };
  const result = await open({ sent: false, reason: "no_surface" }, exited, posted);
  const reopen = [...result!.querySelectorAll<HTMLButtonElement>("button")].find(b => b.textContent === "Reopen terminal")!;
  expect(reopen.disabled).toBe(false);
  await act(async () => { reopen.click(); await new Promise(r => setTimeout(r, 0)); });
  expect(posted).toHaveLength(1);
  expect(posted[0].url).toContain(`/remotea/runs/${A}/reopen`);
  expect(posted[0].body).toMatchObject({ expected_revision: 3 });
  expect(typeof posted[0].body.action_id).toBe("string");
  // The focus answer is cleared, and the header button shows the reopen as pending.
  expect(host.querySelector(".run-focus-result")).toBeNull();
  const pending = [...host.querySelectorAll<HTMLButtonElement>("header button")].find(b => b.textContent === "Reopening terminal…");
  expect(pending?.disabled).toBe(true);
});
