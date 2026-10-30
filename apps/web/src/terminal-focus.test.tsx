import { afterEach, describe, expect, it, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { post } from "./api";
import { RunDrawer, TerminalFallback } from "./Remotea";
import { FOCUS_TEXT, focusSentText, shellWord, terminalFallback, type FocusResult, type RunDetail } from "./remotea-model";

const A = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", B = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb";
const detail = (id: string, values: Partial<RunDetail> = {}): RunDetail => ({
  id, task: { id: "t-1", title: "Task", status: "in_progress" }, state: "executing", state_revision: 3,
  state_reason: "", reason_text: "", state_since: "2026-10-30T10:00:00Z", generation: 1, plan_revision: 1, tool: "claude",
  model: "", branch: "eri/task-1", base_branch: "main", worktree: "/home/davin/wt/eri--task-1", effective_permission_mode: "",
  turn_count: 1, turn_cap: 15, started_at: null, finished_at: null, stop_requested_at: null, stop_mode: "", surf: "surf-eri-abc",
  host: { id: "h", name: "DavisPC", connected: true }, structured_answers: true, session: null, created_at: "2026-10-30T09:00:00Z",
  history: [], turns: [], commands: [], questions: [], plan: null,
  ...values,
});

function button(html: string, label: string) {
  const found = [...html.matchAll(/<button[^>]*>(?:(?!<\/button>).)*<\/button>/g)].map(m => m[0]).find(b => b.replace(/<[^>]+>/g, "").trim() === label);
  if (!found) throw new Error("no button " + label);
  return found.slice(0, found.indexOf(">") + 1);
}

afterEach(() => vi.unstubAllGlobals());

describe("open the run's terminal in remotea", () => {
  it("the drawer offers it once the loaded run is the panel's run and has a terminal", () => {
    const html = renderToStaticMarkup(<RunDrawer id={A} preloaded={detail(A)} onClose={() => {}}/>);
    expect(button(html, "Open terminal in remotea")).not.toContain("disabled");
    // Also for a settled run: its tab may still be open on the host.
    const settled = renderToStaticMarkup(<RunDrawer id={A} preloaded={detail(A, { state: "completed" })} onClose={() => {}}/>);
    expect(button(settled, "Open terminal in remotea")).not.toContain("disabled");
    const noTerminal = renderToStaticMarkup(<RunDrawer id={A} preloaded={detail(A, { surf: "" })} onClose={() => {}}/>);
    expect(button(noTerminal, "Open terminal in remotea")).toContain("disabled");
    const other = renderToStaticMarkup(<RunDrawer id={B} preloaded={detail(A)} onClose={() => {}}/>);
    expect(button(other, "Open terminal in remotea")).toContain("disabled");
  });

  it("posts to the run's focus route and reads both answers", async () => {
    const calls: string[] = [];
    let answer: FocusResult = { sent: true, desktops: 1 };
    vi.stubGlobal("fetch", async (url: string, init: RequestInit = {}) => {
      calls.push(`${init.method} ${url}`);
      return new Response(JSON.stringify(answer), { status: 200 });
    });
    expect(await post<FocusResult>(`/remotea/runs/${A}/focus`)).toEqual({ sent: true, desktops: 1 });
    answer = { sent: false, reason: "no_desktop" };
    expect(await post<FocusResult>(`/remotea/runs/${A}/focus`)).toEqual({ sent: false, reason: "no_desktop" });
    expect(calls).toEqual([`POST /api/v1/remotea/runs/${A}/focus`, `POST /api/v1/remotea/runs/${A}/focus`]);
    expect(focusSentText(detail(A), { sent: true })).toBe("Brought the terminal forward in remotea on DavisPC.");
    expect(focusSentText(detail(A), { sent: false, reason: "no_desktop" })).toBe("");
  });

  it("when the tab cannot come forward, says why and where the terminal is, with commands to copy", () => {
    const info = terminalFallback(detail(A), { sent: false, reason: "no_desktop" });
    expect(info).toEqual({
      message: FOCUS_TEXT.no_desktop, host: "DavisPC", worktree: "/home/davin/wt/eri--task-1", branch: "eri/task-1",
      commands: [
        { label: "Open a shell in the worktree", command: "cd /home/davin/wt/eri--task-1" },
        { label: "See the terminal's screen", command: "remotea screen surf-eri-abc" },
      ],
    });
    for (const reason of ["host_offline", "no_surface", "host_too_old", "no_answer", "failed"]) {
      expect(terminalFallback(detail(A), { sent: false, reason }).message).toBe(FOCUS_TEXT[reason]);
    }
    expect(terminalFallback(detail(A), { sent: false, reason: "something_new" }).message).toBe(FOCUS_TEXT.failed);
    expect(terminalFallback(detail(A), { sent: false, reason: "failed", detail: "interrupted" }).message).toBe(FOCUS_TEXT.failed + " (interrupted)");
    const bare = terminalFallback(detail(A, { worktree: "", surf: "", host: null, branch: "" }), { sent: false, reason: "no_surface" });
    expect(bare).toMatchObject({ host: "the remotea host", worktree: null, branch: null, commands: [] });
    const html = renderToStaticMarkup(<TerminalFallback run={detail(A)} result={{ sent: false, reason: "no_desktop" }}/>);
    expect(html).toContain("No remotea desktop app is connected");
    expect(html).toContain("DavisPC");
    expect(html).toContain("<code>eri/task-1</code>");
    expect(html).toContain("cd /home/davin/wt/eri--task-1");
    expect(button(html.replace(/<small>[^<]*<\/small>/g, ""), "Copy")).toContain('aria-label="Copy: cd /home/davin/wt/eri--task-1"');
  });

  it("quotes a path as one shell word", () => {
    expect(shellWord("/home/davin/wt/eri--task-1")).toBe("/home/davin/wt/eri--task-1");
    expect(shellWord("/home/me/my work")).toBe("'/home/me/my work'");
    expect(shellWord("/tmp/it's; rm -rf ~")).toBe("'/tmp/it'\\''s; rm -rf ~'");
    expect(shellWord("$(boom)")).toBe("'$(boom)'");
    // A surface id that is not a plain id gets no command at all.
    expect(terminalFallback(detail(A, { surf: "surf-x; rm -rf ~" }), null).commands.map(c => c.command)).toEqual(["cd /home/davin/wt/eri--task-1"]);
  });
});
