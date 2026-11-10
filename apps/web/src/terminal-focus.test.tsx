import { afterEach, describe, expect, it, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { post } from "./api";
import { RunDrawer, TerminalFallback } from "./Remotea";
import { FOCUS_TEXT, focusSentText, focusTitle, RESUME_COMMAND_LABEL, reopenButton, shellWord, terminalFallback, type FocusResult, type RunDetail } from "./remotea-model";

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
  const found = [...html.matchAll(/<button[^>]*>(?:(?!<\/button>).)*<\/button>/g)].map(m => m[0]).find(b => b.replace(/<[^>]+>/g, "").trim() === label || b.includes('aria-label="Open terminal in Remotea"') && label === "Open terminal in remotea");
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
      message: FOCUS_TEXT.no_desktop, host: "DavisPC", worktree: "/home/davin/wt/eri--task-1", branch: "eri/task-1", reopen: null,
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

  it("offers the worker's resume command last, as a separate terminal, and says why the button is off", () => {
    const cmd = "remotea open --group Inbox --cwd /x -- claude --resume abc";
    const info = terminalFallback({ ...detail(A), worker_resume_command: cmd }, { sent: false, reason: "no_surface" });
    expect(info.commands.at(-1)).toEqual({ label: RESUME_COMMAND_LABEL, command: cmd });
    expect(RESUME_COMMAND_LABEL).toContain("not connected to this run");
    expect(terminalFallback(detail(A), null).commands.map(c => c.command)).not.toContain(cmd);
    expect(focusTitle(detail(A))).toBe("Open this run's terminal in the remotea app on DavisPC");
    expect(focusTitle(detail(A, { surf: "" }))).toBe("This run has no terminal yet");
  });

  it("puts Reopen terminal first, only when the server offers it, pending while the host has not answered", () => {
    const exited = (reopen: RunDetail["reopen"]) => detail(A, { state: "needs_you", state_reason: "exited", reopen,
      worker_resume_command: "remotea open -- claude --resume abc" });
    expect(reopenButton(exited({ available: true, pending: false, why: null }))).toMatchObject({ enabled: true, label: "Reopen terminal" });
    expect(reopenButton(exited({ available: false, pending: true, why: "A reopen is already on its way to the host." })))
      .toEqual({ enabled: false, label: "Reopening terminal…", reason: "A reopen is already on its way to the host." });
    for (const reopen of [{ available: false, pending: false, why: "Update remotea to reopen an exited agent from here." },
      { available: false, pending: false, why: null }, null, undefined]) expect(reopenButton(exited(reopen))).toBeNull();
    const available = exited({ available: true, pending: false, why: null });
    expect(terminalFallback(available, { sent: false, reason: "no_surface" }).reopen?.label).toBe("Reopen terminal");
    const html = renderToStaticMarkup(<TerminalFallback run={available} result={{ sent: false, reason: "no_surface" }} onReopen={() => {}}/>);
    expect(html.indexOf("Reopen terminal")).toBeGreaterThan(-1);
    expect(html.indexOf("Reopen terminal")).toBeLessThan(html.indexOf("cd /home/davin/wt/eri--task-1"));  // Ahead of the copies.
    expect(html.indexOf("claude --resume abc")).toBeGreaterThan(html.indexOf("remotea screen"));  // The resume command last.
    // Without a handler (or when not offered) there is no button.
    expect(renderToStaticMarkup(<TerminalFallback run={available} result={{ sent: false, reason: "no_surface" }}/>)).not.toContain("Reopen terminal");
    // The drawer header: beside Open terminal in remotea, only while offered.
    expect(button(renderToStaticMarkup(<RunDrawer id={A} preloaded={available} onClose={() => {}}/>), "Reopen terminal")).not.toContain("disabled");
    expect(button(renderToStaticMarkup(<RunDrawer id={A} preloaded={exited({ available: false, pending: true, why: "x" })} onClose={() => {}}/>), "Reopening terminal…")).toContain("disabled");
    expect(renderToStaticMarkup(<RunDrawer id={A} preloaded={detail(A)} onClose={() => {}}/>)).not.toContain("Reopen terminal");
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
