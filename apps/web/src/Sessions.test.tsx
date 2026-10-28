import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { SessionCard, since, type AgentSession } from "./Sessions";

const base: AgentSession = {
  id: "s-1", tool: "claude_code", tool_label: "Claude Code", session_id: "abc", title: "Fix flaky login test",
  summary: "Fixed the login race in the auth middleware.", first_prompt: "The login test fails", last_prompt: "",
  machine: "pc", environment: "wsl", cwd: "/home/davin/acme-app", repo: "acme-app", branch: "davin/eng-12-login",
  turn_count: 3, started_at: null, last_active_at: "2026-10-26T10:00:00Z", pr_urls: ["https://github.com/acme/app/pull/4"],
  issue_refs: [{ identifier: "ENG-12", task_id: "t-1", url: "https://linear.app/acme/issue/ENG-12/x" }],
  resume: { cmd: "cd /home/davin/acme-app && claude --resume abc" },
  filed: { source: "branch", record_id: "r-1", task_id: "t-1", record_title: "Fix login" },
};
const render = (item: AgentSession) => renderToStaticMarkup(<SessionCard item={item} onChanged={() => {}}/>);

describe("agent session cards", () => {
  it("shows where the work lives, how it was filed and how to get back to it", () => {
    const html = render(base);
    expect(html).toContain("Fix flaky login test"); expect(html).toContain("Fixed the login race");
    expect(html).toContain("acme-app · davin/eng-12-login"); expect(html).toContain("3 prompts");
    expect(html).toContain("Filed under <strong>Fix login</strong> from the branch");
    expect(html).toContain('href="https://linear.app/acme/issue/ENG-12/x"'); expect(html).toContain("Copy resume command");
    expect(html).toContain(">PR<"); expect(html).toContain("Refile");
    expect(html).not.toContain("Use automatic filing"); expect(html).not.toContain("Unfiled");
  });
  it("marks unfiled desktop threads and offers their app hint instead of a command", () => {
    const html = render({ ...base, tool_label: "ChatGPT desktop", summary: "", resume: { hint: "ChatGPT desktop → Rill report" },
      filed: { source: null, record_id: null, task_id: null, record_title: null }, issue_refs: [], pr_urls: [] });
    expect(html).toContain("Unfiled"); expect(html).toContain("ChatGPT desktop → Rill report");
    expect(html).toContain("The login test fails"); expect(html).not.toContain("Copy resume command");
  });
  it("lets a hand-filed session return to automatic filing", () => {
    expect(render({ ...base, filed: { ...base.filed, source: "manual" } })).toContain("Use automatic filing");
  });
  it("shows where a session is running instead of a resume command", () => {
    const live = { state: "live" as const, workspace: "Acme", terminal: "surf-1", host: "DavisPC" };
    const html = render({ ...base, resume: { hint: "Live in remotea → Acme", live } });
    expect(html).toContain("Live in remotea · Acme"); expect(html).toContain('title="DavisPC · surf-1"');
    expect(html).not.toContain("Copy resume command"); expect(html).not.toContain("Live in remotea → Acme");
    expect(render({ ...base, resume: { live: { ...live, state: "lost", workspace: "" } } })).toContain("Restoring in remotea · Inbox");
  });
  it("describes recency compactly", () => {
    const now = new Date("2026-10-26T12:00:00Z").getTime();
    expect(since("2026-10-26T11:58:00Z", now)).toBe("2m ago");
    expect(since("2026-10-26T09:00:00Z", now)).toBe("3h ago");
    expect(since("2026-10-23T12:00:00Z", now)).toBe("3d ago");
    expect(since(null, now)).toBe("");
  });
});
