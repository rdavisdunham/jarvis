// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Onboarding } from "./Onboarding";
import { setupEstablished, type SetupSession } from "./setup-model";
import type { Project } from "./types";
import type { Space } from "./productivity";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const space = (archived = false): Space => ({ id: "s1", name: "Home", description: "", archived, revision: 1 });
const project = (archived = false): Project => ({ id: "p1", name: "Kitchen", description: "", archived, revision: 1 });

describe("retiring the setup banner", () => {
  it("counts an account with a live space and a live project as set up", () => {
    expect(setupEstablished({ spaces: [space()], projects: [project()] })).toBe(true);
    expect(setupEstablished({ spaces: [], projects: [project()] })).toBe(false);
    expect(setupEstablished({ spaces: [space()], projects: [] })).toBe(false);
    expect(setupEstablished({ spaces: [space(true)], projects: [project()] })).toBe(false);
    expect(setupEstablished({ spaces: [space()], projects: [project(true)] })).toBe(false);
  });

  let host: HTMLDivElement, root: Root;
  beforeEach(() => { host = document.createElement("div"); document.body.append(host); root = createRoot(host); });
  afterEach(() => { act(() => root.unmount()); host.remove(); vi.unstubAllGlobals(); });
  const progress = (status: string) => ({ revision: 2, status, step: "profile", preferred_name: "Davis", timezone: "America/Chicago",
    purpose: "", classification_name: "", description: "", examples: "", design_notes: "" });
  const banner = (established: boolean) =>
    <Onboarding compact established={established} onChanged={() => {}} onTalk={async () => {}}/>;

  it("stays off Today once the account has structure, even with setup still in progress", async () => {
    vi.stubGlobal("fetch", async () => new Response(JSON.stringify(progress("in_progress"))));
    await act(async () => { root.render(banner(true)); });
    expect(host.textContent).toBe("");
    await act(async () => { root.render(banner(false)); });
    expect(host.textContent).toContain("Make Eridani yours");
  });

  const button = (name: string) => [...host.querySelectorAll("button")].find(b => b.textContent === name);

  it("asks for voice or typing before the setup conversation starts", async () => {
    const modes: string[] = [];
    vi.stubGlobal("fetch", async () => new Response(JSON.stringify(progress("in_progress"))));
    vi.stubGlobal("requestAnimationFrame", (done: () => void) => { done(); return 0; });
    await act(async () => { root.render(<Onboarding compact onChanged={() => {}} onTalk={async mode => { modes.push(mode); }}/>); });
    expect(host.querySelector("section")?.getAttribute("aria-label")).toBe("Set up with Dani");
    expect(button("Fill out the form")).toBeTruthy();
    await act(async () => { button("Set up with Dani")!.click(); });
    expect(host.textContent).toContain("Want to do this by voice?");
    expect(button("Use voice mode")?.className).toContain("btn-primary");
    await act(async () => { button("Type instead")!.click(); });
    await act(async () => { button("Set up with Dani")!.click(); });
    await act(async () => { button("Use voice mode")!.click(); });
    expect(modes).toEqual(["typed", "voice"]);
  });

  it("resumes a typed setup without asking again, and offers an update after completion", async () => {
    const modes: string[] = [];
    vi.stubGlobal("fetch", async () => new Response(JSON.stringify(progress("in_progress"))));
    vi.stubGlobal("requestAnimationFrame", (done: () => void) => { done(); return 0; });
    const session = (status: SetupSession["status"]): SetupSession => ({ status, revision: 1, conversation_id: "c1", mode: "typed",
      focus: "both", notes: {}, tree_proposal_id: null, history_off: false });
    await act(async () => { root.render(<Onboarding onChanged={() => {}} setup={session("interviewing")} onTalk={async mode => { modes.push(mode); }}/>); });
    await act(async () => { button("Resume with Dani")!.click(); });
    expect(modes).toEqual(["typed"]);
    const focuses: (string | undefined)[] = [];
    await act(async () => { root.render(<Onboarding onChanged={() => {}} setup={session("completed")} rerunFocus="organization"
      onTalk={async (_mode, focus) => { focuses.push(focus); }}/>); });
    expect(button("Update with Dani")).toBeTruthy();
    // Settings → Organization re-runs the interview focused on the organization.
    await act(async () => { button("Update with Dani")!.click(); });
    await act(async () => { button("Type instead")!.click(); });
    expect(focuses).toEqual(["organization"]);
  });

  it("keeps a manual way to hide an in-progress setup", async () => {
    const posts: unknown[] = [];
    vi.stubGlobal("fetch", async (_url: string, init?: RequestInit) => {
      if (init?.method === "POST") { posts.push(JSON.parse(String(init.body))); return new Response(JSON.stringify({ data: progress("skipped") })); }
      return new Response(JSON.stringify(progress("in_progress")));
    });
    await act(async () => { root.render(banner(false)); });
    const hide = [...host.querySelectorAll("button")].find(b => b.textContent === "Hide for now")!;
    expect(hide).toBeTruthy();
    await act(async () => { hide.click(); });
    expect(JSON.stringify(posts)).toContain('"status":"skipped"');
  });
});
