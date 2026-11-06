// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Onboarding } from "./Onboarding";
import { setupEstablished } from "./setup-model";
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
