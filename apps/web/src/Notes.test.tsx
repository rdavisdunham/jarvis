// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { FrozenPrdNotice, NotesPage, type NoteRecord } from "./Notes";
import { emptyFilter, emptyOrganization } from "./productivity";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe("a PRD frozen with a live run", () => {
  it("names the run and says the human's edit does not change the frozen plan", () => {
    const html = renderToStaticMarkup(<FrozenPrdNotice runId="0123456789abcdef" />);
    expect(html).toContain("Frozen with run 01234567");
    expect(html).toContain('title="Run 0123456789abcdef"');
    expect(html).toContain("do not change the plan");
  });
});

describe("the notes loading status", () => {
  let host: HTMLDivElement, root: Root;
  beforeEach(() => { host = document.createElement("div"); document.body.append(host); root = createRoot(host); });
  afterEach(() => { act(() => root.unmount()); host.remove(); vi.unstubAllGlobals(); });
  const note: NoteRecord = { id: "n1", title: "Kitchen remodel", excerpt: "Tile and grout", tags: [], project_id: null,
    conversation_id: null, archived: false, revision: 1, index_state: "ready", updated_at: "2026-11-01T12:00:00Z", tasks: [] };
  const page = (revision: number) => <NotesPage archived={false} onArchived={() => {}} mode="keyword" onMode={() => {}}
    organization={emptyOrganization} organizationFilter={emptyFilter} onOrganizationFilter={() => {}} query=""
    projects={[]} project="" onProject={() => {}} revision={revision} onVisible={() => {}} onOpen={() => {}} onNew={() => {}}/>;
  const status = () => [...host.querySelectorAll('[role="status"]')].map(n => n.textContent).join(" | ");

  it("shows only for an empty first load, not above cards already drawn", async () => {
    const pending: ((r: Response) => void)[] = [];
    vi.stubGlobal("fetch", (url: string) => String(url).includes("/notes?")
      ? new Promise<Response>(resolve => pending.push(resolve))
      : Promise.resolve(new Response(JSON.stringify({ items: [] }))));
    await act(async () => { root.render(page(1)); });
    expect(status()).toContain("Loading notes…");
    await act(async () => { pending.shift()!(new Response(JSON.stringify({ items: [note], next_offset: null }))); });
    expect(host.textContent).toContain("Kitchen remodel");
    expect(status()).not.toContain("Loading notes…");
    // A background refresh (a new note revision) keeps the cards and does not announce loading again.
    await act(async () => { root.render(page(2)); });
    expect(pending).toHaveLength(1);
    expect(host.textContent).toContain("Kitchen remodel");
    expect(status()).not.toContain("Loading notes…");
    expect(host.querySelector(".note-grid")!.getAttribute("aria-busy")).toBe("true");
  });
});
