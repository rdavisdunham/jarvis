// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { PersonPage } from "./PersonPage";
import type { OpenLoop, PersonView } from "./context-api";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const cite = (segment = 0) => ({ source_id: "src-1", source_title: "Pricing sync", occurred_at: "2026-11-03T15:00:00Z", segment, speaker: "Sam", quote: "We move pricing to annual plans." });
const person = (over: Partial<PersonView> = {}): PersonView => ({
  actor: { id: "act-sam", name: "Sam Lee", kind: "person", archived: false, revision: 2, is_me: false },
  identities: [{ id: "i1", namespace: "email", value: "sam@acme.test", status: "exact", source_scope: null },
    { id: "i2", namespace: "alias", value: "Sammy", status: "inferred", source_scope: null }],
  context: [
    { assertion_id: "a-pref", revision: 1, kind: "preference", statement: "Sam prefers annual plans.", lifecycle: "current", inferred: true, modality: "asserted", valid_from: null, citation: cite() },
    { assertion_id: "a-rel", revision: 1, kind: "relationship", statement: "Sam reports to Kim.", lifecycle: "current", inferred: false, modality: "asserted", valid_from: null, citation: cite(1) },
  ],
  projects: [{ project_id: "p1", name: "Pricing", context_items: 3, tasks: 1 }],
  tasks: [{ id: "t1", revision: 1, title: "Draft the FAQ", status: "open", due_date: "2026-11-10", completed_at: null }],
  meetings: [{ source_id: "src-1", title: "Pricing sync", occurred_at: "2026-11-03T15:00:00Z" }],
  links_to_review: [{ link_id: "l-1", revision: 3, mention: "Sammy", status: "inferred", role: "participant", assertion_id: "a-pref", score: 0.7, statement: "Sam prefers annual plans." }],
  merges: [],
  merge_candidates: [{ actor_id: "act-sam2", name: "Sam Lee", reason: "Both are called 'sam lee'.", identities: [{ namespace: "email", value: "sam@home.test", status: "exact" }] }],
  gaps: [], ...over,
});
const loop = (over: Partial<OpenLoop>): OpenLoop => ({
  assertion_id: "a-loop", revision: 1, statement: "Sam sends the deck.", bucket: "waiting_for", due_at: "2026-11-07T00:00:00Z", basis: "stated",
  responsible: [{ actor_id: "act-sam", name: "Sam Lee", is_me: false, resolved: true, collective: false }], recipient: null,
  task_state: "none", tasks: [], citation: cite(2), ...over,
});

type Call = { url: string; method: string; body: Record<string, unknown> };
type Reply = [number, unknown];
let host: HTMLDivElement, root: Root, calls: Call[], current: PersonView, queued: Record<string, Reply[]>;
const P = "/api/v1/context/people/act-sam";
beforeEach(() => {
  host = document.createElement("div"); document.body.append(host); root = createRoot(host);
  calls = []; current = person(); queued = {};
  vi.stubGlobal("fetch", vi.fn(async (url: string, init: RequestInit = {}) => {
    const method = (init.method ?? "GET").toUpperCase(), path = url.split("?")[0];
    calls.push({ url, method, body: init.body ? JSON.parse(String(init.body)) : {} });
    const reply: Reply = queued[method + " " + path]?.shift()
      ?? (method === "GET" && path === P ? [200, current]
        : method === "GET" && path === "/api/v1/context/people/act-sam2" ? [200, person({ actor: { ...current.actor, id: "act-sam2" }, merge_candidates: [] })]
        : path === "/api/v1/context/brief" ? [200, { basis: "stored", stale: false, built_at: null, gaps: [], sections: [
          { name: "current_state_and_why", claims: [{ text: "Sam owns pricing.", assertion_ids: ["a-pref"], citations: [{ assertion_id: "a-pref", source_id: "src-1", segment: 0 }] }] }] }]
        : path === "/api/v1/context/open-loops" ? [200, { gaps: [], items: [loop({}),
          loop({ assertion_id: "a-own", statement: "I review Sam's FAQ.", bucket: "own", task_state: "covered",
            responsible: [{ actor_id: "me", name: "owner", is_me: true, resolved: true, collective: false }],
            tasks: [{ id: "t9", revision: 1, title: "Review FAQ", status: "completed", due_date: null, completed_at: null }] })] }]
        : method === "GET" && path === "/api/v1/context/people" ? [200, { items: [{ id: "act-kim", name: "Kim", is_me: false, identities: [] }] }]
        : method === "GET" && path.startsWith("/api/v1/context/evidence/") ? [200, { id: "a-pref", gaps: [], evidence: [
          { evidence_id: "e1", role: "support", origin: "independent", source_id: "src-1", source_title: "Pricing sync", occurred_at: null,
            segment: 0, speaker: "Sam", quote: "Annual is better for us.", current_revision: true }] }]
        : method === "POST" ? [200, { person: current }] : [404, { error: { code: "NOT_FOUND", message: "Not found." } }]);
    return new Response(JSON.stringify(reply[1]), { status: reply[0] });
  }));
});
afterEach(() => { act(() => root.unmount()); host.remove(); vi.unstubAllGlobals(); });

async function settle() { for (let i = 0; i < 6; i++) await act(async () => { await Promise.resolve(); }); }
async function render(props: Partial<React.ComponentProps<typeof PersonPage>> = {}) {
  await act(async () => { root.render(<PersonPage actorId="act-sam" onMeetings={() => {}} onMeeting={() => {}} {...props}/>); });
  await settle();
}
const text = () => host.textContent ?? "";
const section = (label: string) => host.querySelector<HTMLElement>(`section[aria-label="${label}"]`)!;
const button = (label: string, scope: ParentNode = host) => [...scope.querySelectorAll("button")].find(b => b.textContent?.trim().startsWith(label))!;
const click = async (el: HTMLElement) => { await act(async () => { el.click(); }); await settle(); };
const posts = (suffix: string) => calls.filter(c => c.method === "POST" && c.url === P + suffix);
function type(el: HTMLInputElement, value: string) {
  const setter = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(el), "value")!.set!;
  act(() => { setter.call(el, value); el.dispatchEvent(new Event("input", { bubbles: true })); });
}

describe("PersonPage", () => {
  it("shows current context, open work by direction, projects, observations, meetings and identity from the context API only", async () => {
    await render();
    expect(host.querySelector("h2")?.textContent).toBe("Sam Lee");
    expect(section("Current context").textContent).toContain("Sam owns pricing.");
    const work = section("Open work");
    expect(work.querySelector('[aria-label="Waiting for"]')?.textContent).toContain("Sam sends the deck.");
    expect(work.querySelector('[aria-label="You promised"]')?.textContent).toContain("Task done");
    expect(work.querySelector('[aria-label="You promised"] del')?.textContent).toBe("Review FAQ");
    expect(section("Projects and relationships").textContent).toContain("Pricing");
    expect(section("Projects and relationships").textContent).toContain("Sam reports to Kim.");
    expect(section("Work and observations").textContent).toContain("Sam prefers annual plans.");
    expect(section("Work and observations").textContent).toContain("Draft the FAQ");
    expect(section("Work and observations").querySelector(".badge-inferred")).not.toBeNull();
    expect(section("Identity").textContent).toContain("sam@acme.test");
    expect(section("Identity").querySelectorAll(".badge-inferred").length).toBe(2); // the alias and the mention
    expect(calls.every(c => c.url.startsWith("/api/v1/context/"))).toBe(true);
    const loopsCall = calls.find(c => c.url === "/api/v1/context/open-loops")!;
    expect(loopsCall.body).toEqual({ entity: { entity_type: "actor", entity_id: "act-sam" } });
  });

  it("drills into evidence and opens meetings", async () => {
    const onMeeting = vi.fn();
    await render({ onMeeting });
    const row = section("Work and observations").querySelector(".context-row")!;
    await click(button("Evidence", row));
    expect(calls.some(c => c.url === "/api/v1/context/evidence/a-pref")).toBe(true);
    expect(row.textContent).toContain("Annual is better for us.");
    await click(button("Pricing sync", row.querySelector(".context-evidence")!));
    expect(onMeeting).toHaveBeenLastCalledWith("src-1");
    await click(button("Pricing sync", section("Meetings")));
    expect(onMeeting).toHaveBeenCalledTimes(2);
  });

  it("confirms or rejects an inferred mention with its revision and a command ID", async () => {
    await render();
    await click(button("Confirm", section("Identity")));
    const [confirm] = posts("/links/l-1");
    expect(confirm.body).toMatchObject({ action: "confirm", expected_revision: 3 });
    expect(typeof confirm.body.command_id).toBe("string");
    expect(text()).toContain("Confirmed that this is Sam Lee.");
    queued[`POST ${P}/links/l-1`] = [[409, { error: { code: "REVISION_CONFLICT", message: "Changed." } }]];
    await click(button("Not Sam Lee", section("Identity")));
    expect(posts("/links/l-1")[1].body).toMatchObject({ action: "reject" });
    expect(host.querySelector(".meeting-stale")).not.toBeNull();
  });

  it("proposes a merge, confirms which record to keep, and follows the person kept", async () => {
    await render();
    await click(button("Same person…"));
    const form = host.querySelector<HTMLFormElement>('form[aria-label="Confirm merge"]')!;
    expect(form.textContent).toContain("You can undo this.");
    const radios = form.querySelectorAll<HTMLInputElement>("input[type=radio]");
    await click(radios[1]); // keep the other record
    queued[`POST ${P}/merge`] = [[200, { person: person({ actor: { ...current.actor, id: "act-sam2" }, merge_candidates: [],
      merges: [{ merge_id: "m-1", winner_actor_id: "act-sam2", loser_actor_id: "act-sam", winner_name: "Sam Lee", loser_name: "Sam Lee", undone_at: null, created_at: "2026-11-04T10:00:00Z" }] }) }]];
    await click(button("Merge", form));
    const [merge] = posts("/merge");
    expect(merge.body).toMatchObject({ winner_actor_id: "act-sam2", loser_actor_id: "act-sam" });
    expect(typeof merge.body.command_id).toBe("string");
    expect(calls.some(c => c.method === "GET" && c.url === "/api/v1/context/people/act-sam2")).toBe(true);
  });

  it("records 'not the same person' and finds someone else to merge with", async () => {
    await render();
    await click(button("Not the same person"));
    expect(posts("/not-same")[0].body).toMatchObject({ other_actor_id: "act-sam2" });
    const search = section("Identity").querySelector<HTMLInputElement>("input[type=search]")!;
    type(search, "kim");
    await click(button("Find", section("Identity")));
    expect(calls.some(c => c.method === "GET" && c.url.startsWith("/api/v1/context/people?") && c.url.includes("q=kim"))).toBe(true);
    await click(button("Choose", section("Identity")));
    expect(host.querySelector('form[aria-label="Confirm merge"]')?.textContent).toContain("Merge Sam Lee and Kim");
    await click(button("Merge", host.querySelector('form[aria-label="Confirm merge"]')!));
    expect(posts("/merge")[0].body).toMatchObject({ winner_actor_id: "act-sam", loser_actor_id: "act-kim" });
  });

  it("undoes a merge and points a merged-away record at the person kept", async () => {
    current = person({ actor: { ...person().actor, archived: true }, merge_candidates: [],
      merges: [{ merge_id: "m-1", winner_actor_id: "act-sam2", loser_actor_id: "act-sam", winner_name: "Samuel Lee", loser_name: "Sam Lee", undone_at: null, created_at: "2026-11-04T10:00:00Z" }] });
    await render();
    expect(text()).toContain("This record was merged into");
    expect(section("Identity").querySelector('[aria-label="Possible duplicates"]')).toBeNull();
    await click(button("Unmerge"));
    expect(posts("/unmerge")[0].body).toMatchObject({ merge_id: "m-1" });
    expect(text()).toContain("Merge undone.");
  });

  it("says when meeting memory is off and asks for a person when none is chosen", async () => {
    const onMeetings = vi.fn();
    await render({ actorId: null, onMeetings });
    expect(text()).toContain("Choose a person");
    await click(button("Open Meetings"));
    expect(onMeetings).toHaveBeenCalled();
    queued[`GET ${P}`] = [[403, { error: { code: "CONTEXT_DISABLED", message: "Off." } }]];
    await render();
    expect(text()).toContain("Meeting memory is not on");
  });
});
