// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { importOutcome, importParams, SourcesSettings, type SourcesOverview } from "./SourcesSettings";
import { processingLine, type ProcessingStatus } from "./MeetingMemorySettings";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const overview = (granola: Partial<SourcesOverview["granola"]> = {}): SourcesOverview => ({
  adapters: [{ key: "upload", description: "", live: true }, { key: "api", description: "", live: true }],
  deliveries: [{ connector: "api", label: "recorder", sources: 3, last_delivery_at: "2026-11-03T15:00:00Z" }],
  submission: { method: "PUT", url: "https://eri.example/api/v1/external/sources/{connector}/{external_id}",
    scope: "sources:write", labels: ["recorder", "zoom", "granola", "other"],
    bridge: "ERIDANI_SOURCES_KEY=<key> python scripts/bridge_submit.py ./exports --base https://eri.example --connector recorder" },
  limits: { max_source_chars: 400000, max_segments: 20000, max_body_bytes: 900000, formats: ["txt", "markdown", "vtt", "srt", "json"],
    extensions: [".json", ".md", ".srt", ".txt", ".vtt"] },
  granola: { enabled: false, connections: [], ...granola },
  preferences: { synthesize_briefs: false, semantic_supersede: false, lint_budget_usd: 0.5, review_destination: "eridani" },
  preference_limits: { lint_budget_usd: { min: 0, max: 5, default: 0.5, per_call: 0.05 }, review_destination: ["eridani", "linear"] },
});

type Call = { url: string; init: RequestInit };
const running: ProcessingStatus = { epoch: 2, paused: false, paused_at: null, queue: { jobs: {}, runs: { queued: 1, running: 1, succeeded: 9 } } };
let host: HTMLDivElement, root: Root, calls: Call[], replies: ((call: Call) => [number, unknown])[], current: SourcesOverview;
beforeEach(() => {
  host = document.createElement("div"); document.body.append(host); root = createRoot(host);
  calls = []; replies = []; current = overview();
  vi.stubGlobal("fetch", vi.fn(async (url: string, init: RequestInit = {}) => {
    const call = { url, init }; calls.push(call);
    const [status, body] = (replies.shift() ?? (() => [200, url.endsWith("/context/admin/status") ? running : current]))(call);
    return new Response(JSON.stringify(body), { status });
  }));
});
afterEach(() => { act(() => root.unmount()); host.remove(); vi.unstubAllGlobals(); });

async function render(node: React.ReactNode) {
  await act(async () => { root.render(node); });
  await act(async () => { await Promise.resolve(); });
}
const text = () => host.textContent ?? "";
const button = (label: string) => [...host.querySelectorAll("button")].find(b => b.textContent?.includes(label))!;

describe("SourcesSettings", () => {
  it("shows file import and the submission endpoint as live sources without Granola", async () => {
    await render(<SourcesSettings/>);
    expect(calls[0].url).toBe("/api/v1/context/settings");
    expect(text()).toContain("Meeting sources");
    expect(text()).toContain("Windows recorder");
    expect(text()).toContain("3 meetings");
    expect(host.querySelector<HTMLInputElement>("input[type=file]")!.accept).toBe(".json,.md,.srt,.txt,.vtt");
    const endpoint = host.querySelector<HTMLInputElement>("input[aria-label=Endpoint]")!;
    expect(endpoint.value).toBe("PUT https://eri.example/api/v1/external/sources/{connector}/{external_id}");
    expect(host.querySelector<HTMLInputElement>("input[aria-label='Bridge script']")!.value).toContain("bridge_submit.py");
    expect(text()).toContain("Connected agents");
    expect(text()).not.toContain("Granola API");
    expect(host.querySelector("input[aria-label='Granola API key']")).toBeNull();
  });

  it("stays hidden when meeting context is off or the workspace is shared", async () => {
    for (const code of ["CONTEXT_DISABLED", "PERSONAL_WORKSPACE"]) {
      replies.push(() => [403, { error: { code, message: "no" } }]);
      await render(<SourcesSettings key={code}/>);
      expect(host.innerHTML).toBe("");
    }
  });

  it("shows the Granola key form only when the server enables the adapter, and never echoes the key", async () => {
    current = overview({ enabled: true });
    await render(<SourcesSettings initial={current}/>);
    expect(text()).toContain("Granola API (optional)");
    const [email, key] = [host.querySelector<HTMLInputElement>("input[type=email]")!, host.querySelector<HTMLInputElement>("input[aria-label='Granola API key']")!];
    expect(key.type).toBe("password");
    await act(async () => {
      for (const [input, value] of [[email, "davin@example.com"], [key, "test-granola-key"]] as const) {
        Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!.call(input, value);
        input.dispatchEvent(new Event("input", { bubbles: true }));
      }
    });
    replies.push(() => [400, { error: { code: "GRANOLA_NOT_ENTITLED", message: "Granola's API needs a Business or Enterprise workspace." } }]);
    await act(async () => { button("Connect Granola").click(); });
    await act(async () => { await Promise.resolve(); });
    const sent = calls.find(c => c.url === "/api/v1/context/settings/granola")!;
    expect(JSON.parse(String(sent.init.body))).toEqual({ account: "davin@example.com", api_key: "test-granola-key" });
    expect(text()).toContain("Business or Enterprise");
    expect(text()).not.toContain("test-granola-key");
  });

  it("shows a connection's health and queues a sync", async () => {
    const connection = { id: "c1", connector: "granola" as const, account: "davin@example.com", has_key: true, enabled: true,
      health: "auth_failed", last_error: "401 from Granola", last_success_at: null, created_at: null };
    current = overview({ enabled: true, connections: [connection] });
    await render(<SourcesSettings initial={current}/>);
    expect(text()).toContain("Key rejected");
    expect(text()).toContain("401 from Granola");
    expect(text()).toContain("Not yet synced");
    replies.push(() => [200, { job_id: "j", status: "queued", already_queued: false }]);
    await act(async () => { button("Sync now").click(); });
    await act(async () => { await Promise.resolve(); });
    expect(calls.some(c => c.url === "/api/v1/context/settings/granola/c1/sync" && c.init.method === "POST")).toBe(true);
    expect(text()).toContain("Sync queued.");
  });
});

describe("import helpers", () => {
  it("sends the meeting time as an instant with the browser zone", () => {
    const params = new URLSearchParams(importParams({ name: "standup.vtt" }, " Standup ", "2026-11-03T10:00", "America/New_York"));
    expect(params.get("filename")).toBe("standup.vtt");
    expect(params.get("title")).toBe("Standup");
    expect(params.get("occurred_at")).toBe(new Date("2026-11-03T10:00").toISOString());
    expect(params.get("timezone")).toBe("America/New_York");
    expect(new URLSearchParams(importParams({ name: "e.json" }, "", "", "")).has("occurred_at")).toBe(false);
  });
  it("describes each ingestion outcome", () => {
    const base = { id: "s", external_id: "x", title: "Retro", format: "md", revision: 1, processing_status: "queued" };
    expect(importOutcome({ ...base, ingestion_status: "created" })).toBe("Imported “Retro”.");
    expect(importOutcome({ ...base, ingestion_status: "none" })).toContain("already imported");
  });
});

describe("Meeting memory", () => {
  const input = (label: string) => [...host.querySelectorAll<HTMLElement>(".setting-row, label.field")]
    .find(row => row.textContent?.includes(label))!.querySelector<HTMLInputElement | HTMLSelectElement>("input, select")!;
  const sent = (url: string) => calls.filter(c => c.url === url).map(c => ({ method: c.init.method, body: c.init.body && JSON.parse(String(c.init.body)) }));

  it("shows the processing status and current preferences, without an engine or rollback control", async () => {
    await render(<SourcesSettings/>);
    expect(calls.map(c => c.url)).toContain("/api/v1/context/admin/status");
    expect(text()).toContain("Meeting memory");
    expect(text()).toContain("Running · epoch 2 · 2 runs outstanding");
    expect((input("Model-written briefs") as HTMLInputElement).checked).toBe(false);
    expect(text()).toContain("one model call per changed person or project");
    expect((input("Weekly cleanup budget") as HTMLInputElement).value).toBe("0.5");
    expect((input("Approved tasks go to") as HTMLSelectElement).value).toBe("eridani");
    expect(text()).not.toMatch(/engine|rollback/i);
  });

  it("saves a switch through PATCH and pauses through the admin route", async () => {
    await render(<SourcesSettings/>);
    replies.push(() => [200, { preferences: { ...current.preferences, synthesize_briefs: true } }]);
    await act(async () => { (input("Model-written briefs") as HTMLInputElement).click(); });
    await act(async () => { await Promise.resolve(); });
    expect(sent("/api/v1/context/settings/preferences")).toEqual([{ method: "PATCH", body: { synthesize_briefs: true } }]);
    expect((input("Model-written briefs") as HTMLInputElement).checked).toBe(true);
    replies.push(() => [200, { ...running, paused: true, epoch: 3, paused_at: "2026-11-04T12:00:00Z" }]);
    await act(async () => { button("Pause").click(); });
    await act(async () => { await Promise.resolve(); });
    expect(sent("/api/v1/context/admin/pause")).toEqual([{ method: "POST", body: {} }]);
    expect(text()).toContain("Paused since");
    expect(text()).toContain("epoch 3");
    expect(button("Resume")).toBeTruthy();
  });

  it("shows a refused change and keeps the saved value", async () => {
    await render(<SourcesSettings/>);
    replies.push(() => [422, { error: { code: "INVALID_ARGUMENT", message: "Input should be 'eridani' or 'linear'" } }]);
    const select = input("Approved tasks go to") as HTMLSelectElement;
    await act(async () => { select.value = "linear"; select.dispatchEvent(new Event("change", { bubbles: true })); });
    await act(async () => { await Promise.resolve(); });
    expect(text()).toContain("Input should be");
    expect(select.value).toBe("eridani");
  });

  it("describes paused and running states", () => {
    expect(processingLine({ ...running, queue: { jobs: {}, runs: { queued: 1 } } })).toBe("Running · epoch 2 · 1 run outstanding");
    expect(processingLine({ ...running, paused: true, paused_at: null })).toContain("Paused since an unknown time");
  });
});
