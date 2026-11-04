// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { importOutcome, importParams, SourcesSettings, type SourcesOverview } from "./SourcesSettings";

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
});

type Call = { url: string; init: RequestInit };
let host: HTMLDivElement, root: Root, calls: Call[], replies: ((call: Call) => [number, unknown])[], current: SourcesOverview;
beforeEach(() => {
  host = document.createElement("div"); document.body.append(host); root = createRoot(host);
  calls = []; replies = []; current = overview();
  vi.stubGlobal("fetch", vi.fn(async (url: string, init: RequestInit = {}) => {
    const call = { url, init }; calls.push(call);
    const [status, body] = (replies.shift() ?? (() => [200, current]))(call);
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
      for (const [input, value] of [[email, "davin@example.com"], [key, "gk_secret_123456"]] as const) {
        Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!.call(input, value);
        input.dispatchEvent(new Event("input", { bubbles: true }));
      }
    });
    replies.push(() => [400, { error: { code: "GRANOLA_NOT_ENTITLED", message: "Granola's API needs a Business or Enterprise workspace." } }]);
    await act(async () => { button("Connect Granola").click(); });
    await act(async () => { await Promise.resolve(); });
    const sent = calls.find(c => c.url === "/api/v1/context/settings/granola")!;
    expect(JSON.parse(String(sent.init.body))).toEqual({ account: "davin@example.com", api_key: "gk_secret_123456" });
    expect(text()).toContain("Business or Enterprise");
    expect(text()).not.toContain("gk_secret_123456");
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
