import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { ConnectionsSettings } from "./ConnectionsSettings";
import { BotKeyList, liveKey, type BotKey } from "./BotSettings";
import { readConnectionTab } from "./connection-tabs";
import { readSettingsSection } from "./settings-sections";

const link = (search: string) => new URLSearchParams(search);

describe("Connections deep links", () => {
  it("reads the tab from connection=, a tab id in section=, or Google's OAuth return", () => {
    expect(readConnectionTab(link("?view=settings&section=integrations&connection=linear"))).toBe("linear");
    expect(readConnectionTab(link("?view=settings&section=remotea"))).toBe("remotea");
    expect(readConnectionTab(link("?view=settings&section=integrations&google=connected"))).toBe("google");
    expect(readConnectionTab(link("?view=settings&section=integrations&connection=nope"))).toBe("agents");
    expect(readConnectionTab(link("?view=settings&section=integrations"))).toBe("agents");
  });
  it("opens Connections for a tab id given as the section", () => {
    expect(readSettingsSection("linear")).toBe("integrations");
    expect(readSettingsSection("meetings")).toBe("integrations");
    expect(readSettingsSection("integrations")).toBe("integrations");
    expect(readSettingsSection("system")).toBe("system");
    expect(readSettingsSection("unknown")).toBe("profile");
  });
});

describe("Connections tabs", () => {
  const panels = { agents: <p>agent keys panel</p>, remotea: <p>remotea panel</p>, linear: <p>linear panel</p>, google: <p>google panel</p> };
  it("shows one tab's panel at a time", () => {
    const html = renderToStaticMarkup(<ConnectionsSettings tab="linear" onTab={() => {}} panels={panels}/>);
    expect(html).toContain('role="tablist" aria-label="Connections"');
    expect(html).toContain('id="connection-tab-linear" role="tab"');
    expect(html).toMatch(/id="connection-tab-linear"[^>]*aria-selected="true"/);
    expect(html).toContain("linear panel");
    expect(html).not.toContain("agent keys panel");
    expect(html).not.toContain("google panel");
    expect(html).not.toContain("connection-tab-meetings");
  });
  it("offers only the tabs this workspace has, falling back to the first", () => {
    const shared = renderToStaticMarkup(<ConnectionsSettings tab="google" onTab={() => {}} panels={{ agents: panels.agents, remotea: panels.remotea }}/>);
    expect(shared).not.toContain("connection-tab-google");
    expect(shared).not.toContain("connection-tab-linear");
    expect(shared).toMatch(/id="connection-tab-agents"[^>]*aria-selected="true"/);
    expect(shared).toContain("agent keys panel");
    const personal = renderToStaticMarkup(<ConnectionsSettings tab="agents" onTab={() => {}} panels={panels} meetings/>);
    expect(personal).toContain(">Meetings</button>");
  });
});

describe("agent keys", () => {
  const now = Date.parse("2026-11-05T12:00:00Z");
  const key = (over: Partial<BotKey>): BotKey => ({ id: "k", name: "Codex", prefix: "eri_abc", scopes: ["tasks:write"],
    expires_at: "2027-01-01T00:00:00Z", created_at: "2026-10-01T00:00:00Z", revoked_at: null, last_used_at: null, ...over });
  it("folds revoked and expired keys into a closed disclosure", () => {
    const items = [key({ id: "live", name: "Live agent" }), key({ id: "gone", name: "Old agent", revoked_at: "2026-10-02T00:00:00Z" }),
      key({ id: "late", name: "Lapsed agent", expires_at: "2026-11-01T00:00:00Z" })];
    expect(items.map(item => liveKey(item, now))).toEqual([true, false, false]);
    const html = renderToStaticMarkup(<BotKeyList items={items} busy={false} onRevoke={() => {}} now={now}/>);
    const [open, folded] = html.split('<details class="settings-block bot-key-inactive">');
    expect(open).toContain("Live agent");
    expect(open).toContain(">Revoke</button>");
    expect(folded).toContain("<summary>Revoked and expired keys (2)</summary>");
    expect(folded).toContain("Old agent");
    expect(folded).toContain("Lapsed agent");
    expect(folded).not.toContain(">Revoke</button>");
  });
  it("says when only inactive keys remain, and has no disclosure without them", () => {
    expect(renderToStaticMarkup(<BotKeyList items={[key({ revoked_at: "2026-10-02T00:00:00Z" })]} busy={false} onRevoke={() => {}} now={now}/>))
      .toContain("No active agent keys.");
    const none = renderToStaticMarkup(<BotKeyList items={[]} busy={false} onRevoke={() => {}} now={now}/>);
    expect(none).toContain("No agents connected yet.");
    expect(none).not.toContain("<details");
  });
});
