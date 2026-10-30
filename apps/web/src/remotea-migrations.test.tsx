import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { MigrationRow } from "./Remotea";
import { migrationBody, migrationDraft, migrationSummary, type MigrationTarget } from "./remotea-model";

const now = Date.parse("2026-10-28T12:00:00Z");
const target = (values: Partial<MigrationTarget> = {}): MigrationTarget => ({
  host_id: "h", host_name: "desk", host_target: "jarvis", configured: true, source: "host", dir: "migrations/versions",
  pattern: null, host_dir: "migrations/versions", head: 32, override: null, last_reserved: null, next: 33,
  reported_at: "2026-10-28T11:58:00Z", ...values,
});
const blank = { dir: "", pattern: "", override: "" };

describe("migration settings", () => {
  it("renders the folder, pattern and override as editable fields", () => {
    const html = renderToStaticMarkup(<MigrationRow item={target()} readOnly={false} busy={false} onSave={() => {}}/>);
    expect(html).toContain('name="dir"');
    expect(html).toContain('name="pattern"');
    expect(html).toContain('name="override"');
    expect(html).toContain(">Save</button>");
    expect(html).toContain("migrations/versions (from the host)");
    const read = renderToStaticMarkup(<MigrationRow item={target()} readOnly busy={false} onSave={() => {}}/>);
    expect(read).not.toContain(">Save</button>");
    expect(read).toContain("disabled");
  });

  it("starts from Eridani's own values, blank when the host's config applies", () => {
    expect(migrationDraft(target())).toEqual(blank);
    expect(migrationDraft(target({ source: "eridani", dir: "db/migrations", pattern: "^(\\d{3})-", override: 7 })))
      .toEqual({ dir: "db/migrations", pattern: "^(\\d{3})-", override: "7" });
  });

  it("builds the save body and refuses what the server would", () => {
    expect(migrationBody(target(), { dir: " db/migrations ", pattern: "^(\\d{3})-", override: "40" }))
      .toEqual({ host_id: "h", host_target: "jarvis", dir: "db/migrations", pattern: "^(\\d{3})-", override: 40 });
    expect(migrationBody(target(), blank)).toEqual({ host_id: "h", host_target: "jarvis", dir: null, pattern: null, override: null });
    expect(migrationBody(target(), { ...blank, dir: "/abs" })).toMatch(/relative/);
    expect(migrationBody(target(), { ...blank, dir: "a/../b" })).toMatch(/relative/);
    expect(migrationBody(target(), { ...blank, pattern: "^(\\d)" })).toMatch(/together/);
    expect(migrationBody(target(), { ...blank, dir: "m", pattern: "^\\d+_" })).toMatch(/group/);
    expect(migrationBody(target(), { ...blank, dir: "m", pattern: "(" })).toMatch(/not a valid/);
    expect(migrationBody(target(), { ...blank, override: "x1" })).toMatch(/migration number/);
  });

  it("summarizes where the numbers come from", () => {
    expect(migrationSummary(target(), now)).toBe("migrations/versions (from the host) · head 0032 on the host (2m ago) · next 0033");
    expect(migrationSummary(target({ source: "eridani", dir: "alembic/versions", head: null, override: 40, last_reserved: 41, next: 42 }), now))
      .toBe("alembic/versions · the host reports migrations/versions, not this folder · override 0040 · last reserved 0041 · next 0042");
    expect(migrationSummary(target({ configured: false, source: null, dir: null, host_dir: null, head: null, next: null }), now)).toMatch(/^Off:/);
  });
});
