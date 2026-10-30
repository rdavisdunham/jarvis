import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { HostSelfCheck, SetupCodeBox, TargetCard } from "./RemoteaSetup";
import { harnessText, hostLine, mcpText, selfCheck, setupSteps, TOO_OLD, verifyText, withProbe, type SetupHost, type SetupReport, type SetupTarget } from "./remotea-setup";

const host: SetupHost = {
  id: "h1", name: "desk", connected: true, last_seen_at: "2026-10-30T10:00:00Z", hello_at: "2026-10-30T09:59:00Z",
  version: "0.1.0", build: "76c5f62", build_dirty: false, caps_version: 2, degraded_reason: null, problems: [],
};
const target = (over: Partial<SetupTarget> = {}): SetupTarget => ({
  key: "t1", host_id: "h1", host_name: "desk", host_target: "jarvis", name: "jarvis", repo: "/home/davin/jarvis",
  execution_target_id: "t1", advertised: true, harnesses: {}, verify: [], mcp: { configured: true }, problems: [], notes: [],
  ready: true, line: null, ...over,
});

describe("setupSteps", () => {
  it("asks for a server when none has connected", () => {
    const [h, t, r] = setupSteps({ ready: false, hosts: [], targets: [], lines: [] });
    expect(h.tone).toBe("todo");
    expect(h.detail).toContain("remotea eridani setup");
    expect(t.tone).toBe("unknown");
    expect(r.tone).toBe("unknown");
  });

  it("walks host, targets and readiness", () => {
    const report: SetupReport = { ready: true, hosts: [host], lines: [], targets: [
      target(), target({ key: "t2", name: "remotea", ready: false, problems: [{ code: "MCP_NOT_CONFIGURED", message: "m", fix: "f", blocks: true }] }),
    ] };
    const [h, t, r] = setupSteps(report);
    expect(h).toMatchObject({ tone: "done", title: "remotea server connected" });
    expect(h.detail).toBe("desk · remotea 0.1.0 · build 76c5f62 · caps v2");
    expect(t).toMatchObject({ tone: "done", title: "2 targets advertised", detail: "jarvis, remotea" });
    expect(r).toMatchObject({ tone: "warn", title: "Ready to run", detail: "jarvis; not ready: remotea" });
  });

  it("says an offline or degraded host needs attention", () => {
    const offline = setupSteps({ ready: false, hosts: [{ ...host, connected: false }], targets: [], lines: [] }, Date.parse("2026-10-30T10:05:00Z"));
    expect(offline[0].title).toBe("remotea server offline");
    expect(offline[0].detail).toMatch(/^desk: last seen/);
    const degraded = setupSteps({ ready: false, lines: [], targets: [],
      hosts: [{ ...host, problems: [{ code: "HOST_DEGRADED", message: "Host journal degraded: ENOSPC", fix: "free space", blocks: true }] }] });
    expect(degraded[0]).toMatchObject({ tone: "warn", detail: "Host journal degraded: ENOSPC" });
    expect(degraded[2].title).toBe("No target can start a run yet");
  });
});

describe("readiness lines", () => {
  it("names each harness state, including an older host's unknowns", () => {
    expect(harnessText({ label: "Claude Code", ready: true, enabled: true, version: "2.1.312", error: null, launch: "probed", reason: null }))
      .toBe("Claude Code: ready (2.1.312)");
    expect(harnessText({ label: "Codex", ready: false, enabled: true, version: null, error: "codex: command not found", launch: "probed", reason: "x" }))
      .toBe("Codex: not installed: codex: command not found");
    expect(harnessText({ label: "Codex", ready: false, enabled: false, version: null, error: null, launch: "not enabled", reason: "x" }))
      .toBe("Codex: not enabled");
    expect(harnessText({ label: "Codex", ready: false, enabled: null, version: null, error: null, launch: TOO_OLD, reason: "The jarvis target does not enable Codex." }))
      .toBe(`Codex: The jarvis target does not enable Codex. (launch probe ${TOO_OLD})`);
  });

  it("names verify readiness and isolation, and MCP", () => {
    expect(verifyText({ id: "tests", ready: false, reason: "its program is not on the service PATH" }))
      .toBe("tests: not ready: its program is not on the service PATH");
    expect(verifyText({ id: "tests", ready: true, isolated: false, isolation_reason: "bwrap is not installed" }))
      .toBe("tests: ready · not sandboxed: bwrap is not installed");
    expect(verifyText({ id: "lint", ready: true, isolated: true })).toBe("lint: ready · sandboxed");
    expect(mcpText(null)).toBe(`Eridani tools for agents: ${TOO_OLD}`);
    expect(mcpText({ configured: false, reason: "no key file" })).toBe("Eridani tools for agents: missing (no key file)");
    expect(hostLine({ name: "old", version: null, build: null, caps_version: null })).toBe("old · caps unknown (host too old)");
  });

  it("replaces a probed target in place", () => {
    const report: SetupReport = { ready: false, hosts: [host], lines: [], targets: [target({ ready: false }), target({ key: "t2" })] };
    const next = withProbe(report, { source: "hello", hello_at: null, note: "n", host, target: target({ ready: true }) });
    expect(next.targets.map(t => [t.key, t.ready])).toEqual([["t1", true], ["t2", true]]);
  });
});

describe("setup views", () => {
  it("shows the setup code once, masked, with how to use it", () => {
    const html = renderToStaticMarkup(<SetupCodeBox code="eridani-setup:1:abc" onDone={() => {}}/>);
    expect(html).toContain('type="password"');
    expect(html).toContain("Run <code>remotea eridani setup</code> in");
    expect(html).toContain("paste this code when asked");
    expect(html).toContain("<code>remotea eridani setup --code -</code>");
    expect(html).toContain("Copy setup code");
  });

  it("lists a target's facts, each problem with its fix, and the probe button", () => {
    const html = renderToStaticMarkup(<TargetCard onProbed={() => {}} target={target({
      ready: false, verify: null, mcp: null, notes: [`Verify readiness: ${TOO_OLD}.`],
      harnesses: { codex: { label: "Codex", ready: false, enabled: true, version: null, error: "codex: command not found", launch: "probed", reason: "r" } },
      problems: [{ code: "HARNESS_NOT_INSTALLED", message: "Codex is enabled but not installed: r", fix: "Install Codex", blocks: false }],
    })}/>);
    expect(html).toContain("Not ready");
    expect(html).toContain("Codex: not installed: codex: command not found");
    expect(html).toContain(`Verify: ${TOO_OLD}`);
    expect(html).toContain(`Eridani tools for agents: ${TOO_OLD}`);
    expect(html).toContain('class="warns"><strong>Codex is enabled but not installed: r</strong><span>Install Codex</span>');
    expect(html).toContain("Run a readiness probe");
  });
});

describe("host self-check (remotea doctor)", () => {
  const now = Date.parse("2026-10-30T10:05:00Z");
  const doctor = { at: "2026-10-30T10:00:00+00:00", received_at: "2026-10-30T10:00:01+00:00", ok: 30, warn: 1, fail: 2, items: [
    { id: "service.active", level: "fail" as const, summary: "remotea.service is not running", fix: "systemctl --user start remotea" },
    { id: "target.jarvis.verify.test", level: "fail" as const, summary: "verify test: uv not on PATH" },
    { id: "hooks.codex.trust", level: "warn" as const, summary: "Codex hooks are not trusted", fix: "Run codex once" },
  ] };

  it("summarizes counts and age, or why there is no report", () => {
    expect(selfCheck({ doctor, doctor_supported: true }, now)).toEqual({ tone: "todo", headline: "2 failing · 1 warning · 30 ok", age: "checked 5m ago", note: null });
    expect(selfCheck({ doctor: { ...doctor, fail: 0, warn: 0, items: [] } }, now).headline).toBe("All 30 checks ok");
    expect(selfCheck({ doctor: { ...doctor, truncated: true } }, now).note).toContain("remotea doctor");
    expect(selfCheck({ doctor: null, doctor_supported: false }).headline).toBe(TOO_OLD);
    expect(selfCheck({}).headline).toBe(TOO_OLD);
    expect(selfCheck({ doctor: null, doctor_supported: true }).headline).toBe("Not reported yet");
  });

  it("renders each warn and fail line with its fix", () => {
    const html = renderToStaticMarkup(<HostSelfCheck host={{ ...host, doctor, doctor_supported: true }} now={now}/>);
    expect(html).toContain("<strong>Host self-check</strong> <span>2 failing · 1 warning · 30 ok</span>");
    expect(html).toContain("checked 5m ago");
    expect(html).toContain('class="blocks"><strong><span class="remotea-check-level">Fail</span> remotea.service is not running</strong><span>systemctl --user start remotea</span>');
    expect(html).toContain('class="warns"><strong><span class="remotea-check-level">Warn</span> Codex hooks are not trusted</strong><span>Run codex once</span>');
    expect(html).toContain("<code>target.jarvis.verify.test</code>");
    const old = renderToStaticMarkup(<HostSelfCheck host={host}/>);
    expect(old).toContain(`<span>${TOO_OLD}</span>`);
    expect(old).not.toContain("<ul");
  });
});
