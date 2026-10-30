// The guided remotea setup page (Settings → Connections → remotea, and the remotea page while nothing is ready):
// `GET /remotea/setup` and `POST /remotea/targets/{key}/probe` (agent_control.setup_report / target_probe).
import { ago, type DoctorSummary } from "./remotea-model";

export type SetupProblem = { code: string; message: string; fix: string; blocks: boolean };
export type SetupHost = {
  id: string; name: string; connected: boolean; last_seen_at: string | null; hello_at: string | null;
  version: string | null; build: string | null; build_dirty: boolean | null; caps_version: number | null;
  degraded_reason: string | null; problems: SetupProblem[];
  /** `remotea doctor`'s summary (null: none received); `doctor_supported`: the host advertises `doctor`. Both
   * absent from an Eridani without the doctor mirror. */
  doctor?: DoctorSummary | null; doctor_supported?: boolean;
};
export type SetupHarness = {
  label: string; ready: boolean; enabled: boolean | null; version: string | null; error: string | null;
  /** "probed", "not probed" (enabled, no probe reported), "not enabled", or "unknown (host too old)". */
  launch: string; reason: string | null;
};
export type SetupVerify = { id: string; ready: boolean; reason?: string; isolated?: boolean; isolation?: "bwrap" | "none"; isolation_reason?: string };
export type SetupTarget = {
  key: string; host_id: string | null; host_name: string | null; host_target: string; name: string; repo: string;
  execution_target_id: string | null; advertised: boolean; harnesses: Record<string, SetupHarness>;
  verify: SetupVerify[] | null; mcp: { configured: boolean; reason?: string } | null;
  problems: SetupProblem[]; notes: string[]; ready: boolean; line: string | null;
};
export type SetupReport = { ready: boolean; hosts: SetupHost[]; targets: SetupTarget[]; lines: string[] };
export type ProbeResult = { source: "hello" | string; hello_at: string | null; note: string; host: SetupHost | null; target: SetupTarget };

export const SETUP_COMMAND = "remotea eridani setup";
export const SETUP_STDIN = "remotea eridani setup --code -";
export const TOO_OLD = "unknown (host too old)";

export type StepTone = "done" | "todo" | "warn" | "unknown";
export type SetupStep = { id: "host" | "targets" | "ready"; title: string; tone: StepTone; detail: string };

/** The three steps the page walks: a host connected and healthy, targets advertised, a target ready to run. */
export function setupSteps(report: SetupReport, now = Date.now()): SetupStep[] {
  const live = report.hosts.filter(h => h.connected);
  const healthy = live.filter(h => !h.problems.length);
  const host: SetupStep = !report.hosts.length
    ? { id: "host", title: "Connect a remotea server", tone: "todo", detail: `No remotea server has connected yet. Run \`${SETUP_COMMAND}\` in WSL and paste a setup code from below.` }
    : healthy.length
      ? { id: "host", title: "remotea server connected", tone: "done", detail: healthy.map(hostLine).join("; ") }
      : live.length
        ? { id: "host", title: "remotea server needs attention", tone: "warn", detail: live.map(h => h.problems[0]?.message ?? "").join("; ") }
        : { id: "host", title: "remotea server offline", tone: "todo",
            detail: report.hosts.map(h => `${h.name}: last seen ${ago(h.last_seen_at, now)}`).join("; ") };
  const advertised = report.targets.filter(t => t.advertised);
  const targets: SetupStep = advertised.length
    ? { id: "targets", title: `${advertised.length} ${advertised.length === 1 ? "target" : "targets"} advertised`, tone: "done",
        detail: advertised.map(t => t.name).join(", ") }
    : { id: "targets", title: "No target advertised", tone: live.length ? "todo" : "unknown",
        detail: live.length ? "Add a repository: `remotea eridani setup --scan <folder>` in WSL, or targets[] in ~/.config/remotea/eridani.json."
          : "Shown once a remotea server connects." };
  const ready = report.targets.filter(t => t.ready && t.advertised);
  const unready = report.targets.filter(t => !t.ready);
  const run: SetupStep = ready.length && healthy.length
    ? { id: "ready", title: "Ready to run", tone: unready.length ? "warn" : "done",
        detail: ready.map(t => t.name).join(", ") + (unready.length ? `; not ready: ${unready.map(t => t.name).join(", ")}` : "") }
    : { id: "ready", title: "No target can start a run yet", tone: advertised.length ? "todo" : "unknown",
        detail: unready.length ? "See the problems below." : "" };
  return [host, targets, run];
}

/** "remotea 0.1.0 · build 76c5f62 · caps v2" for a host, or its name alone. */
export function hostLine(host: Pick<SetupHost, "name" | "version" | "build" | "caps_version">) {
  return [host.name, host.version ? "remotea " + host.version : "", host.build ? "build " + host.build : "",
    host.caps_version ? "caps v" + host.caps_version : "caps unknown (host too old)"].filter(Boolean).join(" · ");
}

/** One harness's state: ready with its version, enabled but broken, not enabled, or unknown on an older host. */
export function harnessText(h: SetupHarness) {
  if (h.ready) return `${h.label}: ready${h.version ? " (" + h.version + ")" : ""}`;
  if (h.enabled === false) return `${h.label}: not enabled`;
  if (h.enabled === null && h.launch === TOO_OLD) return `${h.label}: ${h.reason ?? "cannot start"} (launch probe ${TOO_OLD})`;
  return `${h.label}: ${h.error ? "not installed: " + h.error : h.reason ?? "cannot start"}`;
}

/** One verify entry: ready or why not, and whether it runs in the sandbox (when the host says). */
export function verifyText(v: SetupVerify) {
  const sandbox = v.isolated === true ? " · sandboxed" : v.isolated === false
    ? ` · not sandboxed${v.isolation_reason ? ": " + v.isolation_reason : ""}` : "";
  return `${v.id}: ${v.ready ? "ready" : "not ready" + (v.reason ? ": " + v.reason : "")}${sandbox}`;
}

export function mcpText(mcp: SetupTarget["mcp"]) {
  if (!mcp) return `Eridani tools for agents: ${TOO_OLD}`;
  return mcp.configured ? "Eridani tools for agents: configured" : `Eridani tools for agents: missing${mcp.reason ? " (" + mcp.reason + ")" : ""}`;
}

export type SelfCheck = { tone: "done" | "warn" | "todo" | "unknown"; headline: string; age: string | null; note: string | null };

/** The "Host self-check" block's header: counts and the report's age, or why there is none. */
export function selfCheck(host: Pick<SetupHost, "doctor" | "doctor_supported">, now = Date.now()): SelfCheck {
  const d = host.doctor;
  if (!d) {
    return host.doctor_supported
      ? { tone: "unknown", headline: "Not reported yet", age: null, note: "remotea checks itself every 5 minutes; `remotea doctor` in WSL shows it now." }
      : { tone: "unknown", headline: TOO_OLD, age: null, note: "Update remotea to see its self-check here; `remotea doctor` needs a current remotea too." };
  }
  const parts = [d.fail ? `${d.fail} failing` : "", d.warn ? `${d.warn} ${d.warn === 1 ? "warning" : "warnings"}` : "", `${d.ok} ok`].filter(Boolean);
  return {
    tone: d.fail ? "todo" : d.warn ? "warn" : "done",
    headline: d.fail || d.warn ? parts.join(" · ") : `All ${d.ok} checks ok`,
    age: `checked ${ago(d.at, now)}`,
    note: d.truncated ? "More findings than shown: run `remotea doctor` in WSL for the full list." : null,
  };
}

/** Replace one target in the report with a probe's fresh copy (matched by its key). */
export function withProbe(report: SetupReport, probe: ProbeResult): SetupReport {
  return { ...report, targets: report.targets.map(t => (t.key === probe.target.key ? probe.target : t)) };
}

/** After opening Settings → Connections, scroll the guided setup view (RemoteaReadiness) into sight once the lazy
 * settings chunk has rendered it; gives up quietly after about two seconds. */
export function revealSetup(tries = 20) {
  const view = document.querySelector('section[aria-label="remotea setup"]');
  if (view) view.scrollIntoView({ block: "start" });
  else if (tries > 0) window.setTimeout(() => revealSetup(tries - 1), 100);
}
