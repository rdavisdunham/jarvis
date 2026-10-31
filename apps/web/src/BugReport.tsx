/**
 * The Report a bug button: floats bottom-left on every page (shell.css `.bug-report-launcher`), takes a
 * screenshot of what was on screen before its dialog opens, and files the report with the page's
 * diagnostics (bug-report.ts) to POST /api/v1/bug-reports, which also opens a task in "Eridani bug
 * reports". Shortcut: Alt+Shift+B.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { Bug, Check, ChevronRight, X } from "lucide-react";
import { ApiError, post } from "./api";
import { collectContext, type BugContext, type PageState } from "./bug-report";
import { Dialog } from "./ux";

export type BugReportResult = { id: string; task_id: string | null; title: string };
type Shot = { url: string | null; error: string | null };

/** Alt+Shift+B (by key position, so macOS Option combinations work too). */
export const isReportShortcut = (e: Pick<KeyboardEvent, "altKey" | "shiftKey" | "ctrlKey" | "metaKey" | "code">) =>
  e.altKey && e.shiftKey && !e.ctrlKey && !e.metaKey && e.code === "KeyB";

async function capture(): Promise<Shot> {
  try {
    const { captureScreenshot } = await import("./bug-report-capture");
    return { url: await captureScreenshot(), error: null };
  } catch (e) {
    return { url: null, error: "capture failed: " + (e instanceof Error ? e.message : String(e)).slice(0, 200) };
  }
}
const nextFrame = () => new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));

export function BugReportButton({ page, onOpenTask, captureShot = capture }: {
  page: () => PageState;
  onOpenTask: (taskId: string) => void;
  /** Tests replace the screenshot step. */
  captureShot?: () => Promise<Shot>;
}) {
  const [phase, setPhase] = useState<"idle" | "capturing" | "open">("idle");
  const [shot, setShot] = useState<Shot>({ url: null, error: null });
  const [context, setContext] = useState<BugContext | null>(null);
  const [sent, setSent] = useState<BugReportResult | null>(null);
  const pageRef = useRef(page);
  pageRef.current = page;
  const start = useCallback(async () => {
    if (phase !== "idle") return;
    setSent(null);
    setPhase("capturing");
    // Let the button disappear before the page is drawn; the capture also filters it out.
    try {
      await nextFrame();
      const taken = await captureShot();
      setContext(await collectContext(pageRef.current()));
      setShot(taken);
      setPhase("open");
    } catch {
      setPhase("idle");
    }
  }, [phase, captureShot]);
  useEffect(() => {
    const listen = (e: KeyboardEvent) => {
      if (!isReportShortcut(e)) return;
      e.preventDefault();
      void start();
    };
    window.addEventListener("keydown", listen);
    return () => window.removeEventListener("keydown", listen);
  }, [start]);
  useEffect(() => {
    if (!sent) return;
    const timer = setTimeout(() => setSent(null), 8000);
    return () => clearTimeout(timer);
  }, [sent]);
  return <>
    {phase === "idle" && <button type="button" className="bug-report-launcher" data-bug-report-ignore aria-label="Report a bug"
      title="Report a bug (Alt+Shift+B)" onClick={() => void start()}>
      <Bug size={20} aria-hidden="true"/>
    </button>}
    {phase === "open" && context && <BugReportDialog shot={shot} context={context}
      onClose={() => setPhase("idle")}
      onSent={(result) => { setPhase("idle"); setSent(result); }}/>}
    {sent && <div className="toast" role="status" data-bug-report-ignore>
      <Check size={16}/>Reported. Thanks
      {sent.task_id && <button type="button" onClick={() => { onOpenTask(sent.task_id!); setSent(null); }}>Open task</button>}
    </div>}
  </>;
}

/** The detail rows the dialog lists, exactly as sent. */
function detailRows(context: BugContext, screenshot: boolean, shot: Shot) {
  const { page, build, device, service_worker: sw } = context;
  return [
    ["Page", `${page.view}${page.run_id ? ` · run ${page.run_id}` : ""}${page.task_id ? ` · task ${page.task_id}` : ""}`],
    ["Address", page.url],
    ["Open dialogs", page.dialogs.join(", ") || "None"],
    ["App build", `${build.version} · ${build.commit}${build.built_at ? " · " + build.built_at : ""}`],
    ["Service worker", sw.controlled ? `${"version" in sw && sw.version ? sw.version : "unknown version"}` : "Not active"],
    ["Device", `${device.platform} · ${device.viewport} @${device.device_pixel_ratio}x · ${device.display_mode}${device.online ? "" : " · offline"}`],
    ["Browser", device.user_agent],
    ["Theme and locale", `${device.resolved_theme} (${device.theme}) · ${device.locale} · ${device.time_zone}`],
    ["Saved settings", Object.keys(context.storage).length + " keys, key-like values removed"],
    ["Recent errors", String(context.errors.length)],
    ["Recent requests", `${context.requests.length} (method, path without query values, status, timing)`],
    ["Recent pages", String(context.navigations.length)],
    ["Screenshot", screenshot && shot.url ? "Included" : shot.error ? `Not captured (${shot.error})` : "Not included"],
  ] as const;
}

export function BugReportDialog({ shot, context, onClose, onSent }: {
  shot: Shot; context: BugContext; onClose: () => void; onSent: (result: BugReportResult) => void;
}) {
  const [description, setDescription] = useState(""), [expected, setExpected] = useState("");
  const [include, setInclude] = useState(!!shot.url), [busy, setBusy] = useState(false), [error, setError] = useState("");
  const close = () => { if (!busy) onClose(); };
  async function send() {
    if (!description.trim() || busy) return;
    setBusy(true);
    setError("");
    try {
      const result = await post<BugReportResult>("/bug-reports", {
        description, expected, context,
        screenshot: include && shot.url ? shot.url : null,
        screenshot_error: !shot.url ? shot.error : include ? null : "left out by the reporter",
      });
      onSent(result);
    } catch (e) {
      setError(e instanceof ApiError || e instanceof Error ? e.message : "The report could not be sent.");
      setBusy(false);
    }
  }
  return <Dialog as="form" className="dialog bug-report-dialog" aria-labelledby="bug-report-title" onEscape={close} onBackdrop={close}
    data-bug-report-ignore onSubmit={(e) => { e.preventDefault(); void send(); }}>
    <div className="dialog-heading">
      <h2 id="bug-report-title">Report a bug</h2>
      <button type="button" className="icon-button" aria-label="Close bug report" onClick={close} disabled={busy}><X size={20}/></button>
    </div>
    {error && <p role="alert" className="error-banner">{error}</p>}
    <label>
      What went wrong?
      <textarea required data-autofocus rows={4} maxLength={10000} value={description} onChange={(e) => setDescription(e.target.value)}/>
    </label>
    <label>
      <span>What did you expect? <span className="field-optional">Optional</span></span>
      <textarea rows={2} maxLength={5000} value={expected} onChange={(e) => setExpected(e.target.value)}/>
    </label>
    {shot.url ? <div className="bug-report-shot">
      <label className="bug-report-include">
        <input type="checkbox" checked={include} onChange={(e) => setInclude(e.target.checked)}/>
        Include screenshot
      </label>
      {include && <img src={shot.url} alt="Screenshot of the page when you clicked Report a bug"/>}
    </div> : <p className="footnote">No screenshot: {shot.error ?? "capture failed"}. The report still sends.</p>}
    <details className="bug-report-details">
      <summary><ChevronRight size={14} aria-hidden="true"/>Details that will be attached</summary>
      <dl>{detailRows(context, include, shot).map(([name, value]) => <div key={name}><dt>{name}</dt><dd>{value}</dd></div>)}</dl>
      <p className="footnote">Never sent: passwords, cookies, sign-in or bot keys, request and response bodies, query values in addresses.</p>
    </details>
    <div className="dialog-actions">
      <button type="button" className="btn" onClick={close} disabled={busy}>Cancel</button>
      <button className="btn btn-primary" disabled={busy || !description.trim()}>{busy ? "Sending…" : "Send"}</button>
    </div>
  </Dialog>;
}
