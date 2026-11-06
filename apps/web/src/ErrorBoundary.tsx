/**
 * Render-error boundaries. One wraps the whole app (main.tsx) so a crash never leaves a blank page; one
 * wraps the main view (App.tsx) so a crashing view leaves the sidebar, chat and other views working.
 * The notice offers Try again (views), Reload, and Report this, which opens the Report a bug dialog with
 * the page diagnostics (React logs the error to the console ring that report attaches).
 */
import { Component, useState, type ErrorInfo, type ReactNode } from "react";
import { collectContext, type BugContext } from "./bug-report";
import { captureError } from "./monitoring";

type Props = {
  /** Where the boundary sits, for reports ("app", or the view's name). */
  name: string;
  scope?: "app" | "view";
  /** A change (e.g. switching views) clears the error. */
  resetKey?: unknown;
  children: ReactNode;
};

export class ErrorBoundary extends Component<Props, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: unknown) {
    return { error: error instanceof Error ? error : new Error(String(error)) };
  }
  componentDidCatch(error: unknown, info: ErrorInfo) {
    captureError(error, { boundary: this.props.name, componentStack: info.componentStack });
  }
  componentDidUpdate(previous: Props) {
    if (this.state.error && previous.resetKey !== this.props.resetKey) this.setState({ error: null });
  }
  render() {
    if (!this.state.error) return this.props.children;
    return <CrashNotice scope={this.props.scope ?? "view"} name={this.props.name} error={this.state.error}
      onRetry={() => this.setState({ error: null })}/>;
  }
}

type ReportDialog = typeof import("./BugReport").BugReportDialog;

function CrashNotice({ scope, name, error, onRetry }: { scope: "app" | "view"; name: string; error: Error; onRetry: () => void }) {
  const [report, setReport] = useState<{ context: BugContext; Dialog: ReportDialog } | null>(null);
  const [sent, setSent] = useState(false), [failed, setFailed] = useState(false);
  const app = scope === "app";
  // Loaded on demand (keeps it out of the entry chunk); a failed load, e.g. after a deploy, says so instead of throwing.
  const open = async () => {
    try {
      const [{ BugReportDialog }, context] = await Promise.all([import("./BugReport"), collectContext({ view: name })]);
      setReport({ context, Dialog: BugReportDialog });
    } catch {
      setFailed(true);
    }
  };
  return <section className={"crash-notice" + (app ? " crash-app" : "")} role="alert">
    <h2>{app ? "Eridani hit a problem" : "This view hit a problem"}</h2>
    <p>{app ? "Reloading usually fixes it. Anything already saved is safe." : "The rest of Eridani still works. Try again, or reload the page."}</p>
    <div className="crash-actions">
      {!app && <button type="button" className="btn" onClick={onRetry}>Try again</button>}
      <button type="button" className="btn btn-primary" onClick={() => location.reload()}>Reload</button>
      {sent ? <span role="status">Reported. Thanks</span>
        : <button type="button" className="btn" onClick={() => void open()}>Report this</button>}
    </div>
    {failed && <p className="footnote">The report form could not open. Reload, then use Report a bug (Alt+Shift+B).</p>}
    {report && <report.Dialog context={report.context} shot={{ url: null, error: "not captured after a crash" }}
      initialDescription={(app ? "Eridani crashed: " : `The ${name} view crashed: `) + error.message.slice(0, 500)}
      onClose={() => setReport(null)} onSent={() => { setReport(null); setSent(true); }}/>}
  </section>;
}
