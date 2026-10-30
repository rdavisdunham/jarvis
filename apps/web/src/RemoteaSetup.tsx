import { useCallback, useEffect, useState } from "react";
import { Check, Copy, RefreshCw, Stethoscope } from "lucide-react";
import { api, post } from "./api";
import { useAgentRefresh } from "./remotea-refresh";
import { ago } from "./remotea-model";
import {
  harnessText, hostLine, mcpText, selfCheck, SETUP_COMMAND, SETUP_STDIN, setupSteps, templateText, verifyText, withProbe,
  type ProbeResult, type SetupHost, type SetupProblem, type SetupReport, type SetupTarget,
} from "./remotea-setup";

/** The one-time setup code after Enable or New setup code: copy it, then paste it into `remotea eridani setup`. */
export function SetupCodeBox({ code, onDone }: { code: string; onDone: () => void }) {
  const [copied, setCopied] = useState(false), [error, setError] = useState("");
  const copy = () => navigator.clipboard.writeText(code).then(() => setCopied(true))
    .catch(() => setError("Copy is unavailable here. Select the text to copy it."));
  return <div className="bot-secret remotea-setup-code" role="region" aria-label="remotea setup code">
    <strong>Copy this setup code now</strong>
    <p>It holds the link key and the key for your coding agents, and is shown only once. Run <code>{SETUP_COMMAND}</code> in
      WSL and paste this code when asked (or pipe it to <code>{SETUP_STDIN}</code>). A new code revokes the keys in the previous one.</p>
    <label className="sr-only" htmlFor="remotea-setup-code">remotea setup code</label>
    <input id="remotea-setup-code" readOnly type="password" value={code} autoComplete="off" onFocus={e => e.target.select()}/>
    {error && <p role="alert" className="work-error">{error}</p>}
    <div className="bot-key-actions">
      <button className="btn btn-primary btn-sm" onClick={() => void copy()}>{copied ? <Check size={14}/> : <Copy size={14}/>} {copied ? "Copied" : "Copy setup code"}</button>
      <button className="btn btn-ghost btn-sm" onClick={onDone}>Done</button>
    </div>
  </div>;
}

function Problems({ items }: { items: SetupProblem[] }) {
  if (!items.length) return null;
  return <ul className="remotea-setup-problems">
    {items.map(p => <li key={p.code + p.message} className={p.blocks ? "blocks" : "warns"}>
      <strong>{p.message}</strong><span>{p.fix}</span>
    </li>)}
  </ul>;
}

/** `remotea doctor` as the host last reported it: counts, each warn/fail line with its fix, the report's age. */
export function HostSelfCheck({ host, now }: { host: SetupHost; now?: number }) {
  const view = selfCheck(host, now);
  return <div className={"remotea-self-check tone-" + view.tone} role="group" aria-label={"Host self-check for " + host.name}>
    <p><strong>Host self-check</strong> <span>{view.headline}</span>{view.age && <small> · {view.age}</small>}</p>
    {!!host.doctor?.items.length && <ul className="remotea-setup-problems">
      {host.doctor.items.map(i => <li key={i.id} className={i.level === "fail" ? "blocks" : "warns"}>
        <strong><span className="remotea-check-level">{i.level === "fail" ? "Fail" : "Warn"}</span> {i.summary}</strong>
        {i.fix && <span>{i.fix}</span>}
        <small><code>{i.id}</code></small>
      </li>)}
    </ul>}
    {view.note && <p className="footnote">{view.note}</p>}
  </div>;
}

export function TargetCard({ target, onProbed }: { target: SetupTarget; onProbed: (probe: ProbeResult) => void }) {
  const [busy, setBusy] = useState(false), [note, setNote] = useState(""), [error, setError] = useState("");
  const probe = async () => {
    setBusy(true); setError("");
    try {
      const result = await post<ProbeResult>(`/remotea/targets/${encodeURIComponent(target.key)}/probe`, {});
      setNote(result.note + (result.hello_at ? ` Last hello ${ago(result.hello_at)}.` : ""));
      onProbed(result);
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  };
  const harnesses = Object.values(target.harnesses);
  return <article className={"remotea-setup-target" + (target.ready ? " ready" : "")} aria-label={"Target " + target.name}>
    <header>
      <strong>{target.name}</strong>
      <span className={"chip " + (target.ready ? "chip-done" : "chip-due-today")}>{target.ready ? "Ready" : "Not ready"}</span>
      <small>{[target.host_name, target.repo].filter(Boolean).join(" · ")}{!target.advertised && " · not advertised"}</small>
    </header>
    {!!harnesses.length && <ul className="remotea-setup-facts" aria-label="Coding agents">{harnesses.map(h => <li key={h.label}>{harnessText(h)}</li>)}</ul>}
    {target.advertised && <ul className="remotea-setup-facts" aria-label="Verify and tools">
      {target.verify === null ? <li>Verify: unknown (host too old)</li> : !target.verify.length ? <li>Verify: none configured</li>
        : target.verify.map(v => <li key={v.id}>Verify {verifyText(v)}</li>)}
      <li>{mcpText(target.mcp)}</li>
      {templateText(target) && <li>{templateText(target)}</li>}
    </ul>}
    <Problems items={target.problems}/>
    {!!target.notes.length && <p className="footnote">{target.notes.join(" ")}</p>}
    <div className="settings-form-actions">
      <button className="btn btn-ghost btn-sm" disabled={busy} onClick={() => void probe()}><Stethoscope size={14}/> {busy ? "Checking…" : "Run a readiness probe"}</button>
    </div>
    {note && <p className="footnote" role="status">{note}</p>}
    {error && <p className="work-error" role="alert">{error}</p>}
  </article>;
}

/** Host connected? → targets advertised? → each target's readiness with a fix per problem, and a probe button. */
export function RemoteaReadiness({ onlyWhenNotReady = false }: { onlyWhenNotReady?: boolean }) {
  const [report, setReport] = useState<SetupReport | null>(null), [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const refresh = useCallback(() => {
    setLoading(true);
    api<SetupReport>("/remotea/setup").then(r => { setReport(r); setError(""); })
      .catch(e => setError((e as Error).message)).finally(() => setLoading(false));
  }, []);
  useEffect(() => { refresh(); }, [refresh]);
  useAgentRefresh(refresh);
  if (onlyWhenNotReady && (!report || report.ready)) return null;
  const steps = report ? setupSteps(report) : [];
  return <section className="remotea-setup" aria-label="remotea setup" aria-busy={loading}>
    <header>
      <h3>{report?.ready ? "remotea setup" : "Set up remotea"}</h3>
      <button className="btn-icon" aria-label="Refresh setup" disabled={loading} onClick={refresh}><RefreshCw size={15}/></button>
    </header>
    {error && <p className="error-banner" role="alert">{error}</p>}
    {!report && !error && <p role="status" className="settings-empty">Checking…</p>}
    {report && <ol className="remotea-setup-steps">
      {steps.map(s => <li key={s.id} className={"tone-" + s.tone}><strong>{s.title}</strong>{s.detail && <span>{s.detail}</span>}</li>)}
    </ol>}
    {report?.hosts.map(h => <div key={h.id} className="remotea-setup-host">
      <small>{hostLine(h)} · {h.connected ? "connected" : "last seen " + ago(h.last_seen_at)}{h.hello_at && ` · last hello ${ago(h.hello_at)}`}{h.build_dirty && " · dirty build"}</small>
      <Problems items={h.problems}/>
      <HostSelfCheck host={h}/>
    </div>)}
    {report?.targets.map(t => <TargetCard key={t.key} target={t} onProbed={p => setReport(r => (r ? withProbe(r, p) : r))}/>)}
  </section>;
}
