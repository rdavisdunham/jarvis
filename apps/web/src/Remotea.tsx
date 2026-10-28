import { useCallback, useEffect, useState } from "react";
import { Check, ChevronDown, ChevronUp, Copy, GitBranch, MonitorSmartphone, Play, RefreshCw, Square, X } from "lucide-react";
import { api, ApiError, post } from "./api";
import { SettingRow, SettingsGroup } from "./SettingsLayout";
import {
  ago, buildAnswers, criteriaEvidence, hostSummary, interviewAnswers, isSettled, latestAttempt, linkedRun, orderRuns, orderSessions,
  cutoverSummary, linearOutcomes, paritySummary, place, POLICY_TEXT, questionMode, REMOTEA_POLL_MS, runButton, runControls, runLabel, stateLabel, targetOptions, timeline,
  unmergedSummary, type Interview, type NeedsTarget, type Parity, type RemoteaSession, type RemoteaStatus, type Run, type RunDetail,
  type RunQuestion, type Turn, type UnmergedWork,
} from "./remotea-model";
import "./remotea.css";

async function copyText(value: string) {
  await navigator.clipboard.writeText(value);
}

/** Refresh on agent/session change-feed signals (coalesced), with a slower poll as the fallback. */
function useAgentRefresh(refresh: () => void) {
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | null = null;
    const soon = () => { if (!timer) timer = setTimeout(() => { timer = null; refresh(); }, 300); };
    const poll = window.setInterval(() => { if (document.visibilityState === "visible") refresh(); }, REMOTEA_POLL_MS);
    window.addEventListener("eri-agents-changed", soon);
    document.addEventListener("visibilitychange", soon);
    return () => {
      if (timer) clearTimeout(timer);
      window.clearInterval(poll);
      window.removeEventListener("eri-agents-changed", soon);
      document.removeEventListener("visibilitychange", soon);
    };
  }, [refresh]);
}

function TurnText({ session, turn }: { session: string; turn: Turn }) {
  const [open, setOpen] = useState(false), [full, setFull] = useState<string | null>(null), [error, setError] = useState("");
  const expand = async () => {
    setOpen(!open);
    if (open || full !== null || !turn.clipped) return;
    try {
      const turns = await api<{ items: Turn[] }>(`/remotea/sessions/${session}/turns?limit=1`);
      setFull(turns.items[0]?.text ?? turn.text);
    } catch (e) { setError((e as Error).message); }
  };
  const text = open ? (full ?? turn.text) : turn.text.slice(0, 280);
  const more = turn.text.length > 280 || turn.clipped;
  return <div className="remotea-turn">
    <p className={open ? "open" : ""}>{text}{!open && more && "…"}</p>
    <div className="remotea-turn-meta">
      <small>Turn {turn.seq} · {ago(turn.at)}{turn.truncated && " · final response truncated"}</small>
      {more && <button className="btn btn-ghost btn-sm" aria-expanded={open} onClick={() => void expand()}>
        {open ? <ChevronUp size={14}/> : <ChevronDown size={14}/>}{open ? "Less" : "Full response"}</button>}
    </div>
    {error && <p role="alert" className="work-error">{error}</p>}
  </div>;
}

function SessionRow({ item }: { item: RemoteaSession }) {
  const state = item.live_state?.state ?? "";
  return <article className={"remotea-session state-" + state} data-session-id={item.id}>
    <header>
      <span className={"remotea-state state-" + state}>{stateLabel(state)}</span>
      <span className="remotea-since">since {ago(item.live_state?.since ?? null)}</span>
      <span className="remotea-host">{item.host ? item.host.name || "remotea" : ""}{item.host && !item.host.connected && " (offline)"}</span>
    </header>
    <h3>{item.title}</h3>
    <p className="remotea-place">{item.tool_label}{place(item) && " · " + place(item)}{item.filed.record_title && " · " + item.filed.record_title}</p>
    {item.live_state?.reason && <p className="remotea-reason">{item.live_state.reason}</p>}
    {item.last_turn ? <TurnText session={item.id} turn={item.last_turn}/> : <p className="footnote">No final response captured yet.</p>}
  </article>;
}

export function Remotea({ run }: { run?: string | null } = {}) {
  const [open, setOpen] = useState<string | null>(() => run ?? linkedRun(location.search));
  useEffect(() => { if (run) setOpen(run); }, [run]);
  const [status, setStatus] = useState<RemoteaStatus | null>(null);
  const [items, setItems] = useState<RemoteaSession[] | null>(null);
  const [parity, setParity] = useState<Parity | null>(null);
  const [error, setError] = useState(""), [loading, setLoading] = useState(false);
  const refresh = useCallback(() => {
    setLoading(true);
    Promise.all([
      api<RemoteaStatus>("/remotea/status"),
      api<{ items: RemoteaSession[] }>("/remotea/sessions"),
      api<Parity>("/remotea/parity"),
    ]).then(([s, list, p]) => { setStatus(s); setItems(orderSessions(list.items)); setParity(p); setError(""); })
      .catch(e => setError((e as Error).message))
      .finally(() => setLoading(false));
  }, []);
  useEffect(() => { refresh(); }, [refresh]);
  useAgentRefresh(refresh);
  const hosts = hostSummary(status?.hosts ?? []);
  const cutover = cutoverSummary(parity);
  return <div className="remotea-page">
    <section className={"remotea-hosts tone-" + hosts.tone} aria-label="remotea server">
      <MonitorSmartphone size={18} aria-hidden="true"/><span>{hosts.text}</span>
      <button className="btn-icon" aria-label="Refresh remotea" disabled={loading} onClick={refresh}><RefreshCw size={16}/></button>
    </section>
    {error && <p className="error-banner" role="alert">{error}</p>}
    <RunList onOpen={setOpen}/>
    <UnmergedList/>
    {open && <RunDrawer id={open} onClose={() => setOpen(null)}/>}
    <section className="remotea-sessions" aria-label="Agent sessions" aria-busy={loading}>
      {items && !items.length && <p className="activity-empty">No agent sessions are running in remotea right now.</p>}
      {items?.map(item => <SessionRow key={item.id} item={item}/>)}
      {!items && !error && <p role="status" className="settings-empty">Loading sessions…</p>}
    </section>
    <details className="remotea-parity">
      <summary>Collector parity{cutover?.ready ? " · ready for cutover" : ""}</summary>
      {cutover && <p className={"chip " + (cutover.ready ? "chip-done" : "chip-due-today")} role="status">{cutover.text}</p>}
      <p>{paritySummary(parity)}</p>
      {!!parity?.recent.length && <ul>{parity.recent.slice(0, 8).map(r =>
        <li key={r.tool + r.session_id}><code>{r.session_id.slice(0, 12)}</code> {r.missing ? "not reported by the current collector" : Object.keys(r.diff).join(", ")}</li>)}</ul>}
    </details>
  </div>;
}

export function RemoteaSettings({ readOnly, onChanged }: { readOnly: boolean; onChanged: () => void }) {
  const [status, setStatus] = useState<RemoteaStatus | null>(null), [token, setToken] = useState("");
  const [busy, setBusy] = useState(false), [error, setError] = useState(""), [copied, setCopied] = useState("");
  useEffect(() => { void api<RemoteaStatus>("/remotea/status").then(setStatus).catch(e => setError(e.message)); }, []);
  const change = async (path: string, body?: unknown) => {
    setBusy(true); setError("");
    try {
      const next = await post<RemoteaStatus>(path, body);
      setStatus(next); setToken(next.token ?? ""); onChanged();
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  };
  const copy = (value: string, label: string) => copyText(value).then(() => setCopied(label))
    .catch(() => setError("Copy is unavailable here. Select the text to copy it."));
  const hosts = hostSummary(status?.hosts ?? []);
  return <SettingsGroup className="remotea-settings" icon={<MonitorSmartphone size={18}/>} title="remotea"
    description="Follow Claude Code and Codex sessions running in your remotea server from this app.">
    {error && <p className="error-banner" role="alert">{error}</p>}
    <SettingRow as="label" label="Show remotea" hint="Adds the remotea page and links your remotea server. Off by default.">
      <input type="checkbox" checked={!!status?.enabled} disabled={!status || busy || readOnly}
        onChange={e => void change(e.target.checked ? "/remotea/enable" : "/remotea/disable", e.target.checked ? {} : undefined)}/>
    </SettingRow>
    {status?.enabled && <>
      <SettingRow label="Server" hint={status.key ? `Link key ${status.key.prefix}… · expires ${new Date(status.key.expires_at).toLocaleDateString()}` : "No active link key."}>
        <span className={"remotea-host-chip tone-" + hosts.tone}>{hosts.tone === "connected" ? "Connected" : hosts.tone === "never" ? "Never connected" : ago(status.hosts.map(h => h.last_seen_at).filter(Boolean).sort().at(-1) ?? null)}</span>
      </SettingRow>
      {token && <div className="bot-secret" role="region" aria-label="remotea link key">
        <strong>Copy this key now</strong><p>This is the only time it will be shown. On your desktop, run the command below and paste the key when asked.</p>
        <label className="sr-only" htmlFor="remotea-secret">remotea link key</label>
        <input id="remotea-secret" readOnly type="password" value={token} autoComplete="off" onFocus={e => e.target.select()}/>
        <div className="bot-key-actions">
          <button className="btn btn-primary btn-sm" onClick={() => void copy(token, "key")}>{copied === "key" ? <Check size={14}/> : <Copy size={14}/>} {copied === "key" ? "Copied" : "Copy key"}</button>
          <button className="btn btn-ghost btn-sm" onClick={() => setToken("")}>Done</button>
        </div>
      </div>}
      <div className="copy-field">
        <label className="field"><span className="field-label-text">Connect command</span>
          <input readOnly aria-label="Connect command" value={status.connect_command} onFocus={e => e.target.select()}/></label>
        <button className="btn-icon" aria-label="Copy connect command" onClick={() => void copy(status.connect_command, "command")}>{copied === "command" ? <Check size={15}/> : <Copy size={15}/>}</button>
      </div>
      {!readOnly && <div className="settings-form-actions">
        <button className="btn btn-ghost btn-sm" disabled={busy} onClick={() => void change("/remotea/enable", { rotate: true })}>New link key</button>
      </div>}
      <SettingRow as="label" label="Complete tasks automatically"
        hint="When the judge finds every success criterion met, the task is completed and closed (in Linear too) without your tap. Off by default; leave it off until the outcome eval is clean.">
        <input type="checkbox" checked={!!status.auto_complete} disabled={busy || readOnly}
          onChange={e => void change("/remotea/auto-complete", { enabled: e.target.checked })}/>
      </SettingRow>
      {status.auto_complete && <p className="remotea-warning" role="note">Automatic completion is on: agents' tasks close without asking you.</p>}
    </>}
  </SettingsGroup>;
}

// ---------------------------------------------------------------- runs (stage C)

function message(e: unknown) {
  return (e as Error).message || "Something went wrong.";
}

function RunList({ onOpen }: { onOpen: (id: string) => void }) {
  const [runs, setRuns] = useState<Run[] | null>(null), [error, setError] = useState("");
  const refresh = useCallback(() => {
    api<{ items: Run[] }>("/remotea/runs?limit=20").then(r => { setRuns(orderRuns(r.items)); setError(""); }).catch(e => setError(message(e)));
  }, []);
  useEffect(() => { refresh(); }, [refresh]);
  useAgentRefresh(refresh);
  if (!runs?.length && !error) return null;
  const active = runs?.filter(r => !isSettled(r)) ?? [], recent = runs?.filter(isSettled).slice(0, 5) ?? [];
  return <section className="remotea-runs" aria-label="Agent runs">
    <h2>Runs</h2>
    {error && <p role="alert" className="work-error">{error}</p>}
    {!active.length && <p className="footnote">No runs are active.</p>}
    {[...active, ...recent].map(r => <button key={r.id} className={"remotea-run-row state-" + r.state} onClick={() => onOpen(r.id)}>
      <span className={"remotea-state run-state-" + r.state}>{runLabel(r.state)}</span>
      <span className="remotea-run-title">{r.task?.title ?? "Task"}</span>
      <small>{r.reason_text || ago(r.state_since)}</small>
    </button>)}
  </section>;
}

function UnmergedList() {
  const [items, setItems] = useState<UnmergedWork[]>([]);
  const refresh = useCallback(() => { api<{ items: UnmergedWork[] }>("/remotea/unmerged").then(r => setItems(r.items)).catch(() => setItems([])); }, []);
  useEffect(() => { refresh(); }, [refresh]);
  useAgentRefresh(refresh);
  if (!items.length) return null;
  return <details className="remotea-unmerged">
    <summary>Unmerged work ({items.length})</summary>
    <ul>{items.map(w => <li key={w.host_id + (w.path ?? w.branch ?? "")}>
      <GitBranch size={14} aria-hidden="true"/> <code>{w.branch ?? w.path}</code> <small>{w.target ? w.target + " · " : ""}{unmergedSummary(w)}{w.active ? " · run active" : ""}</small>
    </li>)}</ul>
  </details>;
}

function QuestionCard({ question, run, onDone }: { question: RunQuestion; run: RunDetail; onDone: () => void }) {
  const mode = questionMode(question, run);
  const [picked, setPicked] = useState<Record<string, string[]>>({}), [typed, setTyped] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false), [error, setError] = useState("");
  if (mode === "closed") return null;
  const answers = buildAnswers(question, picked, typed);
  const toggle = (item: string, choice: string, multi: boolean) => setPicked(current => {
    const now = current[item] ?? [];
    return { ...current, [item]: multi ? (now.includes(choice) ? now.filter(c => c !== choice) : [...now, choice]) : [choice] };
  });
  const send = async () => {
    if (!answers) return;
    setBusy(true); setError("");
    try { await post(`/remotea/questions/${question.id}/answer`, { answers }); onDone(); }
    catch (e) { setError(message(e)); } finally { setBusy(false); }
  };
  return <section className={"remotea-question mode-" + mode} aria-label="Agent question">
    {question.items.map(item => <fieldset key={item.id} disabled={mode !== "answer" || busy}>
      <legend>{item.header ? <strong>{item.header}: </strong> : null}{item.text}</legend>
      {mode === "answer" && !!item.choices?.length && <div className="remotea-choices">{item.choices.map(c =>
        <button key={c.id} type="button" aria-pressed={(picked[item.id] ?? []).includes(c.id)} title={c.description}
          className={"btn btn-sm" + ((picked[item.id] ?? []).includes(c.id) ? " btn-primary" : "")}
          onClick={() => toggle(item.id, c.id, !!item.multi)}>{c.label}</button>)}</div>}
      {mode === "answer" && <input aria-label={"Answer: " + item.text} placeholder={item.choices?.length ? "Or type an answer" : "Your answer"}
        value={typed[item.id] ?? ""} onChange={e => setTyped({ ...typed, [item.id]: e.target.value })}/>}
    </fieldset>)}
    {mode === "answer" && <button className="btn btn-primary btn-sm" disabled={!answers || busy} onClick={() => void send()}>{busy ? "Sending…" : "Send answer"}</button>}
    {mode === "pending" && <p role="status" className="footnote">Answer sent; waiting for the agent to take it.</p>}
    {mode === "terminal" && <p className="footnote">{question.status === "timed_out" ? "This question timed out here. " : ""}Open the terminal on your remotea server to answer.</p>}
    {error && <p role="alert" className="work-error">{error}</p>}
  </section>;
}

export function InterviewForm({ interview, onDone }: { interview: Interview; onDone: () => void }) {
  const [picked, setPicked] = useState<Record<string, string[]>>({}), [typed, setTyped] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false), [error, setError] = useState("");
  const answers = interviewAnswers(interview, picked, typed);
  const toggle = (q: string, label: string, multi: boolean) => setPicked(current => {
    const now = current[q] ?? [];
    return { ...current, [q]: multi ? (now.includes(label) ? now.filter(l => l !== label) : [...now, label]) : now.includes(label) ? [] : [label] };
  });
  const send = async () => {
    if (!answers) return;
    setBusy(true); setError("");
    try { await post(`/remotea/interviews/${interview.id}/answers`, { answers }); onDone(); }
    catch (e) { setError(message(e)); } finally { setBusy(false); }
  };
  return <section className="remotea-question remotea-interview" aria-label="Planning questions">
    <p className="footnote">The agent needs these answers to write the plan.</p>
    {interview.questions.map(q => <fieldset key={q.id} disabled={busy}>
      <legend>{q.header ? <strong>{q.header}: </strong> : null}{q.question}{q.multi && <small> (choose any)</small>}</legend>
      {!!q.options.length && <div className="remotea-choices">{q.options.map(o =>
        <button key={o.label} type="button" aria-pressed={(picked[q.id] ?? []).includes(o.label)} title={o.description || undefined}
          className={"btn btn-sm" + ((picked[q.id] ?? []).includes(o.label) ? " btn-primary" : "")}
          onClick={() => toggle(q.id, o.label, q.multi)}>{o.label}</button>)}</div>}
      {q.free_text && <input aria-label={"Answer: " + q.question} placeholder={q.options.length ? "Or type an answer" : "Your answer"}
        value={typed[q.id] ?? ""} onChange={e => setTyped({ ...typed, [q.id]: e.target.value })}/>}
    </fieldset>)}
    <button className="btn btn-primary btn-sm" disabled={!answers || busy} onClick={() => void send()}>{busy ? "Sending…" : "Send answers"}</button>
    {error && <p role="alert" className="work-error">{error}</p>}
  </section>;
}

export function Criteria({ run }: { run: RunDetail }) {
  const rows = criteriaEvidence(run);
  if (!rows.length) return run.plan ? <p className="footnote">This plan has no success criteria, so the judge cannot complete it; you approve.</p> : null;
  const judge = latestAttempt(run, "judge"), verify = latestAttempt(run, "verify");
  return <details className="remotea-plan remotea-criteria" open={run.state === "awaiting_approval"}>
    <summary>Success criteria · {rows.filter(r => r.status === "met").length}/{rows.length} met{judge?.turn_seq ? ` (turn ${judge.turn_seq})` : ""}</summary>
    <ul>{rows.map(r => <li key={r.id} className={"criterion-" + r.status}>
      <strong>{r.status === "met" ? "Met" : r.status === "pending" ? "Not judged" : r.status === "unclear" ? "Unclear" : "Not met"}</strong> {r.text}
      {r.kind === "verify" && <small> · needs {r.verify_id} to pass</small>}
      {r.evidence && <p>{r.evidence}</p>}
    </li>)}</ul>
    {judge?.gap && judge.verdict !== "met" && <p className="remotea-reason">{judge.gap}</p>}
    {verify && <p className="footnote">Last verification: {verify.verify_id} {verify.status.replace("_", " ")}{verify.exit_code != null ? ` (exit ${verify.exit_code})` : ""}</p>}
  </details>;
}

export function RunDrawer({ id, onClose }: { id: string; onClose: () => void }) {
  const [run, setRun] = useState<RunDetail | null>(null), [error, setError] = useState("");
  const [text, setText] = useState(""), [busy, setBusy] = useState("");
  const refresh = useCallback(() => {
    api<RunDetail>(`/remotea/runs/${id}`).then(r => { setRun(r); setError(""); }).catch(e => setError(message(e)));
  }, [id]);
  useEffect(() => { setRun(null); refresh(); }, [refresh]);
  useAgentRefresh(refresh);
  useEffect(() => {
    const key = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [onClose]);
  const act = async (what: string, path: string, body: object) => {
    setBusy(what); setError("");
    try { await post(`/remotea/runs/${id}/${path}`, body); if (path === "message") setText(""); refresh(); }
    catch (e) { setError(message(e)); } finally { setBusy(""); }
  };
  const controls = run ? runControls(run) : null;
  return <aside className="remotea-drawer" role="dialog" aria-label="Agent run">
    <header>
      <div>
        {run && <span className={"remotea-state run-state-" + run.state}>{runLabel(run.state)}</span>}
        <h2>{run?.task?.title ?? "Run"}</h2>
      </div>
      <button className="btn-icon" aria-label="Close run" onClick={onClose}><X size={18}/></button>
    </header>
    {error && <p role="alert" className="work-error">{error}</p>}
    {!run ? !error && <p role="status">Loading run…</p> : <>
      <p className="remotea-place">
        {run.tool}{run.branch && <> · <code>{run.branch}</code></>}{run.base_branch && <> from <code>{run.base_branch}</code></>}
        {run.host && <> · {run.host.name || "remotea"}{!run.host.connected && " (offline)"}</>} · turn {run.turn_count}/{run.turn_cap}
      </p>
      {run.reason_text && <p className="remotea-reason">{run.reason_text}</p>}
      {run.questions.map(q => <QuestionCard key={q.id + q.status} question={q} run={run} onDone={refresh}/>)}
      {run.state === "interviewing" && run.interviews?.filter(i => i.status === "open").map(i =>
        <InterviewForm key={i.id} interview={i} onDone={refresh}/>)}
      {!!run.integration?.length && <ul className="remotea-integration" aria-label="Sub-task integration">{run.integration.map(i =>
        <li key={i.task_id}><GitBranch size={14} aria-hidden="true"/> {i.branch ? <code>{i.branch}</code> : "Sub-task"}: {i.reason}</li>)}</ul>}
      <Criteria run={run}/>
      {run.completion_policy && run.completion_policy !== "code_ready" && <p className="footnote">{POLICY_TEXT[run.completion_policy] ?? run.completion_policy}</p>}
      {!!linearOutcomes(run).length && <ul className="remotea-linear" aria-label="Linear sync">{linearOutcomes(run).map(o =>
        <li key={o.key} className={"tone-" + o.tone}>{o.url ? <a href={o.url} target="_blank" rel="noreferrer">{o.text}</a> : o.text}</li>)}</ul>}
      {!!run.plan?.steps?.length && <details className="remotea-plan">
        <summary>Plan · {run.plan.steps.filter(s => s.status === "done").length}/{run.plan.steps.length} steps</summary>
        <ol>{run.plan.steps.map(s => <li key={s.id} className={"step-" + s.status}>{s.title}</li>)}</ol>
      </details>}
      {controls && !isSettled(run) && <div className="remotea-controls">
        <textarea aria-label="Message the agent" rows={2} placeholder={controls.message.enabled ? "Message the agent" : controls.message.reason}
          disabled={!controls.message.enabled || !!busy} value={text} onChange={e => setText(e.target.value)}/>
        <div className="remotea-control-row">
          <button className="btn btn-sm" disabled={!controls.message.enabled || !text.trim() || !!busy} title="Queued for after the current turn"
            onClick={() => void act("message", "message", { text, mode: "queue" })}>Message</button>
          <button className="btn btn-sm" disabled={!controls.steer.enabled || !text.trim() || !!busy} title="Interrupts the current turn"
            onClick={() => void act("steer", "message", { text, mode: "steer" })}>Steer</button>
          <button className="btn btn-sm btn-danger" disabled={!controls.stop.enabled || !!busy || controls.stop.label === "Stopping…"}
            onClick={() => void act("stop", "stop", { mode: controls.stop.mode })}><Square size={14}/>{busy === "stop" ? "Stopping…" : controls.stop.label}</button>
          <button className={"btn btn-sm" + (controls.approve.primary ? " btn-primary" : "")} disabled={!controls.approve.enabled || !!busy}
            title="Mark the task complete and end the run" onClick={() => void act("approve", "approve", {})}><Check size={14}/>Approve</button>
        </div>
        <div className="remotea-control-row">
          <button className="btn btn-sm btn-ghost" disabled={!controls.replan.enabled || !!busy} title={controls.replan.reason}
            onClick={() => void act("replan", "replan", { reason: text })}>{busy === "replan" ? "Re-planning…" : "Re-plan"}</button>
          {run.plan?.verify && <button className="btn btn-sm btn-ghost" disabled={!!busy || !run.surf} title="Run the plan's verification now"
            onClick={() => void act("verify", "verify", {})}>{busy === "verify" ? "Starting…" : "Verify now"}</button>}
        </div>
        {controls.offline && <p className="footnote">The remotea server is offline; commands wait until it reconnects.</p>}
      </div>}
      <ol className="remotea-timeline" aria-label="Run timeline">{timeline(run).map((e, i) =>
        <li key={i} className={"kind-" + e.kind + " tone-" + e.tone}>
          <div><strong>{e.title}</strong> <small>{ago(e.at)}</small></div>
          {e.detail && <p>{e.detail}</p>}
        </li>)}</ol>
    </>}
  </aside>;
}

function TargetPicker({ need, onSaved, onCancel }: { need: NeedsTarget; onSaved: (id: string) => void; onCancel: () => void }) {
  const options = targetOptions(need);
  const [key, setKey] = useState(options[0]?.key ?? ""), [scope, setScope] = useState(need.records[0]?.id ?? "task");
  const [busy, setBusy] = useState(false), [error, setError] = useState("");
  const save = async () => {
    const option = options.find(o => o.key === key);
    if (!option) return;
    setBusy(true); setError("");
    try {
      if ("saved" in option && scope === "task") return onSaved(option.saved.id);
      const source = "saved" in option ? option.saved : option.advertised;
      const saved = await post<{ id: string }>("/remotea/targets", {
        host_id: source.host_id, host_target: source.host_target, repo: source.repo, worktree_root: source.worktree_root,
        ...(scope === "task" ? { task_id: need.task_id } : { record_id: scope }),
      });
      onSaved(saved.id);
    } catch (e) { setError(message(e)); } finally { setBusy(false); }
  };
  return <div className="remotea-target-picker" role="group" aria-label="Choose where the agent runs">
    <p>Choose where this task's agent should run.</p>
    {!options.length ? <p className="footnote">Your remotea server has not listed any repositories yet.</p> : <>
      <select aria-label="Repository" value={key} onChange={e => setKey(e.target.value)}>
        {options.map(o => <option key={o.key} value={o.key}>{o.label}{o.detail ? " — " + o.detail : ""}</option>)}
      </select>
      <select aria-label="Remember for" value={scope} onChange={e => setScope(e.target.value)}>
        {need.records.map(r => <option key={r.id} value={r.id}>Remember for {r.title}</option>)}
        <option value="task">Only this task</option>
      </select>
    </>}
    <div className="remotea-control-row">
      <button className="btn btn-sm" onClick={onCancel}>Cancel</button>
      <button className="btn btn-sm btn-primary" disabled={!options.length || busy} onClick={() => void save()}>Save and run</button>
    </div>
    {error && <p role="alert" className="work-error">{error}</p>}
  </div>;
}

/** The task detail's Run control: starts a run, asks for a target when needed, and opens the active run. */
export function RunTask({ taskId, canEdit, onOpenRun }: { taskId: string; canEdit: boolean; onOpenRun: (id: string) => void }) {
  const [status, setStatus] = useState<RemoteaStatus | null>(null), [active, setActive] = useState<Run | null>(null);
  const [need, setNeed] = useState<NeedsTarget | null>(null), [busy, setBusy] = useState(false), [error, setError] = useState("");
  const refresh = useCallback(() => {
    Promise.all([api<RemoteaStatus>("/remotea/status"), api<{ items: Run[] }>(`/remotea/runs?task_id=${taskId}&active=true&limit=1`)])
      .then(([s, r]) => { setStatus(s); setActive(r.items[0] ?? null); }).catch(() => {});
  }, [taskId]);
  useEffect(() => { refresh(); }, [refresh]);
  useAgentRefresh(refresh);
  if (!status) return null;
  const button = runButton(status.enabled, status.hosts, active);
  if (!button.visible) return null;
  const start = async (target?: string) => {
    setBusy(true); setError(""); setNeed(null);
    try {
      const run = await post<Run>("/remotea/runs", { task_id: taskId, ...(target ? { execution_target_id: target } : {}) });
      setActive(run); onOpenRun(run.id);
    } catch (e) {
      if (e instanceof ApiError && e.code === "NEEDS_TARGET") setNeed(e.data as NeedsTarget);
      else setError(message(e));
    } finally { setBusy(false); }
  };
  return <section className="detail-section remotea-run-task" aria-label="Agent run">
    <div className="remotea-control-row">
      <button className={"btn btn-sm" + (active ? "" : " btn-primary")} disabled={button.disabled || busy || (!active && !canEdit)}
        title={button.reason || undefined} onClick={() => active ? onOpenRun(active.id) : void start()}>
        <Play size={14}/>{busy ? "Starting…" : button.label}</button>
      {button.reason && <small>{button.reason}</small>}
    </div>
    {need && <TargetPicker need={need} onSaved={id => void start(id)} onCancel={() => setNeed(null)}/>}
    {error && <p role="alert" className="work-error">{error}</p>}
  </section>;
}
