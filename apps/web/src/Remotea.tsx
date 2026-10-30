import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { Check, ChevronDown, ChevronUp, Copy, GitBranch, MonitorSmartphone, PanelRightClose, Play, RefreshCw, Square, X } from "lucide-react";
import { api, ApiError, post } from "./api";
import { useAgentRefresh } from "./remotea-refresh";
import { useRemotea, useRemoteaActions } from "./remotea-context";
import { PendingControls } from "./PendingControls";
import { RunRow } from "./RunsPanel";
import { SessionsLedger } from "./Sessions";
import { SettingRow, SettingsGroup } from "./SettingsLayout";
import {
  ago, ATTENTION_STATES, baseLine, buildAnswers, composerAction, criteriaEvidence, hostSummary, interviewAnswers, isSettled, latestAttempt, orderRuns, orderSessions,
  cutoverSummary, linearOutcomes, orderOptions, paritySummary, place, planReviewBody, POLICY_TEXT, questionMode, recommended, REMOTEA_POLL_MS,
  defaultScope, runButton, runControls, runLabel, stateLabel, targetOptions, timeline, unmergedSummary,
  type Interview, type NeedsTarget, type Parity, type PlanDecision, type RemoteaSession, type RemoteaStatus, type Run, type RunDetail,
  type RunQuestion, type TimelineEntry, type Turn, type UnmergedWork, versionLine, type MigrationTarget, type MigrationDraft, migrationBody, migrationDraft, migrationSummary,
} from "./remotea-model";
import "./remotea.css";
import "./runs.css";

async function copyText(value: string) {
  await navigator.clipboard.writeText(value);
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

export function Remotea() {
  // The run detail lives in the app shell's side panel (RunPanel), so it stays open across pages.
  const { openRun } = useRemoteaActions();
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
    <PendingControls/>
    <RunList onOpen={openRun}/>
    <UnmergedList/>
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
    <section className="remotea-ledger" aria-label="Session ledger">
      <h2>Session ledger</h2>
      <SessionsLedger/>
    </section>
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
    <SettingRow as="label" label="Show remotea" hint="Adds the remotea page and links your remotea server. Off by default. Turning it off revokes the link key; turning it on again shows a new one. Personal workspace only.">
      <input type="checkbox" checked={!!status?.enabled} disabled={!status || busy || readOnly}
        onChange={e => void change(e.target.checked ? "/remotea/enable" : "/remotea/disable", e.target.checked ? {} : undefined)}/>
    </SettingRow>
    {status?.enabled && <>
      <SettingRow label="Server" hint={status.key ? `Link key ${status.key.prefix}… · expires ${new Date(status.key.expires_at).toLocaleDateString()}` : "No active link key."}>
        <span className={"remotea-host-chip tone-" + hosts.tone}>{hosts.tone === "connected" ? "Connected" : hosts.tone === "never" ? "Never connected" : ago(status.hosts.map(h => h.last_seen_at).filter(Boolean).sort().at(-1) ?? null)}</span>
      </SettingRow>
      {status.hosts.some(h => h.status === "degraded") && <p className="remotea-warning" role="note">Today's agent text limit is reached: new turns are recorded without their text until tomorrow (UTC).</p>}
      {status.hosts.map(h => ({ h, line: versionLine(h) })).filter(x => x.line).map(({ h, line }) =>
        <SettingRow key={h.id} label={h.name || h.machine || "remotea"}
          hint={line!.update ? `The latest release is ${line!.update.latest}; this server is more than one release behind.` : undefined}>
          <span className={"remotea-version" + (line!.update ? " outdated" : "")}>{line!.update?.url
            ? <a href={line!.update.url} target="_blank" rel="noreferrer">{line!.text}</a> : line!.text}</span>
        </SettingRow>)}
      <MigrationSettings readOnly={readOnly}/>
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
      <SettingRow as="label" label="Agent questions break quiet hours"
        hint="Off by default: an agent's question waits for your quiet hours to end like any other notification. On makes it urgent.">
        <input type="checkbox" checked={!!status.quiet_bypass} disabled={busy || readOnly}
          onChange={e => void change("/remotea/quiet-bypass", { enabled: e.target.checked })}/>
      </SettingRow>
    </>}
  </SettingsGroup>;
}

/** Every advertised target: where its migrations live (Settings, else the host's eridani.json), the chain head and
 * its override, the next number Eridani reserves when it freezes a plan that adds a migration. */
function MigrationSettings({ readOnly }: { readOnly: boolean }) {
  const [items, setItems] = useState<MigrationTarget[] | null>(null), [error, setError] = useState("");
  const [busy, setBusy] = useState("");
  useEffect(() => { void api<{ items: MigrationTarget[] }>("/remotea/migrations").then(r => setItems(r.items)).catch(e => setError(message(e))); }, []);
  if (!items?.length && !error) return null;
  const save = async (m: MigrationTarget, draft: MigrationDraft) => {
    const body = migrationBody(m, draft);
    if (typeof body === "string") { setError(body); return; }
    setBusy(m.host_id + "/" + m.host_target); setError("");
    try { setItems((await post<{ items: MigrationTarget[] }>("/remotea/migrations", body)).items); }
    catch (e) { setError(message(e)); } finally { setBusy(""); }
  };
  return <>
    {error && <p className="error-banner" role="alert">{error}</p>}
    {items?.map(m => <MigrationRow key={m.host_id + "/" + m.host_target + "/" + (m.dir ?? "") + "/" + (m.override ?? "")}
      item={m} readOnly={readOnly} busy={busy === m.host_id + "/" + m.host_target} onSave={d => void save(m, d)}/>)}
  </>;
}

export function MigrationRow({ item, readOnly, busy, onSave }: {
  item: MigrationTarget; readOnly: boolean; busy: boolean; onSave: (draft: MigrationDraft) => void;
}) {
  const [draft, setDraft] = useState<MigrationDraft>(() => migrationDraft(item));
  const field = (name: keyof MigrationDraft, label: string, placeholder: string) =>
    <label className="field"><span className="field-label-text">{label}</span>
      <input name={name} value={draft[name]} placeholder={placeholder} disabled={readOnly || busy}
        inputMode={name === "override" ? "numeric" : undefined} onChange={e => setDraft(d => ({ ...d, [name]: e.target.value }))}/></label>;
  return <SettingRow label={`Migrations · ${item.host_target}`} hint={migrationSummary(item)}>
    <span className="remotea-migration-form">
      {field("dir", "Folder", item.host_dir ?? "migrations/versions")}
      {field("pattern", "File pattern", "^(\\d{4})_")}
      {field("override", "Chain head override", item.head != null ? String(item.head) : "none")}
      {!readOnly && <button className="btn btn-ghost btn-sm" disabled={busy} onClick={() => onSave(draft)}>Save</button>}
    </span>
  </SettingRow>;
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
    {[...active, ...recent].map(r => <RunRow key={r.id} run={r} onOpen={onOpen}/>)}
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
    <p className="remotea-card-kicker">The agent asks</p>
    {question.items.map(item => <fieldset key={item.id} disabled={mode !== "answer" || busy}>
      <legend>{item.header ? <strong>{item.header}: </strong> : null}{item.text}</legend>
      {mode === "answer" && !!item.choices?.length && <div className="remotea-choices">{orderOptions(item.choices).map(c => {
        const on = (picked[item.id] ?? []).includes(c.id);
        return <button key={c.id} type="button" aria-pressed={on} title={c.description}
          className={"btn btn-sm" + (on ? " btn-primary" : recommended(c) ? " btn-recommended" : "")}
          onClick={() => toggle(item.id, c.id, !!item.multi)}>{c.label}</button>;
      })}</div>}
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
    <p className="remotea-card-kicker">Planning questions</p>
    <p className="footnote">The agent needs these answers to write the plan. Pick an option or type your own.</p>
    {interview.questions.map(q => <fieldset key={q.id} disabled={busy}>
      <legend>{q.header ? <strong>{q.header}: </strong> : null}{q.question}{q.multi && <small> (choose any)</small>}</legend>
      {!!q.options.length && <div className="remotea-choices">{orderOptions(q.options).map(o => {
        const on = (picked[q.id] ?? []).includes(o.label);
        return <button key={o.label} type="button" aria-pressed={on} title={o.description || undefined}
          className={"btn btn-sm" + (on ? " btn-primary" : recommended(o) ? " btn-recommended" : "")}
          onClick={() => toggle(q.id, o.label, q.multi)}>{o.label}</button>;
      })}</div>}
      <input aria-label={"Answer: " + q.question} placeholder={q.options.length ? "Or type an answer" : "Your answer"}
        value={typed[q.id] ?? ""} onChange={e => setTyped({ ...typed, [q.id]: e.target.value })}/>
    </fieldset>)}
    <button className="btn btn-primary btn-sm" disabled={!answers || busy} onClick={() => void send()}>{busy ? "Sending…" : "Send answers"}</button>
    {error && <p role="alert" className="work-error">{error}</p>}
  </section>;
}

/** The agent proposed changes to criteria you wrote: accept or keep each one, then the plan freezes and work starts. */
export function PlanReview({ run, onDone }: { run: RunDetail; onDone: () => void }) {
  const proposals = run.plan?.proposals;
  const [decisions, setDecisions] = useState<Record<string, PlanDecision>>(() =>
    Object.fromEntries((proposals?.criteria ?? []).map(c => [c.id, { accept: true, text: c.proposed ?? "" }])));
  const [verify, setVerify] = useState<"accept" | "keep">("accept"), [policy, setPolicy] = useState<"accept" | "keep">("accept");
  const [busy, setBusy] = useState(false), [error, setError] = useState("");
  if (!proposals || run.state_reason !== "plan_review" || run.state !== "needs_you") return null;
  const decide = (id: string, accept: boolean) => setDecisions(d => ({ ...d, [id]: { ...(d[id] ?? { text: "" }), accept } }));
  const send = async () => {
    setBusy(true); setError("");
    try { await post(`/remotea/runs/${run.id}/plan-review`, planReviewBody(proposals, decisions, verify, policy, run.state_revision)); onDone(); }
    catch (e) { setError(message(e)); } finally { setBusy(false); }
  };
  const choice = (on: boolean, label: string, good: boolean, onClick: () => void) =>
    <button type="button" aria-pressed={on} className={"btn btn-sm" + (on ? " btn-primary" : good ? " btn-recommended" : "")} onClick={onClick}>{label}</button>;
  return <section className="remotea-question remotea-plan-review" aria-label="Plan review">
    <p className="remotea-card-kicker">Plan review</p>
    <p className="footnote">The agent wants to change what you set. Accept or keep each change; nothing changes until you tap Start.</p>
    {(proposals.criteria ?? []).map(c => {
      const d = decisions[c.id] ?? { accept: true, text: c.proposed ?? "" };
      return <fieldset key={c.id} disabled={busy} className={"remotea-proposal change-" + c.change}>
        <legend><code>{c.id}</code> · {c.change === "remove" ? "remove this criterion" : "change the wording"}</legend>
        <p className="remotea-proposal-was"><span>Yours</span>{c.text}</p>
        {c.change === "reword" && (d.accept
          ? <label className="remotea-proposal-new"><span>Agent proposes (edit if you like)</span>
              <textarea rows={2} value={d.text} onChange={e => setDecisions(all => ({ ...all, [c.id]: { accept: true, text: e.target.value } }))}/></label>
          : <p className="remotea-proposal-new"><span>Agent proposes</span>{c.proposed}</p>)}
        {c.because && <blockquote className="remotea-proposal-why">{c.because}</blockquote>}
        <div className="remotea-choices">
          {choice(d.accept, c.change === "remove" ? "Remove it" : "Accept", true, () => decide(c.id, true))}
          {choice(!d.accept, "Keep mine", false, () => decide(c.id, false))}
        </div>
      </fieldset>;
    })}
    {proposals.verify && <fieldset disabled={busy} className="remotea-proposal">
      <legend>Verification</legend>
      <p className="remotea-proposal-was"><span>Yours</span>{proposals.verify.current.verify_id}</p>
      <p className="remotea-proposal-new"><span>Agent proposes</span>{proposals.verify.proposed.verify_id}</p>
      <div className="remotea-choices">{choice(verify === "accept", "Accept", true, () => setVerify("accept"))}{choice(verify === "keep", "Keep mine", false, () => setVerify("keep"))}</div>
    </fieldset>}
    {proposals.completion_policy && <fieldset disabled={busy} className="remotea-proposal">
      <legend>Done means</legend>
      <p className="remotea-proposal-was"><span>Yours</span>{POLICY_TEXT[proposals.completion_policy.current] ?? proposals.completion_policy.current}</p>
      <p className="remotea-proposal-new"><span>Agent proposes</span>{POLICY_TEXT[proposals.completion_policy.proposed] ?? proposals.completion_policy.proposed}</p>
      <div className="remotea-choices">{choice(policy === "accept", "Accept", true, () => setPolicy("accept"))}{choice(policy === "keep", "Keep mine", false, () => setPolicy("keep"))}</div>
    </fieldset>}
    <button className="btn btn-primary btn-sm" disabled={busy} onClick={() => void send()}>{busy ? "Starting…" : "Freeze the plan and start"}</button>
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

const BUBBLE_FOLD = 700;

/** One timeline entry as a chat bubble: the agent on the left, you on the right, Eridani's own events as system lines. */
export function Bubble({ entry }: { entry: TimelineEntry }) {
  const [open, setOpen] = useState(false);
  const long = entry.detail.length > BUBBLE_FOLD;
  const text = open || !long ? entry.detail : entry.detail.slice(0, BUBBLE_FOLD) + "…";
  const system = entry.who === "eridani" && (entry.kind === "state" || entry.kind === "command") && !entry.detail;
  return <li className={`remotea-bubble who-${entry.who} kind-${entry.kind} tone-${entry.tone}${system ? " system" : ""}`}>
    <div className="remotea-bubble-head"><strong>{entry.title}</strong> <small>{ago(entry.at)}</small></div>
    {entry.detail && <p>{text}</p>}
    {long && <button className="btn btn-ghost btn-sm" aria-expanded={open} onClick={() => setOpen(!open)}>
      {open ? <ChevronUp size={14}/> : <ChevronDown size={14}/>}{open ? "Less" : "Show all"}</button>}
  </li>;
}

/** The run detail as a chat. Rendered inside the app shell's RunPanel (a layout column, or a sheet on phones). */
export function RunDrawer({ id, onClose, onCollapse }: { id: string; onClose: () => void; onCollapse?: () => void }) {
  const [run, setRun] = useState<RunDetail | null>(null), [error, setError] = useState("");
  const [text, setText] = useState(""), [busy, setBusy] = useState("");
  const log = useRef<HTMLDivElement>(null), stuck = useRef(true);
  const refresh = useCallback(() => {
    api<RunDetail>(`/remotea/runs/${id}`).then(r => { setRun(r); setError(""); }).catch(e => setError(message(e)));
  }, [id]);
  useEffect(() => { setRun(null); stuck.current = true; refresh(); }, [refresh]);
  useAgentRefresh(refresh);
  // Chat behaviour: follow new entries only while the reader is already at the bottom.
  useLayoutEffect(() => { const el = log.current; if (el && stuck.current) el.scrollTop = el.scrollHeight; }, [run]);
  const onScroll = () => { const el = log.current; if (el) stuck.current = el.scrollHeight - el.scrollTop - el.clientHeight < 48; };
  const act = async (what: string, path: string, body: object) => {
    setBusy(what); setError("");
    try { await post(`/remotea/runs/${id}/${path}`, body); if (path === "message") setText(""); refresh(); }
    catch (e) { setError(message(e)); } finally { setBusy(""); }
  };
  const controls = run ? runControls(run) : null;
  const attention = run ? ATTENTION_STATES.includes(run.state) : false;
  const entries = run ? timeline(run) : [];
  return <section className="run-view remotea-chat" aria-label="Agent run">
    <header className="remotea-chat-head">
      <div className="remotea-chat-title">
        {run && <span className={"remotea-state run-state-" + run.state}>{runLabel(run.state)}</span>}
        <h2>{run?.task?.title ?? "Run"}</h2>
        {onCollapse && <button className="btn-icon" aria-label="Collapse run panel" title="Collapse" onClick={onCollapse}><PanelRightClose size={18}/></button>}
        <button className="btn-icon" aria-label="Close run" onClick={onClose}><X size={18}/></button>
      </div>
      {run && <p className="remotea-place">
        {run.tool}{run.branch && <> · <code>{run.branch}</code></>}{run.base_branch && <> from <code>{run.base_branch}</code></>}
        {run.host && <> · {run.host.name || "remotea"}{!run.host.connected && " (offline)"}</>} · turn {run.turn_count}/{run.turn_cap}
      </p>}
      {run && baseLine(run.base) && <p className={"remotea-place remotea-base" + (run.base?.fetch && !run.base.fetch.ok ? " stale" : "")}>
        <GitBranch size={13} aria-hidden="true"/> {baseLine(run.base)}
      </p>}
      {run?.reason_text && <div className={"remotea-reason-box" + (attention ? " attention" : "")} role={attention ? "status" : undefined}>
        <strong>{run.reason_text}</strong>
        {run.reason_detail && run.reason_detail !== run.reason_text && <p>{run.reason_detail}</p>}
      </div>}
      {run && <div className="remotea-chat-facts">
        <Criteria run={run}/>
        {!!run.plan?.steps?.length && <details className="remotea-plan">
          <summary>Plan · {run.plan.steps.filter(s => s.status === "done").length}/{run.plan.steps.length} steps</summary>
          <ol>{run.plan.steps.map(s => <li key={s.id} className={"step-" + s.status}>{s.title}</li>)}</ol>
        </details>}
        {run.completion_policy && run.completion_policy !== "code_ready" && <p className="footnote">{POLICY_TEXT[run.completion_policy] ?? run.completion_policy}</p>}
        {!!run.integration?.length && <ul className="remotea-integration" aria-label="Sub-task integration">{run.integration.map(i =>
          <li key={i.task_id}><GitBranch size={14} aria-hidden="true"/> {i.branch ? <code>{i.branch}</code> : "Sub-task"}: {i.reason}</li>)}</ul>}
        {!!linearOutcomes(run).length && <ul className="remotea-linear" aria-label="Linear sync">{linearOutcomes(run).map(o =>
          <li key={o.key} className={"tone-" + o.tone}>{o.url ? <a href={o.url} target="_blank" rel="noreferrer">{o.text}</a> : o.text}</li>)}</ul>}
      </div>}
      {error && <p role="alert" className="work-error">{error}</p>}
    </header>
    <div className="remotea-chat-log" ref={log} onScroll={onScroll}>
      {!run ? !error && <p role="status">Loading run…</p> : <>
        <ol className="remotea-timeline" aria-label="Run timeline">{entries.map((e, i) => <Bubble key={e.kind + e.at + i} entry={e}/>)}</ol>
        <PlanReview key={run.plan?.revision ?? 0} run={run} onDone={refresh}/>
        {run.questions.map(q => <QuestionCard key={q.id + q.status} question={q} run={run} onDone={refresh}/>)}
        {run.state === "interviewing" && run.interviews?.filter(i => i.status === "open").map(i =>
          <InterviewForm key={i.id} interview={i} onDone={refresh}/>)}
        {isSettled(run) && <p className="remotea-chat-end">{runLabel(run.state)}{run.finished_at ? " · " + ago(run.finished_at) : ""}</p>}
      </>}
    </div>
    {run && controls && !isSettled(run) && <footer className="remotea-composer">
      <textarea aria-label="Message the agent" rows={2} placeholder={controls.message.enabled ? "Message the agent" : controls.message.reason}
        disabled={!controls.message.enabled || !!busy} value={text} onChange={e => setText(e.target.value)}
        onKeyDown={e => {
          // Enter queues, Ctrl/Cmd+Enter steers, Shift+Enter is a new line.
          const action = composerAction(e);
          if (!action || !text.trim()) return;
          if (action === "queue" ? !controls.message.enabled : !controls.steer.enabled) { if (action === "queue") e.preventDefault(); return; }
          e.preventDefault();
          void act(action === "queue" ? "message" : "steer", "message", { text, mode: action });
        }}/>
      <div className="remotea-control-row">
        <button className="btn btn-sm" disabled={!controls.message.enabled || !text.trim() || !!busy} title="Queued for after the current turn (Enter)"
          onClick={() => void act("message", "message", { text, mode: "queue" })}>Message</button>
        <button className="btn btn-sm" disabled={!controls.steer.enabled || !text.trim() || !!busy} title="Interrupts the current turn (Ctrl+Enter)"
          onClick={() => void act("steer", "message", { text, mode: "steer" })}>Steer</button>
        <span className="remotea-control-gap"/>
        <button className="btn btn-sm btn-ghost" disabled={!controls.replan.enabled || !!busy} title={controls.replan.reason}
          onClick={() => void act("replan", "replan", { reason: text })}>{busy === "replan" ? "Re-planning…" : "Re-plan"}</button>
        {run.plan?.verify && <button className="btn btn-sm btn-ghost" disabled={!!busy || !run.surf} title="Run the plan's verification now"
          onClick={() => void act("verify", "verify", {})}>{busy === "verify" ? "Starting…" : "Verify now"}</button>}
        <button className="btn btn-sm btn-danger" disabled={!controls.stop.enabled || !!busy || controls.stop.label === "Stopping…"}
          onClick={() => void act("stop", "stop", { mode: controls.stop.mode })}><Square size={14}/>{busy === "stop" ? "Stopping…" : controls.stop.label}</button>
        <button className={"btn btn-sm" + (controls.approve.primary ? " btn-primary" : "")} disabled={!controls.approve.enabled || !!busy}
          title="Mark the task complete and end the run" onClick={() => void act("approve", "approve", {})}><Check size={14}/>Approve</button>
      </div>
      {controls.offline && <p className="footnote">The remotea server is offline; commands wait until it reconnects.</p>}
    </footer>}
  </section>;
}

export function TargetPicker({ need, onSaved, onCancel }: { need: NeedsTarget; onSaved: (id: string) => void; onCancel: () => void }) {
  const options = targetOptions(need);
  const [key, setKey] = useState(options[0]?.key ?? ""), [scope, setScope] = useState(() => defaultScope(need));
  const [busy, setBusy] = useState(false), [error, setError] = useState("");
  const save = async () => {
    const option = options.find(o => o.key === key);
    if (!option) return;
    setBusy(true); setError("");
    try {
      // Every choice is saved: for a record (its project by default), or "Only this task" binds the task itself.
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

/** The task detail's Run control: opens the Run dialog, or the task's active run in the side panel. Reads the
 * app-wide remotea state (no fetch of its own); hidden while remotea is off. */
export function RunTask({ taskId, canEdit, onOpenRun }: { taskId: string; canEdit: boolean; onOpenRun?: (id: string) => void }) {
  const { enabled, status, runsByTask } = useRemotea();
  const { openRun, openStart } = useRemoteaActions();
  if (!enabled) return null;
  const active = runsByTask.get(taskId) ?? null;
  const button = runButton(enabled, status?.hosts ?? [], active);
  if (!button.visible) return null;
  const open = (id: string) => { onOpenRun?.(id); openRun(id); };
  return <section className="detail-section remotea-run-task" aria-label="Agent run">
    <div className="remotea-control-row">
      <button className={"btn btn-sm" + (active ? "" : " btn-primary")} disabled={button.disabled || (!active && !canEdit)}
        title={button.reason || undefined} onClick={() => active ? open(active.id) : openStart(taskId)}>
        <Play size={14}/>{button.label}</button>
      {button.reason && <small>{button.reason}</small>}
    </div>
  </section>;
}
