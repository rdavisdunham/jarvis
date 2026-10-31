import { type ReactNode, useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { Check, ChevronDown, ChevronUp, Copy, GitBranch, MonitorSmartphone, PanelRightClose, Play, RefreshCw, Square, SquareTerminal, X } from "lucide-react";
import { api, ApiError, post } from "./api";
import { useAgentRefresh } from "./remotea-refresh";
import { useRemotea, useRemoteaActions } from "./remotea-context";
import { PendingControls } from "./PendingControls";
import { AttentionList, Facts, useNow } from "./RunsPanel";
import { SessionsLedger } from "./Sessions";
import { SettingRow, SettingsGroup } from "./SettingsLayout";
import {
  ago, ATTENTION_STATES, attentionGroups, baseLine, criteriaBody, reviewPacket, type Host, type ReviewPacket, buildAnswers, composerAction, criteriaEvidence, hostSummary, interviewAnswers, isSettled, latestAttempt, orderRuns, orderSessions,
  cutoverSummary, linearOutcomes, orderOptions, paritySummary, place, planReviewBody, POLICY_TEXT, questionMode, recommended, REMOTEA_POLL_MS,
  defaultScope, runButton, runControls, runLabel, stateLabel, targetOptions, timeline, unmergedSummary, verifyLine,
  isRevisionConflict, latestOnly, latestRun, RUN_CHANGED, runAction, type LatestOnly, type RunAction,
  type Interview, type NeedsTarget, type Parity, type PlanDecision, type RemoteaSession, type RemoteaStatus, type Run, type RunDetail,
  type RunQuestion, type TimelineEntry, type Turn, type UnmergedWork, versionLine, type MigrationTarget, type MigrationDraft, migrationBody, migrationDraft, migrationSummary,
} from "./remotea-model";
import { postAction } from "./api";
import { RemoteaReadiness, SetupCodeBox } from "./RemoteaSetup";
import { WorktreeMaintenance } from "./WorktreeMaintenance";
import { SwarmList } from "./Swarm";
import { anchoredScrollTop, earlierLine, mergeDetail, nextOffset, scrollAnchor, type FullTurn } from "./remotea-model";
import { focusSentText, terminalFallback, type FocusResult } from "./remotea-model";
import { APPROVAL_FACTS, freshnessFacts, type Fact } from "./remotea-model";
import { actorChip, hostBusyText, modelLabel, planApprovalBody, PLAN_STOPS, reviewFindings, reviewTitle, type PlanReviewRow } from "./remotea-model";

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
  const unmerged = useUnmerged();
  return <div className="remotea-page">
    <section className={"remotea-hosts tone-" + hosts.tone} aria-label="remotea server">
      <MonitorSmartphone size={18} aria-hidden="true"/><span>{hosts.text}
        {(status?.hosts ?? []).filter(h => h.connected).map(h => ({ h, line: versionLine(h) })).filter(x => x.line).map(({ h, line }) =>
          <small key={h.id} className="remotea-host-build"> · {line!.text}{line!.dirty && <span className="remotea-dirty" title="Built with uncommitted changes">dirty</span>}</small>)}
      </span>
      <button className="btn-icon" aria-label="Refresh remotea" disabled={loading} onClick={refresh}><RefreshCw size={16}/></button>
    </section>
    {hosts.warning && <p className="remotea-warning" role="alert">{hosts.warning}</p>}
    {error && <p className="error-banner" role="alert">{error}</p>}
    <RemoteaReadiness onlyWhenNotReady/>
    <PendingControls/>
    <RunList onOpen={openRun} hosts={status?.hosts ?? []} unmerged={unmerged}/>
    <SwarmList onOpenRun={openRun}/>
    <UnmergedList items={unmerged}/>
    <WorktreeMaintenance/>
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
  const [code, setCode] = useState("");
  const [busy, setBusy] = useState(false), [error, setError] = useState(""), [copied, setCopied] = useState("");
  useEffect(() => { void api<RemoteaStatus>("/remotea/status").then(setStatus).catch(e => setError(e.message)); }, []);
  const change = async (path: string, body?: unknown) => {
    setBusy(true); setError("");
    try {
      const next = await post<RemoteaStatus>(path, body);
      setStatus(next); setToken(next.token ?? ""); setCode(next.setup_code ?? ""); onChanged();
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  };
  const copy = (value: string, label: string) => copyText(value).then(() => setCopied(label))
    .catch(() => setError("Copy is unavailable here. Select the text to copy it."));
  const hosts = hostSummary(status?.hosts ?? []);
  return <SettingsGroup className="remotea-settings" icon={<MonitorSmartphone size={18}/>} title="remotea"
    description="Follow Claude Code and Codex sessions running in your remotea server from this app.">
    {error && <p className="error-banner" role="alert">{error}</p>}
    <SettingRow as="label" label="Show remotea" hint="Adds the remotea page and links your remotea server. Off by default. Turning it on shows a one-time setup code; turning it off revokes its keys. Personal workspace only.">
      <input type="checkbox" checked={!!status?.enabled} disabled={!status || busy || readOnly}
        onChange={e => void change(e.target.checked ? "/remotea/enable" : "/remotea/disable", e.target.checked ? {} : undefined)}/>
    </SettingRow>
    {status?.enabled && <>
      <SettingRow label="Server" hint={status.key ? `Link key ${status.key.prefix}… · expires ${new Date(status.key.expires_at).toLocaleDateString()}` : "No active link key."}>
        <span className={"remotea-host-chip tone-" + hosts.tone}>{hosts.tone === "connected" ? "Connected" : hosts.tone === "never" ? "Never connected" : ago(status.hosts.map(h => h.last_seen_at).filter(Boolean).sort().at(-1) ?? null)}</span>
      </SettingRow>
      {status.hosts.some(h => h.status === "degraded") && <p className="remotea-warning" role="note">Today's agent text limit is reached: new turns are recorded without their text until tomorrow (UTC).</p>}
      {hosts.warning && <p className="remotea-warning" role="alert">{hosts.warning}</p>}
      {status.hosts.map(h => ({ h, line: versionLine(h) })).filter(x => x.line).map(({ h, line }) =>
        <SettingRow key={h.id} label={h.name || h.machine || "remotea"}
          hint={line!.update ? `The latest release is ${line!.update.latest}; this server is more than one release behind.` : undefined}>
          <span className={"remotea-version" + (line!.update ? " outdated" : "")}>{line!.update?.url
            ? <a href={line!.update.url} target="_blank" rel="noreferrer">{line!.text}</a> : line!.text}
            {line!.dirty && <span className="remotea-dirty" title="Built with uncommitted changes">dirty</span>}</span>
        </SettingRow>)}
      {status.mcp_key && <SettingRow label="Agent tools key"
        hint={`MCP key ${status.mcp_key.prefix}… · expires ${new Date(status.mcp_key.expires_at).toLocaleDateString()}. The setup code gives it to the coding agents a run starts; a new setup code replaces it.`}>
        <span className="remotea-host-chip">{status.mcp_key.last_used_at ? "Used " + ago(status.mcp_key.last_used_at) : "Not used yet"}</span>
      </SettingRow>}
      {code && <SetupCodeBox code={code} onDone={() => { setCode(""); setToken(""); }}/>}
      {!readOnly && <div className="settings-form-actions">
        <button className="btn btn-ghost btn-sm" disabled={busy} onClick={() => void change("/remotea/setup-code", {})}>New setup code</button>
      </div>}
      <RemoteaReadiness/>
      <MigrationSettings readOnly={readOnly}/>
      {token && <details className="remotea-older-key" open={!code}>
        <summary>Older remotea without <code>eridani setup</code>: the link key alone</summary>
        <div className="bot-secret" role="region" aria-label="remotea link key">
          <strong>Copy this key now</strong><p>This is the only time it will be shown. On your desktop, run the command below and paste the key when asked.</p>
          <label className="sr-only" htmlFor="remotea-secret">remotea link key</label>
          <input id="remotea-secret" readOnly type="password" value={token} autoComplete="off" onFocus={e => e.target.select()}/>
          <div className="bot-key-actions">
            <button className="btn btn-primary btn-sm" onClick={() => void copy(token, "key")}>{copied === "key" ? <Check size={14}/> : <Copy size={14}/>} {copied === "key" ? "Copied" : "Copy key"}</button>
            <button className="btn btn-ghost btn-sm" onClick={() => setToken("")}>Done</button>
          </div>
        </div>
      </details>}
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

/** The page's runs, grouped by the next human action (attentionGroups); the five most recent settled runs close it. */
export function RunList({ onOpen, hosts, unmerged, preloaded }: { onOpen: (id: string) => void; hosts: Host[]; unmerged: UnmergedWork[];
  /** Runs already in hand (tests); the list still loads. */
  preloaded?: Run[] }) {
  const [runs, setRuns] = useState<Run[] | null>(preloaded ?? null), [error, setError] = useState("");
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
    <AttentionList groups={attentionGroups([...active, ...recent], [], hosts, unmerged)} onOpen={onOpen}/>
  </section>;
}

function useUnmerged() {
  const [items, setItems] = useState<UnmergedWork[]>([]);
  const refresh = useCallback(() => { api<{ items: UnmergedWork[] }>("/remotea/unmerged").then(r => setItems(r.items)).catch(() => setItems([])); }, []);
  useEffect(() => { refresh(); }, [refresh]);
  useAgentRefresh(refresh);
  return items;
}

function UnmergedList({ items }: { items: UnmergedWork[] }) {
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
export function PlanReview({ run, onDone, onConflict }: { run: RunDetail; onDone: () => void; onConflict?: () => void }) {
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
    catch (e) {
      if (!isRevisionConflict(e)) setError(message(e));
      else if (onConflict) onConflict();
      else { setError(RUN_CHANGED); onDone(); }
    } finally { setBusy(false); }
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
      <p className="remotea-proposal-was"><span>{proposals.verify.current ? "Yours" : "Plan had"}</span>{proposals.verify.current?.verify_id ?? "no verify command"}</p>
      <p className="remotea-proposal-new"><span>{proposals.verify.current ? "Agent proposes" : "Eridani picked"}</span>{proposals.verify.proposed.verify_id}
        {proposals.verify.options && proposals.verify.options.length > 1 && <small> · also offered: {proposals.verify.options.filter(o => o !== proposals.verify!.proposed.verify_id).join(", ")}</small>}</p>
      {proposals.verify.because && <p className="remotea-proposal-why">{proposals.verify.because}</p>}
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

/** Hyperplan's final-plan stop (needs_you(plan_approval)): the frozen plan and the review history; Approve starts the
 * work (or hands the plan to the worker), Revise with note sends it back to the planner. */
export function PlanApproval({ run, onDone, onConflict }: { run: RunDetail; onDone: () => void; onConflict?: () => void }) {
  const [note, setNote] = useState(""), [busy, setBusy] = useState<"" | "approve" | "revise">(""), [error, setError] = useState("");
  if (run.state !== "needs_you" || run.state_reason !== "plan_approval") return null;
  const send = async (action: "approve" | "revise") => {
    setBusy(action); setError("");
    try { await post(`/remotea/runs/${run.id}/plan-review`, planApprovalBody(action, note, run.state_revision)); setNote(""); onDone(); }
    catch (e) {
      if (!isRevisionConflict(e)) setError(message(e));
      else if (onConflict) onConflict();
      else { setError(RUN_CHANGED); onDone(); }
    } finally { setBusy(""); }
  };
  const plan = run.plan, reviews = run.reviews ?? [], worker = run.roles?.worker;
  return <section className="remotea-question remotea-plan-approval" aria-label="Final plan approval">
    <p className="remotea-card-kicker">Final plan</p>
    <p className="footnote">{reviews.length ? `Reviewed ${reviews.length} time${reviews.length === 1 ? "" : "s"}. ` : ""}Approve to start the work
      {worker ? ` with ${modelLabel(worker.harness, worker.model)} in a fresh session` : ""}, or send it back to the planner with a note.</p>
    {plan ? <>
      <p><strong>{plan.title}</strong> <small>revision {plan.revision}</small></p>
      {plan.prd_text && <details open><summary>Plan</summary><p className="remotea-plan-text">{plan.prd_text}</p></details>}
      {!!plan.steps.length && <details><summary>Steps · {plan.steps.length}</summary><ol>{plan.steps.map(s => <li key={s.id}>{s.title}</li>)}</ol></details>}
      {!!plan.criteria.length && <details open><summary>Success criteria · {plan.criteria.length}</summary>
        <ul>{plan.criteria.map(c => <li key={c.id}>{c.text}{c.kind === "verify" && <small> · needs {c.verify_id} to pass</small>}</li>)}</ul></details>}
    </> : <p className="footnote">The plan is still loading.</p>}
    {!!reviews.length && <ul className="remotea-review-history" aria-label="Review history">{reviews.map(r => <li key={r.id}>{reviewTitle(r)}</li>)}</ul>}
    <label className="remotea-proposal-new"><span>Note for the planner (to revise)</span>
      <textarea rows={3} maxLength={4000} value={note} disabled={!!busy} onChange={e => setNote(e.target.value)}/></label>
    <div className="remotea-control-row">
      <button className="btn btn-sm" disabled={!!busy || !note.trim()} onClick={() => void send("revise")}>{busy === "revise" ? "Sending…" : "Revise with note"}</button>
      <button className="btn btn-sm btn-primary" disabled={!!busy} onClick={() => void send("approve")}><Check size={14}/>{busy === "approve" ? "Approving…" : "Approve"}</button>
    </div>
    {error && <p role="alert" className="work-error">{error}</p>}
  </section>;
}

const DECISION_TEXT = { accepted: "accepted", rejected: "rejected", deferred: "deferred" } as const;

/** A review round's findings (severity, area, issue, suggestion) with the planner's answer to each, and the
 * reviewer's questions for you, folded under the round's timeline entry. */
export function ReviewFindings({ review }: { review: PlanReviewRow }) {
  const findings = reviewFindings(review), questions = review.questions ?? [];
  if (!findings.length && !questions.length && review.via !== "turn") return null;
  const parts = [findings.length ? `${findings.length} finding${findings.length === 1 ? "" : "s"}` : "",
    questions.length ? `${questions.length} question${questions.length === 1 ? "" : "s"} for you` : ""].filter(Boolean);
  return <details className="remotea-review-findings">
    <summary>{parts.join(" · ") || "Details"}</summary>
    {review.via === "turn" && <p>The reviewer did not file a structured review; its reply is the review.</p>}
    {!!findings.length && <ul>{findings.map(f => <li key={f.id} className={"severity-" + f.severity}>
      <strong>{f.severity}</strong>{f.area ? " · " + f.area : ""} <code>{f.id}</code>
      <p>{f.issue}</p>
      {f.suggestion && <p>Suggestion: {f.suggestion}</p>}
      {f.resolution && <p className="resolution">Planner {DECISION_TEXT[f.resolution.decision] ?? f.resolution.decision}{f.resolution.note ? ": " + f.resolution.note : ""}</p>}
    </li>)}</ul>}
    {!!questions.length && <><p>Questions for you (the planner decides which to ask):</p><ul>{questions.map((q, i) => <li key={i}>{q}</li>)}</ul></>}
  </details>;
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

const PACKET_STATUS = { met: "Met", "not met": "Not met", unknown: "Unknown" } as const;

/** E-Q3: everything a reviewer needs beside Approve, from reviewPacket. Read-only in needs_you and settled runs. */
export function ReviewPacketCard({ packet, children }: { packet: ReviewPacket; children?: ReactNode }) {
  return <details className={"remotea-plan review-packet" + (packet.readOnly ? " read-only" : "")} open={!packet.readOnly || undefined}
    aria-label="Review packet">
    <summary>Review · {packet.text}</summary>
    {packet.decision && <p className={"review-decision kind-" + packet.decision.kind}>{packet.decision.text}</p>}
    <h4>Success criteria · {packet.met}/{packet.total} met</h4>
    {packet.judgedEarlierPlan && <p className="footnote attention">Judged against an earlier plan: the criteria changed since. Judge again to check them.</p>}
    {packet.criteria.length ? <ul>{packet.criteria.map(c => <li key={c.id} className={"criterion-" + c.status.replace(" ", "_")}>
      <strong>{PACKET_STATUS[c.status]}</strong> {c.text}{c.evidence && <p>{c.evidence}</p>}
    </li>)}</ul> : <p className="footnote">No success criteria: the judge cannot complete this run; you decide.</p>}
    <h4>Changes</h4>
    <p>{packet.prs.length ? packet.prs.map((p, i) => <span key={p.url}>{i ? ", " : ""}<a href={p.url} target="_blank" rel="noreferrer">{p.label}</a></span>) : null}
      {packet.prs.length ? " " + packet.prState : packet.changesLine}
      {packet.branch && <> · <code>{packet.branch}</code></>}{packet.worktree && <small> in <code>{packet.worktree}</code></small>}</p>
    <h4>Tests</h4>
    <p>{packet.testsLine}</p>
    {!!packet.tests?.markers.length && <ul className="review-markers">{packet.tests.markers.map(m => <li key={m}>{m}</li>)}</ul>}
    {!!packet.questions.length && <><h4>Open questions</h4><ul>{packet.questions.map((q, i) => <li key={i}>{q}</li>)}</ul></>}
    {packet.reason && <><h4>The judge's reason</h4><p className="review-reason">{packet.reason}</p></>}
    {children}
  </details>;
}

/** ba56714a: edit the success criteria (and the verify command) of a parked run, then Judge again. */
export function CriteriaEditor({ run, onDone, onConflict, onCancel }: { run: RunDetail; onDone: () => void; onConflict: () => void; onCancel: () => void }) {
  const [rows, setRows] = useState(() => (run.plan?.criteria ?? []).map(c => ({ id: c.id as string | undefined, text: c.text, kind: c.kind ?? "text" })));
  const options = run.verify_options ?? [], current = run.plan?.verify?.verify_id ?? null;
  const [verifyId, setVerifyId] = useState(current ?? "");
  const [busy, setBusy] = useState(""), [error, setError] = useState("");
  const body = criteriaBody(rows, verifyId || null, current);
  const set = (i: number, values: Partial<(typeof rows)[number]>) => setRows(all => all.map((r, j) => j === i ? { ...r, ...values } : r));
  const save = async (judge: boolean) => {
    if (!body) return;
    setBusy(judge ? "judge" : "save"); setError("");
    try {
      const saved = await postAction<RunDetail>(`/remotea/runs/${run.id}/criteria`, { ...body, expected_revision: run.state_revision });
      if (judge) await postAction(`/remotea/runs/${run.id}/judge`, { expected_revision: saved.state_revision });
      onDone();
    } catch (e) { if (isRevisionConflict(e)) onConflict(); else setError(message(e)); } finally { setBusy(""); }
  };
  return <section className="remotea-question criteria-editor" aria-label="Edit criteria">
    <p className="remotea-card-kicker">Edit criteria</p>
    <p className="footnote">Your edit freezes the next plan revision; the agent is judged against it from now on.</p>
    {rows.map((r, i) => <fieldset key={r.id ?? "new-" + i} disabled={!!busy}>
      <textarea aria-label={"Criterion " + (i + 1)} rows={2} value={r.text} onChange={e => set(i, { text: e.target.value })}/>
      <div className="remotea-choices">
        {(verifyId || r.kind === "verify") && <label className="remotea-toggle"><input type="checkbox" checked={r.kind === "verify"}
          onChange={e => set(i, { kind: e.target.checked ? "verify" : "text" })}/><span>Needs the verify command to pass</span></label>}
        <button type="button" className="btn btn-ghost btn-sm" onClick={() => setRows(all => all.filter((_, j) => j !== i))}>Remove</button>
      </div>
    </fieldset>)}
    <button type="button" className="btn btn-ghost btn-sm" disabled={!!busy} onClick={() => setRows(all => [...all, { id: undefined, text: "", kind: "text" }])}>Add criterion</button>
    {!!options.length && <label className="remotea-verify-line">Verify command
      <select aria-label="Verify command" value={verifyId} disabled={!!busy} onChange={e => setVerifyId(e.target.value)}>
        {!current && <option value="">none</option>}
        {options.map(o => <option key={o} value={o}>{o}</option>)}
      </select></label>}
    <div className="remotea-control-row">
      <button className="btn btn-sm btn-ghost" disabled={!!busy} onClick={onCancel}>Cancel</button>
      <button className="btn btn-sm" disabled={!body || !!busy} onClick={() => void save(false)}>{busy === "save" ? "Saving…" : "Save"}</button>
      <button className="btn btn-sm btn-primary" disabled={!body || !!busy} onClick={() => void save(true)}>{busy === "judge" ? "Judging…" : "Save and judge again"}</button>
    </div>
    {error && <p role="alert" className="work-error">{error}</p>}
  </section>;
}

const BUBBLE_FOLD = 700;

const CLIPPED_HINT = "Cut at 4,000 characters for display. Full response loads the rest.";
const TRUNCATED_HINT = "remotea cut this final response before sending it; the rest was never stored.";

/** One timeline entry as a chat bubble: the agent on the left, you on the right, Eridani's own events as system lines.
 * An agent response says how complete it is: Clipped (cut for display; Full response fetches the whole turn once and
 * keeps it), Truncated by host (the rest never reached Eridani) or Expired (retention deleted it since the page loaded). */
export function Bubble({ entry, runId }: { entry: TimelineEntry; runId?: string }) {
  const [open, setOpen] = useState(false), [full, setFull] = useState<string | null>(null);
  const [loading, setLoading] = useState(false), [expired, setExpired] = useState(""), [error, setError] = useState("");
  const turn = entry.turn;
  const fetchable = !!turn?.clipped && full === null && !expired && !!runId;
  const body = full ?? entry.detail;
  const long = body.length > BUBBLE_FOLD;
  const text = open || !long ? body : body.slice(0, BUBBLE_FOLD) + "…";
  const system = entry.who === "eridani" && (entry.kind === "state" || entry.kind === "command") && !entry.detail;
  const loadFull = async () => {
    setLoading(true); setError("");
    try {
      const whole = await api<FullTurn>(`/remotea/runs/${runId}/turns/${turn!.id}`);
      setFull(whole.text); setOpen(true);
    } catch (e) {
      if (e instanceof ApiError && e.code === "TURN_EXPIRED") setExpired(e.message); else setError(message(e));
    } finally { setLoading(false); }
  };
  return <li data-entry={entry.id} className={`remotea-bubble who-${entry.who} kind-${entry.kind} tone-${entry.tone}${system ? " system" : ""}`}>
    <div className="remotea-bubble-head"><strong>{entry.title}</strong>
      {turn?.clipped && full === null && !expired && <span className="remotea-label label-clipped" title={CLIPPED_HINT}>Clipped</span>}
      {turn?.truncated && <span className="remotea-label label-truncated" title={TRUNCATED_HINT}>Truncated by host</span>}
      {expired && <span className="remotea-label label-expired" title={expired}>Expired</span>}
      <small>{ago(entry.at)}</small></div>
    {entry.facts && <Facts facts={entry.facts} label="Verification facts"/>}
    {entry.detail && <p>{text}</p>}
    {entry.review && <ReviewFindings review={entry.review}/>}
    {fetchable ? <button className="btn btn-ghost btn-sm" disabled={loading} title={CLIPPED_HINT} onClick={() => void loadFull()}>
      <ChevronDown size={14}/>{loading ? "Loading…" : "Full response"}</button>
      : long && <button className="btn btn-ghost btn-sm" aria-expanded={open} onClick={() => setOpen(!open)}>
        {open ? <ChevronUp size={14}/> : <ChevronDown size={14}/>}{open ? "Less" : "Show all"}</button>}
    {expired && <p className="remotea-bubble-note">{expired}</p>}
    {error && <p role="alert" className="work-error">{error}</p>}
  </li>;
}

/** Where the run's terminal is, when remotea could not bring its tab forward: host, worktree, branch, and commands
 * to copy. */
export function TerminalFallback({ run, result }: { run: Pick<Run, "host" | "worktree" | "branch" | "surf">; result: FocusResult | null }) {
  const info = terminalFallback(run, result);
  const [copied, setCopied] = useState("");
  return <div className="remotea-terminal-fallback" role="status">
    {info.message && <p><strong>{info.message}</strong></p>}
    <p>The terminal is on <strong>{info.host}</strong>
      {info.worktree && <> in <code>{info.worktree}</code></>}{info.branch && <> on branch <code>{info.branch}</code></>}.</p>
    {info.commands.map(c => <div key={c.command} className="remotea-copy-row">
      <small>{c.label}</small>
      <code>{c.command}</code>
      <button className="btn btn-ghost btn-sm" aria-label={"Copy: " + c.command}
        onClick={() => void copyText(c.command).then(() => setCopied(c.command), () => setCopied(""))}>
        <Copy size={14}/>{copied === c.command ? "Copied" : "Copy"}</button>
    </div>)}
  </div>;
}

/** The run detail as a chat. Rendered inside the app shell's RunPanel (a layout column, or a sheet on phones). */
export function RunDrawer({ id, onClose, onCollapse, preloaded }: {
  id: string; onClose: () => void; onCollapse?: () => void;
  /** A run already in hand (tests); the drawer still loads `id` and binds its controls to whichever run is shown. */
  preloaded?: RunDetail;
}) {
  const [run, setRun] = useState<RunDetail | null>(preloaded ?? null), [error, setError] = useState(""), [notice, setNotice] = useState("");
  const [text, setText] = useState(""), [busy, setBusy] = useState("");
  // "Open terminal in remotea": the host's answer (null until asked); a refusal shows where the terminal is instead.
  const [focus, setFocus] = useState<FocusResult | null>(null);
  // Older pages of turns and commands the reader loaded (merged); kept across polls, which reload only the newest page.
  const [earlier, setEarlier] = useState<RunDetail | null>(null), [loadingEarlier, setLoadingEarlier] = useState(false);
  const log = useRef<HTMLDivElement>(null), stuck = useRef(true);
  const anchor = useRef<{ id: string; offset: number } | null>(null);
  // Only the newest request for this drawer's run may install its response (out-of-order replies are dropped).
  const requests = useRef<LatestOnly | null>(null);
  requests.current ??= latestOnly();
  const refresh = useCallback(() => {
    void latestRun(requests.current!, id, signal => api<RunDetail>(`/remotea/runs/${id}`, { signal })).then(result => {
      if (!result) return;
      if ("run" in result) { setRun(result.run); setError(""); } else setError(message(result.error));
    });
  }, [id]);
  useEffect(() => { stuck.current = true; setEarlier(null); refresh(); return () => requests.current?.cancel(); }, [refresh]);
  useAgentRefresh(refresh);
  // Chat behaviour: follow new entries only while the reader is already at the bottom.
  useLayoutEffect(() => { const el = log.current; if (el && stuck.current) el.scrollTop = el.scrollHeight; }, [run]);
  const onScroll = () => { const el = log.current; if (el) stuck.current = el.scrollHeight - el.scrollTop - el.clientHeight < 48; };
  // Load earlier keeps the entry that was first on screen where it was (older rows land above it, or between older
  // history rows above it), instead of letting the content jump.
  const entryRows = (el: HTMLDivElement) => [...el.querySelectorAll<HTMLElement>("[data-entry]")].map(row => {
    const box = row.getBoundingClientRect();
    return { id: row.dataset.entry!, top: box.top, bottom: box.bottom };
  });
  useLayoutEffect(() => {
    const el = log.current, mark = anchor.current;
    anchor.current = null;
    if (!el || !mark) return;
    const row = entryRows(el).find(r => r.id === mark.id);
    if (row) el.scrollTop = anchoredScrollTop(el.scrollTop, mark, row.top - el.getBoundingClientRect().top);
  }, [earlier]);
  // Controls act on the run on screen, and only once it is the run the panel asked for.
  const bound = !!run && run.id === id;
  const off = !!busy || !bound;
  const changed = () => { setNotice(RUN_CHANGED); refresh(); };
  const act = async (what: RunAction, values: Record<string, unknown> = {}) => {
    const action = runAction(run, id, what, values);
    if (!action) return;
    setBusy(what); setError(""); setNotice("");
    // Routes with a receipt get one action_id and a retry after a lost response; Verify now has no receipt, so no retry.
    try { await (what === "verify" ? post(action.url, action.body) : postAction(action.url, action.body)); if (what === "message" || what === "steer") setText(""); refresh(); }
    catch (e) { if (isRevisionConflict(e)) changed(); else setError(message(e)); }
    finally { setBusy(""); }
  };
  // Not a run action: it changes nothing on the run and stores no receipt, so a plain post (a repeat only raises the
  // tab again). The route waits a few seconds for the host's answer.
  const openTerminal = async () => {
    if (!run || !bound) return;
    setBusy("focus"); setError(""); setNotice(""); setFocus(null);
    try {
      const result = await post<FocusResult>(`/remotea/runs/${run.id}/focus`);
      setFocus(result);
      if (result.sent) setNotice(focusSentText(run, result));
    } catch (e) { setError(message(e)); }
    finally { setBusy(""); }
  };
  const shown = run && earlier ? mergeDetail(run, earlier) : run;
  const loadEarlier = async () => {
    const offset = shown && nextOffset(shown);
    if (!shown || offset === null || !bound) return;
    const el = log.current;
    anchor.current = el ? scrollAnchor(entryRows(el), el.getBoundingClientRect().top) : null;
    stuck.current = false;
    setLoadingEarlier(true);
    try {
      const page = await api<RunDetail>(`/remotea/runs/${shown.id}?offset=${offset}&limit=${shown.more!.limit}`);
      if (page.id === id) setEarlier(prev => prev ? mergeDetail(prev, page) : page);
    } catch (e) { anchor.current = null; setError(message(e)); }
    finally { setLoadingEarlier(false); }
  };
  const controls = run ? runControls(run) : null;
  const attention = run ? ATTENTION_STATES.includes(run.state) : false;
  const entries = shown ? timeline(shown) : [];
  const pagedOut = earlierLine(shown?.more);
  const verify = run ? verifyLine(run) : null;
  const [verifyPick, setVerifyPick] = useState("");
  const packet = run ? reviewPacket(run) : null;
  // Edit criteria / Judge again: only while the run is parked for you (not during a plan review).
  const parked = !!run && bound && !run.freestyle && (run.state === "awaiting_approval" || (run.state === "needs_you" && !PLAN_STOPS.includes(run.state_reason)));
  const actor = run ? actorChip(run) : null, busyHost = run ? hostBusyText(run) : null;
  const [editing, setEditing] = useState(false);
  // Host lease, transcript, state, verification, workspace and unconfirmed controls, each with its own age (E-S6).
  const now = useNow();
  const fresh = run ? freshnessFacts(run, null, now) : [];
  return <section className="run-view remotea-chat" aria-label="Agent run">
    <header className="remotea-chat-head">
      <div className="remotea-chat-title">
        {run && <span className={"remotea-state run-state-" + run.state}>{runLabel(run.state)}</span>}
        {actor && <span className="run-actor" title="Who acts on this run now">{actor}</span>}
        <h2>{run?.task?.title ?? "Run"}</h2>
        {onCollapse && <button className="btn-icon" aria-label="Collapse run panel" title="Collapse" onClick={onCollapse}><PanelRightClose size={18}/></button>}
        <button className="btn-icon" aria-label="Close run" onClick={onClose}><X size={18}/></button>
      </div>
      {run && <p className="remotea-place">
        {run.tool}{run.branch && <> · <code>{run.branch}</code></>}{run.base_branch && <> from <code>{run.base_branch}</code></>}
        {run.host && <> · {run.host.name || "remotea"}{!run.host.connected && " (offline)"}</>} · turn {run.turn_count}/{run.turn_cap}
      </p>}
      {run && <div className="remotea-control-row remotea-terminal-row">
        <button className="btn btn-sm btn-ghost" disabled={off || !run.surf}
          title={run.surf ? "Bring this run's terminal tab to the front in the remotea app" : "This run has no terminal yet"}
          onClick={() => void openTerminal()}>
          <SquareTerminal size={14}/>{busy === "focus" ? "Opening…" : "Open terminal in remotea"}</button>
      </div>}
      {run && focus && !focus.sent && <TerminalFallback run={run} result={focus}/>}
      {run && baseLine(run.base) && <p className={"remotea-place remotea-base" + (run.base?.fetch && !run.base.fetch.ok ? " stale" : "")}>
        <GitBranch size={13} aria-hidden="true"/> {baseLine(run.base)}
      </p>}
      {run && !isSettled(run) && <Facts facts={fresh} label="How current this is"/>}
      {run?.reason_text && <div className={"remotea-reason-box" + (attention ? " attention" : "")} role={attention ? "status" : undefined}>
        <strong>{run.reason_text}</strong>
        {run.reason_detail && run.reason_detail !== run.reason_text && <p>{run.reason_detail}</p>}
        {busyHost && <p>{busyHost}</p>}
      </div>}
      {run && !isSettled(run) && <div className="remotea-chat-modes">
        <label className="remotea-toggle" title="Talk to the agent directly: turns are recorded but not verified, judged or counted toward the cap.">
          <input type="checkbox" checked={!!run.freestyle} disabled={off || run.state === "stopping"}
            onChange={e => void act("freestyle", { enabled: e.target.checked })}/>
          <span>Freestyle</span><small>{run.freestyle ? "judge and prompts off" : "judge on"}</small>
        </label>
        {verify?.text && !run.freestyle && <span className={"remotea-verify-line" + (verify.current ? "" : " attention")}>
          {verify.text}
          {verify.canSet && <>
            <select aria-label="Verify command" disabled={!bound} value={verifyPick || verify.current || verify.options[0]} onChange={e => setVerifyPick(e.target.value)}>
              {verify.options.map(o => <option key={o} value={o}>{o}</option>)}
            </select>
            <button className="btn btn-sm" disabled={off} onClick={() => void act("verify-config",
              { verify_id: verifyPick || verify.current || verify.options[0] })}>
              {busy === "verify-config" ? "Setting…" : verify.current ? "Change" : "Set verify"}</button>
          </>}
        </span>}
      </div>}
      {run && <div className="remotea-chat-facts">
        {packet ? <ReviewPacketCard packet={packet}>
          {parked && !editing && <div className="remotea-control-row">
            <button className="btn btn-sm" disabled={off} onClick={() => setEditing(true)}>Edit criteria</button>
            <button className="btn btn-sm" disabled={off || !run.turn_seq || !run.plan?.criteria?.length} title="Check the last turn again against the criteria as they are now"
              onClick={() => void act("judge")}>{busy === "judge" ? "Judging…" : "Judge again"}</button>
          </div>}
        </ReviewPacketCard> : <Criteria run={run}/>}
        {parked && editing && <CriteriaEditor key={run.plan?.revision ?? 0} run={run} onCancel={() => setEditing(false)}
          onDone={() => { setEditing(false); refresh(); }} onConflict={() => { setEditing(false); changed(); }}/>}
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
      {run && !bound && <p role="status" className="footnote">Loading the selected run…</p>}
      {notice && <p role="status" className="remotea-reason">{notice}</p>}
      {error && <p role="alert" className="work-error">{error}</p>}
    </header>
    <div className="remotea-chat-log" ref={log} onScroll={onScroll}>
      {!run ? !error && <p role="status">Loading run…</p> : <>
        {shown?.retention?.expired && <p className="remotea-log-note">
          <span className="remotea-label label-expired">Expired</span> This run ended more than 90 days ago, so its turns and commands were deleted.</p>}
        {pagedOut && <div className="remotea-log-note">
          <span className="remotea-label label-paged">Paged out</span> {pagedOut}
          <button className="btn btn-ghost btn-sm" disabled={loadingEarlier || !bound} onClick={() => void loadEarlier()}>
            <ChevronUp size={14}/>{loadingEarlier ? "Loading…" : "Load earlier"}</button>
        </div>}
        <ol className="remotea-timeline" aria-label="Run timeline">{entries.map(e => <Bubble key={e.id} entry={e} runId={run.id}/>)}</ol>
        {bound && <>
          <PlanReview key={run.plan?.revision ?? 0} run={run} onDone={refresh} onConflict={changed}/>
          <PlanApproval key={"approval-" + run.plan_revision} run={run} onDone={refresh} onConflict={changed}/>
          {run.questions.map(q => <QuestionCard key={q.id + q.status} question={q} run={run} onDone={refresh}/>)}
          {run.state === "interviewing" && run.interviews?.filter(i => i.status === "open").map(i =>
            <InterviewForm key={i.id} interview={i} onDone={refresh}/>)}
        </>}
        {isSettled(run) && <p className="remotea-chat-end">{runLabel(run.state)}{run.finished_at ? " · " + ago(run.finished_at) : ""}</p>}
      </>}
    </div>
    {run && controls && !isSettled(run) && <footer className="remotea-composer">
      {/* What Approve rests on: the verification, its workspace and sandbox, the host, anything unconfirmed. */}
      {run.state === "awaiting_approval" && <Facts facts={fresh.filter(APPROVAL_FACTS)} label="What approval rests on" className="remotea-approve-facts"/>}
      <textarea aria-label="Message the agent" rows={2} placeholder={controls.message.enabled ? "Message the agent" : controls.message.reason}
        disabled={!controls.message.enabled || off} value={text} onChange={e => setText(e.target.value)}
        onKeyDown={e => {
          // Enter queues, Ctrl/Cmd+Enter steers, Shift+Enter is a new line.
          const action = composerAction(e);
          if (!action || !text.trim() || off) return;
          if (action === "queue" ? !controls.message.enabled : !controls.steer.enabled) { if (action === "queue") e.preventDefault(); return; }
          e.preventDefault();
          void act(action === "queue" ? "message" : "steer", { text, mode: action });
        }}/>
      <div className="remotea-control-row">
        <button className="btn btn-sm" disabled={!controls.message.enabled || !text.trim() || off} title="Queued for after the current turn (Enter)"
          onClick={() => void act("message", { text, mode: "queue" })}>Message</button>
        <button className="btn btn-sm" disabled={!controls.steer.enabled || !text.trim() || off} title="Interrupts the current turn (Ctrl+Enter)"
          onClick={() => void act("steer", { text, mode: "steer" })}>Steer</button>
        <span className="remotea-control-gap"/>
        <button className="btn btn-sm btn-ghost" disabled={!controls.replan.enabled || off} title={controls.replan.reason}
          onClick={() => void act("replan", { reason: text })}>{busy === "replan" ? "Re-planning…" : "Re-plan"}</button>
        {run.plan?.verify && <button className="btn btn-sm btn-ghost" disabled={off || !run.surf} title="Run the plan's verification now"
          onClick={() => void act("verify")}>{busy === "verify" ? "Starting…" : "Verify now"}</button>}
        {/* Stop sends no revision (runAction): it is the emergency path and must work from a stale view. */}
        <button className="btn btn-sm btn-danger" disabled={!controls.stop.enabled || off || controls.stop.label === "Stopping…"}
          onClick={() => void act("stop", { mode: controls.stop.mode })}><Square size={14}/>{busy === "stop" ? "Stopping…" : controls.stop.label}</button>
        <button className={"btn btn-sm" + (controls.approve.primary ? " btn-primary" : "")} disabled={!controls.approve.enabled || off}
          title="Your decision: mark the task complete as you and end the run" onClick={() => void act("approve")}><Check size={14}/>{controls.approve.label}</button>
      </div>
      {packet && run.state === "awaiting_approval" && <p className="review-beside-approve" role="note">
        <strong>{packet.text}.</strong> {packet.decision?.text}</p>}
      {controls.offline && <p className="footnote">The remotea server is offline; commands wait until it reconnects.</p>}
    </footer>}
  </section>;
}

export function TargetPicker({ need, onSaved, onCancel, saveLabel = "Save and run" }: {
  need: NeedsTarget; onSaved: (id: string) => void; onCancel: () => void; saveLabel?: string;
}) {
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
      <button className="btn btn-sm btn-primary" disabled={!options.length || busy} onClick={() => void save()}>{saveLabel}</button>
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
