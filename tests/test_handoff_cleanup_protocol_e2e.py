"""Real API/link/host handoff retires only an attached worker's separate planner."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from jarvis import agent_plan
from jarvis import run_coordinator as coordinator
from jarvis.db import session_scope
from jarvis.domain import execute
from jarvis.models import Task
from jarvis.session_models import AgentRun, AgentSession, AgentTurn
from protocol_harness import MANAGED_FORCE, LinkProxy, RemoteaHost, missing, remotea_checkout, until
from sqlalchemy import select
from test_protocol_e2e import stack  # noqa: F401 - fixture imported for pytest discovery

REASON = missing()
pytestmark = [pytest.mark.e2e_protocol, pytest.mark.skipif(REASON is not None, reason=REASON or "")]
OWNER = "davin"
PLANNER_MODEL, WORKER_MODEL = "fixture-planner", "fixture-worker"
PLANNER_SESSION, WORKER_SESSION = "handoff-planner-conversation", "handoff-worker-conversation"

# This JSONL provider only handles deterministic protocol messages. Its files,
# conversations, executable and HOME belong to the disposable RemoteaHost.
PROVIDER = r"""#!__NODE__
const fs = require('fs');
const readline = require('readline');
if (process.argv.includes('--version')) { console.log('codex-cli 0.166.0'); process.exit(0); }
const send = x => process.stdout.write(JSON.stringify(x)+'\n');
const log = x => fs.appendFileSync(__LOG__, JSON.stringify(x)+'\n');
const history = __HISTORY__;
let session, role, turns = 0;
const when = (file, action) => {
  const timer = setInterval(() => { if (fs.existsSync(file)) { clearInterval(timer); action(); } }, 25);
};
readline.createInterface({input:process.stdin}).on('line', line => {
  const m = JSON.parse(line);
  if (m.method === 'initialize') send({id:m.id,result:{}});
  if (m.method === 'model/list') send({id:m.id,result:{data:['fixture-planner','fixture-worker'].map(model => ({model,displayName:model,isDefault:model==='fixture-planner',supportedReasoningEfforts:[{reasoningEffort:'low'}],defaultReasoningEffort:'low'})),nextCursor:null}});
  if (m.method === 'thread/start' || m.method === 'thread/resume') {
    role = m.params.model === 'fixture-worker' ? 'worker' : 'planner';
    session = `handoff-${role}-conversation`;
    const start = () => {
      fs.writeFileSync(`${history}/${session}.json`, JSON.stringify({session,role,inputs:[]}));
      log({event:'session_started',role,session});
      send({id:m.id,result:{thread:{id:session}}});
    };
    if (role === 'worker') {
      log({event:'worker_waiting_for_attach',session});
      when(__WORKER_GATE__, start);
    } else start();
  }
  if (m.method === 'turn/start') {
    const turn = `${role}-turn-${++turns}`;
    const saved = JSON.parse(fs.readFileSync(`${history}/${session}.json`));
    saved.inputs.push(m.params.input); fs.writeFileSync(`${history}/${session}.json`,JSON.stringify(saved));
    log({event:'turn_started',role,session,turn,input:m.params.input});
    send({id:m.id,result:{turn:{id:turn}}});
    send({method:'turn/started',params:{threadId:session,turn:{id:turn}}});
    if (role === 'planner') when(__PLAN_GATE__, () => {
      send({method:'item/completed',params:{item:{type:'agentMessage',text:'The fixture plan is ready for approval.'}}});
      send({method:'turn/completed',params:{threadId:session,turn:{id:turn,status:'completed'}}});
      log({event:'planner_turn_completed',session});
    });
    // The worker deliberately keeps working, so planner retirement cannot be
    // confused with terminal-run cleanup or a worker's own process exit.
  }
  if (m.method === 'turn/interrupt') send({id:m.id,result:{}});
});
"""


def test_separate_planner_closes_only_after_real_worker_attachment(stack):  # noqa: F811
    eri = stack.eridani.start()
    enabled = eri.post("/api/v1/remotea/enable")
    stack.proxy = proxy = LinkProxy(f"ws://127.0.0.1:{eri.port}/api/v1/external/agents/link").start()
    stack.host = host = RemoteaHost(remotea_checkout(), stack.tmp, proxy.url, enabled["token"])
    host.configure_mcp(enabled["setup_code"], eri.base)
    provider_log = host.bin / "handoff-provider.jsonl"
    history_dir = host.home / "provider-history"
    history_dir.mkdir()
    worker_gate, plan_gate = host.bin / "allow-worker-attach", host.bin / "complete-planner-turn"
    provider = host.bin / "codex"
    provider.write_text(PROVIDER.replace("__NODE__", host.node).replace("__LOG__", json.dumps(str(provider_log)))
                        .replace("__HISTORY__", json.dumps(str(history_dir)))
                        .replace("__WORKER_GATE__", json.dumps(str(worker_gate)))
                        .replace("__PLAN_GATE__", json.dumps(str(plan_gate))))
    provider.chmod(0o755)
    config = json.loads((host.config / "eridani.json").read_text())
    config["targets"][0]["launch"] = {"codex": {"bin": str(provider)}}
    host.write("eridani.json", config)
    codex_home = host.home / ".codex"
    codex_home.mkdir()
    (codex_home / "models_cache.json").write_text(json.dumps({"models": [
        {"slug": model, "display_name": model, "default": model == PLANNER_MODEL,
         "supported_reasoning_levels": [{"effort": "low"}], "default_reasoning_level": "low"}
        for model in (PLANNER_MODEL, WORKER_MODEL)]}))
    original_env = host.env
    # These fakes speak the managed transports but not remotea's conformance probe (remotea PR #51): admit their
    # versions without one, as an operator would with a launcher they trust.
    host.env = lambda: {**original_env(), "PATH": str(host.bin) + os.pathsep + "/usr/bin:/bin", **MANAGED_FORCE}
    host.start()
    connected = until(lambda: next((h for h in eri.get("/api/v1/remotea/status")["hosts"] if h["connected"]), None),
                      45, "handoff-capable isolated host connected")
    assert "run.cleanup.handoff" in connected["caps"], connected
    with session_scope() as db:
        task = execute(db, OWNER, str(uuid4()), "task.create", {"title": "Separate planner handoff fixture"})["data"]
    eri.post("/api/v1/remotea/targets", {"host_id": connected["id"], "host_target": host.TARGET,
             "repo": str(host.repo), "worktree_root": str(host.worktrees), "task_id": task["id"]})
    run = eri.post("/api/v1/remotea/runs", {"task_id": task["id"], "harness": "codex", "model": PLANNER_MODEL,
                   "interview": False, "approve_final": True,
                   "worker": {"harness": "codex", "model": WORKER_MODEL}})
    run_path = f"/api/v1/remotea/runs/{run['id']}"

    def registry():
        return json.loads((host.config / "registry.json").read_text())

    def events():
        return [json.loads(line) for line in provider_log.read_text().splitlines()] if provider_log.exists() else []

    def attached(session_id):
        with session_scope() as db:
            current = db.get(AgentRun, run["id"])
            session = db.get(AgentSession, current.session_pk) if current.session_pk else None
            return session if session and session.session_id == session_id else None

    planner_session = until(lambda: attached(PLANNER_SESSION), 30, "real planner SessionStart attached")
    planner_row = registry()["sessions"][planner_session.surf]
    planner = {"surf": planner_row["id"], "incarnation": planner_row["incarnation"],
               "generation": planner_row["eridani"]["generation"]}
    worktree = Path(planner_row["cwd"])
    keep = worktree / "keep-handoff-work.txt"
    keep.write_text("The handoff must preserve this worktree.\n")
    planner_history = history_dir / f"{PLANNER_SESSION}.json"
    assert planner_history.exists()
    assert not proxy.frames("eridani", "run.cleanup"), "planning must not retire any terminal"

    # Seed through the ordinary plan service; the approval and every handoff
    # effect below cross the real HTTP API, websocket link and Node host.
    with session_scope() as db:
        current = db.get(AgentRun, run["id"])
        agent_plan.submit_plan(db, SimpleNamespace(owner_id=OWNER, id=None, scopes=["agents:session"]),
            agent_plan.SubmitPlanInput(run_token=coordinator.run_token(current), submission_id=uuid4(),
                expected_plan_revision=db.get(Task, task["id"]).plan_revision or 0,
                prd="Keep the approved plan and both provider conversations through handoff.",
                steps=[{"title": "Implement the approved fixture plan"}],
                success_criteria=[{"text": "Worker remains live after the planner terminal is retired."}]))
    plan_gate.touch()
    until(lambda: any(e.get("event") == "planner_turn_completed" for e in events()), 15, "planner turn completed")
    decision = until(lambda: next((d for d in eri.get(run_path)["decisions"]
                                  if d["kind"] == "plan_approval" and d["status"] == "pending"), None),
                     20, "actual pending plan approval")
    eri.post(f"/api/v1/remotea/decisions/{decision['id']}/respond",
             {"action_id": str(uuid4()), "expected_revision": decision["revision"], "action": "approve"})
    try:
        until(lambda: any(e.get("event") == "worker_waiting_for_attach" for e in events()), 30,
              "worker provider initialization reached the controlled attachment gate")
    except AssertionError:
        current = eri.get(run_path)
        print("handoff diagnostics", {"state": current["state"], "reason": current.get("state_reason"),
              "handoff": current.get("handoff"), "commands": current.get("commands")})
        raise
    planner_stop = proxy.wait("eridani", "stop", lambda f: f.get("role") == "planner" and f["run_id"] == run["id"],
                              timeout=20)["frame"]
    proxy.wait("host", "command.outcome", lambda f: f["command_id"] == planner_stop["op_id"] and f["state"] == "executed",
               timeout=20)
    assert eri.get(run_path)["state"] == "handing_off"
    assert not attached(WORKER_SESSION)
    waiting = registry()
    assert planner["surf"] in waiting["sessions"], "a stopped planner stays inspectable while the worker is pending"
    assert "s:" + planner["surf"] not in waiting["tombstones"]
    assert not [r for r in proxy.frames("eridani", "run.cleanup") if r["frame"].get("scope") == "retired_planner"], \
        "no planner retirement may be queued before worker attachment"
    assert keep.read_text() == "The handoff must preserve this worktree.\n"

    worker_gate.touch()
    try:
        worker_session = until(lambda: attached(WORKER_SESSION), 30, "real separate worker SessionStart attached")
    except AssertionError:
        current = eri.get(run_path)
        print("worker attachment diagnostics", {"state": current["state"], "reason": current.get("state_reason"),
              "handoff": current.get("handoff"), "commands": current.get("commands"),
              "sessions": [(row["id"], row["state"], row.get("agent"))
                           for row in registry()["sessions"].values()]})
        raise
    until(lambda: eri.get(run_path)["state"] == "executing", 15, "worker owns the active run")
    cleanup = proxy.wait("eridani", "run.cleanup", lambda f: f["run_id"] == run["id"] and
                         f.get("scope") == "retired_planner", timeout=30)["frame"]
    assert cleanup["surfaces"] == [planner]
    assert cleanup["handoff_to"]["surf"] == worker_session.surf
    assert cleanup["handoff_to"]["surf"] != planner["surf"]
    assert "terminal_state" not in cleanup

    def planner_closed():
        saved = registry()
        return planner["surf"] not in saved["sessions"] and "s:" + planner["surf"] in saved["tombstones"]

    until(planner_closed, 30, "retired planner removed from desktop registry")
    worker_row = registry()["sessions"][worker_session.surf]
    assert cleanup["handoff_to"] == {"surf": worker_row["id"], "incarnation": worker_row["incarnation"],
                                     "generation": worker_row["eridani"]["generation"]}
    assert worker_row["state"] not in ("exited", "lost")
    assert worker_row["eridani"]["controller"] == "eridani"
    assert Path(worker_row["cwd"]) == worktree and keep.exists()
    assert until(lambda: any(e.get("event") == "turn_started" and e.get("role") == "worker" for e in events()),
                 15, "worker received the frozen plan")
    for session_id in (PLANNER_SESSION, WORKER_SESSION):
        assert json.loads((history_dir / f"{session_id}.json").read_text())["inputs"]
    saved_conversations = [json.loads(path.read_text()) for path in (host.config / "managed").glob("*.json")]
    assert {PLANNER_SESSION, WORKER_SESSION}.issubset({row["sessionId"] for row in saved_conversations})
    with session_scope() as db:
        sessions = db.scalars(select(AgentSession).where(AgentSession.run_id == run["id"])).all()
        assert {PLANNER_SESSION, WORKER_SESSION}.issubset({session.session_id for session in sessions})
        assert db.scalar(select(AgentTurn.id).where(AgentTurn.run_id == run["id"], AgentTurn.role == "assistant"))
    assert eri.get(run_path)["history"], "planner retirement retains the run's history"
    assert registry()["sessions"][worker_session.surf]["state"] not in ("exited", "lost")
