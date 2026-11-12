"""Real API/link/host recovery with a deterministic JSONL provider (no model calls)."""
import json
import os
from pathlib import Path
from uuid import uuid4

import pytest
from jarvis.db import session_scope
from jarvis.domain import execute
from jarvis.session_models import AgentSession, AgentTurn
from protocol_harness import MANAGED_FORCE, LinkProxy, RemoteaHost, missing, remotea_checkout, until
from sqlalchemy import select
from test_protocol_e2e import stack as stack  # noqa: F401

REASON = missing()
pytestmark = [pytest.mark.e2e_protocol, pytest.mark.skipif(REASON is not None, reason=REASON or "")]

PROVIDER = r"""#!__NODE__
const fs = require('fs');
const readline = require('readline');
if (process.argv.includes('--version')) { console.log('codex-cli 0.166.0'); process.exit(0); }
const send = x => process.stdout.write(JSON.stringify(x)+'\n');
const log = x => fs.appendFileSync(__LOG__, JSON.stringify(x)+'\n');
let turn = 'turn-' + process.pid;
const requestId = 'approval-' + process.pid;
readline.createInterface({input:process.stdin}).on('line', line => {
  const m = JSON.parse(line);
  if (m.method === 'initialize') send({id:m.id,result:{}});
  if (m.method === 'model/list') send({id:m.id,result:{data:[{model:'gpt-6.1-sol',displayName:'GPT-6.1 Sol',isDefault:true,supportedReasoningEfforts:[{reasoningEffort:'low'},{reasoningEffort:'high'}],defaultReasoningEffort:'low'}],nextCursor:null}});
  if (m.method === 'thread/start' || m.method === 'thread/resume') {
    log({start:m.method}); send({id:m.id,result:{thread:{id:'durable-conversation'}}});
  }
  if (m.method === 'turn/start') {
    log({input:m.params.input,operation:m.params.clientUserMessageId});
    send({id:m.id,result:{turn:{id:turn}}});
    send({method:'turn/started',params:{threadId:'durable-conversation',turn:{id:turn}}});
    send({id:requestId,method:'item/commandExecution/requestApproval',params:{threadId:'durable-conversation',turnId:turn,itemId:'item-'+process.pid,command:'printf durable-approval',cwd:process.cwd(),reason:'Review exact fixture action'}});
  }
  if (m.method === 'turn/interrupt') {
    send({id:m.id,result:{}});
    send({method:'turn/completed',params:{threadId:'durable-conversation',turn:{id:turn,status:'interrupted'}}});
  }
  if (m.id === requestId && m.result) {
    log({answer:m.result});
    send({method:'serverRequest/resolved',params:{threadId:'durable-conversation',requestId}});
    send({method:'item/completed',params:{item:{type:'agentMessage',text:'Fixture action completed.'}}});
    send({method:'turn/completed',params:{threadId:'durable-conversation',turn:{id:turn,status:'completed'}}});
  }
});
"""


def test_paused_question_survives_real_host_restart_and_receipt_replay(stack):
    eri = stack.eridani.start()
    enabled = eri.post("/api/v1/remotea/enable")
    token = enabled["token"]
    stack.proxy = proxy = LinkProxy(f"ws://127.0.0.1:{eri.port}/api/v1/external/agents/link").start()
    stack.host = host = RemoteaHost(remotea_checkout(), stack.tmp, proxy.url, token)
    host.configure_mcp(enabled["setup_code"], eri.base)
    provider = host.bin / "codex"
    provider_log = host.bin / "provider.jsonl"
    provider.write_text(PROVIDER.replace("__NODE__", host.node).replace("__LOG__", json.dumps(str(provider_log))))
    provider.chmod(0o755)
    config = json.loads((host.config / "eridani.json").read_text())
    config["targets"][0]["launch"] = {"codex": {"bin": str(provider)}}
    host.write("eridani.json", config)
    (host.home / ".codex").mkdir()
    (host.home / ".codex" / "models_cache.json").write_text(json.dumps({"models": [{
        "slug": "gpt-6.1-sol", "display_name": "GPT-6.1 Sol", "default": True,
        "supported_reasoning_levels": [{"effort": "low"}, {"effort": "high"}], "default_reasoning_level": "low"}]}))
    original_env = host.env
    # These fakes speak the managed transports but not remotea's conformance probe (remotea PR #51): admit their
    # versions without one, as an operator would with a launcher they trust.
    host.env = lambda: {**original_env(), "PATH": str(host.bin) + os.pathsep + "/usr/bin:/bin", **MANAGED_FORCE}
    host.start()
    connected = until(lambda: next((h for h in eri.get("/api/v1/remotea/status")["hosts"] if h["connected"]), None),
                      45, "managed host connected")
    assert "managed.sessions" in connected["caps"], connected
    with session_scope() as db:
        task = execute(db, "davin", str(uuid4()), "task.create", {"title": "Durable protocol fixture"})["data"]
    eri.post("/api/v1/remotea/targets", {"host_id": connected["id"], "host_target": host.TARGET,
             "repo": str(host.repo), "worktree_root": str(host.worktrees), "task_id": task["id"]})
    run = eri.post("/api/v1/remotea/runs", {"task_id": task["id"], "harness": "codex",
                                          "model": "gpt-6.1-sol", "effort": "high", "freestyle": True})
    run_path = f"/api/v1/remotea/runs/{run['id']}"

    def questions():
        return [d for d in eri.get(run_path)["decisions"] if d["kind"] == "question"]

    original = until(lambda: next(iter(questions()), None), 30, "durable permission card")
    assert original["payload"]["action"]["tool"] == "item/commandExecution/requestApproval"
    current = eri.get(run_path)
    eri.post(run_path + "/pause", {"action_id": str(uuid4()), "expected_revision": current["state_revision"]})
    until(lambda: eri.get(run_path)["paused"], 10, "run paused")
    proxy.wait("eridani", "run.pause", timeout=20)
    # Wait for persisted host acknowledgement before simulating process loss.
    pause = proxy.frames("eridani", "run.pause")[-1]["frame"]
    proxy.wait("host", "command.outcome", lambda f: f["command_id"] == pause["op_id"] and f["state"] == "executed", timeout=20)
    host.stop()
    host.start()
    until(lambda: any(h["connected"] for h in eri.get("/api/v1/remotea/status")["hosts"]), 45, "host reconnected")
    assert eri.get(run_path)["paused"]
    assert [q["id"] for q in questions()] == [original["id"]]
    action_id = str(uuid4())
    body = {"action_id": action_id, "expected_revision": original["revision"], "action": "answer",
            "answers": [{"item_id": "approval", "choice_ids": ["allow_once"]}]}
    answer_path = f"/api/v1/remotea/decisions/{original['id']}/respond"
    saved = eri.post(answer_path, body)
    assert eri.get(run_path)["paused"], "saving an answer must not resume the run"
    assert saved["receipt"]["action_id"] == action_id
    current = eri.get(run_path)
    eri.post(run_path + "/resume", {"action_id": str(uuid4()), "expected_revision": current["state_revision"]})
    try:
        until(lambda: eri.get(f"/api/v1/remotea/actions/{action_id}")["status"] == "delivered", 45,
              "saved answer delivered to recreated callback")
    except AssertionError:
        receipt = eri.get(f"/api/v1/remotea/actions/{action_id}")
        current = eri.get(run_path)
        print("delivery diagnostic", {"receipt": receipt, "state": current["state"],
              "reason": current.get("state_reason"), "paused": current["paused"],
              "incarnation": current.get("incarnation"), "plan_revision": current.get("plan_revision")})
        raise
    repeated = eri.post(answer_path, body)
    assert repeated["receipt"]["status"] == "delivered"
    events = [json.loads(line) for line in provider_log.read_text().splitlines()]
    assert len([e for e in events if "answer" in e]) == 1, events
    assert any(e.get("start") == "thread/resume" for e in events)
    registry = json.loads((host.config / "registry.json").read_text())
    surfaces = [row for row in registry["sessions"].values() if row.get("eridani", {}).get("runId") == run["id"]]
    assert surfaces
    worktrees = [Path(row["cwd"]) for row in surfaces]
    assert all(path.exists() for path in worktrees)
    eri.post(run_path + "/cancel", {"action_id": str(uuid4())})
    until(lambda: eri.get(run_path)["state"] == "stopped", 30, "run reached its final state")
    proxy.wait("eridani", "run.cleanup", lambda f: f["run_id"] == run["id"], timeout=30)

    def closed():
        saved = json.loads((host.config / "registry.json").read_text())
        return all(row["id"] not in saved["sessions"] and "s:" + row["id"] in saved["tombstones"]
                   for row in surfaces)

    until(closed, 30, "finalized terminals removed from registry")
    assert all(path.exists() for path in worktrees), "closing a terminal must preserve its worktree"
    saved_conversations = [json.loads(path.read_text()) for path in (host.config / "managed").glob("*.json")]
    assert any(row["sessionId"] == "durable-conversation" for row in saved_conversations)
    assert provider_log.exists(), "provider history remains available"
    with session_scope() as db:
        assert db.scalar(select(AgentSession.id).where(AgentSession.run_id == run["id"]))
        assert db.scalar(select(AgentTurn.id).where(AgentTurn.run_id == run["id"]))
    assert eri.get(run_path)["history"], "Eridani history remains readable after terminal cleanup"
