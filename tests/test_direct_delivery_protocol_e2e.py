"""Direct input and public output through the real API, link, host and JSONL transports.

Only the model providers are synthetic. Their first turn stays open until the test
releases it, making a next-turn queue distinguishable from actual harness delivery.
"""

import json
import os
from uuid import uuid4

import pytest
from jarvis.db import session_scope
from jarvis.domain import execute
from protocol_harness import LinkProxy, RemoteaHost, missing, remotea_checkout, until
from test_protocol_e2e import stack as stack  # noqa: PLC0414 -- export the shared pytest fixture

REASON = missing()
pytestmark = [pytest.mark.e2e_protocol, pytest.mark.skipif(REASON is not None, reason=REASON or "")]

PROVIDER = r"""#!__NODE__
const fs = require('fs');
const readline = require('readline');
const harness = __HARNESS__;
const logPath = __LOG__;
const control = __CONTROL__;
if (process.argv.includes('--version')) {
  console.log(harness === 'codex' ? 'codex-cli 0.166.0' : '2.1.317 (Claude Code)');
  process.exit(0);
}
const send = x => process.stdout.write(JSON.stringify(x)+'\n');
const log = x => fs.appendFileSync(logPath, JSON.stringify(x)+'\n');
const sessionFlag = process.argv.includes('--session-id') ? '--session-id' : '--resume';
const session = harness === 'claude' ? process.argv[process.argv.indexOf(sessionFlag)+1] : 'direct-fixture-conversation';
let active = false, turns = 0, message = 0;
let turn = '', pendingEchoes = [];
function output(text) {
  const id = 'message-' + (++message);
  if (harness === 'codex') {
    send({method:'item/reasoning/textDelta',params:{threadId:session,turnId:turn,delta:'PRIVATE_REASONING_FIXTURE'}});
    send({method:'item/started',params:{threadId:session,turnId:turn,item:{id,type:'agentMessage',text:'',phase:'commentary'}}});
    send({method:'item/agentMessage/delta',params:{threadId:session,turnId:turn,itemId:id,delta:text}});
    send({method:'item/completed',params:{threadId:session,turnId:turn,item:{id,type:'agentMessage',text,phase:'commentary'}}});
  } else {
    send({type:'stream_event',session_id:session,uuid:id+'-start',event:{type:'message_start',message:{id,role:'assistant',content:[]}}});
    send({type:'stream_event',session_id:session,uuid:id+'-private',event:{type:'content_block_delta',index:0,delta:{type:'thinking_delta',thinking:'PRIVATE_REASONING_FIXTURE'}}});
    send({type:'stream_event',session_id:session,uuid:id+'-block',event:{type:'content_block_start',index:1,content_block:{type:'text',text:''}}});
    send({type:'stream_event',session_id:session,uuid:id+'-text',event:{type:'content_block_delta',index:1,delta:{type:'text_delta',text}}});
    send({type:'assistant',session_id:session,uuid:id,message:{id,role:'assistant',content:[{type:'text',text}]}});
  }
}
function start() {
  active = true; turn = 'turn-' + (++turns);
  if (harness === 'codex') send({method:'turn/started',params:{threadId:session,turn:{id:turn}}});
  output('Published progress while the fixture is still working.');
}
setInterval(() => {
  if (fs.existsSync(control+'-echo')) {
    for (const m of pendingEchoes.splice(0)) send(m);
  }
  if (active && fs.existsSync(control+'-finish')) {
    fs.unlinkSync(control+'-finish'); active = false;
    output('Fixture turn finished.');
    if (harness === 'codex') send({method:'turn/completed',params:{threadId:session,turn:{id:turn,status:'completed'}}});
    else send({type:'result',subtype:'success',session_id:session,uuid:'result-'+turn,result:'Fixture turn finished.',num_turns:turns});
    log({finished:turn});
  }
  if (!active && fs.existsSync(control+'-continue')) {
    fs.unlinkSync(control+'-continue'); start(); log({continued:turn});
  }
}, 25);
readline.createInterface({input:process.stdin}).on('line', line => {
  const m = JSON.parse(line);
  if (m.method === 'initialize') send({id:m.id,result:{}});
  if (m.method === 'model/list') send({id:m.id,result:{data:[{model:'gpt-6.1-sol',displayName:'GPT-6.1 Sol',isDefault:true,supportedReasoningEfforts:[{reasoningEffort:'low'}],defaultReasoningEffort:'low'}],nextCursor:null}});
  if (m.method === 'thread/start' || m.method === 'thread/resume') send({id:m.id,result:{thread:{id:session}}});
  if (m.method === 'turn/start') {
    log({method:m.method,operation:m.params.clientUserMessageId,text:m.params.input[0].text,active});
    send({id:m.id,result:{turn:{id:'turn-'+(turns+1)}}}); start();
  }
  if (m.method === 'turn/steer') {
    log({method:m.method,operation:m.params.clientUserMessageId,text:m.params.input[0].text,active,expected:m.params.expectedTurnId});
    if (!active || m.params.expectedTurnId !== turn) send({id:m.id,error:{code:-32600,message:'no active turn'}});
    else { send({id:m.id,result:{turnId:turn}}); output('Published response to the direct message.'); }
  }
  if (m.method === 'turn/interrupt') { log({interrupt:true}); send({id:m.id,result:{}}); }
  if (m.type === 'control_request' && m.request.subtype === 'initialize') {
    send({type:'control_response',response:{subtype:'success',request_id:m.request_id,response:{models:[{value:'sonnet',displayName:'Sonnet',supportedEffortLevels:['low'],supportsAutoMode:true}]}}});
    send({type:'system',subtype:'init',session_id:session});
  }
  if (m.type === 'control_request' && m.request.subtype === 'interrupt') log({interrupt:true});
  if (m.type === 'user') {
    log({method:'stream/user',operation:m.uuid,text:m.message.content,active});
    if (!active) { send(m); start(); }
    else { pendingEchoes.push(m); output('Published response to the direct message.'); }
  }
});
"""


class OfflineProxy(LinkProxy):
    """A controllable link outage, without stopping the worker process."""

    offline = False

    async def _relay(self, down):
        if self.offline:
            await down.close(1013, "synthetic network outage")
            return
        try:
            await super()._relay(down)
        except ConnectionRefusedError:
            await down.close(1013, "synthetic API restarting")


def setup_direct(stack, harness):
    eri = stack.eridani.start()
    enabled = eri.post("/api/v1/remotea/enable")
    stack.proxy = proxy = OfflineProxy(f"ws://127.0.0.1:{eri.port}/api/v1/external/agents/link").start()
    stack.host = host = RemoteaHost(remotea_checkout(), stack.tmp, proxy.url, enabled["token"])
    host.configure_mcp(enabled["setup_code"], eri.base)
    provider = host.bin / harness
    provider_log = host.bin / "direct-provider.jsonl"
    control = host.bin / "control"
    source = (PROVIDER.replace("__NODE__", host.node).replace("__HARNESS__", json.dumps(harness))
              .replace("__LOG__", json.dumps(str(provider_log))).replace("__CONTROL__", json.dumps(str(control))))
    provider.write_text(source)
    provider.chmod(0o755)
    config = json.loads((host.config / "eridani.json").read_text())
    config["targets"][0]["launch"] = {harness: {"bin": str(provider)}}
    host.write("eridani.json", config)
    if harness == "codex":
        (host.home / ".codex").mkdir()
        (host.home / ".codex" / "models_cache.json").write_text(json.dumps({"models": [{
            "slug": "gpt-6.1-sol", "display_name": "GPT-6.1 Sol", "default": True,
            "supported_reasoning_levels": [{"effort": "low"}], "default_reasoning_level": "low"}]}))
    original_env = host.env
    host.env = lambda: {**original_env(), "PATH": str(host.bin) + os.pathsep + "/usr/bin:/bin"}
    host.start()
    connected = until(lambda: next((h for h in eri.get("/api/v1/remotea/status")["hosts"] if h["connected"]), None),
                      45, "direct-capable host connected")
    assert "input.direct" in connected["caps"], connected
    with session_scope() as db:
        task = execute(db, "davin", str(uuid4()), "task.create", {"title": "Direct message fixture"})["data"]
    eri.post("/api/v1/remotea/targets", {"host_id": connected["id"], "host_target": host.TARGET,
             "repo": str(host.repo), "worktree_root": str(host.worktrees), "task_id": task["id"]})
    run = eri.post("/api/v1/remotea/runs", {"task_id": task["id"], "harness": harness,
                  "model": "gpt-6.1-sol" if harness == "codex" else "sonnet", "effort": "low", "freestyle": True})
    return eri, proxy, host, run, provider_log, control


def entries(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def delivered(eri, action_id):
    receipt = eri.get(f"/api/v1/remotea/actions/{action_id}")
    assert receipt["status"] not in ("failed", "uncertain"), json.dumps({
        "status": receipt["status"], "commands": receipt["commands"]}, indent=2)
    return receipt["status"] == "delivered"


@pytest.mark.parametrize("harness", ["codex", "claude"])
def test_direct_message_reaches_busy_harness_and_survives_offline_reconnect(stack, harness):
    eri, proxy, _host, run, provider_log, control = setup_direct(stack, harness)
    run_path = f"/api/v1/remotea/runs/{run['id']}"
    initial = proxy.wait("host", "output.live", lambda f: f.get("run_id") == run["id"]
                         and f.get("text") == "Published progress while the fixture is still working.",
                         timeout=30)["frame"]
    assert initial["text"] == "Published progress while the fixture is still working."
    until(lambda: "Published progress while the fixture is still working." in json.dumps(eri.get(run_path)),
          15, "published output saved before the turn finishes")
    assert not [t for t in eri.get(run_path)["turns"] if t["role"] == "assistant"]

    action_id = str(uuid4())
    body = {"text": "Approved", "mode": "direct", "action_id": action_id}
    eri.post(run_path + "/message", body)
    held = eri.get(run_path)
    assert held["automation"]["held"] and not held["paused"]
    generation = held["generation"]
    direct = until(lambda: next((e for e in entries(provider_log) if e.get("text") == "Approved"), None),
                   15, "literal direct input reached the still-active harness")
    assert direct["active"] is True
    assert direct["method"] == ("turn/steer" if harness == "codex" else "stream/user")
    if harness == "codex":
        assert direct["expected"] == "turn-1"
    assert not [e for e in entries(provider_log) if e.get("interrupt")]
    assert not [e for e in entries(provider_log) if e.get("finished")]
    receipt_path = f"/api/v1/remotea/actions/{action_id}"
    until(lambda: eri.get(receipt_path)["status"] == "delivered", 15, "harness handoff receipt")
    # Claude's user-message UUID has deliberately not been echoed yet. A write
    # receipt must not wait for that echo or claim provider acknowledgment.
    if harness == "claude":
        handed = proxy.wait("host", "input.receipt", lambda f: f.get("input_id") == direct["operation"]
                            and f.get("stage") == "handed_to_harness", timeout=10)["frame"]
        assert handed["session_id"] == initial["session_id"]
        assert not [r for r in proxy.frames("host", "input.receipt")
                    if r["frame"].get("input_id") == direct["operation"]
                    and r["frame"].get("stage") == "provider_acknowledged"]
    eri.post(run_path + "/message", body)
    assert len([e for e in entries(provider_log) if e.get("text") == "Approved"]) == 1
    assert eri.get(run_path)["generation"] == generation, "direct send must not interrupt/change the generation"

    # Restart only the synthetic API. The same provider remains in turn 1;
    # durable hold and live output must be readable after reload/reconnection.
    previous_connection = proxy.connections
    eri.stop()
    eri.start()
    until(lambda: proxy.connections > previous_connection and proxy.current, 30, "link after API restart")
    restored = eri.get(run_path)
    assert restored["automation"]["held"]
    assert "Published response to the direct message." in json.dumps(restored)
    assert not [t for t in restored["turns"] if t["role"] == "assistant"]

    # Make the host unreachable without stopping its provider. A second Send
    # is saved and keeps automation held; no invisible next-turn fallback.
    proxy.offline = True
    proxy.cut()
    until(lambda: not any(h["connected"] for h in eri.get("/api/v1/remotea/status")["hosts"]),
          15, "synthetic network outage visible")
    offline_id = str(uuid4())
    eri.post(run_path + "/message", {"text": "Offline follow-up", "mode": "direct", "action_id": offline_id})
    second_offline_id = str(uuid4())
    eri.post(run_path + "/message", {"text": "Second offline follow-up", "mode": "direct", "action_id": second_offline_id})
    saved = eri.get(run_path)
    assert saved["automation"]["held"] and not saved["paused"]
    assert eri.get(f"/api/v1/remotea/actions/{offline_id}")["status"] == "queued"
    assert not [e for e in entries(provider_log) if e.get("text") == "Offline follow-up"]
    refused = eri.http.post(run_path + "/automation/resume", json={"action_id": str(uuid4()),
                            "expected_automation_revision": saved["automation"]["revision"]})
    assert refused.status_code == 409, refused.text
    assert eri.get(run_path)["automation"]["held"]
    proxy.offline = False
    offline_direct = until(lambda: next((e for e in entries(provider_log) if e.get("text") == "Offline follow-up"), None),
                           45, "offline direct message handed to the same active worker")
    assert offline_direct["active"] is True
    until(lambda: delivered(eri, offline_id),
          15, "offline message delivery receipt")
    until(lambda: delivered(eri, second_offline_id),
          15, "both offline messages delivered despite a newer hold revision")
    second_offline_direct = next(e for e in entries(provider_log) if e.get("text") == "Second offline follow-up")
    assert second_offline_direct["active"] is True
    assert [e["text"] for e in entries(provider_log) if e.get("text") in
            ("Offline follow-up", "Second offline follow-up")] == ["Offline follow-up", "Second offline follow-up"]
    assert len([e for e in entries(provider_log) if e.get("method") == "turn/start"]) == (1 if harness == "codex" else 0)
    assert not [e for e in entries(provider_log) if e.get("interrupt")]

    # Provider acknowledgment may arrive much later than handoff. It is monotonic
    # and cannot create a second user submission or a completed assistant turn.
    control.with_name(control.name + "-echo").touch()
    if harness == "claude":
        proxy.wait("host", "input.receipt", lambda f: f.get("input_id") == offline_direct["operation"]
                   and f.get("stage") == "provider_acknowledged", timeout=15)
    assert not [t for t in eri.get(run_path)["turns"] if t["role"] == "assistant"]
    current = eri.get(run_path)
    eri.post(run_path + "/automation/resume", {"action_id": str(uuid4()),
             "expected_automation_revision": current["automation"]["revision"]})
    until(lambda: not eri.get(run_path)["automation"]["held"], 20, "host-confirmed automation release")
    assert not [e for e in entries(provider_log) if e.get("interrupt")]

    control.with_name(control.name + "-finish").touch()
    until(lambda: len([t for t in eri.get(run_path)["turns"] if t["role"] == "assistant"]) == 1,
          20, "one completed turn after explicit provider finish")
    public = eri.get(run_path)
    assert "PRIVATE_REASONING_FIXTURE" not in json.dumps(public)
    assert "PRIVATE_REASONING_FIXTURE" not in json.dumps(proxy.frames("host", "output.live"))
    assert len([e for e in entries(provider_log) if e.get("text") == "Approved"]) == 1
    assert len([e for e in entries(provider_log) if e.get("text") == "Offline follow-up"]) == 1
    assert len([e for e in entries(provider_log) if e.get("text") == "Second offline follow-up"]) == 1
    if harness == "claude":
        # A streaming provider may acknowledge input before choosing a later turn
        # to work on it. New public output after the first result must create a
        # distinct turn even when no second UUID echo arrives at that boundary.
        control.with_name(control.name + "-continue").touch()
        until(lambda: any(e.get("continued") == "turn-2" for e in entries(provider_log)),
              10, "provider began the later turn after its earlier input acknowledgment")
        control.with_name(control.name + "-finish").touch()
        until(lambda: len([t for t in eri.get(run_path)["turns"] if t["role"] == "assistant"]) == 2,
              20, "distinct completed Claude turns despite early input acknowledgments")
        finished = [t for t in eri.get(run_path)["turns"] if t["role"] == "assistant"]
        assert len({turn["provider_turn_id"] for turn in finished}) == 2
