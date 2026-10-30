"""One real protocol path: the remotea server process connecting outbound to the Eridani app (task c9aa8f54, E-S5).

test_agent_link.py drives Eridani with a fake host, and remotea's tests drive remotea with a fake Eridani; each
side can drift from the other and both suites stay green. This test runs the real two:

    remotea-server (node, dist/) --ws--> LinkProxy (records) --ws--> uvicorn jarvis.api:app --> test database

and walks one run through connect -> hello -> spawn -> prompt -> turn -> verify -> judge -> stop, with a
reconnect in the middle that makes both sides deal with duplicates and an older peer. The frames each side sent
are compared with the recorded shapes in tests/fixtures/protocol/<caps_version>/ (keys and types, not values),
so a field renamed on either side fails here. How to run it: docs/AGENT_CONTROL.md, "Testing the protocol end to
end". It skips when node, the remotea checkout or its dist/ is missing (CI has no remotea checkout).
"""

import json
import os
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from jarvis import agent_judge as judge
from jarvis.config import get_settings
from jarvis.db import session_scope
from jarvis.domain import execute
from jarvis.models import Job, Note, Task
from jarvis.session_models import AgentAttempt, AgentCommand, AgentHost, AgentRun, AgentTurn
from protocol_harness import (
    FIXTURES,
    LINK_PATH,
    OLD_ERIDANI_CAPS_DROPPED,
    OLD_HOST_CAPS_DROPPED,
    EridaniApp,
    LinkProxy,
    RemoteaHost,
    check_fixtures,
    drift,
    missing,
    observed_shapes,
    remotea_checkout,
    tail,
    until,
)
from sqlalchemy import func, select

OWNER = "davin"
REASON = missing()
pytestmark = [pytest.mark.e2e_protocol, pytest.mark.skipif(REASON is not None, reason=REASON or "")]

STEP = 30  # Seconds any one step may take; the whole test stays well under two minutes.
GAP = "Also add a CHANGELOG entry for the fix."


# ---------------------------------------------------------------- the worker's side (this process)


def verdict(kind, gap=""):
    return judge.Verdict(verdict=kind, gap=gap, summary="stub judge",
                         criteria=[judge.CriterionVerdict(id="c1", status="met" if kind == "met" else "not_met",
                                                          evidence="the verify attempt")], steps=[])


def judge_once(run_id, answer):
    """Run the queued judge job as the worker would, with a stubbed model instead of Luna."""
    def queued():
        with session_scope() as db:
            attempts = select(AgentAttempt.id).where(AgentAttempt.run_id == run_id, AgentAttempt.kind == "judge",
                                                     AgentAttempt.status == "queued")
            jobs = db.scalars(select(Job).where(Job.kind == "agent_judge", Job.status == "queued")).all()
            ids = set(db.scalars(attempts))
            return next((j.id for j in jobs if (j.payload or {}).get("attempt") in ids), None)
    job_id = until(queued, STEP, "a queued judge job")
    calls = []

    def model(evidence):
        calls.append(evidence)
        return answer
    judge.process(job_id, model=model)
    assert len(calls) == 1, "the judge asked the model once"
    return calls[0]


def run_state(run_id):
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        return SimpleNamespace(state=row.state, reason=row.state_reason, revision=row.state_revision,
                               turn_count=row.turn_count, generation=row.generation)


def command_status(op_id):
    with session_scope() as db:
        return db.scalar(select(AgentCommand.status).where(AgentCommand.op_id == op_id))


def turns(run_id):
    with session_scope() as db:
        return db.scalar(select(func.count()).select_from(AgentTurn).where(AgentTurn.run_id == run_id))


def planned_task():
    """A task with a PRD, one verify criterion and the target's verify command: the stage D loop, no interview."""
    with session_scope() as db:
        note = Note(owner_id=OWNER, title="PRD: fix the flaky login test", content="Fix the flaky login test.")
        db.add(note)
        db.flush()
        note_id = note.id
    with session_scope() as db:
        created = execute(db, OWNER, str(uuid.uuid4()), "task.create", {"title": "Fix the flaky login test"})["data"]
    with session_scope() as db:
        execute(db, OWNER, str(uuid.uuid4()), "task.update", {
            "task_id": created["id"], "expected_revision": created["revision"], "prd_note_id": note_id,
            "success_criteria": [{"id": "c1", "text": "the test suite passes", "kind": "verify",
                                  "verify_id": RemoteaHost.VERIFY_ID}],
            "verify": {"target_id": RemoteaHost.TARGET, "verify_id": RemoteaHost.VERIFY_ID}})
    return created["id"]


# ---------------------------------------------------------------- the test


@pytest.fixture
def stack(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_COST_TRACKING_ENABLED", "false")
    get_settings.cache_clear()
    parts = SimpleNamespace(eridani=EridaniApp(tmp_path), proxy=None, host=None, tmp=tmp_path)
    try:
        yield parts
    finally:
        failed = False
        for name, stop in (("host", parts.host), ("proxy", parts.proxy), ("eridani", parts.eridani)):
            if stop is not None:
                try:
                    stop.stop()
                except Exception as exc:  # noqa: BLE001 - keep stopping the others
                    failed = True
                    print(f"stopping {name}: {exc}")
        get_settings.cache_clear()
        assert not failed


def test_real_remotea_drives_a_run_through_the_real_eridani(stack):
    started = time.monotonic()
    eri = stack.eridani.start()
    token = eri.post("/api/v1/remotea/enable")["token"]
    stack.proxy = proxy = LinkProxy(f"ws://127.0.0.1:{eri.port}{LINK_PATH}").start()
    stack.host = host = RemoteaHost(remotea_checkout(), stack.tmp, proxy.url, token)
    host.start()
    update = os.environ.get("UPDATE_FIXTURES") == "1"
    try:
        drive(eri, proxy, host)
    except BaseException:
        # A drifted field is often why a step failed: show the drift with the logs (nothing is recorded here).
        hellos = proxy.frames("host", "hello")
        if hellos and not update:
            folder = FIXTURES / str(hellos[0]["frame"].get("caps_version", 1))
            shapes = observed_shapes(proxy.records)
            for name in sorted(shapes):
                if (folder / f"{name}.json").exists():
                    recorded = json.loads((folder / f"{name}.json").read_text())["shape"]
                    for problem in drift(recorded, shapes[name]):
                        print(f"drift: {name}: {problem}")
        print(f"--- remotea-server log\n{tail(host.log)}\n--- eridani log\n{tail(eri.log)}")
        raise
    finally:
        if os.environ.get("PROTOCOL_E2E_FRAMES"):  # Every frame both ways, for protocol review (test tokens only).
            Path(os.environ["PROTOCOL_E2E_FRAMES"]).write_text(json.dumps(proxy.records, indent=1))
    elapsed = time.monotonic() - started
    assert elapsed < 120, f"the protocol test took {elapsed:.0f}s"

    # Wire fixtures: what each side actually sent, as keys and types.
    hello = proxy.frames("host", "hello")[0]["frame"]
    print(f"remotea build {hello.get('build_sha', 'unknown')} caps_version {hello.get('caps_version', 1)}; "
          f"{len(proxy.records)} frames in {elapsed:.1f}s")
    problems, written = check_fixtures(proxy.records, hello.get("caps_version", 1), update=update)
    if written:
        print(f"recorded {len(written)} wire fixture(s): {', '.join(written)}")
    assert not problems, ("The link protocol drifted from tests/fixtures/protocol (re-record with UPDATE_FIXTURES=1 "
                          "only if the change is intended and both sides handle it):\n  " + "\n  ".join(problems))


def drive(eri, proxy, host):
    # 1. connect -> hello -> welcome -> sessions.snapshot -> runs.snapshot
    hello = proxy.wait("host", "hello", timeout=45, what="remotea's hello")["frame"]
    welcome = proxy.wait("eridani", "welcome", timeout=STEP)["frame"]
    proxy.wait("host", "sessions.snapshot", timeout=STEP)
    proxy.wait("eridani", "runs.snapshot", timeout=STEP)
    assert hello["proto"] == welcome["proto"] == 1
    assert {"spawn", "input", "stop", "verify", "turn", "prompt", "runs.snapshot"} <= set(hello["caps"]), hello["caps"]
    assert hello["harnesses"]["claude"]["spawn"] is True, hello["harnesses"]["claude"]
    target = next(t for t in hello["targets"] if t["id"] == host.TARGET)
    assert target["harnesses"] == ["claude"] and target["verify"] == [host.VERIFY_ID], target
    assert {"prompt", "spawn", "verify", "context.scope"} <= set(welcome["caps"])
    hosts = until(lambda: [h for h in eri.get("/api/v1/remotea/status")["hosts"] if h["connected"]], STEP,
                  "the host to show as connected")
    host_id = hosts[0]["id"]

    # 2. Start a run over the browser routes: the target the host advertised, a plan with a verify command.
    task_id = planned_task()
    eri.post("/api/v1/remotea/targets", {"host_id": host_id, "host_target": host.TARGET, "repo": str(host.repo),
                                         "worktree_root": str(host.worktrees), "task_id": task_id})
    # The host will not hear Eridani's acks for its first prompt and turn: it resends both after the reconnect.
    proxy.drop_acks_for = {"turn", "prompt"}
    run_id = eri.post("/api/v1/remotea/runs", {"task_id": task_id, "harness": "claude", "model": "sonnet",
                                               "interview": False})["id"]

    # 3. spawn: accepted (journaled) in the ack, then executed in a worktree for the run's branch.
    spawn = proxy.wait("eridani", "spawn", lambda f: f["run_id"] == run_id, timeout=STEP)["frame"]
    assert spawn["harness"] == "claude" and spawn["target"]["id"] == host.TARGET and spawn["digest"]
    ack = proxy.wait("host", "ack", lambda f: f.get("op_id") == spawn["op_id"], timeout=STEP)["frame"]
    assert ack["ok"] and ack["outcome"]["state"] == "accepted" and ack["outcome"]["surf"].startswith("surf-eri-"), ack
    spawned = proxy.wait("host", "command.outcome", lambda f: f["command_id"] == spawn["op_id"], timeout=STEP)["frame"]
    assert spawned["state"] == "executed", spawned
    assert spawned["result"]["branch"] == spawn["target"]["branch"]

    # 4. SessionStart asks for context; the prompt is reported as the turn starts, then the turn itself.
    context = proxy.wait("host", "context.request", lambda f: f.get("run_id") == run_id, timeout=STEP)["frame"]
    reply = proxy.wait("eridani", "context.response", lambda f: f["req_id"] == context["req_id"], timeout=STEP)["frame"]
    assert reply["scope"] == "run" and reply["text"]
    prompt = proxy.wait("host", "prompt", lambda f: f.get("run_id") == run_id, timeout=STEP)["frame"]
    turn = proxy.wait("host", "turn", lambda f: f.get("run_id") == run_id, timeout=STEP)["frame"]
    assert prompt["seq"] == turn["seq"] and turn["root"] is True and turn["text"].startswith("Stub turn 1")
    assert prompt["text"] == spawn["prompt"][:4000], (prompt["text"][:300], spawn["prompt"][:300])
    until(lambda: run_state(run_id).turn_count == 1, STEP, "Eridani to count the first turn")

    # 5. verify: Eridani asks for the plan's verify command; the host runs it in the worktree (one run).
    verify = proxy.wait("eridani", "verify", lambda f: f["run_id"] == run_id, timeout=STEP)["frame"]
    assert verify["verify_id"] == host.VERIFY_ID and verify["turn_seq"] == turn["seq"]
    result = proxy.wait("host", "verify.result", lambda f: f["attempt_id"] == verify["attempt_id"], timeout=STEP)["frame"]
    assert result["exit_code"] == 0 and "3 passed" in result["tail"] and result["stale_workspace"] is False, result
    until(lambda: run_state(run_id).state == "judging", STEP, "the run to be judged")
    proxy.wait("host", "command.outcome", lambda f: f["command_id"] == verify["op_id"], timeout=STEP)
    until(lambda: command_status(verify["op_id"]) == "executed", STEP, "the verify command to settle")

    # 6. Reconnect with duplicates, to an older peer on both sides. Eridani acked the first prompt and turn,
    # the host never heard it; after the reconnect it sends both again and Eridani must store neither twice.
    dropped = {r["acks"] for r in proxy.frames("eridani", "ack") if r.get("dropped")}
    assert dropped == {"turn", "prompt"}, dropped
    stored = turns(run_id)
    proxy.drop_acks_for = set()
    proxy.old_host = proxy.old_eridani = True
    mark = len(proxy.records)
    first = proxy.cut()
    welcome2 = proxy.wait("eridani", "welcome", lambda f: f["conn_epoch"] > welcome["conn_epoch"], timeout=STEP,
                          since=mark)["frame"]
    proxy.wait("eridani", "runs.snapshot", timeout=STEP, since=mark)
    later = [r for r in proxy.records[mark:] if r["conn"] > first]

    def resent(kind):
        def found():
            frames = [r["frame"] for r in proxy.frames("host", kind, since=mark)
                      if r["frame"].get("run_id") == run_id and r["frame"]["seq"] == turn["seq"]]
            return frames[0] if frames else None
        return until(found, STEP, f"the host to resend its unacked {kind}")
    turn_again, prompt_again = resent("turn"), resent("prompt")
    assert turn_again["id"] != turn["id"] and turn_again["provider_turn_id"] == turn["provider_turn_id"]
    for again in (turn_again, prompt_again):
        dup = proxy.wait("eridani", "ack", lambda f, a=again: f.get("acked_id") == a["id"], timeout=STEP, since=mark)
        assert dup["frame"]["ok"] and dup["frame"]["outcome"] == {"duplicate": True}, dup["frame"]
    assert turns(run_id) == stored

    # Eridani's resend of a command it already delivered (same op_id and digest, new envelope on the new epoch):
    # the host answers with the outcome it journaled and starts nothing.
    replay = {**spawn, "id": str(uuid.uuid4()), "conn_epoch": welcome2["conn_epoch"], "at": int(time.time() * 1000)}
    proxy.send_to_host(replay)
    again = proxy.wait("host", "ack", lambda f: f.get("acked_id") == replay["id"], timeout=STEP, since=mark)["frame"]
    assert again["ok"] and again["op_id"] == spawn["op_id"] and again["outcome"]["state"] == "executed", again
    assert len(host.stub_entries("start")) == 1, host.stub_entries("start")

    # The older-peer subset reached both sides: Eridani stored a host without the newer caps.
    with session_scope() as db:
        caps = set(db.scalar(select(AgentHost.caps).where(AgentHost.id == host_id))["caps"])
    assert not caps & OLD_HOST_CAPS_DROPPED and {"spawn", "verify", "turn"} <= caps
    assert later, "the reconnect carried frames"

    # 7. The judge (stubbed model) says not met: the gap is typed into the agent as its next prompt.
    evidence = judge_once(run_id, verdict("not_met", GAP))
    assert evidence["verify"]["fresh"] is True and evidence["verify"]["status"] == "passed", evidence["verify"]
    assert (run_state(run_id).state, run_state(run_id).reason) == ("executing", "not_met")
    gap = proxy.wait("eridani", "input", lambda f: f["run_id"] == run_id, timeout=STEP, since=mark)["frame"]
    assert GAP in gap["text"] and gap["mode"] == "queue"
    typed = proxy.wait("host", "command.outcome", lambda f: f["command_id"] == gap["op_id"], timeout=STEP)["frame"]
    assert typed["state"] == "executed", typed

    # An Eridani without `prompt` gets no prompt frame: the turn carries the typed text instead.
    turn2 = proxy.wait("host", "turn", lambda f: f.get("run_id") == run_id and f["seq"] == turn["seq"] + 1,
                       timeout=STEP)["frame"]
    assert GAP in (turn2.get("prompt") or ""), turn2
    assert not [r for r in proxy.frames("host", "prompt", since=mark) if r["frame"]["seq"] == turn2["seq"]]
    verify2 = proxy.wait("eridani", "verify", lambda f: f["run_id"] == run_id and f["turn_seq"] == turn2["seq"],
                         timeout=STEP)["frame"]
    proxy.wait("host", "verify.result", lambda f: f["attempt_id"] == verify2["attempt_id"], timeout=STEP)
    until(lambda: run_state(run_id).state == "judging", STEP, "the second turn to be judged")
    detail = eri.get(f"/api/v1/remotea/runs/{run_id}")
    assistant = [t for t in detail["turns"] if t["role"] == "assistant" and t["seq"] == turn2["seq"]]
    assert assistant and GAP in (assistant[0]["prompt"] or ""), detail["turns"]

    # 8. Met: awaiting approval; the approval stops the agent and completes the run and the task.
    judge_once(run_id, verdict("met"))
    current = run_state(run_id)
    assert (current.state, current.reason) == ("awaiting_approval", "judge_met")
    eri.post(f"/api/v1/remotea/runs/{run_id}/approve", {"expected_revision": current.revision})
    stop = proxy.wait("eridani", "stop", lambda f: f["run_id"] == run_id, timeout=STEP)["frame"]
    stopped = proxy.wait("host", "command.outcome", lambda f: f["command_id"] == stop["op_id"], timeout=STEP)["frame"]
    assert stopped["state"] == "executed", stopped
    proxy.wait("host", "exit", lambda f: f["surf"] == spawned["result"].get("surf", ack["outcome"]["surf"]),
               timeout=STEP)
    until(lambda: run_state(run_id).state == "completed", STEP, "the run to complete")
    with session_scope() as db:
        assert db.get(Task, task_id).status == "completed"

    # The agent saw exactly what Eridani sent, once each; the verify command ran once per turn.
    prompts = [e["prompt"] for e in host.stub_entries("prompt")]
    assert len(prompts) == 2 and prompts[0] == spawn["prompt"] and GAP in prompts[1], prompts
    assert host.stub_entries("exit")[-1]["why"] in ("Ctrl-C", "SIGINT")
    assert host.verify_runs() == 2, host.verify_runs()
    assert OLD_ERIDANI_CAPS_DROPPED <= set(welcome2["caps"])  # Recorded before the rewrite: Eridani sent them.

    # Neither side refused anything the other sent, and Eridani never answered an ack with an ack.
    refused = [(r["dir"], r.get("acks"), r["frame"].get("error")) for r in proxy.records
               if r["frame"].get("t") == "ack" and r["frame"].get("ok") is False]
    assert not refused, refused
