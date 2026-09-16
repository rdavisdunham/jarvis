import asyncio
import json
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet
from jarvis import agent_work, work_intake, work_runner
from jarvis.config import get_settings
from jarvis.db import session_scope
from jarvis.domain import DomainError, execute
from jarvis.models import AgentWork, AuthSession, Command, Job, Outbox, Task, VoiceInbox
from jarvis.work_crypto import unseal
from sqlalchemy import select


@pytest.fixture(autouse=True)
def recovery_settings(monkeypatch):
    monkeypatch.setenv("JARVIS_INTEGRATION_ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("JARVIS_OPENAI_API_KEY", "synthetic")
    monkeypatch.setenv("JARVIS_GEMINI_API_KEY", "synthetic")
    monkeypatch.setenv("JARVIS_EXTERNAL_SERVICES_ENABLED", "true")
    monkeypatch.setenv("JARVIS_COST_TRACKING_ENABLED", "false")
    get_settings.cache_clear()

    async def memories(*args):
        return ""

    monkeypatch.setattr(work_runner, "prompt_context", memories)
    yield
    get_settings.cache_clear()


def accept(client, message="Add Alpha", turn=None):
    conv = client.post("/api/v1/conversations", json={}).json()
    payload = {"turn_id": turn or str(uuid4()), "conversation_id": conv["id"], "message": message}
    response = client.post("/api/v1/work", json=payload)
    assert response.status_code == 200, response.text
    return response.json(), payload


def action(client, message="Add Alpha"):
    work, _payload = accept(client, message)
    with session_scope() as db:
        db.get(Job, work["id"]).kind = "agent_action"
    return work


def response(calls=None, message=None):
    return {
        "id": str(uuid4()),
        "usage": {"prompt_tokens": 10, "completion_tokens": 10},
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": message,
                    "tool_calls": [
                        {
                            "id": str(uuid4()),
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)},
                        }
                        for name, args in calls or []
                    ],
                }
            }
        ],
    }


def responder(monkeypatch, responses):
    async def model(*args, **kwargs):
        assert responses, "Unexpected provider call"
        item = responses.pop(0)
        return item() if callable(item) else item

    monkeypatch.setattr(work_runner, "request_model", model)
    return model


def test_acceptance_is_idempotent_and_input_encrypted(client):
    work, payload = accept(client)
    again = client.post("/api/v1/work", json=payload)
    assert again.status_code == 200 and again.json()["id"] == work["id"]
    with session_scope() as db:
        row = db.get(AgentWork, work["id"])
        assert "Alpha" not in row.input_ciphertext
        assert unseal(row.input_ciphertext)["message"] == "Add Alpha"
        assert db.get(Outbox, row.id)
        assert len(list(db.scalars(select(AgentWork)))) == 1
    payload["message"] = "Different"
    assert client.post("/api/v1/work", json=payload).status_code == 409


@pytest.mark.asyncio
async def test_crash_after_commit_replays_same_command_once(client, monkeypatch):
    work = action(client)
    responder(monkeypatch, [response([("task_create", {"title": "Alpha"})])])
    real_checkpoint = work_runner.checkpoint

    def crash(identity, state):
        if state["tool_index"] == 1:
            raise asyncio.CancelledError()
        real_checkpoint(identity, state)

    monkeypatch.setattr(work_runner, "checkpoint", crash)
    with pytest.raises(asyncio.CancelledError):
        await work_runner.run(work["id"])
    with session_scope() as db:
        assert len(list(db.scalars(select(Task)))) == 1
        assert db.get(Job, work["id"]).status == "running"
    monkeypatch.setattr(work_runner, "checkpoint", real_checkpoint)
    responder(monkeypatch, [response(message="Added Alpha.")])
    await work_runner.run(work["id"])
    with session_scope() as db:
        assert len(list(db.scalars(select(Task)))) == 1
        assert len(list(db.scalars(select(Command)))) == 1
        assert db.get(Job, work["id"]).status == "succeeded"


@pytest.mark.asyncio
async def test_cancel_during_provider_call_prevents_next_write(client, monkeypatch):
    work = action(client)

    def cancel_before_reply():
        assert client.post("/api/v1/work/" + work["id"] + "/cancel").status_code == 200
        return response([("task_create", {"title": "Must not exist"})])

    responder(monkeypatch, [cancel_before_reply])
    await work_runner.run(work["id"])
    with session_scope() as db:
        assert not list(db.scalars(select(Task)))
        assert db.get(Job, work["id"]).status == "cancelled"


@pytest.mark.asyncio
async def test_correction_during_model_call_rebases_request(client, monkeypatch):
    work = action(client)

    def correction():
        result = client.post(
            "/api/v1/work/" + work["id"] + "/revise",
            json={"message": "Call it Beta instead", "expected_revision": 1},
        )
        assert result.status_code == 200, result.text
        return response([("task_create", {"title": "Wrong Alpha"})])

    responder(
        monkeypatch,
        [correction, response([("task_create", {"title": "Beta"})]), response(message="Added Beta.")],
    )
    await work_runner.run(work["id"])
    with session_scope() as db:
        assert [t.title for t in db.scalars(select(Task))] == ["Beta"]


@pytest.mark.asyncio
async def test_revert_preserves_unrelated_later_edits_and_blocks_conflicts(client, monkeypatch):
    with session_scope() as db:
        task = execute(db, get_settings().owner_id, str(uuid4()), "task.create", {"title": "Before"})["data"]
    work = action(client, "Rename to After")
    responder(
        monkeypatch,
        [
            response(
                [
                    (
                        "task_update",
                        {"task_id": task["id"], "expected_revision": task["revision"], "title": "After"},
                    )
                ]
            ),
            response(message="Renamed."),
        ],
    )
    await work_runner.run(work["id"])
    card = next(c for c in client.get("/api/v1/work/" + work["id"]).json()["actions"] if c["kind"] == "task")
    assert card["can_revert"], card
    with session_scope() as db:
        row = db.get(Task, task["id"])
        execute(
            db,
            row.owner_id,
            str(uuid4()),
            "task.update",
            {"task_id": row.id, "expected_revision": row.revision, "notes": "Keep later notes"},
        )
    result = client.post("/api/v1/work/actions/" + card["id"] + "/revert", json={"command_id": str(uuid4())})
    assert result.status_code == 200, result.text
    with session_scope() as db:
        row = db.get(Task, task["id"])
        assert row.title == "Before" and row.notes == "Keep later notes"
    assert client.get("/api/v1/work/" + work["id"]).json()["actions"][0]["reverted"]


@pytest.mark.asyncio
async def test_created_record_revert_archives_and_repeated_revert_is_safe(client, monkeypatch):
    work = action(client)
    responder(monkeypatch, [response([("task_create", {"title": "Alpha"})]), response(message="Added.")])
    await work_runner.run(work["id"])
    card = next(c for c in client.get("/api/v1/work/" + work["id"]).json()["actions"] if c["kind"] == "task")
    assert card["can_revert"], card
    for _ in range(2):
        result = client.post(
            "/api/v1/work/actions/" + card["id"] + "/revert", json={"command_id": str(uuid4())}
        )
        assert result.status_code == 200, result.text
    with session_scope() as db:
        assert db.get(Task, card["entity_id"]).archived


def test_voice_capture_survives_close_and_deduplicates(client):
    work, _payload = accept(client)
    with session_scope() as db:
        accepted = db.get(AgentWork, work["id"])
        inbox = work_intake.open_voice(
            db,
            str(uuid4()),
            accepted.owner_id,
            accepted.account_id,
            accepted.device_id,
            accepted.conversation_id,
        )
        sid = inbox.id
        for event_id, delta, start in [
            ("a", "Add Alpha.", 0),
            ("b", " Also add Beta.", 3000),
            ("b", " Also add Beta.", 3000),
        ]:
            work_intake.append_voice(db, sid, event_id, "user", delta, start, start + 1000)
        captured = work_intake.claim_voice(db, inbox, close=True)
        assert "Alpha" in unseal(captured.input_ciphertext)["message"]
        turns = list(db.scalars(select(AgentWork).where(AgentWork.voice_session_id == sid)))
        assert sorted(unseal(t.input_ciphertext)["message"] for t in turns) == [
            "Add Alpha.",
            "Also add Beta.",
        ]
        assert all(db.get(Job, t.id).kind == "agent_action" for t in turns)
        assert work_intake.claim_voice(db, inbox) is None
        assert db.get(VoiceInbox, sid).closed
        assert db.get(Outbox, captured.id)


def test_explicit_cancel_keeps_saved_actions(client):
    work = action(client)
    with session_scope() as db:
        row = db.get(AgentWork, work["id"])
        execute(db, row.owner_id, row.id + ":0", "task.create", {"title": "Saved"})
    result = client.post("/api/v1/work/" + work["id"] + "/cancel")
    assert result.json()["status"] == "cancelled"
    with session_scope() as db:
        assert [t.title for t in db.scalars(select(Task))] == ["Saved"]


def test_other_account_cannot_inspect_cancel_or_revert_work(client):
    from test_accounts import client_for

    work, _ = accept(client)
    other = client_for("other")
    assert other.get("/api/v1/work").json()["items"] == []
    assert other.get("/api/v1/work/" + work["id"]).status_code == 404
    assert other.post("/api/v1/work/" + work["id"] + "/cancel").status_code == 404


def test_execution_survives_cookie_expiry_but_checks_account_scope(client):
    work = action(client)
    with session_scope() as db:
        row = db.get(AgentWork, work["id"])
        db.query(AuthSession).delete()
        db.expunge(row)
    with agent_work.principal_for(row), session_scope() as db:
        execute(db, row.owner_id, row.id + ":0", "task.create", {"title": "Accepted before disconnect"})
    with session_scope() as db:
        assert len(list(db.scalars(select(Task)))) == 1


def test_voice_preserves_long_input_and_clears_closed_capture(client):
    work, _ = accept(client)
    text = "Add a task with notes " + ("exact words " * 2500) + " END"
    with session_scope() as db:
        accepted = db.get(AgentWork, work["id"])
        inbox = work_intake.open_voice(
            db,
            str(uuid4()),
            accepted.owner_id,
            accepted.account_id,
            accepted.device_id,
            accepted.conversation_id,
        )
        work_intake.append_voice(db, inbox.id, "long", "user", text, 0, 1000)
        captured = work_intake.claim_voice(db, inbox, close=True)
        assert unseal(captured.input_ciphertext)["message"] == text
        assert unseal(inbox.content_ciphertext) == {"entries": []}


@pytest.mark.asyncio
async def test_pending_dependency_waits_then_backend_receives_cancelled_outcome(client):
    first = action(client)
    second = action(client, "Rename Alpha")
    with session_scope() as db:
        row = db.get(AgentWork, second["id"])
        row.dependencies = [first["id"]]
        job = db.get(Job, second["id"])
        assert not agent_work.eligible(db, job)
        agent_work.cancel(db, db.get(AgentWork, first["id"]))
        assert agent_work.eligible(db, job)
        assert job.status == "queued"


def test_per_account_capacity_leaves_other_accounts_eligible(client):
    from test_accounts import client_for

    ids = [action(client)["id"] for _ in range(3)]
    other = action(client_for("other"))
    with session_scope() as db:
        for identity in ids[:2]:
            db.get(Job, identity).status = "running"
        assert not agent_work.eligible(db, db.get(Job, ids[2]))
        assert agent_work.eligible(db, db.get(Job, other["id"]))


@pytest.mark.asyncio
async def test_revoked_member_cannot_finish_accepted_write(client, monkeypatch):
    from jarvis.models import WorkspaceMember
    from test_accounts import client_for, shared

    guest = client_for("guest")
    workspace = shared(client, guest)
    work = action(guest)
    with session_scope() as db:
        db.get(WorkspaceMember, (workspace["id"], "guest")).active = False
    responder(monkeypatch, [])
    await work_runner.run(work["id"])
    with session_scope() as db:
        assert not list(db.scalars(select(Task)))
        assert db.get(Job, work["id"]).status == "failed"


@pytest.mark.asyncio
async def test_revert_same_field_conflict_is_blocked(client, monkeypatch):
    work = action(client)
    responder(monkeypatch, [response([("task_create", {"title": "Alpha"})]), response(message="Added.")])
    await work_runner.run(work["id"])
    card = next(c for c in client.get("/api/v1/work/" + work["id"]).json()["actions"] if c["kind"] == "task")
    with session_scope() as db:
        row = db.get(Task, card["entity_id"])
        execute(
            db,
            row.owner_id,
            str(uuid4()),
            "task.update",
            {"task_id": row.id, "expected_revision": row.revision, "title": "Later edit"},
        )
    result = client.post("/api/v1/work/actions/" + card["id"] + "/revert", json={"command_id": str(uuid4())})
    assert result.status_code == 409
    with session_scope() as db:
        assert db.get(Task, card["entity_id"]).title == "Later edit"


@pytest.mark.asyncio
async def test_device_bridge_waits_for_ack_and_does_not_replay_acknowledged_action(client):
    from jarvis import device_bridge
    from jarvis.ui_control import UIContext, UISync

    work = action(client)
    with session_scope() as db:
        row = db.get(AgentWork, work["id"])
        db.expunge(row)
    action_data = {"id": row.id + ":0", "kind": "activity", "mode": "open"}
    with agent_work.principal_for(row):
        task = asyncio.create_task(device_bridge.dispatch(row.owner_id, row.device_id, action_data))
        await asyncio.sleep(0.01)
        assert not task.done()
        assert device_bridge.sync(row.owner_id, "wrong", UISync(context=UIContext())) == []
        assert device_bridge.sync(row.owner_id, row.device_id, UISync(context=UIContext())) == [action_data]
        device_bridge.sync(
            row.owner_id,
            row.device_id,
            UISync(
                context=UIContext(activity_open=True),
                results=[{"id": action_data["id"], "status": "displayed"}],
            ),
        )
        assert (await task)["status"] == "displayed"
        assert (await device_bridge.dispatch(row.owner_id, row.device_id, action_data))[
            "status"
        ] == "displayed"


def test_oauth_return_target_preserves_invite_and_rejects_external_redirects():
    from jarvis.google_auth import return_destination

    invite = str(uuid4())
    result = return_destination("/?invite=" + invite)
    assert invite in result and "sharing=1" in result and "view=settings" in result
    for url in ("https://evil.example/", "//evil.example/", "/\\evil.example/", "/?view=tasks#external"):
        with pytest.raises(DomainError):
            return_destination(url)


def test_invitation_details_only_reveal_to_matching_google_account(client):
    from datetime import timedelta

    from jarvis.models import WorkspaceInvite, now
    from test_accounts import client_for, post

    guest = client_for("guest")
    wrong = client_for("wrong")
    space = post(client, "/accounts/workspaces", {"name": "Team", "kind": "space"})
    invite = post(
        client,
        "/accounts/invitations",
        {"workspace_id": space["id"], "email": "guest@example.test", "role": "editor"},
    )
    path = "/api/v1/accounts/invitations/" + invite["id"]
    assert wrong.get(path).status_code == 403
    assert guest.get(path).json()["status"] == "pending"
    with session_scope() as db:
        db.get(WorkspaceInvite, invite["id"]).expires_at = now() - timedelta(seconds=1)
    assert guest.get(path).json()["status"] == "expired"


def test_real_worker_resumes_command_after_process_kill(client, test_database, tmp_path):
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path

    work = action(client)
    with session_scope() as db:
        for outbox in db.scalars(select(Outbox)):
            if outbox.job_id != work["id"]:
                db.delete(outbox)
    marker = tmp_path / "committed"
    script = tmp_path / "worker.py"
    script.write_text("""import asyncio,json,os,runpy
from pathlib import Path
from jarvis import work_runner
async def model(agent,messages,definitions,**kwargs):
    prior=any(m.get("role")=="tool" and m.get("name")=="task_create" for m in messages)
    return {"id":"synthetic-response-final" if prior else "synthetic-response-create","usage":{"prompt_tokens":10,"completion_tokens":10},"choices":[{"message":{"role":"assistant","content":"Added once." if prior else None,"tool_calls":[] if prior else [{"id":"synthetic-call","type":"function","function":{"name":"task_create","arguments":json.dumps({"title":"Survived worker kill"})}}]}}]}
async def memory(*args):return ""
work_runner.request_model=model
work_runner.prompt_context=memory
original=work_runner.checkpoint
def checkpoint(identity,state):
    marker=Path(os.environ["SYNTHETIC_CRASH_MARKER"])
    if state["tool_index"]==1 and not marker.exists():
        marker.write_text("committed")
        os.kill(os.getpid(),9)
    original(identity,state)
work_runner.checkpoint=checkpoint
runpy.run_module("jarvis.worker",run_name="__main__")
""")
    env = {
        **os.environ,
        "JARVIS_DATABASE_URL": test_database,
        "JARVIS_WORKER_ENABLED": "true",
        "JARVIS_WORKER_INTERVAL_SECONDS": "1",
        "JARVIS_VAPID_PRIVATE_KEY": "",
        "SYNTHETIC_CRASH_MARKER": str(marker),
    }
    root = Path(__file__).resolve().parents[1]
    with (tmp_path / "worker.log").open("w") as log:
        process = subprocess.Popen([sys.executable, str(script)], cwd=root, env=env, stdout=log, stderr=log)
        try:
            process.wait(timeout=40)
            assert marker.exists(), "The worker did not reach the committed step"
            assert process.returncode == -9
            process = subprocess.Popen(
                [sys.executable, str(script)], cwd=root, env=env, stdout=log, stderr=log
            )
            deadline = time.monotonic() + 40
            while time.monotonic() < deadline:
                with session_scope() as db:
                    if db.get(Job, work["id"]).status == "succeeded":
                        break
                time.sleep(0.25)
            else:
                raise AssertionError("The restarted worker did not finish its saved plan")
            with session_scope() as db:
                assert [t.title for t in db.scalars(select(Task))] == ["Survived worker kill"]
                assert (
                    len(list(db.scalars(select(Command).where(Command.id.startswith(work["id"] + ":"))))) == 1
                )
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


@pytest.mark.asyncio
async def test_recovered_argument_error_does_not_report_partial_success(client, monkeypatch):
    work = action(client)
    responder(
        monkeypatch,
        [
            response([("task_create", {})]),
            response([("task_create", {"title": "Alpha"})]),
            response(message="Added Alpha."),
        ],
    )
    await work_runner.run(work["id"])
    assert client.get("/api/v1/work/" + work["id"]).json()["status"] == "succeeded"


@pytest.mark.asyncio
async def test_browser_saved_receipt_is_linked_to_work_card(client, monkeypatch):
    work = action(client)

    async def browser_save(owner, turn, index, name, args, **kwargs):
        with session_scope() as db:
            result = execute(db, owner, str(uuid4()), "task.create", {"title": "Saved through inline card"})
        return {"status": "displayed", "data": {"command_id": result["command_id"]}}

    monkeypatch.setattr(work_runner, "call_tool", browser_save)
    responder(
        monkeypatch,
        [
            response([("task_create", {"title": "Synthetic browser acknowledgment"})]),
            response(message="Saved."),
        ],
    )
    await work_runner.run(work["id"])
    card = client.get("/api/v1/work/" + work["id"]).json()
    assert card["status"] == "succeeded"
    assert [a["title"] for a in card["actions"] if a["kind"] == "task"] == ["Saved through inline card"]


@pytest.mark.asyncio
async def test_natural_speech_runs_directly_without_an_intake_model(client, monkeypatch):
    speech = "Okay, um, add a task to call Alex.\nAlso, at 10 p.m., write down a visit to Hayden."
    work, _ = accept(client, speech)
    calls = []

    async def model(agent, messages, definitions, **kwargs):
        calls.append(messages)
        assert all(t["name"] != "route_requests" for t in definitions)
        if len(calls) == 1:
            assert messages[-1]["content"] == speech
            return response(
                [
                    ("task_create", {"title": "Call Alex"}),
                    (
                        "task_create",
                        {
                            "title": "Visit Hayden",
                            "due_date": "2026-09-16",
                            "due_time": "22:00",
                            "due_timezone": "America/Chicago",
                        },
                    ),
                ]
            )
        return response(message="Added both.")

    monkeypatch.setattr(work_runner, "request_model", model)
    await work_runner.run(work["id"])
    assert len(calls) == 2  # One tool plan and its confirmation; no routing round.
    result = client.get("/api/v1/work/" + work["id"]).json()
    assert result["status"] == "succeeded" and len(result["actions"]) == 2
    assert not result["children"]


@pytest.mark.asyncio
async def test_independent_backend_requests_execute_concurrently(client, monkeypatch):
    first = action(client, "Add Call Alex")
    second = action(client, "Add Buy milk")
    entered = 0
    both_entered = asyncio.Event()

    async def model(agent, messages, definitions, **kwargs):
        nonlocal entered
        if any(m.get("role") == "tool" for m in messages):
            return response(message="Saved.")
        entered += 1
        if entered == 2:
            both_entered.set()
        await asyncio.wait_for(both_entered.wait(), 3)
        title = "Call Alex" if messages[-1]["content"] == "Add Call Alex" else "Buy milk"
        return response([("task_create", {"title": title})])

    monkeypatch.setattr(work_runner, "request_model", model)
    await asyncio.gather(work_runner.run(first["id"]), work_runner.run(second["id"]))
    with session_scope() as db:
        assert sorted(t.title for t in db.scalars(select(Task))) == ["Buy milk", "Call Alex"]
        assert all(db.get(Job, identity).status == "succeeded" for identity in (first["id"], second["id"]))


@pytest.mark.asyncio
@pytest.mark.parametrize("already_finished", [False, True])
async def test_followup_uses_original_saved_task_running_or_completed(client, monkeypatch, already_finished):
    first = action(client, "Add a task to call Alex")
    milk = action(client, "Add Buy milk")
    follow = action(client, "Actually make that call tomorrow")
    if not already_finished:
        responder(monkeypatch, [response([("work_followup", {"request_id": first["id"]})])])
        await work_runner.run(follow["id"])
        with session_scope() as db:
            row = db.get(AgentWork, follow["id"])
            assert row.dependencies == [first["id"]]
            assert row.result["related_request_id"] == first["id"]
            assert db.get(Job, row.id).status == "queued"
            assert unseal(row.checkpoint_ciphertext)["pending"][0]["function"]["name"] == "work_followup"
            assert not agent_work.eligible(db, db.get(Job, row.id))
            assert agent_work.eligible(db, db.get(Job, milk["id"]))
    responder(
        monkeypatch,
        [response([("task_create", {"title": "Call Alex"})]), response(message="Added Call Alex.")],
    )
    await work_runner.run(first["id"])
    responder(
        monkeypatch, [response([("task_create", {"title": "Buy milk"})]), response(message="Added milk.")]
    )
    await work_runner.run(milk["id"])

    def edit_from_receipt(agent, messages, definitions, **kwargs):
        replies = [m for m in messages if m.get("name") == "work_followup"]
        if not replies:
            return response([("work_followup", {"request_id": first["id"]})])
        earlier = json.loads(replies[-1]["content"])
        assert earlier["status"] == "succeeded"
        record = earlier["saved_records"][0]["data"]
        if any(m.get("name") == "task_update" for m in messages):
            return response(message="Call Alex is due tomorrow.")
        return response(
            [
                (
                    "task_update",
                    {
                        "task_id": record["id"],
                        "expected_revision": record["revision"],
                        "due_date": "2026-09-17",
                    },
                )
            ]
        )

    async def model(*args, **kwargs):
        return edit_from_receipt(*args, **kwargs)

    monkeypatch.setattr(work_runner, "request_model", model)
    await work_runner.run(follow["id"])
    result = client.get("/api/v1/work/" + follow["id"]).json()
    assert result["status"] == "succeeded", result
    assert result["related_request_id"] == first["id"]
    with session_scope() as db:
        tasks = list(db.scalars(select(Task)))
        assert len(tasks) == 2
        assert next(t for t in tasks if t.title == "Call Alex").due_date.isoformat() == "2026-09-17"
    change = next(a for a in result["actions"] if a["kind"] == "task")
    assert change["summary"] == "Updated task: Call Alex" and change["request_id"] == follow["id"]
    reverted = client.post(
        "/api/v1/work/actions/" + change["id"] + "/revert", json={"command_id": str(uuid4())}
    )
    assert reverted.status_code == 200, reverted.text
    with session_scope() as db:
        assert db.get(Task, change["entity_id"]).due_date is None
    original = client.get("/api/v1/work/" + first["id"]).json()["actions"][0]
    assert not original["can_revert"]  # Creation cannot silently erase intervening edits.


@pytest.mark.asyncio
async def test_failed_predecessor_is_reported_without_inventing_a_saved_record(client, monkeypatch):
    first = action(client)
    follow = action(client, "Make that tomorrow")
    with session_scope() as db:
        agent_work.finish(db, db.get(AgentWork, first["id"]), "failed", "No record was saved.")
    responder(
        monkeypatch,
        [
            response([("work_followup", {"request_id": first["id"]})]),
            response(
                [
                    (
                        "work_needs_input",
                        {"question": "The task was not created. Should I create it for tomorrow?"},
                    )
                ]
            ),
        ],
    )
    await work_runner.run(follow["id"])
    with session_scope() as db:
        row = db.get(AgentWork, follow["id"])
        state = unseal(row.checkpoint_ciphertext)
        outcome = json.loads(
            next(m["content"] for m in state["messages"] if m.get("name") == "work_followup")
        )
        assert outcome["status"] == "failed" and not outcome["saved_records"]
        assert not list(db.scalars(select(Task)))


def test_record_reservations_wait_for_conflicts_but_allow_different_records(client):
    from jarvis import work_coordination as coordination

    ids = [action(client)["id"] for _ in range(3)]
    with session_scope() as db:
        rows = [db.get(AgentWork, i) for i in ids]
        for row in rows:
            db.expunge(row)
    coordination.reserve(rows[0], "task_update", {"task_id": "one"}, 0)
    coordination.reserve(rows[1], "task_update", {"task_id": "two"}, 0)
    with pytest.raises(coordination.WorkDeferred) as waiting:
        coordination.reserve(rows[2], "task_update", {"task_id": "one"}, 0)
    assert waiting.value.dependencies == [ids[0]]


def test_newer_request_waits_for_unknown_older_scope_without_deadlocking_older(client):
    from jarvis import work_coordination as coordination

    first, second = action(client), action(client)
    with session_scope() as db:
        a, b = db.get(AgentWork, first["id"]), db.get(AgentWork, second["id"])
        db.expunge_all()
    with pytest.raises(coordination.WorkDeferred):
        coordination.reserve(b, "task_update", {"task_id": "same"}, 0)
    coordination.reserve(a, "task_update", {"task_id": "same"}, 0)


def test_scope_expansion_cannot_overwrite_newer_reserved_work(client):
    from jarvis import work_coordination as coordination

    first, second = action(client), action(client)
    with session_scope() as db:
        a, b = db.get(AgentWork, first["id"]), db.get(AgentWork, second["id"])
        db.expunge_all()
    coordination.reserve(a, "task_update", {"task_id": "one"}, 0)
    coordination.reserve(b, "task_update", {"task_id": "two"}, 0)
    with pytest.raises(DomainError, match="newer request"):
        coordination.reserve(a, "task_update", {"task_id": "two"}, 1)


def test_followup_cannot_target_future_work_or_another_account(client):
    from jarvis import work_coordination as coordination
    from test_accounts import client_for

    first, second = action(client), action(client)
    other = action(client_for("other"))
    with session_scope() as db:
        row = db.get(AgentWork, first["id"])
        db.expunge(row)
    for target in (second["id"], other["id"]):
        with pytest.raises(DomainError):
            coordination.followup(row, {"request_id": target})


@pytest.mark.asyncio
async def test_legacy_intake_retry_executes_original_request_without_quote_validation(client, monkeypatch):
    work, _ = accept(client, "Um, at 10 p.m., please add a visit.")
    with session_scope() as db:
        db.get(Job, work["id"]).kind = "agent_intake"
    responder(monkeypatch, [response([("task_create", {"title": "Visit"})]), response(message="Added.")])
    await work_intake.run(work["id"])
    assert client.get("/api/v1/work/" + work["id"]).json()["status"] == "succeeded"


@pytest.mark.asyncio
async def test_direct_voice_backend_ends_voice_and_hides_conversation_only_cards(client, monkeypatch):
    work, _ = accept(client, "That will be all, goodbye")
    with session_scope() as db:
        row = db.get(AgentWork, work["id"])
        inbox = work_intake.open_voice(
            db, str(uuid4()), row.owner_id, row.account_id, row.device_id, row.conversation_id
        )
        row.voice_session_id = inbox.id
        sid = inbox.id
    responder(monkeypatch, [response([("voice_end", {})])])
    await work_runner.run(work["id"])
    with session_scope() as db:
        assert db.get(VoiceInbox, sid).end_requested
        assert db.get(Job, work["id"]).status == "succeeded"
        assert db.get(AgentWork, work["id"]).result["quiet"]
    assert not client.get("/api/v1/work").json()["items"]


@pytest.mark.asyncio
async def test_partial_request_clarification_preserves_saved_effects(client, monkeypatch):
    work = action(client, "Add Alpha and move that other thing")
    responder(
        monkeypatch,
        [
            response([("task_create", {"title": "Alpha"})]),
            response([("work_needs_input", {"question": "Which other task?"})]),
        ],
    )
    await work_runner.run(work["id"])
    card = client.get("/api/v1/work/" + work["id"]).json()
    assert card["status"] == "needs_input" and len(card["actions"]) == 1
    assert (
        client.post(
            "/api/v1/work/" + work["id"] + "/revise",
            json={"message": "Never mind the other thing", "expected_revision": 1},
        ).status_code
        == 200
    )
    responder(monkeypatch, [response(message="Alpha is saved. I left the others alone.")])
    await work_runner.run(work["id"])
    with session_scope() as db:
        assert len(list(db.scalars(select(Task)))) == 1


@pytest.mark.asyncio
async def test_parallel_request_context_excludes_later_user_instructions(client):
    first, payload = accept(client, "Add Call Alex")
    later = client.post(
        "/api/v1/work", json={**payload, "turn_id": str(uuid4()), "message": "Actually delete everything"}
    )
    assert later.status_code == 200
    with session_scope() as db:
        row = db.get(AgentWork, first["id"])
        db.expunge(row)
    with agent_work.principal_for(row):
        state = await work_runner.initial_state(
            row, __import__("jarvis.agent_models", fromlist=["catalog"]).catalog()["luna"]
        )
    assert not any("Actually delete everything" in m.get("content", "") for m in state["messages"])


@pytest.mark.asyncio
async def test_previous_turns_are_reference_data_not_replayed_user_commands(client):
    _first, payload = accept(client, "Add Call Alex")
    result = client.post(
        "/api/v1/work", json={**payload, "turn_id": str(uuid4()), "message": "Also add Buy milk"}
    ).json()
    with session_scope() as db:
        row = db.get(AgentWork, result["id"])
        db.expunge(row)
    with agent_work.principal_for(row):
        state = await work_runner.initial_state(
            row, __import__("jarvis.agent_models", fromlist=["catalog"]).catalog()["luna"]
        )
    assert [m["content"] for m in state["messages"] if m["role"] == "user"] == ["Also add Buy milk"]
    assert any(
        "EARLIER CONVERSATION DATA" in m["content"] and "Add Call Alex" in m["content"]
        for m in state["messages"]
    )


@pytest.mark.asyncio
async def test_revert_creation_checks_backlinks_that_did_not_change_task_revision(client, monkeypatch):
    from jarvis.models import Note, NoteTaskLink

    work = action(client)
    responder(monkeypatch, [response([("task_create", {"title": "Alpha"})]), response(message="Added.")])
    await work_runner.run(work["id"])
    card = client.get("/api/v1/work/" + work["id"]).json()["actions"][0]
    with session_scope() as db:
        task = db.get(Task, card["entity_id"])
        revision = task.revision
        note = Note(owner_id=task.owner_id, title="Later note", content="Keep this linked work")
        db.add(note)
        db.flush()
        db.add(NoteTaskLink(note_id=note.id, task_id=task.id, linked=True))
        assert task.revision == revision
    card = client.get("/api/v1/work/" + work["id"]).json()["actions"][0]
    assert not card["can_revert"] and "linked" in card["revert_reason"]
    result = client.post("/api/v1/work/actions/" + card["id"] + "/revert", json={"command_id": str(uuid4())})
    assert result.status_code == 409
    with session_scope() as db:
        assert not db.get(Task, card["entity_id"]).archived
