import asyncio
from unittest.mock import AsyncMock

import pytest
from cryptography.fernet import Fernet
from jarvis import budget, live_voice
from jarvis.config import get_settings
from jarvis.db import session_scope
from jarvis.domain import preferences
from jarvis.live_voice import LiveController
from jarvis.models import AgentWork, BudgetReservation, Conversation, Job, Source, Usage, VoiceInbox
from jarvis.work_crypto import unseal
from jarvis.work_intake import open_voice
from sqlalchemy import select


@pytest.fixture
def controller(monkeypatch):
    monkeypatch.setattr(get_settings(), "integration_encryption_key", Fernet.generate_key().decode())
    monkeypatch.setattr(get_settings(), "openai_api_key", "synthetic")
    monkeypatch.setattr(get_settings(), "gemini_api_key", "synthetic")
    with session_scope() as db:
        conv = Conversation(owner_id="davin", device_id="device", private=True, learning=False)
        db.add(conv)
        db.flush()
        c = LiveController("davin", "device", conv.id, None, preferences(db, "davin"))
        budget.reserve(db, "davin", c.id, 0.1, c.model)
        from jarvis.accounts import ensure_account
        ensure_account(db, "davin")
        open_voice(db, c.id, "davin", "davin", "device", conv.id)
    c.send = AsyncMock()
    return c


async def test_live_fragments_do_not_execute_tools_and_private_history_is_not_saved(controller):
    c = controller
    fragment = {
        "type": "session.input_transcript.delta",
        "event_id": "a",
        "delta": "Create a task",
        "start_ms": 0,
        "end_ms": 1000,
    }
    await c.event(fragment)
    await c.event(fragment)
    assert c.groups[0]["content"] == "Create a task"
    assert not c.work
    await c.close()
    with session_scope() as db:
        assert db.scalar(select(Source)) is None


async def test_delegation_id_and_same_input_are_deduplicated(controller):
    c = controller
    await c.event({"type": "session.input_transcript.delta", "event_id": "request-a",
                   "delta": "Add one task.", "start_ms": 0, "end_ms": 1000})
    event = {"type": "session.delegation.created", "delegation": {"id": "delegation", "target": "client"}}
    await c.event(event)
    await c.event(event)
    await asyncio.gather(*list(c.work))
    await c.delegate("second-delegation")
    with session_scope() as db:
        assert len(list(db.scalars(select(AgentWork)))) == 1
    assert c.receipts == []
    assert c.send.await_args.args[0]["type"] == "session.thinking.append"
    await c.close()



async def test_new_speech_cannot_replace_an_accepted_request(controller):
    c = controller
    await c.event({"type": "session.input_transcript.delta", "event_id": "a",
                   "delta": "Add Alpha.", "start_ms": 0, "end_ms": 1000})
    await c.delegate("first")
    await c.event({"type": "session.input_transcript.delta", "event_id": "b",
                   "delta": "Also add Beta.", "start_ms": 3000, "end_ms": 4000})
    await c.delegate("second")
    with session_scope() as db:
        rows = list(db.scalars(select(AgentWork)))
        assert len(rows) == 2
        assert {unseal(row.input_ciphertext)["message"] for row in rows} == {"Add Alpha.", "Also add Beta."}
        assert not any(row.cancel_requested for row in rows)
    await c.close()



async def test_duration_snapshots_are_cumulative_and_minimum_is_credited(controller):
    c = controller
    c.record_seconds(15)
    c.record_seconds(12)
    c.record_seconds(30)
    c.record_seconds(30)
    with session_scope() as db:
        row = db.get(BudgetReservation, c.id)
        assert float(row.actual) == pytest.approx(0.025)
        assert len(list(db.scalars(select(Usage)))) == 2


async def test_graceful_close_waits_for_final_usage(controller):
    c = controller
    c.session_created = True
    c.ws = AsyncMock()

    async def send(event):
        assert event["type"] == "session.close"
        assert not c.closed
        await c.event({"type": "session.closed", "usage": {"seconds": 42}, "reason": "close_requested"})

    c.send = send
    await c.close()
    assert c.closed and c.finalized.is_set()
    with session_scope() as db:
        row = db.get(BudgetReservation, c.id)
        assert row.state == "closed"
        assert float(row.actual) == pytest.approx(0.035)


async def test_live_interrupt_does_not_cancel_committed_task_work(controller):
    c = controller
    c.receipts = ["saved"]
    await c.interrupt()
    assert c.receipts == ["saved"]
    assert c.send.await_args.args[0]["type"] == "session.instructions.append"


async def test_uncertain_creation_keeps_budget_reservation(controller):
    c = controller
    c.creation_attempted = True
    await c.close(uncertain=True)
    with session_scope() as db:
        assert db.get(BudgetReservation, c.id).state == "uncertain"


def test_invalid_provider_voice_is_rejected_before_provider_call(client):
    conv = client.post("/api/v1/conversations", json={}).json()
    result = client.post(
        "/api/v1/voice/sessions",
        json={"conversation_id": conv["id"], "sdp": "synthetic", "provider": "live", "voice": "cedar"},
    )
    assert result.status_code == 400
    assert result.json()["error"]["code"] == "INVALID_ARGUMENT"


async def test_quiet_memory_update_is_bounded_deduplicated_and_acknowledged(controller, monkeypatch):
    c = controller
    monkeypatch.setattr(
        live_voice,
        "semantic_search",
        AsyncMock(
            return_value=[
                {
                    "id": "memory-id",
                    "revision": 1,
                    "source_id": "source-id",
                    "content": "The user prefers tea.",
                }
            ]
        ),
    )
    c.groups = [{"role": "user", "content": "What drink do I prefer?", "saved": True}]
    await c.refresh_memory_context(c.input_revision)
    message = c.send.await_args.args[0]
    assert message["type"] == "session.thinking.append" and message["delegation_id"] is None
    assert len(message["content"].encode()) <= 480
    await c.event({"type": "session.thinking.appended", "client_event_id": message["event_id"]})
    assert not c.context_pending
    await c.refresh_memory_context(c.input_revision)
    c.send.assert_awaited_once()
    await c.close()


async def test_close_cancels_pending_context_lookup_before_send(controller, monkeypatch):
    c = controller
    started = asyncio.Event()

    async def lookup(*_):
        started.set()
        await asyncio.sleep(60)

    monkeypatch.setattr(live_voice, "semantic_search", lookup)
    c.groups = [{"role": "user", "content": "What drink do I prefer?", "saved": True}]
    c.memory_task = asyncio.create_task(c.refresh_memory_context(c.input_revision))
    await started.wait()
    await c.close()
    assert c.memory_task.cancelled()
    assert not any(call.args[0]["type"] == "session.thinking.append" for call in c.send.await_args_list)


def test_realtime_is_paused_before_starting_or_replacing_a_session(client, monkeypatch):
    from jarvis import voice
    from jarvis.voice import VoiceInput

    existing = type("Existing", (), {"device": "any", "closed": False, "close": AsyncMock()})()
    monkeypatch.setattr(voice, "controllers", {"existing": existing})
    result = client.post(
        "/api/v1/voice/sessions",
        json={
            "conversation_id": "00000000-0000-0000-0000-000000000001",
            "sdp": "synthetic",
            "provider": "realtime",
        },
    )
    assert result.status_code == 400
    assert "temporarily disabled" in result.json()["error"]["message"]
    existing.close.assert_not_awaited()
    assert (
        VoiceInput(conversation_id="00000000-0000-0000-0000-000000000001", sdp="synthetic").provider == "live"
    )
    options = client.get("/api/v1/bootstrap").json()["voice_options"]
    assert list(options) == ["live"]
    assert "willow" in options["live"]["voices"]
    assert "cedar" not in options["live"]["voices"]


async def test_late_answer_keeps_previous_speech_context_without_claiming_saved_actions(controller):
    c = controller
    await c.event({"type": "session.input_transcript.delta", "delta": "Add this to my project.", "start_ms": 0, "end_ms": 1000})
    await c.delegate("clarify")
    await c.event({"type": "session.output_transcript.delta", "delta": "Which project?", "start_ms": 1100, "end_ms": 1800})
    await c.event({"type": "session.input_transcript.delta", "delta": "Home, please.", "start_ms": 2200, "end_ms": 3000})
    await c.delegate("answer")
    with session_scope() as db:
        rows = list(db.scalars(select(AgentWork).order_by(AgentWork.updated_at)))
        assert len(rows) == 2
        data = unseal(rows[-1].input_ciphertext)
        assert "Home" in data["message"]
        assert any("my project" in m["content"] for m in data["context"])
    assert c.receipts == []
    assert all(call.args[0]["type"] == "session.thinking.append" for call in c.send.await_args_list)
    await c.close()



async def test_failed_intake_can_recover_in_same_live_session(controller, monkeypatch):
    c = controller
    c.receipts = ["previously-confirmed"]
    await c.event({"type": "session.input_transcript.delta", "delta": "Find my task.", "start_ms": 0, "end_ms": 1000})
    original = live_voice.claim_voice
    def unavailable(*args, **kwargs):
        raise RuntimeError("Synthetic unavailable intake")
    monkeypatch.setattr(live_voice, "claim_voice", unavailable)
    await c.delegate("failed")
    assert c.error and not c.closed and c.receipts == ["previously-confirmed"]
    monkeypatch.setattr(live_voice, "claim_voice", original)
    await c.delegate("recovered")
    assert c.error is None and c.state == "listening"
    with session_scope() as db:
        assert len(list(db.scalars(select(AgentWork)))) == 1
    await c.close()



async def test_delegated_voice_end_preserves_accepted_work_and_closes_once(controller):
    c = controller
    await c.event({"type": "session.input_transcript.delta", "delta": "Add Alpha.", "start_ms": 0, "end_ms": 1000})
    await c.delegate("save")
    with session_scope() as db:
        db.get(VoiceInbox, c.id).end_requested = True
    await c.report_work()
    assert c.end_requested
    c.session_created = True
    c.ws = type("Socket", (), {"close": AsyncMock()})()
    async def send(event):
        if event["type"] == "session.close":
            await c.event({"type": "session.closed", "reason": "client_requested", "usage": {"seconds": 15}})
    c.send = AsyncMock(side_effect=send)
    await c.close()
    await c.close()
    assert c.closed and c.finalized.is_set()
    c.ws.close.assert_awaited_once()
    with session_scope() as db:
        row = db.scalar(select(AgentWork))
        assert db.get(Job, row.id).status == "queued" and not row.cancel_requested



async def test_voice_end_still_closes_if_final_intake_is_rejected(controller, monkeypatch):
    c = controller
    def rejected(*args, **kwargs):
        from jarvis.domain import DomainError
        raise DomainError("QUEUE_FULL", "Synthetic full queue")
    monkeypatch.setattr(live_voice, "claim_voice", rejected)
    c.request_end()
    await c.close()
    assert c.closed


async def test_read_only_backend_result_is_spoken_even_without_an_action_card(controller):
    from datetime import timedelta
    from uuid import uuid4

    from jarvis.agent_work import enqueue, finish
    from jarvis.models import now
    c=controller
    with session_scope() as db:
        # This fixture uses legacy private history; only the verified synthetic result is needed here.
        row=enqueue(db,c.owner,c.owner,c.device,c.conversation_id,str(uuid4()),"What is due?",voice_session_id=c.id)
        row.transient=False
        finish(db,row,"succeeded","You have two tasks due today.",quiet=True,tool_calls=1)
        db.get(VoiceInbox,c.id).last_input_at=now()-timedelta(seconds=5)
    await c.report_work()
    events=[call.args[0] for call in c.send.await_args_list]
    assert any(e['type']=='session.commentary.append' and 'two tasks due today' in e['content'] for e in events)
    c.send.reset_mock()
    await c.report_work()
    c.send.assert_not_awaited()


async def test_pending_question_keeps_exact_identity_and_cleared_work_is_not_spoken(controller):
    from datetime import timedelta
    from uuid import uuid4

    from jarvis.agent_work import enqueue, finish
    from jarvis.models import now
    c = controller
    with session_scope() as db:
        row = enqueue(db, c.owner, c.owner, c.device, c.conversation_id, str(uuid4()), "Schedule review", voice_session_id=c.id)
        row.transient = False
        finish(db, row, "needs_input", "What time should I use?")
        question = row.result["clarification"]
        ignored = enqueue(db, c.owner, c.owner, c.device, c.conversation_id, str(uuid4()), "Old request", voice_session_id=c.id)
        ignored.transient = False
        finish(db, ignored, "failed", "Do not announce cleared error")
        ignored.result = {**ignored.result, "archived_at": now().isoformat()}
        db.get(VoiceInbox, c.id).last_input_at = now() - timedelta(seconds=5)
    await c.report_work()
    events = [call.args[0] for call in c.send.await_args_list]
    assert any(e["type"] == "session.thinking.append" and question["id"] in e["content"] and question["request_id"] in e["content"] for e in events)
    assert any(e["type"] == "session.commentary.append" and "What time" in e["content"] for e in events)
    c.send.reset_mock()
    await c.report_work()
    c.send.assert_not_awaited()


async def test_continuation_result_is_announced_in_the_new_voice_session(controller):
    from datetime import timedelta
    from uuid import uuid4

    from jarvis.agent_work import enqueue, finish
    from jarvis.models import now
    from jarvis.work_continuation import WorkContinued, answer
    c = controller
    with session_scope() as db:
        root = enqueue(db, c.owner, c.owner, c.device, c.conversation_id, str(uuid4()), "Schedule review", voice_session_id="older-session")
        root.transient = False
        finish(db, root, "needs_input", "What time?")
        question = root.result["clarification"]
        follow = enqueue(db, c.owner, c.owner, c.device, c.conversation_id, str(uuid4()), "9 a.m.", voice_session_id=c.id)
        follow.transient = False
        db.flush()
        db.expunge(follow)
    with pytest.raises(WorkContinued):
        answer(follow, {"request_id": root.id, "clarification_id": question["id"]})
    with session_scope() as db:
        finish(db, db.get(AgentWork, follow.id), "succeeded", "The review deadline is set.")
        db.get(VoiceInbox, c.id).last_input_at = now() - timedelta(seconds=5)
    await c.report_work()
    assert any('The review deadline is set.' in call.args[0]['content'] for call in c.send.await_args_list)
