import asyncio
from unittest.mock import AsyncMock

import pytest
from jarvis import budget, live_voice
from jarvis.db import session_scope
from jarvis.domain import preferences
from jarvis.live_voice import LiveController
from jarvis.models import BudgetReservation, Conversation, Source, Usage
from sqlalchemy import select


@pytest.fixture
def controller():
    with session_scope() as db:
        conv = Conversation(owner_id="davin", device_id="device", private=True, learning=False)
        db.add(conv)
        db.flush()
        c = LiveController("davin", "device", conv.id, None, preferences(db, "davin"))
        budget.reserve(db, "davin", c.id, 0.1, c.model)
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


async def test_delegation_id_and_same_input_are_deduplicated(controller, monkeypatch):
    c = controller
    backend = AsyncMock(
        return_value={"message": "Saved one task.", "actions": [{"command_id": "receipt"}], "ui_actions": []}
    )
    monkeypatch.setattr(live_voice, "chat", backend)
    await c.event(
        {"type": "session.input_transcript.delta", "delta": "Save one task.", "start_ms": 0, "end_ms": 1000}
    )
    event = {"type": "session.delegation.created", "delegation": {"id": "delegation", "target": "client"}}
    await c.event(event)
    await c.event(event)
    await asyncio.gather(*list(c.work))
    await c.delegate("second-delegation")
    backend.assert_awaited_once()
    assert c.receipts == ["receipt"]
    assert c.send.await_args.kwargs == {}
    assert c.send.await_args.args[0]["delegation_id"] == "second-delegation"


async def test_new_speech_guard_blocks_stale_tool_request(controller, monkeypatch):
    c = controller

    async def backend(*args, tool_guard, **kwargs):
        assert tool_guard() is None
        await c.event(
            {
                "type": "session.input_transcript.delta",
                "delta": "Actually, cancel that.",
                "start_ms": 3000,
                "end_ms": 4000,
            }
        )
        assert "continued speaking" in tool_guard()
        assert tool_guard() is None  # The next model step has received the correction.
        c.closing = True
        assert "closing" in tool_guard()
        return {"message": "Nothing else changed.", "actions": []}

    monkeypatch.setattr(live_voice, "chat", backend)
    await c.delegate("delegate")


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
    conv = client.post("/api/v1/conversations", json={"private": True}).json()
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


async def test_clarification_and_answer_keep_context_without_claiming_early_save(controller, monkeypatch):
    c = controller
    backend = AsyncMock(side_effect=[
        {"message": "Which project should I use?", "actions": [], "ui_actions": []},
        {"message": "Saved in Home.", "actions": [{"command_id": "saved-home"}], "ui_actions": []},
    ])
    monkeypatch.setattr(live_voice, "chat", backend)
    c.groups = [{"role": "user", "content": "Add this to my project.", "saved": True}]
    c.input_revision = 1
    await c.delegate("clarify")
    assert c.receipts == []
    assert c.send.await_args.args[0]["content"] == "Which project should I use?"
    c.groups.append({"role": "user", "content": "Home, please.", "saved": True})
    c.input_revision += 1
    await c.delegate("answer")
    context = backend.await_args.kwargs["live_context"]
    assert any("Which project" in row["content"] for row in context)
    assert any("Home, please." in row["content"] for row in context)
    assert c.receipts == ["saved-home"] and c.state == "listening"
    await c.close()


async def test_failed_delegation_can_recover_in_same_live_session(controller, monkeypatch):
    c = controller
    c.receipts = ["previously-confirmed"]
    backend = AsyncMock(side_effect=[
        RuntimeError("Synthetic unavailable backend"),
        {"message": "Found your saved task.", "actions": [], "ui_actions": []},
    ])
    monkeypatch.setattr(live_voice, "chat", backend)
    c.groups = [{"role": "user", "content": "Find my task.", "saved": True}]
    c.input_revision = 1
    await c.delegate("failed")
    assert c.error and not c.closed and c.receipts == ["previously-confirmed"]
    assert "do not claim further success" in c.send.await_args.args[0]["content"]
    c.input_revision += 1
    await c.delegate("recovered")
    assert c.error is None and c.state == "listening"
    assert c.receipts == ["previously-confirmed"]
    assert c.send.await_args.args[0]["content"] == "Found your saved task."
    await c.close()
