import pytest
from jarvis import budget
from jarvis.db import session_scope
from jarvis.domain import preferences
from jarvis.voice import Controller


@pytest.fixture
def controller():
    with session_scope() as db:
        prefs = preferences(db, "davin")
    c = Controller("davin", "device", "conversation", None, prefs)
    with session_scope() as db:
        budget.reserve(db, "davin", c.id, 5, c.model)
    c.sent = []

    async def send(event):
        c.sent.append(event)

    c.send = send
    return c


async def test_cancelled_gate_cannot_start_a_response(controller):
    c = controller
    c.epoch = 2
    await c.finish_response({"output": [{"content": [{"text": "RESPOND"}]}]}, {"epoch": 1, "phase": "gate"})
    assert c.sent == []


async def test_silence_and_wait_never_create_audio(controller):
    c = controller
    for result, state in [("SILENT", "listening"), ("WAIT", "waiting")]:
        await c.finish_response({"output": [{"content": [{"text": result}]}]}, {"epoch": 0, "phase": "gate"})
        assert c.state == state
        assert not c.sent


async def test_gate_has_no_tools_and_invalid_result_is_unresolved(controller):
    c = controller
    await c.event({"type": "input_audio_buffer.committed", "item_id": "input-1"})
    response = c.sent[0]["response"]
    assert response["conversation"] == "none" and response["tools"] == []
    assert response["output_modalities"] == ["text"]
    await c.finish_response({"output": [{"content": [{"text": "SURE!"}]}]}, {"epoch": 0, "phase": "gate"})
    assert c.state == "unresolved"


async def test_unowned_provider_response_cancelled(controller):
    c = controller
    await c.event({"type": "response.created", "response": {"id": "unexpected", "metadata": {}}})
    assert c.sent == [{"type": "response.cancel", "response_id": "unexpected"}]


async def test_interruption_cancels_classifier_and_clears_audio(controller):
    c = controller
    c.responses["gate-response"] = {"phase": "gate", "epoch": 0}
    await c.interrupt()
    assert c.epoch == 1
    assert {"type": "response.cancel", "response_id": "gate-response"} in c.sent
    assert {"type": "output_audio_buffer.clear"} in c.sent


@pytest.mark.parametrize("state", ["listening", "waiting"])
async def test_live_browser_has_no_elapsed_turn_or_silence_cap(controller, monkeypatch, state):
    import asyncio
    from unittest.mock import AsyncMock

    from jarvis import voice

    c = controller
    c.started = c.activity = 0
    c.turns = 100
    c.client_seen = 10000
    c.state = state
    c.close = AsyncMock()
    monkeypatch.setattr(voice.time, "monotonic", lambda: 10000)
    ticks = 0

    async def tick(_):
        nonlocal ticks
        if ticks:
            raise asyncio.CancelledError()
        ticks += 1

    monkeypatch.setattr(voice.asyncio, "sleep", tick)
    with pytest.raises(asyncio.CancelledError):
        await c.watch()
    c.close.assert_not_awaited()
    assert c.state == state


async def test_abandoned_browser_session_is_closed(controller, monkeypatch):
    import asyncio
    from unittest.mock import AsyncMock

    from jarvis import voice

    c = controller
    c.client_seen = 0
    c.close = AsyncMock()
    monkeypatch.setattr(voice.time, "monotonic", lambda: 31)
    ticks = 0

    async def tick(_):
        nonlocal ticks
        if ticks:
            raise asyncio.CancelledError()
        ticks += 1

    monkeypatch.setattr(voice.asyncio, "sleep", tick)
    with pytest.raises(asyncio.CancelledError):
        await c.watch()
    c.close.assert_awaited_once()


async def test_recovered_turn_clears_stale_provider_error(controller):
    c = controller
    await c.event({"type": "error", "error": {"code": "server_error"}})
    assert c.error
    await c.finish_response({"output": [{"content": [{"text": "SILENT"}]}]}, {"epoch": 0, "phase": "gate"})
    assert c.error is None
    assert c.state == "listening"


async def test_new_speech_clears_error_and_finished_action_cannot_be_submitted_again(controller):
    c = controller
    c.error = "old error"
    await c.event({"type": "input_audio_buffer.speech_started", "item_id": "new"})
    assert c.error is None
    c.current_item = "new"
    c.state = "waiting"
    assert c.can_submit()
    c.tool_index = 1
    assert not c.can_submit()
    c.tool_index = 0
    c.answered_items.add("new")
    assert not c.can_submit()


@pytest.mark.parametrize("private", [True, False])
async def test_voice_final_transcript_retention_and_deduplication(controller, client, private):
    from jarvis.models import Source
    from sqlalchemy import select

    c = controller
    conv = client.post("/api/v1/conversations", json={"private": private}).json()
    c.conversation_id = conv["id"]
    await c.event({"type": "input_audio_buffer.speech_started", "item_id": "spoken"})
    event = {
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": "spoken",
        "transcript": "A synthetic transcript.",
    }
    await c.event(event)
    await c.event(event)
    with session_scope() as db:
        sources = list(db.scalars(select(Source).where(Source.conversation_id == conv["id"])))
        assert len(sources) == (0 if private else 1)
        if sources:
            assert sources[0].content == event["transcript"]
            assert sources[0].created_at == c.input_started["spoken"]
