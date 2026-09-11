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
