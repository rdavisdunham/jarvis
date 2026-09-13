import asyncio

import pytest
from jarvis.tools import call_tool
from jarvis.ui_control import UIContext, UISync, dispatch, pending, states, sync


@pytest.fixture(autouse=True)
def clear():
    pending.clear()
    states.clear()
    yield
    pending.clear()
    states.clear()


async def test_ui_action_waits_for_correct_device_acknowledgement():
    task = asyncio.create_task(dispatch("owner", "phone", {"id": "one", "kind": "chat", "mode": "close"}))
    await asyncio.sleep(0)
    assert sync("owner", "laptop", UISync(context=UIContext()))["actions"] == []
    assert sync("owner", "phone", UISync(context=UIContext()))["actions"][0]["id"] == "one"
    sync("owner", "laptop", UISync(context=UIContext(), results=[{"id": "one", "status": "displayed"}]))
    assert not task.done()
    sync(
        "owner",
        "phone",
        UISync(context=UIContext(chat_open=False), results=[{"id": "one", "status": "displayed"}]),
    )
    assert (await task)["status"] == "displayed"
    assert not pending


async def test_invalid_ui_controls_cannot_cross_browser_boundary():
    from jarvis.domain import DomainError

    with pytest.raises(DomainError):
        await call_tool("davin", "turn", 0, "ui_chat", {"mode": "execute_javascript"})
    with pytest.raises(DomainError):
        await call_tool("davin", "turn", 0, "ui_search", {"query": "hello", "view": "http://evil.invalid"})


async def test_editor_ack_preserves_draft_data_and_refusal():
    action = asyncio.create_task(dispatch("owner", "phone", {"id": "read", "kind": "editor", "operation": "read"}))
    await asyncio.sleep(0)
    context = UIContext(editor={"kind": "note", "record_id": "note-id", "dirty": True, "busy": False, "fields": ["title", "content"]})
    sync("owner", "phone", UISync(context=context, results=[{
        "id": "read", "status": "displayed", "data": {"saved": False, "values": {"content": "Draft  with whitespace"}}
    }]))
    result = await action
    assert result["data"]["saved"] is False
    assert result["data"]["values"]["content"] == "Draft  with whitespace"
    action = asyncio.create_task(dispatch("owner", "phone", {"id": "close", "kind": "editor", "operation": "close"}))
    await asyncio.sleep(0)
    sync("owner", "phone", UISync(context=context, results=[{
        "id": "close", "status": "failed", "message": "This draft has unsaved changes."
    }]))
    result = await action
    assert result["status"] == "failed"
    assert "unsaved" in result["message"]


def test_oversized_editor_ack_is_rejected_before_transport():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        UISync(context=UIContext(), results=[{"id": "large", "status": "displayed", "data": {"value": "x" * 150001}}])
