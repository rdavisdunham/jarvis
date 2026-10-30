import asyncio
from uuid import uuid4

import pytest
from jarvis.domain import DomainError
from jarvis.tools import call_tool
from jarvis.ui_control import UIContext, UISync, dispatch, pending, states, sync
from test_agent_runs import host, start  # noqa: F401 - pytest fixture
from test_organization_foundation import create


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


async def test_calendar_detail_requires_a_record_and_preserves_detail_mode():
    from jarvis.domain import DomainError
    from pydantic import ValidationError

    with pytest.raises(DomainError, match="saved calendar item"):
        await call_tool("owner", "turn", 0, "ui_calendar", {"date": "2026-09-13", "open_details": True})
    context = UIContext(editor={"kind": "task", "record_id": "task-id", "mode": "detail",
                                "dirty": False, "busy": False, "fields": []})
    action = asyncio.create_task(dispatch("owner", "phone", {
        "id": "details", "kind": "editor", "operation": "read"
    }))
    await asyncio.sleep(0)
    sync("owner", "phone", UISync(context=context, results=[{
        "id": "details", "status": "displayed",
        "data": {"mode": "detail", "saved": True, "values": {"title": "Saved task"}}
    }]))
    result = await action
    assert result["data"]["saved"] is True
    assert result["data"]["mode"] == "detail"
    with pytest.raises(ValidationError):
        UIContext(editor={"kind": "task", "mode": "javascript"})


async def test_private_chat_control_is_retired_but_old_context_can_sync():
    from jarvis.domain import DomainError
    from jarvis.ui_contracts import UI_TOOLS

    assert "private_chat" not in UI_TOOLS["ui_device"]["parameters"]["properties"]
    context = UIContext(device_preferences={"private_chat": True})
    assert "private_chat" not in context.model_dump()["device_preferences"]
    with pytest.raises(DomainError):
        await call_tool("davin", "turn", 0, "ui_device", {"private_chat": True})


def test_long_screen_lists_are_sampled_instead_of_rejected():
    """A Calendar page with 79 visible items once rejected the whole chat send (October 6)."""
    from jarvis.ui_control import UIContext
    ids = [f"{i:036d}" for i in range(79)]
    context = UIContext(view="calendar", visible_ids=ids, selected_task_ids=ids * 2)
    assert context.visible_ids == ids[:60]
    assert len(context.selected_task_ids) == 100


async def test_exact_calendar_entry_requires_detail_target_and_is_bounded():
    from jarvis.domain import DomainError

    for arguments in (
        {"date": "2026-10-27", "calendar_entry_id": "entry"},
        {"date": "2026-10-27", "entity_id": "record", "calendar_entry_id": "entry"},
        {"date": "2026-10-27", "entity_id": "record", "open_details": True, "calendar_entry_id": "x" * 301},
    ):
        with pytest.raises(DomainError):
            await call_tool("owner", "turn", 0, "ui_calendar", arguments)


def test_every_web_view_is_a_valid_device_context():
    """The browser reports its current page as device context; a page the backend does not know (remotea was
    missing) made every chat message from that page fail validation before Eri saw it."""
    from jarvis.ui_contracts import VIEWS
    from jarvis.ui_control import View
    from typing import get_args

    assert set(get_args(View)) == set(VIEWS)
    assert UIContext(view="remotea").view == "remotea"
    with pytest.raises(ValueError):
        UIContext(view="http://evil.invalid")


async def test_ui_show_says_remotea_is_off():
    with pytest.raises(DomainError) as off:
        await call_tool("davin", "turn", 0, "ui_show", {"view": "remotea"})
    assert off.value.code == "REMOTEA_DISABLED"


async def test_ui_show_opens_remotea_and_one_of_your_runs(host):  # noqa: F811
    """Eri can open the remotea page or one run (what the push deep link ?run= opens), not another
    namespace's run or an unknown one."""
    run_id = start(host, create("task", "Open from Eri")["task_id"])["id"]
    action = (await call_tool("davin", "turn", 0, "ui_show", {"view": "remotea", "entity_id": run_id}))["ui_action"]
    assert (action["view"], action["entity_id"]) == ("remotea", run_id)
    page = (await call_tool("davin", "turn", 0, "ui_show", {"view": "remotea"}))["ui_action"]
    assert page["entity_id"] is None
    for owner, identity in (("stranger", run_id), ("davin", str(uuid4()))):
        with pytest.raises(DomainError):
            await call_tool(owner, "turn", 0, "ui_show", {"view": "remotea", "entity_id": identity})


def test_open_run_id_is_a_validated_uuid_and_named_in_the_prompt():
    """The remotea page sends the open run (a later web PR); control tools without a target then mean it."""
    from uuid import uuid4

    from pydantic import ValidationError
    from jarvis.ui_control import context_prompt

    run_id = str(uuid4())
    assert UIContext(open_run_id=run_id).open_run_id == run_id
    assert UIContext().open_run_id is None and UIContext(open_run_id="").open_run_id is None
    with pytest.raises(ValidationError):
        UIContext(open_run_id="not-a-run")
    assert f"Open run: {run_id}" in context_prompt(UIContext(open_run_id=run_id).model_dump())
    assert "Open run:" not in context_prompt(UIContext().model_dump()) and "Open run:" not in context_prompt(None)
