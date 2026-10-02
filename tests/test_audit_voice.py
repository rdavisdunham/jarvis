"""Audit H1/§4: Live owns the conversation until it delegates; no request is lost."""

import asyncio
import time
from datetime import timedelta
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet
from jarvis import budget, work_intake
from jarvis.agent_work import enqueue, finish
from jarvis.config import get_settings
from jarvis.db import session_scope
from jarvis.domain import advisory, preferences
from jarvis.live_voice import LiveController
from jarvis.models import AgentWork, Conversation, Job, VoiceInbox, now
from jarvis.work_continuation import touch_root
from jarvis.work_crypto import unseal
from jarvis.work_intake import flush_voice, open_voice
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
    c.settle_min = c.settle_quiet = 0.05
    return c


async def say(c, role, text, start, end=None):
    kind = "session.input_transcript.delta" if role == "user" else "session.output_transcript.delta"
    await c.event({"type": kind, "event_id": str(uuid4()), "delta": text, "start_ms": start, "end_ms": end or start + 900})


def quiet(c, seconds=10):
    with session_scope() as db:
        db.get(VoiceInbox, c.id).last_input_at = now() - timedelta(seconds=seconds)


def scan():
    with session_scope() as db:
        flush_voice(db)


def jobs():
    with session_scope() as db:
        rows = list(db.scalars(select(AgentWork).order_by(AgentWork.updated_at)))
        return [(row.id, unseal(row.input_ciphertext), row.parent_id) for row in rows]


async def test_sept_24_incomplete_turn_waits_for_live_then_one_job(controller):
    c = controller
    await say(c, "user", "Add this to Todo.", 0)
    quiet(c)
    scan()
    assert jobs() == []  # Quiet flush retains; it never enqueues.
    await say(c, "assistant", "What should I add?", 1200)
    await say(c, "user", "Milk.", 3000)
    quiet(c)
    scan()
    assert jobs() == []
    await c.delegate("complete")
    [(_, data, _)] = jobs()
    assert data["message"] == "Add this to Todo.\nMilk."
    assert {"role": "assistant", "content": "What should I add?"} in data["context"]
    await c.delegate("repeat")  # Nothing new: no second job.
    assert len(jobs()) == 1
    await c.close()
    assert len(jobs()) == 1


async def test_mid_utterance_pause_is_one_job(controller):
    c = controller
    await say(c, "user", "Add a task to call", 0, 1000)
    await say(c, "user", " mom tomorrow at noon.", 4500, 5500)  # 3.5 s pause
    quiet(c, 4)
    scan()
    assert jobs() == []
    await c.delegate("one")
    [(_, data, _)] = jobs()
    assert "call" in data["message"] and "mom tomorrow" in data["message"]
    await c.close()


async def test_delegation_waits_for_the_final_transcript(controller):
    c = controller
    c.settle_min, c.settle_quiet = 0.05, 0.3
    await say(c, "user", "Add Alpha", 0)
    task = asyncio.create_task(c.delegate("early"))
    await asyncio.sleep(0.15)
    await say(c, "user", " and Beta.", 1000)
    await task
    [(_, data, _)] = jobs()
    assert data["message"] == "Add Alpha and Beta."
    await c.close()


async def test_goodbye_and_chitchat_are_not_enqueued_at_close(controller):
    c = controller
    await say(c, "user", "How was your weekend?", 0)
    await say(c, "assistant", "Quiet, thank you.", 1200)
    await say(c, "user", "Okay, thanks. Bye!", 3000)
    await c.close()
    assert jobs() == []
    with session_scope() as db:
        assert db.get(VoiceInbox, c.id).closed


async def test_undelegated_action_at_close_is_a_draft_without_the_farewell(controller):
    c = controller
    await say(c, "user", "Remind me to water the plants.", 0)
    await say(c, "user", "Thank you, goodbye.", 3000)
    await c.close()
    assert jobs() == []
    with session_scope() as db:
        draft = unseal(db.get(VoiceInbox, c.id).content_ciphertext)["draft"]
        assert draft["message"] == "Remind me to water the plants."


async def test_dismissed_request_at_close_is_dropped(controller):
    c = controller
    await say(c, "user", "Add this to Todo.", 0)
    await say(c, "user", "Actually never mind.", 3000)
    await c.close()
    assert jobs() == []


async def test_pending_delegation_is_claimed_in_full_when_voice_closes(controller):
    c = controller
    c.settle_min = 5
    await say(c, "user", "Okay, bye", 0)
    await c.event({"type": "session.delegation.created", "delegation": {"id": "farewell", "target": "client"}})
    await asyncio.sleep(0.05)
    await c.close()
    [(_, data, _)] = jobs()
    assert data["message"] == "Okay, bye"  # Live delegated it, so the backend can call voice_end.


def pending_question(c, text="Which list should I use?"):
    with session_scope() as db:
        row = enqueue(db, c.owner, c.owner, c.device, c.conversation_id, str(uuid4()), "Add milk", voice_session_id=c.id)
        row.transient = False
        finish(db, row, "needs_input", text)
        db.get(VoiceInbox, c.id).last_input_at = now() - timedelta(seconds=5)
        return row.id, row.result["clarification"]


async def test_needs_input_question_is_spoken_once_and_not_on_revision_bumps(controller):
    c = controller
    work_id, _ = pending_question(c)
    await c.report_work()
    events = [call.args[0] for call in c.send.await_args_list]
    assert [e["type"] for e in events] == ["session.commentary.append"]
    assert "Which list" in events[0]["content"] and "add no question of your own" in events[0]["content"]
    c.send.reset_mock()
    with session_scope() as db:
        row = db.get(AgentWork, work_id)
        row.revision += 1  # e.g. touch_root/continuation bookkeeping
        touch_root(db, row)
    await c.report_work()
    c.send.assert_not_awaited()


async def test_heard_question_does_not_capture_the_next_request(controller):
    # Regression (2026-10-02): after Eri asked a question, every later spoken request was
    # attached to it as the "answer", so new requests (e.g. Linear changes) never ran.
    c = controller
    work_id, _ = pending_question(c)
    await c.report_work()
    await say(c, "user", "Move the Linear issue about onboarding to In Progress.", 0)
    await c.delegate("new request")
    with session_scope() as db:
        other = db.scalar(select(AgentWork).where(AgentWork.id != work_id))
        assert other.parent_id is None
        assert "answered_clarification_id" not in other.result
        assert db.get(Job, work_id).status == "needs_input"
    await c.close()


async def test_unheard_question_does_not_capture_an_unrelated_turn(controller):
    c = controller
    work_id, _ = pending_question(c)
    await say(c, "user", "Add eggs to Todo.", 0)
    await c.delegate("independent")
    with session_scope() as db:
        other = db.scalar(select(AgentWork).where(AgentWork.id != work_id))
        assert other.parent_id is None and db.get(Job, work_id).status == "needs_input"
    await c.close()


async def test_stale_question_is_not_spoken_after_later_voice_work_started(controller):
    c = controller
    work_id, _ = pending_question(c)
    with session_scope() as db:
        later = enqueue(db, c.owner, c.owner, c.device, c.conversation_id, str(uuid4()), "Add eggs", voice_session_id=c.id)
        later.transient = False
        db.get(Job, later.id).status = "running"
        db.get(Job, later.id).created_at = db.get(Job, work_id).created_at + timedelta(seconds=1)
        db.get(VoiceInbox, c.id).last_input_at = now() - timedelta(seconds=5)
    await c.report_work()
    assert not any("Which list" in call.args[0].get("content", "") for call in c.send.await_args_list)
    assert not c.pending_question


async def test_abandoned_inbox_is_closed_by_flush_with_one_draft(controller):
    c = controller
    await say(c, "user", "Add Alpha.", 0)
    await say(c, "user", "Add Beta.", 4000)
    quiet(c, 60)
    scan()
    assert jobs() == []
    quiet(c, 3600)
    scan()
    assert jobs() == []
    with session_scope() as db:
        row = db.get(VoiceInbox, c.id)
        assert row.closed
        assert unseal(row.content_ciphertext)["draft"]["message"] == "Add Alpha.\nAdd Beta."


async def test_flush_skips_an_inbox_locked_by_capture(controller):
    c = controller
    await say(c, "user", "Add Alpha.", 0)
    quiet(c, 3600)
    with session_scope() as holder:
        advisory(holder, "voice-inbox:" + c.id)
        started = time.monotonic()
        scan()
        assert time.monotonic() - started < 5
    assert jobs() == []
    with session_scope() as db:
        assert not db.get(VoiceInbox, c.id).closed


async def test_expiry_takes_locks_and_skips_running_work(controller):
    c = controller
    with session_scope() as db:
        running = enqueue(db, c.owner, c.owner, c.device, c.conversation_id, str(uuid4()), "Running")
        queued = enqueue(db, c.owner, c.owner, c.device, c.conversation_id, str(uuid4()), "Queued")
        db.get(Job, running.id).status = "running"
        running.expires_at = queued.expires_at = now() - timedelta(seconds=1)
        ids = running.id, queued.id
    scan()
    with session_scope() as db:
        assert db.get(Job, ids[0]).status == "running" and db.get(AgentWork, ids[0]).input_ciphertext
        assert db.get(Job, ids[1]).status == "expired" and db.get(AgentWork, ids[1]).input_ciphertext is None


def test_live_instructions_gather_detail_and_respect_backend_questions():
    from jarvis.agent_instructions import live_instructions
    with session_scope() as db:
        text = live_instructions(preferences(db, "davin"))
    assert "gather the minimum missing detail yourself" in text
    assert "never ask a competing or duplicate question" in text


def test_close_policy_filters():
    turns = lambda *texts: [{"role": "user", "content": t} for t in texts]
    assert work_intake.undelegated(turns("Thanks so much, goodbye!")) == ""
    assert work_intake.undelegated(turns("That's all for now.")) == ""
    assert work_intake.undelegated(turns("What's the capital of Peru?")) == "What's the capital of Peru?"
    assert work_intake.undelegated(turns("Move my dentist to Friday", "okay bye")) == "Move my dentist to Friday"


async def test_quiet_zero_tool_question_is_still_spoken(controller):
    c = controller
    with session_scope() as db:
        row = enqueue(db, c.owner, c.owner, c.device, c.conversation_id, str(uuid4()), "Add this to Todo", voice_session_id=c.id)
        row.transient = False
        finish(db, row, "needs_input", "What should I add?", quiet=True, tool_calls=0)
        db.get(VoiceInbox, c.id).last_input_at = now() - timedelta(seconds=5)
    await c.report_work()
    [call] = c.send.await_args_list
    assert "What should I add?" in call.args[0]["content"]
