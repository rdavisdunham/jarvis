"""Set up with Dani in voice: GPT-Live holds the interview (docs/SETUP_WITH_DANI_PLAN.md, PR 3)."""

import httpx
import pytest
from cryptography.fernet import Fernet

from jarvis import budget, live_http, setup
from jarvis.agent_instructions import LIVE_DELEGATION, LIVE_SETUP, live_instructions
from jarvis.config import get_settings
from jarvis.db import session_scope
from jarvis.domain import DomainError, preferences
from jarvis.live_voice import LiveController
from jarvis.models import Conversation, OwnerSettings, Source

PREFS = {"timezone": "UTC", "preferred_name": "Davis"}


def test_setup_mode_replaces_the_delegation_policy():
    plain = live_instructions(PREFS)
    interview = live_instructions(PREFS, mode="setup", setup_notes={"covered": ["you"]})
    assert LIVE_DELEGATION in plain and LIVE_SETUP not in plain
    assert LIVE_SETUP in interview and LIVE_DELEGATION not in interview
    assert "Do not generate <|SpawnThinking|> for interview answers" in interview
    assert 'Setup notes DATA: {"covered": ["you"]}' in interview
    assert "Setup notes DATA" not in live_instructions(PREFS, mode="setup")


def test_voice_delegated_setup_turns_get_the_narrow_block():
    with session_scope() as db:
        started = setup.start(db, "davin", "device", setup.Start(mode="voice"))
        assert setup.interview_block(db, "davin", started["conversation_id"], voice=True) == setup.SETUP_VOICE_TURN
        assert "SETUP INTERVIEW" in setup.interview_block(db, "davin", started["conversation_id"])
        other = Conversation(owner_id="davin", device_id="device")
        db.add(other)
        db.flush()
        assert setup.interview_block(db, "davin", other.id, voice=True) == ""


async def connect_payload(monkeypatch, conversation_id):
    monkeypatch.setattr(get_settings(), "integration_encryption_key", Fernet.generate_key().decode())
    monkeypatch.setattr(get_settings(), "openai_api_key", "synthetic")
    sent = {}

    class Client:
        async def post(self, url, headers=None, json=None):
            sent.update(json["session"])
            return httpx.Response(500, json={})

    monkeypatch.setattr(live_http, "client", lambda: Client())
    monkeypatch.setattr("jarvis.about_you.for_voice", lambda *args: "")
    monkeypatch.setattr("jarvis.access.assert_current", lambda *args: None)
    with session_scope() as db:
        from jarvis.accounts import ensure_account
        ensure_account(db, "davin")
        c = LiveController("davin", "device", conversation_id, None, preferences(db, "davin"))
        budget.reserve(db, "davin", c.id, 0.1, c.model)
    with pytest.raises(DomainError):
        await c.connect("synthetic")
    return sent


async def test_setup_voice_session_reads_the_kept_transcript_with_the_setup_policy(monkeypatch):
    with session_scope() as db:
        db.add(OwnerSettings(owner_id="davin", values={"history_enabled": False}))
    with session_scope() as db:
        started = setup.start(db, "davin", "device", setup.Start(mode="voice"))
        for n in range(30):
            db.add(Source(owner_id="davin", conversation_id=started["conversation_id"], native_id=f"t{n}",
                          kind="transcript", role="user", content=f"answer {n}"))
    session = await connect_payload(monkeypatch, started["conversation_id"])
    assert LIVE_SETUP in session["instructions"]
    texts = [item["content"][0]["text"] for item in session["input"]]
    # History is off, but the setup conversation is kept and read: the opening plus 30 answers fit in the window.
    assert texts[0] == setup.OPENING and len(texts) == 31


async def test_ordinary_voice_session_is_unchanged(monkeypatch):
    with session_scope() as db:
        conv = Conversation(owner_id="davin", device_id="device")
        db.add(conv)
        db.flush()
        conv_id = conv.id
    session = await connect_payload(monkeypatch, conv_id)
    assert LIVE_DELEGATION in session["instructions"] and LIVE_SETUP not in session["instructions"]


def test_setup_turns_run_on_terra(monkeypatch):
    from uuid import uuid4

    from jarvis import agent_work
    from jarvis.models import Job

    monkeypatch.setattr(get_settings(), "integration_encryption_key", Fernet.generate_key().decode())
    monkeypatch.setattr(get_settings(), "openai_api_key", "synthetic")
    with session_scope() as db:
        started = setup.start(db, "davin", "device", setup.Start(mode="voice"))
        other = Conversation(owner_id="davin", device_id="device")
        db.add(other)
        db.flush()
        spoken = agent_work.enqueue(db, "davin", "davin", "device", started["conversation_id"], str(uuid4()),
                                    "Propose my organization")
        plain = agent_work.enqueue(db, "davin", "davin", "device", other.id, str(uuid4()), "Add milk")
        assert db.get(Job, spoken.id).payload["profile"] == setup.SETUP_PROFILE
        assert db.get(Job, plain.id).payload["profile"] != setup.SETUP_PROFILE
