import json
from pathlib import Path

import pytest
from sqlalchemy import select

from jarvis.assistant_identity import available_assistant_name, wake_request
from jarvis.db import session_scope
from jarvis.models import Actor
from jarvis.personality import SYSTEM_PROMPT
from jarvis.productivity import defaults

CASES = json.loads((Path(__file__).parent / "fixtures/assistant-wake.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["text"])
def test_shared_wake_grammar(case):
    assert wake_request(case["text"]) == case["request"]


def test_assistant_identity_is_separate_from_app_and_allows_both_pronunciations():
    assert "You are Dani" in SYSTEM_PROMPT
    assert "Eridani app" in SYSTEM_PROMPT
    assert "DAN-ee" in SYSTEM_PROMPT and "DAH-nee" in SYSTEM_PROMPT


def test_pronouns_follow_the_chosen_voice():
    from jarvis.agent_instructions import backend_instructions, live_instructions
    from jarvis.voice_options import LIVE_VOICES, pronouns

    assert pronouns("marin") == pronouns(None) == pronouns("Gleam") == "she/her"
    assert pronouns("stone") == pronouns("meridian") == "he/him"
    assert pronouns("custom-voice") == "they/them"
    assert all(pronouns(v) != "they/them" for v in LIVE_VOICES)
    prefs = {"timezone": "America/Chicago", "default_reminder_hour": 9}
    assert live_instructions(prefs, "stone").startswith("You are Dani (he/him)")
    assert live_instructions(prefs, "willow").startswith("You are Dani (she/her)")
    typed = backend_instructions(prefs, ui_context={"device_preferences": {"voice": "cinder"}})
    assert "Your pronouns: he/him" in typed and "Your pronouns: she/her" in backend_instructions(prefs)


def test_personality_is_charming_with_restrained_wit():
    from jarvis.personality import PERSONALITY
    assert PERSONALITY in SYSTEM_PROMPT
    assert "charming" in PERSONALITY and "Most replies have none" in PERSONALITY
    assert "never stiff, formal" in PERSONALITY and "formal but" not in SYSTEM_PROMPT


def test_name_collision_preserves_existing_people_and_agents():
    assert available_assistant_name([]) == "Dani"
    assert available_assistant_name(["dani"]) == "Dani (assistant)"
    assert available_assistant_name(["Dani", "Dani (assistant)"]) == "Dani (assistant 2)"


def test_defaults_seed_dani_once_and_preserve_user_renames_and_archives():
    with session_scope() as db:
        defaults(db, "fresh-account")
        assistant = db.scalar(select(Actor).where(Actor.owner_id == "fresh-account", Actor.kind == "agent"))
        assert assistant.name == "Dani"
        assistant.name, assistant.archived = "Custom helper", True
        db.flush()
        defaults(db, "fresh-account")
        agents = list(db.scalars(select(Actor).where(Actor.owner_id == "fresh-account", Actor.kind == "agent")))
        assert [(a.id, a.name, a.archived) for a in agents] == [(assistant.id, "Custom helper", True)]


def test_defaults_do_not_mistake_a_person_named_dani_for_the_assistant():
    with session_scope() as db:
        person = Actor(owner_id="collision", name="Dani", kind="person")
        db.add(person)
        db.flush()
        defaults(db, "collision")
        actors = list(db.scalars(select(Actor).where(Actor.owner_id == "collision")))
        assert person.name == "Dani"
        assert any(a.name == "Dani (assistant)" and a.kind == "agent" for a in actors)
