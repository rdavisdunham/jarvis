"""The owner's "About you" summary: material, the daily sweep, the job, privacy, and voice session start."""

import json
from datetime import timedelta
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from jarvis import about_you
from jarvis.cost_features import LABELS
from jarvis.db import session_scope
from jarvis.domain import DomainError, preferences
from jarvis.model_map import JOBS
from jarvis.models import AboutYou, Actor, Job, Memory, OwnerSettings, Source, now
from sqlalchemy import select
from test_live_voice import controller as controller  # noqa: F401 - fixture

OWNER = "davin"
EXAMPLE = ("Davis is a product engineer who builds Eridani, his personal planner and voice assistant, and runs AI "
           "call-grading work for ABC Home & Commercial, where Janiece leads customer service. He lives in Chicago "
           "with his cat Hayes and prefers tea over coffee. He wants short, direct answers, likes plans written as "
           "plain steps, and asks before anything is sent to other people. His priorities this season are shipping "
           "Dani's voice features and keeping Mondays free for deep work.")


def fact(content, owner=OWNER, **changes):
    with session_scope() as db:
        source = Source(owner_id=owner, native_id="fact:" + uuid4().hex, kind="chat", role="user", content=content)
        db.add(source)
        db.flush()
        row = Memory(owner_id=owner, source_id=source.id, content=content, fingerprint=uuid4().hex, **changes)
        db.add(row)
        db.flush()
        return row.id


def settings(**values):
    with session_scope() as db:
        row = db.get(OwnerSettings, OWNER)
        if row:
            row.values = {**row.values, **values}
        else:
            db.add(OwnerSettings(owner_id=OWNER, values=values))


def sweep(clock=None):
    with session_scope() as db:
        return about_you.queue_due(db, clock)


def jobs(owner=OWNER):
    with session_scope() as db:
        return [j.id for j in db.scalars(select(Job).where(Job.kind == about_you.KIND, Job.owner_id == owner)
                                         .order_by(Job.created_at))]


def stored():
    with session_scope() as db:
        row = db.get(AboutYou, OWNER)
        return None if row is None else {"summary": row.summary, "ids": list(row.memory_ids),
                                         "generated_at": row.generated_at, "next": row.next_check_at}


def due_now(**changes):
    with session_scope() as db:
        row = db.get(AboutYou, OWNER)
        row.next_check_at = now() - timedelta(minutes=1)
        for key, value in changes.items():
            setattr(row, key, value)


class Model:
    """Stands in for Luna: records each request's material and answers with a fixed summary."""

    def __init__(self, text=EXAMPLE):
        self.text, self.calls = text, []

    def __call__(self, body):
        self.calls.append(json.loads(body["messages"][1]["content"]))
        return about_you.Summary(summary=self.text), {"model": "gpt-5.6-luna", "cost": 0.0004}


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setattr(about_you, "enabled", lambda: True)


def build(model):
    """One sweep plus the job it queued."""
    before = set(jobs())
    sweep()
    new = [j for j in jobs() if j not in before]
    for job_id in new:
        about_you.process(job_id, model=model)
    return new


# ---------------------------------------------------------------- generation


def test_cost_feature_and_model_route_are_registered():
    assert LABELS["about_you"] == "About you summary"
    assert JOBS["about_you"].cost_feature == "about_you" and JOBS["about_you"].path == "extraction"
    body = about_you.request({"saved_facts_newest_first": ["x"]})
    assert body["model"] == "gpt-5.6-luna" and body["store"] is False
    assert "untrusted" in body["messages"][0]["content"]


def test_generation_reads_only_live_memories_and_is_skipped_when_unchanged(on):
    live = fact("The user prefers tea over coffee.")
    fact("The user once lived in Denver.", suppressed=True)
    merged_into = fact("The user has a cat named Hayes.")
    fact("The user has a cat called Hayes.", merged_into_id=merged_into)
    gone = fact("The user's old phone number is private.")
    with session_scope() as db:
        db.get(Source, db.get(Memory, gone).source_id).deleted_at = now()
    fact("Someone else's fact.", owner="other-owner")
    with session_scope() as db:
        db.add(Actor(owner_id=OWNER, name="Janiece", kind="person"))
    model = Model()
    assert len(build(model)) == 1
    material = model.calls[0]
    assert material["saved_facts_newest_first"] == ["The user has a cat named Hayes.",
                                                    "The user prefers tea over coffee."]
    assert material["people"] == ["Janiece"] and material["profile"]["preferred_name"]
    row = stored()
    assert row["summary"] == EXAMPLE and set(row["ids"]) == {live, merged_into}
    assert 400 <= len(row["summary"]) <= 700 and "{" not in row["summary"]
    # Unchanged material: the next daily look queues nothing and a stray job makes no call.
    due_now()
    assert sweep() == 0 and len(jobs()) == 1
    with session_scope() as db:
        stray = about_you.schedule(db, OWNER)
    about_you.process(stray, model=model)
    assert len(model.calls) == 1
    with session_scope() as db:
        assert db.get(Job, stray).result == {"stored": False, "reason": "no_change"}
    # Not due yet: nothing is even looked at.
    assert sweep() == 0


def test_regenerates_on_change_or_when_older_than_seven_days(on):
    fact("The user prefers tea over coffee.")
    model = Model()
    build(model)
    first = stored()
    assert first["next"] > now() + timedelta(hours=23)
    # Within the day, a new fact waits for the daily look.
    fact("The user works with Janiece at ABC.")
    assert build(model) == []
    due_now()
    assert len(build(model)) == 1
    assert "The user works with Janiece at ABC." in model.calls[-1]["saved_facts_newest_first"]
    # Nothing changed, but the summary is over a week old.
    due_now(generated_at=now() - timedelta(days=8))
    assert len(build(model)) == 1 and len(model.calls) == 3
    assert stored()["generated_at"] > now() - timedelta(minutes=1)


def test_a_forgotten_fact_is_withheld_at_once_and_dropped_by_the_next_pass(on):
    keep, forget = fact("The user prefers tea over coffee."), fact("The user is training for a marathon.")
    model = Model()
    build(model)
    prefs = {"preferred_name": "Davis", "memory_learning": True, "live_memory_updates": True}
    assert about_you.for_voice(OWNER, prefs) == about_you.VOICE_PREFIX + EXAMPLE
    with session_scope() as db:
        db.get(Memory, forget).suppressed = True
    # The summary still mentions it, so it is not sent, and the owner is due now.
    assert about_you.for_voice(OWNER, prefs) == ""
    assert stored()["next"] <= now()
    assert len(build(model)) == 1
    assert model.calls[-1]["saved_facts_newest_first"] == ["The user prefers tea over coffee."]
    assert stored()["ids"] == [keep] and about_you.for_voice(OWNER, prefs)
    # With no memories left, the summary is cleared and nothing is generated.
    with session_scope() as db:
        db.get(Memory, keep).suppressed = True
    due_now()
    assert build(model) == [] and stored()["summary"] == "" and len(model.calls) == 2


def test_a_fact_forgotten_during_the_call_is_not_stored(on):
    fact("The user prefers tea over coffee.")
    forget = fact("The user is training for a marathon.")
    sweep()

    def racing(body):
        with session_scope() as db:
            db.get(Memory, forget).suppressed = True
        return about_you.Summary(summary=EXAMPLE)

    about_you.process(jobs()[0], model=racing)
    row = stored()
    assert row["summary"] == "" and row["next"] <= now()


def test_failures_keep_the_previous_summary_and_retry_later(on):
    fact("The user prefers tea over coffee.")
    build(Model())
    due_now(generated_at=now() - timedelta(days=8))

    def failing(body):
        raise about_you.NoSummary("timeout", "slow")

    build(failing)
    row = stored()
    assert row["summary"] == EXAMPLE and row["next"] > now() + timedelta(hours=5)
    with session_scope() as db:
        assert db.get(AboutYou, OWNER).last_error["kind"] == "timeout"


def test_summary_is_capped_and_owners_without_memories_get_nothing(on):
    model = Model("word " * 400)
    assert build(model) == []  # No memories: no owner to look at.
    fact("The user prefers tea over coffee.")
    build(model)
    assert len(stored()["summary"]) <= about_you.MAX_CHARS


def test_no_model_key_means_no_job(monkeypatch):
    monkeypatch.setattr(about_you, "enabled", lambda: False)
    fact("The user prefers tea over coffee.")
    assert sweep() == 0 and jobs() == []


# ---------------------------------------------------------------- privacy


def test_memory_off_clears_and_never_generates_or_sends(on):
    fact("The user prefers tea over coffee.")
    model = Model()
    build(model)
    settings(memory_learning=False)
    with session_scope() as db:
        prefs = preferences(db, OWNER)
    assert about_you.for_voice(OWNER, prefs) == ""
    due_now()
    assert build(model) == [] and stored()["summary"] == ""
    with session_scope() as db:
        job = about_you.schedule(db, OWNER)
    about_you.process(job, model=model)
    assert len(model.calls) == 1
    with session_scope() as db:
        assert db.get(Job, job).result == {"stored": False, "reason": "disabled"}
        assert about_you.public(db, OWNER, preferences(db, OWNER))["enabled"] is False


def test_memory_during_voice_off_keeps_the_summary_but_sends_nothing(on):
    fact("The user prefers tea over coffee.")
    build(Model())
    settings(live_memory_updates=False)
    with session_scope() as db:
        prefs = preferences(db, OWNER)
        page = about_you.public(db, OWNER, prefs)
    assert about_you.for_voice(OWNER, prefs) == ""
    assert page["summary"] == EXAMPLE and page["in_voice"] is False


def test_shared_workspace_never_gets_a_summary(on):
    from jarvis.models import SharedWorkspace, UserAccount

    with session_scope() as db:
        db.add(UserAccount(id="member", name="Member"))
        db.flush()
        db.add(SharedWorkspace(id="team", name="Team", kind="space", creator_id="member"))
    fact("A fact somehow filed under the workspace.", owner="team")
    model = Model()
    sweep()
    assert jobs("team") == [] and model.calls == []
    with session_scope() as db:
        row = db.get(AboutYou, "team")  # The sweep looked and kept it empty.
        assert row.summary == ""
        row.summary = "Should never be sent."
    with session_scope() as db:
        prefs = preferences(db, "team")
    assert prefs["shared_workspace"] and about_you.for_voice("team", prefs) == ""


# ---------------------------------------------------------------- voice session start


@pytest.fixture
def live(controller, monkeypatch):
    """A Live controller whose provider call records the session and is then rejected (no media starts)."""
    from jarvis import live_http

    sent = []

    class Client:
        async def post(self, url, headers=None, json=None):
            sent.append(json["session"])
            return httpx.Response(500, json={})

    monkeypatch.setattr(live_http, "client", lambda: Client())
    monkeypatch.setattr("jarvis.access.assert_current", lambda *args: None)
    return controller, sent


async def test_voice_session_instructions_carry_the_summary_and_no_memory_json(live, monkeypatch):
    from jarvis import live_voice

    c, sent = live
    fact("The user prefers tea over coffee.")
    build(Model())
    retrieval = AsyncMock(side_effect=AssertionError("no retrieval at session start"))
    monkeypatch.setattr(live_voice, "semantic_search", retrieval)
    monkeypatch.setattr(about_you, "ask", lambda *a: pytest.fail("never generated during connect"))
    calls_before = len(jobs())
    with pytest.raises(DomainError):
        await c.connect("synthetic")
    instructions = sent[0]["instructions"]
    assert instructions.endswith("\n" + about_you.VOICE_PREFIX + EXAMPLE)
    for leaked in ('"source_id"', '"attribution"', '"saved_at"', "JSON DATA", "The user prefers tea"):
        assert leaked not in instructions
    assert c.context_sent == set() and len(jobs()) == calls_before
    retrieval.assert_not_awaited()
    await c.close()


async def test_voice_session_without_a_summary_sends_no_memory(live, monkeypatch):
    c, sent = live
    fact("The user prefers tea over coffee.")  # Saved, but no summary has been written yet.
    monkeypatch.setattr(about_you, "ask", lambda *a: pytest.fail("never generated during connect"))
    with pytest.raises(DomainError):
        await c.connect("synthetic")
    instructions = sent[0]["instructions"]
    assert "About you" not in instructions and "tea" not in instructions
    assert jobs() == [] and stored() is None
    await c.close()


# ---------------------------------------------------------------- the Memory page


def test_memory_page_shows_the_summary_and_when_it_was_written(client, on):
    fact("The user prefers tea over coffee.")
    assert client.get("/api/v1/memory").json()["about"] == {"enabled": True, "in_voice": True, "summary": "",
                                                            "updated_at": None}
    build(Model())
    about = client.get("/api/v1/memory").json()["about"]
    assert about["summary"] == EXAMPLE and about["updated_at"] and about["in_voice"] is True
