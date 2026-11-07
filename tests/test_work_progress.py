"""Mid-run progress notes: the backend records its latest step, Activity shows it and voice relays it."""

import time
from unittest.mock import AsyncMock

import pytest
from jarvis import budget, live_voice, work_runner
from jarvis.agent_work import finish
from jarvis.db import session_scope
from jarvis.domain import preferences
from jarvis.live_voice import LiveController
from jarvis.models import AgentWork, Conversation, Job
from jarvis.work_crypto import unseal
from test_agent_work import action, recovery_settings, responder, response  # noqa: F401 - fixtures and helpers
from test_live_voice import queued_voice_work, quiet_since_last_input, spoken


def card(client, work):
    return client.get("/api/v1/work/" + work["id"]).json()


@pytest.mark.parametrize("tool,target,expected", [
    ("task_list", {}, ("looking_up", "looking that up")),
    ("record_search", {"query": "x"}, ("looking_up", "looking that up")),
    ("note_read", {"note_id": "n"}, ("looking_up", "looking that up")),
    ("record_browse", {}, ("looking_up", "looking that up")),
    ("session_search", {}, ("looking_up", "looking that up")),
    ("navigation_find", {}, ("looking_up", "looking that up")),
    ("task_create", {"title": "Call Alex"}, ("saving", "saving Call Alex")),
    ("project_create", {"name": "Launch"}, ("saving", "saving Launch")),
    ("note_create", {}, ("saving", "saving that")),
    ("task_update", {"task_id": "t", "title": "Call Alex"}, ("updating", "updating Call Alex")),
    ("task_complete", {"task_id": "t"}, ("updating", "updating that")),
    ("task_reopen", {"task_id": "t"}, ("updating", "updating that")),
    ("task_batch", {}, ("updating", "updating that")),
    ("structure_apply", {}, ("updating", "updating that")),
    ("work_followup", {"request_id": "r"}, ("checking", "checking the earlier request")),
    ("tools_load", {}, None),
    ("ui_show", {}, None),
    ("work_needs_input", {}, None),
    ("work_answer", {}, None),
    ("voice_end", {}, None),
    ("notification_read", {}, None),
    ("memory_forget", {}, None),
    ("brand_new_tool", {"title": "X"}, None),
])
def test_note_templates_follow_the_tool_and_never_claim_an_outcome(tool, target, expected):
    assert work_runner.progress_note(tool, target) == expected
    if expected:
        assert not any(word in expected[1] for word in ("done", "added", "saved", "finished"))


def test_titles_are_trimmed_to_sixty_characters():
    stage, note = work_runner.progress_note("task_create", {"title": "  Call   " + "x" * 100})
    assert stage == "saving" and note == "saving " + ("Call " + "x" * 100)[:60]


@pytest.mark.asyncio
async def test_each_tool_step_sets_progress_and_finish_clears_it(client, monkeypatch):
    work = action(client, "Find Alex's number and add a call")
    seen = []

    def after(step):
        def round_():
            seen.append(card(client, work)["progress"])
            return step
        return round_

    responder(monkeypatch, [
        response([("task_list", {})]),
        after(response([("task_create", {"title": "Call Alex"})])),
        after(response(message="Added Call Alex.")),
    ])
    await work_runner.run(work["id"])
    assert [(p["stage"], p["note"], p["ok"]) for p in seen] == [
        ("looking_up", "looking that up", True), ("saving", "saving Call Alex", True)]
    assert seen[0]["seq"] < seen[1]["seq"]
    done = card(client, work)
    assert done["status"] == "succeeded" and done["progress"] is None
    with session_scope() as db:
        assert "progress" not in db.get(AgentWork, work["id"]).result


@pytest.mark.asyncio
async def test_a_failed_step_is_followed_by_a_retry_note_only_when_the_model_continues(client, monkeypatch):
    work = action(client, "Move the dentist")
    seen = []

    def check(step):
        def round_():
            seen.append(card(client, work)["progress"])
            return step
        return round_

    responder(monkeypatch, [
        response([("task_update", {"task_id": "missing", "expected_revision": 1, "title": "Dentist"})]),
        check(response([("task_list", {})])),
        check(response(message="I couldn't find it.")),
    ])
    await work_runner.run(work["id"])
    assert (seen[0]["stage"], seen[0]["note"]) == ("retrying", work_runner.RETRY_NOTE)
    # The next step clears the retry: one failure gives one retry note.
    assert seen[1]["stage"] == "looking_up"
    assert card(client, work)["progress"] is None


@pytest.mark.asyncio
async def test_parked_followup_says_it_is_waiting_and_clears_it_on_resume(client, monkeypatch):
    first = action(client, "Add a task to call Alex")
    follow = action(client, "Actually make that call tomorrow")
    responder(monkeypatch, [response([("work_followup", {"request_id": first["id"]})])])
    await work_runner.run(follow["id"])
    waiting = card(client, follow)
    assert waiting["status"] == "queued" and waiting["waiting"]
    assert (waiting["progress"]["stage"], waiting["progress"]["note"]) == ("waiting", work_runner.WAITING_NOTE)
    responder(monkeypatch, [response([("task_create", {"title": "Call Alex"})]), response(message="Added.")])
    await work_runner.run(first["id"])
    seen = []

    def resumed():
        seen.append(card(client, follow)["progress"])
        return response(message="It already moved.")

    responder(monkeypatch, [resumed])
    await work_runner.run(follow["id"])
    assert seen[0]["stage"] == "checking" and seen[0]["seq"] > waiting["progress"]["seq"]
    with session_scope() as db:
        assert db.get(Job, follow["id"]).status == "succeeded"
        assert "progress" not in db.get(AgentWork, follow["id"]).result


# Voice: LiveController.report_work relays notes with gating.

@pytest.fixture
def live():
    """A voice session with its inbox, as in test_live_voice (whose fixture this module cannot share)."""
    from jarvis.accounts import ensure_account
    from jarvis.work_intake import open_voice
    with session_scope() as db:
        conv = Conversation(owner_id="davin", device_id="device", private=True, learning=False)
        db.add(conv)
        db.flush()
        c = LiveController("davin", "device", conv.id, None, preferences(db, "davin"))
        budget.reserve(db, "davin", c.id, 0.1, c.model)
        ensure_account(db, "davin")
        open_voice(db, c.id, "davin", "davin", "device", conv.id)
    c.send = AsyncMock()
    return c


def progress(work, seq, note, stage="looking_up", ok=True):
    with session_scope() as db:
        row = db.get(AgentWork, work)
        row.result = {**row.result, "progress": {"seq": seq, "stage": stage, "note": note, "ok": ok}}


def busy_for(c, work, seconds):
    c.busy_since[work] = time.monotonic() - seconds


def notes(c):
    return [text for text in spoken(c) if text.startswith(live_voice.PROGRESS_PREFIX)]


def allow_next(c):
    c.last_progress_at -= live_voice.PROGRESS_SPACING_SECONDS + 1


async def test_voice_waits_four_seconds_then_relays_a_new_note_once(live):
    c = live
    work = queued_voice_work(c)
    quiet_since_last_input(c)
    progress(work, 1, "looking that up")
    await c.report_work()
    assert notes(c) == []  # Busy for less than four seconds.
    busy_for(c, work, live_voice.PROGRESS_MIN_BUSY_SECONDS + 0.5)
    await c.report_work()
    assert notes(c) == ["Progress, not a result: the backend is looking that up. "
                        "Say so in a few words, once, without guessing the outcome."]
    allow_next(c)
    await c.report_work()
    progress(work, 2, "looking that up")  # A newer step with the same words is not repeated.
    await c.report_work()
    assert len(notes(c)) == 1
    assert len(notes(c)[0].encode()) < 420


async def test_voice_spaces_notes_eight_seconds_and_stops_at_three_per_request(live):
    c = live
    work = queued_voice_work(c)
    quiet_since_last_input(c)
    busy_for(c, work, 5)
    progress(work, 1, "looking that up")
    await c.report_work()
    progress(work, 2, "saving Call Alex")
    await c.report_work()
    assert len(notes(c)) == 1  # Within eight seconds of the last note.
    allow_next(c)
    await c.report_work()
    for seq, note in [(3, "updating Call Alex"), (4, "checking the earlier request")]:
        progress(work, seq, note)
        allow_next(c)
        await c.report_work()
    assert [n.split("backend is ")[1].split(".")[0] for n in notes(c)] == [
        "looking that up", "saving Call Alex", "updating Call Alex"]
    # The generic twelve-second note is suppressed, not stacked on top.
    busy_for(c, work, live_voice.SLOW_WORK_SECONDS + 1)
    allow_next(c)
    await c.report_work()
    assert len(spoken(c)) == 3


async def test_voice_holds_notes_while_the_owner_talks_or_a_question_is_open(live):
    c = live
    work = queued_voice_work(c)
    quiet_since_last_input(c)
    busy_for(c, work, 5)
    progress(work, 1, "saving Call Alex", "saving")
    c.input_speaking = True
    await c.report_work()
    c.input_speaking = False
    asking = queued_voice_work(c)
    with session_scope() as db:
        finish(db, db.get(AgentWork, asking), "needs_input", "Which Alex?")
    await c.report_work()
    assert c.pending_question and "Which Alex?" in spoken(c)[-1]
    await c.report_work()
    assert notes(c) == []  # The open question owns the next turn.
    with session_scope() as db:
        finish(db, db.get(AgentWork, asking), "cancelled", "Cancelled.")
    quiet_since_last_input(c, 1)
    await c.report_work()
    assert notes(c) == []  # Spoke within 2.5 s.
    quiet_since_last_input(c)
    await c.report_work()
    assert notes(c) == [] and "Cancelled." in spoken(c)[-1]  # The result goes first.
    await c.report_work()
    assert notes(c) == [live_voice.PROGRESS_PREFIX + "saving Call Alex" + live_voice.PROGRESS_SUFFIX]
    progress(work, 2, "saving Call Alex", "saving", ok=False)
    allow_next(c)
    await c.report_work()
    assert len(notes(c)) == 1  # A failed step is not relayed; the retry note follows it.
    progress(work, 3, work_runner.RETRY_NOTE, "retrying")
    await c.report_work()
    assert notes(c)[-1] == live_voice.PROGRESS_PREFIX + work_runner.RETRY_NOTE + live_voice.PROGRESS_SUFFIX


async def test_results_win_over_progress_in_the_same_tick(live):
    c = live
    done, running = queued_voice_work(c), queued_voice_work(c)
    quiet_since_last_input(c)
    busy_for(c, running, 5)
    progress(running, 1, "looking that up")
    with session_scope() as db:
        finish(db, db.get(AgentWork, done), "succeeded", "Added milk.", tool_calls=1)
    await c.report_work()
    assert len(spoken(c)) == 1 and "Added milk." in spoken(c)[0]
    await c.report_work()
    assert notes(c) == [live_voice.PROGRESS_PREFIX + "looking that up" + live_voice.PROGRESS_SUFFIX]


async def test_two_concurrent_requests_both_get_notes_least_recently_heard_first(live):
    c = live
    first, second = queued_voice_work(c), queued_voice_work(c)
    quiet_since_last_input(c)
    busy_for(c, first, 5)
    busy_for(c, second, 5)
    progress(first, 1, "saving Call Alex", "saving")
    progress(second, 1, "saving Buy milk", "saving")
    await c.report_work()
    allow_next(c)
    progress(first, 2, "updating Call Alex", "updating")
    await c.report_work()
    allow_next(c)
    await c.report_work()
    said = [n.split("backend is ")[1].split(".")[0] for n in notes(c)]
    assert said == ["saving Call Alex", "saving Buy milk", "updating Call Alex"]


async def test_generic_still_working_note_counts_toward_spacing(live):
    c = live
    work = queued_voice_work(c)
    quiet_since_last_input(c)
    busy_for(c, work, live_voice.SLOW_WORK_SECONDS + 1)
    await c.report_work()
    assert "still working on an earlier request" in spoken(c)[0]
    progress(work, 1, "looking that up")
    await c.report_work()
    assert len(spoken(c)) == 1  # Eight seconds have not passed.
    allow_next(c)
    await c.report_work()
    assert spoken(c)[-1] == live_voice.PROGRESS_PREFIX + "looking that up" + live_voice.PROGRESS_SUFFIX
    with session_scope() as db:
        assert unseal(db.get(AgentWork, work).input_ciphertext)["message"] == "Synthetic task"
