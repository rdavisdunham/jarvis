import asyncio
import json
from datetime import timedelta
from uuid import uuid4

import pytest
from jarvis import agent_work, work_continuation, work_coordination, work_runner
from jarvis.db import session_scope
from jarvis.domain import DomainError
from jarvis.models import AgentWork, Command, Job, Task, now
from sqlalchemy import select
from test_agent_work import accept, action, recovery_settings, responder, response  # noqa: F401


def reply(client, original, message):
    return client.post("/api/v1/work", json={
        "conversation_id": original["conversation_id"], "turn_id": str(uuid4()), "message": message,
    }).json()


def pending(client, message="Schedule the review", question="What time and deadline or block?"):
    item = action(client, message)
    with session_scope() as db:
        row = db.get(AgentWork, item["id"])
        agent_work.finish(db, row, "needs_input", question)
    return client.get("/api/v1/work/" + item["id"]).json()


def answer_args(item):
    return {"request_id": item["clarification"]["request_id"], "clarification_id": item["clarification"]["id"]}


def detached(identity):
    with session_scope() as db:
        row = db.get(AgentWork, identity)
        db.expunge(row)
        return row


@pytest.mark.asyncio
async def test_spoken_answer_completes_original_card_without_repeating_saved_action(client, monkeypatch):
    original = action(client, "Add Review docs and schedule its deadline tomorrow")
    responder(monkeypatch, [response([("task_create", {"title": "Review docs"})]),
        response([("work_needs_input", {"question": "What time should its deadline be?"})])])
    await work_runner.run(original["id"])
    original = client.get("/api/v1/work/" + original["id"]).json()
    answered = reply(client, original, "9 a.m., just a deadline")
    responder(monkeypatch, [response([("work_answer", answer_args(original))])])
    await work_runner.run(answered["id"])
    queued = client.get("/api/v1/work").json()["items"]
    assert len(queued) == 1 and queued[0]["id"] == original["id"]
    assert queued[0]["status"] == "queued" and len(queued[0]["actions"]) == 1
    with session_scope() as db:
        assert agent_work.eligible(db, db.get(Job, answered["id"]))
        task = db.scalar(select(Task))
        task_id, revision = task.id, task.revision
        assert db.get(Job, original["id"]).status == "continued"

    async def model(agent, messages, definitions, **kwargs):
        text = json.dumps(messages)
        assert "What time should its deadline be?" in text
        assert "9 a.m., just a deadline" in text and "Add Review docs" in text
        assert task_id in text and "Verified previous actions" in text
        if any(m.get("name") == "task_update" for m in messages):
            return response(message="Deadline set.")
        return response([("task_update", {"task_id": task_id, "expected_revision": revision,
            "due_date": "2026-09-17", "due_time": "09:00"})])

    monkeypatch.setattr(work_runner, "request_model", model)
    await work_runner.run(answered["id"])
    # Re-delivery of either attempt cannot replay an effect.
    await work_runner.run(answered["id"])
    await work_runner.run(original["id"])
    cards = client.get("/api/v1/work").json()["items"]
    assert len(cards) == 1
    card = cards[0]
    assert card["id"] == original["id"] and card["status"] == "succeeded"
    assert not card["children"] and not card["clarification"]
    assert len(card["actions"]) == 2
    assert card["clarification_history"] == [{"question": "What time should its deadline be?", "answer": "9 a.m., just a deadline"}]
    with session_scope() as db:
        assert len(list(db.scalars(select(Task)))) == 1
        assert db.get(Task, task_id).due_date.isoformat() == "2026-09-17"
    edit = next(a for a in card["actions"] if a["operation"] == "updated")
    assert client.post("/api/v1/work/actions/" + edit["id"] + "/revert", json={"command_id": str(uuid4())}).status_code == 200
    with session_scope() as db:
        assert db.get(Task, task_id).due_date is None


@pytest.mark.asyncio
async def test_incomplete_answers_stay_on_one_card_and_keep_question_history(client, monkeypatch):
    original = pending(client)
    first = reply(client, original, "Tomorrow")
    responder(monkeypatch, [response([("work_answer", answer_args(original))])])
    await work_runner.run(first["id"])
    responder(monkeypatch, [response([("work_needs_input", {"question": "And what time?"})])])
    await work_runner.run(first["id"])
    waiting = client.get("/api/v1/work/" + original["id"]).json()
    assert waiting["status"] == "needs_input" and waiting["message"] == "And what time?"
    assert waiting["clarification"]["request_id"] == first["id"]
    second = reply(client, original, "9 a.m.")
    responder(monkeypatch, [response([("work_answer", answer_args(waiting))])])
    await work_runner.run(second["id"])
    responder(monkeypatch, [response(message="The remaining work is done.")])
    await work_runner.run(second["id"])
    cards = client.get("/api/v1/work").json()["items"]
    assert len(cards) == 1 and cards[0]["status"] == "succeeded"
    assert [a["answer"] for a in cards[0]["clarification_history"]] == ["Tomorrow", "9 a.m."]
    with session_scope() as db:
        assert len(db.get(AgentWork, original["id"]).result["continuation_ids"]) == 2


def test_answer_handoff_is_idempotent_and_rejects_late_answer_to_consumed_question(client):
    original = pending(client)
    first, second = reply(client, original, "9 a.m."), reply(client, original, "Actually 10")
    args = answer_args(original)
    with pytest.raises(work_continuation.WorkContinued):
        work_continuation.answer(detached(first["id"]), args)
    assert work_continuation.answer(detached(first["id"]), args)["status"] == "continued"
    with pytest.raises(DomainError) as stale:
        work_continuation.answer(detached(second["id"]), args)
    assert stale.value.code == "STALE_CLARIFICATION"
    with session_scope() as db:
        assert db.get(AgentWork, original["id"]).result["continuation_ids"] == [first["id"]]
    # A later correction waits for the actual continuation, not the old paused attempt.
    with pytest.raises(work_coordination.WorkDeferred) as waiting:
        work_coordination.followup(detached(second["id"]), {"request_id": original["id"]})
    assert waiting.value.dependencies == [first["id"]]


@pytest.mark.asyncio
async def test_two_concurrent_answers_cannot_both_claim_one_question(client):
    original = pending(client)
    replies = [reply(client, original, text) for text in ("9 a.m.", "10 a.m.")]
    def adopt(item):
        try:
            work_continuation.answer(detached(item["id"]), answer_args(original))
        except work_continuation.WorkContinued:
            return "adopted"
        except DomainError as exc:
            return exc.code
    result = await asyncio.gather(*[asyncio.to_thread(adopt, item) for item in replies])
    assert sorted(result) == ["STALE_CLARIFICATION", "adopted"]


def test_pending_followup_returns_exact_question_and_independent_work_stays_separate(client):
    original = pending(client)
    milk = reply(client, original, "Also add Milk")
    answer = reply(client, original, "9 a.m.")
    outcome = work_coordination.followup(detached(answer["id"]), {"request_id": original["id"]})
    assert outcome["clarification"] == original["clarification"] and "work_answer" in outcome["next_step"]
    with pytest.raises(work_continuation.WorkContinued):
        work_continuation.answer(detached(answer["id"]), answer_args(original))
    with session_scope() as db:
        assert not db.get(AgentWork, milk["id"]).parent_id
        assert agent_work.eligible(db, db.get(Job, milk["id"]))


def test_wrong_question_account_or_cancelled_request_cannot_be_adopted(client):
    from test_accounts import client_for
    original = pending(client)
    answer = reply(client, original, "9 a.m.")
    with pytest.raises(DomainError):
        work_continuation.answer(detached(answer["id"]), {**answer_args(original), "clarification_id": str(uuid4())})
    other = action(client_for("other"), "9 a.m.")
    with pytest.raises(DomainError):
        work_continuation.answer(detached(other["id"]), answer_args(original))
    client.post("/api/v1/work/" + original["id"] + "/cancel", json={})
    with pytest.raises(DomainError):
        work_continuation.answer(detached(answer["id"]), answer_args(original))


@pytest.mark.asyncio
async def test_manual_revision_preserves_question_and_resumes_latest_attempt(client, monkeypatch):
    original = pending(client)
    answer = reply(client, original, "Tomorrow")
    responder(monkeypatch, [response([("work_answer", answer_args(original))])])
    await work_runner.run(answer["id"])
    responder(monkeypatch, [response([("work_needs_input", {"question": "What time?"})])])
    await work_runner.run(answer["id"])
    card = client.get("/api/v1/work/" + original["id"]).json()
    updated = client.post("/api/v1/work/" + original["id"] + "/revise", json={
        "message": "9 a.m.", "expected_revision": card["revision"], "continue_work": True})
    assert updated.status_code == 200, updated.text
    async def model(agent, messages, definitions, **kwargs):
        text = json.dumps(messages)
        assert "What time?" in text and "9 a.m." in text
        return response(message="Done.")
    monkeypatch.setattr(work_runner, "request_model", model)
    await work_runner.run(answer["id"])
    assert client.get("/api/v1/work/" + original["id"]).json()["status"] == "succeeded"


@pytest.mark.asyncio
async def test_clear_history_preserves_saved_records_and_is_account_scoped(client, monkeypatch):
    from test_accounts import client_for
    saved = action(client)
    responder(monkeypatch, [response([("task_create", {"title": "Keep me"})]), response(message="Saved.")])
    await work_runner.run(saved["id"])
    waiting = pending(client)
    failed = action(client)
    with session_scope() as db:
        agent_work.finish(db, db.get(AgentWork, failed["id"]), "failed", "Old error")
    other_client = client_for("other")
    other = action(other_client)
    cutoff = now()
    fresh = action(client)
    result = client.post("/api/v1/work/clear", json={"before": cutoff.isoformat()})
    assert result.status_code == 200 and result.json()["cleared"] == 3
    assert [i["id"] for i in client.get("/api/v1/work").json()["items"]] == [fresh["id"]]
    assert other_client.get("/api/v1/work").json()["items"][0]["id"] == other["id"]
    with session_scope() as db:
        assert len(list(db.scalars(select(Task)))) == 1
        assert len(list(db.scalars(select(Command)))) == 1
        assert db.get(Job, waiting["id"]).status == "cancelled"
        assert not agent_work.eligible(db, db.get(Job, waiting["id"]))
        assert work_coordination.recent(db, db.get(AgentWork, fresh["id"])) == []
    assert client.post("/api/v1/work/clear", json={"before": cutoff.isoformat()}).json()["cleared"] == 0
    card = client.get("/api/v1/work/" + waiting["id"]).json()
    assert client.post("/api/v1/work/" + waiting["id"] + "/revise", json={"message": "retry", "expected_revision": card["revision"]}).status_code == 409
    assert client.post("/api/v1/work/clear", json={"before": (now() + timedelta(days=1)).isoformat()}).status_code == 400


def test_linked_waiting_request_cannot_be_bypassed_with_a_standalone_edit(client):
    original = pending(client)
    follow = reply(client, original, "9 a.m.")
    row = detached(follow["id"])
    work_coordination.followup(row, {"request_id": original["id"]})
    with pytest.raises(DomainError) as blocked:
        work_coordination.reserve(row, "task_update", {"task_id": str(uuid4())}, 1)
    assert blocked.value.code == "CLARIFICATION_REQUIRED"
    with pytest.raises(work_continuation.WorkContinued):
        work_continuation.answer(row, answer_args(original))


def test_legacy_pending_questions_receive_stable_ids_without_a_migration(client):
    original = pending(client)
    with session_scope() as db:
        row = db.get(AgentWork, original["id"])
        row.result = {"message": "What time?"}
    card = client.get("/api/v1/work/" + original["id"]).json()
    follow = reply(client, card, "9 a.m.")
    with pytest.raises(work_continuation.WorkContinued):
        work_continuation.answer(detached(follow["id"]), answer_args(card))


@pytest.mark.asyncio
async def test_clear_during_model_call_stops_late_effect_and_card_cannot_reappear(client, monkeypatch):
    original = action(client)
    entered, released = asyncio.Event(), asyncio.Event()
    async def model(*args, **kwargs):
        entered.set()
        await released.wait()
        return response([("task_create", {"title": "Must not be saved"})])
    monkeypatch.setattr(work_runner, "request_model", model)
    execution = asyncio.create_task(work_runner.run(original["id"]))
    await asyncio.wait_for(entered.wait(), 3)
    assert client.post("/api/v1/work/clear", json={}).status_code == 200
    released.set()
    await execution
    assert client.get("/api/v1/work").json()["items"] == []
    with session_scope() as db:
        assert not list(db.scalars(select(Task)))
        assert db.get(Job, original["id"]).status == "cancelled"
