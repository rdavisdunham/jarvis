"""Synthetic DB regressions for the conversational Quick Sort state machine."""
from uuid import uuid4
import pytest
from sqlalchemy import select
from jarvis import agent_work, quick_sort, structure, work_continuation, work_runner
from jarvis.db import session_scope
from jarvis.domain import DomainError, execute
from jarvis.models import AgentWork, Job, Task, Note
from jarvis.structure_models import QuickSortSession, StructureRecord, RoutingObservation, RoutingPattern, RoutingReview
from test_agent_work import action, recovery_settings, responder, response  # noqa: F401
from test_work_continuation import reply as new_turn, detached

OWNER = "davin"


def seed(kind="task", title="File this", **kwargs):
    with session_scope() as db:
        schema = structure.ensure(db, OWNER)
        return execute(db, OWNER, "seed:" + str(uuid4()), "record.create",
                       {"type_id": kind, "title": title, "schema_revision": schema.revision, **kwargs})["data"]


def command(work, tool, args, *, index=None, finish=True):
    with session_scope() as db:
        row = db.get(AgentWork, work["id"])
        with agent_work.principal_for(row):
            result = execute(db, OWNER, row.id + ":" + (index or str(uuid4())[:8]), tool, args)
            data = result["data"]
            if finish:
                agent_work.finish(db, row, "needs_input" if data.get("question") else "succeeded",
                                  data.get("question") or data.get("message", "Done"))
            return data


def begin(client, **kwargs):
    work = action(client, "Help me sort")
    result = command(work, "quicksort.start", kwargs)
    return work, result


def answer(client, original, text):
    card = client.get("/api/v1/work/" + original["id"]).json()
    following = new_turn(client, card, text)
    question = card["clarification"]
    with pytest.raises(work_continuation.WorkContinued):
        work_continuation.answer(detached(following["id"]), {
            "request_id": question["request_id"], "clarification_id": question["id"]})
    return following


def choose(client, work, sorting, text, **kwargs):
    following = answer(client, work, text)
    return following, command(following, "quicksort.reply",
                              {"session_id": sorting["session_id"], **kwargs})


def observations():
    with session_scope() as db:
        return [(r.source_key, r.suppressed, r.evidence) for r in db.scalars(select(RoutingObservation))]


def test_snapshot_only_active_unfiled_leaf_tasks_and_bound(client):
    parent = seed("project", "Container")
    seed(title="Inside", parent_id=parent["id"])
    first = seed(title="First")
    seed(title="Second")
    seed("note", "A note")
    done = seed(title="Done")
    with session_scope() as db:
        db.get(Task, done["task_id"]).status = "completed"
    work, sorting = begin(client, limit=1)
    assert sorting["current"]["id"] == first["id"]
    assert sorting["progress"] == {"total": 1, "reviewed": 0, "filed": 0, "skipped": 0}
    seed(title="Arrived later")
    with session_scope() as db:
        assert db.get(QuickSortSession, sorting["session_id"]).state["items"] == [first["id"]]
    assert client.get("/api/v1/work/" + work["id"]).json()["actions"] == []


def test_notes_selected_and_home_scopes(client):
    home = seed("client", "ABC")
    note = seed("note", "Useful note", parent_id=home["id"])
    seed(title="Unfiled task")
    work, sorting = begin(client, scope="selected", kind="notes", record_ids=[note["id"]])
    assert sorting["current"]["id"] == note["id"] and sorting["progress"]["total"] == 1
    stop = new_turn(client, work, "stop")
    command(stop, "quicksort.control", {"session_id": sorting["session_id"], "action": "stop"})
    _, scoped = begin(client, scope="home", home_id=home["id"], kind="notes")
    assert scoped["current"]["id"] == note["id"]


def test_exact_named_home_ancestry_and_home_only_evidence(client):
    client_home = seed("client", "ABC")
    home = seed("project", "Transcript Intelligence", parent_id=client_home["id"])
    task = seed(title="Finish the central docs", values={"priority": 2})
    work, sorting = begin(client)
    _, saved = choose(client, work, sorting, "Put it in Transcript Intelligence")
    assert saved["status"] == "completed"
    with session_scope() as db:
        assert db.get(StructureRecord, task["id"]).parent_id == home["id"]
        item = db.scalar(select(RoutingObservation))
        assert item.evidence["assignment"] == {"parent_id": home["id"], "values": {}}
        assert item.evidence["provenance"]["kind"] == "quick_sort"
    assert "ABC / Transcript Intelligence" in saved["message"]
    cards = client.get("/api/v1/work/" + work["id"]).json()["actions"]
    assert len(cards) == 1 and cards[0]["kind"] == "record"
    assert cards[0]["command_id"].endswith(":home") and cards[0]["can_revert"]


def test_duplicate_names_require_specific_candidate_confirmation(client):
    a, b = seed("client", "ABC"), seed("client", "Other")
    homes = [seed("project", "Website", parent_id=p["id"]) for p in (a, b)]
    task = seed()
    work, sorting = begin(client)
    follow, proposed = choose(client, work, sorting, "Website")
    assert len(proposed["candidates"]) == 2 and not observations()
    _, saved = choose(client, follow, proposed, "second one")
    with session_scope() as db:
        assert db.get(StructureRecord, task["id"]).parent_id == proposed["candidates"][1]["id"]
    assert saved["status"] == "completed"


def test_exact_home_name_with_conjunction_is_not_a_compound_guess(client):
    home = seed("client", "Research and Development")
    task = seed()
    work, sorting = begin(client)
    choose(client, work, sorting, "Research and Development")
    with session_scope() as db:
        assert db.get(StructureRecord, task["id"]).parent_id == home["id"]


def test_fuzzy_home_proposal_requires_later_yes_and_cannot_change_target(client):
    home, alternative = seed("client", "ABC"), seed("client", "Personal")
    task = seed()
    work, sorting = begin(client)
    follow, proposed = choose(client, work, sorting, "the pest control company", home_id=home["id"])
    assert proposed["question"].startswith('File "') and not observations()
    _, saved = choose(client, follow, proposed, "yes", home_id=alternative["id"])
    with session_scope() as db:
        assert db.get(StructureRecord, task["id"]).parent_id == home["id"]
    assert saved["status"] == "completed"


@pytest.mark.parametrize("text", ["Not ABC", "ABC or Personal", "ABC and Personal", "I am not sure"])
def test_ambiguous_negated_partial_answers_do_not_file(client, text):
    home = seed("client", "ABC")
    seed()
    work, sorting = begin(client)
    _, result = choose(client, work, sorting, text, home_id=home["id"] if text != "I am not sure" else None)
    assert result["progress"]["filed"] == 0 and not observations()


def test_unrelated_turn_cannot_be_adopted_or_become_human_evidence(client):
    seed("client", "ABC")
    seed()
    work, sorting = begin(client)
    unrelated = new_turn(client, work, "add a task to call someone at ABC")
    card = client.get("/api/v1/work/" + work["id"]).json()
    with pytest.raises(DomainError) as error:
        work_continuation.answer(detached(unrelated["id"]), {
            "request_id": card["clarification"]["request_id"],
            "clarification_id": card["clarification"]["id"]})
    assert error.value.code == "UNRELATED_SORT_ANSWER"
    with session_scope() as db:
        assert db.get(Job, work["id"]).status == "needs_input"
        assert not db.get(AgentWork, unrelated["id"]).parent_id
    with pytest.raises(DomainError) as missing:
        command(unrelated, "quicksort.reply", {"session_id": sorting["session_id"]}, finish=False)
    assert missing.value.code == "SORT_ANSWER_REQUIRED" and not observations()


def test_one_captured_answer_cannot_file_two_items_and_retry_is_idempotent(client):
    home = seed("client", "ABC")
    seed(title="First")
    seed(title="Second")
    work, sorting = begin(client)
    follow = answer(client, work, "ABC")
    args = {"session_id": sorting["session_id"]}
    first = command(follow, "quicksort.reply", args, index="once")
    again = command(follow, "quicksort.reply", args, index="once", finish=False)
    assert again == first
    with pytest.raises(DomainError):
        command(follow, "quicksort.reply", args, index="twice", finish=False)
    assert len(observations()) == 1


@pytest.mark.parametrize("changed", ["item", "schema", "proposal"])
def test_stale_item_schema_or_proposed_home_requires_new_answer(client, changed):
    home = seed("client", "ABC")
    task = seed()
    work, sorting = begin(client)
    text = "ABC"
    if changed == "proposal":
        work, sorting = choose(client, work, sorting, "that company", home_id=home["id"])
        text = "yes"
    following = answer(client, work, text)
    with session_scope() as db:
        if changed == "schema":
            structure.ensure(db, OWNER).revision += 1
        else:
            record = db.get(StructureRecord, home["id"] if changed == "proposal" else task["id"])
            record.title += " changed"
            record.revision += 1
    result = command(following, "quicksort.reply", {"session_id": sorting["session_id"]})
    assert result["progress"]["filed"] == 0 and not observations()
    assert "changed" in result["message"]


def test_skip_pause_resume_stop_retire_pending_cards_and_replay_is_blocked(client):
    home = seed("client", "ABC")
    filed = seed(title="Already filed")
    seed(title="First")
    seed(title="Second")
    original, sorting = begin(client)
    original, sorting = choose(client, original, sorting, "ABC")
    skip = new_turn(client, original, "leave it unfiled")
    args = {"session_id": sorting["session_id"], "action": "skip"}
    result = command(skip, "quicksort.control", args)
    assert result["progress"]["skipped"] == 1
    with session_scope() as db:
        assert db.get(Job, original["id"]).status == "cancelled"
    with pytest.raises(DomainError) as used:
        command(skip, "quicksort.control", args, finish=False)
    assert used.value.code == "SORT_ANSWER_USED"
    pause = new_turn(client, skip, "pause")
    assert command(pause, "quicksort.control", {**args, "action": "pause"})["status"] == "paused"
    with session_scope() as db:
        assert db.get(Job, skip["id"]).status == "cancelled"
    resume = action(client, "resume quick sort")
    assert command(resume, "quicksort.control", {**args, "action": "resume"})["status"] == "active"
    stop = new_turn(client, resume, "stop sorting")
    assert command(stop, "quicksort.control", {**args, "action": "stop"})["status"] == "stopped"
    assert len(observations()) == 1
    with session_scope() as db:
        assert db.get(StructureRecord, filed["id"]).parent_id == home["id"]


def test_activity_continue_captures_answer_and_cancel_creates_no_evidence(client):
    home = seed("client", "ABC")
    task = seed()
    work, sorting = begin(client)
    card = client.get("/api/v1/work/" + work["id"]).json()
    result = client.post("/api/v1/work/" + work["id"] + "/revise", json={
        "message": "ABC", "continue_work": True, "expected_revision": card["revision"]})
    assert result.status_code == 200, result.text
    command(work, "quicksort.reply", {"session_id": sorting["session_id"]})
    assert len(observations()) == 1
    seed(title="Another")
    pending, pending_sort = begin(client)
    client.post("/api/v1/work/" + pending["id"] + "/cancel")
    with session_scope() as db:
        assert db.get(QuickSortSession, pending_sort["session_id"]).status == "paused"
    assert len(observations()) == 1


def test_user_revert_retracts_evidence_without_labeling_inverse_or_tags(client):
    home = seed("client", "ABC")
    note = seed("note", "Company lunch")
    with session_scope() as db:
        db.get(Note, note["note_id"]).tags = ["model-guessed"]
    work, sorting = begin(client, kind="notes")
    choose(client, work, sorting, "ABC")
    cards = client.get("/api/v1/work/" + work["id"]).json()["actions"]
    assert len(cards) == 1
    with session_scope() as db:
        observation = db.scalar(select(RoutingObservation))
        assert db.get(Note, note["note_id"]).tags == ["model-guessed"]
        assert observation.evidence["assignment"] == {"parent_id": home["id"], "values": {}}
        rule = RoutingPattern(owner_id=OWNER, fingerprint="test", condition={"type_id": "note", "phrase": "lunch"},
            assignment={"parent_id": home["id"]}, reason="test", schema_revision=1,
            evidence_ids=[observation.id], status="active")
        db.add(rule)
        db.flush()
        review = RoutingReview(owner_id=OWNER, period="undo-test", status="pending",
            questions=[{"id": rule.id, "question": "File lunch notes under ABC?"}])
        db.add(review)
    result = client.post("/api/v1/work/actions/" + cards[0]["id"] + "/revert",
                         json={"command_id": str(uuid4())})
    assert result.status_code == 200, result.text
    assert len(observations()) == 1 and observations()[0][1]
    with session_scope() as db:
        assert db.get(StructureRecord, note["id"]).parent_id is None
        assert db.scalar(select(RoutingPattern)).status == "paused"
        review = db.scalar(select(RoutingReview))
        assert review.status == "completed" and review.questions == []
        assert review.answers[-1]["answer"] == "evidence_withdrawn"


def test_learning_disabled_preserves_filing_and_no_agent_guess_evidence(client):
    home = seed("client", "ABC")
    task = seed()
    with session_scope() as db:
        execute(db, OWNER, "seed:settings", "settings.update", {"routing_learning": False})
    work, sorting = begin(client)
    choose(client, work, sorting, "ABC")
    assert not observations()
    with session_scope() as db:
        assert db.get(StructureRecord, task["id"]).parent_id == home["id"]


def test_sort_state_is_account_scoped_and_requires_human_execution_for_writes(client):
    seed()
    work, sorting = begin(client)
    with session_scope() as db:
        with pytest.raises(DomainError):
            quick_sort.state(db, "other", sorting["session_id"])
        with pytest.raises(DomainError) as denied:
            execute(db, OWNER, str(uuid4()), "quicksort.start", {})
        assert denied.value.code == "HUMAN_CONVERSATION_REQUIRED"


@pytest.mark.asyncio
async def test_runner_stops_after_server_question_then_files_on_captured_continuation(client, monkeypatch):
    home = seed("client", "ABC")
    task = seed()
    original = action(client, "Help me sort unfiled tasks")
    responder(monkeypatch, [
        response([("tools_load", {"groups": ["quick_sort"]})]),
        response([("quicksort_start", {}), ("task_create", {"title": "Must not run after a question"})])])
    await work_runner.run(original["id"])
    card = client.get("/api/v1/work/" + original["id"]).json()
    assert card["status"] == "needs_input" and not card["actions"]
    with session_scope() as db:
        sorting = db.scalar(select(QuickSortSession))
        sort_id = sorting.id
        assert sorting.state["question_id"] == card["clarification"]["id"]
        assert list(db.scalars(select(Task.title))) == ["File this"]
    following = new_turn(client, original, "ABC")
    responder(monkeypatch, [response([("work_answer", {
        "request_id": card["clarification"]["request_id"], "clarification_id": card["clarification"]["id"]})])])
    await work_runner.run(following["id"])
    responder(monkeypatch, [
        response([("tools_load", {"groups": ["quick_sort"]})]),
        response([("quicksort_reply", {"session_id": sort_id})]),
        response(message="Filed under ABC.")])
    await work_runner.run(following["id"])
    card = client.get("/api/v1/work/" + original["id"]).json()
    assert card["status"] == "succeeded" and len(card["actions"]) == 1
    assert card["actions"][0]["command_id"].endswith(":home")
    with session_scope() as db:
        assert db.get(StructureRecord, task["id"]).parent_id == home["id"]
    assert len(observations()) == 1


def test_empty_activity_continue_preserves_the_answerable_question(client):
    seed()
    work, sorting = begin(client)
    card = client.get("/api/v1/work/" + work["id"]).json()
    result = client.post("/api/v1/work/" + work["id"] + "/revise", json={
        "message": "", "continue_work": True, "expected_revision": card["revision"]})
    assert result.status_code == 409, result.text
    current = client.get("/api/v1/work/" + work["id"]).json()
    assert current["status"] == "needs_input" and current["revision"] == card["revision"]
    assert current["clarification"] == card["clarification"]


def test_mismatched_question_pauses_and_resume_retires_it(client):
    seed()
    work, sorting = begin(client)
    following = answer(client, work, "I do not know")
    with session_scope() as db:
        row = db.get(AgentWork, following["id"])
        agent_work.finish(db, row, "needs_input", "A model-authored question?")
        assert db.get(QuickSortSession, sorting["session_id"]).status == "paused"
    resume = new_turn(client, following, "resume quick sort")
    result = command(resume, "quicksort.control", {
        "session_id": sorting["session_id"], "action": "resume"})
    assert result["status"] == "active"
    with session_scope() as db:
        assert db.get(Job, following["id"]).status == "cancelled"


def test_skip_reply_cannot_be_replayed_as_another_control(client):
    seed(title="First")
    seed(title="Second")
    original, sorting = begin(client)
    following, result = choose(client, original, sorting, "skip")
    assert result["progress"]["skipped"] == 1
    with pytest.raises(DomainError) as used:
        command(following, "quicksort.control",
                {"session_id": sorting["session_id"], "action": "skip"}, finish=False)
    assert used.value.code == "SORT_ANSWER_USED"


def test_later_agent_home_correction_withdraws_old_example_without_forging_new_one(client):
    home, other = seed("client", "ABC"), seed("client", "Personal")
    task = seed()
    work, sorting = begin(client)
    choose(client, work, sorting, "ABC")
    with session_scope() as db:
        row = db.get(StructureRecord, task["id"])
        execute(db, OWNER, "agent-correction:1", "record.update", {
            "record_id": row.id, "schema_revision": structure.ensure(db, OWNER).revision,
            "expected_revision": row.revision, "parent_id": other["id"]})
    assert len(observations()) == 1 and observations()[0][1]


def test_rejected_proposal_does_not_file_or_train(client):
    home = seed("client", "ABC")
    task = seed()
    work, sorting = begin(client)
    following, proposal = choose(client, work, sorting, "that company", home_id=home["id"])
    _, result = choose(client, following, proposal, "no")
    assert result["progress"]["filed"] == 0 and not observations()
    with session_scope() as db:
        assert db.get(StructureRecord, task["id"]).parent_id is None


def test_illegal_home_proposal_does_not_change_item(client):
    home = seed("note", "Not a legal Home")
    task = seed()
    with session_scope() as db:
        import copy
        schema = structure.ensure(db, OWNER)
        design = copy.deepcopy(schema.definition)
        next(t for t in design["types"] if t["id"] == "task")["parent_types"] = ["client"]
        schema.definition = design
        schema.revision += 1
    work, sorting = begin(client, scope="selected", record_ids=[task["id"]])
    following = answer(client, work, "that place")
    with pytest.raises(DomainError) as denied:
        command(following, "quicksort.reply", {
            "session_id": sorting["session_id"], "home_id": home["id"]}, finish=False)
    assert denied.value.code == "INVALID_PARENT"
    assert not observations()
