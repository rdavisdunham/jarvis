"""Operator answers may select options, add notes, or use text alone without losing either."""
from uuid import uuid4

import pytest
from jarvis import agent_plan
from jarvis.db import session_scope
from jarvis.domain import DomainError
from jarvis.session_models import AgentCommand, AgentInterview
from pydantic import ValidationError
from sqlalchemy import select
from test_agent_plan import call, interviewing
from test_agent_runs import host as host  # noqa: F401 - fixture


def clean(question, answer):
    body = agent_plan.InterviewAnswers(answers={"q": answer})
    return agent_plan.clean_answers([{"id": "q", "options": [{"label": "A"}, {"label": "B"}],
                                      "free_text": False, **question}], body.answers)["q"]


@pytest.mark.parametrize(("answer", "expected"), [
    ("My own answer", "My own answer"),
    ("  My own answer  ", "My own answer"),
    ("A\n\nNotes: Keep this detail.\nAnd this one.", "A\n\nNotes: Keep this detail.\nAnd this one."),
    (["A"], "A"),
])
def test_old_single_choice_interviews_allow_text_and_notes(answer, expected):
    assert clean({}, answer) == expected


def test_multichoice_notes_do_not_use_up_an_option_slot():
    labels = [str(i) + "x" * 199 for i in range(8)]
    notes = "n" * 4000
    answer = "Selected options: " + "; ".join(labels) + "\n\nNotes: " + notes
    assert clean({"multi": True, "options": [{"label": label} for label in labels]},
                 [answer]) == [answer]
    assert clean({"multi": True}, ["Something else"]) == ["Something else"]


@pytest.mark.parametrize("answer", [
    "  ",
    [],
    ["A", "B"],
])
def test_selection_and_required_answer_validation_remains(answer):
    with pytest.raises(DomainError) as error:
        clean({}, answer)
    assert error.value.code == "INVALID_ARGUMENT"


@pytest.mark.parametrize("answer", [
    "n" * 6001,
    ["A"] * 9,
    {"choices": ["A"], "text": "Unsupported shape"},
])
def test_existing_answer_size_and_shape_remain_bounded(answer):
    with pytest.raises(ValidationError):
        agent_plan.InterviewAnswers(answers={"q": answer})


@pytest.mark.parametrize("route", ["decisions", "interviews"])
def test_saved_notes_reach_the_planning_prompt_once_through_both_answer_routes(client, host, route):
    _, run_id = interviewing(host)
    asked = call(agent_plan.task_interview, agent_plan.InterviewInput, run_id, questions=[
        {"id": "scope", "question": "Which widgets?", "options": [{"label": "All"}], "free_text": False},
        {"id": "details", "question": "What matters?", "options": [{"label": "Speed"}], "free_text": False}])
    identity = asked["interview_id"]
    with session_scope() as db:
        interview = db.get(AgentInterview, identity)
        assert all(q["free_text"] for q in interview.questions)
        # Existing forms keep their saved question metadata across this update.
        interview.questions = [{**q, "free_text": False} for q in interview.questions]
    body = {"action_id": str(uuid4()), "answers": {
        "scope": "All\n\nNotes: Start with charts.\nPreserve keyboard controls.",
        "details": "Accessibility matters too"}}
    suffix = "respond" if route == "decisions" else "answers"
    if route == "decisions":
        body.update(action="answer", expected_revision=1)
    url = f"/api/v1/remotea/{route}/{identity}/{suffix}"
    first = client.post(url, json=body)
    assert first.status_code == 200, first.text
    again = client.post(url, json=body)
    assert again.status_code == 200, again.text
    expected = {"scope": "All\n\nNotes: Start with charts.\nPreserve keyboard controls.",
                "details": "Accessibility matters too"}
    with session_scope() as db:
        interview = db.get(AgentInterview, identity)
        assert interview.status == "answered" and interview.answers == expected
        commands = db.scalars(select(AgentCommand).where(AgentCommand.run_id == run_id,
                                                        AgentCommand.kind == "input")).all()
        assert len(commands) == 1
        assert "All" in commands[0].payload["text"]
        assert "Start with charts." in commands[0].payload["text"]
        assert "Preserve keyboard controls." in commands[0].payload["text"]
        assert "Accessibility matters too" in commands[0].payload["text"]
