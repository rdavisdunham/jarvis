"""Conservative filing-language boundaries; arbitrary mentions are not human choices."""
import pytest
from jarvis.quick_sort import destination, is_answer, control_word


@pytest.mark.parametrize("text", [
    "Add a task to call someone at ABC.",
    "Please create a task for ABC.",
    "Show me the tasks under ABC.",
    "What is in Transcript Intelligence?",
    "ABC and also add a task to buy milk",
    "Put that in ABC then delete my notes",
])
def test_unrelated_or_mixed_requests_are_not_sort_answers(text):
    assert not is_answer(text)
    assert destination(text) is None


@pytest.mark.parametrize("text", [
    "Not ABC",
    "Don't put it in ABC",
    "Anything except ABC",
    "ABC or Personal",
    "ABC and Transcript Intelligence",
])
def test_negation_and_multiple_destinations_do_not_choose_a_home(text):
    assert destination(text) is None


@pytest.mark.parametrize(("text", "expected"), [
    ("ABC", "abc"),
    ("Put that in Transcript Intelligence", "transcript intelligence"),
    ("File this task under ABC please", "abc"),
    ("It belongs in Transcript Intelligence", "transcript intelligence"),
    ("Please move it to ABC", "abc"),
    ("Work / BrainForge / ABC / Transcript Intelligence",
     "work brainforge abc transcript intelligence"),
])
def test_explicit_named_destination_keeps_the_full_name(text, expected):
    assert is_answer(text)
    assert destination(text) == expected


@pytest.mark.parametrize(("text", "expected"), [
    ("skip", "skip"), ("pause quick sort", "pause"),
    ("resume sorting", "resume"), ("stop sorting", "stop"),
])
def test_controls_remain_explicit(text, expected):
    assert control_word(text) == expected


@pytest.mark.parametrize("text", ["thanks", "goodbye", "end voice"])
def test_voice_farewells_are_not_home_choices(text):
    assert not is_answer(text)
