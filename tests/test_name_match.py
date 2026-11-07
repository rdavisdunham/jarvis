"""Sound-alike and near-spelling name matching (name_match.py)."""
import pytest
from jarvis.name_match import alike, score, sound


@pytest.mark.parametrize("heard, known", [
    ("Janice", "Janiece"), ("Janis", "Janiece"), ("Jeanice", "Janiece"), ("Shreya", "Sreya"), ("Sriya", "Sreya"),
    ("Steven", "Stephen"), ("Bryan", "Brian"), ("Catherine", "Katherine"), ("Evette", "Yvette"), ("Jon", "John"),
    ("Zoë", "Zoe"),
])
def test_misheard_names_match(heard, known):
    assert alike(heard, known)


@pytest.mark.parametrize("heard, known", [
    ("Janet", "Janiece"), ("Sorry", "Sarah"), ("Maria", "Priya"), ("Market", "Marco"), ("Ann", "Anne"),
])
def test_different_names_and_words_do_not(heard, known):
    assert not alike(heard, known)


def test_scores_and_keys():
    assert score("Janiece", "janiece") == 1.0 and score("", "Janiece") == 0.0
    assert sound("Shreya") == sound("Sreya") == "sra"
    assert sound("Yvette")[0] == sound("Evette")[0] == "a"
