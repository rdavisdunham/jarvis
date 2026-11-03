import json
from pathlib import Path

import pytest
from jarvis.voice_end_policy import EndPolicy, decide

CASES = json.loads((Path(__file__).parent / "fixtures/voice-ending.json").read_text())

@pytest.mark.parametrize("case", CASES)
def test_shared_decisions(case):
    values = {k:v for k,v in case.items() if k != "decision"}
    assert decide(**values) == case["decision"]


def test_offer_requires_ambiguous_input_and_observed_approved_question():
    p = EndPolicy()
    p.input("thanks", 0, 100, 0)
    assert p.offer(clock=1) is None
    p.assistant("Anything else?", 2)  # invented, not approved
    assert p.input("No", 200, 300, 3)["question"] is None
    p.input("I'm done", 2000, 2100, 4)
    assert p.offer(pending=True, clock=5) is None
    assert p.offer(clock=5) == "Will that be all for now?"
    assert p.offer(clock=6) is None
    p.assistant("Will that be all for now?", 6)
    ctx = p.input("Yes, thanks", 2200, 2400, 7)
    assert decide(**ctx) == "confirm"
    p.input("I'm good", 4000, 4100, 8)
    assert p.offer(clock=180) is None
    assert p.offer(clock=186)


def test_expired_unrelated_and_new_requests_consume_context():
    p = EndPolicy()
    p.input("that's all", 0, 100, 0); p.offer(clock=1)
    p.assistant("Will that be all for now?", 2)
    assert decide(**p.input("Yes", 200, 300, 63)) == "keep"
    p.assistant("Will that be all for now?", 64)
    assert decide(**p.input("Yes", 400, 500, 65)) == "keep"


def test_task_question_reply_never_generates_a_closing_offer():
    p = EndPolicy()
    p.input("Add milk", 0,100,0)
    p.assistant("Which list?",1)
    context=p.input("I'm done",200,300,2)
    assert context["pending"]
    assert p.offer(clock=3) is None
    assert decide(**context) == "keep"
