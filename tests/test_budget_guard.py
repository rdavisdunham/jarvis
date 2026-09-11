from uuid import uuid4

import pytest
from jarvis import budget
from jarvis.db import session_scope
from jarvis.domain import DomainError, execute


def test_continuations_need_additional_reservation():
    rid = str(uuid4())
    with session_scope() as db:
        execute(db, "davin", str(uuid4()), "settings.update", {"monthly_budget_usd": 1})
        budget.reserve(db, "davin", rid, 0.1, "gpt-5.4-mini")
        budget.ensure_room(db, "davin", rid, 0.8)
        budget.record_usage(db, "davin", rid, "request-1", "gpt-5.4-mini", {}, 0.7)
        budget.ensure_room(db, "davin", rid, 0.2)
    with session_scope() as db, pytest.raises(DomainError, match="spending is paused"):
        budget.ensure_room(db, "davin", rid, 0.4)


def test_current_realtime_rates_include_text_and_cached_modalities():
    usage = {
        "input_token_details": {
            "audio_tokens": 200,
            "text_tokens": 100,
            "cached_tokens_details": {"audio_tokens": 100, "text_tokens": 50},
        },
        "output_token_details": {"audio_tokens": 100, "text_tokens": 100},
    }
    assert budget.realtime_cost(usage, "gpt-realtime-2.1") == pytest.approx(0.01226)
    assert budget.realtime_cost(usage, "gpt-realtime-2.1-mini") == pytest.approx(0.003303)
