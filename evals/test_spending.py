import json

import httpx
import pytest

from jarvis.agent_models import AgentModel
from scripts.app_eval.spending import EvalLimit, SpendingMeter


def agent():
    return AgentModel("openai", "gpt-5.6-luna", "Luna", "https://api.openai.com/v1/responses", "",
                      max_output_tokens=8192, profile="luna", api="responses")


def request(a):
    return httpx.Request("POST", a.endpoint, json={"model": a.model, "max_output_tokens": 8192, "input": []})


def test_preflight_counts_uncertain_retries_and_refuses_next_request(tmp_path):
    a = agent()
    meter = SpendingMeter(.025, tmp_path / "cost.json")
    for _ in range(2):
        row = meter.reserve(a, request(a))
        meter.settle(row, a, httpx.Response(429, json={"error": "rate limited"}))
    with pytest.raises(EvalLimit):
        meter.reserve(a, request(a))
    assert len(json.loads(meter.path.read_text())["calls"]) == 2


def test_usage_releases_unused_bound_and_cached_input_discount(tmp_path):
    a = agent()
    meter = SpendingMeter(.02, tmp_path / "cost.json")
    row = meter.reserve(a, request(a))
    meter.settle(row, a, httpx.Response(200, json={"usage": {
        "input_tokens": 1000, "output_tokens": 100,
        "input_tokens_details": {"cached_tokens": 500},
    }}))
    assert float(meter.total("luna")) == pytest.approx(.00023)
    meter.reserve(a, request(a))


@pytest.mark.parametrize("cap", [0, -1, 11, "nan", "inf"])
def test_invalid_caps_are_refused(tmp_path, cap):
    with pytest.raises(ValueError):
        SpendingMeter(cap, tmp_path / "cost.json")


def test_long_context_upper_bound_and_bill():
    a = agent()
    assert a.reserve_cost(300000) == pytest.approx((300000 * .5 + 8192 * 1.8) / 1000000)
    assert a.usage_cost({"prompt_tokens": 300000, "completion_tokens": 100}) == pytest.approx(.12018)


def test_gemini_max_tokens_and_independent_profile_ceiling(tmp_path):
    from dataclasses import replace
    luna = agent()
    gemini = replace(luna, provider="gemini", model="gemini-3.8-flash", profile="gemini", api="chat_completions")
    meter = SpendingMeter(.1, tmp_path / "cost.json")
    body = {"model": gemini.model, "max_tokens": gemini.max_output_tokens, "messages": []}
    row = meter.reserve(gemini, httpx.Request("POST", gemini.endpoint, json=body))
    meter.settle(row, gemini, httpx.Response(200, json={"usage": {"prompt_tokens": 1000, "completion_tokens": 100}}))
    assert float(meter.total("gemini")) == pytest.approx(.001125)
    meter.reserve(luna, request(luna))
    assert len(meter.calls) == 2
