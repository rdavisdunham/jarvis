import copy
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
s = importlib.import_module("summarize_expert_agents")


def report(repeats=3):
    rows = []
    for model in s.MODELS:
        for repeat in range(1, repeats + 1):
            rows.append(
                {
                    "model": model,
                    "case": "scope",
                    "repeat": repeat,
                    "outcome": "clean_success",
                    "task_seconds": 10.0,
                    "provider_seconds": 8.0,
                    "fixture_hash": "same",
                    "fixture_prompts_sha256": "same",
                    "injected_memory_sha256": "same",
                    "provider_calls": [
                        {
                            "system_prompt_sha256": "same",
                            "response_model": model,
                            "usage": {
                                "input_tokens": 1000,
                                "output_tokens": 100,
                                "output_tokens_details": {"reasoning_tokens": 50},
                            },
                        }
                    ],
                    "tools": [],
                    "turns": [],
                    "checks": {"state": True},
                }
            )
    return {
        "state": "completed",
        "diagnostic_only": False,
        "planned_runs": 2 * repeats,
        "manifest": {
            "case_metadata": {"scope": {"title": "Scope", "category": "Precision"}},
            "repeats": repeats,
            "models": {m: {} for m in s.MODELS},
        },
        "results": rows,
    }


def test_failed_trial_stays_in_denominator_and_cost():
    r = report()
    r["results"][0]["outcome"] = "task_failure"
    a = s.summarize(r)
    luna = a["models"][s.MODELS[0]]
    assert luna["successful_trials"] == 2
    assert luna["planned_trials"] == 3
    assert luna["consistent_cases_all_repeats"] == 0
    assert luna["normalized_cost_per_success_usd"] == pytest.approx(0.00096 / 2)


def test_reasoning_is_not_charged_twice():
    a = s.summarize(report(1))["models"][s.MODELS[0]]["usage"]
    assert a["output_tokens"] == 100 and a["reasoning_tokens"] == 50
    assert a["uncached_rate_equivalent_usd"] == pytest.approx(0.00032)
    assert a.get("cached_input_tokens") is None


def test_missing_usage_is_unavailable_not_zero_cost():
    r = report(1)
    r["results"][0]["provider_calls"][0].pop("usage")
    a = s.summarize(r)["models"][s.MODELS[0]]["usage"]
    assert not a["usage_complete"]
    assert a["uncached_rate_equivalent_usd"] is None


def test_missing_trial_not_success():
    r = report()
    r["results"].pop()
    a = s.summarize(r)
    assert a["models"][s.MODELS[1]]["not_recorded"] == 1
    assert a["models"][s.MODELS[1]]["success_fraction_of_planned"] == pytest.approx(2 / 3)
    assert a["paired"]["complete_case_clusters"] == 0


def test_duplicate_trial_is_rejected():
    r = report()
    r["results"].append(copy.deepcopy(r["results"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        s.summarize(r)


def test_diagnostics_need_explicit_opt_in():
    r = report()
    r["diagnostic_only"] = True
    with pytest.raises(ValueError, match="Diagnostic"):
        s.summarize(r)
    assert s.summarize(r, allow_diagnostic=True)["diagnostic_only"]


def test_paired_cluster_bootstrap_keeps_repeats_together():
    r = report()
    for row in r["results"]:
        if row["model"] == s.MODELS[0]:
            row["outcome"] = "task_failure"
    a = s.summarize(r)["paired"]
    assert a["complete_case_clusters"] == 1
    assert a["gemini_minus_luna_success_fraction"] == 1
    assert a["cluster_bootstrap_95_percent_interval"] == [1, 1]


def test_fixture_mismatch_is_visible():
    r = report()
    r["results"][0]["fixture_hash"] = "wrong"
    a = s.summarize(r)["paired"]
    assert len(a["fixture_or_prompt_mismatches"]) == 1


def test_manual_review_does_not_rewrite_automatic_grade():
    manual = {
        "reviews": [
            {
                "model": s.MODELS[0],
                "case": "scope",
                "repeat": 1,
                "verdict": "misleading",
                "notes": "Claimed unverified success.",
            }
        ],
        "completed": True,
    }
    a = s.summarize(report(), manual)
    assert a["models"][s.MODELS[0]]["successful_trials"] == 3
    assert a["manual_review"] == manual


def test_report_escapes_untrusted_text():
    r = report(1)
    r["results"][0]["turns"] = [{"user": "<script>alert(1)</script>", "assistant": "Saved."}]
    page = s.render(r, s.summarize(r))
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page


def test_skipped_times_do_not_make_latency_look_fast():
    r = report()
    r["results"][0].update(outcome="skipped", task_seconds=0)
    a = s.summarize(r)["models"][s.MODELS[0]]
    assert a["task_latency_all_attempts"]["median"] == 10
    assert a["task_latency_all_attempts"]["samples"] == 2


def test_uncertain_response_is_not_confirmed_joint_success():
    r = report(1)
    manual = {
        "reviews": [{"model": s.MODELS[0], "case": "scope", "repeat": 1, "factual_response_pass": None}],
        "completed": True,
    }
    q = s.summarize(r, manual)["models"][s.MODELS[0]]["qualitative_review"]
    assert q["state_and_factual_confirmed"] == 0
    assert q["state_success_factual_uncertain"] == 1


def test_manual_review_cannot_attach_to_wrong_trial():
    manual = {
        "reviews": [
            {"model": s.MODELS[0], "case": "not_in_suite", "repeat": 1, "factual_response_pass": True}
        ]
    }
    with pytest.raises(ValueError, match="uniquely match"):
        s.summarize(report(), manual)
