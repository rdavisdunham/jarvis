import importlib.util
import json
from pathlib import Path

# Console-script pytest does not add the repository root to sys.path in CI.
spec = importlib.util.spec_from_file_location(
    "latency_summary", Path(__file__).resolve().parents[1] / "scripts/summarize_latency.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
summarize = module.summarize


def line(stage, identity, utc, **fields):
    return "latency " + json.dumps({"stage": stage, "request_id": identity, "utc": utc, **fields})


def test_joined_summary_counts_failure_and_censored_and_deduplicates_usage():
    lines = [
        line("work_http_received", "good", "2026-10-04T00:00:00+00:00"),
        line("work_accepted", "good", "2026-10-04T00:00:00.050000+00:00", profile="luna-none", channel="work"),
        line("runner_started", "good", "2026-10-04T00:00:00.100000+00:00", profile="luna-none", channel="work"),
        line("model_attempt", "good", "2026-10-04T00:00:00.200000+00:00", revision=1, round=0, attempt=1),
        line("model_attempt", "good", "2026-10-04T00:00:00.300000+00:00", revision=1, round=0, attempt=2),
        line("model_usage", "good", "2026-10-04T00:00:00.500000+00:00", revision=1,
             response_id="resp1", input_tokens=100, output_tokens=20, cost_usd=.0001),
        line("model_usage", "good", "2026-10-04T00:00:00.500000+00:00", revision=1,
             response_id="resp1", input_tokens=100, output_tokens=20, cost_usd=.0001),
        line("work_finished", "good", "2026-10-04T00:00:00.800000+00:00", revision=1,
             outcome="succeeded", receipts=1, profile="luna-none", reasoning="none", channel="work"),
        line("work_card_rendered", "good", "2026-10-04T00:00:01+00:00",
             revision=1, elapsed_ms=1000, source="client_reported"),
        line("work_accepted", "failed", "2026-10-04T00:00:00+00:00", profile="luna-none", channel="work"),
        line("work_finished", "failed", "2026-10-04T00:00:01+00:00",
             outcome="failed", profile="luna-none", reasoning="none", channel="work"),
        line("work_accepted", "censored", "2026-10-04T00:00:00+00:00",
             profile="luna-none", channel="work"),
    ]
    result = summarize(lines)
    group = result["groups"]["work/luna-none/none"]
    assert group["attempted"] == 3 and group["succeeded"] == 1
    assert group["failed_or_censored"] == 2
    good = next(r for r in result["requests"] if r["request_id"] == "good")
    assert good["retries"] == 1 and good["usage"]["responses"] == 1
    assert good["usage"]["recorded_cost_usd"] == .0001
    assert good["send_to_render_client_ms"] == 1000
    assert next(r for r in result["requests"] if r["request_id"] == "censored")["outcome"] == "censored"


def test_negative_cross_clock_difference_is_flagged_not_a_fast_success():
    result = summarize([
        line("work_http_received", "skew", "2026-10-04T00:00:02+00:00"),
        line("work_finished", "skew", "2026-10-04T00:00:03+00:00", outcome="succeeded"),
        line("work_card_rendered", "skew", "2026-10-04T00:00:01+00:00", source="client_reported"),
    ])
    row = result["requests"][0]
    assert row["cross_clock_skew"] and row["ingress_to_render_observed_ms"] is None
    assert row["audible_completion_ms"] is None

def test_failure_spend_is_included_and_incomplete_is_not_a_free_success():
    records = [
        line("work_accepted", "ok", "2026-10-04T00:00:00Z", profile="luna", channel="work"),
        line("model_usage", "ok", "2026-10-04T00:00:01Z", response_id="ok", cost_usd=.001),
        line("work_finished", "ok", "2026-10-04T00:00:02Z", outcome="succeeded"),
        line("accepted_precommit", "bad", "2026-10-04T00:00:00Z", profile="luna", channel="work"),
        line("model_usage", "bad", "2026-10-04T00:00:01Z", response_id="bad", cost_usd=.002),
        line("work_finished", "bad", "2026-10-04T00:00:02Z", outcome="failed"),
        line("accepted_precommit", "unknown", "2026-10-04T00:00:00Z", profile="luna", channel="work"),
    ]
    report = summarize(records + [records[1]])  # duplicated export line
    group = report["groups"]["work/luna/low"]
    assert group["attempted"] == 3
    assert group["recorded_cost_per_success_usd"] == .003
    assert group["cost_evidence"] == "incomplete_lower_bound"
    assert group["incomplete_cost_cases"] == 1
    assert len(report["requests"]) == 3


def test_text_reply_and_resumed_runs_keep_measurement_and_attempt_identity():
    report = summarize([
        line("work_accepted", "r", "2026-10-04T00:00:00Z", profile="luna-none", channel="work"),
        line("runner_started", "r", "2026-10-04T00:00:01Z", run_id="a"),
        line("model_attempt", "r", "2026-10-04T00:00:02Z", run_id="a", round=0, attempt=1),
        line("runner_started", "r", "2026-10-04T00:00:03Z", run_id="b"),
        line("model_attempt", "r", "2026-10-04T00:00:04Z", run_id="b", round=0, attempt=1),
        line("model_attempt", "r", "2026-10-04T00:00:05Z", run_id="b", round=0, attempt=2),
        line("work_finished", "r", "2026-10-04T00:00:06Z", outcome="succeeded"),
        line("work_reply_rendered", "r", "2026-10-04T00:00:07Z", elapsed_ms=7000),
    ])
    row = report["requests"][0]
    assert row["invocations"] == 2 and row["retries"] == 1
    assert row["send_to_render_client_ms"] == 7000
    assert row["missing_usage"]


def test_tier_groups_do_not_mix_and_estimated_cost_is_labelled():
    events = []
    for identity, tier, served, basis in [
        ("standard", "default", "default", "returned_service_tier"),
        ("fast", "fast", "priority", "returned_service_tier"),
        ("downgrade", "fast", "default", "returned_service_tier"),
        ("estimate", "fast", None, "requested_service_tier_estimate"),
    ]:
        events.extend([
            line("work_accepted", identity, "2026-10-04T00:00:00Z",
                 profile="luna", channel="work", requested_service_tier=tier),
            line("model_usage", identity, "2026-10-04T00:00:01Z", response_id=identity,
                 served_service_tier=served, cost_basis=basis, cost_usd=.001),
            line("work_finished", identity, "2026-10-04T00:00:02Z", outcome="succeeded"),
        ])
    report = summarize(events)
    assert report["groups"]["work/luna/low/default"]["attempted"] == 1
    fast = report["groups"]["work/luna/low/fast"]
    assert fast["attempted"] == 3 and fast["estimated_tier_cost_cases"] == 1
    assert fast["cost_evidence"] == "includes_tier_estimates"
    assert next(r for r in report["requests"] if r["request_id"] == "downgrade")["served_service_tiers"] == ["default"]
