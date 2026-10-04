"""Join content-free latency events by durable request and revision.

Usage: python scripts/summarize_latency.py < api-and-worker.log
No database or provider access. Browser render is client-reported; append is not
audible playback. A precommit event cannot establish a successful request.
"""
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * fraction) - 1] if ordered else None


def metrics(values):
    return {"n": len(values), "median": statistics.median(values) if values else None,
            "p90": percentile(values, .90), "p95": percentile(values, .95)}


def parse(lines):
    for line in lines:
        if "latency " not in line:
            continue
        try:
            event = json.loads(line.split("latency ", 1)[1])
            if not isinstance(event, dict) or not isinstance(event.get("stage"), str):
                continue
            stamp = datetime.fromisoformat(event["utc"])
            if stamp.tzinfo is None:
                continue
            event["_at"] = stamp
            yield event
        except (ValueError, TypeError, KeyError):
            continue


def summarize(lines):
    events = list({json.dumps({k: v for k, v in e.items() if k != "_at"}, sort_keys=True): e
                   for e in parse(lines)}.values())
    stages = defaultdict(lambda: {"events": 0, "errors": 0, "metrics": defaultdict(list)})
    by_request = defaultdict(list)
    for event in events:
        stage = stages[event["stage"]]
        stage["events"] += 1
        stage["errors"] += event.get("outcome") in {"error", "failed", "partial", "cancelled", "expired"}
        for field in ("duration_ms", "queue_ms", "dispatch_ms"):
            if finite(event.get(field)):
                stage["metrics"][field].append(event[field])
        if isinstance(event.get("request_id"), str):
            by_request[(event["request_id"], event.get("revision") or 1)].append(event)

    rows = []
    for (identity, revision), records in sorted(by_request.items()):
        # Session-only or UI-action-only IDs have no durable runner/work event.
        if not any(e["stage"] in {"work_http_received", "accepted_precommit", "runner_started", "work_accepted", "work_finished", "voice_claimed"} for e in records):
            continue
        records.sort(key=lambda e: e["_at"])
        first = {}
        for event in records:
            first.setdefault(event["stage"], event)
        terminal = next((e for e in reversed(records) if e["stage"] == "work_finished"), None)
        runner = first.get("runner_started")
        profile_event = {}
        for event in records:
            for field in ("profile", "model", "channel", "reasoning", "requested_service_tier"):
                if event.get(field) is not None:
                    profile_event[field] = event[field]
        # These profile IDs have fixed effort semantics, including older partial logs.
        profile_event.setdefault("reasoning", {"luna": "low", "luna-none": "none"}.get(profile_event.get("profile")))
        usage = {e["response_id"]: e for e in records if e["stage"] == "model_usage" and e.get("response_id")}
        attempts = [e for e in records if e["stage"] == "model_attempt"]
        attempts_by_round = defaultdict(int)
        for attempt in attempts:
            attempts_by_round[(attempt.get("run_id"), attempt.get("round", 0))] += 1
        rendered = next((e for e in records if e["stage"] in {"work_card_rendered", "work_reply_rendered"}), None)
        start = first.get("work_http_received") or first.get("voice_claimed")
        wall_ms = None
        skew = False
        if start and rendered:
            wall_ms = round((rendered["_at"] - start["_at"]).total_seconds() * 1000, 3)
            if wall_ms < 0:
                skew, wall_ms = True, None
        client_ms = rendered.get("elapsed_ms") if rendered and finite(rendered.get("elapsed_ms")) else None
        outcome = terminal.get("outcome") if terminal else "censored"
        def interval(start_stage, end_stage):
            begin, end = first.get(start_stage), first.get(end_stage)
            if not begin or not end:
                return None
            elapsed = (end["_at"] - begin["_at"]).total_seconds() * 1000
            return round(elapsed, 3) if elapsed >= 0 else None

        row = {
            "request_id": identity, "revision": revision,
            "channel": profile_event.get("channel") or ("live" if first.get("voice_claimed") else "unknown"),
            "profile": profile_event.get("profile"), "model": profile_event.get("model"),
            "reasoning": profile_event.get("reasoning"),
            "requested_service_tier": profile_event.get("requested_service_tier"),
            "served_service_tiers": sorted({e["served_service_tier"] for e in usage.values()
                                           if e.get("served_service_tier")}),
            "estimated_tier_cost": any(e.get("cost_basis") == "requested_service_tier_estimate"
                                       for e in usage.values()),
            "outcome": outcome,
            "receipts": terminal.get("receipts") if terminal else None,
            "invocations": len({e.get("run_id") for e in records if e["stage"] == "runner_started"}),
            "uncertain_spend": any(e.get("uncertain_spend") for e in records),
            "missing_usage": bool(attempts) and not usage,
            "stage_ms": {name: sum(e["duration_ms"] for e in records
                                   if e["stage"] == name and finite(e.get("duration_ms")))
                         for name in ("context", "model", "tool", "tool_discovery")},
            "accept_to_runner_ms": None if not runner or not first.get("work_accepted") else
                (runner["_at"] - first["work_accepted"]["_at"]).total_seconds() * 1000,
            "model_rounds": len({(e.get("run_id"), e.get("round")) for e in records if e["stage"] == "model"}),
            "model_attempts": len(attempts),
            "retries": sum(max(0, n - 1) for n in attempts_by_round.values()),
            "tool_calls": sum(e["stage"] == "tool" for e in records),
            "tool_discovery": sum(e["stage"] == "tool_discovery" for e in records),
            "usage": {
                "input_tokens": sum(e.get("input_tokens") or 0 for e in usage.values()),
                "output_tokens": sum(e.get("output_tokens") or 0 for e in usage.values()),
                "reasoning_tokens": sum(e.get("reasoning_tokens") or 0 for e in usage.values()),
                "cached_tokens": sum(e.get("cached_tokens") or 0 for e in usage.values()),
                "cache_write_tokens": sum(e.get("cache_write_tokens") or 0 for e in usage.values()),
                "recorded_cost_usd": round(sum(e.get("cost_usd") or 0 for e in usage.values()), 9),
                "responses": len(usage),
            },
            "accept_to_finish_ms": interval("work_accepted", "work_finished"),
            "runner_to_finish_ms": interval("runner_started", "work_finished"),
            "claim_to_append_ms": interval("voice_claimed", "live_append_sent"),
            "send_to_render_client_ms": client_ms,
            "ingress_to_render_observed_ms": wall_ms,
            "cross_clock_skew": skew,
            "browser_render_reported": bool(rendered),
            "live_append_reported": bool(first.get("live_append_sent")),
            "audible_completion_ms": None,
        }
        if row["accept_to_runner_ms"] is not None and row["accept_to_runner_ms"] < 0:
            row["cross_clock_skew"] = True
            row["accept_to_runner_ms"] = None
        rows.append(row)

    groups = defaultdict(list)
    for row in rows:
        groups[(row["channel"], row["profile"], row["reasoning"], row["requested_service_tier"])].append(row)
    grouped = {}
    for (channel, profile, reasoning, tier), group in groups.items():
        successful = [r for r in group if r["outcome"] == "succeeded"]
        measured = [r["send_to_render_client_ms"] for r in successful if r["send_to_render_client_ms"] is not None]
        observed = [r["ingress_to_render_observed_ms"] for r in successful if r["ingress_to_render_observed_ms"] is not None]
        costs = [r["usage"]["recorded_cost_usd"] for r in successful if r["usage"]["responses"]]
        total_cost = round(sum(r["usage"]["recorded_cost_usd"] for r in group), 9)
        uncertain = sum(r["uncertain_spend"] or r["missing_usage"] or r["outcome"] == "censored" for r in group)
        estimates = sum(r["estimated_tier_cost"] for r in group)
        group_key = "/".join((channel or "unknown", profile or "unknown", reasoning or "unknown"))
        if tier:
            group_key += "/" + tier
        grouped[group_key] = {
            "attempted": len(group), "distinct_requests": len({r["request_id"] for r in group}), "succeeded": len(successful),
            "failed_or_censored": len(group) - len(successful),
            "accept_to_finish_ms": metrics([r["accept_to_finish_ms"] for r in successful if r["accept_to_finish_ms"] is not None]),
            "client_send_to_render_ms": metrics(measured),
            "ingress_to_render_observed_ms": metrics(observed),
            "successful_request_recorded_cost_usd": metrics(costs),
            "recorded_cost_per_success_usd": round(total_cost / len(successful), 9) if successful else None,
            "cost_evidence": ("includes_tier_estimates" if estimates else
                              "incomplete_lower_bound" if uncertain else "recorded_model_usage_only"),
            "estimated_tier_cost_cases": estimates,
            "incomplete_cost_cases": uncertain,
            "total_recorded_cost_usd": total_cost,
            "model_attempts": sum(r["model_attempts"] for r in group),
            "retries": sum(r["retries"] for r in group),
            "clock_skew_cases": sum(r["cross_clock_skew"] for r in group),
        }
    return {
        "stages": {name: {"events": stage["events"], "errors": stage["errors"],
                         "metrics": {key: metrics(values) for key, values in stage["metrics"].items()}}
                   for name, stage in stages.items()},
        "requests": rows, "groups": grouped,
        "limits": ["Client render is an observed painted-card approximation, not proof of requested content visibility.",
                   "Live append is not audible completion; device speech timing remains unmeasured.",
                   "Precommit-only and missing-terminal requests are censored, not successes.",
                   "Attempted counts request revisions; distinct_requests deduplicates revisions.",
                   "Cost per success includes recorded spend on failed/censored requests; uncertain spend is a lower bound.",
                   "Initial client rendering covers same-tab typed requests only; historic cards and voice speech are unmeasured.",
                   "Cross-process wall time uses UTC and may include clock skew; client elapsed uses one monotonic clock.",
                   "Recorded model usage is counted once per response ID; absent usage or uncertain spend is not zero-cost proof."],
    }


if __name__ == "__main__":
    print(json.dumps(summarize(sys.stdin), indent=2))
