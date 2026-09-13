"""Reproducible analysis and a self-contained report for the expert suite."""

import argparse
import gzip
import hashlib
import html
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

MODELS = ("gpt-5.6-luna", "gemini-3.8-flash")
RATES = {
    "gpt-5.6-luna": {
        "input": 0.20,
        "output": 1.20,
        "source": "https://developers.openai.com/api/docs/models/gpt-5.6-luna",
    },
    "gemini-3.8-flash": {
        "input": 0.75,
        "output": 3.75,
        "source": "https://ai.google.dev/gemini-api/docs/pricing",
    },
}
SUCCESS = {"clean_success", "recovered_success"}


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    index = (len(ordered) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def timings(rows, key):
    values = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and r.get("outcome") != "skipped"]
    return {
        "samples": len(values),
        "mean": statistics.mean(values) if values else None,
        "median": percentile(values, 0.5),
        "p90": percentile(values, 0.9),
        "max": max(values) if values else None,
    }


def usage_totals(rows, model):
    totals = Counter()
    for row in rows:
        for call in row.get("provider_calls", []):
            totals["provider_requests"] += 1
            usage = call.get("usage")
            if not isinstance(usage, dict):
                totals["requests_without_usage"] += 1
                continue
            incoming = usage.get("input_tokens", usage.get("prompt_tokens"))
            outgoing = usage.get("output_tokens", usage.get("completion_tokens"))
            if incoming is None or outgoing is None:
                totals["requests_without_usage"] += 1
                continue
            totals["requests_with_usage"] += 1
            totals["input_tokens"] += incoming
            totals["output_tokens"] += outgoing
            details = usage.get("input_tokens_details", usage.get("prompt_tokens_details")) or {}
            output = usage.get("output_tokens_details", usage.get("completion_tokens_details")) or {}
            for field, key in (
                ("cached_tokens", "cached_input_tokens"),
                ("cache_write_tokens", "cache_write_tokens"),
            ):
                if field in details:
                    totals[key] += details[field] or 0
                    totals[key + "_reporting_requests"] += 1
            if "reasoning_tokens" in output:
                totals["reasoning_tokens"] += output["reasoning_tokens"] or 0
                totals["reasoning_reporting_requests"] += 1
    measured = totals["requests_with_usage"]
    price = RATES[model]
    return {
        **dict(totals),
        "uncached_rate_equivalent_usd": (
            (totals["input_tokens"] * price["input"] + totals["output_tokens"] * price["output"]) / 1_000_000
            if measured
            else None
        ),
        "usage_complete": totals["requests_without_usage"] == 0,
        "note": "Published uncached-rate equivalent for reported usage, not a bill. Output already includes reasoning. Cache-reporting fields absent from a provider are unavailable, not zero.",
    }


def paired_analysis(rows, metadata, repeats, samples=10000):
    bykey = {(r["case"], r["repeat"], r["model"]): r for r in rows}
    deltas, cases = [], []
    mismatches = []
    for case in metadata:
        pairs = []
        for repeat in range(1, repeats + 1):
            pair = [bykey.get((case, repeat, m)) for m in MODELS]
            if not all(pair) or any(r.get("outcome") in {"skipped", "infrastructure_error"} for r in pair):
                continue
            for field in ("fixture_hash", "fixture_prompts_sha256", "injected_memory_sha256"):
                if pair[0].get(field) != pair[1].get(field):
                    mismatches.append({"case": case, "repeat": repeat, "field": field})
            first_prompts = [
                r.get("provider_calls", [{}])[0].get("system_prompt_sha256")
                if r.get("provider_calls")
                else None
                for r in pair
            ]
            if all(first_prompts) and first_prompts[0] != first_prompts[1]:
                mismatches.append({"case": case, "repeat": repeat, "field": "first_system_prompt"})
            pairs.append(int(pair[1]["outcome"] in SUCCESS) - int(pair[0]["outcome"] in SUCCESS))
        if len(pairs) == repeats:
            delta = statistics.mean(pairs)
            deltas.append(delta)
            cases.append({"case": case, "gemini_minus_luna": delta})
    rng = random.Random(20260913)
    boot = [statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(samples)] if deltas else []
    return {
        "complete_case_clusters": len(deltas),
        "total_cases": len(metadata),
        "gemini_minus_luna_success_fraction": statistics.mean(deltas) if deltas else None,
        "cluster_bootstrap_95_percent_interval": [percentile(boot, 0.025), percentile(boot, 0.975)]
        if boot
        else None,
        "bootstrap_samples": samples,
        "case_differences": cases,
        "fixture_or_prompt_mismatches": mismatches,
        "note": "Paired bootstrap resamples whole scenarios, retaining their repetitions. Describes this authored suite, not general real-world reliability. Infrastructure/skipped pairs excluded from this comparison.",
    }


def summarize(report, manual=None, allow_diagnostic=False):
    if report.get("diagnostic_only") and not allow_diagnostic:
        raise ValueError("Diagnostic data cannot silently become a scored report")
    rows = report["results"]
    keys = [(r["case"], r["repeat"], r["model"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate trial keys")
    reviews = (manual or {}).get("reviews", [])
    review_by_key = {(r["case"], r["repeat"], r["model"]): r for r in reviews}
    if len(review_by_key) != len(reviews) or set(review_by_key) - set(keys):
        raise ValueError("Qualitative reviews must uniquely match actual trial keys")
    manifest = report["manifest"]
    metadata, repeats = manifest["case_metadata"], manifest["repeats"]
    results = {}
    bycase = []
    for model in manifest["models"]:
        chosen = [r for r in rows if r["model"] == model]
        counts = Counter(r["outcome"] for r in chosen)
        success = sum(r["outcome"] in SUCCESS for r in chosen)
        usage = usage_totals(chosen, model)
        planned = len(metadata) * repeats
        cats = defaultdict(lambda: {"planned": 0, "observed": 0, "success": 0, "safety_failures": 0})
        consistent = 0
        for case, meta in metadata.items():
            subset = [r for r in chosen if r["case"] == case]
            wins = sum(r["outcome"] in SUCCESS for r in subset)
            consistent += int(wins == repeats and len(subset) == repeats)
            category = cats[meta["category"]]
            category["planned"] += repeats
            category["observed"] += len(subset)
            category["success"] += wins
            category["safety_failures"] += sum(r["outcome"] == "safety_failure" for r in subset)
            bycase.append(
                {
                    "model": model,
                    "case": case,
                    **meta,
                    "success": wins,
                    "planned": repeats,
                    "outcomes": dict(Counter(r["outcome"] for r in subset)),
                    "task_latency": timings(subset, "task_seconds"),
                }
            )
        audited = [
            (r, review_by_key[(r["case"], r["repeat"], r["model"])])
            for r in chosen
            if (r["case"], r["repeat"], r["model"]) in review_by_key
        ]
        qualitative = {
            "reviewed_trials": len(audited),
            "factual_response_supported": sum(a.get("factual_response_pass") is True for r, a in audited),
            "factual_response_incorrect": sum(a.get("factual_response_pass") is False for r, a in audited),
            "factual_response_uncertain": sum(a.get("factual_response_pass") is None for r, a in audited),
            "state_and_factual_confirmed": sum(
                r["outcome"] in SUCCESS and a.get("factual_response_pass") is True for r, a in audited
            ),
            "state_success_factual_uncertain": sum(
                r["outcome"] in SUCCESS and a.get("factual_response_pass") is None for r, a in audited
            ),
            "reviewer": "Codex subagent, unblinded qualitative review; original automatic grades unchanged",
        }
        results[model] = {
            "qualitative_review": qualitative,
            "planned_trials": planned,
            "recorded_trials": len(chosen),
            "not_recorded": max(0, planned - len(chosen)),
            "successful_trials": success,
            "success_fraction_of_planned": success / planned,
            "outcomes": dict(counts),
            "consistent_cases_all_repeats": consistent,
            "cases": len(metadata),
            "categories": dict(cats),
            "task_latency_all_attempts": timings(chosen, "task_seconds"),
            "task_latency_successful": timings(
                [r for r in chosen if r["outcome"] in SUCCESS], "task_seconds"
            ),
            "provider_latency_sum_per_trial": timings(chosen, "provider_seconds"),
            "usage": usage,
            "normalized_cost_per_success_usd": usage["uncached_rate_equivalent_usd"] / success
            if success and usage["uncached_rate_equivalent_usd"] is not None
            else None,
            "tool_calls": sum(len(r.get("tools", [])) for r in chosen),
            "tool_errors": dict(
                Counter(c["error"] for r in chosen for c in r.get("tools", []) if c.get("error"))
            ),
            "returned_models": sorted(
                {
                    c["response_model"]
                    for r in chosen
                    for c in r.get("provider_calls", [])
                    if c.get("response_model")
                }
            ),
        }
    return {
        "state": report["state"],
        "diagnostic_only": report.get("diagnostic_only", False),
        "planned_trials": report["planned_runs"],
        "recorded_trials": len(rows),
        "models": results,
        "cases": bycase,
        "paired": paired_analysis(rows, metadata, repeats),
        "rates": RATES,
        "rates_as_of": "2026-09-13",
        "pricing_note": "Gemini introductory rates expire December 31, 2026. Published uncached-rate equivalents ignore discounts/premiums and are not invoices.",
        "manual_review": manual or {"reviews": [], "completed": False},
        "manual_review_policy": "Original automatic grades remain unchanged. Qualitative findings are reported separately, including any proposed corrections.",
    }


def fmt_seconds(value):
    return f"{value:.2f}s" if value is not None else "unavailable"


def render(report, summary):
    esc = lambda x: html.escape(str(x)).replace("\n", "&#10;")
    labels = {"gpt-5.6-luna": "Luna · low reasoning", "gemini-3.8-flash": "Gemini 3.8 Flash · low reasoning"}
    cards = []
    for model, data in summary["models"].items():
        latency = data["task_latency_all_attempts"]
        cost = data["usage"]["uncached_rate_equivalent_usd"]
        cards.append(
            f'<section class="metric"><h2>{labels.get(model, esc(model))}</h2><strong>{data["successful_trials"]}/{data["planned_trials"]}</strong><p>Completed with no unrequested changes</p><p>{data["consistent_cases_all_repeats"]}/{data["cases"]} scenarios passed every repeat</p><p>Median task time: {fmt_seconds(latency["median"])} · P90: {fmt_seconds(latency["p90"])}</p><p>Uncached-rate equivalent: {"$" + format(cost, ".3f") if cost is not None else "unavailable"}</p></section>'
        )
    case_rows = []
    for case, meta in report["manifest"]["case_metadata"].items():
        cells = []
        for model in summary["models"]:
            data = next(r for r in summary["cases"] if r["case"] == case and r["model"] == model)
            cells.append(f"<td>{data['success']}/{data['planned']}</td>")
        case_rows.append(
            f'<tr><th scope="row">{esc(meta["title"])}<small>{esc(meta["category"])}</small></th>{"".join(cells)}</tr>'
        )
    details = []
    manual_bykey = {
        (r["model"], r["case"], r["repeat"]): r for r in summary["manual_review"].get("reviews", [])
    }
    for row in report["results"]:
        failed = [k for k, v in row.get("checks", {}).items() if not v]
        review = manual_bykey.get((row["model"], row["case"], row["repeat"]))
        text = "".join(
            f"<h4>User</h4><p>{esc(t['user'])}</p><h4>Eri</h4><p>{esc(t['assistant'])}</p>"
            for t in row.get("turns", [])
        )
        checks = (
            "<ul>" + "".join(f"<li>{esc(k)}</li>" for k in failed) + "</ul>"
            if failed
            else "<p>All automatic checks passed.</p>"
        )
        audit = (
            f"<h4>Transcript review: {esc(review['verdict'])}</h4><p>{esc(review['notes'])}</p>"
            if review
            else ""
        )
        details.append(
            f'<details data-model="{esc(row["model"])}" data-failed="{str(bool(failed) or bool(review and review.get("factual_response_pass") is not True)).lower()}"><summary>{esc(labels[row["model"]])} · {esc(row["case"])} · repeat {row["repeat"]} — {esc(row["outcome"])}</summary>{checks}{audit}{text}<details><summary>Tool trace</summary><pre>{esc(json.dumps(row.get("tools", []), indent=2, ensure_ascii=False))}</pre></details></details>'
        )
    pair = summary["paired"]
    delta = pair["gemini_minus_luna_success_fraction"]
    interval = pair["cluster_bootstrap_95_percent_interval"]
    comparison = (
        f"Gemini minus Luna: {delta * 100:+.1f} percentage points; paired scenario-bootstrap 95% interval {interval[0] * 100:+.1f} to {interval[1] * 100:+.1f} points."
        if delta is not None
        else "Paired comparison unavailable."
    )
    mismatch = pair["fixture_or_prompt_mismatches"]
    outcome_labels = {
        "clean_success": "clean completions",
        "recovered_success": "completions after recovery",
        "task_failure": "incomplete workflows",
        "safety_failure": "unrequested changes",
        "infrastructure_error": "execution errors",
        "skipped": "skipped trials",
    }
    outcome_text = " · ".join(
        labels[m]
        + ": "
        + ", ".join(str(n) + " " + outcome_labels.get(k, k) for k, n in d["outcomes"].items())
        for m, d in summary["models"].items()
    )
    run_label = "Diagnostic sample" if summary["diagnostic_only"] else "Scored run " + summary["state"]
    truth = Counter(
        "supported" if r.get("factual_response_pass") is True
        else "misleading" if r.get("factual_response_pass") is False else "uncertain"
        for r in summary["manual_review"].get("reviews", [])
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Eridani expert task-agent evaluation</title>
<style>
:root{{color-scheme:light dark;font-family:system-ui,sans-serif;color:light-dark(#202427,#e7ecef);background:light-dark(#fafbfc,#171b1f)}}
body{{max-width:1120px;margin:auto;padding:28px;line-height:1.55}}h1{{font-size:26px;margin-bottom:8px}}h2{{font-size:19px}}h3{{font-size:17px}}h4{{margin-bottom:4px}}p{{max-width:90ch}}a{{color:light-dark(#165a9a,#8bc6f9)}}.lede{{font-size:18px}}.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:32px;margin:28px 0}}.metric{{border-top:2px solid light-dark(#44657a,#9db8c9);padding-top:8px}}.metric strong{{font-size:36px;font-weight:600}}.metric p{{margin:6px 0}}small{{display:block;font-weight:400;opacity:.8}}table{{border-collapse:collapse;width:100%;margin:20px 0}}th,td{{padding:12px 8px;border-bottom:1px solid light-dark(#d4dce1,#394249);text-align:left;vertical-align:top}}td{{font-variant-numeric:tabular-nums}}th:first-child{{width:64%;font-weight:500}}details{{padding:12px 0;border-bottom:1px solid light-dark(#d4dce1,#394249)}}summary{{cursor:pointer;font-weight:500;overflow-wrap:anywhere}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}details p{{white-space:pre-wrap}}.controls{{display:flex;gap:16px;flex-wrap:wrap;margin:18px 0}}select{{font:inherit;padding:7px;max-width:100%}}label{{display:flex;align-items:center;gap:8px}}footer{{border-top:1px solid light-dark(#d4dce1,#394249);margin-top:32px;padding-top:16px}}[hidden]{{display:none!important}}@media(max-width:500px){{body{{padding:16px}}h1{{font-size:23px}}th,td{{padding:10px 4px}}.metrics{{gap:14px}}}}
</style></head><body><h1>Eridani expert task-agent evaluation</h1><p class="lede">{len(report["manifest"]["case_metadata"])} scenarios · {report["manifest"]["repeats"]} repetitions · {len(summary["models"])} production task-agent routes</p><p>{esc(run_label)} · September 13, 2026. Every effect used synthetic records. External services and UI responses were simulated; model APIs were real.</p><div class="metrics">{"".join(cards)}</div>
<h2>What the comparison establishes</h2><p>{comparison} These are repeated samples of 24 authored scenarios, not a general reliability estimate.</p><p>{esc(outcome_text)}</p><p>Pair integrity: {len(mismatch)} fixture/prompt mismatches. Transcript audit: {truth.get("supported", 0)} supported replies, {truth.get("misleading", 0)} with incorrect claims, {truth.get("uncertain", 0)} uncertain. Original automatic grades are preserved.</p><h2>Supplementary reply review</h2><p>A Codex subagent reviewed every reply against the trace. This review was not blinded or independently human-annotated; it is a separate, qualitative signal.</p><p>{" · ".join(esc(labels[m]) + ": " + str(d["qualitative_review"]["state_and_factual_confirmed"]) + "/" + str(d["planned_trials"]) + " with both successful state and supported factual replies; " + str(d["qualitative_review"]["state_success_factual_uncertain"]) + " additional successful-state replies uncertain." for m, d in summary["models"].items())}</p>
<h2>Completion by scenario</h2><table><thead><tr><th>Scenario</th>{"".join("<th>" + esc(labels[m]) + "</th>" for m in summary["models"])}</tr></thead><tbody>{"".join(case_rows)}</tbody></table>
<h2>Inspect every trial</h2><div class="controls"><label>Model <select id="model"><option value="">Both models</option><option value="gpt-5.6-luna">Luna</option><option value="gemini-3.8-flash">Gemini</option></select></label><label><input type="checkbox" id="failures" checked>Only findings</label></div><div id="trials">{"".join(details)}</div>
<footer><p>Task times include the actual app tool loop and state-audit overhead, excluding fixture setup and final grading. Costs are published uncached-rate equivalents for measured tokens, not invoices; output includes reasoning. Native reasoning content is not retained.</p><p><a href="../EXPERT_AGENT_EVALUATION.md">Protocol and scope</a> · <a href="../AGENT_TOOL_IMPROVEMENTS.md">Tool-improvement backlog</a> · <a href="expert-fixtures-2026-09-13.json.gz">Synthetic ground truth</a> · <a href="expert-agents-2026-09-13.json.gz">Full raw trace</a> · <a href="https://developers.openai.com/api/docs/models/gpt-5.6-luna">OpenAI rates</a> · <a href="https://ai.google.dev/gemini-api/docs/pricing">Google rates</a></p></footer>
<script>const model=document.getElementById('model'),fail=document.getElementById('failures');function filter(){{document.querySelectorAll('#trials > details').forEach(x=>x.hidden=(model.value&&x.dataset.model!==model.value)||(fail.checked&&x.dataset.failed!=='true'));}}model.addEventListener('change',filter);fail.addEventListener('change',filter);filter();</script></body></html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--html", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--manual", type=Path)
    parser.add_argument("--allow-diagnostic", action="store_true")
    args = parser.parse_args()
    raw = args.input.read_bytes()
    report = json.loads(raw)
    manual = json.loads(args.manual.read_text()) if args.manual else None
    result = summarize(report, manual, args.allow_diagnostic)
    result["raw_report_sha256"] = hashlib.sha256(raw).hexdigest()
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    if args.html:
        args.html.parent.mkdir(parents=True, exist_ok=True)
        args.html.write_text(render(report, result), encoding="utf-8")
    if args.archive:
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        args.archive.write_bytes(gzip.compress(raw, mtime=0))
    print(
        json.dumps(
            {
                "state": result["state"],
                "models": {
                    k: {
                        "success": v["successful_trials"],
                        "planned": v["planned_trials"],
                        "outcomes": v["outcomes"],
                    }
                    for k, v in result["models"].items()
                },
            }
        )
    )


if __name__ == "__main__":
    main()
