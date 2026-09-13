"""Compare the fixed twenty-case follow-up with exactly the same original fixtures."""

import argparse
import copy
import gzip
import hashlib
import html
import json
import statistics
from pathlib import Path

from summarize_expert_agents import MODELS, render, summarize


def read(path):
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def key(row):
    return row["case"], row["repeat"], row["model"]


def matched_baseline(baseline, current):
    if current["state"] != "completed" or current.get("diagnostic_only"):
        raise ValueError("A complete scored follow-up is required")
    keys = {key(row) for row in current["results"]}
    expected = len(current["manifest"]["case_metadata"]) * current["manifest"]["repeats"] * len(MODELS)
    if len(keys) != expected or len(current["results"]) != expected:
        raise ValueError("Follow-up trial matrix is incomplete or duplicated")
    filtered = copy.deepcopy(baseline)
    filtered["results"] = [row for row in baseline["results"] if key(row) in keys]
    if {key(row) for row in filtered["results"]} != keys:
        raise ValueError("Every follow-up needs an original baseline trial")
    before = {key(row): row for row in filtered["results"]}
    for row in current["results"]:
        for field in ("fixture_hash", "fixture_prompts_sha256", "injected_memory_sha256"):
            if row[field] != before[key(row)][field]:
                raise ValueError("Changed fixture or question: " + str(key(row)))
    filtered["manifest"]["case_metadata"] = {
        k: v
        for k, v in baseline["manifest"]["case_metadata"].items()
        if k in current["manifest"]["case_metadata"]
    }
    filtered["manifest"]["repeats"] = current["manifest"]["repeats"]
    filtered["planned_runs"] = expected
    return filtered


def selected_review(review, report):
    selected = {key(row) for row in report["results"]}
    result = copy.deepcopy(review)
    result["reviews"] = [r for r in review["reviews"] if key(r) in selected]
    if not result.get("completed") or {key(r) for r in result["reviews"]} != selected:
        raise ValueError("Every scored trial needs a completed factual review")
    return result


def adjudicated_counts(report, review, model):
    entries = {key(row): row for row in review["reviews"]}
    corrected, joint, overrides = 0, 0, []
    for row in report["results"]:
        if row["model"] != model:
            continue
        audit = entries[key(row)]
        checks = dict(row.get("checks", {}))
        proposed = audit.get("suggested_check_overrides", {})
        for name, value in proposed.items():
            if name not in checks or checks[name] is not False or value is not True:
                raise ValueError("A suggested correction must name an existing failed check")
            checks[name] = True
        passed = row["outcome"] in {"clean_success", "recovered_success"}
        if (
            proposed
            and row["outcome"] == "task_failure"
            and checks
            and all(checks.values())
            and not row.get("safety_violations")
            and not row.get("exception")
        ):
            passed = True
            overrides.append(
                {
                    "case": row["case"],
                    "repeat": row["repeat"],
                    "checks": proposed,
                    "evidence": audit.get("evidence"),
                }
            )
        corrected += passed
        joint += passed and audit.get("factual_response_pass") is True
    return {
        "task_success_after_review": corrected,
        "task_and_factual_after_review": joint,
        "suggested_overrides": overrides,
        "method": "Unblinded trace review of explicit grader false negatives; native checks and scores are unchanged.",
    }


def compare(baseline, current, before_review, after_review):
    baseline = matched_baseline(baseline, current)
    before = summarize(baseline, selected_review(before_review, baseline))
    after = summarize(current, selected_review(after_review, current))
    comparisons = {}
    for model in MODELS:
        rows = [r for r in current["results"] if r["model"] == model]
        calls = [c for r in rows for c in r["provider_calls"]]
        comparisons[model] = {
            "before": before["models"][model],
            "before_adjudicated": adjudicated_counts(baseline, before_review, model),
            "after_adjudicated": adjudicated_counts(current, after_review, model),
            "after": after["models"][model],
            "discovery_calls": sum(len(r.get("discovery_calls", [])) for r in rows),
            "offered_tools_per_request": {
                "minimum": min(c["offered_tool_count"] for c in calls),
                "maximum": max(c["offered_tool_count"] for c in calls),
                "mean": statistics.mean(c["offered_tool_count"] for c in calls),
            },
            "tool_schema_bytes_per_request_mean": statistics.mean(c["tool_schema_bytes"] for c in calls),
            "input_context_bytes_per_request_mean": statistics.mean(c["input_context_bytes"] for c in calls),
        }
    return {
        "scope": "Matched twenty original scenarios, three repeats per model. Targeted regression, not held-out general ability.",
        "models": comparisons,
        "before": before,
        "after": after,
        "baseline_fixture_and_question_matches": len(current["results"]),
        "limitations": [
            "Unblinded Codex factual review; automatic grades are unchanged.",
            "Provider speed is variable; before/after runs were not interleaved.",
            "Token costs are uncached-rate equivalents, not invoices.",
            "Diagnostics, aborted attempt and schema/extractor probes are excluded from scored totals.",
            "Aborted-run recorded usage may omit requests from the interrupted in-flight trial.",
            "Real Google/Linear/mobile voice behavior was simulated or excluded.",
            "The scored fixture read validator returned generic errors; production returns specific schema errors. A separate paired diagnostic checks this mismatch without replacing scores.",
            "The constraint fixture availability adapter models a connected calendar, but a fallback calendar_list sees no real connection. Its free/busy evidence is inconsistent across the two paths.",
            "The queued-write fixture omits production create-tool internal status polling; its separate-status-call requirement is implementation-specific.",
        ],
    }


def report_html(current, comparison):
    result = render(current, comparison["after"])
    result = result.replace(
        "samples of 24 authored scenarios", "samples of 20 preselected authored scenarios"
    )
    result = result.replace("expert-fixtures-", "tool-refinement-fixtures-")
    result = result.replace("expert-agents-", "tool-refinement-agents-")
    result = result.replace("../EXPERT_AGENT_EVALUATION.md", "../TOOL_REFINEMENT_EVAL_PLAN.md")
    reviewed = "; ".join(
        html.escape(model) + " " + str(info["after_adjudicated"]["task_success_after_review"]) + "/60"
        for model, info in comparison["models"].items()
    )
    result = result.replace(
        '<div class="metrics">',
        "<p><strong>Reviewed workflow completions: " + reviewed + ".</strong> "
        "The cards and scenario scores below retain the original automatic grades, including three explicit phrase-grader false negatives. "
        "The before/after section reports both versions. <a href='../TOOL_REFINEMENT_RESULTS.md'>Read findings and fixture limitations</a>.</p>"
        '<div class="metrics">',
    )
    sections = []
    for model, info in comparison["models"].items():
        a, b = info["before"], info["after"]
        rows = [
            ("Original automated completions", a["successful_trials"], b["successful_trials"]),
            (
                "Completions after explicit grader review",
                info["before_adjudicated"]["task_success_after_review"],
                info["after_adjudicated"]["task_success_after_review"],
            ),
            (
                "State plus supported reply (reviewed)",
                info["before_adjudicated"]["task_and_factual_after_review"],
                info["after_adjudicated"]["task_and_factual_after_review"],
            ),
            (
                "Mean task time (seconds)",
                round(a["task_latency_all_attempts"]["mean"], 2),
                round(b["task_latency_all_attempts"]["mean"], 2),
            ),
            ("Reported input tokens", a["usage"]["input_tokens"], b["usage"]["input_tokens"]),
            (
                "Reported output tokens (includes reasoning)",
                a["usage"]["output_tokens"],
                b["usage"]["output_tokens"],
            ),
            (
                "Uncached-rate equivalent (USD)",
                round(a["usage"]["uncached_rate_equivalent_usd"], 4),
                round(b["usage"]["uncached_rate_equivalent_usd"], 4),
            ),
        ]
        body = "".join(
            "<tr><th scope='row'>"
            + html.escape(label)
            + "</th><td>"
            + str(old)
            + "</td><td>"
            + str(new)
            + "</td></tr>"
            for label, old, new in rows
        )
        sections.append(
            f"<h3>{html.escape(model)}</h3><table><thead><tr><th>Metric</th><th>Before</th><th>After</th></tr></thead><tbody>{body}</tbody></table>"
        )
    block = (
        "<h2>Same questions, before and after</h2>"
        "<p>Each column covers the same 60 trials per model. The four excluded original scenarios do not enter either denominator. "
        "Source: retained September 13 baseline and this complete follow-up. These are development regressions, not held-out tests. Explicit false-negative corrections are from an unblinded trace review; original scores above are unchanged. Timing is observed across separate runs; costs exclude diagnostic overhead.</p>"
        + "".join(sections)
        + "<p><a href='tool-refinement-comparison-2026-09-13.json'>Machine-readable comparison</a> · "
        "<a href='../TOOL_DESIGN_RESEARCH.md'>Research and implementation</a> · "
        "<a href='../TOOL_REFINEMENT_RESULTS.md'>Findings and fixture limitations</a> · "
        "<a href='tool-refinement-aborted-v1-2026-09-13.json.gz'>Interrupted schema-failure attempt</a></p>"
    )
    return result.replace("<h2>Completion by scenario</h2>", block + "<h2>Completion by scenario</h2>")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("current", type=Path)
    parser.add_argument("--baseline", type=Path, default=Path("docs/evals/expert-agents-2026-09-13.json.gz"))
    parser.add_argument(
        "--before-review", type=Path, default=Path("docs/evals/expert-response-review-2026-09-13.json")
    )
    parser.add_argument("--after-review", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, default=Path("docs/evals"))
    args = parser.parse_args()
    current = read(args.current)
    result = compare(read(args.baseline), current, read(args.before_review), read(args.after_review))
    raw = args.current.read_bytes()
    if args.current.suffix == ".gz":
        raw = gzip.decompress(raw)
    result["raw_report_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    prefix = args.output_directory / "tool-refinement"
    Path(str(prefix) + "-comparison-2026-09-13.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    Path(str(prefix) + "-report-2026-09-13.html").write_text(report_html(current, result), encoding="utf-8")
    Path(str(prefix) + "-agents-2026-09-13.json.gz").write_bytes(gzip.compress(raw, mtime=0))
    print(
        json.dumps(
            {
                m: {"before": v["before"]["successful_trials"], "after": v["after"]["successful_trials"]}
                for m, v in result["models"].items()
            }
        )
    )


if __name__ == "__main__":
    main()
