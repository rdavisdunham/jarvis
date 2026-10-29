"""Outcome eval for the run judge (decision 6: auto_complete stays off until this is acceptable).

    uv run --no-sync python -m evals.agent_control.run            # stubbed model (default, free)
    uv run --no-sync python -m evals.agent_control.run --live     # Luna through the extraction helper (paid)

Each case is fixed evidence plus a label. The judge's hard rules run either way; the stub stands in
for the model with a plausible (sometimes deliberately fooled) answer. Reported: false completions
(met where the label is not met), stale completions (false completions on stale or changed-source
evidence), and per-case results. Loop cases check the no-progress rule on a verdict sequence.
"""

import argparse
import json
import sys
from pathlib import Path

from jarvis import agent_judge as judge
from jarvis import run_coordinator as coordinator

CASES = Path(__file__).with_name("cases.json")
STALE = {"stale_test_result", "source_change_mid_run"}


def accepts(expect, verdict):
    return verdict in ("not_met", "needs_you") if expect == "not_met_or_needs_you" else verdict == expect


def evaluate(cases=None, model_for=None):
    """model_for(case) → a callable(evidence) returning a Verdict; the default replays the case's stub."""
    cases = cases if cases is not None else json.loads(CASES.read_text())["cases"]
    model_for = model_for or (lambda case: lambda evidence: judge.Verdict.model_validate(case["stub"]))
    results = []
    for case in cases:
        if case.get("kind") == "loop":
            got = coordinator.stalled(list(reversed(case["verdicts"])))
            results.append({"id": case["id"], "category": case["category"], "ok": got == case["expect_stalled"],
                            "stalled": got})
            continue
        try:
            final = judge.decide(case["evidence"], model_for(case))
            verdict = final["verdict"]
        except judge.NonVerdict as exc:
            final, verdict = {"rules": [], "non_verdict": exc.kind}, "non_verdict"
        false_completion = verdict == "met" and case["expect"] != "met"
        results.append({"id": case["id"], "category": case["category"], "expect": case["expect"], "verdict": verdict,
                        "ok": accepts(case["expect"], verdict), "rules": final.get("rules", []),
                        "false_completion": false_completion,
                        "stale_completion": false_completion and case["category"] in STALE})
    return {
        "cases": len(results), "passed": sum(r["ok"] for r in results),
        "false_completions": sum(r.get("false_completion", False) for r in results),
        "stale_completions": sum(r.get("stale_completion", False) for r in results),
        "missed_completions": sum(1 for r in results if r.get("expect") == "met" and r.get("verdict") != "met"),
        "results": results,
    }


def live_model(owner):
    def factory(case):
        return lambda evidence: judge.ask(owner, evidence)
    return factory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--live", action="store_true", help="ask the real judge model (costs money)")
    parser.add_argument("--owner", default="davin", help="budget owner for --live")
    args = parser.parse_args(argv)
    report = evaluate(model_for=live_model(args.owner) if args.live else None)
    report["mode"] = "live" if args.live else "stub"
    print(json.dumps(report, indent=2))
    return 0 if report["false_completions"] == 0 and report["stale_completions"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
