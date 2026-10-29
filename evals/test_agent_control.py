"""The run judge's outcome eval with the stubbed model: no false or stale completions on the labelled suite."""

from evals.agent_control.run import evaluate


def test_stubbed_outcome_eval_has_no_false_or_stale_completions():
    report = evaluate()
    failed = [r for r in report["results"] if not r["ok"]]
    assert not failed, failed
    assert report["false_completions"] == 0 and report["stale_completions"] == 0
    assert report["cases"] >= 14
    categories = {r["category"] for r in report["results"]}
    assert categories >= {"correct_completion", "plausible_false_success", "missing_criteria", "missing_evidence",
                          "stale_test_result", "ambiguous_blocker", "repeated_no_progress", "source_change_mid_run",
                          "recovery"}


def test_without_hard_rules_the_fooled_stub_would_complete_falsely():
    """The suite has teeth: the stubs that say met on bad evidence are caught by the rules, not by the stub."""
    report = evaluate()
    caught = [r for r in report["results"] if r.get("rules")]
    assert len(caught) >= 5
