import copy
import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "comparison_report", ROOT / "scripts/compare_tool_refinement.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sample():
    baseline = json.loads(
        gzip.decompress((ROOT / "docs/evals/expert-agents-2026-09-13.json.gz").read_bytes())
    )
    current = copy.deepcopy(baseline)
    exclusions = {"subtask_reparent", "cross_zone", "memory_override", "all_day_span"}
    current["manifest"]["case_metadata"] = {
        k: v for k, v in current["manifest"]["case_metadata"].items() if k not in exclusions
    }
    current["results"] = [r for r in current["results"] if r["case"] not in exclusions]
    current["planned_runs"] = 120
    return baseline, current


def test_comparison_uses_exact_sixty_trial_denominators():
    baseline, current = sample()
    matched = module.matched_baseline(baseline, current)
    assert matched["planned_runs"] == len(matched["results"]) == 120
    assert len(matched["manifest"]["case_metadata"]) == 20
    assert len(baseline["results"]) == 144
    review = json.loads((ROOT / "docs/evals/expert-response-review-2026-09-13.json").read_text())
    assert len(module.selected_review(review, matched)["reviews"]) == 120


@pytest.mark.parametrize("corruption", ["question", "duplicate", "incomplete"])
def test_comparison_rejects_invalid_pairing(corruption):
    baseline, current = sample()
    if corruption == "question":
        current["results"][0]["fixture_prompts_sha256"] = "changed"
    elif corruption == "duplicate":
        current["results"][-1] = current["results"][0]
    else:
        current["state"] = "interrupted"
    with pytest.raises(ValueError):
        module.matched_baseline(baseline, current)
