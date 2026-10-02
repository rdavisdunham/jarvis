import concurrent.futures
import json
from decimal import Decimal

import pytest

from scripts.app_eval.ledger import Ledger
from scripts.app_eval.spending import EvalLimit
from scripts.app_eval.registry import jobs, registry, select, validate_registry
from scripts.app_eval.reporting import aggregate
from scripts.app_eval.judge import validate_judgment


def reserve_worker(path, index):
    ledger = Ledger(path)
    try:
        return ledger.reserve(str(index), "agent", "gpt-5.6-luna", ".03")
    except EvalLimit:
        return None


def test_parallel_reservations_share_one_limit_and_survive_restart(tmp_path):
    path = tmp_path / "budget.sqlite"
    ledger = Ledger(path)
    ledger.initialize(".1", ".01")
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        identities = list(pool.map(reserve_worker, [path] * 10, range(10)))
    assert len([i for i in identities if i]) == 3
    assert Decimal(Ledger(path).snapshot()["including_uncertain_usd"]) == Decimal(".09")
    with pytest.raises(ValueError, match="Resume"):
        ledger.initialize(10, 1)


def test_settlement_is_idempotent_and_unknown_calls_are_not_forgiven(tmp_path):
    ledger = Ledger(tmp_path / "budget.sqlite")
    ledger.initialize(".1", ".01")
    known = ledger.reserve("a", "agent", "luna", ".03")
    ledger.reserve("b", "embedding", "embed", ".02")
    ledger.settle(known, ".001")
    ledger.settle(known, ".001")
    assert ledger.snapshot()["including_uncertain_usd"] == "0.021"
    with pytest.raises(ValueError, match="Conflicting"):
        ledger.settle(known, ".002")


def test_judge_sublimit_stop_and_provider_overage(tmp_path):
    ledger = Ledger(tmp_path / "budget.sqlite")
    ledger.initialize(".1", ".01")
    with pytest.raises(EvalLimit, match="judge"):
        ledger.reserve("a", "judge", "luna", ".02")
    reservation = ledger.reserve("a", "agent", "luna", ".001")
    with pytest.raises(EvalLimit, match="exceeded"):
        ledger.settle(reservation, ".002")
    assert ledger.snapshot()["estimated_usd"] == ".002".replace(".002", "0.002")
    with pytest.raises(EvalLimit, match="stopped"):
        ledger.reserve("b", "agent", "luna", ".001")


@pytest.mark.parametrize("amount", [0, -1, 11, "nan", "inf"])
def test_invalid_campaign_limits(tmp_path, amount):
    with pytest.raises(ValueError):
        Ledger(tmp_path / "budget.sqlite").initialize(amount, 0)


def test_every_catalog_entry_has_a_type_and_honest_coverage():
    assert validate_registry()["cases"] == 1001
    rows = registry()
    assert len({r["id"] for r in rows}) == 1001
    for row in rows:
        bound = {c for v in row["variants"] if v["level"] == "acceptance" for c in v["criteria"]}
        assert (row["requirement"] is None) == (bound == row["criteria_map"].keys())


def test_shared_suites_run_once_per_repeat_and_filters_validate():
    rows = select(features="task_capture")
    work = jobs(rows, ["offline"], repeats=2)
    suite = [j for j in work if j["adapter"] == "pytest" and j["target"] == "tests/test_domain.py"]
    assert len(suite) == 2 and len(suite[0]["consumers"]) == 25
    with pytest.raises(ValueError, match="Unknown"):
        select(types="made_up")


def test_component_and_related_suite_success_never_pass_acceptance():
    rows = select(cases="task_capture.01")
    work = jobs(rows, ["offline"])
    manifest = {"cases": rows, "jobs": work}
    result = aggregate(manifest, {j["id"]: {"status": "passed"} for j in work})
    assert result[0]["status"] != "passed"


def test_aggregation_preserves_failed_repeat_and_missing_criterion():
    case = {
        "id": "x",
        "title": "X",
        "type": "agent",
        "feature": "x",
        "criteria_map": {"expected.1": "Saved"},
        "requirement": None,
    }
    consumer = {"case_id": "x", "variant": "v", "level": "acceptance", "criteria": ["expected.1"]}
    manifest = {
        "cases": [case],
        "jobs": [{"id": str(i), "repeat": i, "mode": "offline", "consumers": [consumer]} for i in (1, 2)],
    }
    assert aggregate(manifest, {"1": {"status": "passed"}})[0]["status"] != "passed"
    assert (
        aggregate(manifest, {"1": {"status": "passed"}, "2": {"status": "failed"}})[0]["status"] == "failed"
    )


@pytest.mark.parametrize(
    "judgments",
    [
        [],
        [{"criterion": "other", "verdict": "met", "reason": "x"}],
        [{"criterion": "a", "verdict": "yes", "reason": "x"}],
        [{"criterion": "a", "verdict": "met", "reason": ""}],
    ],
)
def test_judge_cannot_omit_or_invent_criteria(judgments):
    with pytest.raises(ValueError):
        validate_judgment({"a": "Fact supported"}, {"judgments": judgments})


def test_stop_file_prevents_another_reservation(tmp_path):
    ledger = Ledger(tmp_path / "budget.sqlite")
    ledger.initialize()
    (tmp_path / "STOP").touch()
    with pytest.raises(EvalLimit):
        ledger.reserve("a", "agent", "luna", ".001")


def test_paid_worker_blank_isolation_values_do_not_hide_configured_key(monkeypatch):
    from scripts.app_eval.environment import settings_env, DEFAULT_URL

    monkeypatch.setattr(
        "dotenv.dotenv_values",
        lambda path: {"OPENAI_API_KEY": "synthetic-configured", "GEMINI_API_KEY": "must-not-load"},
    )
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("JARVIS_OPENAI_API_KEY", "")
    result = settings_env(DEFAULT_URL, live=True, providers=("OPENAI",))
    assert result["JARVIS_OPENAI_API_KEY"] == "synthetic-configured"
    assert result["JARVIS_GEMINI_API_KEY"] == ""
    assert settings_env(DEFAULT_URL)["JARVIS_OPENAI_API_KEY"] == ""
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-explicit")
    assert (
        settings_env(DEFAULT_URL, live=True, providers=("OPENAI",))["JARVIS_OPENAI_API_KEY"]
        == "synthetic-explicit"
    )


def test_external_judge_preserves_evidence_without_provider_calls(tmp_path, monkeypatch):
    import httpx
    from scripts.app_eval.judge import evaluate, external_grading

    def forbidden(*args, **kwargs):
        raise AssertionError("External grading must not call a provider")

    monkeypatch.setattr(httpx.Client, "post", forbidden)
    with external_grading(tmp_path):
        result = evaluate(["Saved both movies"], {"saved": ["Arrival"]})
        assert result["status"] == "needs_review"
        request = json.loads((tmp_path / result["grading_request"]).read_text())
        assert request == {
            "criteria": ["Saved both movies"],
            "evidence": {"saved": ["Arrival"]},
            "hard_failures": [],
        }
        failed = evaluate(["Saved both movies"], {}, hard_failures=["Missing movie"])
        assert failed["status"] == "failed"
        assert len(list(tmp_path.glob("external-grade-*.json"))) == 1
    from scripts.app_eval.judge import _EXTERNAL

    assert _EXTERNAL.get() is None
