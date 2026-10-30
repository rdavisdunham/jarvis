"""The backend model eval (scripts/model_eval.py, evals/models) and the per-job model map (jarvis.model_map).

No test here reaches a provider: the eval runs on its deterministic fake, and production calls are monkeypatched.
"""

import json
import re
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import model_eval
from jarvis import (
    agent_judge,
    agent_models,
    agent_sessions,
    cost_features,
    memory_learning,
    model_map,
    routing,
)
from jarvis.config import get_settings

from evals.models import schema as case_schema

PREFS = {"timezone": "America/Chicago", "default_reminder_hour": 10}


# ---------------------------------------------------------------- case files


def test_case_files_match_the_schema_and_cover_every_job():
    assert case_schema.SCHEMA.read_text() == case_schema.schema_text(), \
        "evals/models/schema.json is stale: run `python -m evals.models.schema`"
    loaded = case_schema.load()
    assert set(loaded) == set(case_schema.JOBS)
    for job, (spec, cases) in loaded.items():
        assert len(cases) >= 5, job
        assert spec.prompt_source
        if spec.grader == "text":
            assert spec.rubric, f"{job} needs a rubric for the grader model"
    # The judge reuses the agent_control outcome cases (minus loop cases, which test a pure rule).
    control = json.loads((ROOT / "evals/agent_control/cases.json").read_text())["cases"]
    assert {c["id"] for c in loaded["run_judge"][1]} == {c["id"] for c in control if c.get("kind") != "loop"}


def test_validator_rejects_a_stub_that_production_would_not_parse(tmp_path):
    data = json.loads((case_schema.CASES / "session_summary.json").read_text())
    data["cases"][0]["stub"] = {"summary": "ok", "extra": 1}  # Summary forbids extra keys.
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="login-fix-with-pr"):
        case_schema.validate_file(bad)
    data["cases"][0]["stub"] = {"summary": "ok"}
    data["cases"][0]["expect"] = {"max_chars": 300, "must_mention": "login"}  # Groups are lists of terms.
    bad.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        case_schema.validate_file(bad)


class Forced(model_eval.StubProvider):
    def __init__(self, kind):
        self.kind = kind

    def plan(self, p, case, repeat):
        return self.kind if self.kind != "wrong" or case.get("stub_wrong") is not None else "skip"


def run_all(kind, profile="luna", grader=None):
    runner = model_eval.Run(Forced(kind), max_requests=10_000, max_usd=1, grader=grader)
    rows = []
    for job, (spec, cases) in case_schema.load().items():
        for case in cases:
            if runner.provider.plan(None, case, 0) == "skip":
                continue
            rows.append(runner.case(job, spec, case, model_eval.profile(profile), 0))
    return rows


@pytest.mark.parametrize("profile", ["luna", "luna-fast", "haiku-5.5", "deepseek"])
def test_every_recorded_answer_passes_its_own_grader(profile):
    """The stubs are correct answers in production shape, through each wire format and adapter."""
    rows = run_all("ok", profile)
    failed = [(r["job"], r["case"], r["outcome"], r["detail"]) for r in rows if r["outcome"] != "pass"]
    assert not failed


def test_the_graders_have_teeth():
    """Every recorded wrong answer fails; refusals and broken output are non-verdicts, never passes."""
    wrong = run_all("wrong")
    graded = [r for r in wrong if r["job"] != "run_judge"]
    assert graded and all(r["outcome"] == "fail" for r in graded), [r for r in graded if r["outcome"] != "fail"]
    assert {r["job"] for r in wrong} == set(case_schema.JOBS)
    assert all(r["outcome"] == "refusal" for r in run_all("refusal", "haiku-5.5"))
    assert all(r["outcome"] == "invalid" for r in run_all("invalid", "sonnet-5.5"))
    judged = [r for r in wrong if r["job"] == "run_judge"]
    # The judge's hard rules overrule many fooled verdicts (that is their job), so a wrong model answer can still
    # end right; the model's own verdict is wrong every time and the summary reports it separately.
    assert all(r["outcome"] in ("pass", "fail") and r["detail"]["model_ok"] is False for r in judged)
    assert any(r["outcome"] == "fail" for r in judged) and any(r["outcome"] == "pass" for r in judged)
    rows = model_eval.summarize(judged)
    assert rows[0]["model_pass_rate"] == 0 and rows[0]["pass_rate"] > 0


def test_rubric_grader_runs_only_on_text_jobs_and_counts_separately():
    runner = model_eval.Run(Forced("ok"), max_requests=10_000, max_usd=1, grader="sonnet-5.5")
    spec, cases = case_schema.load(("session_summary",))["session_summary"]
    row = runner.case("session_summary", spec, cases[0], model_eval.profile("haiku-5.5"), 0)
    assert row["outcome"] == "pass" and row["detail"]["grader"]["passed"] is True
    assert runner.requests == 2 and runner.grader_spent > 0 and runner.spent > 0


# ---------------------------------------------------------------- the stub run


def test_stub_run_writes_results_and_summary(tmp_path):
    assert model_eval.main(["--stub", "--quiet", "--out", str(tmp_path), "--date", "2026-10-30"]) == 0
    report = json.loads((tmp_path / "2026-10-30-stub.json").read_text())
    summary = (tmp_path / "2026-10-30-stub.md").read_text()
    assert report["mode"] == "stub" and report["stopped"] is None
    # Eri chat runs on the Settings choice: Luna at the configured interactive tier (Fast by default).
    tier = "luna-fast" if get_settings().agent_service_tier == "fast" else "luna"
    assert report["baselines"] == {**{job: "luna" for job in case_schema.JOBS}, "eri_chat": tier}
    assert set(report["decisions"]) == set(case_schema.JOBS)
    rows = {(r["job"], r["profile"]): r for r in report["summary"]}
    for job in case_schema.JOBS:
        for pid in (report["baselines"][job], "haiku-5.5", "sonnet-5.5", "deepseek", "kimi", "minimax"):
            row = rows[(job, pid)]
            assert row["runs"] == report["jobs"][job]["cases"]
            assert 0 <= row["pass_rate"] <= 1 and 0 <= row["non_verdict_rate"] <= 1
            assert row["latency_ms_p50"] is not None
    # Today's model replays every recorded answer; placeholders have no price, so no cost and no decision.
    assert all(rows[(job, report["baselines"][job])]["pass_rate"] == 1 for job in case_schema.JOBS)
    assert rows[("session_summary", "deepseek")]["cost_per_completed"] is None
    assert rows[("session_summary", "haiku-5.5")]["cost_per_completed"] > 0
    assert "STUB" in summary and "none is a measurement" in summary and "## Decision" in summary
    # Deterministic: the same command writes the same report.
    model_eval.main(["--stub", "--quiet", "--out", str(tmp_path / "again"), "--date", "2026-10-30"])
    again = json.loads((tmp_path / "again/2026-10-30-stub.json").read_text())
    assert again["summary"] == report["summary"] and again["decisions"] == report["decisions"]


def test_request_ceiling_stops_the_run_with_partial_results(tmp_path):
    code = model_eval.main(["--stub", "--quiet", "--out", str(tmp_path), "--max-requests", "7", "--jobs", "run_judge"])
    report = json.loads(next(tmp_path.glob("*-stub.json")).read_text())
    assert code == 2 and "request ceiling" in report["stopped"] and len(report["results"]) == 7


def test_a_real_run_skips_profiles_without_keys_or_models(monkeypatch):
    for name in ("OPENAI_API_KEY", "JARVIS_OPENAI_API_KEY", "ANTHROPIC_API_KEY", "MODEL_EVAL_KIMI_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(model_eval, "api_key", lambda p, environ=None: "" if p.provider in ("openai", "anthropic")
                        else "set")
    _baselines, per_job, skipped = model_eval.select_profiles(["haiku-5.5", "kimi"], ("session_summary",), stub=False)
    assert per_job["session_summary"] == []
    assert skipped == {"luna": "no OPENAI_API_KEY", "haiku-5.5": "no ANTHROPIC_API_KEY",
                       "kimi": "placeholder: set MODEL_EVAL_KIMI_MODEL to run it"}
    monkeypatch.setenv("MODEL_EVAL_KIMI_MODEL", "kimi-k3")
    monkeypatch.setenv("MODEL_EVAL_KIMI_PRICE", "0.5,2")
    _b, per_job, skipped = model_eval.select_profiles(["kimi"], ("session_summary",), stub=False)
    [kimi] = per_job["session_summary"]
    assert kimi.model == "kimi-k3" and not kimi.placeholder
    assert model_eval.price_for(kimi)["input"] == 0.5


# ---------------------------------------------------------------- prices


def test_price_table_is_dated_and_matches_production_rates():
    for pid, p in model_map.PROFILES.items():
        price = model_eval.price_for(model_eval.profile(pid, {}), {})
        if p.placeholder:
            assert price is None, pid
            continue
        assert price, f"{pid} has no price"
        assert date.fromisoformat(price["checked"]) <= date(2026, 12, 31) and price["source"]
    # Luna's rates are the ones the production budget ledger charges.
    for tier, key in (("default", "gpt-5.6-luna"), ("fast", "gpt-5.6-luna@fast")):
        agent = agent_models.catalog()["luna"]
        incoming, outgoing = agent.rates(service_tier=tier)
        assert (model_eval.PRICES[key]["input"], model_eval.PRICES[key]["output"]) == (incoming, outgoing)
    # The task's reference prices (2026-10-28).
    assert (model_eval.PRICES["claude-haiku-5-5"]["input"], model_eval.PRICES["claude-haiku-5-5"]["output"]) == (0.10, 0.50)
    assert (model_eval.PRICES["gpt-5.6-luna"]["input"], model_eval.PRICES["gpt-5.6-luna"]["output"]) == (0.20, 1.20)


def test_cost_uses_cached_and_long_prompt_rates():
    haiku, luna = model_eval.PRICES["claude-haiku-5-5"], model_eval.PRICES["gpt-5.6-luna"]
    assert model_eval.cost(haiku, {"input_tokens": 100_000, "output_tokens": 1_000_000}) == pytest.approx(0.51)
    # Above 100K input tokens Haiku's long-prompt rates apply to the whole request.
    assert model_eval.cost(haiku, {"input_tokens": 200_000, "output_tokens": 0}) == pytest.approx(0.10)
    assert model_eval.cost(luna, {"input_tokens": 1000, "cached_input_tokens": 1000, "output_tokens": 0}) \
        == pytest.approx(0.00002)
    assert model_eval.cost(None, {"input_tokens": 1, "output_tokens": 1}) is None
    # The eval's Luna estimate agrees with the ledger's for an ordinary request.
    agent = agent_models.catalog()["luna"]
    usage = {"prompt_tokens": 5000, "completion_tokens": 700, "prompt_tokens_details": {"cached_tokens": 2000}}
    assert model_eval.cost(luna, {"input_tokens": 5000, "cached_input_tokens": 2000, "output_tokens": 700}) \
        == pytest.approx(agent.usage_cost(usage))


# ---------------------------------------------------------------- the model map


def test_model_map_defaults_keep_todays_model_for_every_job():
    assert set(model_map.MODEL_MAP) == set(model_map.JOBS) == set(model_map.DEFAULTS)
    assert model_map.RULES == ()
    for job, spec in model_map.JOBS.items():
        chosen = model_map.route(job, 10**6)
        assert chosen.reason == "map" and chosen.profile == model_map.MODEL_MAP[job] == model_map.DEFAULTS[job]
        assert chosen.cost_feature == spec.cost_feature
        if spec.path == "extraction":
            p = model_map.PROFILES[chosen.profile]
            # extraction_request: Luna, Responses API, low reasoning, Standard tier.
            assert (p.model, p.api, p.reasoning_effort, p.service_tier) == \
                (memory_learning.EXTRACTION_MODEL, "openai_responses", "low", "default")
        else:
            assert chosen.profile == model_map.SELECTED
            assert model_map.agent_model(job, PREFS) == agent_models.selected(PREFS)
    for job in case_schema.JOBS:
        assert job in model_map.JOBS


def test_router_rules_by_job_and_prompt_length():
    rules = (model_map.Rule("sonnet-5.5", min_prompt_tokens=100_001, name="long"),
             model_map.Rule("haiku-5.5", jobs=("session_summary",), max_prompt_tokens=100_000, name="short-summaries"))
    assert model_map.route("session_summary", 2_000, rules=rules).profile == "haiku-5.5"
    assert model_map.route("session_summary", 150_000, rules=rules).reason == "rule:long"
    assert model_map.route("run_judge", 2_000, rules=rules).profile == "luna"
    assert model_map.route("run_judge", 2_000, rules=rules, model_map={"run_judge": "sonnet-5.5"}).profile == "sonnet-5.5"
    with pytest.raises(ValueError):
        model_map.route("run_judge", rules=(model_map.Rule("nope"),))
    with pytest.raises(ValueError):
        model_map.route("unknown_job")
    assert model_map.estimate_tokens("x" * 400) == 100


def test_profile_request_is_extraction_request_by_default(monkeypatch):
    calls = []
    monkeypatch.setattr(memory_learning, "extraction_request", lambda *a, **k: calls.append(a) or {"ok": True})
    payload = agent_sessions.summary_request({"title": "x"})
    for job, spec in model_map.JOBS.items():
        if spec.path != "extraction":
            with pytest.raises(ValueError):
                memory_learning.profile_request("owner", job, payload, 0.02)
            continue
        assert memory_learning.profile_request("owner", job, payload, 0.02) == {"ok": True}
    assert all(a[1:4] == ("chat/completions", payload, memory_learning.EXTRACTION_MODEL) for a in calls)
    # A route to any other model is refused in production (the budget ledger prices Luna only).
    monkeypatch.setattr(model_map, "RULES", (model_map.Rule("haiku-5.5", jobs=("session_summary",)),))
    with pytest.raises(ValueError, match="not enabled"):
        memory_learning.profile_request("owner", "session_summary", payload, 0.02)
    monkeypatch.setattr(model_map, "RULES", (model_map.Rule("haiku-5.5", jobs=("eri_chat",)),))
    with pytest.raises(ValueError, match="not enabled"):
        model_map.agent_model("eri_chat", PREFS)


def test_attributed_records_under_the_jobs_cost_feature():
    assert all(spec.cost_feature in cost_features.LABELS for spec in model_map.JOBS.values())
    with model_map.attributed("routing_review"):
        assert cost_features.current_feature.get() == "organization_learning"
    assert cost_features.current_feature.get() == "unattributed"


def test_each_jobs_production_call_records_under_its_mapped_feature(monkeypatch):
    seen = {}

    def fake(owner, path, payload, model, allowance, timeout=45):
        seen["feature"] = cost_features.current_feature.get()
        return {"choices": [{"message": {"content": json.dumps(stub)}}], "usage": {}}

    monkeypatch.setattr(memory_learning, "extraction_request", fake)
    stub = {"summary": "Fixed it."}
    assert agent_sessions.summarize_text("owner", {"title": "x"}) == "Fixed it."
    assert seen["feature"] == model_map.JOBS["session_summary"].cost_feature
    stub = json.loads((case_schema.ROOT / "evals/agent_control/cases.json").read_text())["cases"][0]["stub"]
    agent_judge.ask("owner", {"plan": {}})
    assert seen["feature"] == model_map.JOBS["run_judge"].cost_feature
    # The others are worker jobs that need a database: their decorator is checked in the source instead.
    for spec in model_map.JOBS.values():
        module, _, function = spec.site.partition(".")
        path = ROOT / "apps/api/jarvis" / f"{module}.py"
        if not path.exists():
            continue  # run_summaries is not on main yet.
        source = path.read_text()
        if function:
            assert re.search(rf'@feature\("{spec.cost_feature}"\)\ndef {function}\(', source), spec.site
        else:
            assert f'feature="{spec.cost_feature}"' in source, spec.site


def test_routing_infer_sends_the_request_the_eval_sends(monkeypatch):
    captured = {}

    def fake(owner, path, payload, model, allowance, timeout=45):
        captured.update(payload=payload, model=model, allowance=allowance)
        return {"choices": [{"message": {"content": json.dumps({"candidates": []})}}]}

    monkeypatch.setattr(memory_learning, "extraction_request", fake)
    payload = {"definition": {}, "examples": [], "note_lists": [], "homes": []}
    assert routing.infer("owner", routing.Candidates, routing.REVIEW_PROMPT, payload).candidates == []
    assert captured["payload"] == routing.infer_request(routing.Candidates, routing.REVIEW_PROMPT, payload)
    assert captured["model"] == memory_learning.EXTRACTION_MODEL and captured["allowance"] == 0.08
    assert captured["payload"]["messages"][0]["content"].startswith("Find precise task or note title phrases")
    assert captured["payload"]["max_completion_tokens"] == 4000


# ---------------------------------------------------------------- provider adapters


def test_adapters_build_and_parse_each_wire_format():
    structured = agent_sessions.summary_request({"title": "x"})
    luna = model_map.PROFILES["luna"]
    body = model_map.build(luna, structured)
    # The same body extraction_request sends (plus the service tier provider_request adds).
    assert body["reasoning"] == {"effort": "low"} and body["max_output_tokens"] == 8192
    assert body["text"]["format"]["name"] == "session_summary" and body["service_tier"] == "default"
    haiku = model_map.PROFILES["haiku-5.5"]
    body = model_map.build(haiku, structured)
    assert body["system"].startswith("You write recall summaries") and body["messages"][0]["role"] == "user"
    assert body["output_config"]["format"]["type"] == "json_schema" and body["output_config"]["effort"] == "low"
    assert model_map.headers(haiku, "k")["anthropic-version"]
    refusal = model_map.parse(haiku, {"content": [], "stop_reason": "refusal", "stop_details": {"category": "cyber"},
                                      "usage": {"input_tokens": 10, "output_tokens": 0}})
    assert refusal["refusal"] == "cyber" and refusal["text"] is None
    used = model_map.parse(haiku, {"content": [{"type": "tool_use", "id": "t", "name": "task_list", "input": {}}],
                                   "stop_reason": "tool_use", "usage": {"input_tokens": 5, "output_tokens": 3,
                                                                        "cache_read_input_tokens": 100}})
    assert used["tool_calls"][0]["function"]["name"] == "task_list"
    assert used["usage"] == {"input_tokens": 105, "cached_input_tokens": 100, "output_tokens": 3}
    deepseek = model_map.PROFILES["deepseek"]
    body = model_map.build(deepseek, structured)
    assert body["response_format"] == {"type": "json_object"} and "JSON schema" in body["messages"][-1]["content"]
    chat = model_eval.chat_request(case_schema.load(("eri_chat",))["eri_chat"][0].config, "hi")
    body = model_map.build(model_map.PROFILES["luna-fast"], chat)
    assert body["service_tier"] == "fast" and {t["name"] for t in body["tools"]} >= {"task_create", "session_search"}
    assert model_map.build(deepseek, chat)["tool_choice"] == "auto"
