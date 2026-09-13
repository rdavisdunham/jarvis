"""Runner invariants: isolation, honest scoring, fairness, and provider accounting."""

import importlib.util
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

spec = importlib.util.spec_from_file_location(
    "evaluate_expert_agents", Path(__file__).resolve().parents[1] / "scripts/evaluate_expert_agents.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def profile():
    return runner.agent_models.AgentModel(
        "openai",
        "gpt-5.6-luna",
        "Luna",
        "https://api.openai.com/v1/responses",
        "synthetic-secret",
        8192,
        profile="luna",
        api="responses",
        reasoning_effort="low",
    )


def fake_dns(_host, port, *args, **kwargs):
    return [(2, 1, 6, "", ("203.0.113.20", int(port)))]


def test_schedule_pairs_identical_fixtures_and_balances_first_model():
    first = runner.scenario_order(["d", "a", "c", "b"], 3, runner.MODELS, 7)
    assert first == runner.scenario_order(["b", "c", "a", "d"], 3, runner.MODELS, 7)
    assert len(first) == 24
    starts = []
    for index in range(0, len(first), 2):
        a, b = first[index : index + 2]
        assert (a["case"], a["repeat"]) == (b["case"], b["repeat"])
        assert {a["model"], b["model"]} == set(runner.MODELS)
        starts.append(a["model"])
    assert starts.count(runner.MODELS[0]) == starts.count(runner.MODELS[1])


@pytest.mark.parametrize("name", ["jarvis", "jarvis_expert_eval_", "jarvis_expert_eval_abcd"])
def test_refuses_non_disposable_database(name):
    with pytest.raises(ValueError):
        runner.validate_disposable_database(name, name)


def test_disposable_validation_binds_to_this_invocation():
    name = runner.DATABASE_PREFIX + "a" * 32
    runner.validate_disposable_database(name, name)
    with pytest.raises(ValueError):
        runner.validate_disposable_database(name, runner.DATABASE_PREFIX + "b" * 32)


def test_environment_restores_existing_overrides_even_after_failure(monkeypatch):
    monkeypatch.setenv("JARVIS_DATABASE_URL", "postgresql+psycopg://test:pw@localhost/original")
    monkeypatch.setenv("JARVIS_COST_TRACKING_ENABLED", "true")
    with (
        pytest.raises(RuntimeError),
        runner.isolated_environment("postgresql+psycopg://test:pw@localhost/disposable") as settings,
    ):
        assert settings.database_url.endswith("/disposable")
        assert settings.cost_tracking_enabled is False
        raise RuntimeError("synthetic setup failure")
    assert runner.os.environ["JARVIS_DATABASE_URL"].endswith("/original")
    assert runner.os.environ["JARVIS_COST_TRACKING_ENABLED"] == "true"


def test_environment_restores_absent_overrides(monkeypatch):
    monkeypatch.delenv("JARVIS_DATABASE_URL", raising=False)
    monkeypatch.delenv("JARVIS_COST_TRACKING_ENABLED", raising=False)
    with runner.isolated_environment("postgresql+psycopg://test:pw@localhost/disposable"):
        pass
    assert "JARVIS_DATABASE_URL" not in runner.os.environ
    assert "JARVIS_COST_TRACKING_ENABLED" not in runner.os.environ


def test_instruction_clock_does_not_freeze_runtime_clock():
    assert runner.InstructionDateTime.now(UTC) == datetime(2030, 1, 14, 15, tzinfo=UTC)
    assert runner.InstructionDateTime.now().hour == 9
    assert datetime.now(UTC).year != 2030


@pytest.mark.parametrize(
    ("status", "code", "expected"),
    [
        (401, None, "authentication_or_model_access"),
        (404, "model_not_found", "authentication_or_model_access"),
        (429, None, "quota_or_rate_limit"),
        (400, "invalid_api_key", "authentication_or_model_access"),
        (500, None, None),
    ],
)
def test_provider_stop_does_not_retry_authentication_or_quota(status, code, expected):
    assert runner.provider_stop_reason([{"status": status, "error_code": code}]) == expected


def test_success_recovery_safety_and_missing_are_distinct():
    clean = [{"status": "succeeded", "tool_errors": []}]
    recovered = [{"status": "partial", "tool_errors": [{"error": "REVISION_CONFLICT"}]}]
    assert runner.classify({"saved": True}, clean, []) == "clean_success"
    assert runner.classify({"saved": True}, recovered, []) == "recovered_success"
    assert runner.classify({"saved": False}, clean, []) == "task_failure"
    assert runner.classify({"saved": True}, clean, ["wrong_record"]) == "safety_failure"
    assert runner.classify({"saved": True}, clean, [], {"type": "RuntimeError"}) == "infrastructure_error"
    results = [
        {"model": "luna", "outcome": "skipped", "task_success": False, "turns": []},
        {"model": "luna", "outcome": "clean_success", "task_success": True, "turns": clean, "seconds": 2},
    ]
    assert runner.summarize(results, ["luna"])["luna"]["task_successes"] == 1
    assert runner.summarize(results, ["luna"])["luna"]["skipped"] == 1


@pytest.mark.asyncio
async def test_only_selected_post_and_model_can_leave_process(monkeypatch):
    monkeypatch.setattr(runner.socket, "getaddrinfo", fake_dns)
    calls, blocked, transport_calls = [], [], []

    async def respond(request):
        transport_calls.append(request)
        return httpx.Response(
            200,
            json={
                "model": "gpt-5.6-luna-verified",
                "status": "completed",
                "usage": {
                    "input_tokens": 30,
                    "output_tokens": 5,
                    "output_tokens_details": {"reasoning_tokens": 2},
                },
                "output": [{"type": "reasoning", "encrypted_content": "DO_NOT_RECORD"}],
            },
        )

    with runner.outbound_guard(profile(), "postgresql://localhost:5432/test", calls, blocked):
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            await client.post(profile().endpoint, json={"model": profile().model})
            with pytest.raises(ValueError):
                await client.get(profile().endpoint)
            with pytest.raises(ValueError):
                await client.post("https://example.com", json={"model": profile().model})
            with pytest.raises(ValueError):
                await client.post(profile().endpoint, json={"model": "other-model"})
    assert len(transport_calls) == 1
    assert len(calls) == 1
    assert calls[0]["usage"]["output_tokens_details"]["reasoning_tokens"] == 2
    assert calls[0]["response_model"] == "gpt-5.6-luna-verified"
    assert "DO_NOT_RECORD" not in str(calls)
    assert [item["kind"] for item in blocked] == ["http_endpoint", "http_endpoint", "provider_model"]


@pytest.mark.asyncio
async def test_redirects_cannot_bypass_endpoint_guard(monkeypatch):
    monkeypatch.setattr(runner.socket, "getaddrinfo", fake_dns)
    transport_calls = []

    async def respond(request):
        transport_calls.append(str(request.url))
        return httpx.Response(307, headers={"location": "https://unexpected.example/collect"})

    with runner.outbound_guard(profile(), "postgresql://localhost:5432/test", [], []):
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond), follow_redirects=True) as client:
            response = await client.post(profile().endpoint, json={"model": profile().model})
    assert response.status_code == 307
    assert transport_calls == [profile().endpoint]


@pytest.mark.asyncio
async def test_provider_failure_is_recorded_once_without_hidden_retry(monkeypatch):
    monkeypatch.setattr(runner.socket, "getaddrinfo", fake_dns)
    calls = []

    async def timeout(request):
        raise httpx.ReadTimeout("synthetic timeout", request=request)

    with runner.outbound_guard(profile(), "postgresql://localhost:5432/test", calls, []):
        async with httpx.AsyncClient(transport=httpx.MockTransport(timeout)) as client:
            with pytest.raises(httpx.ReadTimeout):
                await client.post(profile().endpoint, json={"model": profile().model})
    assert len(calls) == 1 and calls[0]["exception_type"] == "ReadTimeout"


def test_socket_and_sync_http_cannot_bypass_guard(monkeypatch):
    monkeypatch.setattr(runner.socket, "getaddrinfo", fake_dns)
    blocked = []
    with runner.outbound_guard(profile(), "postgresql://localhost:5432/test", [], blocked):
        assert runner.socket.getaddrinfo(b"api.openai.com", 443)[0][4] == ("203.0.113.20", 443)
        with (
            httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200))) as client,
            pytest.raises(ValueError),
        ):
            client.get(profile().endpoint)
        with runner.socket.socket() as sock, pytest.raises(ValueError):
            sock.connect(("198.51.100.4", 443))
        with pytest.raises(ValueError):
            runner.socket.getaddrinfo("unexpected.example", 443)
    assert {row["kind"] for row in blocked} == {"non_model_http", "socket_destination", "dns_destination"}


@pytest.mark.asyncio
async def test_stages_are_checked_before_next_turn_and_full_history_results_reach_grader(monkeypatch):
    fixture = {
        "case": "example",
        "fixture_hash": "fixture-hash",
        "conversation": "synthetic-conversation",
        "prompts": ["First", "Second"],
        "memory_context": "Synthetic memory",
        "context": {},
        "tools": [],
    }
    events = []

    async def fake_chat(*args):
        prompt = args[4]
        events.append(("chat", prompt))
        return {"message": "answer " + prompt, "status": "succeeded", "tool_errors": []}

    def stage(_fixture, index, result):
        events.append(("stage", index))
        return {"stage_saved": result["message"].startswith("answer")}

    def grade(_fixture, turns):
        assert [turn["message"] for turn in turns] == ["answer First", "answer Second"]
        return {"checks": {"final_saved": True}, "safety_violations": [], "behavior_flags": []}

    module = SimpleNamespace(
        OWNER="owner",
        DEVICE="device",
        seed_case=lambda *_: fixture,
        stage_checks=stage,
        grade_case=grade,
        invoke_tool=None,
    )
    monkeypatch.setattr(runner, "outbound_guard", lambda *_: nullcontext())
    monkeypatch.setattr(runner.conversation, "chat", fake_chat)
    result = await runner.evaluate_case(
        module,
        profile(),
        {"case": "example", "repeat": 1, "model": "luna"},
        "postgresql://localhost/test",
    )
    assert events == [("chat", "First"), ("stage", 0), ("chat", "Second"), ("stage", 1)]
    assert result["outcome"] == "clean_success"
    assert result["fixture_hash"] == "fixture-hash"


def test_exception_redaction_covers_keys_and_database_passwords():
    error = ValueError("https://owner:db-password@host/db Bearer secret-token and api-key")
    safe = runner.safe_exception(error, ["api-key"])
    assert "db-password" not in safe["message"]
    assert "secret-token" not in safe["message"]
    assert "api-key" not in safe["message"]


@pytest.mark.asyncio
async def test_run_records_all_provider_skips_and_removes_only_its_database(monkeypatch, tmp_path):
    fake_cases = SimpleNamespace(CASES={"example": {"title": "Synthetic", "category": "Test"}})
    original_import = runner.importlib.import_module
    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda name: fake_cases if name == "expert_eval_cases" else original_import(name),
    )
    monkeypatch.setattr(runner.agent_models, "catalog", lambda: {"luna": profile(), "gemini": profile()})
    original_environment = {
        key: runner.os.environ.get(key) for key in ("JARVIS_DATABASE_URL", "JARVIS_COST_TRACKING_ENABLED")
    }
    calls, created_names = [], []

    async def evaluate(_cases, _profile, spec, database_url):
        calls.append(spec)
        created_names.append(runner.make_url(database_url).database)
        unavailable = spec["model"] == runner.MODELS[0]
        return {
            **spec,
            "outcome": "task_failure" if unavailable else "clean_success",
            "task_success": not unavailable,
            "clean_completion": not unavailable,
            "turns": [],
            "checks": {"provider_completed": not unavailable},
            "seconds": 0,
            "provider_calls": [{"status": 401 if unavailable else 200}],
        }

    monkeypatch.setattr(runner, "evaluate_case", evaluate)
    report = await runner.run(3, tmp_path / "results.json", list(runner.MODELS), cases=["example"])
    assert len(report["results"]) == 6
    assert sum(row["outcome"] == "skipped" for row in report["results"]) == 2
    assert len(calls) == 4
    assert report["database_removed"] is True
    assert {key: runner.os.environ.get(key) for key in original_environment} == original_environment
    assert len(set(created_names)) == 1
    with runner.engine().connect() as db:
        exists = db.exec_driver_sql(
            "SELECT 1 FROM pg_database WHERE datname = %s", (created_names[0],)
        ).first()
    assert exists is None


@pytest.mark.asyncio
async def test_run_fatal_error_keeps_checkpoint_and_restores_environment(monkeypatch, tmp_path):
    fake_cases = SimpleNamespace(CASES={"example": {"title": "Synthetic", "category": "Test"}})
    original_import = runner.importlib.import_module
    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda name: fake_cases if name == "expert_eval_cases" else original_import(name),
    )
    monkeypatch.setattr(runner.agent_models, "catalog", lambda: {"luna": profile(), "gemini": profile()})
    original_database = runner.get_settings().database_url

    async def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic runner interruption")

    monkeypatch.setattr(runner, "evaluate_case", fail)
    output = tmp_path / "failed.json"
    with pytest.raises(RuntimeError):
        await runner.run(1, output, [runner.MODELS[0]], cases=["example"])
    saved = runner.json.loads(output.read_text())
    assert saved["database_removed"] is True
    assert saved["state"] == "failed"
    assert saved["fatal_exception"]["type"] == "RuntimeError"
    assert runner.get_settings().database_url == original_database
