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

    async def evaluate(_cases, _profile, spec, database_url, **_kwargs):
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


def expert_cases():
    case_spec = importlib.util.spec_from_file_location(
        "expert_eval_cases", Path(__file__).resolve().parents[1] / "scripts/expert_eval_cases.py"
    )
    module = importlib.util.module_from_spec(case_spec)
    case_spec.loader.exec_module(module)
    return module


def test_tool_refinement_fixed_selection_includes_failures_and_has_exact_exclusions():
    cases = expert_cases()
    selected = runner.select_cases(cases.CASES, "tool-refinement20")
    assert len(selected) == 20
    assert set(cases.CASES) - set(selected) == set(runner.TOOL_REFINEMENT_EXCLUSIONS)
    assert {"bulk_pagination", "evidence_extraction", "dst_gap", "pending_remote", "goal_rewire"} <= set(
        selected
    )
    with pytest.raises(ValueError):
        runner.select_cases(cases.CASES, "tool-refinement20", ["cross_zone"])


def test_context_metrics_measure_actual_offered_catalog_without_retaining_payload():
    body = {
        "model": "synthetic",
        "tools": [{"type": "function", "name": "task_list", "parameters": {"type": "object"}}],
        "input": [
            {"role": "system", "content": "Synthetic instruction"},
            {"type": "reasoning", "encrypted_content": "DO_NOT_RETAIN"},
            {"role": "user", "content": "Find the task"},
        ],
    }
    metrics = runner.request_context_metrics(body, 1234)
    assert metrics["offered_tool_names"] == ["task_list"]
    assert metrics["offered_tool_count"] == 1
    assert metrics["tool_schema_bytes"] == runner.json_bytes(body["tools"])
    assert metrics["input_context_bytes"] == runner.json_bytes(body["input"])
    assert metrics["request_body_bytes"] == 1234
    assert "DO_NOT_RETAIN" not in str(metrics)


@pytest.mark.parametrize("api", ["responses", "chat_completions"])
def test_function_projection_preserves_discovery_arguments_without_native_reasoning(api):
    arguments = '{"groups":["notes"]}'
    data = {
        "output": [
            {"type": "reasoning", "encrypted_content": "SECRET_NATIVE"},
            {"type": "function_call", "name": "tools_load", "call_id": "call-1", "arguments": arguments},
        ],
        "choices": [
            {
                "message": {
                    "extra_content": {"thought_signature": "SECRET_NATIVE"},
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "function": {"name": "tools_load", "arguments": arguments},
                            "extra_content": {"thought_signature": "SECRET_NATIVE"},
                        }
                    ],
                },
            }
        ],
    }
    calls = runner.requested_functions(data, api)
    assert calls == [{"name": "tools_load", "call_id": "call-1", "arguments": {"groups": ["notes"]}}]
    assert "SECRET_NATIVE" not in str(calls)


@pytest.mark.asyncio
async def test_discovery_observer_preserves_production_loader_and_records_outcome(monkeypatch):
    fixture = {
        "case": "example",
        "fixture_hash": "fixture-hash",
        "conversation": "synthetic-conversation",
        "prompts": ["Read the note"],
        "memory_context": "",
        "context": {},
        "tools": [],
    }

    async def fake_chat(*args):
        session = runner.conversation.ToolSession(runner.jarvis_tools.registry())
        assert "note_update" not in session.names
        loaded = session.load({"groups": ["notes"]})
        assert loaded["status"] == "loaded" and "note_update" in session.names
        assert runner.agent_instructions.datetime.now(UTC) == datetime(2030, 1, 14, 15, tzinfo=UTC)
        return {"message": "Read it.", "status": "succeeded", "tool_errors": []}

    module = SimpleNamespace(
        OWNER="owner",
        DEVICE="device",
        seed_case=lambda *_: fixture,
        stage_checks=lambda *_: {},
        grade_case=lambda *_: {"checks": {"read": True}, "safety_violations": [], "behavior_flags": []},
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
    assert result["outcome"] == "clean_success"
    assert result["turns"][0]["discovery_call_range"] == [0, 1]
    load = result["discovery_calls"][0]
    assert load["arguments"] == {"groups": ["notes"]}
    assert "note_update" not in load["offered_tool_names_before"]
    assert "note_update" in load["offered_tool_names_after"]
    assert load["turn_id"] and load["definition_bytes_after"] > 0
    assert set(load["outcome"]) == {"status", "tools", "message"}
    assert set(load["outcome"]["tools"]) <= set(load["offered_tool_names_after"])
    assert "parameters" not in runner.json.dumps(load["outcome"])
    session = runner.ToolSession(runner.jarvis_tools.registry())
    session.load({"groups": ["notes"]})
    definitions = [
        {"type": "function", "function": {key: value for key, value in tool.items() if key != "type"}}
        for tool in session.definitions()
    ]
    for model in runner.agent_models.catalog().values():
        body = model.request([{"role": "system", "content": "Synthetic"}], definitions)
        metrics = runner.request_context_metrics(body, runner.json_bytes(body))
        assert metrics["offered_tool_names"] == session.names
        assert len(set(metrics["offered_tool_names"])) == metrics["offered_tool_count"]


@pytest.mark.asyncio
async def test_selection_adapter_uses_exact_production_selection_and_original_oracle():
    cases = expert_cases()
    fixture = cases.seed_case("bulk_pagination", 2)
    read = await cases.invoke_tool(
        fixture,
        runner.jarvis_tools.call_tool,
        cases.OWNER,
        "selection-test",
        0,
        "task_list",
        {
            "project_id": fixture["p1"],
            "status": "open",
            "assignee": "owner",
            "tags_all": ["ship"],
            "tags_none": ["hold"],
            "due_from": "2030-01-15",
            "due_through": "2030-01-18",
            "limit": 3,
        },
        device=cases.DEVICE,
        conversation_id=fixture["conversation"],
    )
    assert read["match_count"] == 24 and len(read["tasks"]) == 3
    selection = fixture["tools"][0]["authoritative_selection"]
    assert {row["id"] for row in selection["records"]} == set(fixture["targets"])
    write = await cases.invoke_tool(
        fixture,
        runner.jarvis_tools.call_tool,
        cases.OWNER,
        "selection-test",
        1,
        "task_selection_update",
        {"selection_id": read["selection_id"], "changes": {"priority": 2, "planned_date": "2030-01-16"}},
        device=cases.DEVICE,
        conversation_id=fixture["conversation"],
    )
    assert write["data"]["applied_count"] == 24
    assert fixture["tools"][1]["authoritative_selection"]["match_count"] == 24
    graded = cases.grade_case(fixture, [{"message": "Updated24 tasks.", "status": "succeeded"}])
    assert all(graded["checks"].values())
    assert not graded["safety_violations"]


@pytest.mark.asyncio
async def test_selection_update_still_receives_concurrent_revision_fault():
    from jarvis.domain import DomainError

    cases = expert_cases()
    fixture = cases.seed_case("stale_revision", 1)
    read = await cases.invoke_tool(
        fixture,
        runner.jarvis_tools.call_tool,
        cases.OWNER,
        "selection-race",
        0,
        "task_list",
        {"query": "Migration"},
        device=cases.DEVICE,
        conversation_id=fixture["conversation"],
    )
    with pytest.raises(DomainError) as error:
        await cases.invoke_tool(
            fixture,
            runner.jarvis_tools.call_tool,
            cases.OWNER,
            "selection-race",
            1,
            "task_selection_update",
            {"selection_id": read["selection_id"], "changes": {"priority": 2}},
            device=cases.DEVICE,
            conversation_id=fixture["conversation"],
        )
    assert error.value.code == "REVISION_CONFLICT"
    assert fixture["faults"] == ["stale_revision"]
    with runner.engine().connect() as db:
        rows = db.exec_driver_sql(
            "SELECT priority FROM tasks WHERE id = ANY(%s)", (fixture["targets"],)
        ).all()
    assert all(row[0] == 0 for row in rows)


@pytest.mark.asyncio
async def test_refinement_stops_both_models_after_provider_rejection(monkeypatch, tmp_path):
    fake_cases = SimpleNamespace(CASES={"bulk_pagination": {"title": "Synthetic", "category": "Test"}})
    original_import = runner.importlib.import_module
    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda name: fake_cases if name == "expert_eval_cases" else original_import(name),
    )
    monkeypatch.setattr(runner.agent_models, "catalog", lambda: {"luna": profile(), "gemini": profile()})
    called = []

    async def reject(_cases, _profile, spec, _database_url, **_kwargs):
        called.append(spec)
        return {
            **spec,
            "outcome": "task_failure",
            "task_success": False,
            "clean_completion": False,
            "turns": [],
            "checks": {"provider_completed": False},
            "seconds": 0,
            "provider_calls": [{"status": 400, "error_code": "invalid_function_parameters"}],
        }

    monkeypatch.setattr(runner, "evaluate_case", reject)
    report = await runner.run(
        3,
        tmp_path / "blocked.json",
        list(runner.MODELS),
        cases=["bulk_pagination"],
        suite="tool-refinement20",
    )
    assert len(called) == 1
    assert report["state"] == "provider_blocked" and report["database_removed"] is True
    assert len(report["results"]) == 6
    assert sum(row["outcome"] == "skipped" for row in report["results"]) == 5
    assert not any(row["task_success"] for row in report["results"])


@pytest.mark.asyncio
async def test_schema_rejection_keeps_safe_diagnostic_and_redacts_credentials(monkeypatch):
    monkeypatch.setattr(runner.socket, "getaddrinfo", fake_dns)
    calls = []

    async def reject(_request):
        return httpx.Response(
            400,
            json={
                "error": {
                    "code": "invalid_function_parameters",
                    "param": "tools[4].parameters",
                    "message": "Invalid schema for ui_select; credential synthetic-secret must be scrubbed.",
                }
            },
        )

    with runner.outbound_guard(profile(), "postgresql://localhost:5432/test", calls, []):
        async with httpx.AsyncClient(transport=httpx.MockTransport(reject)) as client:
            await client.post(profile().endpoint, json={"model": profile().model})
    assert calls[0]["error_param"] == "tools[4].parameters"
    assert "ui_select" in calls[0]["schema_error"]
    assert "synthetic-secret" not in calls[0]["schema_error"]


@pytest.mark.asyncio
async def test_stop_file_waits_until_both_models_finish_fixture(monkeypatch, tmp_path):
    fake_cases = SimpleNamespace(CASES={"bulk_pagination": {"title": "Synthetic", "category": "Test"}})
    original_import = runner.importlib.import_module
    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda name: fake_cases if name == "expert_eval_cases" else original_import(name),
    )
    monkeypatch.setattr(runner.agent_models, "catalog", lambda: {"luna": profile(), "gemini": profile()})
    output = tmp_path / "paired.json"
    called = []

    async def evaluate(_cases, _profile, spec, _database_url, **_kwargs):
        called.append(spec)
        output.with_suffix(".stop").touch()
        return {
            **spec,
            "outcome": "clean_success",
            "task_success": True,
            "clean_completion": True,
            "turns": [],
            "checks": {"saved": True},
            "seconds": 0,
            "provider_calls": [{"status": 200}],
        }

    monkeypatch.setattr(runner, "evaluate_case", evaluate)
    report = await runner.run(
        3,
        output,
        list(runner.MODELS),
        cases=["bulk_pagination"],
        suite="tool-refinement20",
    )
    assert len(called) == 2 and {row["model"] for row in called} == set(runner.MODELS)
    assert report["state"] == "stopped" and report["database_removed"]
    assert len(report["results"]) == 2 and len(report["remaining_unattempted"]) == 4
    assert report["active_run"] is None


@pytest.mark.asyncio
async def test_interruption_preserves_current_meter_and_tool_evidence(monkeypatch, tmp_path):
    fake_cases = SimpleNamespace(CASES={"bulk_pagination": {"title": "Synthetic", "category": "Test"}})
    original_import = runner.importlib.import_module
    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda name: fake_cases if name == "expert_eval_cases" else original_import(name),
    )
    monkeypatch.setattr(runner.agent_models, "catalog", lambda: {"luna": profile(), "gemini": profile()})

    async def interrupted(_cases, _profile, spec, _database_url, on_progress=None):
        partial = {**spec, "provider_calls": [], "tools": [], "turns": []}
        on_progress(partial)
        partial["provider_calls"].append({"status": 200, "usage": {"input_tokens": 80, "output_tokens": 5}})
        partial["tools"].append({"name": "task_list", "arguments": {"query": "Synthetic"}})
        raise runner.asyncio.CancelledError()

    monkeypatch.setattr(runner, "evaluate_case", interrupted)
    output = tmp_path / "interrupted.json"
    with pytest.raises(runner.asyncio.CancelledError):
        await runner.run(
            1,
            output,
            list(runner.MODELS),
            cases=["bulk_pagination"],
            suite="tool-refinement20",
        )
    report = runner.json.loads(output.read_text())
    assert report["state"] == "interrupted" and report["database_removed"]
    assert report["results"] == []
    assert report["active_run"]["incomplete"] is True
    assert report["active_run"]["provider_calls"][0]["usage"]["input_tokens"] == 80
    assert report["active_run"]["tools"][0]["arguments"] == {"query": "Synthetic"}


@pytest.mark.asyncio
@pytest.mark.parametrize("minutes", [1, 481])
async def test_mock_read_validation_error_matches_production(minutes):
    from jarvis.domain import DomainError

    cases = expert_cases()
    arguments = {
        "start": "2030-01-15T09:00:00-06:00",
        "end": "2030-01-15T16:00:00-06:00",
        "minutes": minutes,
    }
    with pytest.raises(DomainError) as actual:
        await runner.jarvis_tools.call_tool(
            "synthetic-owner", "validation-parity", 0, "calendar_availability", arguments
        )
    with pytest.raises(DomainError) as synthetic:
        cases._validate("calendar_availability", arguments)
    assert synthetic.value.code == actual.value.code == "INVALID_ARGUMENT"
    assert str(synthetic.value) == str(actual.value)
    assert str(minutes) in str(synthetic.value)


@pytest.mark.asyncio
async def test_diagnostic_only_label_preserves_three_repeat_schedule(monkeypatch, tmp_path):
    fake_cases = SimpleNamespace(CASES={"bulk_pagination": {"title": "Synthetic", "category": "Test"}})
    original_import = runner.importlib.import_module
    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda name: fake_cases if name == "expert_eval_cases" else original_import(name),
    )
    monkeypatch.setattr(runner.agent_models, "catalog", lambda: {"luna": profile(), "gemini": profile()})

    async def complete(_cases, _profile, spec, _database_url, **_kwargs):
        return {
            **spec,
            "outcome": "clean_success",
            "task_success": True,
            "clean_completion": True,
            "turns": [],
            "checks": {"saved": True},
            "seconds": 0,
            "provider_calls": [],
        }

    monkeypatch.setattr(runner, "evaluate_case", complete)
    report = await runner.run(
        3,
        tmp_path / "diagnostic.json",
        list(runner.MODELS),
        cases=["bulk_pagination"],
        suite="tool-refinement20",
        diagnostic_only=True,
    )
    assert report["diagnostic_only"] is True
    assert report["planned_runs"] == len(report["results"]) == 6
    assert {row["repeat"] for row in report["results"]} == {1, 2, 3}
    assert report["state"] == "completed" and report["database_removed"] is True


def test_fresh_suite_selection_keeps_all_eight_cases_and_48_paired_trials():
    cases = {
        name: {"title": name}
        for name in [
            "sparse_note_unicode",
            "note_link_race",
            "schedule_three_blocks",
            "calendar_not_confirmed",
            "mobile_workspace_controls",
            "unsaved_view_recovery",
            "cross_type_entity",
            "infeasible_explain",
        ]
    }
    selected = runner.select_cases(cases, "reliability8")
    assert selected == cases
    schedule = runner.scenario_order(selected, 3, runner.MODELS, 20260913)
    assert len(schedule) == 48
    for i in range(0, 48, 2):
        assert schedule[i]["case"] == schedule[i + 1]["case"]
        assert schedule[i]["repeat"] == schedule[i + 1]["repeat"]


@pytest.mark.asyncio
async def test_case_hooks_preserve_stateful_ui_and_integration_traces(monkeypatch):
    fixture = {
        "case": "device-fixture",
        "fixture_hash": "fresh-hash",
        "conversation": "synthetic-conversation",
        "prompts": ["Show notes", "Keep those notes"],
        "memory_context": "",
        "context": {"view": "all"},
        "tools": [],
        "ui_trace": [],
        "integration_calls": [],
        "communication_expectations": ["Separate semantic review."],
    }
    screens = []

    def prepare(f, index):
        if index:
            f["fixture_events"].append({"kind": "owner_followup"})

    def display(f, action):
        f["context"]["view"] = action["view"]
        result = {"ui_action": action, "status": "displayed", "screen": dict(f["context"])}
        f["ui_trace"].append(result)
        return result

    async def fake_chat(*_args):
        screens.append(runner.conversation.get_context("owner", "device")["view"])
        await runner.jarvis_tools.dispatch(
            "owner", "device", {"id": str(len(screens)), "kind": "show", "view": "notes"}
        )
        fixture["integration_calls"].append({"kind": "synthetic_test_trace"})
        return {"message": "Notes are visible.", "status": "succeeded", "tool_errors": []}

    module = SimpleNamespace(
        OWNER="owner",
        DEVICE="device",
        seed_case=lambda *_: fixture,
        prepare_turn=prepare,
        dispatch_ui=display,
        stage_checks=lambda *_: {},
        grade_case=lambda *_: {"checks": {"saved": True}, "safety_violations": [], "behavior_flags": []},
        invoke_tool=None,
    )
    monkeypatch.setattr(runner, "outbound_guard", lambda *_: nullcontext())
    monkeypatch.setattr(runner.conversation, "chat", fake_chat)
    result = await runner.evaluate_case(
        module,
        profile(),
        {"case": "device-fixture", "repeat": 1, "model": "luna"},
        "postgresql://localhost/test",
    )
    assert screens == ["all", "notes"]
    assert len(result["ui_trace"]) == len(result["integration_calls"]) == 2
    assert result["fixture_events"] == [{"kind": "owner_followup"}]
    assert result["communication_expectations"] == ["Separate semantic review."]
    assert result["outcome"] == "clean_success"
