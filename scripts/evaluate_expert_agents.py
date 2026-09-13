"""Opt-in, isolated expert comparison of Eridani's production task-agent routes.

The suite uses synthetic fixtures, actual local domain commands, mocked external
integrations, and deterministic state graders. It never starts a worker or
changes the deployed application. Provider calls are billable and opt-in.
"""

import argparse
import asyncio
import copy
import hashlib
import importlib
import json
import os
import random
import re
import socket
import subprocess
import time
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit
from uuid import uuid4
from zoneinfo import ZoneInfo

import httpx
from jarvis import agent_instructions, agent_models, conversation
from jarvis import tools as jarvis_tools
from jarvis.config import get_settings
from jarvis.db import engine
from jarvis.models import Base
from jarvis.personality import SYSTEM_PROMPT
from jarvis.tool_catalog import ToolSession
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("gpt-5.6-luna", "gemini-3.8-flash")
DATABASE_PREFIX = "jarvis_expert_eval_"
FIXED_TIME = datetime(2030, 1, 14, 9, tzinfo=ZoneInfo("America/Chicago"))
TOOL_REFINEMENT_CASES = (
    "bulk_pagination",
    "selected_not_visible",
    "ambiguous_followup",
    "singular_ambiguous",
    "stale_revision",
    "lost_ack",
    "dst_gap",
    "dst_fold",
    "clear_deadline",
    "goal_rewire",
    "note_preservation",
    "evidence_extraction",
    "note_injection",
    "unsupported_email",
    "calendar_unknown",
    "constraint_schedule",
    "impossible_schedule",
    "pending_remote",
    "ui_refusal",
    "recurring_occurrence",
)
TOOL_REFINEMENT_EXCLUSIONS = {
    "subtask_reparent": "Previously clean; relationship preservation remains covered by goal_rewire and note_preservation.",
    "cross_zone": "Previously clean; offset-sensitive resolution remains covered by dst_gap and dst_fold.",
    "memory_override": "Previously clean; untrusted-data restraint remains covered by note_injection and unavailable-capability cases.",
    "all_day_span": "Previously clean; retained scheduling and DST cases stress the revised time tools more directly.",
}


def json_bytes(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())


def request_context_metrics(body, request_bytes):
    inputs = body.get("input", body.get("messages", []))
    definitions = body.get("tools", [])
    names = [tool.get("name") or tool.get("function", {}).get("name") for tool in definitions]
    system = [item.get("content") for item in inputs if item.get("role") == "system"]
    return {
        "offered_tool_names": names,
        "offered_tool_count": len(definitions),
        "tool_schema_bytes": json_bytes(definitions),
        "system_prompt_bytes": json_bytes(system),
        "input_context_bytes": json_bytes(inputs),
        "request_body_bytes": request_bytes,
    }


def requested_functions(data, api):
    """Project only executable function calls, never native reasoning or signatures."""
    if api == "responses":
        calls = [
            {"name": item.get("name"), "call_id": item.get("call_id"), "arguments": item.get("arguments")}
            for item in data.get("output", [])
            if item.get("type") == "function_call"
        ]
    else:
        message = (data.get("choices") or [{}])[0].get("message", {})
        calls = [
            {
                "name": item.get("function", {}).get("name"),
                "call_id": item.get("id"),
                "arguments": item.get("function", {}).get("arguments"),
            }
            for item in message.get("tool_calls", [])
        ]
    for call in calls:
        try:
            call["arguments"] = json.loads(call["arguments"])
        except (TypeError, ValueError):
            call["arguments_valid_json"] = False
    return calls


def select_cases(all_cases, suite, requested=None):
    if suite not in {"expert24", "tool-refinement20", "reliability8"}:
        raise ValueError("Unknown evaluation suite")
    names = list(TOOL_REFINEMENT_CASES) if suite == "tool-refinement20" else list(all_cases)
    if requested:
        if set(requested) - set(names):
            raise ValueError("Requested case is outside this fixed evaluation suite")
        names = [name for name in names if name in requested]
    return {name: all_cases[name] for name in names}


class InstructionDateTime(datetime):
    """Freeze only the date named in the system instructions."""

    @classmethod
    def now(cls, tz=None):
        return FIXED_TIME.astimezone(tz) if tz is not None else FIXED_TIME.replace(tzinfo=None)


def scenario_order(cases, repeats, models, seed):
    """Interleave providers per fixture, rotating the first provider each case."""
    schedule = []
    for repeat in range(1, repeats + 1):
        names = sorted(cases)
        random.Random(seed + repeat).shuffle(names)
        for index, case in enumerate(names):
            order = list(models)
            if (repeat + index) % 2 == 0:
                order.reverse()
            for model in order:
                schedule.append({"case": case, "repeat": repeat, "model": model})
    return schedule


def safe_exception(exc, secrets=()):
    message = str(exc)
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[REDACTED]")
    # Connection exceptions may include a DSN; remove password and credential-like values.
    message = re.sub(r"(://[^:/\s]+:)[^@\s]+@", r"\1[REDACTED]@", message)
    message = re.sub(r"(?i)(bearer\s+)[^\s]+", r"\1[REDACTED]", message)
    return {"type": type(exc).__name__, "message": message[:1500]}


def provider_stop_reason(calls):
    for call in calls:
        status = call.get("status")
        code = str(call.get("error_code") or "").lower()
        if status in {401, 403, 404}:
            return "authentication_or_model_access"
        if status == 429 or "quota" in code or "billing" in code:
            return "quota_or_rate_limit"
        if status == 400:
            if any(word in code for word in ("model", "permission", "api_key", "auth")):
                return "authentication_or_model_access"
            return "provider_request_rejected"
    return None


def checkpoint(output, report):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(output)


def digest_file(path):
    path = ROOT / path
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def source_manifest(profiles, settings, cases, repeats, seed):
    source_paths = [
        "scripts/evaluate_expert_agents.py",
        "scripts/expert_eval_cases.py",
        "scripts/eval_integrations.py",
        "scripts/reliability_eval_cases.py",
        "scripts/eval_ui.py",
        "scripts/export_reliability_fixtures.py",
        "apps/api/jarvis/planner.py",
        "apps/api/jarvis/planner_schema.py",
        "apps/api/jarvis/note_schema.py",
        "apps/api/jarvis/record_references.py",
        "apps/api/jarvis/ui_contracts.py",
        "apps/web/src/site-actions.ts",
        "apps/web/src/site-validation.ts",
        "apps/web/src/editor-control.tsx",
        "apps/web/src/App.tsx",
        "apps/web/src/Notes.tsx",
        "apps/web/src/Workspace.tsx",
        "apps/web/src/Productivity.tsx",
        "apps/web/src/work-views.ts",
        "apps/web/src/workspace.ts",
        "apps/web/src/productivity.ts",
        "apps/web/src/copilot.tsx",
        "apps/api/jarvis/conversation.py",
        "apps/api/jarvis/tools.py",
        "apps/api/jarvis/tool_catalog.py",
        "apps/api/jarvis/agent_instructions.py",
        "apps/api/jarvis/task_tools.py",
        "apps/api/jarvis/time_tools.py",
        "apps/api/jarvis/remote_status.py",
        "apps/api/jarvis/personality.py",
        "apps/api/jarvis/responses_adapter.py",
        "apps/api/jarvis/agent_models.py",
        "apps/api/jarvis/domain.py",
        "apps/api/jarvis/productivity.py",
        "apps/api/jarvis/task_context.py",
        "apps/api/jarvis/notes.py",
        "apps/api/jarvis/google_writes.py",
        "apps/api/jarvis/ui_control.py",
        "apps/api/jarvis/memory_learning.py",
        "apps/api/jarvis/models.py",
        "apps/api/jarvis/config.py",
    ]
    source_paths = sorted(
        set(source_paths) | {str(path.relative_to(ROOT)) for path in (ROOT / "apps/api/jarvis").glob("*.py")}
    )
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    schemas = jarvis_tools.registry()
    initial_schemas = ToolSession(schemas).definitions()
    return {
        "started_at": datetime.now(UTC).isoformat(),
        "git_commit": commit,
        "tracked_worktree_dirty": dirty,
        "source_sha256": {path: digest_file(path) for path in source_paths},
        "tool_schema_sha256": hashlib.sha256(
            json.dumps(schemas, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
        "system_personality_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "initial_catalog": {
            "tool_names": [item["name"] for item in initial_schemas],
            "tool_count": len(initial_schemas),
            "definition_bytes": json_bytes(initial_schemas),
        },
        "full_catalog": {"tool_count": len(schemas), "definition_bytes": json_bytes(schemas)},
        "context_metric_encoding": "UTF-8 bytes of compact JSON; input_context_bytes includes opaque reasoning state size without its contents",
        "fixed_instruction_time": FIXED_TIME.isoformat(),
        "runtime_clock_frozen": False,
        "seed": seed,
        "repeats": repeats,
        "case_metadata": cases,
        "models": {
            key: {
                "provider": profile.provider,
                "model": profile.model,
                "api": profile.api,
                "endpoint": profile.endpoint,
                "reasoning_effort": profile.reasoning_effort,
                "max_output_tokens": profile.max_output_tokens,
            }
            for key, profile in profiles.items()
        },
        "limits": {
            "max_tool_calls_per_request": settings.max_tool_calls_per_request,
            "max_model_rounds_per_request": settings.max_model_rounds_per_request,
            "provider_timeout_seconds": 60,
            "history_source_limit": 12,
            "history_source_character_limit": 3000,
        },
        "isolation": {
            "database": "new randomly named disposable PostgreSQL database",
            "fixtures": "synthetic only; identical scenario seed for each paired model run",
            "memory": "synthetic prompt context; no extraction or embedding provider",
            "integrations": "local mocks, no Google/Linear writes or worker",
            "http": "only POST to the selected model endpoint; HTTP redirects disabled",
            "python_sockets": "only selected provider:443 and configured PostgreSQL host:port",
            "transcripts": "ordinary user/assistant/tool traces; native reasoning content excluded",
            "cost_tracking": False,
            "provider_retries": 0,
        },
        "timing": {
            "seconds": "harness elapsed including fixture seed, DNS setup, chat turns, and grading",
            "task_seconds": "sum of conversation.chat elapsed time across the scenario turns",
            "provider_seconds": "sum of measured model HTTP request elapsed time",
        },
        "grading": {
            "kind": "deterministic database state and explicit trace checks; no model judge",
            "clean_completion": "all state checks, no safety violations, no tool errors, all turns succeeded",
            "recovered_success": "state checks pass safely but a recoverable tool error or partial status occurred",
            "missing_runs": "explicit skipped entries; never counted as passes",
        },
    }


@contextmanager
def isolated_environment(database_url):
    """Restore prior values, including a caller-supplied database override."""
    overrides = {
        "JARVIS_DATABASE_URL": database_url,
        "JARVIS_COST_TRACKING_ENABLED": "false",
    }
    original = {key: os.environ.get(key) for key in overrides}
    if engine.cache_info().currsize:
        engine().dispose()
    os.environ.update(overrides)
    get_settings.cache_clear()
    engine.cache_clear()
    try:
        yield get_settings()
    finally:
        # Dispose the isolated pool before restoring configuration.
        if engine.cache_info().currsize:
            engine().dispose()
        engine.cache_clear()
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()


def validate_disposable_database(name, expected_name):
    if name != expected_name or not re.fullmatch(DATABASE_PREFIX + r"[a-f0-9]{32}", name or ""):
        raise ValueError("Refusing an operation outside the newly created expert-evaluation database")


@contextmanager
def outbound_guard(profile, database_url, provider_calls, blocked_calls):
    """Guard every HTTP send and Python socket connection during a model run."""
    endpoint = profile.endpoint
    provider = urlsplit(endpoint)
    database = make_url(database_url)
    allowed_hosts = {
        (provider.hostname, provider.port or 443),
        (database.host or "127.0.0.1", database.port or 5432),
    }
    original_resolve = socket.getaddrinfo
    allowed_addresses = set()
    for host, port in allowed_hosts:
        for row in original_resolve(host, port, type=socket.SOCK_STREAM):
            allowed_addresses.add((row[4][0], port))
    allowed_hosts |= allowed_addresses
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex
    real_async_send = httpx.AsyncClient.send
    real_async_init = httpx.AsyncClient.__init__

    def blocked(kind):
        blocked_calls.append({"kind": kind})
        raise ValueError("Evaluation blocked unexpected outbound access")

    def check_address(address):
        if not isinstance(address, tuple) or len(address) < 2:
            blocked("socket_destination")
        if (address[0], int(address[1])) not in allowed_addresses:
            blocked("socket_destination")

    def resolve(host, port, *args, **kwargs):
        # AnyIO/httpcore supplies IDNA hostnames as bytes; compare the same hostname.
        normalized_host = host.decode("ascii") if isinstance(host, bytes) else host
        if (normalized_host, int(port)) not in allowed_hosts:
            blocked("dns_destination")
        rows = original_resolve(host, port, *args, **kwargs)
        for row in rows:
            allowed_addresses.add((row[4][0], int(port)))
        return rows

    def connect(sock, address):
        check_address(address)
        return real_connect(sock, address)

    def connect_ex(sock, address):
        check_address(address)
        return real_connect_ex(sock, address)

    def async_init(client, *args, **kwargs):
        kwargs["trust_env"] = False  # Do not route synthetic data through inherited proxies.
        return real_async_init(client, *args, **kwargs)

    async def send(client, request, *args, **kwargs):
        if str(request.url) != endpoint or request.method != "POST":
            blocked("http_endpoint")
        try:
            body = json.loads(request.content)
            request_model = body.get("model")
        except (TypeError, ValueError):
            blocked("provider_payload")
        if request_model != profile.model:
            blocked("provider_model")
        input_messages = body.get("input", body.get("messages", []))
        system_content = [item.get("content") for item in input_messages if item.get("role") == "system"]
        meter = {
            "request_model": request_model,
            "system_prompt_sha256": hashlib.sha256(
                json.dumps(system_content, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest(),
        }
        meter.update(request_context_metrics(body, len(request.content)))
        provider_calls.append(meter)
        started = time.perf_counter()
        try:
            kwargs["follow_redirects"] = False
            response = await real_async_send(client, request, *args, **kwargs)
            meter["status"] = response.status_code
            try:
                data = response.json()
            except ValueError:
                data = {}
                meter["invalid_json"] = True
            if response.is_success:
                # Usage metadata has numeric token counts, never reasoning contents.
                meter.update(
                    response_model=data.get("model"),
                    usage=data.get("usage", {}),
                    requested_tools=requested_functions(data, profile.api),
                )
                meter["finish_reason"] = (
                    data.get("status")
                    if profile.api == "responses"
                    else (data.get("choices") or [{}])[0].get("finish_reason")
                )
            else:
                error = data.get("error", {})
                code = error.get("code") if isinstance(error, dict) else None
                meter["error_code"] = (
                    str(code) if code is not None and re.fullmatch(r"[\w.-]{1,100}", str(code)) else None
                )
                parameter = error.get("param") if isinstance(error, dict) else None
                if isinstance(parameter, str) and re.fullmatch(r"[A-Za-z0-9_.\[\]-]{1,200}", parameter):
                    meter["error_param"] = parameter
                if code == "invalid_function_parameters" and isinstance(error.get("message"), str):
                    meter["schema_error"] = safe_exception(ValueError(error["message"]), (profile.api_key,))[
                        "message"
                    ]
            return response
        except Exception as exc:
            meter["exception_type"] = type(exc).__name__
            raise
        finally:
            meter["seconds"] = round(time.perf_counter() - started, 4)

    def deny_http(*_args, **_kwargs):
        blocked("non_model_http")

    with ExitStack() as stack:
        stack.enter_context(patch.object(httpx.AsyncClient, "__init__", async_init))
        stack.enter_context(patch.object(httpx.AsyncClient, "send", send))
        stack.enter_context(patch.object(httpx.Client, "send", deny_http))
        stack.enter_context(patch("urllib.request.urlopen", deny_http))
        try:
            requests = importlib.import_module("requests.sessions")
        except ImportError:
            pass
        else:
            stack.enter_context(patch.object(requests.Session, "request", deny_http))
        stack.enter_context(patch.object(socket, "getaddrinfo", resolve))
        stack.enter_context(patch.object(socket.socket, "connect", connect))
        stack.enter_context(patch.object(socket.socket, "connect_ex", connect_ex))
        yield


def classify(checks, turns, safety_violations, exception=None):
    if safety_violations:
        return "safety_failure"
    if exception:
        return "infrastructure_error"
    if not checks or not all(checks.values()):
        return "task_failure"
    if any(turn.get("status") != "succeeded" or turn.get("tool_errors") for turn in turns):
        return "recovered_success"
    return "clean_success"


def summarize(results, models):
    summary = {}
    for model in models:
        rows = [row for row in results if row["model"] == model]
        attempted = [row for row in rows if row["outcome"] != "skipped"]
        successful = [row for row in attempted if row.get("task_success")]
        summary[model] = {
            "scheduled": len(rows),
            "attempted": len(attempted),
            "skipped": len(rows) - len(attempted),
            "task_successes": len(successful),
            "clean_completions": sum(row["outcome"] == "clean_success" for row in attempted),
            "recovered_successes": sum(row["outcome"] == "recovered_success" for row in attempted),
            "safety_failures": sum(row["outcome"] == "safety_failure" for row in attempted),
            "infrastructure_errors": sum(row["outcome"] == "infrastructure_error" for row in attempted),
            "tool_errors": sum(
                len(turn.get("tool_errors", [])) for row in attempted for turn in row["turns"]
            ),
            "seconds": round(sum(row.get("seconds", 0) for row in attempted), 3),
            "task_seconds": round(sum(row.get("task_seconds", 0) for row in attempted), 3),
            "provider_seconds": round(sum(row.get("provider_seconds", 0) for row in attempted), 3),
            "provider_requests": sum(len(row.get("provider_calls", [])) for row in attempted),
        }
    return summary


async def evaluate_case(cases_module, profile, run_spec, database_url, on_progress=None):
    provider_calls, blocked_calls, turns, checks = [], [], [], {}
    discovery_calls = []
    active_turn = {"index": None, "id": None}
    current = {
        **run_spec,
        "provider_calls": provider_calls,
        "turns": turns,
        "discovery_calls": discovery_calls,
        "tools": [],
    }
    if on_progress:
        on_progress(current)
    started = time.perf_counter()
    fixture = None
    exception = None
    grading = {"checks": {}, "safety_violations": [], "behavior_flags": []}
    secrets = [profile.api_key, make_url(database_url).password]
    try:
        fixture = cases_module.seed_case(run_spec["case"], run_spec["repeat"])
        current["tools"] = fixture["tools"]
        current["integration_calls"] = fixture.get("integration_calls", [])
        current["integration_fixture"] = copy.deepcopy(fixture.get("calendar_fixture"))
        current["ui_trace"] = fixture.get("ui_trace", [])
        current["communication_expectations"] = fixture.get("communication_expectations", [])
        fixture.setdefault("fixture_events", [])
        current["fixture_events"] = fixture["fixture_events"]
        current["fixture_hash"] = fixture["fixture_hash"]
        current["fixture_prompts_sha256"] = hashlib.sha256(
            json.dumps(fixture["prompts"], ensure_ascii=False).encode()
        ).hexdigest()
        current["injected_memory_sha256"] = hashlib.sha256(fixture["memory_context"].encode()).hexdigest()
        real_tool = conversation.call_tool

        class ObservedToolSession(ToolSession):
            def load(self, arguments):
                trace = {
                    "name": "tools_load",
                    "arguments": copy.deepcopy(arguments),
                    "turn_id": active_turn["id"],
                    "turn_index": active_turn["index"],
                    "provider_call_index": len(provider_calls) - 1,
                    "offered_tool_names_before": list(self.names),
                }
                discovery_calls.append(trace)
                load_started = time.perf_counter()
                try:
                    result = super().load(arguments)
                    trace["outcome"] = copy.deepcopy(result)
                    return result
                except Exception as exc:
                    trace["error"] = getattr(exc, "code", type(exc).__name__)
                    trace["message"] = str(exc)
                    raise
                finally:
                    trace["seconds"] = round(time.perf_counter() - load_started, 4)
                    trace["offered_tool_names_after"] = list(self.names)
                    trace["definition_bytes_after"] = json_bytes(self.definitions())

        async def invoke(owner, turn, index, name, arguments, **kwargs):
            return await cases_module.invoke_tool(
                fixture, real_tool, owner, turn, index, name, arguments, **kwargs
            )

        async def memory(*_args, **_kwargs):
            return fixture["memory_context"]

        def context(*_args, **_kwargs):
            return copy.deepcopy(fixture["context"])

        async def displayed(_owner, _device, action):
            custom = getattr(cases_module, "dispatch_ui", None)
            if custom:
                return custom(fixture, action)
            if fixture["case"] == "ui_refusal":
                return {
                    "ui_action": action,
                    "status": "failed",
                    "message": "Navigation blocked by an unsaved editor. Save or discard the edit first.",
                }
            return {
                "ui_action": action,
                "status": "displayed",
                "message": "Synthetic UI acknowledged display.",
            }

        with (
            outbound_guard(profile, database_url, provider_calls, blocked_calls),
            patch.object(agent_models, "selected", return_value=profile),
            patch.object(conversation, "call_tool", invoke),
            patch.object(conversation, "prompt_context", memory),
            patch.object(conversation, "get_context", context),
            patch.object(jarvis_tools, "get_context", context),
            patch.object(jarvis_tools, "dispatch", displayed),
            patch.object(agent_instructions, "datetime", InstructionDateTime),
            patch.object(conversation, "ToolSession", ObservedToolSession),
        ):
            for turn_index, prompt in enumerate(fixture["prompts"]):
                prepare = getattr(cases_module, "prepare_turn", None)
                if prepare:
                    prepare(fixture, turn_index)
                turn_started = time.perf_counter()
                first_call = len(provider_calls)
                first_discovery = len(discovery_calls)
                active_turn.update(index=turn_index, id=str(uuid4()))
                current["pending_user_message"] = prompt
                result = await conversation.chat(
                    cases_module.OWNER,
                    cases_module.DEVICE,
                    active_turn["id"],
                    fixture["conversation"],
                    prompt,
                )
                turn = {
                    "user": prompt,
                    "assistant": result["message"],
                    "message": result["message"],
                    "status": result["status"],
                    "tool_errors": result.get("tool_errors", []),
                    "actions": result.get("actions", []),
                    "ui_actions": result.get("ui_actions", []),
                    "tool_calls": result.get("tool_calls", 0),
                    "seconds": round(time.perf_counter() - turn_started, 4),
                    "provider_call_range": [first_call, len(provider_calls)],
                    "discovery_call_range": [first_discovery, len(discovery_calls)],
                    "result": result,
                }
                turns.append(turn)
                current.pop("pending_user_message", None)
                checks.update(
                    {
                        f"turn_{turn_index + 1}.{name}": bool(value)
                        for name, value in cases_module.stage_checks(fixture, turn_index, result).items()
                    }
                )
                checks[f"turn_{turn_index + 1}.provider_completed"] = result["status"] not in {
                    "failed",
                    "cancelled",
                }
                if provider_stop_reason(provider_calls):
                    break
    except Exception as exc:  # noqa: BLE001 - preserve each failed trial without hiding or retrying it
        exception = safe_exception(exc, secrets)
    if fixture is not None:
        if fixture.get("harness_errors"):
            exception = {"type": "UnsupportedFixtureTransition", "details": fixture["harness_errors"]}
        try:
            grading = cases_module.grade_case(fixture, turns)
        except Exception as exc:  # noqa: BLE001 - a grader failure is an explicit infrastructure error
            grading_error = safe_exception(exc, secrets)
            exception = {**(exception or {}), "grading_error": grading_error}
    checks.update({name: bool(value) for name, value in grading["checks"].items()})
    checks["all_conversation_turns_attempted"] = bool(fixture and len(turns) == len(fixture["prompts"]))
    checks["no_unexpected_outbound_access"] = not blocked_calls
    safety = list(grading["safety_violations"])
    if blocked_calls:
        exception = {
            **(exception or {}),
            "egress_guard": "Unexpected outbound access was blocked; inspect harness/tool trace before attribution.",
        }
    outcome = classify(checks, turns, safety, exception)
    current.update(
        seconds=round(time.perf_counter() - started, 4),
        task_seconds=round(sum(turn["seconds"] for turn in turns), 4),
        provider_seconds=round(sum(call["seconds"] for call in provider_calls), 4),
        checks=checks,
        safety_violations=safety,
        behavior_flags=grading["behavior_flags"],
        blocked_outbound_calls=blocked_calls,
        tools=fixture["tools"] if fixture else [],
        discovery_calls=discovery_calls,
        exception=exception,
        outcome=outcome,
        task_success=outcome in {"clean_success", "recovered_success"},
        clean_completion=outcome == "clean_success",
    )
    return current


async def run(
    repeats,
    output,
    models,
    cases=None,
    seed=20260913,
    preflight=False,
    suite="expert24",
    stop_file=None,
    diagnostic_only=False,
):
    stop_file = Path(stop_file) if stop_file else Path(output).with_suffix(".stop")
    if stop_file.exists():
        raise ValueError("A stop file already exists for this output; use a fresh output path.")
    cases_module = importlib.import_module(
        "reliability_eval_cases" if suite == "reliability8" else "expert_eval_cases"
    )
    selected_cases = select_cases(cases_module.CASES, suite, cases)
    catalog = agent_models.catalog()
    profiles = {key: catalog["luna" if key == "gpt-5.6-luna" else "gemini"] for key in models}
    if any(profile.max_output_tokens != 8192 for profile in profiles.values()):
        raise ValueError("Both evaluation routes must retain the agreed 8192-token output cap")
    if not all(profile.available for profile in profiles.values()):
        raise ValueError("Selected models need their API keys; no credentials are printed")

    if preflight:
        repeats = 1
        if not cases and suite != "reliability8":
            selected_cases = {
                name: selected_cases[name] for name in ("selected_not_visible", "ambiguous_followup")
            }
    original = get_settings().database_url
    name = DATABASE_PREFIX + uuid4().hex
    database_url = make_url(original).set(database=name).render_as_string(hide_password=False)
    admin = create_engine(original, isolation_level="AUTOCOMMIT")
    created = False
    report = {
        "manifest": source_manifest(profiles, get_settings(), selected_cases, repeats, seed),
        "suite": suite,
        "baseline_artifact": "docs/evals/expert-agents-2026-09-13.json.gz"
        if suite == "tool-refinement20"
        else None,
        "excluded_cases": TOOL_REFINEMENT_EXCLUSIONS if suite == "tool-refinement20" else {},
        "diagnostic_only": preflight or diagnostic_only,
        "state": "running",
        "planned_runs": len(selected_cases) * repeats * len(models),
        "results": [],
        "blocked_models": {},
        "active_run": None,
        "stop_file": str(stop_file),
        "database_removed": False,
    }
    checkpoint(output, report)
    try:
        with admin.connect() as db:
            validate_disposable_database(name, name)
            db.exec_driver_sql(f'CREATE DATABASE "{name}"')
        created = True
        with isolated_environment(database_url):
            validate_disposable_database(make_url(get_settings().database_url).database, name)
            Base.metadata.create_all(engine())
            schedule = scenario_order(selected_cases, repeats, models, seed)
            for run_index, spec in enumerate(schedule):
                reason = None
                if spec["model"] in report["blocked_models"]:
                    result = {
                        **spec,
                        "outcome": "skipped",
                        "skip_reason": report["blocked_models"][spec["model"]],
                        "task_success": False,
                        "clean_completion": False,
                        "turns": [],
                        "checks": {},
                    }
                else:
                    result = await evaluate_case(
                        cases_module,
                        profiles[spec["model"]],
                        spec,
                        database_url,
                        on_progress=lambda current: report.update(active_run=current),
                    )
                    reason = provider_stop_reason(result["provider_calls"])
                    if reason:
                        report["blocked_models"][spec["model"]] = reason
                report["results"].append(result)
                report["active_run"] = None
                report["summary"] = summarize(report["results"], models)
                checkpoint(output, report)
                print(
                    json.dumps(
                        {
                            "completed": len(report["results"]),
                            "planned": report["planned_runs"],
                            **{key: result[key] for key in ("model", "repeat", "case", "outcome")},
                            "seconds": result.get("seconds"),
                            "failed_checks": [key for key, passed in result["checks"].items() if not passed],
                            "exception": result.get("exception"),
                        }
                    ),
                    flush=True,
                )
                if reason and suite in {"tool-refinement20", "reliability8"}:
                    report["state"] = "provider_blocked"
                    report["halt_reason"] = {"model": spec["model"], "reason": reason}
                    for remaining in schedule[run_index + 1 :]:
                        report["results"].append(
                            {
                                **remaining,
                                "outcome": "skipped",
                                "skip_reason": "paired_suite_halted_after_provider_rejection",
                                "task_success": False,
                                "clean_completion": False,
                                "turns": [],
                                "checks": {},
                            }
                        )
                    report["summary"] = summarize(report["results"], models)
                    checkpoint(output, report)
                    print(
                        json.dumps(
                            {
                                "paired_suite_halted": True,
                                "attempted_before_halt": run_index + 1,
                                "remaining_skipped": len(schedule) - run_index - 1,
                            }
                        ),
                        flush=True,
                    )
                    break
                next_spec = schedule[run_index + 1] if run_index + 1 < len(schedule) else None
                pair_complete = next_spec is None or (next_spec["case"], next_spec["repeat"]) != (
                    spec["case"],
                    spec["repeat"],
                )
                if pair_complete and stop_file.exists():
                    report["state"] = "stopped"
                    report["stop_reason"] = "requested_at_completed_pair_boundary"
                    report["remaining_unattempted"] = schedule[run_index + 1 :]
                    checkpoint(output, report)
                    break
        if report["state"] == "running":
            report["state"] = "completed"
    except BaseException as exc:
        report["state"] = (
            "interrupted" if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)) else "failed"
        )
        report["fatal_exception"] = safe_exception(
            exc, [profile.api_key for profile in profiles.values()] + [make_url(original).password]
        )
        raise
    finally:
        if report.get("active_run") is not None:
            report["active_run"]["incomplete"] = True
        if created:
            # This name is generated once by this invocation, never read from the CLI.
            validate_disposable_database(name, name)
            with admin.connect() as db:
                db.exec_driver_sql(f'DROP DATABASE "{name}" WITH (FORCE)')
            report["database_removed"] = True
        admin.dispose()
        report["finished_at"] = datetime.now(UTC).isoformat()
        report["summary"] = summarize(report["results"], models)
        checkpoint(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, choices=range(1, 11), default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--suite", choices=["expert24", "tool-refinement20", "reliability8"], default="expert24"
    )
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument(
        "--stop-file", type=Path, help="Create this file to stop after the current model pair."
    )
    parser.add_argument(
        "--preflight", action="store_true", help="Diagnostic-only run; excluded from final comparison"
    )
    parser.add_argument(
        "--diagnostic-only", action="store_true", help="Exclude from scored comparison; preserve --repeats"
    )
    args = parser.parse_args()
    output = args.output or Path(
        ".runtime/tool-refinement-evaluation.json"
        if args.suite == "tool-refinement20"
        else ".runtime/reliability-heldout.json"
        if args.suite == "reliability8"
        else ".runtime/expert-evaluation.json"
    )
    asyncio.run(
        run(
            args.repeats,
            output,
            args.models,
            args.cases,
            args.seed,
            args.preflight,
            args.suite,
            args.stop_file,
            args.diagnostic_only,
        )
    )


if __name__ == "__main__":
    main()
