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
from jarvis import agent_models, conversation
from jarvis import tools as jarvis_tools
from jarvis.config import get_settings
from jarvis.db import engine
from jarvis.models import Base
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("gpt-5.6-luna", "gemini-3.8-flash")
DATABASE_PREFIX = "jarvis_expert_eval_"
FIXED_TIME = datetime(2030, 1, 14, 9, tzinfo=ZoneInfo("America/Chicago"))


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
        "apps/api/jarvis/conversation.py",
        "apps/api/jarvis/tools.py",
        "apps/api/jarvis/personality.py",
        "apps/api/jarvis/responses_adapter.py",
        "apps/api/jarvis/agent_models.py",
        "apps/api/jarvis/domain.py",
        "apps/api/jarvis/productivity.py",
        "apps/api/jarvis/task_context.py",
        "apps/api/jarvis/models.py",
        "apps/api/jarvis/config.py",
    ]
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
    return {
        "started_at": datetime.now(UTC).isoformat(),
        "git_commit": commit,
        "tracked_worktree_dirty": dirty,
        "source_sha256": {path: digest_file(path) for path in source_paths},
        "tool_schema_sha256": hashlib.sha256(
            json.dumps(schemas, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
        "system_personality_sha256": hashlib.sha256(jarvis_tools.SYSTEM_PROMPT.encode()).hexdigest(),
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
                meter.update(response_model=data.get("model"), usage=data.get("usage", {}))
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


async def evaluate_case(cases_module, profile, run_spec, database_url):
    provider_calls, blocked_calls, turns, checks = [], [], [], {}
    current = {**run_spec, "provider_calls": provider_calls, "turns": turns}
    started = time.perf_counter()
    fixture = None
    exception = None
    grading = {"checks": {}, "safety_violations": [], "behavior_flags": []}
    secrets = [profile.api_key, make_url(database_url).password]
    try:
        fixture = cases_module.seed_case(run_spec["case"], run_spec["repeat"])
        current["fixture_hash"] = fixture["fixture_hash"]
        current["fixture_prompts_sha256"] = hashlib.sha256(
            json.dumps(fixture["prompts"], ensure_ascii=False).encode()
        ).hexdigest()
        current["injected_memory_sha256"] = hashlib.sha256(fixture["memory_context"].encode()).hexdigest()
        real_tool = conversation.call_tool

        async def invoke(owner, turn, index, name, arguments, **kwargs):
            return await cases_module.invoke_tool(
                fixture, real_tool, owner, turn, index, name, arguments, **kwargs
            )

        async def memory(*_args, **_kwargs):
            return fixture["memory_context"]

        def context(*_args, **_kwargs):
            return copy.deepcopy(fixture["context"])

        async def displayed(_owner, _device, action):
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
            patch.object(jarvis_tools, "datetime", InstructionDateTime),
        ):
            for turn_index, prompt in enumerate(fixture["prompts"]):
                turn_started = time.perf_counter()
                first_call = len(provider_calls)
                result = await conversation.chat(
                    cases_module.OWNER,
                    cases_module.DEVICE,
                    str(uuid4()),
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
                    "result": result,
                }
                turns.append(turn)
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
        exception=exception,
        outcome=outcome,
        task_success=outcome in {"clean_success", "recovered_success"},
        clean_completion=outcome == "clean_success",
    )
    return current


async def run(repeats, output, models, cases=None, seed=20260913, preflight=False):
    cases_module = importlib.import_module("expert_eval_cases")
    selected_cases = {
        name: metadata for name, metadata in cases_module.CASES.items() if not cases or name in cases
    }
    if cases and set(cases) - set(selected_cases):
        raise ValueError("Unknown expert-evaluation case")
    catalog = agent_models.catalog()
    profiles = {key: catalog["luna" if key == "gpt-5.6-luna" else "gemini"] for key in models}
    if any(profile.max_output_tokens != 8192 for profile in profiles.values()):
        raise ValueError("Both evaluation routes must retain the agreed 8192-token output cap")
    if not all(profile.available for profile in profiles.values()):
        raise ValueError("Selected models need their API keys; no credentials are printed")

    if preflight:
        repeats = 1
        if not cases:
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
        "diagnostic_only": preflight,
        "state": "running",
        "planned_runs": len(selected_cases) * repeats * len(models),
        "results": [],
        "blocked_models": {},
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
            for spec in scenario_order(selected_cases, repeats, models, seed):
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
                    result = await evaluate_case(cases_module, profiles[spec["model"]], spec, database_url)
                    reason = provider_stop_reason(result["provider_calls"])
                    if reason:
                        report["blocked_models"][spec["model"]] = reason
                report["results"].append(result)
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
    parser.add_argument("--output", type=Path, default=Path(".runtime/expert-evaluation.json"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument(
        "--preflight", action="store_true", help="Diagnostic-only run; excluded from final comparison"
    )
    args = parser.parse_args()
    asyncio.run(run(args.repeats, args.output, args.models, args.cases, args.seed, args.preflight))


if __name__ == "__main__":
    main()
