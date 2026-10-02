"""Metered provider transport for isolated worker processes, synchronous and async."""

import contextvars
import importlib
import json
import socket
import time
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from decimal import Decimal
from unittest.mock import patch

import httpx

from .spending import EvalLimit

AUTHORIZED = contextvars.ContextVar("eval_authorized_http", default=False)
CATEGORY = contextvars.ContextVar("eval_billing_category", default="agent")
ENDPOINTS = {
    "https://api.openai.com/v1/responses": "gpt-5.6-luna",
    "https://api.openai.com/v1/chat/completions": "gpt-5.6-luna",
    "https://api.openai.com/v1/embeddings": "text-embedding-3-small",
}


@contextmanager
def category(name):
    token = CATEGORY.set(name)
    try:
        yield
    finally:
        CATEGORY.reset(token)


@contextmanager
def metered(ledger, trial, trace, *, embeddings=False):
    from jarvis.agent_models import catalog

    luna = catalog()["luna"]
    real_async, real_sync = httpx.AsyncClient.send, httpx.Client.send
    real_async_init, real_sync_init = httpx.AsyncClient.__init__, httpx.Client.__init__
    real_dns, real_connect, real_connect_ex = (
        socket.getaddrinfo,
        socket.socket.connect,
        socket.socket.connect_ex,
    )
    failed_limits = []
    transport_errors = []
    active_requests = 0

    def deny(*args, **kwargs):
        raise ValueError("Eval blocked unmetered outbound access")

    def reserve(request):
        endpoint = str(request.url)
        if request.method != "POST" or endpoint not in ENDPOINTS:
            return deny()
        payload = json.loads(request.content)
        if payload.get("model") != ENDPOINTS[endpoint] or payload.get("stream"):
            return deny()
        embedding = endpoint.endswith("/embeddings")
        if embedding and not embeddings:
            return deny()
        if any(tool.get("type") != "function" for tool in payload.get("tools", [])):
            return deny()
        if embedding:
            agent = None
            bound = Decimal(len(request.content) + 4096) * Decimal("0.02") / 1_000_000
            kind = "embedding"
        else:
            limit = payload.get(
                "max_output_tokens", payload.get("max_completion_tokens", payload.get("max_tokens"))
            )
            if type(limit) is not int or not 0 < limit <= 8192:
                return deny()
            agent = replace(
                luna,
                max_output_tokens=limit,
                api="responses" if endpoint.endswith("/responses") else "chat_completions",
            )
            bound = agent.reserve_cost(len(request.content) + 4096)
            kind = CATEGORY.get()
        try:
            reservation = ledger.reserve(trial, kind, payload["model"], bound)
        except EvalLimit as exc:
            failed_limits.append(str(exc))
            raise
        row = {
            "reservation": reservation,
            "kind": kind,
            "model": payload["model"],
            "started": time.monotonic(),
        }
        trace.append(row)
        return row, agent

    def settle(row, agent, response):
        row["http_status"] = response.status_code
        row["duration_seconds"] = round(time.monotonic() - row.pop("started"), 4)
        row["request_id"] = response.headers.get("x-request-id")
        try:
            usage = response.json()["usage"]
            if agent is None:
                tokens = usage.get("prompt_tokens", usage.get("total_tokens"))
                if type(tokens) is not int or tokens < 0:
                    raise ValueError()
                cost = Decimal(tokens) * Decimal("0.02") / 1_000_000
            else:
                if agent.api == "responses":
                    usage = {
                        "prompt_tokens": usage["input_tokens"],
                        "completion_tokens": usage["output_tokens"],
                        "prompt_tokens_details": usage.get("input_tokens_details") or {},
                    }
                if any(
                    type(usage[k]) is not int or usage[k] < 0 for k in ("prompt_tokens", "completion_tokens")
                ):
                    raise ValueError()
                details = usage.get("prompt_tokens_details") or {}
                if any(
                    type(details.get(k, 0)) is not int or details.get(k, 0) < 0
                    for k in ("cached_tokens", "cache_write_tokens")
                ):
                    raise ValueError()
                cost = agent.usage_cost(usage)
            ledger.settle(
                row["reservation"],
                cost,
                {"http_status": response.status_code, "request_id": row["request_id"]},
            )
            row["usage"] = usage
        except (ValueError, KeyError, TypeError):
            row["usage_uncertain"] = True

    async def async_send(client, request, *args, **kwargs):
        nonlocal active_requests
        if kwargs.get("stream"):
            return deny()
        row, agent = reserve(request)
        token = AUTHORIZED.set(True)
        active_requests += 1
        try:
            kwargs["follow_redirects"] = False
            if kwargs.get("stream"):
                return deny()
            response = await real_async(client, request, *args, **kwargs)
            settle(row, agent, response)
            if response.status_code >= 400:
                transport_errors.append("HTTP " + str(response.status_code))
            return response
        except Exception as exc:
            row["transport_error"] = type(exc).__name__
            transport_errors.append(type(exc).__name__)
            raise
        finally:
            active_requests -= 1
            AUTHORIZED.reset(token)

    def sync_send(client, request, *args, **kwargs):
        nonlocal active_requests
        if kwargs.get("stream"):
            return deny()
        row, agent = reserve(request)
        token = AUTHORIZED.set(True)
        active_requests += 1
        try:
            kwargs["follow_redirects"] = False
            if kwargs.get("stream"):
                return deny()
            response = real_sync(client, request, *args, **kwargs)
            settle(row, agent, response)
            if response.status_code >= 400:
                transport_errors.append("HTTP " + str(response.status_code))
            return response
        except Exception as exc:
            row["transport_error"] = type(exc).__name__
            transport_errors.append(type(exc).__name__)
            raise
        finally:
            active_requests -= 1
            AUTHORIZED.reset(token)

    def dns(host, port, *args, **kwargs):
        name = host.decode() if isinstance(host, bytes) else host
        if name not in {"localhost", "127.0.0.1", "::1", None} and not (
            (AUTHORIZED.get() or active_requests > 0) and name == "api.openai.com" and int(port) == 443
        ):
            return deny()
        return real_dns(host, port, *args, **kwargs)

    def check(address):
        if isinstance(address, tuple) and address[0] not in {"localhost", "127.0.0.1", "::1"}:
            if not AUTHORIZED.get() or int(address[1]) != 443:
                deny()

    def connect(sock, address):
        check(address)
        return real_connect(sock, address)

    def connect_ex(sock, address):
        check(address)
        return real_connect_ex(sock, address)

    def ainit(client, *args, **kwargs):
        kwargs["trust_env"] = False
        return real_async_init(client, *args, **kwargs)

    def sinit(client, *args, **kwargs):
        kwargs["trust_env"] = False
        return real_sync_init(client, *args, **kwargs)

    with ExitStack() as stack:
        for cls, name, fn in (
            (httpx.AsyncClient, "send", async_send),
            (httpx.Client, "send", sync_send),
            (httpx.AsyncClient, "__init__", ainit),
            (httpx.Client, "__init__", sinit),
            (socket, "getaddrinfo", dns),
            (socket.socket, "connect", connect),
            (socket.socket, "connect_ex", connect_ex),
        ):
            stack.enter_context(patch.object(cls, name, fn))
        stack.enter_context(patch("urllib.request.urlopen", deny))
        try:
            sessions = importlib.import_module("requests.sessions")
        except ImportError:
            pass
        else:
            stack.enter_context(patch.object(sessions.Session, "request", deny))
        yield
    if transport_errors and not failed_limits:
        # Application workers may catch provider errors; do not grade those as model quality.
        raise RuntimeError("Eval provider transport failed: " + transport_errors[-1])
    if failed_limits:
        # Some application jobs catch provider errors. Still report budget termination accurately.
        raise EvalLimit(failed_limits[-1])
