import asyncio
from dataclasses import replace

import httpx
import pytest

from scripts.app_eval.ledger import Ledger
from scripts.app_eval.transport import category, metered


@pytest.fixture
def ledger(tmp_path):
    result = Ledger(tmp_path / "budget.sqlite")
    result.initialize(".2", ".02")
    return result


def reply(request):
    return httpx.Response(200, json={"usage": {"input_tokens": 100, "output_tokens": 20}})


def test_sync_and_async_calls_share_meter_and_real_attempt_counts(ledger):
    trace = []
    with metered(ledger, "t", trace):
        with httpx.Client(transport=httpx.MockTransport(reply)) as client:
            client.post(
                "https://api.openai.com/v1/responses",
                json={"model": "gpt-5.6-luna", "max_output_tokens": 2048, "input": []},
            )

        async def second():
            async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
                with category("judge"):
                    await client.post(
                        "https://api.openai.com/v1/responses",
                        json={"model": "gpt-5.6-luna", "max_output_tokens": 2048, "input": []},
                    )

        asyncio.run(second())
    snapshot = ledger.snapshot()
    assert snapshot["requests"] == 2 and len(trace) == 2
    assert {c["kind"] for c in snapshot["calls"]} == {"agent", "judge"}
    assert all(c["state"] == "recorded" for c in snapshot["calls"])


@pytest.mark.parametrize(
    "url,body",
    [
        ("https://example.org", {"model": "gpt-5.6-luna", "max_output_tokens": 100}),
        ("https://api.openai.com/v1/responses", {"model": "different", "max_output_tokens": 100}),
        ("https://api.openai.com/v1/responses", {"model": "gpt-5.6-luna", "max_output_tokens": 999999}),
        (
            "https://api.openai.com/v1/responses",
            {"model": "gpt-5.6-luna", "max_output_tokens": 100, "stream": True},
        ),
        (
            "https://api.openai.com/v1/responses",
            {"model": "gpt-5.6-luna", "max_output_tokens": 100, "tools": [{"type": "web_search"}]},
        ),
    ],
)
def test_unsupported_billing_paths_block_before_network(ledger, url, body):
    calls = []
    with (
        metered(ledger, "t", []),
        httpx.Client(transport=httpx.MockTransport(lambda r: calls.append(r))) as client,
    ):
        with pytest.raises(ValueError):
            client.post(url, json=body)
    assert not calls and ledger.snapshot()["requests"] == 0


def test_embedding_is_explicit_and_missing_usage_is_uncertain(ledger):
    with metered(ledger, "t", [], embeddings=True):
        with httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"data": []}))
        ) as client:
            client.post(
                "https://api.openai.com/v1/embeddings",
                json={"model": "text-embedding-3-small", "input": ["cat"]},
            )
    assert ledger.snapshot()["calls"][0]["state"] == "uncertain"
    assert float(ledger.snapshot()["including_uncertain_usd"]) > 0


def test_raw_socket_and_non_httpx_client_cannot_bypass_guard(ledger):
    import socket
    import urllib.request

    with metered(ledger, "t", []):
        with pytest.raises(ValueError):
            socket.getaddrinfo("example.org", 443)
        with pytest.raises(ValueError):
            urllib.request.urlopen("https://api.openai.com/v1/responses")
        with socket.socket() as sock, pytest.raises(ValueError):
            sock.connect(("8.8.8.8", 443))


def test_asyncio_dns_executor_can_resolve_only_authorized_provider(ledger, monkeypatch):
    import socket

    calls = []
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port, *a, **k: calls.append((host, port)) or [])

    async def respond(request):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, socket.getaddrinfo, b"api.openai.com", 443)
        with pytest.raises(ValueError):
            await loop.run_in_executor(None, socket.getaddrinfo, "example.org", 443)
        return reply(request)

    async def run():
        with metered(ledger, "dns", []):
            async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
                await client.post(
                    "https://api.openai.com/v1/responses",
                    json={"model": "gpt-5.6-luna", "max_output_tokens": 16, "input": "OK"},
                )
            with pytest.raises(ValueError):
                await asyncio.get_running_loop().run_in_executor(
                    None, socket.getaddrinfo, "api.openai.com", 443
                )

    asyncio.run(run())
    assert calls == [(b"api.openai.com", 443)]


def test_swallowed_transport_error_still_fails_worker_as_infrastructure(ledger):
    def fail(request):
        raise httpx.ConnectError("synthetic")

    with pytest.raises(RuntimeError, match="transport"):
        with metered(ledger, "t", []):
            with httpx.Client(transport=httpx.MockTransport(fail)) as client:
                try:
                    client.post(
                        "https://api.openai.com/v1/responses",
                        json={"model": "gpt-5.6-luna", "max_output_tokens": 16, "input": "OK"},
                    )
                except httpx.ConnectError:
                    pass
    assert ledger.snapshot()["calls"][0]["state"] == "uncertain"
