"""Bounded HTTP connection reuse, created lazily by a confirmed voice start.

This pool holds no provider media session. The application lifespan closes it
only after active controllers have settled their usage.
"""
import httpx

_client = None


def client():
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=30, limits=httpx.Limits(
            max_connections=10, max_keepalive_connections=5, keepalive_expiry=10))
    return _client


async def close():
    global _client
    previous, _client = _client, None
    if previous is not None:
        await previous.aclose()
