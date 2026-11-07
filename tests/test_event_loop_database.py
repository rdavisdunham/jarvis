"""Database waits in the response guard and change feed must leave the event loop available."""
import threading
from contextlib import contextmanager
from datetime import timedelta
from types import SimpleNamespace

import pytest
from fastapi import Request
from fastapi.responses import JSONResponse
from jarvis import access, api
from jarvis.auth import Identity
from jarvis.domain import DomainError
from jarvis.models import now


@pytest.mark.asyncio
async def test_response_access_recheck_runs_off_the_event_loop(monkeypatch):
    loop_thread = threading.get_ident()
    checked = []

    def authenticate(request):
        checked.append(threading.get_ident())
        assert threading.get_ident() != loop_thread

    async def next_response(request):
        return JSONResponse({"ok": True})

    monkeypatch.setattr(api, "authenticate", authenticate)
    request = Request({"type": "http", "method": "GET", "path": "/api/v1/bootstrap", "headers": []})
    response = await api.security(request, next_response)
    assert response.status_code == 200
    assert len(checked) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["valid", "expired", "switched", "revoked"])
async def test_event_feed_reads_off_loop_and_closes_transaction_before_sending(monkeypatch, state):
    loop_thread = threading.get_ident()
    transactions = []
    session = SimpleNamespace(expires_at=now() + timedelta(hours=1), device_id="device")
    event = SimpleNamespace(id=17)

    class Database:
        def get(self, *args):
            return None if state == "expired" else session

        def scalars(self, query):
            return [event]

    @contextmanager
    def session_scope():
        assert threading.get_ident() != loop_thread
        transactions.append("opened")
        try:
            yield Database()
        finally:
            transactions.append("closed")

    def identity(db, row):
        if state == "revoked":
            raise DomainError("ACCESS_REVOKED", "Access ended", 403)
        return ("other" if state == "switched" else "owner"), "owner"

    async def connected():
        return False

    monkeypatch.setattr(api, "session_scope", session_scope)
    monkeypatch.setattr(api, "serial", lambda row: {"id": row.id})
    monkeypatch.setattr(access, "identity", identity)
    request = SimpleNamespace(headers={}, cookies={"jarvis_session": "synthetic"}, is_disconnected=connected)
    response = await api.events(request, Identity("owner", "device", "csrf"))
    stream = response.body_iterator
    try:
        assert "connected" in await anext(stream)
        if state == "expired":
            with pytest.raises(StopAsyncIteration):
                await anext(stream)
        else:
            chunk = await anext(stream)
            expected = "id: 17" if state == "valid" else "event: access_revoked"
            assert expected in chunk
            if state == "switched":
                assert "WORKSPACE_CHANGED" in chunk
        assert transactions == ["opened", "closed"]
    finally:
        await stream.aclose()
