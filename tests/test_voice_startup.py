"""Offline creation, pooling and cleanup contracts; no provider connections."""
import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from jarvis import live_http
from jarvis.db import session_scope
from jarvis.domain import DomainError
from jarvis.models import BudgetReservation
from test_live_voice import controller as controller


async def test_http_pool_is_lazy_bounded_reused_and_closed(monkeypatch):
    created=[]
    class Client:
        is_closed=False
        def __init__(self, **options): self.options=options;created.append(self)
        async def aclose(self): self.is_closed=True
    monkeypatch.setattr(live_http,"_client",None)
    monkeypatch.setattr(live_http.httpx,"AsyncClient",Client)
    assert created==[]
    first=live_http.client();assert live_http.client() is first
    assert first.options["limits"].max_connections==10
    assert first.options["limits"].keepalive_expiry==10
    await live_http.close();assert first.is_closed and live_http._client is None
    assert live_http.client() is not first
    await live_http.close()


async def test_late_created_provider_is_closed_without_reviving_cancelled_controller(controller,monkeypatch):
    c=controller
    entered,release=asyncio.Event(),asyncio.Event()
    class Client:
        async def post(self,*args,**kwargs):
            entered.set();await release.wait()
            return httpx.Response(200,json={"session":{"id":"late-created"},"transport":{"sdp":"answer"}})
    monkeypatch.setattr(live_http,"client",lambda:Client())
    monkeypatch.setattr("jarvis.live_voice.prompt_context",AsyncMock(return_value=""))
    monkeypatch.setattr("jarvis.access.assert_current",lambda *args:None)
    c.close_late_creation=AsyncMock()
    task=asyncio.create_task(c.connect("synthetic"));await entered.wait()
    await c.close();assert c.closed
    release.set()
    with pytest.raises(DomainError,match="cancelled"):
        await task
    c.close_late_creation.assert_awaited_once()
    assert c.closed and c.receiver is None and c.timer is None


async def test_late_creation_cleanup_reads_final_usage_without_reopening_media(controller,monkeypatch):
    c=controller;c.closed=True;c.closing=True;c.call_id="late-created"
    sent=[]
    class Socket:
        async def send(self,value): sent.append(json.loads(value))
        def __aiter__(self):return self
        async def __anext__(self):
            if getattr(self,"done",False):raise StopAsyncIteration
            self.done=True
            return json.dumps({"type":"session.closed","usage":{"seconds":42}})
    class Connection:
        async def __aenter__(self):return Socket()
        async def __aexit__(self,*args):pass
    monkeypatch.setattr("jarvis.live_voice.websockets.connect",lambda *a,**k:Connection())
    await c.close_late_creation()
    assert [event["type"] for event in sent]==["session.close"]
    assert c.closed and c.seconds==42
    with session_scope() as db:
        assert db.get(BudgetReservation,c.id).state=="closed"


@pytest.mark.parametrize("account,name",[("first","Rowan"),("second","Mara")])
async def test_startup_greeting_uses_only_current_account_and_cancellation_is_monotonic(controller,account,name):
    from uuid import uuid4
    from jarvis.models import OwnerSettings,UserAccount
    c=controller;c.account=account;c.attempt_id=str(uuid4());c.ready.set()
    c.end_policy.text="Hey Eri";c.last_user_delta=0
    with session_scope() as db:
        db.add(UserAccount(id=account,name=name))
        db.add(OwnerSettings(owner_id=account,values={"preferred_name":name}))
    assert (await c.startup_control(c.attempt_id,1,"ready"))["accepted"]
    content=c.send.await_args.args[0]["content"]
    assert name in content and ("Mara" if name=="Rowan" else "Rowan") not in content
    assert not (await c.startup_control(c.attempt_id,1,"ready"))["accepted"]
    assert (await c.startup_control(c.attempt_id,2,"cancel"))["accepted"]
    assert not (await c.startup_control(c.attempt_id,1,"ready"))["accepted"]
    assert not (await c.startup_control(str(uuid4()),3,"ready"))["accepted"]
    assert len([a for a in c.send.await_args_list if a.args[0]["type"]=="session.commentary.append"])==1
    await c.close()


async def test_startup_cancel_before_ready_and_substantive_speech_never_greet(controller):
    from uuid import uuid4
    c=controller;c.attempt_id=str(uuid4());c.ready.set();c.end_policy.text="Eri add milk";c.last_user_delta=0
    assert not (await c.startup_control(c.attempt_id,1,"ready"))["accepted"]
    assert (await c.startup_control(c.attempt_id,2,"cancel"))["accepted"]
    c.end_policy.text="Eri"
    assert not (await c.startup_control(c.attempt_id,2,"ready"))["accepted"]
    c.send.assert_not_awaited();await c.close()


def test_startup_schema_forbids_client_prompt_name_and_foreign_target():
    from pydantic import ValidationError
    from jarvis.voice import StartupInput
    from uuid import uuid4
    for field in ("prompt","preferred_name","owner","device_id","session_id"):
        with pytest.raises(ValidationError):
            StartupInput(attempt_id=uuid4(),activity_epoch=1,state="ready",**{field:"untrusted"})


def test_voice_controls_are_account_workspace_and_device_bound(controller,monkeypatch):
    from jarvis import voice
    from jarvis.auth import Identity
    c=controller;c.account="account-a";monkeypatch.setitem(voice.controllers,c.id,c)
    for owner,account,device in [(c.owner,"account-b",c.device),("foreign-workspace","account-a",c.device),(c.owner,"account-a","foreign-device")]:
        with pytest.raises(DomainError,match="ended"):
            voice.control(c.id,Identity(owner,device,"csrf",account))
    assert voice.control(c.id,Identity(c.owner,c.device,"csrf","account-a")) is c


async def test_candidate_stop_acknowledges_before_graceful_close_finishes(controller,monkeypatch):
    from jarvis import voice
    c=controller
    monkeypatch.setattr(voice,"control",lambda *args:c)
    monkeypatch.setattr(c,"confirm_end",lambda identity:identity=="fresh")
    release=asyncio.Event()
    async def closing():await release.wait()
    original_close=c.close;c.close=AsyncMock(side_effect=closing)
    result=await voice.stop(c.id,None,{"candidate_id":"fresh"})
    assert result=={"closed":False,"accepted":True,"status":"closing"}
    await asyncio.sleep(0);c.close.assert_awaited_once();release.set();await asyncio.sleep(0)
    c.close=original_close;await c.close()


async def test_status_exposes_remaining_candidate_lifetime_without_transcript(controller,monkeypatch):
    import time
    from jarvis import voice
    c=controller;c.end_candidate={"id":"opaque","source":{"sequence":1,"event_id":"e","end_ms":100},"category":"confirm"}
    c.end_candidate_deadline=time.monotonic()+2
    monkeypatch.setattr(voice,"control",lambda *args:c)
    candidate=(await voice.status(c.id,None))["end_candidate"]
    assert 0<candidate["expires_in_ms"]<=2000 and "text" not in candidate
    c.end_candidate_deadline=0
    assert (await voice.status(c.id,None))["end_candidate"] is None
    await c.close()
