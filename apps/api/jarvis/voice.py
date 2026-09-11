"""Server-owned Realtime sessions. Media is enabled only after sideband readiness."""

import asyncio
import hashlib
import json
import logging
import secrets
import time
from typing import Annotated
from urllib.parse import quote
from uuid import UUID

import httpx
import websockets
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select

from . import budget
from .auth import Identity, authenticate
from .config import get_settings
from .db import session_scope
from .domain import DomainError, capture_source, enqueue_job, owned, preferences
from .models import Conversation, Source, uid
from .tools import call_tool, instructions, registry

router = APIRouter()
User = Annotated[Identity, Depends(authenticate)]
controllers = {}

GATE = """Classify the latest speech in conversational context. Return exactly one word:
SILENT: a final thanks, backchannel, filler or remark needing no response.
WAIT: an incomplete thought, or explicit request to pause/let the speaker think.
END_SESSION: explicitly finished talking, with no remaining request.
RESPOND: a genuine request, question or conversational statement inviting a reply.
Do not execute tools, answer the question, or add punctuation. 'I wonder what is on tomorrow'
can be a genuine request. Do not suppress requests just because they are indirect."""


class VoiceInput(BaseModel):
    conversation_id: UUID
    sdp: str = Field(min_length=1, max_length=64000)
    focus: str | None = None


class Controller:
    def __init__(self, owner, device, conversation_id, focus, preferences):
        self.id, self.owner, self.device = uid(), owner, device
        self.conversation_id, self.focus, self.preferences = conversation_id, focus, preferences
        self.call_id, self.ws = None, None
        self.epoch, self.current_item, self.tool_index, self.turns = 0, None, 0, 0
        self.allowed, self.responses, self.seen_items = {}, {}, set()
        self.ready, self.closed = asyncio.Event(), False
        self.state, self.error, self.last_text = "connecting", None, ""
        self.started = self.activity = self.client_seen = time.monotonic()
        self.receiver, self.timer, self.action = None, None, None
        self.model = get_settings().realtime_model
        self.send_lock = asyncio.Lock()
        self.receipts = []

    async def send(self, event):
        if self.ws and not self.closed:
            async with self.send_lock:
                await self.ws.send(json.dumps(event))

    async def request(self, phase, **response):
        # 128k input tokens at the highest audio rate, plus capped output and transcription headroom.
        # Pending/cancelled responses retain their allowance until provider completion.
        with session_scope() as db:
            budget.ensure_room(db, self.owner, self.id, (len(self.allowed) + len(self.responses) + 1) * 4.3)
        nonce = secrets.token_hex(16)
        self.allowed[nonce] = {"phase": phase, "epoch": self.epoch, "created": time.monotonic()}
        await self.send(
            {
                "type": "response.create",
                "response": {
                    "metadata": {"jarvis_request": nonce},
                    "max_output_tokens": 128 if phase == "gate" else 1024,
                    **response,
                },
            }
        )

    async def interrupt(self):
        self.epoch += 1
        self.activity = time.monotonic()
        self.state = "listening"
        for response_id in list(self.responses):
            await self.send({"type": "response.cancel", "response_id": response_id})
        await self.send({"type": "output_audio_buffer.clear"})
        # Committed domain effects are retained. Only future stale actions are rejected.

    async def run(self):
        settings = get_settings()
        try:
            async with websockets.connect(
                f"wss://api.openai.com/v1/realtime?call_id={quote(self.call_id)}",
                additional_headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                open_timeout=15,
                max_size=2_000_000,
            ) as ws:
                self.ws = ws
                await self.send(
                    {
                        "type": "session.update",
                        "session": {
                            "type": "realtime",
                            "instructions": instructions(self.preferences, self.focus),
                            "tools": registry(),
                            "tool_choice": "auto",
                            "max_output_tokens": 1024,
                            "audio": {
                                "input": {
                                    "turn_detection": {
                                        "type": "semantic_vad",
                                        "eagerness": "low",
                                        "create_response": False,
                                        "interrupt_response": True,
                                    },
                                    "transcription": {"model": "gpt-4o-mini-transcribe"},
                                },
                                "output": {"voice": "marin"},
                            },
                        },
                    }
                )
                async for message in ws:
                    event = json.loads(message)
                    await self.event(event)
                    if self.closed:
                        break
        except Exception as exc:  # noqa: BLE001 - isolate failures without logging speech or credentials
            logging.getLogger("jarvis.voice").warning("Realtime sideband ended (%s)", type(exc).__name__)
            if not self.closed:
                self.error, self.state = "Voice connection ended. Your saved work is safe.", "disconnected"
        finally:
            if not self.closed:
                await self.close(uncertain=True)

    async def event(self, event):
        kind = event.get("type")
        if kind == "session.updated":
            if not self.ready.is_set():
                # Restore a compact text history; never replay prior function calls.
                with session_scope() as db:
                    conv = owned(db, Conversation, self.conversation_id, self.owner)
                    history = (
                        []
                        if conv.private
                        else list(
                            db.scalars(
                                select(Source)
                                .where(Source.conversation_id == conv.id, Source.deleted_at.is_(None))
                                .order_by(Source.created_at.desc())
                                .limit(8)
                            )
                        )
                    )
                for source in reversed(history):
                    await self.send(
                        {
                            "type": "conversation.item.create",
                            "item": {
                                "type": "message",
                                "role": source.role if source.role in {"user", "assistant"} else "user",
                                "content": [
                                    {
                                        "type": "input_text" if source.role != "assistant" else "text",
                                        "text": source.content[:1500],
                                    }
                                ],
                            },
                        }
                    )
                self.state = "listening"
                self.ready.set()
        elif kind == "input_audio_buffer.speech_started":
            await self.interrupt()
        elif kind == "input_audio_buffer.committed":
            item = event["item_id"]
            if item in self.seen_items:
                return
            self.seen_items.add(item)
            self.current_item, self.tool_index = item, 0
            self.turns += 1
            self.state, self.activity = "evaluating", time.monotonic()
            await self.request(
                "gate",
                conversation="none",
                output_modalities=["text"],
                tools=[],
                tool_choice="none",
                instructions=GATE,
            )
        elif kind == "conversation.item.input_audio_transcription.completed":
            with session_scope() as db:
                conv = owned(db, Conversation, self.conversation_id, self.owner)
                source = capture_source(
                    db,
                    self.owner,
                    event.get("transcript", ""),
                    f"voice:{self.id}:{event['item_id']}",
                    conversation=conv,
                )
                if source and conv.learning:
                    enqueue_job(db, self.owner, "extract_memory", {"source_id": source.id})
        elif kind == "response.created":
            response = event["response"]
            nonce = response.get("metadata", {}).get("jarvis_request")
            request = self.allowed.pop(nonce, None)
            if request is None or request["epoch"] != self.epoch:
                await self.send({"type": "response.cancel", "response_id": response["id"]})
            else:
                self.responses[response["id"]] = request
        elif kind == "response.done":
            response = event["response"]
            request = self.responses.pop(response["id"], None)
            usage = response.get("usage")
            if usage:
                with session_scope() as db:
                    budget.record_usage(
                        db,
                        self.owner,
                        self.id,
                        response["id"],
                        self.model,
                        usage,
                        budget.realtime_cost(usage, self.model),
                    )
            if not request or request["epoch"] != self.epoch or response.get("status") != "completed":
                return
            # Keep receiving speech/cancel events while asynchronous tool reads are pending.
            self.action = asyncio.create_task(self.finish_response(response, request))
        elif kind == "error":
            code = event.get("error", {}).get("code", "")
            if code not in {"response_cancel_not_active", "output_audio_buffer_clear_empty"}:
                self.error = "Voice could not complete that turn. Tap submit or use text."
                self.state = "unresolved"

    async def finish_response(self, response, request):
        epoch, phase = request["epoch"], request["phase"]
        if epoch != self.epoch or self.closed:
            return
        outputs = response.get("output", [])
        content = "".join(
            c.get("text", c.get("transcript", "")) for item in outputs for c in item.get("content", [])
        )
        try:
            if phase == "gate":
                decision = content.strip()
                if decision == "END_SESSION":
                    await self.close()
                elif decision in {"WAIT", "SILENT"}:
                    self.state = "waiting" if decision == "WAIT" else "listening"
                elif decision == "RESPOND":
                    self.state = "thinking"
                    await self.request("plan", output_modalities=["text"])
                else:
                    self.state, self.error = "unresolved", "Tap submit to handle this turn."
            elif phase == "plan":
                calls = [o for o in outputs if o.get("type") == "function_call"]
                if calls:
                    self.state = "acting"
                    for call in calls:
                        if self.epoch != epoch or self.closed:
                            return
                        try:
                            if self.tool_index >= 4:
                                raise DomainError("LIMIT_EXCEEDED", "This turn reached its tool limit.")
                            turn_id = hashlib.sha256(f"{self.id}:{self.current_item}".encode()).hexdigest()[
                                :40
                            ]
                            result = await call_tool(
                                self.owner,
                                turn_id,
                                self.tool_index,
                                call["name"],
                                json.loads(call["arguments"]),
                            )
                            self.receipts.append(result.get("command_id"))
                        except (DomainError, ValueError) as exc:
                            result = {"error": getattr(exc, "code", "INVALID_ARGUMENT"), "message": str(exc)}
                        self.tool_index += 1
                        await self.send(
                            {
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "function_call_output",
                                    "call_id": call["call_id"],
                                    "output": json.dumps(result),
                                },
                            }
                        )
                    if self.epoch == epoch:
                        await self.request(
                            "plan",
                            output_modalities=["text"],
                            tool_choice="none" if self.tool_index >= 4 else "auto",
                        )
                else:
                    # Planning/tool turns are text-only. Audible confirmation begins only after receipts commit.
                    self.last_text, self.state = content, "speaking"
                    with session_scope() as db:
                        conv = owned(db, Conversation, self.conversation_id, self.owner)
                        capture_source(
                            db,
                            self.owner,
                            content,
                            f"voice:{self.id}:{response['id']}",
                            role="assistant",
                            conversation=conv,
                        )
                    await self.request(
                        "speak",
                        conversation="none",
                        output_modalities=["audio"],
                        tools=[],
                        tool_choice="none",
                        input=[],
                        instructions="Read this response naturally and exactly, without additions: "
                        + content[:4000],
                    )
            elif phase == "speak":
                self.state, self.activity = "listening", time.monotonic()
        except Exception:  # noqa: BLE001 - isolate provider/process failures without exposing personal data
            self.state, self.error = (
                "unresolved",
                "That turn could not finish. Check your saved tasks or use text.",
            )

    async def watch(self):
        while not self.closed:
            await asyncio.sleep(1)
            current = time.monotonic()
            # A live browser renews this lease through status polls. This is
            # orphan cleanup, not an elapsed-time, speech-turn, or silence cap.
            if current - self.client_seen > 30:
                await self.close()
            elif any(current - r["created"] > 20 for r in [*self.responses.values(), *self.allowed.values()]):
                await self.interrupt()
                self.allowed.clear()
                self.state, self.error = "unresolved", "Tap submit to retry this turn, or use text."

    async def close(self, uncertain=False):
        if self.closed:
            return
        uncertain = uncertain or bool(self.responses or self.allowed)
        self.closed = True
        self.state = "closed"
        if self.call_id:
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    response = await client.post(
                        f"https://api.openai.com/v1/realtime/calls/{quote(self.call_id)}/hangup",
                        headers={"Authorization": f"Bearer {get_settings().openai_api_key}"},
                    )
                    response.raise_for_status()
            except httpx.HTTPError:
                uncertain = True
        if self.ws:
            await self.ws.close()
        with session_scope() as db:
            budget.close(db, self.owner, self.id, uncertain=uncertain)
        # Closed controllers are removed after the client has a chance to observe shutdown.
        asyncio.get_running_loop().call_later(60, lambda: controllers.pop(self.id, None))


def control(session_id, user):
    c = controllers.get(session_id)
    if c is None or c.owner != user.owner_id or c.device != user.device_id:
        raise DomainError("NOT_FOUND", "Voice session ended. Start a new session.", 404)
    c.client_seen = time.monotonic()
    return c


@router.post("/voice/sessions")
async def start(body: VoiceInput, user: User):
    settings = get_settings()
    if not settings.openai_api_key:
        raise DomainError(
            "INTEGRATION_UNAVAILABLE",
            "Realtime needs an OpenAI API key. Text and task controls are ready.",
            503,
        )
    for existing in list(controllers.values()):
        if existing.device == user.device_id and not existing.closed:
            await existing.close()
    with session_scope() as db:
        conv = owned(db, Conversation, str(body.conversation_id), user.owner_id)
        if conv.device_id != user.device_id:
            raise DomainError("NOT_AUTHORIZED", "Conversation belongs to another device.", 403)
        prefs = preferences(db, user.owner_id)
        c = Controller(user.owner_id, user.device_id, conv.id, body.focus, prefs)
        budget.reserve(db, user.owner_id, c.id, 5, settings.realtime_model)
    controllers[c.id] = c
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(
                "https://api.openai.com/v1/realtime/calls",
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                files={
                    "sdp": (None, body.sdp),
                    "session": (
                        None,
                        json.dumps(
                            {
                                "type": "realtime",
                                "model": settings.realtime_model,
                                "audio": {
                                    "input": {
                                        "turn_detection": {
                                            "type": "semantic_vad",
                                            "create_response": False,
                                            "interrupt_response": True,
                                        }
                                    },
                                    "output": {"voice": "marin"},
                                },
                            }
                        ),
                    ),
                },
            )
            response.raise_for_status()
        c.call_id = response.headers["location"].rstrip("/").split("/")[-1]
        c.receiver = asyncio.create_task(c.run())
        c.timer = asyncio.create_task(c.watch())
        await asyncio.wait_for(c.ready.wait(), 20)
        return {"session_id": c.id, "sdp": response.text, "ready": True}
    except Exception:  # noqa: BLE001 - isolate provider/process failures without exposing personal data
        await c.close(uncertain=True)
        raise DomainError(
            "INTEGRATION_UNAVAILABLE", "Voice could not connect. Use text while the connection recovers.", 503
        )


@router.get("/voice/sessions/{session_id}")
async def status(session_id: str, user: User):
    c = control(session_id, user)
    return {
        "state": c.state,
        "error": c.error,
        "text": c.last_text,
        "closed": c.closed,
        "receipts": [r for r in c.receipts[-10:] if r],
    }


@router.post("/voice/sessions/{session_id}/stop")
async def stop(session_id: str, user: User):
    await control(session_id, user).close()
    return {"closed": True}


@router.post("/voice/sessions/{session_id}/interrupt")
async def interrupt(session_id: str, user: User):
    await control(session_id, user).interrupt()
    return {"interrupted": True}


@router.post("/voice/sessions/{session_id}/submit")
async def submit(session_id: str, user: User):
    c = control(session_id, user)
    if not c.current_item or c.closed:
        raise DomainError("INVALID_ARGUMENT", "Speak first, then submit.")
    await c.interrupt()
    c.state = "thinking"
    await c.request("plan", output_modalities=["text"])
    return {"submitted": True}
