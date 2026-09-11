"""Server-owned Realtime sessions. Media is enabled only after sideband readiness."""

import asyncio
import hashlib
import json
import logging
import secrets
import time
from typing import Annotated, Literal
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
from .memory_service import prompt_context
from .models import Conversation, Source, now, uid
from .tools import call_tool, instructions, registry
from .ui_control import get_context
from .voice_options import OPTIONS

router = APIRouter()
User = Annotated[Identity, Depends(authenticate)]
controllers = {}

GATE = """Classify the latest speech in conversational context. Return exactly one word:
SILENT: a backchannel, filler or remark needing no response.
WAIT: an incomplete thought, or explicit request to pause/let the speaker think.
END_SESSION: a standalone goodbye, closing thank you/thanks, or explicitly finished talking, with no remaining request. A thanks followed by a new request is RESPOND. Quoted farewell words are not a request to end voice.
RESPOND: a genuine request, question or conversational statement inviting a reply.
Do not execute tools, answer the question, or add punctuation. 'I wonder what is on tomorrow'
can be a genuine request. Do not suppress requests just because they are indirect."""


class VoiceInput(BaseModel):
    conversation_id: UUID
    sdp: str = Field(min_length=1, max_length=64000)
    focus: str | None = None
    provider: Literal["realtime", "live"] = "realtime"
    voice: str = "marin"


class Controller:
    def __init__(self, owner, device, conversation_id, focus, preferences):
        self.id, self.owner, self.device = uid(), owner, device
        self.conversation_id, self.focus, self.preferences = conversation_id, focus, preferences
        self.call_id, self.ws = None, None
        self.epoch, self.current_item, self.tool_index, self.turns = 0, None, 0, 0
        self.plan_rounds = 0
        self.allowed, self.responses, self.seen_items = {}, {}, set()
        self.ready, self.closed = asyncio.Event(), False
        self.closing = False
        self.close_lock = asyncio.Lock()
        self.state, self.error, self.last_text = "connecting", None, ""
        self.started = self.activity = self.client_seen = time.monotonic()
        self.receiver, self.timer, self.action = None, None, None
        self.model = get_settings().realtime_model
        self.send_lock = asyncio.Lock()
        self.receipts = []
        self.ui_actions = []
        self.voice = "marin"
        self.provider = "realtime"
        self.text_id = None
        self.input_started = {}
        self.answered_items = set()
        self.transcripts = {}
        self.memory_context = ""
        self.ui_context = {}
        self.unreported = set()
        self.pending_transcription = set()

    async def send(self, event):
        if self.ws and not self.closed:
            async with self.send_lock:
                await self.ws.send(json.dumps(event))

    def can_submit(self):
        return bool(
            self.current_item
            and self.current_item not in self.answered_items
            and self.tool_index == 0
            and self.state in {"listening", "waiting", "unresolved"}
            and not self.closed
        )

    async def request(self, phase, **response):
        if self.closed or self.closing:
            return
        if phase == "plan":
            self.plan_rounds += 1
            if self.plan_rounds > get_settings().max_model_rounds_per_request:
                response["tool_choice"] = "none"
        # 128k input tokens at the highest audio rate, plus capped output and transcription headroom.
        # Pending/cancelled responses retain their allowance until provider completion.
        with session_scope() as db:
            budget.ensure_room(db, self.owner, self.id, (len(self.allowed) + len(self.responses) + 1) * 4.3)
        self.error = None
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
        self.error = None
        self.epoch += 1
        self.activity = time.monotonic()
        self.state = "listening"
        for response_id in list(self.responses):
            await self.send({"type": "response.cancel", "response_id": response_id})
        await self.send({"type": "output_audio_buffer.clear"})
        # Committed domain effects are retained. Only future stale actions are rejected.

    async def run(self):
        settings = get_settings()
        self.memory_context = await prompt_context(self.owner)
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
                            "instructions": instructions(
                                self.preferences, self.focus, get_context(self.owner, self.device)
                            )
                            + "\n"
                            + self.memory_context,
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
                                    "transcription": {"model": "gpt-live-transcribe", "delay": "low"},
                                },
                                "output": {"voice": self.voice},
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
            if not self.closed and not self.closing:
                await self.close(uncertain=True)

    async def event(self, event):
        kind = event.get("type")
        if self.closing and kind not in {
            "response.created",
            "response.done",
            "conversation.item.input_audio_transcription.completed",
        }:
            return
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
            self.input_started.setdefault(event["item_id"], now())
            self.pending_transcription.add(event["item_id"])
            await self.interrupt()
        elif kind == "input_audio_buffer.committed":
            item = event["item_id"]
            if item in self.seen_items:
                return
            self.seen_items.add(item)
            self.current_item, self.tool_index = item, 0
            self.plan_rounds = 0
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
            self.transcripts[event["item_id"]] = event.get("transcript", "")
            usage = event.get("usage", {})
            cost = budget.transcription_cost(usage)
            if cost is not None:
                with session_scope() as db:
                    budget.record_usage(
                        db,
                        self.owner,
                        self.id,
                        f"{self.id}:transcription:{event['item_id']}",
                        "gpt-live-transcribe",
                        usage,
                        cost,
                    )
                self.pending_transcription.discard(event["item_id"])
            with session_scope() as db:
                conv = owned(db, Conversation, self.conversation_id, self.owner)
                source = capture_source(
                    db,
                    self.owner,
                    event.get("transcript", ""),
                    f"voice:{self.id}:{event['item_id']}",
                    conversation=conv,
                )
                if source and event["item_id"] in self.input_started:
                    source.created_at = self.input_started[event["item_id"]]
                if source and conv.learning:
                    enqueue_job(db, self.owner, "extract_memory", {"source_id": source.id})
        elif kind == "response.created":
            response = event["response"]
            nonce = response.get("metadata", {}).get("jarvis_request")
            request = self.allowed.pop(nonce, None)
            if nonce in self.unreported:
                self.unreported.discard(nonce)
                self.unreported.add(response["id"])
            if request is None:
                self.unreported.add(response["id"])
            if request is not None:
                self.responses[response["id"]] = request
            if request is None or request["epoch"] != self.epoch or self.closing:
                await self.send({"type": "response.cancel", "response_id": response["id"]})
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
            if not usage:
                self.unreported.add(response["id"])
            else:
                self.unreported.discard(response["id"])
            if not request or request["epoch"] != self.epoch:
                return
            if response.get("status") != "completed":
                if response.get("status") == "failed":
                    self.state, self.error = (
                        "unresolved",
                        "That response did not finish. Speak again or choose Respond now.",
                    )
                return
            self.error = None
            # Keep receiving speech/cancel events while asynchronous tool reads are pending.
            self.action = asyncio.create_task(self.finish_response(response, request))
        elif kind == "error":
            code = event.get("error", {}).get("code", "")
            if code not in {"response_cancel_not_active", "output_audio_buffer_clear_empty"}:
                safe_code = "".join(c for c in str(code)[:80] if c.isalnum() or c in "_.-")
                logging.getLogger("jarvis.voice").warning("Realtime provider event (%s)", safe_code)
                self.error = "That response hit a problem. Speak again or choose Respond now."
                self.state = "unresolved"

    async def finish_response(self, response, request):
        epoch, phase = request["epoch"], request["phase"]
        if epoch != self.epoch or self.closed or self.closing:
            return
        self.error = None
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
                    # Give transcription a short chance to arrive; keep the receiver free
                    # to accept speech interrupts while retrieval runs.
                    item = self.current_item
                    for _ in range(10):
                        if item in self.transcripts or self.epoch != epoch:
                            break
                        await asyncio.sleep(0.1)
                    self.memory_context = await prompt_context(self.owner, self.transcripts.get(item, ""))
                    if self.epoch != epoch or self.closed or self.closing:
                        return
                    await self.send(
                        {
                            "type": "session.update",
                            "session": {
                                "type": "realtime",
                                "instructions": instructions(
                                    self.preferences, self.focus, get_context(self.owner, self.device)
                                )
                                + "\n"
                                + self.memory_context,
                            },
                        }
                    )
                    await self.request("plan", output_modalities=["text"])
                else:
                    self.state, self.error = "unresolved", "Choose Respond now to handle this turn."
            elif phase == "plan":
                calls = [o for o in outputs if o.get("type") == "function_call"]
                if calls:
                    self.state = "acting"
                    for call in calls:
                        if self.epoch != epoch or self.closed or self.closing:
                            return
                        try:
                            if (
                                self.tool_index >= get_settings().max_tool_calls_per_request
                                or self.plan_rounds > get_settings().max_model_rounds_per_request
                            ):
                                raise DomainError(
                                    "LIMIT_EXCEEDED",
                                    "The request allowance was reached. Report saved changes and unfinished work separately.",
                                )
                            turn_id = hashlib.sha256(f"{self.id}:{self.current_item}".encode()).hexdigest()[
                                :40
                            ]
                            result = await call_tool(
                                self.owner,
                                turn_id,
                                self.tool_index,
                                call["name"],
                                json.loads(call["arguments"]),
                                device=self.device,
                            )
                            self.receipts.append(result.get("command_id"))
                            if result.get("ui_action"):
                                self.ui_actions.append(result["ui_action"])
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
                            tool_choice="none"
                            if self.tool_index >= get_settings().max_tool_calls_per_request
                            else "auto",
                        )
                else:
                    # Planning/tool turns are text-only. Audible confirmation begins only after receipts commit.
                    self.last_text, self.state = content, "speaking"
                    self.text_id = f"voice:{self.id}:{response['id']}"
                    self.answered_items.add(self.current_item)
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
            with session_scope() as db:
                budget.touch(db, self.owner, self.id)
            # A live browser renews this lease through status polls. This is
            # orphan cleanup, not an elapsed-time, speech-turn, or silence cap.
            if current - self.client_seen > 30:
                await self.close()
            elif any(current - r["created"] > 20 for r in [*self.responses.values(), *self.allowed.values()]):
                await self.interrupt()
                self.unreported.update(self.allowed)
                self.allowed.clear()
                self.state, self.error = "unresolved", "Speak again or choose Respond now to retry this turn."

    async def close(self, uncertain=False):
        async with self.close_lock:
            if self.closed:
                return
            self.closing, self.state = True, "closing"
            self.epoch += 1
            # Keep sideband alive briefly to collect usage for cancelled responses.
            # A browser may already have hung up media; a 404 then means "already ended".
            for response_id in list(self.responses):
                try:
                    await self.send({"type": "response.cancel", "response_id": response_id})
                except Exception:  # noqa: BLE001 - the remaining reservation covers uncertain usage
                    break
            deadline = time.monotonic() + 2
            if asyncio.current_task() != self.receiver:
                while (
                    self.responses or self.allowed or self.pending_transcription
                ) and time.monotonic() < deadline:
                    await asyncio.sleep(0.05)
            unsettled = bool(self.responses or self.allowed or self.unreported or self.pending_transcription)
            self.closed, self.state = True, "closed"
            if self.call_id:
                try:
                    async with httpx.AsyncClient(timeout=5) as client:
                        response = await client.post(
                            f"https://api.openai.com/v1/realtime/calls/{quote(self.call_id)}/hangup",
                            headers={"Authorization": f"Bearer {get_settings().openai_api_key}"},
                        )
                        if response.status_code not in {404, 409}:
                            response.raise_for_status()
                except httpx.HTTPError:
                    # Realtime bills generated/input tokens, not connected silence.
                    # Keep a hold for unresolved responses, never for unused idle headroom.
                    logging.getLogger("jarvis.voice").info(
                        "Realtime hangup was already disconnected or unavailable"
                    )
            if self.ws:
                await self.ws.close()
            with session_scope() as db:
                budget.close(db, self.owner, self.id, uncertain=unsettled)
            for task in (self.receiver, self.timer):
                if task and task != asyncio.current_task():
                    task.cancel()
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
    if body.voice not in OPTIONS[body.provider]["voices"]:
        raise DomainError("INVALID_ARGUMENT", "Choose a voice supported by this provider.")
    # Authorize the requested conversation before affecting an existing session.
    with session_scope() as db:
        conv = owned(db, Conversation, str(body.conversation_id), user.owner_id)
        if conv.device_id != user.device_id:
            raise DomainError("NOT_AUTHORIZED", "Conversation belongs to another device.", 403)
    for existing in list(controllers.values()):
        if existing.device == user.device_id and not existing.closed:
            await existing.close()
    with session_scope() as db:
        conv = owned(db, Conversation, str(body.conversation_id), user.owner_id)
        if conv.device_id != user.device_id:
            raise DomainError("NOT_AUTHORIZED", "Conversation belongs to another device.", 403)
        prefs = preferences(db, user.owner_id)
        if body.provider == "live":
            from .live_voice import LiveController

            c = LiveController(user.owner_id, user.device_id, conv.id, body.focus, prefs)
        else:
            c = Controller(user.owner_id, user.device_id, conv.id, body.focus, prefs)
        c.voice = body.voice
        budget.reserve(db, user.owner_id, c.id, 0.10 if body.provider == "live" else 5, c.model)
    controllers[c.id] = c
    try:
        if body.provider == "live":
            return await c.connect(body.sdp)
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
                                    "output": {"voice": c.voice},
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
    except DomainError:
        await c.close(uncertain=True)
        raise
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
        "provider": c.provider,
        "ui_actions": c.ui_actions[-30:],
        "error": c.error,
        "text": c.last_text,
        "text_id": c.text_id,
        "can_submit": c.can_submit(),
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
    if not c.can_submit():
        raise DomainError("INVALID_ARGUMENT", "Wait until Eri is listening, then speak a new request.")
    await c.interrupt()
    c.state = "thinking"
    await c.request("plan", output_modalities=["text"])
    return {"submitted": True}
