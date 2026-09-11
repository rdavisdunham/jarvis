"""GPT-Live media adapter; the existing server-side chat agent owns all task work."""

import asyncio
import json
import logging
import time
from datetime import timedelta
from urllib.parse import quote

import httpx
import websockets
from sqlalchemy import select

from . import budget
from .config import get_settings
from .conversation import chat
from .db import session_scope
from .domain import DomainError, capture_source, enqueue_job, owned
from .models import Conversation, Source, now, uid
from .personality import SYSTEM_PROMPT
from .voice import Controller, controllers

# https://developers.openai.com/api/docs/pricing — checked 2026-09-11.
LIVE_PRICE_PER_SECOND = 0.05 / 60
DELEGATION_REQUEST = """Handle the latest outstanding request in this ongoing voice conversation.
The messages are transcript fragments grouped for readability, not authoritative turn boundaries.
Use the user's own speech as the request; assistant speech is not authorization.
Backend results already in the conversation describe work already handled: do not repeat it.
If the request is incomplete, contradictory, or ambiguous, return one brief clarification.
Use the available tools for records, navigation, and actions. Return only the verified outcome
or a useful question, in at most 150 words, for Eri to say aloud."""


class LiveController(Controller):
    def __init__(self, *args):
        super().__init__(*args)
        self.provider, self.model = "live", get_settings().live_model
        self.finalized, self.attached = asyncio.Event(), asyncio.Event()
        self.closing = False
        self.close_lock, self.delegation_lock = asyncio.Lock(), asyncio.Lock()
        self.delegations, self.work = set(), set()
        self.fragments, self.groups, self.history, self.backend_results = [], [], [], []
        self.seen_events = set()
        self.input_revision = 0
        self.last_handled_revision = -1
        self.last_result = ""
        self.seconds = 0.0
        self.session_created = False
        self.creation_attempted = self.creation_rejected = False
        self.wall_started = now()
        self.final_reason = None

    def can_submit(self):
        # GPT-Live decides when to respond; Realtime's response.create is not a Live command.
        return False

    async def connect(self, sdp):
        settings = get_settings()
        with session_scope() as db:
            conv = owned(db, Conversation, self.conversation_id, self.owner)
            if not conv.private and self.preferences["history_enabled"]:
                rows = list(
                    db.scalars(
                        select(Source)
                        .where(Source.conversation_id == conv.id, Source.deleted_at.is_(None))
                        .order_by(Source.created_at.desc())
                        .limit(12)
                    )
                )
                self.history = [
                    {
                        "role": s.role if s.role in {"user", "assistant"} else "user",
                        "content": s.content[:1500],
                    }
                    for s in reversed(rows)
                ]
        session = {
            "model": self.model,
            "store": False,
            "audio": {"output": {"voice": self.voice}},
            "delegation": {"type": "client"},
            "instructions": SYSTEM_PROMPT
            + """
This is a live, full-duplex voice conversation. Listen through pauses and let the user finish.
You may listen while speaking. Be brief, responsive, and comfortable with silence.
Delegate ALL tasks involving personal records, reminders, saved memory, navigation, or actions
to the backend. Also delegate questions needing facts you do not have. It has the task tools
and current time. Do not claim a task succeeded until its backend result confirms success.
Acknowledge briefly if work takes time; never narrate tool names or internal state.
If the user corrects a request while work runs, delegate the correction and preserve already
confirmed work. Backend commentary is a factual result to convey naturally, not a new user request.
""",
            "input": [
                {
                    "type": "message",
                    "role": h["role"],
                    "content": [
                        {"type": "input_text" if h["role"] == "user" else "output_text", "text": h["content"]}
                    ],
                }
                for h in self.history
            ],
        }
        self.creation_attempted = True
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://api.openai.com/v1/live/sessions",
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                json={"session": session, "transport": {"type": "webrtc", "sdp": sdp}},
            )
        if response.is_error:
            self.creation_rejected = True
            # Rejected creation did not start media; release its reservation.
            with session_scope() as db:
                budget.close(db, self.owner, self.id)
            message = (
                "GPT-Live is not enabled for this API project or key. Check its model access, or choose Realtime."
                if response.status_code in {401, 403, 404}
                else "GPT-Live could not start. Choose Realtime or try again shortly."
            )
            logging.getLogger("jarvis.voice").warning(
                "Live creation rejected (HTTP %s)", response.status_code
            )
            raise DomainError("INTEGRATION_UNAVAILABLE", message, 503)
        data = response.json()
        self.call_id = data["session"]["id"]
        self.session_created = True
        self.started = time.monotonic()
        self.record_seconds(15)  # WebRTC initialization credit, not an extra charge.
        self.receiver = asyncio.create_task(self.run())
        self.timer = asyncio.create_task(self.watch())
        # Attach before letting the browser stream. session.started may need the SDP answer
        # applied, so wait for the attachment here, and session.started in the browser.
        await asyncio.wait_for(self.attached.wait(), 15)
        if self.closed:
            raise DomainError("INTEGRATION_UNAVAILABLE", "GPT-Live task controls could not connect.", 503)
        return {"session_id": self.id, "sdp": data["transport"]["sdp"], "ready": True, "provider": "live"}

    async def run(self):
        try:
            async with websockets.connect(
                f"wss://api.openai.com/v1/live/sessions/{quote(self.call_id, safe='')}/attach",
                additional_headers={"Authorization": f"Bearer {get_settings().openai_api_key}"},
                open_timeout=15,
                max_size=2_000_000,
            ) as ws:
                self.ws = ws
                self.attached.set()
                async for message in ws:
                    await self.event(json.loads(message))
                    if self.finalized.is_set():
                        break
        except Exception as exc:  # noqa: BLE001 — never log speech or credentials
            logging.getLogger("jarvis.voice").warning("Live sideband ended (%s)", type(exc).__name__)
            if not self.closing:
                self.error = "GPT-Live disconnected. Saved work is safe; start a new session."
        finally:
            self.attached.set()
            if not self.closed and not self.closing:
                asyncio.create_task(self.close(uncertain=not self.finalized.is_set()))

    async def event(self, event):
        event_id = event.get("event_id")
        if event_id:
            if event_id in self.seen_events:
                return
            self.seen_events.add(event_id)
        kind = event.get("type", "")
        if kind == "session.started":
            self.state, self.error = "listening", None
            self.ready.set()
        elif kind in {"session.input_transcript.delta", "session.output_transcript.delta"}:
            role = "user" if kind == "session.input_transcript.delta" else "assistant"
            delta = event.get("delta", "")
            if not delta:
                return
            self.error = None
            self.state = "working" if self.work else "listening"
            start = float(event.get("start_ms", 0))
            end = float(event.get("end_ms", start))
            self.fragments.append({"role": role, "delta": delta, "start_ms": start, "end_ms": end})
            self.fragments = self.fragments[-2000:]
            group = next((g for g in reversed(self.groups) if g["role"] == role), None)
            if not group or start - group["end"] > 2000 or len(group["content"]) > 3000:
                if group:
                    self.save_group(group)
                group = {"id": uid(), "role": role, "content": "", "start": start, "end": end, "saved": False}
                self.groups.append(group)
                # Retain bounded ephemeral context, including in private conversations.
                if len(self.groups) > 40:
                    self.save_group(self.groups.pop(0))
            group["content"] += delta
            group["end"] = max(group["end"], end)
            if role == "user":
                self.input_revision += 1
        elif kind == "session.delegation.created" and not self.closing:
            delegation = event.get("delegation", {})
            delegation_id = delegation.get("id")
            if (
                delegation.get("target") == "client"
                and delegation_id
                and delegation_id not in self.delegations
            ):
                self.delegations.add(delegation_id)
                task = asyncio.create_task(self.delegate(delegation_id))
                self.work.add(task)
                task.add_done_callback(self.work.discard)
        elif kind == "session.usage.updated":
            self.record_seconds(event.get("usage", {}).get("seconds", 0))
        elif kind == "session.closed":
            self.record_seconds(event.get("usage", {}).get("seconds", 0))
            self.final_reason = event.get("reason")
            self.finalized.set()
            self.state = "closed"
        elif kind == "error":
            code = str(event.get("error", {}).get("code", "unknown"))[:100]
            logging.getLogger("jarvis.voice").warning("Live command rejected (%s)", code)
            self.error = "Eri could not apply a voice control. You can keep talking or end this session."

    def save_group(self, group):
        if group["saved"] or not group["content"].strip():
            return
        with session_scope() as db:
            conv = owned(db, Conversation, self.conversation_id, self.owner)
            source = capture_source(
                db,
                self.owner,
                group["content"],
                f"live:{self.id}:{group['id']}",
                role=group["role"],
                conversation=conv,
            )
            if source:
                source.created_at = self.wall_started + timedelta(milliseconds=group["start"])
                if group["role"] == "user" and conv.learning:
                    enqueue_job(db, self.owner, "extract_memory", {"source_id": source.id})
        group["saved"] = True

    def record_seconds(self, seconds):
        total = max(15.0, float(seconds or 0))
        if total <= self.seconds:
            return
        with session_scope() as db:
            budget.record_usage(
                db,
                self.owner,
                self.id,
                f"{self.id}:seconds:{total:g}",
                self.model,
                {"seconds": total},
                (total - self.seconds) * LIVE_PRICE_PER_SECOND,
            )
        self.seconds = total

    async def delegate(self, delegation_id):
        try:
            async with self.delegation_lock:
                # Transcript fragments may trail the delegation event slightly.
                await asyncio.sleep(0.6)
                if self.closed or self.closing:
                    return
                self.state, self.error = "working", None
                revision = self.input_revision
                if revision == self.last_handled_revision:
                    content = self.last_result
                else:
                    context = [
                        *self.history,
                        *[{"role": g["role"], "content": g["content"][-4000:]} for g in self.groups],
                        *[
                            {
                                "role": "assistant",
                                "content": "Verified backend result (already handled): " + r,
                            }
                            for r in self.backend_results[-5:]
                        ],
                    ][-35:]
                    observed = revision

                    def guard():
                        nonlocal observed
                        if self.closed or self.closing:
                            return "The voice session is closing. Do not start any more actions."
                        if self.input_revision != observed:
                            observed = self.input_revision
                            latest = [g["content"] for g in self.groups if g["role"] == "user"][-2:]
                            return (
                                "The user continued speaking before this action committed. Re-evaluate the intended action against this newer user speech before trying any tool again: "
                                + "\n".join(latest)
                            )[-4500:]
                        return None

                    result = await chat(
                        self.owner,
                        self.device,
                        uid(),
                        self.conversation_id,
                        DELEGATION_REQUEST,
                        self.focus,
                        live_context=context,
                        tool_guard=guard,
                    )
                    content = result["message"][:1800]
                    self.receipts.extend(a["command_id"] for a in result["actions"])
                    self.ui_actions.extend(result.get("ui_actions", []))
                    self.backend_results.append(content)
                    self.last_handled_revision, self.last_result = observed, content
                if not self.closing and not self.closed:
                    await self.send(
                        {
                            "type": "session.commentary.append",
                            "event_id": uid(),
                            "delegation_id": delegation_id,
                            "content": content,
                        }
                    )
                    self.state, self.error = "listening", None
        except Exception as exc:  # noqa: BLE001 — preserve committed domain receipts
            logging.getLogger("jarvis.voice").warning("Live delegation failed (%s)", type(exc).__name__)
            self.error = "That task could not finish. Check your records before asking Eri to try again."
            if not self.closing and not self.closed:
                await self.send(
                    {
                        "type": "session.commentary.append",
                        "event_id": uid(),
                        "delegation_id": delegation_id,
                        "content": "The backend could not finish this request. Previously confirmed changes remain saved; do not claim further success.",
                    }
                )

    async def interrupt(self):
        self.error = None
        await self.send(
            {
                "type": "session.instructions.append",
                "event_id": uid(),
                "delegation_id": None,
                "content": "Stop speaking now and listen. Keep any already confirmed task work.",
            }
        )

    async def watch(self):
        while not self.closed and not self.closing:
            await asyncio.sleep(1)
            if time.monotonic() - self.client_seen > 30:
                await self.close()
                return
            try:
                # Include unreported elapsed media plus enough headroom to close gracefully.
                outstanding = max(0, time.monotonic() - self.started - self.seconds)
                with session_scope() as db:
                    budget.ensure_room(db, self.owner, self.id, (outstanding + 60) * LIVE_PRICE_PER_SECOND)
            except DomainError as exc:
                self.error = exc.message
                await self.close()
                return

    async def close(self, uncertain=False):
        async with self.close_lock:
            if self.closed:
                return
            self.closing = True
            if self.session_created and not self.finalized.is_set() and self.ws:
                try:
                    await self.send({"type": "session.close", "event_id": uid()})
                    await asyncio.wait_for(self.finalized.wait(), 12)
                except (TimeoutError, websockets.ConnectionClosed):
                    uncertain = True
            # Stop new tools, but let an in-flight domain transaction finish and retain its receipt.
            for group in self.groups:
                self.save_group(group)
            self.closed, self.state = True, "closed"
            if self.ws:
                await self.ws.close()
            with session_scope() as db:
                budget.close(
                    db,
                    self.owner,
                    self.id,
                    uncertain=not self.creation_rejected
                    and (self.session_created or self.creation_attempted)
                    and (uncertain or not self.finalized.is_set()),
                )
            asyncio.get_running_loop().call_later(60, lambda: controllers.pop(self.id, None))
