import asyncio
import hashlib
import json

import httpx
from sqlalchemy import select

from . import budget
from .config import get_settings
from .db import session_scope
from .domain import DomainError, advisory, capture_source, enqueue_job, owned, preferences
from .memory_service import prompt_context
from .models import Conversation, Job, Source, now
from .tools import call_tool, instructions, registry
from .ui_control import get_context


def turn_hash(conversation_id, message, focus):
    return hashlib.sha256(json.dumps([conversation_id, message, focus]).encode()).hexdigest()


async def chat(
    owner, device, turn_id, conversation_id, message, focus=None, *, live_context=None, tool_guard=None
):
    settings = get_settings()
    if not (settings.openai_api_key or settings.groq_api_key):
        raise DomainError(
            "INTEGRATION_UNAVAILABLE", "Add an API key to use chat. You can still manage tasks directly.", 503
        )
    with session_scope() as db:
        conv = owned(db, Conversation, conversation_id, owner)
        if conv.device_id != device:
            raise DomainError("NOT_AUTHORIZED", "This conversation belongs to another device.", 403)
        private = conv.private
        key = turn_hash(conversation_id, message, focus)
        advisory(db, f"chat:{owner}:{turn_id}")
        prior = db.get(Job, turn_id)
        if prior:
            if prior.owner_id != owner or prior.payload.get("request_hash") != key:
                raise DomainError("REVISION_CONFLICT", "This turn ID was already used.", 409)
            if prior.result:
                return prior.result
            raise DomainError(
                "IN_PROGRESS", "This request is still being resolved. Its saved actions remain visible.", 409
            )
        prefs = preferences(db, owner)
        private = private or not prefs["history_enabled"]
        model = settings.text_model if settings.openai_api_key else "openai/gpt-oss-120b"
        budget.reserve(db, owner, turn_id, 0.10, model)
        job = Job(
            id=turn_id,
            owner_id=owner,
            kind="chat",
            status="running",
            payload={"request_hash": key, "conversation_id": conv.id},
        )
        db.add(job)
        history = []
        if not private:
            rows = list(
                db.scalars(
                    select(Source)
                    .where(Source.conversation_id == conv.id, Source.deleted_at.is_(None))
                    .order_by(Source.created_at.desc())
                    .limit(12)
                )
            )
            history = [
                {"role": s.role if s.role in {"user", "assistant"} else "user", "content": s.content[:3000]}
                for s in reversed(rows)
            ]
        source = (
            None
            if live_context is not None
            else capture_source(db, owner, message, f"chat:{turn_id}:user", conversation=conv)
        )
        if source and conv.learning:
            enqueue_job(db, owner, "extract_memory", {"source_id": source.id})
    if live_context is not None:
        history = live_context
    memory_query = (
        message
        if live_context is None
        else " ".join(m["content"] for m in live_context if m["role"] == "user")[-1500:]
    )
    try:
        memory_context = await prompt_context(owner, memory_query)
    except asyncio.CancelledError:
        with session_scope() as db:
            job = db.get(Job, turn_id)
            job.status, job.finished_at = "cancelled", now()
            job.result = {
                "turn_id": turn_id,
                "status": "cancelled",
                "message": "Voice ended before any actions started.",
                "actions": [],
                "ui_actions": [],
            }
            budget.close(db, owner, turn_id)
        raise
    messages = [
        {
            "role": "system",
            "content": instructions(prefs, focus, get_context(owner, device)) + "\n" + memory_context,
        },
        *history,
        {"role": "user", "content": message},
    ]
    api_key = settings.openai_api_key or settings.groq_api_key
    endpoint = (
        "https://api.openai.com/v1/chat/completions"
        if settings.openai_api_key
        else "https://api.groq.com/openai/v1/chat/completions"
    )
    tools = [
        {"type": "function", "function": {k: v for k, v in t.items() if k != "type"}} for t in registry()
    ]
    actions, tool_index = [], 0
    ui_actions = []
    reply, failed, limited = "", False, False
    cancelled, provider_pending = False, False
    tool_errors = []
    try:
        async with httpx.AsyncClient(timeout=35) as client:
            for step in range(settings.max_model_rounds_per_request + 1):
                final_round = step == settings.max_model_rounds_per_request
                if final_round or tool_index >= settings.max_tool_calls_per_request:
                    messages.append(
                        {
                            "role": "system",
                            "content": "The application tool/round allowance is reached. Do not request more tools. Summarize only verified saved results and explicitly identify unfinished work.",
                        }
                    )
                    limited = True
                # Byte count is a conservative token upper bound; reserve each continuation.
                input_bound = len(json.dumps([messages, tools], ensure_ascii=False).encode()) + 1024
                if input_bound > 250000:
                    raise DomainError(
                        "LIMIT_EXCEEDED",
                        "This request needs a narrower scope. Saved actions remain available.",
                    )
                rates = (0.75, 4.5) if settings.openai_api_key else (0.15, 0.60)
                with session_scope() as db:
                    budget.ensure_room(
                        db, owner, turn_id, (input_bound * rates[0] + 1200 * rates[1]) / 1_000_000
                    )
                provider_pending = True
                response = await client.post(
                    endpoint,
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": model,
                        "messages": messages,
                        "tools": tools,
                        "tool_choice": "none" if limited else "auto",
                        "max_completion_tokens": 1200,
                    },
                )
                if 400 <= response.status_code < 500 and response.status_code != 408:
                    provider_pending = False  # Explicit rejection, not an unknown timeout.
                response.raise_for_status()
                data = response.json()
                usage = data.get("usage")
                if (
                    not isinstance(usage, dict)
                    or "prompt_tokens" not in usage
                    or "completion_tokens" not in usage
                ):
                    raise ValueError("Provider response omitted usage")
                rates = (0.75, 4.5) if settings.openai_api_key else (0.15, 0.60)
                cost = (
                    usage.get("prompt_tokens", 0) * rates[0] + usage.get("completion_tokens", 0) * rates[1]
                ) / 1_000_000
                with session_scope() as db:
                    budget.record_usage(db, owner, turn_id, data["id"], model, usage, cost)
                provider_pending = False
                msg = data["choices"][0]["message"]
                messages.append({k: v for k, v in msg.items() if k in {"role", "content", "tool_calls"}})
                calls = msg.get("tool_calls", [])
                if not calls:
                    reply = (
                        msg.get("content")
                        or "I couldn't finish that response. Your saved changes are still visible."
                    )
                    break
                batch_guard_error = None
                for call in calls:
                    fn = call["function"]
                    try:
                        if final_round or tool_index >= settings.max_tool_calls_per_request:
                            limited = True
                            raise DomainError(
                                "LIMIT_EXCEEDED",
                                f"This request reached its allowance of {settings.max_tool_calls_per_request} tool calls or {settings.max_model_rounds_per_request} model rounds. Saved changes remain; report any unfinished work.",
                            )
                        args = json.loads(fn["arguments"])
                        if tool_guard:
                            changed = batch_guard_error or tool_guard()
                            if changed:
                                batch_guard_error = changed
                                raise DomainError("REQUEST_CHANGED", changed)
                        outcome = await call_tool(
                            owner,
                            turn_id,
                            tool_index,
                            fn["name"],
                            args,
                            device=device,
                            conversation_id=conversation_id,
                        )
                        if outcome.get("ui_action"):
                            ui_actions.append(outcome["ui_action"])
                        # Store references only; retrieved personal context is not another transcript store.
                        if outcome.get("command_id"):
                            actions.append({"command_id": outcome["command_id"], "status": outcome["status"]})
                    except (DomainError, ValueError) as exc:
                        outcome = {
                            "error": getattr(exc, "code", "INVALID_ARGUMENT"),
                            "message": str(exc),
                            "data": getattr(exc, "data", None),
                        }
                    if outcome.get("error"):
                        tool_errors.append({"tool": fn["name"], **outcome})
                    tool_index += 1
                    messages.append(
                        {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(outcome)}
                    )
    except asyncio.CancelledError:
        cancelled = True
        reply = "Voice ended. Already saved actions remain; no further actions were started."
    except DomainError as exc:
        reply = exc.message
        limited = True
    except (httpx.HTTPError, ValueError, KeyError):
        failed = True
        reply = "I lost the model connection. Any changes already saved are still in your task list; I haven't repeated them."
    if not reply:
        reply = "I could not finish this batch. Previously saved changes remain; some requested work is unfinished."
        limited = True
    result = {
        "turn_id": turn_id,
        "message": reply,
        "actions": actions,
        "ui_actions": ui_actions,
        "status": "cancelled"
        if cancelled
        else "failed"
        if failed
        else "partial"
        if limited or tool_errors
        else "succeeded",
        "tool_calls": tool_index,
        "tool_errors": tool_errors,
        "limits": {
            "tool_calls": settings.max_tool_calls_per_request,
            "model_rounds": settings.max_model_rounds_per_request,
        },
    }
    with session_scope() as db:
        job = db.get(Job, turn_id)
        job.status, job.finished_at = result["status"], now()
        private = private or not preferences(db, owner)["history_enabled"]
        retained = {**result, "tool_errors": [{"tool": e["tool"], "error": e["error"]} for e in tool_errors]}
        job.result = (
            {**retained, "message": "Private response was not retained."}
            if private or live_context is not None
            else retained
        )
        conv = owned(db, Conversation, conversation_id, owner)
        if not private and live_context is None:
            capture_source(db, owner, reply, f"chat:{turn_id}:assistant", role="assistant", conversation=conv)
        budget.close(db, owner, turn_id, uncertain=provider_pending)
    if cancelled:
        raise asyncio.CancelledError()
    return result
