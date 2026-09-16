import asyncio
import hashlib
import json

import httpx
from sqlalchemy import select

from . import agent_models, budget
from .config import get_settings
from .db import session_scope
from .domain import DomainError, advisory, capture_source, enqueue_job, owned, preferences
from .memory_service import prompt_context
from .models import Conversation, Job, Source, now
from .tool_catalog import ToolSession
from .tools import call_tool, instructions, registry
from .ui_control import get_context


def turn_hash(conversation_id, message, focus):
    return hashlib.sha256(json.dumps([conversation_id, message, focus]).encode()).hexdigest()


async def chat(
    owner, device, turn_id, conversation_id, message, focus=None, *, live_context=None, tool_guard=None, end_voice=None
):
    from .access import assert_current
    assert_current(owner,device)
    settings = get_settings()
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
        from .access import person_preferences
        prefs = person_preferences(db,owner,device,preferences(db, owner))
        private = private or not prefs["history_enabled"]
        # Pin the route for this entire turn, even if Settings changes during a tool call.
        agent = agent_models.selected(prefs, require_key=True)
        model = agent.model
        budget.reserve(db, owner, turn_id, 0.10, model)
        job = Job(
            id=turn_id,
            owner_id=owner,
            kind="chat",
            status="running",
            payload={
                "request_hash": key,
                "conversation_id": conv.id,
                "provider": agent.provider,
                "profile": agent.profile_id,
                "model": model,
            },
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
        memory_context = "" if prefs.get("shared_workspace") else await prompt_context(owner, memory_query)
        assert_current(owner,device)
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
    tool_session = ToolSession(registry())
    voice_ended = False
    if end_voice is not None:
        from .voice_control import VOICE_END_POLICY, VOICE_END_TOOL

        messages[0]["content"] += "\n" + VOICE_END_POLICY
        tool_session.catalog["voice_end"] = VOICE_END_TOOL
        tool_session.names.append("voice_end")
    actions, tool_index = [], 0
    ui_actions = []
    reply, failed, limited = "", False, False
    cancelled, provider_pending = False, False
    tool_errors = []
    try:
        async with httpx.AsyncClient(
            timeout=60 if agent.provider == "gemini" or agent.api == "responses" else 35
        ) as client:
            for step in range(settings.max_model_rounds_per_request + 1):
                assert_current(owner,device)
                final_round = step == settings.max_model_rounds_per_request
                if final_round or tool_index >= settings.max_tool_calls_per_request:
                    messages.append(
                        {
                            "role": "system",
                            "content": "The application tool/round allowance is reached. Do not request more tools. Summarize only verified saved results and explicitly identify unfinished work.",
                        }
                    )
                    limited = True
                tools = [
                    {"type": "function", "function": {k: v for k, v in t.items() if k != "type"}}
                    for t in tool_session.definitions()
                ]
                offered_tools = set(tool_session.names)
                # Byte count is a conservative token upper bound; reserve each continuation.
                input_bound = len(json.dumps([messages, tools], ensure_ascii=False).encode()) + 1024
                if input_bound > 250000:
                    raise DomainError(
                        "LIMIT_EXCEEDED",
                        "This request needs a narrower scope. Saved actions remain available.",
                    )
                with session_scope() as db:
                    budget.ensure_room(
                        db,
                        owner,
                        turn_id,
                        agent.reserve_cost(input_bound),
                    )
                provider_pending = True
                response = await client.post(
                    agent.endpoint,
                    headers={"Authorization": f"Bearer {agent.api_key}"},
                    json=agent.request(messages, tools, limited),
                )
                if 400 <= response.status_code < 500 and response.status_code != 408:
                    provider_pending = False  # Explicit rejection, not an unknown timeout.
                response.raise_for_status()
                assert_current(owner,device)
                data = agent.normalize(response.json())
                usage = data.get("usage")
                if (
                    not isinstance(usage, dict)
                    or "prompt_tokens" not in usage
                    or "completion_tokens" not in usage
                ):
                    raise ValueError("Provider response omitted usage")
                cost = agent.usage_cost(usage)
                with session_scope() as db:
                    budget.record_usage(db, owner, turn_id, data["id"], model, usage, cost)
                provider_pending = False
                msg = data["choices"][0]["message"]
                # Keep native Responses items and Gemini thought signatures intact in
                # request memory only. Neither becomes another transcript store.
                fields = {"role", "content", "tool_calls", "extra_content"}
                messages.append({k: v for k, v in msg.items() if k in fields})
                if data["choices"][0].get("finish_reason") == "length":
                    raise DomainError("OUTPUT_TRUNCATED",
                        "The task model ran out of output space before finishing. Already saved changes remain; this unfinished response was not executed.")
                calls = msg.get("tool_calls") or []
                if not calls:
                    reply = msg.get("content")
                    if not isinstance(reply, str) or not reply.strip():
                        raise ValueError("Provider returned no answer or tool call")
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
                        try:
                            args = json.loads(fn["arguments"])
                            if not isinstance(args, dict):
                                raise TypeError("Tool arguments must be an object")
                        except (ValueError, TypeError):
                            raise DomainError("MALFORMED_TOOL_ARGUMENTS",
                                "The model returned malformed tool arguments. No action was executed for this call; retry with valid JSON.") from None
                        if tool_guard:
                            changed = batch_guard_error or tool_guard()
                            if changed:
                                batch_guard_error = changed
                                raise DomainError("REQUEST_CHANGED", changed)
                        if fn["name"] not in offered_tools:
                            raise DomainError(
                                "TOOL_NOT_LOADED",
                                "Load the matching group with tools_load before calling this tool.",
                            )
                        if fn["name"] == "voice_end" and end_voice is not None:
                            if args:
                                raise DomainError("INVALID_ARGUMENT", "voice_end takes no arguments.")
                            assert_current(owner, device)
                            outcome = end_voice()
                            voice_ended = True
                        elif fn["name"] == "tools_load":
                            outcome = tool_session.load(args)
                        else:
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
                        {
                            "role": "tool",
                            "name": fn["name"],
                            "tool_call_id": call["id"],
                            "content": json.dumps(outcome),
                        }
                    )
                    if voice_ended:
                        reply = "Voice conversation ended."
                        break
                if voice_ended:
                    break
    except asyncio.CancelledError:
        cancelled = True
        reply = "Voice ended. Already saved actions remain; no further actions were started."
    except DomainError as exc:
        reply = exc.message
        limited = True
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
        failed = True
        reply = (
            f"The connection to {agent.label} failed before a usable response arrived. Already saved changes remain; check their receipts before retrying."
            if isinstance(exc, httpx.HTTPError)
            else f"{agent.label} returned an invalid response format. Already saved changes remain; this is a response-format error, not a lost connection."
        )
        if isinstance(exc, httpx.HTTPStatusError):
            code = exc.response.status_code
            if code in {401, 403, 404}:
                reply = f"{agent.label} could not be accessed. Check its API key and model access, or choose another task agent in Settings. Already saved changes remain."
            elif code == 429:
                reply = f"{agent.label} reached its provider quota or rate limit. Try again later or choose another task agent in Settings. Already saved changes remain."
    if not reply:
        reply = "I could not finish this batch. Previously saved changes remain; some requested work is unfinished."
        limited = True
    result = {
        "profile": agent.profile_id,
        "provider": agent.provider,
        "model": model,
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
        "voice_ended": voice_ended,
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
