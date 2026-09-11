import hashlib
import json

import httpx
from sqlalchemy import select

from . import budget
from .config import get_settings
from .db import session_scope
from .domain import DomainError, advisory, capture_source, enqueue_job, owned, preferences
from .models import Conversation, Job, Source, now
from .tools import call_tool, instructions, registry


def turn_hash(conversation_id, message, focus):
    return hashlib.sha256(json.dumps([conversation_id, message, focus]).encode()).hexdigest()


async def chat(owner, device, turn_id, conversation_id, message, focus=None):
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
        source = capture_source(db, owner, message, f"chat:{turn_id}:user", conversation=conv)
        if source and conv.learning:
            enqueue_job(db, owner, "extract_memory", {"source_id": source.id})
    messages = [
        {"role": "system", "content": instructions(prefs, focus)},
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
    reply, failed = "", False
    try:
        async with httpx.AsyncClient(timeout=35) as client:
            for step in range(5):
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
                response = await client.post(
                    endpoint,
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": model,
                        "messages": messages,
                        "tools": tools,
                        "tool_choice": "none" if step == 4 else "auto",
                        "max_completion_tokens": 1200,
                    },
                )
                response.raise_for_status()
                data = response.json()
                usage = data.get("usage", {})
                rates = (0.75, 4.5) if settings.openai_api_key else (0.15, 0.60)
                cost = (
                    usage.get("prompt_tokens", 0) * rates[0] + usage.get("completion_tokens", 0) * rates[1]
                ) / 1_000_000
                with session_scope() as db:
                    budget.record_usage(db, owner, turn_id, data["id"], model, usage, cost)
                msg = data["choices"][0]["message"]
                messages.append({k: v for k, v in msg.items() if k in {"role", "content", "tool_calls"}})
                calls = msg.get("tool_calls", [])
                if not calls:
                    reply = (
                        msg.get("content")
                        or "I couldn't finish that response. Your saved changes are still visible."
                    )
                    break
                for call in calls:
                    fn = call["function"]
                    try:
                        if tool_index >= 4:
                            raise DomainError("LIMIT_EXCEEDED", "This request reached its action limit.")
                        args = json.loads(fn["arguments"])
                        outcome = await call_tool(owner, turn_id, tool_index, fn["name"], args)
                        # Store references only; retrieved personal context is not another transcript store.
                        if outcome.get("command_id"):
                            actions.append({"command_id": outcome["command_id"], "status": outcome["status"]})
                    except (DomainError, ValueError) as exc:
                        outcome = {
                            "error": getattr(exc, "code", "INVALID_ARGUMENT"),
                            "message": str(exc),
                            "data": getattr(exc, "data", None),
                        }
                    tool_index += 1
                    messages.append(
                        {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(outcome)}
                    )
    except DomainError as exc:
        reply = exc.message
    except (httpx.HTTPError, ValueError, KeyError):
        failed = True
        reply = "I lost the model connection. Any changes already saved are still in your task list; I haven't repeated them."
    result = {
        "turn_id": turn_id,
        "message": reply,
        "actions": actions,
        "status": "failed" if failed else "succeeded",
    }
    with session_scope() as db:
        job = db.get(Job, turn_id)
        job.status, job.finished_at = result["status"], now()
        private = private or not preferences(db, owner)["history_enabled"]
        job.result = {**result, "message": "Private response was not retained."} if private else result
        conv = owned(db, Conversation, conversation_id, owner)
        if not private:
            capture_source(db, owner, reply, f"chat:{turn_id}:assistant", role="assistant", conversation=conv)
        budget.close(db, owner, turn_id, uncertain=failed)
    return result
