"""Opt-in real Gemini tool handshake, using synthetic context and no database writes.

Run from the repository root: uv run python scripts/check_gemini.py
"""

import asyncio
import json

import httpx
from jarvis.agent_models import selected
from jarvis.tools import registry


async def main():
    agent = selected({"agent_provider": "gemini"}, require_key=True)
    tools = [
        {"type": "function", "function": {k: v for k, v in tool.items() if k != "type"}}
        for tool in registry()
    ]
    messages = [
        {
            "role": "system",
            "content": "You are an integration test. Call task_list once, then report the synthetic task's title. Never call any other tool.",
        },
        {"role": "user", "content": "List the task in the test workspace."},
    ]
    async with httpx.AsyncClient(timeout=60) as client:
        body = agent.request(messages, tools)
        body["tool_choice"] = {"type": "function", "function": {"name": "task_list"}}
        first = await client.post(
            agent.endpoint, headers={"Authorization": f"Bearer {agent.api_key}"}, json=body
        )
        first.raise_for_status()
        message = first.json()["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        if not calls or any(c["function"]["name"] != "task_list" for c in calls):
            raise ValueError("Gemini did not return the requested task_list call.")
        messages.append(
            {k: v for k, v in message.items() if k in {"role", "content", "tool_calls", "extra_content"}}
        )
        for call in calls:
            json.loads(call["function"]["arguments"])
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": "task_list",
                    "content": json.dumps(
                        {
                            "tasks": [{"id": "synthetic-task", "title": "Gemini handshake", "revision": 1}],
                            "next_offset": None,
                        }
                    ),
                }
            )
        second = await client.post(
            agent.endpoint,
            headers={"Authorization": f"Bearer {agent.api_key}"},
            json=agent.request(messages, tools, limited=True),
        )
        second.raise_for_status()
        result = second.json()["choices"][0]
        if (
            result["message"].get("tool_calls")
            or "gemini handshake" not in (result["message"].get("content") or "").lower()
        ):
            raise ValueError("Gemini did not summarize the synthetic tool result.")
    print(
        "Gemini 3.8 Flash: tool catalog accepted; signed continuation and synthetic task result passed. No database or external writes."
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except httpx.HTTPStatusError as exc:
        # Never print authorization headers, a credential-bearing URL, or provider response text.
        raise SystemExit(
            f"Gemini returned HTTP {exc.response.status_code}; check model access, API key, and quota."
        ) from None
