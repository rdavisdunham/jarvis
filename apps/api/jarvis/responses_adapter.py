"""Stateless Responses transport for the existing durable conversation/tool loop."""


def request(agent, messages, tools, limited):
    inputs = []
    for message in messages:
        native = (message.get("extra_content") or {}).get("responses_output")
        if native is not None:
            # Replay the entire output, preserving encrypted reasoning, IDs and order.
            inputs.extend(native)
        elif message["role"] == "tool":
            inputs.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": message["content"],
                }
            )
        else:
            inputs.append({"role": message["role"], "content": message["content"]})
    return {
        "model": agent.model,
        "input": inputs,
        "tools": [{"type": "function", **tool["function"], "strict": False} for tool in tools],
        "tool_choice": "none" if limited else "auto",
        "reasoning": {"effort": agent.reasoning_effort},
        "max_output_tokens": agent.max_output_tokens,
        "store": False,
        "include": ["reasoning.encrypted_content"],
    }


def normalize(data):
    usage = data.get("usage")
    if not isinstance(usage, dict) or "input_tokens" not in usage or "output_tokens" not in usage:
        raise ValueError("Responses omitted usage")
    status = data.get("status")
    if status not in {"completed", "incomplete"}:
        raise ValueError("Responses did not complete")
    output = data["output"]
    if not isinstance(output, list):
        raise TypeError("Responses omitted output items")
    calls, text = [], []
    for item in output:
        if item["type"] == "function_call":
            if status == "completed" and item.get("status", "completed") == "completed":
                calls.append(
                    {
                        "id": item["call_id"],
                        "type": "function",
                        "function": {"name": item["name"], "arguments": item["arguments"]},
                    }
                )
        elif item["type"] == "message":
            for part in item["content"]:
                if part["type"] == "output_text":
                    text.append(part["text"])
                elif part["type"] == "refusal":
                    text.append(part["refusal"])
        elif item["type"] != "reasoning":
            raise ValueError("Responses returned an unsupported output item")
    return {
        "id": data["id"],
        "usage": {
            "prompt_tokens": usage["input_tokens"],
            "completion_tokens": usage["output_tokens"],
            "prompt_tokens_details": usage.get("input_tokens_details") or {},
            "completion_tokens_details": usage.get("output_tokens_details") or {},
        },
        "choices": [
            {
                "finish_reason": "length" if status == "incomplete" else "tool_calls" if calls else "stop",
                "message": {
                    "role": "assistant",
                    "content": "\n".join(text) or None,
                    "tool_calls": calls,
                    "extra_content": {"responses_output": output},
                },
            }
        ],
    }
