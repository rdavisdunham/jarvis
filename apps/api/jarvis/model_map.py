"""Which backend model each job type uses, and a rules-based router hook (docs/COST_TRACKING.md, "Model map").

Every backend job that calls a text model has an entry in JOBS: the cost feature its usage is recorded under
(cost_features.LABELS, the Settings → Model usage buckets), the production function that makes the call, and its
default profile. The defaults are today's models: Luna with low reasoning through memory_learning.extraction_request
for the structured jobs, and the task agent chosen in Settings (agent_models.selected) for Eri chat. Nothing in
production reaches a non-default profile yet: memory_learning.profile_request refuses any route that is not the
extraction model, exactly as extraction_request's allowlist does, because the budget ledger prices Luna only.

`route(job, prompt_tokens)` is the router hook: RULES (empty by default) are checked in order, by job type and by
prompt length, and the first match wins; otherwise MODEL_MAP decides. scripts/model_eval.py uses the same
profiles, request builders and parsers (`build`, `parse`) to compare candidates on recorded Eridani cases.
"""

import json
import math
from contextlib import contextmanager
from dataclasses import dataclass, field

from .cost_features import LABELS, current_feature

SELECTED = "selected"  # The Settings → task agent choice (agent_models.selected), not a fixed model.


@dataclass(frozen=True)
class Profile:
    id: str
    provider: str
    model: str
    api: str  # openai_responses | anthropic_messages | openai_chat
    endpoint: str
    key_env: str
    label: str
    reasoning_effort: str | None = None
    # openai_chat providers differ in structured-output support: json_schema, or json_object plus the schema in text.
    json_mode: str = "json_schema"
    placeholder: bool = False  # Model id and endpoint not verified; set them before a real run.
    service_tier: str = "default"  # Luna only: default (Standard) or fast.

    def public(self):
        return {"id": self.id, "provider": self.provider, "model": self.model, "api": self.api,
                "reasoning_effort": self.reasoning_effort, "service_tier": self.service_tier, "label": self.label,
                "placeholder": self.placeholder}


PROFILES = {
    p.id: p
    for p in (
        # Today's backend model for every structured job (memory_learning.EXTRACTION_MODEL, reasoning low).
        Profile("luna", "openai", "gpt-5.6-luna", "openai_responses", "https://api.openai.com/v1/responses",
                "OPENAI_API_KEY", "GPT-5.6 Luna · low reasoning (current backend)", "low"),
        Profile("luna-none", "openai", "gpt-5.6-luna", "openai_responses", "https://api.openai.com/v1/responses",
                "OPENAI_API_KEY", "GPT-5.6 Luna · no reasoning", "none"),
        # Eri chat's default: the Settings task agent is Luna, interactive requests at the Fast tier.
        Profile("luna-fast", "openai", "gpt-5.6-luna", "openai_responses", "https://api.openai.com/v1/responses",
                "OPENAI_API_KEY", "GPT-5.6 Luna · low reasoning · Fast tier (Eri chat default)", "low",
                service_tier="fast"),
        Profile("haiku-5.5", "anthropic", "claude-haiku-5-5", "anthropic_messages",
                "https://api.anthropic.com/v1/messages", "ANTHROPIC_API_KEY", "Claude Haiku 5.5 · low effort", "low"),
        Profile("sonnet-5.5", "anthropic", "claude-sonnet-5-5", "anthropic_messages",
                "https://api.anthropic.com/v1/messages", "ANTHROPIC_API_KEY", "Claude Sonnet 5.5 · low effort", "low"),
        # Generic OpenAI-compatible adapter. Model ids are placeholders: override them for a real run
        # (scripts/model_eval.py: MODEL_EVAL_<ID>_MODEL / _ENDPOINT / _PRICE).
        Profile("deepseek", "deepseek", "deepseek-chat", "openai_chat", "https://api.deepseek.com/chat/completions",
                "DEEPSEEK_API_KEY", "DeepSeek (placeholder)", json_mode="json_object", placeholder=True),
        Profile("kimi", "moonshot", "kimi-latest", "openai_chat", "https://api.moonshot.ai/v1/chat/completions",
                "MOONSHOT_API_KEY", "Kimi (placeholder)", json_mode="json_object", placeholder=True),
        Profile("minimax", "minimax", "MiniMax-latest", "openai_chat", "https://api.minimax.io/v1/chat/completions",
                "MINIMAX_API_KEY", "MiniMax (placeholder)", json_mode="json_object", placeholder=True),
    )
}


@dataclass(frozen=True)
class Job:
    id: str
    cost_feature: str
    site: str  # The production call: module.function, or the module for Eri chat's request loop.
    path: str  # extraction (structured JSON via extraction_request) | agent (tool-calling chat)
    label: str


def _summaries_feature():
    # run_summaries records under its own `agent_summaries` bucket once that lands; until then the judge's.
    return "agent_summaries" if "agent_summaries" in LABELS else "agent_control"


JOBS = {
    j.id: j
    for j in (
        Job("session_summary", "session_ledger", "agent_sessions.summarize_text", "extraction",
            "Agent session recall summaries"),
        Job("routing_review", "organization_learning", "routing.process", "extraction",
            "Weekly organization (routing) review"),
        Job("field_understanding", "field_understanding", "routing.assess", "extraction",
            "Custom field clarification"),
        Job("run_judge", "agent_control", "agent_judge.ask", "extraction", "Run judge verdicts"),
        Job("run_summary", _summaries_feature(), "run_summaries.ask", "extraction",
            "Run-card one-line summaries"),
        Job("memory_extraction", "memory_learning", "memory_learning.process", "extraction",
            "Memory fact extraction"),
        Job("note_tasks", "note_tasks", "notes.suggest_tasks", "extraction", "Tasks extracted from notes"),
        Job("eri_chat", "assistant", "conversation", "agent", "Eri chat turns (tool selection)"),
    )
}

# Today's behaviour: every structured job on Luna (low reasoning), Eri chat on the Settings choice.
DEFAULTS = {job: (SELECTED if spec.path == "agent" else "luna") for job, spec in JOBS.items()}
MODEL_MAP = dict(DEFAULTS)


@dataclass(frozen=True)
class Rule:
    """First matching rule wins. Empty `jobs` matches every job; token bounds are inclusive."""
    profile: str
    jobs: tuple = field(default_factory=tuple)
    min_prompt_tokens: int | None = None
    max_prompt_tokens: int | None = None
    name: str = ""

    def matches(self, job, prompt_tokens):
        return ((not self.jobs or job in self.jobs)
                and (self.min_prompt_tokens is None or prompt_tokens >= self.min_prompt_tokens)
                and (self.max_prompt_tokens is None or prompt_tokens <= self.max_prompt_tokens))


# No rules in production: route() returns MODEL_MAP. Example (docs/COST_TRACKING.md, "Model map"):
#   Rule("haiku-5.5", jobs=("session_summary", "run_summary"), max_prompt_tokens=100_000, name="short-summaries")
RULES: tuple = ()


@dataclass(frozen=True)
class Route:
    job: str
    profile: str
    cost_feature: str
    reason: str


def estimate_tokens(text):
    """A cheap length estimate for routing (about four characters per token), never for billing."""
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False, default=str)
    return math.ceil(len(text) / 4)


def route(job, prompt_tokens=0, *, rules=None, model_map=None):
    spec = JOBS.get(job)
    if spec is None:
        raise ValueError(f"Unknown job type {job!r}")
    for rule in RULES if rules is None else rules:
        if rule.matches(job, prompt_tokens):
            if rule.profile != SELECTED and rule.profile not in PROFILES:
                raise ValueError(f"Rule {rule.name or rule} names unknown profile {rule.profile!r}")
            return Route(job, rule.profile, spec.cost_feature, "rule:" + (rule.name or rule.profile))
    profile = (MODEL_MAP if model_map is None else model_map).get(job, DEFAULTS[job])
    return Route(job, profile, spec.cost_feature, "map")


def agent_model(job, prefs, **selected_kwargs):
    """The AgentModel a tool-calling job uses: the Settings choice; any other route is not enabled yet."""
    from .agent_models import selected

    chosen = route(job)
    if chosen.profile != SELECTED:
        raise ValueError(f"Profile {chosen.profile} is not enabled for production {job} turns")
    return selected(prefs, **selected_kwargs)


@contextmanager
def attributed(job):
    """Record usage made inside the block under the job's cost feature (the per-feature Model usage buckets)."""
    token = current_feature.set(JOBS[job].cost_feature)
    try:
        yield JOBS[job]
    finally:
        current_feature.reset(token)


# ---------------------------------------------------------------- provider adapters (pure)


def _split_system(messages):
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    return system, [m for m in messages if m["role"] != "system"]


def _function(tool):
    """A tool definition in chat-completions shape, from either chat or responses shape."""
    fn = tool.get("function") or {k: tool[k] for k in ("name", "description", "parameters") if k in tool}
    return {"type": "function", "function": fn}


def build(profile, payload):
    """A provider request body from a chat-completions-shaped payload: messages, and either response_format
    (json_schema) or tools, plus max_completion_tokens. Structured Luna requests match extraction_request."""
    messages = payload["messages"]
    fmt = (payload.get("response_format") or {}).get("json_schema")
    tools = [_function(t) for t in payload.get("tools") or []]
    limit = payload.get("max_completion_tokens") or 1200
    if profile.api == "openai_responses":
        if tools:
            from .agent_models import AgentModel
            from .responses_adapter import request

            agent = AgentModel(profile.provider, profile.model, profile.label, profile.endpoint, "", max(limit, 8192),
                               profile=profile.id, api="responses", reasoning_effort=profile.reasoning_effort,
                               service_tier=profile.service_tier)
            return request(agent, messages, tools, False)
        return {"model": profile.model, "input": messages,
                "text": {"format": {"type": "json_schema", **fmt}},
                "reasoning": {"effort": profile.reasoning_effort or "low"},
                "max_output_tokens": max(8192, limit), "store": False, "service_tier": profile.service_tier}
    if profile.api == "anthropic_messages":
        system, rest = _split_system(messages)
        body = {"model": profile.model, "max_tokens": max(8192, limit), "messages": rest}
        if system:
            body["system"] = system
        config = {"effort": profile.reasoning_effort} if profile.reasoning_effort else {}
        if fmt:
            config["format"] = {"type": "json_schema", "schema": fmt["schema"]}
        if config:
            body["output_config"] = config
        if tools:
            body["tools"] = [{"name": t["function"]["name"], "description": t["function"].get("description", ""),
                              "input_schema": t["function"]["parameters"]} for t in tools]
        return body
    if profile.api == "openai_chat":
        body = {"model": profile.model, "messages": list(messages), "max_tokens": max(4096, limit)}
        if fmt and profile.json_mode == "json_schema":
            body["response_format"] = {"type": "json_schema", "json_schema": fmt}
        elif fmt:
            body["response_format"] = {"type": "json_object"}
            body["messages"] = [*messages, {"role": "system", "content": "Answer with one JSON object matching this "
                                            "JSON schema, nothing else:\n" + json.dumps(fmt["schema"])}]
        if tools:
            body.update(tools=tools, tool_choice="auto")
        return body
    raise ValueError(f"Unsupported profile api {profile.api!r}")


def headers(profile, key):
    if profile.api == "anthropic_messages":
        return {"x-api-key": key, "anthropic-version": "2026-10-21", "content-type": "application/json"}
    return {"Authorization": f"Bearer {key}"}


def parse(profile, data):
    """A provider response as one normalized answer: {"text", "tool_calls" (chat shape), "refusal", "finish",
    "usage": {"input_tokens", "cached_input_tokens", "output_tokens"}}; input_tokens includes cached tokens."""
    if profile.api == "openai_responses":
        from .responses_adapter import normalize

        refusal = next((p["refusal"] for item in data.get("output", []) if item.get("type") == "message"
                        for p in item.get("content", []) if p.get("type") == "refusal"), None)
        chat = normalize(data)
        message, usage = chat["choices"][0]["message"], chat["usage"]
        return {"text": None if refusal else message.get("content"), "tool_calls": message.get("tool_calls") or [],
                "refusal": refusal, "finish": chat["choices"][0]["finish_reason"],
                "usage": {"input_tokens": usage["prompt_tokens"],
                          "cached_input_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                          "output_tokens": usage["completion_tokens"]}}
    if profile.api == "anthropic_messages":
        blocks = data.get("content") or []
        calls = [{"id": b["id"], "type": "function",
                  "function": {"name": b["name"], "arguments": json.dumps(b.get("input") or {})}}
                 for b in blocks if b.get("type") == "tool_use"]
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text") or None
        stop, usage = data.get("stop_reason"), data.get("usage") or {}
        cached = usage.get("cache_read_input_tokens") or 0
        refused = stop == "refusal"
        return {"text": None if refused else text, "tool_calls": [] if refused else calls,
                "refusal": ((data.get("stop_details") or {}).get("category") or "refusal") if refused else None,
                "finish": "length" if stop == "max_tokens" else "tool_calls" if calls else "stop",
                "usage": {"input_tokens": (usage.get("input_tokens") or 0) + cached
                          + (usage.get("cache_creation_input_tokens") or 0),
                          "cached_input_tokens": cached, "output_tokens": usage.get("output_tokens") or 0}}
    if profile.api == "openai_chat":
        choice = data["choices"][0]
        message, usage = choice["message"], data.get("usage") or {}
        cached = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                  or usage.get("prompt_cache_hit_tokens") or 0)
        return {"text": message.get("content"), "tool_calls": message.get("tool_calls") or [],
                "refusal": message.get("refusal"), "finish": choice.get("finish_reason") or "stop",
                "usage": {"input_tokens": usage.get("prompt_tokens") or 0, "cached_input_tokens": cached,
                          "output_tokens": usage.get("completion_tokens") or 0}}
    raise ValueError(f"Unsupported profile api {profile.api!r}")
