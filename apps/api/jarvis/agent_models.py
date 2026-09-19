"""Backend text/task model routing; voice and memory embeddings have separate providers."""

from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from .config import get_settings


@dataclass(frozen=True)
class AgentModel:
    provider: str
    model: str
    label: str
    endpoint: str
    api_key: str = field(repr=False)
    max_output_tokens: int = 1200
    profile: str | None = None
    api: str = "chat_completions"
    reasoning_effort: str | None = None

    @property
    def profile_id(self):
        return self.profile or self.provider

    @property
    def available(self):
        return bool(self.api_key.strip())

    def public(self):
        return {
            "id": self.profile_id,
            "provider": self.provider,
            "reasoning_effort": self.reasoning_effort,
            "model": self.model,
            "label": self.label,
            "available": self.available,
        }

    def rates(self, today=None):
        if self.model == "gpt-5.6-luna":
            return (0.20, 1.20)
        if self.provider == "gemini":
            today = today or datetime.now(UTC).date()
            # Published introductory pricing expires at the end of 2026.
            return (0.75, 3.75) if today < date(2027, 1, 1) else (1.50, 7.50)
        return (0.75, 4.50) if self.provider == "openai" else (0.15, 0.60)

    def reserve_cost(self, input_bound):
        incoming, outgoing = self.rates()
        # Luna charges a cache-write premium; this remains an upper bound.
        if self.model == "gpt-5.6-luna" and input_bound > 272000:
            incoming, outgoing = incoming * 2, outgoing * 1.5
        incoming *= 1.25 if self.model == "gpt-5.6-luna" else 1
        return (input_bound * incoming + self.max_output_tokens * outgoing) / 1_000_000

    def usage_cost(self, usage):
        incoming, outgoing = self.rates()
        prompt = usage["prompt_tokens"]
        if self.model == "gpt-5.6-luna" and prompt > 272000:
            incoming, outgoing = incoming * 2, outgoing * 1.5
        cost = prompt * incoming + usage["completion_tokens"] * outgoing
        if self.model == "gpt-5.6-luna":
            details = usage.get("prompt_tokens_details") or {}
            cached = max(0, min(prompt, details.get("cached_tokens", 0)))
            written = max(0, min(prompt - cached, details.get("cache_write_tokens", 0)))
            cost += written * incoming * 0.25 - cached * incoming * 0.90
        return cost / 1_000_000

    def normalize(self, data):
        if self.api == "responses":
            from .responses_adapter import normalize

            return normalize(data)
        return data

    def request(self, messages, tools, limited=False):
        if self.api == "responses":
            from .responses_adapter import request

            return request(self, messages, tools, limited)
        body = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "none" if limited else "auto",
        }
        if self.provider == "gemini":
            # Allow room for both thinking and the answer; Flash 3.8 cannot use none/minimal.
            body.update(max_tokens=self.max_output_tokens, reasoning_effort="low")
        else:
            body["max_completion_tokens"] = self.max_output_tokens
        return body


def catalog():
    settings = get_settings()
    return {
        "luna": AgentModel(
            "openai",
            "gpt-5.6-luna",
            "GPT-5.6 Luna · reasoning",
            "https://api.openai.com/v1/responses",
            settings.openai_api_key,
            8192,
            profile="luna",
            api="responses",
            reasoning_effort="low",
        ),
        "gemini": AgentModel(
            "gemini",
            "gemini-3.8-flash",
            "Gemini 3.8 Flash",
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            settings.gemini_api_key,
            8192,
            reasoning_effort="low",
        ),
        "groq": AgentModel(
            "groq",
            "openai/gpt-oss-120b",
            "GPT-OSS 120B (Groq)",
            "https://api.groq.com/openai/v1/chat/completions",
            settings.groq_api_key,
        ),
    }


def default_provider():
    models = catalog()
    # Preserve the existing provider preference while retiring the old OpenAI model.
    return next((p for p in ("luna", "groq", "gemini") if models[p].available), "luna")


def selected(prefs, *, require_key=False):
    from .domain import DomainError

    profile = prefs.get("agent_profile") or prefs.get("agent_provider") or default_provider()
    # Old saved profile/provider values resolve to Luna, never the retired model.
    if profile == "openai":
        profile = "luna"
    model = catalog().get(profile)
    if model is None:
        raise DomainError("INVALID_ARGUMENT", "Choose a supported task agent in Settings.")
    if require_key and not model.available:
        key_name = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY"}[
            model.provider
        ]
        raise DomainError(
            "INTEGRATION_UNAVAILABLE",
            f"{model.label} needs {key_name} in the server .env file. Recreate the API/worker containers after saving it, or choose another task agent in Settings.",
            503,
        )
    return model
