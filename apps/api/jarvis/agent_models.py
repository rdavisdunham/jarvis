"""Backend text/task model routing; voice and memory embeddings have separate providers."""

from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime

from .config import get_settings


@dataclass(frozen=True)
class Price:
    """USD per million tokens. Cached reads and cache writes are fractions of the input rate; prompts longer than
    `long_above` input tokens are billed at the long-context multipliers for the whole request; `fast` scales
    both rates on the fast or priority tier."""

    input: float
    output: float
    cached_input: float = 0.10
    cache_write: float = 1.25
    long_above: int | None = None
    long_input: float = 1.0
    long_output: float = 1.0
    fast: float = 1.0


# Standard API rates (docs/archive/2026-10-02/BACKGROUND_MODEL_COMPARISON.md for GPT-5.6; Anthropic's published
# pricing for Claude, checked 2026-11-04). The GPT-5.6 family shares Luna's cache-write premium and 272K
# long-context rates. Claude cache writes are the 5-minute TTL rate; Sonnet 5.5 has no long-context premium.
PRICES = {
    "gpt-5.6-luna": Price(0.20, 1.20, long_above=272_000, long_input=2, long_output=1.5, fast=2),
    "gpt-5.6-terra": Price(2.00, 12.00, long_above=272_000, long_input=2, long_output=1.5),
    "claude-sonnet-5-5": Price(2.00, 10.00),
    "claude-haiku-5-5": Price(0.10, 0.50, long_above=100_000, long_input=5, long_output=5),
}


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
    service_tier: str = "default"

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

    @property
    def price(self):
        return PRICES.get(self.model)

    def rates(self, today=None, *, service_tier=None):
        if price := self.price:
            tier = self.service_tier if service_tier is None else service_tier
            fast = price.fast if tier in {"fast", "priority"} else 1
            return price.input * fast, price.output * fast
        if self.provider == "gemini":
            today = today or datetime.now(UTC).date()
            # Published introductory pricing expires at the end of 2026.
            return (0.75, 3.75) if today < date(2027, 1, 1) else (1.50, 7.50)
        return (0.75, 4.50) if self.provider == "openai" else (0.15, 0.60)

    def _long(self, prompt_tokens, incoming, outgoing):
        price = self.price
        if price and price.long_above is not None and prompt_tokens > price.long_above:
            return incoming * price.long_input, outgoing * price.long_output
        return incoming, outgoing

    def reserve_cost(self, input_bound):
        incoming, outgoing = self._long(input_bound, *self.rates())
        # Every input token could be a cache write (a premium over plain input); this remains an upper bound.
        incoming *= self.price.cache_write if self.price else 1
        return (input_bound * incoming + self.max_output_tokens * outgoing) / 1_000_000

    def usage_record(self, data):
        """Retain tier provenance with token accounting; missing tier is an estimate."""
        usage = dict(data.get("usage") or {})
        if self.price and self.price.fast != 1:
            served = data.get("service_tier")
            verified = isinstance(served, str) and served in {"default", "fast", "priority"}
            usage.update(
                requested_service_tier=self.service_tier,
                served_service_tier=served if verified else None,
                cost_basis="returned_service_tier" if verified else "requested_service_tier_estimate",
            )
        return usage

    def usage_cost(self, usage):
        """`prompt_tokens` includes cached reads (`prompt_tokens_details.cached_tokens`) and cache writes
        (`cache_write_tokens`), which are repriced at the model's cache rates."""
        prompt = usage["prompt_tokens"]
        incoming, outgoing = self._long(prompt, *self.rates(service_tier=usage.get("served_service_tier")))
        cost = prompt * incoming + usage["completion_tokens"] * outgoing
        if price := self.price:
            details = usage.get("prompt_tokens_details") or {}
            cached = max(0, min(prompt, details.get("cached_tokens") or 0))
            written = max(0, min(prompt - cached, details.get("cache_write_tokens") or 0))
            cost += written * incoming * (price.cache_write - 1) - cached * incoming * (1 - price.cached_input)
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
        "luna-none": AgentModel(
            "openai",
            "gpt-5.6-luna",
            "Luna · no reasoning",
            "https://api.openai.com/v1/responses",
            settings.openai_api_key,
            8192,
            profile="luna-none",
            api="responses",
            reasoning_effort="none",
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


def selected(prefs, *, require_key=False, service_tier=None):
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
    if model.model == "gpt-5.6-luna":
        tier = get_settings().agent_service_tier if service_tier is None else service_tier
        if tier not in {"default", "fast"}:
            raise ValueError("Unsupported agent service tier")
        model = replace(model, service_tier=tier)
    return model
