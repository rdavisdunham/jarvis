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

    @property
    def available(self):
        return bool(self.api_key.strip())

    def public(self):
        return {
            "provider": self.provider,
            "model": self.model,
            "label": self.label,
            "available": self.available,
        }

    def rates(self, today=None):
        if self.provider == "gemini":
            today = today or datetime.now(UTC).date()
            # Published introductory pricing expires at the end of 2026.
            return (0.75, 3.75) if today < date(2027, 1, 1) else (1.50, 7.50)
        return (0.75, 4.50) if self.provider == "openai" else (0.15, 0.60)

    def request(self, messages, tools, limited=False):
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
        "openai": AgentModel(
            "openai",
            settings.text_model,
            "GPT-5.4 mini" if settings.text_model == "gpt-5.4-mini" else settings.text_model,
            "https://api.openai.com/v1/chat/completions",
            settings.openai_api_key,
        ),
        "gemini": AgentModel(
            "gemini",
            "gemini-3.8-flash",
            "Gemini 3.8 Flash",
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            settings.gemini_api_key,
            8192,
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
    # Preserve the existing OpenAI-first/Groq-second default.
    return next((p for p in ("openai", "groq", "gemini") if models[p].available), "openai")


def selected(prefs, *, require_key=False):
    from .domain import DomainError

    provider = prefs.get("agent_provider") or default_provider()
    model = catalog().get(provider)
    if model is None:
        raise DomainError("INVALID_ARGUMENT", "Choose a supported task agent in Settings.")
    if require_key and not model.available:
        key_name = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY"}[provider]
        raise DomainError(
            "INTEGRATION_UNAVAILABLE",
            f"{model.label} needs {key_name} in the server .env file. Recreate the API/worker containers after saving it, or choose another task agent in Settings.",
            503,
        )
    return model
