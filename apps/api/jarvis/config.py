from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, AwareDatetime, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", ".env.upgrade"),
        extra="ignore",
        env_prefix="JARVIS_",
        hide_input_in_errors=True,
        populate_by_name=True,
    )
    database_url: str = Field(
        default="postgresql+psycopg://jarvis:jarvis@127.0.0.1:54329/jarvis",
        validation_alias=AliasChoices("JARVIS_DATABASE_URL", "DATABASE_URL"),
    )
    deployment_environment: Literal["local", "staging", "production"] = "local"
    pairing_enabled: bool = True
    worker_enabled: bool = True
    external_services_enabled: bool = True
    semantic_search_enabled: bool = False
    maintenance_mode: bool = False
    database_pool_size: int = Field(default=10, ge=1, le=50)
    database_max_overflow: int = Field(default=10, ge=0, le=50)
    dbos_pool_size: int = Field(default=10, ge=5, le=50)
    dbos_client_pool_size: int = Field(default=5, ge=5, le=20)

    @field_validator("database_url")
    @classmethod
    def normalize_database_url(cls, value):
        # Railway supplies a libpq URL. Use our installed psycopg v3 driver.
        for prefix in ("postgres://", "postgresql://"):
            if value.startswith(prefix):
                return "postgresql+psycopg://" + value[len(prefix) :]
        return value

    owner_id: str = "davin"
    owner_name: str = "Davin"
    owner_token: str = ""
    origin: str = "http://localhost:8765"
    timezone: str = "America/Chicago"
    default_reminder_hour: int = 10
    cost_tracking_enabled: bool = True
    budget_enforcement_enabled: bool = True
    cost_tracking_since: AwareDatetime | None = None
    monthly_budget_usd: float = 150
    realtime_model: str = "gpt-realtime-2.1"
    live_model: str = "gpt-live-1"
    text_model: str = "gpt-5.6-luna"  # Legacy env compatibility; routing uses agent_models.
    agent_parallelism: int = Field(default=4, ge=1, le=16)
    agent_account_parallelism: int = Field(default=2, ge=1, le=4)
    agent_request_timeout_seconds: int = Field(default=600, ge=30, le=3600)
    agent_queue_limit: int = Field(default=100, ge=10, le=1000)
    max_tool_calls_per_request: int = Field(default=100, ge=1, le=1000)
    max_model_rounds_per_request: int = Field(default=30, ge=2, le=100)
    openai_api_key: str = ""
    groq_api_key: str = ""
    gemini_api_key: str = ""
    history_days: int = 0
    web_dist: Path = Path("apps/web/dist")
    vapid_private_key: str = ""
    vapid_public_key: str = ""
    vapid_subject: str = "mailto:owner@localhost"
    google_client_id: str = ""
    google_client_secret: str = ""
    integration_encryption_key: str = ""
    google_poll_seconds: int = Field(default=300, ge=60, le=3600)
    home_assistant_url: str = ""
    home_assistant_token: str = ""
    home_entities: str = ""
    session_hours: int = 24 * 30
    worker_interval_seconds: int = 5

    @property
    def secure_cookie(self):
        return self.origin.startswith("https://")


@lru_cache
def get_settings():
    # Existing environment variables remain supported without copying secrets into source files.
    import os

    from dotenv import dotenv_values

    env_override = os.environ.get("JARVIS_ENV_FILE")
    if env_override is None:
        legacy = {**dotenv_values(".env"), **os.environ}
        settings = Settings()
    else:
        # An explicit empty value disables local file fallback for cloud preflight.
        legacy = {**(dotenv_values(env_override) if env_override else {}), **os.environ}
        settings = Settings(_env_file=env_override or None)
    settings.openai_api_key = settings.openai_api_key or legacy.get("OPENAI_API_KEY", "") or ""
    settings.groq_api_key = settings.groq_api_key or legacy.get("GROQ_API_KEY", "") or ""
    settings.gemini_api_key = settings.gemini_api_key or legacy.get("GEMINI_API_KEY", "") or ""
    if not settings.external_services_enabled:
        # A restored staging database must not spend money or send device notifications.
        settings.openai_api_key = settings.groq_api_key = settings.gemini_api_key = ""
        settings.vapid_private_key = settings.home_assistant_token = ""
    return settings


def require_external_services():
    if not get_settings().external_services_enabled:
        from .domain import DomainError

        raise DomainError(
            "EXTERNAL_SERVICES_DISABLED", "External services are paused in this environment.", 503
        )
