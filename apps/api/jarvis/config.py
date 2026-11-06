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
    # Interactive Luna requests; scheduled learning remains explicitly Standard.
    agent_service_tier: Literal["default", "fast"] = "fast"
    agent_tool_policy: Literal["baseline", "discovery-v1", "reads-v1", "lean-v1"] = "discovery-v1"
    agent_parallelism: int = Field(default=4, ge=1, le=16)
    agent_account_parallelism: int = Field(default=2, ge=1, le=4)
    agent_request_timeout_seconds: int = Field(default=600, ge=30, le=3600)
    # A run's verification still running after this long sends the owner one notice (it keeps running up to the verify
    # entry's own timeout); a verify entry's advertised `notify_after_s` overrides it (run_coordinator.verify_notify_after).
    agent_verify_notify_after_seconds: int = Field(default=1800, ge=60, le=4 * 3600)
    # GitHub CI as the final verification (run_ci.py): a run whose criteria are met on a quick local check waits this
    # long in awaiting_ci for the PR head's checks to finish, then needs you (ci_timeout).
    agent_ci_wait_seconds: int = Field(default=90 * 60, ge=300, le=12 * 3600)
    # Verify entry ids that are only a pre-push check (scripts/verify-quick.sh): their pass needs green CI on the PR
    # head to finish the run. Any other entry's pass counts on its own, as before. Comma-separated.
    agent_ci_quick_verify_ids: str = "quick"
    agent_queue_limit: int = Field(default=100, ge=10, le=1000)
    max_tool_calls_per_request: int = Field(default=100, ge=1, le=1000)
    max_model_rounds_per_request: int = Field(default=30, ge=2, le=100)
    openai_api_key: str = ""
    groq_api_key: str = ""
    gemini_api_key: str = ""
    anthropic_api_key: str = ""  # Claude models routed through model_map (JARVIS_ANTHROPIC_API_KEY)
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
    # Context platform (jarvis/context, read through context/settings.py). Off until the pilot.
    context_enabled: bool = False
    context_queue_concurrency: int = Field(default=2, ge=1, le=8)
    # Sized for the whole-meeting action pass on Terra or Sonnet 5.5 plus the chunked passes (about $0.30 to
    # $1.50 for a one-hour meeting depending on the routes); reservations count upper bounds against it.
    context_source_budget_usd: float = Field(default=3.0, ge=0)
    context_model_timeout_s: int = Field(default=90, ge=10, le=600)
    context_poll_minutes: int = Field(default=5, ge=1, le=1440)
    context_max_source_chars: int = Field(default=1_000_000, ge=1_000, le=20_000_000)
    context_granola_enabled: bool = False
    # model_map profile for the meeting action-item pass; empty keeps the model map's route (Luna). When the profile
    # is unknown or its provider key is missing, the pass falls back to that route (context/native/actions.py).
    context_actions_model: str = "sonnet-5.5"
    # "split": action pass + per-chunk detect/enrich/verify; "combined": one whole-meeting call for facts and
    # follow-ups, then one verify call (an experiment; context/native/extract.py).
    context_extraction_mode: Literal["split", "combined"] = "split"
    # Error reporting (observability.py). Off unless set. The API and worker report to JARVIS_SENTRY_DSN; the
    # browser app reads JARVIS_SENTRY_WEB_DSN from /bootstrap, and the CSP allows its ingest host only when set.
    sentry_dsn: str = ""
    sentry_web_dsn: str = ""

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
    settings.anthropic_api_key = settings.anthropic_api_key or legacy.get("ANTHROPIC_API_KEY", "") or ""
    if not settings.external_services_enabled:
        # A restored staging database must not spend money or send device notifications.
        settings.openai_api_key = settings.groq_api_key = settings.gemini_api_key = ""
        settings.anthropic_api_key = ""
        settings.vapid_private_key = settings.home_assistant_token = ""
    return settings


def require_external_services():
    if not get_settings().external_services_enabled:
        from .domain import DomainError

        raise DomainError(
            "EXTERNAL_SERVICES_DISABLED", "External services are paused in this environment.", 503
        )
