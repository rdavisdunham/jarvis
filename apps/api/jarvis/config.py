from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", ".env.upgrade"), extra="ignore", env_prefix="JARVIS_")
    database_url: str = "postgresql+psycopg://jarvis:jarvis@127.0.0.1:54329/jarvis"
    owner_id: str = "davin"
    owner_name: str = "Davin"
    owner_token: str = ""
    origin: str = "http://localhost:8765"
    timezone: str = "America/Chicago"
    default_reminder_hour: int = 10
    monthly_budget_usd: float = 150
    realtime_model: str = "gpt-realtime-2.1"
    live_model: str = "gpt-live-1"
    text_model: str = "gpt-5.4-mini"
    max_tool_calls_per_request: int = Field(default=100, ge=1, le=1000)
    max_model_rounds_per_request: int = Field(default=30, ge=2, le=100)
    openai_api_key: str = ""
    groq_api_key: str = ""
    history_days: int = 0
    web_dist: Path = Path("apps/web/dist")
    vapid_private_key: str = ""
    vapid_public_key: str = ""
    vapid_subject: str = "mailto:owner@localhost"
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

    legacy = {**dotenv_values(".env"), **os.environ}
    settings = Settings()
    settings.openai_api_key = settings.openai_api_key or legacy.get("OPENAI_API_KEY", "") or ""
    settings.groq_api_key = settings.groq_api_key or legacy.get("GROQ_API_KEY", "") or ""
    return settings
