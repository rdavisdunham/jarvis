from functools import lru_cache
from pathlib import Path

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
    text_model: str = "gpt-5.4-mini"
    openai_api_key: str = ""
    groq_api_key: str = ""
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "jarvis_memories"
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
