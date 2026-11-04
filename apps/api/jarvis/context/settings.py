"""Context platform settings: one accessor over config.Settings so the module reads flags in one place.

Recorded decision (2026-11-04, implementation_plan.md section 3): embeddings are pgvector vector(512) columns.
Railway PITR is unavailable on the pgvector image; pgvector stays and another backup is added later. There is
no JSONB fallback.
"""

from dataclasses import dataclass

from ..config import get_settings

EMBEDDING_DIMENSIONS = 512
EMBEDDING_STORAGE = "pgvector"
QUEUE = "jarvis-context"
JOB_PREFIX = "context_"


@dataclass(frozen=True)
class ContextSettings:
    enabled: bool
    queue_concurrency: int
    source_budget_usd: float
    model_timeout_s: int
    poll_minutes: int
    max_source_chars: int
    granola_enabled: bool
    actions_model: str = ""  # model_map profile for the action-item pass; "" keeps the model map's route
    embedding_dimensions: int = EMBEDDING_DIMENSIONS
    embedding_storage: str = EMBEDDING_STORAGE


def current():
    s = get_settings()
    return ContextSettings(
        enabled=s.context_enabled,
        queue_concurrency=s.context_queue_concurrency,
        source_budget_usd=s.context_source_budget_usd,
        model_timeout_s=s.context_model_timeout_s,
        poll_minutes=s.context_poll_minutes,
        max_source_chars=s.context_max_source_chars,
        granola_enabled=s.context_granola_enabled,
        actions_model=s.context_actions_model,
    )


def enabled():
    return get_settings().context_enabled
