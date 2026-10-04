"""Bounded content-free latency events for correlating existing durable work.

Durations use the local monotonic clock; UTC timestamps allow cross-process joins.
Precommit events are diagnostic; after-commit events establish durable outcomes.
"""
import json
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar

from sqlalchemy import event
from datetime import datetime, timezone

logger = logging.getLogger("jarvis.latency")
logger.setLevel(logging.INFO)

class FallbackHandler(logging.StreamHandler):
    """Uvicorn need not configure the root logger; still expose content-free events."""
    def emit(self, record):
        if len(logger.handlers) == 1 and not (logger.parent and logger.parent.hasHandlers()):
            super().emit(record)


if not any(isinstance(handler, FallbackHandler) for handler in logger.handlers):
    logger.addHandler(FallbackHandler())
FIELDS = {
    "tool_policy", "tool_kind", "groups", "newly_loaded_count", "already_available_count",
    "available_tools", "tool_names", "definition_bytes",
    "revision", "round", "tool_index", "duration_ms", "queue_ms", "dispatch_ms",
    "elapsed_ms", "outcome", "voice_session_id", "channel", "profile", "model",
    "requested_service_tier", "served_service_tier", "cost_basis",
    "reasoning", "tool", "retry_count", "attempt", "response_id",
    "input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens",
    "cache_write_tokens", "cost_usd", "receipts", "source", "run_id", "uncertain_spend",
}


_context = ContextVar("latency_context", default={})


@contextmanager
def context(**fields):
    token = _context.set({**_context.get(), **fields})
    try:
        yield
    finally:
        _context.reset(token)


def mark(stage, request_id, **fields):
    event = {"stage": stage, "request_id": request_id,
             "utc": datetime.now(timezone.utc).isoformat(),
             **{k: v for k, v in {**_context.get(), **fields}.items() if k in FIELDS}}
    logger.info("latency %s", json.dumps(event, separators=(",", ":")))


@contextmanager
def span(stage, request_id, **fields):
    start = time.monotonic()
    outcome = "ok"
    try:
        yield
    except BaseException:
        outcome = "error"
        raise
    finally:
        mark(stage, request_id, **fields, duration_ms=round((time.monotonic()-start)*1000, 3), outcome=outcome)

def after_commit(db, stage, request_id, **fields):
    """Emit only once the containing transaction commits; never export record content."""
    if not db.info.get("_latency_registered"):
        db.info["_latency_registered"] = True

        def committed(session):
            if session.in_nested_transaction():
                return
            for name, identity, payload in session.info.pop("_latency_pending", []):
                mark(name, identity, **payload)

        def rolled_back(session):
            # Losing measurement on nested rollback is preferable to reporting a save
            # which did not happen. Application work transitions use root transactions.
            session.info.pop("_latency_pending", None)

        event.listen(db, "after_commit", committed)
        event.listen(db, "after_rollback", rolled_back)
    db.info.setdefault("_latency_pending", []).append((stage, request_id, {**_context.get(), **fields}))
