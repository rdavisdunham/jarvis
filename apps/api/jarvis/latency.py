"""Bounded content-free latency events for correlating existing durable work.

Durations use the local monotonic clock; UTC timestamps allow cross-process joins.
Transaction stages say 'precommit': only later dispatch/runner events prove acceptance.
"""
import json
import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone

logger = logging.getLogger("jarvis.latency")
logger.setLevel(logging.INFO)
FIELDS = {"revision", "round", "tool_index", "duration_ms", "queue_ms", "dispatch_ms", "outcome", "voice_session_id"}


def mark(stage, request_id, **fields):
    event = {"stage": stage, "request_id": request_id,
             "utc": datetime.now(timezone.utc).isoformat(),
             **{k: v for k, v in fields.items() if k in FIELDS}}
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
