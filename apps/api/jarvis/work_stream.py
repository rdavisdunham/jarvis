"""Dani's answer as it is written: text deltas from the worker to the browser.

The worker streams the model's reply (work_runner, OpenAI Responses `stream: true`) and sends the text in small
batches with `pg_notify` on CHANNEL. Every web process LISTENs (lifespan task) and hands each batch to the event
streams (`GET /api/v1/events`) of the same owner and device, which pass it to the browser as a `work_delta`
event. Deltas are a preview only: nothing stores them, a lost one just leaves a gap, and the finished reply still
arrives the durable way (`work.changed`, then the work item's saved message), replacing the preview.

Payload (JSON, kept under NOTIFY's 8000-byte limit): {"o": owner, "d": device, "w": work id, "r": model round,
"s": batch sequence within the round, "t": text}. A new round (after tool calls) starts a new preview.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict

import psycopg
from sqlalchemy import text

logger = logging.getLogger("jarvis.work_stream")
CHANNEL = "work_delta"
FLUSH_SECONDS = 0.08  # Batch deltas: smooth enough to read as streaming, few enough notifies.
MAX_CHARS = 1500  # Per notify, well under 8000 bytes even for multi-byte text once JSON-escaped.
QUEUE_SIZE = 500
LISTEN_IDLE_SECONDS = 30
LISTEN_BACKOFF = (0.5, 30.0)

_subscribers: dict[tuple[str, str], set[asyncio.Queue]] = defaultdict(set)


# ---------------------------------------------------------------- worker side


class Deltas:
    """Collects one request's streamed text and sends it in batches. Never raises: a preview is optional."""

    def __init__(self, owner, device, work_id):
        self.owner, self.device, self.work_id = owner, device, work_id
        self.round, self.seq, self.buffer, self.last = 0, 0, "", 0.0
        self.failed = False

    def start_round(self, number):
        self.flush()
        self.round, self.seq = number, 0

    def add(self, piece):
        if not piece or self.failed:
            return
        self.buffer += piece
        if time.monotonic() - self.last >= FLUSH_SECONDS or len(self.buffer) >= MAX_CHARS:
            self.flush()

    def flush(self):
        if not self.buffer or self.failed:
            self.buffer = ""
            return
        chunks = [self.buffer[i:i + MAX_CHARS] for i in range(0, len(self.buffer), MAX_CHARS)]
        self.buffer, self.last = "", time.monotonic()
        try:
            from .db import engine

            with engine().connect() as conn:
                for chunk in chunks:
                    self.seq += 1
                    conn.execute(text("SELECT pg_notify(:channel, :payload)"), {"channel": CHANNEL, "payload": json.dumps(
                        {"o": self.owner, "d": self.device, "w": self.work_id, "r": self.round, "s": self.seq,
                         "t": chunk}, ensure_ascii=False)})
                conn.commit()
        except Exception as error:  # noqa: BLE001 - the answer itself is unaffected; stop previewing this request
            self.failed = True
            logger.warning("Streaming preview stopped (%s); the reply still arrives when it is done.",
                           type(error).__name__)


# ---------------------------------------------------------------- web side


def subscribe(owner, device) -> asyncio.Queue:
    queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_SIZE)
    _subscribers[(owner, device)].add(queue)
    return queue


def unsubscribe(owner, device, queue):
    group = _subscribers.get((owner, device))
    if group is not None:
        group.discard(queue)
        if not group:
            _subscribers.pop((owner, device), None)


def deliver(payload: str):
    """Hand one notify to this process's matching event streams (owner and device must both match)."""
    try:
        data = json.loads(payload)
        key = (data.pop("o"), data.pop("d"))
    except (ValueError, KeyError, TypeError, AttributeError):
        return
    for queue in list(_subscribers.get(key, ())):
        try:
            queue.put_nowait(data)
        except asyncio.QueueFull:
            pass  # A stalled browser misses preview text; its final reply is unaffected.


async def listen():
    """Lifespan task in each web process: LISTEN on CHANNEL and deliver every batch. Reconnects with backoff."""
    from .agent_link import listen_dsn

    delay = LISTEN_BACKOFF[0]
    while True:
        try:
            async with await psycopg.AsyncConnection.connect(
                    listen_dsn(), autocommit=True, connect_timeout=10, application_name="eridani-work-stream") as conn:
                await conn.execute(f"LISTEN {CHANNEL}")
                delay = LISTEN_BACKOFF[0]
                while True:
                    async for note in conn.notifies(timeout=LISTEN_IDLE_SECONDS):
                        deliver(note.payload)
                    await conn.execute("SELECT 1")  # Raises if the quiet connection died.
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - never take the web process down
            logger.warning("work stream listener disconnected (%s); reconnecting in %.1fs", type(exc).__name__, delay)
        await asyncio.sleep(delay)
        delay = min(delay * 2, LISTEN_BACKOFF[1])


async def drain(queue: asyncio.Queue, seconds: float):
    """Wait up to `seconds` for preview batches; returns what arrived (possibly nothing)."""
    loop = asyncio.get_running_loop()
    deadline, out = loop.time() + seconds, []
    while (left := deadline - loop.time()) > 0:
        try:
            out.append(await asyncio.wait_for(queue.get(), left))
        except TimeoutError:
            break
        while not queue.empty():
            out.append(queue.get_nowait())
        if out:
            break  # Send what arrived now; the next wait continues.
    return out
