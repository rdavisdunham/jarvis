"""Embedding projections for extraction chunks and assertions (plan sections 2 and 9).

Vectors come from `memory_learning.embeddings` (the shared model and its 512 dimensions) under the cost feature
`context_embeddings`; each row records the producing model and time beside the vector. Vectors are computed
outside any transaction (`compute`) and written afterwards (`write`) only where the row is still unembedded and
its text is unchanged, so a purge or edit in between wins. `embedding_model` is always the current
`memory_learning.EMBEDDING_MODEL` (retrieval's semantic leg only uses rows from the current model), so a row
embedded by an older model counts as unembedded and is re-embedded. A failed call leaves the row as it was:
lexical search keeps working and `status` reports what is still unindexed and the last embedding error.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import func, or_, select, update

from ... import memory_learning
from ...cost_features import feature
from ...models import now
from ..models import ContextAssertion, ContextChunk, ContextRun

FEATURE = "context_embeddings"
MODEL = memory_learning.EMBEDDING_MODEL
BATCH = 64
LIMIT = 500  # rows per stage run; the rest waits for the next run
LIVE = ("current", "contested")


def _missing(model):
    """No vector from the current model."""
    return or_(model.embedding.is_(None), model.embedding_model.is_distinct_from(MODEL))


@dataclass(frozen=True)
class Item:
    table: str  # "chunk" | "assertion"
    id: str
    text: str


def pending(db, owner: str, *, revision_id: str | None = None, limit: int = LIMIT) -> list[Item]:
    """Unembedded chunks (of one revision when given) and live assertions, oldest first."""
    c = ContextChunk
    chunks = select(c.id, c.text).where(c.owner_id == owner, _missing(c), c.text != "")
    if revision_id:
        chunks = chunks.where(c.revision_id == revision_id)
    items = [Item("chunk", i, t) for i, t in db.execute(chunks.order_by(c.created_at, c.id).limit(limit))]
    a = ContextAssertion
    rows = db.execute(select(a.id, a.statement).where(a.owner_id == owner, _missing(a), a.lifecycle.in_(LIVE))
                      .order_by(a.created_at, a.id).limit(max(limit - len(items), 0)))
    return items + [Item("assertion", i, t) for i, t in rows]


@feature(FEATURE)
def compute(owner: str, items: Sequence[Item]) -> list[list[float]]:
    """Vectors for `items`, in order. Raises on any provider failure (the caller records it)."""
    vectors: list[list[float]] = []
    for start in range(0, len(items), BATCH):
        vectors += memory_learning.embeddings(owner, [i.text for i in items[start:start + BATCH]])
    return vectors


def write(db, owner: str, items: Sequence[Item], vectors: Sequence[list[float]]) -> int:
    """Store vectors (with `embedding_model` and `embedded_at`) on rows still lacking a current-model vector whose
    text is unchanged. Returns how many were written."""
    stamp, written = now(), 0
    for item, vector in zip(items, vectors, strict=True):
        model, text = (ContextChunk, ContextChunk.text) if item.table == "chunk" else \
            (ContextAssertion, ContextAssertion.statement)
        written += db.execute(update(model).where(
            model.id == item.id, model.owner_id == owner, _missing(model), text == item.text)
            .values(embedding=vector, embedding_model=MODEL, embedded_at=stamp)).rowcount
    db.flush()
    return written


def status(db, owner: str) -> dict:
    """Indexing status for operators and the API: unembedded counts and the last embedding run's outcome."""
    c, a = ContextChunk, ContextAssertion
    chunks = db.scalar(select(func.count()).select_from(c).where(
        c.owner_id == owner, _missing(c), c.text != ""))
    assertions = db.scalar(select(func.count()).select_from(a).where(
        a.owner_id == owner, _missing(a), a.lifecycle.in_(LIVE)))
    last = db.scalar(select(ContextRun).where(ContextRun.owner_id == owner, ContextRun.stage == "embed")
                     .order_by(ContextRun.created_at.desc(), ContextRun.id.desc()).limit(1))
    return {"model": MODEL, "dimensions": memory_learning.DIMENSIONS, "unembedded_chunks": chunks,
            "unembedded_assertions": assertions,
            "last_run": None if last is None else {"id": last.id, "status": last.status, "error": last.error,
                                                   "at": last.updated_at.isoformat()}}
