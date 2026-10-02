"""Campaign-local immutable fixture vector cache. Cold-index trials bypass it."""

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from unittest.mock import patch


@contextmanager
def cached_embeddings(directory, trace, *, cold=False):
    from jarvis import memory_learning, notes

    real = memory_learning.embeddings
    path = directory / "embeddings.sqlite"
    with sqlite3.connect(path, timeout=30) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vector TEXT NOT NULL, model TEXT NOT NULL, dimensions INTEGER NOT NULL)"
        )

    def cached(owner, texts, *args, **kwargs):
        if cold:
            return real(owner, texts, *args, **kwargs)
        keys = [
            hashlib.sha256(
                json.dumps(
                    [
                        memory_learning.EMBEDDING_MODEL,
                        memory_learning.DIMENSIONS,
                        memory_learning.VERSION,
                        text[:12000],
                    ],
                    ensure_ascii=False,
                ).encode()
            ).hexdigest()
            for text in texts
        ]
        # Holding the cache transaction across the bounded embedding call prevents two
        # workers paying for the same immutable fixture at once. Agent calls stay parallel.
        with sqlite3.connect(path, timeout=60, isolation_level=None) as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                found = {
                    key: json.loads(row[0])
                    for key in keys
                    if (row := db.execute("SELECT vector FROM vectors WHERE key=?", (key,)).fetchone())
                }
                missing = list(dict.fromkeys(key for key in keys if key not in found))
                if missing:
                    input_texts = [texts[keys.index(key)] for key in missing]
                    vectors = real(owner, input_texts, *args, **kwargs)
                    for key, vector in zip(missing, vectors, strict=True):
                        db.execute(
                            "INSERT INTO vectors VALUES (?,?,?,?)",
                            (
                                key,
                                json.dumps(vector),
                                memory_learning.EMBEDDING_MODEL,
                                memory_learning.DIMENSIONS,
                            ),
                        )
                        found[key] = vector
                db.commit()
            except BaseException:
                db.rollback()
                raise
        trace.append(
            {
                "kind": "embedding_cache",
                "hits": len(keys) - len(missing),
                "misses": len(missing),
                "model": memory_learning.EMBEDDING_MODEL,
                "keys": keys,
            }
        )
        return [found[key] for key in keys]

    with patch.object(memory_learning, "embeddings", cached), patch.object(notes, "embeddings", cached):
        yield
