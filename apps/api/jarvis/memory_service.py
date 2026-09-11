"""Canonical memory reads plus an isolated, read-only legacy Qdrant adapter."""

import re

import httpx
from sqlalchemy import func, select

from .config import get_settings
from .domain import serial
from .models import Memory, Source


def search(db, owner, query="", limit=10):
    q = (
        select(Memory)
        .join(Source, Source.id == Memory.source_id)
        .where(Memory.owner_id == owner, Memory.suppressed.is_(False), Source.deleted_at.is_(None))
    )
    if query.strip():
        tsquery = func.websearch_to_tsquery("english", query)
        vector = func.to_tsvector("english", Memory.content)
        q = q.where(vector.op("@@")(tsquery)).order_by(func.ts_rank_cd(vector, tsquery).desc())
    else:
        q = q.order_by(Memory.created_at.desc())
    return [serial(m) for m in db.scalars(q.limit(limit))]


async def legacy_search(query, limit=5):
    settings = get_settings()
    if not query.strip():
        return []
    # Read-only scroll: no Mem0 initialization, embedding downloads, collection creation or writes.
    # A bounded lexical bridge keeps old facts accessible while the versioned retrieval service evolves.
    terms = set(re.findall(r"\w+", query.lower())) - {
        "the",
        "and",
        "that",
        "what",
        "about",
        "you",
        "me",
        "my",
        "is",
        "a",
    }
    if not terms:
        return []
    try:
        async with httpx.AsyncClient(timeout=2.5, follow_redirects=False) as client:
            candidates, offset = [], None
            for _ in range(5):
                payload = {
                    "limit": 200,
                    "with_payload": True,
                    "with_vector": False,
                    "filter": {"must": [{"key": "user_id", "match": {"value": settings.owner_id}}]},
                }
                if offset is not None:
                    payload["offset"] = offset
                response = await client.post(
                    f"{settings.qdrant_url}/collections/{settings.qdrant_collection}/points/scroll",
                    json=payload,
                )
                response.raise_for_status()
                result = response.json()["result"]
                for point in result["points"]:
                    p = point.get("payload", {})
                    content = p.get("data", p.get("memory", ""))
                    if not isinstance(content, str):
                        continue
                    score = len(terms & set(re.findall(r"\w+", content.lower())))
                    if score:
                        candidates.append(
                            {
                                "id": str(point["id"]),
                                "content": content,
                                "attribution": "legacy_unverified",
                                "score": score,
                            }
                        )
                offset = result.get("next_page_offset")
                if offset is None:
                    break
            return sorted(candidates, key=lambda x: x["score"], reverse=True)[:limit]
    except (httpx.HTTPError, ValueError, KeyError):
        return []


async def context(db, owner, query):
    current = search(db, owner, query, 5)
    legacy = await legacy_search(query)
    return {"current": current, "legacy": legacy}
