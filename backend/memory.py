"""
Long-term memory for JARVIS using Mem0 + Qdrant.

Provides semantic memory extraction, storage, and retrieval.
All functions gracefully degrade if memory is disabled or Qdrant is unavailable.

Memories are stored in two categories:
- "user": facts about the user (preferences, opinions, experiences, goals)
- "jarvis": facts about what JARVIS said (recommendations, advice, explanations)
"""

import logging
import os
import sys
import threading
import time

# Suppress noisy Mem0/Qdrant logs (Pydantic validation spam from event:'NONE' bug)
for _logger_name in ("mem0", "mem0ai", "qdrant_client", "httpx", "httpcore",
                      "pydantic", "openai", "groq"):
    logging.getLogger(_logger_name).setLevel(logging.CRITICAL)

MEMORY_ENABLED = os.environ.get("MEMORY_ENABLED", "true").lower() == "true"
MEMORY_DEVICE = os.environ.get("MEMORY_DEVICE", "cuda")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))

USER_ID = "davin"
MEMORY_LLM_MODEL = os.environ.get("MEMORY_LLM_MODEL", "llama-3.3-70b-versatile")


def _search_threshold():
    return float(os.environ.get("MEMORY_SEARCH_THRESHOLD", "0.5"))


def _search_limit_user():
    return int(os.environ.get("MEMORY_SEARCH_LIMIT_USER", "10"))


def _search_limit_jarvis():
    return int(os.environ.get("MEMORY_SEARCH_LIMIT_JARVIS", "10"))


# Runtime memory mode: "full" (search+store), "read-only" (search only), "off" (disabled)
_memory_mode = "full"


def set_mode(mode):
    """Set memory mode: 'full', 'read-only', or 'off'."""
    global _memory_mode
    if mode in ("full", "read-only", "off"):
        _memory_mode = mode
        print(f"Memory mode set to: {mode}", file=sys.stderr)
    else:
        print(f"Invalid memory mode: {mode}", file=sys.stderr)


def get_mode():
    return _memory_mode


_mem_instance = None
_mem_lock = threading.Lock()
_mem_failed = False
_mem_retry_time = 0
_RETRY_INTERVAL = 30  # seconds


def _get_memory():
    """Lazy-init Mem0 Memory instance (thread-safe singleton)."""
    global _mem_instance, _mem_failed, _mem_retry_time

    if not MEMORY_ENABLED:
        return None

    if _mem_instance is not None:
        return _mem_instance

    # If init previously failed, wait before retrying
    if _mem_failed and time.time() < _mem_retry_time:
        return None

    with _mem_lock:
        # Double-check after acquiring lock
        if _mem_instance is not None:
            return _mem_instance

        try:
            from mem0 import Memory

            config = {
                "llm": {
                    "provider": "groq",
                    "config": {
                        "model": MEMORY_LLM_MODEL,
                        "temperature": 0.1,
                        "max_tokens": 1000,
                    },
                },
                "embedder": {
                    "provider": "huggingface",
                    "config": {
                        "model": "sentence-transformers/all-MiniLM-L6-v2",
                        "model_kwargs": {"device": MEMORY_DEVICE},
                    },
                },
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "host": QDRANT_HOST,
                        "port": QDRANT_PORT,
                        "collection_name": "jarvis_memories",
                        "embedding_model_dims": 384,
                    },
                },
            }

            _mem_instance = Memory.from_config(config)
            _mem_failed = False
            print("Memory system initialized (Mem0 + Qdrant)", file=sys.stderr)
            return _mem_instance

        except Exception as e:
            _mem_failed = True
            _mem_retry_time = time.time() + _RETRY_INTERVAL
            print(f"Memory init failed (will retry in {_RETRY_INTERVAL}s): {e}", file=sys.stderr)
            return None


def _filtered_search(mem, query, source, limit):
    """Search memories filtered by source tag, applying relevance threshold."""
    threshold = _search_threshold()
    results = mem.search(
        query, user_id=USER_ID,
        filters={"source": source},
        limit=limit,
    )
    if not results or not results.get("results"):
        return []
    memories = results["results"]
    # Apply relevance threshold if scores are available
    filtered = []
    for m in memories:
        if not m.get("memory"):
            continue
        score = m.get("score", 1.0)
        if score >= threshold:
            filtered.append(m["memory"])
    return filtered


def search_memories(query):
    """Search for relevant memories. Returns formatted string or empty string."""
    if _memory_mode == "off":
        return ""

    mem = _get_memory()
    if mem is None:
        return ""

    try:
        t0 = time.time()

        user_memories = _filtered_search(mem, query, "user", _search_limit_user())
        jarvis_memories = _filtered_search(mem, query, "jarvis", _search_limit_jarvis())

        elapsed_ms = (time.time() - t0) * 1000
        print(
            f"[timing] Memory search: {elapsed_ms:.0f}ms "
            f"({len(user_memories)} user, {len(jarvis_memories)} jarvis)",
            file=sys.stderr,
        )

        if not user_memories and not jarvis_memories:
            return ""

        sections = ["## Long-term memory\n"]

        if user_memories:
            sections.append("### What I know about you (Davin)")
            for m in user_memories:
                sections.append(f"- {m}")

        if jarvis_memories:
            sections.append("\n### What I (JARVIS) have said in past conversations")
            for m in jarvis_memories:
                sections.append(f"- {m}")

        return "\n".join(sections)

    except Exception as e:
        print(f"Memory search failed: {e}", file=sys.stderr)
        return ""


def add_memory_background(user_msg, assistant_msg):
    """Extract and store memories in a background thread (both user and JARVIS facts)."""
    if _memory_mode != "full":
        return

    mem = _get_memory()
    if mem is None:
        return

    def _extract():
        try:
            t0 = time.time()
            messages = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]

            # Extract user facts (no agent_id → USER_MEMORY_EXTRACTION_PROMPT)
            user_result = mem.add(
                messages, user_id=USER_ID,
                metadata={"source": "user"},
            )
            n_user = len(user_result.get("results", [])) if user_result else 0

            # Extract JARVIS facts (agent_id → AGENT_MEMORY_EXTRACTION_PROMPT)
            jarvis_result = mem.add(
                messages, user_id=USER_ID, agent_id="jarvis",
                metadata={"source": "jarvis"},
            )
            n_jarvis = len(jarvis_result.get("results", [])) if jarvis_result else 0

            elapsed_ms = (time.time() - t0) * 1000
            print(
                f"[timing] Memory extraction: {elapsed_ms:.0f}ms "
                f"({n_user} user, {n_jarvis} jarvis)",
                file=sys.stderr,
            )
        except Exception as e:
            print(f"Memory extraction failed: {e}", file=sys.stderr)

    thread = threading.Thread(target=_extract, daemon=True)
    thread.start()


def get_all_memories():
    """Return all stored memories (for debugging)."""
    mem = _get_memory()
    if mem is None:
        return []

    try:
        results = mem.get_all(user_id=USER_ID)
        return results.get("results", []) if results else []
    except Exception as e:
        print(f"Memory get_all failed: {e}", file=sys.stderr)
        return []
