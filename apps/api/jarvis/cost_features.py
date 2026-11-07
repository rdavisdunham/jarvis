"""Per-request attribution; ContextVar keeps parallel owners/features independent."""
from contextvars import ContextVar
from functools import wraps
from inspect import iscoroutinefunction

current_feature = ContextVar("cost_feature", default="unattributed")
LABELS = {
    "assistant": "Assistant tasks and chat",
    "voice": "Voice conversations",
    "transcription": "Speech transcription",
    "memory_learning": "Memory learning and indexing",
    "memory_retrieval": "Memory retrieval",
    "memory_review": "Memory dream sequence",
    "about_you": "About you summary",
    "organization_learning": "Organization dream sequence",
    "field_understanding": "Custom field clarification",
    "note_organization": "Note organization",
    "note_indexing": "Note indexing",
    "note_search": "Note search",
    "note_tasks": "Tasks extracted from notes",
    "search_indexing": "Workspace search indexing",
    "semantic_search": "Semantic search",
    "session_ledger": "Session search (Eridani)",
    "agent_control": "Run judging (Eridani)",
    "agent_summaries": "Run summaries (Eridani)",
    "record_overviews": "Client and project overviews (Eridani)",
    "placement_by_meaning": "Filing tasks by meaning (Eridani)",
    "context_extraction": "Meeting context extraction",
    "context_linking": "Meeting people and entity linking",
    "context_reconciliation": "Meeting context reconciliation",
    "context_embeddings": "Meeting context embeddings",
    "context_synthesis": "Meeting briefs and summaries",
    "unattributed": "Earlier or unclassified usage",
}


def feature(name):
    if name not in LABELS:
        raise ValueError("Unknown cost feature")

    def decorate(fn):
        if iscoroutinefunction(fn):
            @wraps(fn)
            async def asynchronous(*args, **kwargs):
                token = current_feature.set(name)
                try:
                    return await fn(*args, **kwargs)
                finally:
                    current_feature.reset(token)
            return asynchronous

        @wraps(fn)
        def synchronous(*args, **kwargs):
            token = current_feature.set(name)
            try:
                return fn(*args, **kwargs)
            finally:
                current_feature.reset(token)
        return synchronous
    return decorate
