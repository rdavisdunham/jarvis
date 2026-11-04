"""Context platform: meeting sources, source-backed assertions, retrieval and briefs (an extractable module).

Boundary rules (plan section 2):
1. This package reaches the rest of Eridani only through entity lookup and candidate search, the command path
   for every planner write (tasks, actors, notifications), and auth, principal and bot-credential scopes.
   Nothing outside it imports from it except the registration points (worker dispatch, route mounting, MCP
   definitions, model import block, cost labels, scope table).
2. Every reader, Eridani's own frontend included, goes through the context REST and MCP API; nothing queries
   `context_*` tables directly.
3. Context IDs are opaque. Eridani entities are referenced as `(entity_type, entity_id)` pairs validated in
   application code, never as foreign keys.

Contracts live in `schema.py`, engine protocols in `engine.py`, deterministic test engines in `doubles.py`.
"""
