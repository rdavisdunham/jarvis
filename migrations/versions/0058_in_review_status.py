"""In review becomes a built-in status meaning; existing work types get an "In review" status.

`tasks.status` is a free string, so the new core meaning needs no DDL. This data migration inserts
`{"id": "in_review", "name": "In review", "meaning": "in_review"}` into every stored work type that lacks one,
right after its last In progress status. It runs once, so a user who later retires the status keeps it retired."""
import json

import sqlalchemy as sa
from alembic import op

revision = "0058_in_review_status"
down_revision = "0057_memory_core"
branch_labels = depends_on = None

STATUS = {"id": "in_review", "name": "In review", "meaning": "in_review"}


def _schemas(bind):
    return bind.execute(sa.text("SELECT owner_id, definition FROM structure_schemas")).fetchall()


def _save(bind, owner, definition):
    bind.execute(
        sa.text(
            "UPDATE structure_schemas SET definition = CAST(:d AS jsonb), revision = revision + 1, updated_at = now() "
            "WHERE owner_id = :o"
        ),
        {"d": json.dumps(definition), "o": owner},
    )


def _work_types(definition):
    for t in (definition or {}).get("types") or []:
        if "work" in (t.get("capabilities") or []) and t.get("statuses"):
            yield t


def add_in_review(statuses):
    """Return statuses with In review inserted, or None when the type already has it."""
    if any(s.get("id") == "in_review" or s.get("meaning") == "in_review" for s in statuses):
        return None
    at = next((i + 1 for i in range(len(statuses) - 1, -1, -1) if statuses[i].get("meaning") == "in_progress"), None)
    if at is None:
        at = next((i for i, s in enumerate(statuses) if s.get("meaning") in ("completed", "cancelled")), len(statuses))
    return [*statuses[:at], dict(STATUS), *statuses[at:]]


def upgrade():
    bind = op.get_bind()
    for owner, definition in _schemas(bind):
        changed = False
        for t in _work_types(definition):
            statuses = add_in_review(t["statuses"])
            if statuses is not None:
                t["statuses"], changed = statuses, True
        if changed:
            _save(bind, owner, definition)


def downgrade():
    bind = op.get_bind()
    bind.execute(sa.text("UPDATE tasks SET status = 'in_progress' WHERE status = 'in_review'"))
    for owner, definition in _schemas(bind):
        changed = False
        for t in (definition or {}).get("types") or []:
            statuses = t.get("statuses") or []
            gone = [s["id"] for s in statuses if s.get("meaning") == "in_review"]
            if not gone:
                continue
            kept = [s for s in statuses if s.get("meaning") != "in_review"]
            target = next((s["id"] for s in kept if s.get("meaning") == "in_progress"), None)
            if target is None:
                target = "in_progress" if any(s["id"] == "in_progress" for s in kept) else (kept[0]["id"] if kept else None)
            bind.execute(
                sa.text(
                    "UPDATE structure_records SET status_id = :target "
                    "WHERE owner_id = :o AND type_id = :t AND status_id = ANY(:gone)"
                ),
                {"target": target, "o": owner, "t": t["id"], "gone": gone},
            )
            t["statuses"], changed = kept, True
        if changed:
            _save(bind, owner, definition)
