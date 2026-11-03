"""Rename the default assistant while retaining actor identities and user content."""

import sqlalchemy as sa
from alembic import op

revision = "0046_dani_assistant"
down_revision = "0045_durable_runs"
branch_labels = depends_on = None


def upgrade():
    db = op.get_bind()
    # Freeze this data migration's names here rather than importing mutable app code.
    actors = list(db.execute(sa.text(
        "SELECT id, owner_id FROM actors WHERE name='Eri' AND kind='agent' ORDER BY owner_id, id"
    )).mappings())
    for actor in actors:
        names = {name.casefold() for name in db.scalars(sa.text(
            "SELECT name FROM actors WHERE owner_id=:owner"
        ), {"owner": actor["owner_id"]})}
        name, number = "Dani", 1
        while name.casefold() in names:
            suffix = "assistant" if number == 1 else f"assistant {number}"
            name = f"Dani ({suffix})"
            number += 1
        values = {"id": actor["id"], "owner": actor["owner_id"], "name": name}
        db.execute(sa.text("UPDATE actors SET name=:name, revision=revision+1 WHERE id=:id"), values)
        db.execute(sa.text(
            "UPDATE tasks SET assignee=:name, revision=revision+1, updated_at=now() "
            "WHERE owner_id=:owner AND assignee_id=:id"
        ), values)


def downgrade():
    # Keep the new display names: older code supports arbitrary actor names too.
    # Reversing by name could overwrite a person's name or an intentional later edit.
    pass
