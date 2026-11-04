"""Allow two actors with the same display name in one workspace.

Meeting ingestion creates people from attendee lists, where names repeat. Identity is the actor ID;
name lookups must resolve only when exactly one non-archived actor matches.
"""

from alembic import op

revision = "0047_actor_name_uniqueness"
down_revision = "0046_dani_assistant"
branch_labels = depends_on = None


def upgrade():
    op.drop_constraint("uq_actor_owner_name", "actors", type_="unique")
    op.create_index("ix_actors_owner_name", "actors", ["owner_id", "name"])


def downgrade():
    # Refuses (IntegrityError) if duplicate names exist; rename or merge them first.
    op.drop_index("ix_actors_owner_name", table_name="actors")
    op.create_unique_constraint("uq_actor_owner_name", "actors", ["owner_id", "name"])
