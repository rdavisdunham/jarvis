"""Old remotea conversations are searchable: `agent_turns.tsv` indexes each turn's prompt and reply for the session
ledger's search (GIN, the same stored-column pattern as context search)."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0060_turn_search"
down_revision = "0059_run_dismiss_origin"
branch_labels = depends_on = None


def upgrade():
    op.add_column("agent_turns", sa.Column(
        "tsv", postgresql.TSVECTOR(), sa.Computed("to_tsvector('english', prompt || ' ' || text)", persisted=True),
        nullable=False))
    op.create_index("ix_agent_turns_tsv", "agent_turns", ["tsv"], postgresql_using="gin")


def downgrade():
    op.drop_index("ix_agent_turns_tsv", table_name="agent_turns")
    op.drop_column("agent_turns", "tsv")
