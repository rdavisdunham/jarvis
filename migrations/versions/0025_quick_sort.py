"""Durable human-guided filing sessions; existing records/queue stay authoritative."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
revision = "0025_quick_sort"
down_revision = "0024_agent_sessions"
branch_labels = depends_on = None

def upgrade():
    op.create_table("quick_sort_sessions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("owner_id", sa.String(100), nullable=False),
        sa.Column("account_id", sa.String(100), nullable=False),
        sa.Column("conversation_id", sa.String(36), nullable=False),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("state", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False))
    op.create_index("ix_quick_sort_sessions_owner_id", "quick_sort_sessions", ["owner_id"])
    op.create_index("ix_quick_sort_sessions_account_id", "quick_sort_sessions", ["account_id"])

def downgrade():
    op.drop_table("quick_sort_sessions")
