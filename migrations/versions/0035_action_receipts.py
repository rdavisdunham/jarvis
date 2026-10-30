"""Run action receipts: a browser retry of message/replan/stop/approve/freestyle/verify-config with the same
action_id returns the first result instead of acting twice."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0035_action_receipts"
down_revision = "0034_run_freestyle"
branch_labels = depends_on = None


def upgrade():
    op.create_table(
        "agent_action_receipts",
        sa.Column("owner_id", sa.String(100), primary_key=True),
        sa.Column("action_id", sa.String(36), primary_key=True),
        sa.Column("run_id", sa.String(36), nullable=False),
        sa.Column("kind", sa.String(30), nullable=False),
        sa.Column("digest", sa.String(64), nullable=False),
        sa.Column("result", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_agent_action_receipts_created_at", "agent_action_receipts", ["created_at"])


def downgrade():
    op.drop_table("agent_action_receipts")
