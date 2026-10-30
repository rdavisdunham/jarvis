"""Swarm mode: a project's open tasks planned into batches of parallel runs (agent_swarms)."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0040_agent_swarms"
down_revision = "0039_linear_structure"
branch_labels = depends_on = None


def upgrade():
    op.create_table(
        "agent_swarms",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("owner_id", sa.String(100), nullable=False),
        sa.Column("account_id", sa.String(100), sa.ForeignKey("user_accounts.id"), nullable=True),
        sa.Column("record_id", sa.String(36), sa.ForeignKey("structure_records.id"), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("state", sa.String(20), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("cap", sa.Integer(), nullable=False),
        sa.Column("auto_continue", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("harness", sa.String(20), nullable=False),
        sa.Column("batches", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("tasks", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("excluded", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("planner", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_agent_swarms_owner_id", "agent_swarms", ["owner_id"])
    op.create_index("ix_agent_swarms_record_id", "agent_swarms", ["record_id"])


def downgrade():
    op.drop_index("ix_agent_swarms_record_id", table_name="agent_swarms")
    op.drop_index("ix_agent_swarms_owner_id", table_name="agent_swarms")
    op.drop_table("agent_swarms")
