"""Compound: what finished runs learned, per repository (run_learnings.py, docs/AGENT_CONTROL.md "Compound").

A new table only; nothing existing changes, so the previous release keeps working."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0064_run_learnings"
down_revision = "0063_task_readiness"
branch_labels = depends_on = None


def upgrade():
    op.create_table(
        "agent_run_learnings",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("owner_id", sa.String(100), nullable=False),
        sa.Column("execution_target_id", sa.String(36), nullable=False),
        sa.Column("run_id", sa.String(36), nullable=True),
        sa.Column("task_id", sa.String(36), nullable=True),
        sa.Column("kind", sa.String(20), nullable=False, server_default="knowledge"),
        sa.Column("title", sa.String(200), nullable=False, server_default=""),
        sa.Column("applies_when", sa.Text(), nullable=False, server_default=""),
        sa.Column("body", sa.Text(), nullable=False, server_default=""),
        sa.Column("paths", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("confirmations", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("confirmed_by", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("status", sa.String(20), nullable=False, server_default="active"),
        sa.Column("meta", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_agent_run_learnings_owner_id", "agent_run_learnings", ["owner_id"])
    op.create_index("ix_agent_run_learnings_target", "agent_run_learnings", ["owner_id", "execution_target_id", "status"])


def downgrade():
    op.drop_index("ix_agent_run_learnings_target", table_name="agent_run_learnings")
    op.drop_index("ix_agent_run_learnings_owner_id", table_name="agent_run_learnings")
    op.drop_table("agent_run_learnings")
