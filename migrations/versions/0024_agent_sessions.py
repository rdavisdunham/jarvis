"""Agent session ledger: external coding/chat sessions and the folder rules that file them."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0024_agent_sessions"
down_revision = "0023_record_reviews"
branch_labels = depends_on = None


def upgrade():
    op.create_table(
        "agent_sessions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("owner_id", sa.String(100), nullable=False),
        sa.Column("tool", sa.String(30), nullable=False),
        sa.Column("session_id", sa.String(100), nullable=False),
        sa.Column("machine", sa.String(100), nullable=False),
        sa.Column("environment", sa.String(20), nullable=False),
        sa.Column("cwd", sa.Text(), nullable=False),
        sa.Column("repo", sa.String(300), nullable=False),
        sa.Column("branch", sa.String(300), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("first_prompt", sa.Text(), nullable=False),
        sa.Column("last_prompt", sa.Text(), nullable=False),
        sa.Column("excerpt", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("pr_urls", postgresql.JSONB(), nullable=False),
        sa.Column("issue_refs", postgresql.JSONB(), nullable=False),
        sa.Column("resume", postgresql.JSONB(), nullable=False),
        sa.Column("turn_count", sa.Integer(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_active_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("binding_source", sa.String(20), nullable=False),
        sa.Column("record_id", sa.String(36), sa.ForeignKey("structure_records.id"), nullable=True),
        sa.Column("task_id", sa.String(36), sa.ForeignKey("tasks.id"), nullable=True),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("summary_fingerprint", sa.String(64), nullable=False),
        sa.Column("embedding", postgresql.JSONB(), nullable=True),
        sa.Column("embedding_model", sa.String(100), nullable=True),
        sa.Column("archived", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("owner_id", "tool", "session_id", name="uq_agent_session"),
    )
    op.create_index("ix_agent_sessions_owner_id", "agent_sessions", ["owner_id"])
    op.create_index("ix_agent_sessions_owner_id_last_active_at", "agent_sessions", ["owner_id", "last_active_at"])
    op.create_index("ix_agent_sessions_record_id", "agent_sessions", ["record_id"])
    op.create_index("ix_agent_sessions_task_id", "agent_sessions", ["task_id"])
    op.create_table(
        "agent_session_rules",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("owner_id", sa.String(100), nullable=False),
        sa.Column("path_prefix", sa.String(500), nullable=False),
        sa.Column("record_id", sa.String(36), sa.ForeignKey("structure_records.id"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("owner_id", "path_prefix", name="uq_agent_session_rule"),
    )
    op.create_index("ix_agent_session_rules_owner_id", "agent_session_rules", ["owner_id"])


def downgrade():
    op.drop_table("agent_session_rules")
    op.drop_table("agent_sessions")
