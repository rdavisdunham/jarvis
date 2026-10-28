"""Stage F (security audit 2026-10-28): pending Eri control confirmations, run-token salt, ingestion quota."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0031_agent_security"
down_revision = "0030_agent_sync"
branch_labels = depends_on = None


def upgrade():
    op.add_column("agent_runs", sa.Column("run_token_salt", sa.String(64), nullable=False, server_default=""))
    op.create_table(
        "agent_pending_controls",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("owner_id", sa.String(100), nullable=False),
        sa.Column("account_id", sa.String(100), nullable=False),
        sa.Column("work_id", sa.String(36)),
        sa.Column("tool", sa.String(40), nullable=False),
        sa.Column("arguments", JSONB(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("run_id", sa.String(36)),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("actor", JSONB(), nullable=False),
        sa.Column("decided_via", sa.String(20), nullable=False),
        sa.Column("result", JSONB()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_agent_pending_controls_owner_id", "agent_pending_controls", ["owner_id"])
    op.create_table(
        "agent_ingest_days",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("owner_id", sa.String(100), nullable=False),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("bytes", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("owner_id", "day", name="uq_agent_ingest_day"),
    )
    op.create_index("ix_agent_ingest_days_owner_id", "agent_ingest_days", ["owner_id"])


def downgrade():
    op.drop_table("agent_ingest_days")
    op.drop_table("agent_pending_controls")
    op.drop_column("agent_runs", "run_token_salt")
