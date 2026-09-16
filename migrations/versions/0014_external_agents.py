"""Scoped bot credentials and attribution for external/durable requests."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0014_external_agents"
down_revision = "0013_agent_work"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "bot_credentials",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("owner_id", sa.String(100), nullable=False),
        sa.Column("account_id", sa.String(100), sa.ForeignKey("user_accounts.id"), nullable=False),
        sa.Column("name", sa.String(80), nullable=False),
        sa.Column("token_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("prefix", sa.String(20), nullable=False),
        sa.Column("scopes", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True)),
        sa.Column("last_used_at", sa.DateTime(timezone=True)),
        sa.Column("rate_window", sa.DateTime(timezone=True), nullable=False),
        sa.Column("rate_count", sa.Integer, nullable=False),
    )
    op.create_index("ix_bot_credentials_owner_id", "bot_credentials", ["owner_id"])
    op.create_index("ix_bot_credentials_account_id", "bot_credentials", ["account_id"])
    op.add_column("agent_work", sa.Column("credential_id", sa.String(36)))
    op.create_index("ix_agent_work_credential_id", "agent_work", ["credential_id"])


def downgrade():
    op.drop_index("ix_agent_work_credential_id", table_name="agent_work")
    op.drop_column("agent_work", "credential_id")
    op.drop_table("bot_credentials")
