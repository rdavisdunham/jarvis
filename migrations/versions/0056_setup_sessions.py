"""Set up with Dani: the interview session and its organization proposals (docs/SETUP_WITH_DANI_PLAN.md)."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0056_setup_sessions"
down_revision = "0055_about_you"
branch_labels = depends_on = None


def upgrade():
    op.create_table(
        "setup_sessions",
        sa.Column("owner_id", sa.String(100), primary_key=True),
        sa.Column("conversation_id", sa.String(36), sa.ForeignKey("conversations.id"), nullable=False),
        sa.Column("mode", sa.String(20), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("focus", sa.String(20), nullable=False),
        sa.Column("notes", postgresql.JSONB(), nullable=False),
        sa.Column("tree_proposal_id", sa.String(36)),
        sa.Column("memory_candidates", postgresql.JSONB(), nullable=False),
        sa.Column("confirmed_memory_ids", postgresql.JSONB(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("model", sa.String(100)),
        sa.Column("cost", sa.Float(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
    )
    op.create_index("ix_setup_sessions_conversation_id", "setup_sessions", ["conversation_id"])
    op.create_table(
        "organization_proposals",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("owner_id", sa.String(100), nullable=False),
        sa.Column("account_id", sa.String(100), nullable=False),
        sa.Column("conversation_id", sa.String(36)),
        sa.Column("tree", postgresql.JSONB(), nullable=False),
        sa.Column("impact", postgresql.JSONB(), nullable=False),
        sa.Column("preview_hash", sa.String(64), nullable=False),
        sa.Column("applied_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("command_id", sa.String(100)),
    )
    op.create_index("ix_organization_proposals_owner_id", "organization_proposals", ["owner_id"])


def downgrade():
    op.drop_table("organization_proposals")
    op.drop_table("setup_sessions")
