"""Separate optional question state from invitation delivery."""

from alembic import op
import sqlalchemy as sa

revision = "0021_review_delivery"
down_revision = "0020_capture_setup"
branch_labels = depends_on = None


def upgrade():
    op.add_column(
        "field_understandings", sa.Column("deferred_until", sa.DateTime(timezone=True), nullable=True)
    )
    op.create_table(
        "review_deliveries",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("owner_id", sa.String(100), nullable=False),
        sa.Column("device_id", sa.String(36), nullable=False),
        sa.Column("conversation_id", sa.String(36), sa.ForeignKey("conversations.id"), nullable=False),
        sa.Column("question_key", sa.String(300), nullable=False),
        sa.Column("question_revision", sa.Integer(), nullable=False),
        sa.Column("channel", sa.String(20), nullable=False),
        sa.Column("state", sa.String(20), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    for column in ("owner_id", "conversation_id"):
        op.create_index("ix_review_deliveries_" + column, "review_deliveries", [column])


def downgrade():
    op.drop_table("review_deliveries")
    op.drop_column("field_understandings", "deferred_until")
