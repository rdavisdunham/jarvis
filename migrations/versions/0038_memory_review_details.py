"""Memory review provenance: how a question's candidates were found (semantic similarity, profile field)."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0038_memory_review_details"
down_revision = "0037_notification_bundles"
branch_labels = depends_on = None


def upgrade():
    op.add_column(
        "memory_reviews",
        sa.Column("details", postgresql.JSONB(), nullable=False, server_default="{}"),
    )


def downgrade():
    op.drop_column("memory_reviews", "details")
