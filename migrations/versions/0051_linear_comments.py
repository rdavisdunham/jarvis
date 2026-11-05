"""Linear comments copied onto each synced issue link, read-only (linear_sync.py)."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0051_linear_comments"
down_revision = "0050_context_review"
branch_labels = depends_on = None


def upgrade():
    op.add_column("linear_issues", sa.Column("comments", postgresql.JSONB(), nullable=False, server_default="[]"))


def downgrade():
    op.drop_column("linear_issues", "comments")
