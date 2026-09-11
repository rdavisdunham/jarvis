"""Keep reminder completion independently from delivery and cancellation."""

import sqlalchemy as sa
from alembic import op

revision = "0002_reminder_completion"
down_revision = "0001_core"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("schedules", sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("notifications", sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True))


def downgrade():
    op.drop_column("notifications", "completed_at")
    op.drop_column("schedules", "completed_at")
