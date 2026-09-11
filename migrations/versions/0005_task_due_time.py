"""Optional local task due times with an explicit timezone."""

import sqlalchemy as sa
from alembic import op

revision = "0005_task_due_time"
down_revision = "0004_memory_review"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("tasks", sa.Column("due_time", sa.String(14), nullable=True))
    op.add_column("tasks", sa.Column("due_timezone", sa.String(100), nullable=True))


def downgrade():
    op.drop_column("tasks", "due_timezone")
    op.drop_column("tasks", "due_time")
