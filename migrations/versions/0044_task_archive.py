"""Durable task outcomes and independent archive grace/metadata."""
import sqlalchemy as sa
from alembic import op

revision = "0044_task_archive"
down_revision = "0043_run_owner_wait"
branch_labels = depends_on = None


def upgrade():
    for name in ("finished_at", "archive_eligible_at", "archived_at"):
        op.add_column("tasks", sa.Column(name, sa.DateTime(timezone=True), nullable=True))
    op.add_column("tasks", sa.Column("archive_reason", sa.String(20), nullable=True))
    op.create_index("ix_tasks_archive_eligible_at", "tasks", ["archive_eligible_at"])
    op.execute("UPDATE tasks SET finished_at = completed_at, archive_eligible_at = completed_at + INTERVAL '24 hours' WHERE status = 'completed' AND completed_at IS NOT NULL")
    # Old archive provenance is not reliably distinguishable; retain intentional filing.
    op.execute("UPDATE tasks SET archive_reason = 'manual' WHERE archived = true")


def downgrade():
    op.drop_index("ix_tasks_archive_eligible_at", table_name="tasks")
    for name in ("archive_reason", "archived_at", "archive_eligible_at", "finished_at"):
        op.drop_column("tasks", name)
