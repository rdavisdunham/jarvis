"""Readiness pass: a task's route and what the pass found (readiness.py, docs/AGENT_CONTROL.md).

`tasks.run_route`: auto | direction | plan_review | human, null until decided. `tasks.readiness`: what the pass
found and filled in. Nullable or defaulted, so the previous release's inserts still work."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0063_task_readiness"
down_revision = "0062_run_restart"
branch_labels = depends_on = None


def upgrade():
    op.add_column("tasks", sa.Column("run_route", sa.String(20), nullable=True))
    op.add_column("tasks", sa.Column("readiness", postgresql.JSONB(), nullable=False, server_default="{}"))


def downgrade():
    op.drop_column("tasks", "readiness")
    op.drop_column("tasks", "run_route")
