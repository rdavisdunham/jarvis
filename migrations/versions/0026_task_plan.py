"""Task execution plan: steps, PRD note, success criteria, verify config, turn cap and plan revision."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0026_task_plan"
down_revision = "0025_quick_sort"
branch_labels = depends_on = None


def upgrade():
    op.add_column("tasks", sa.Column("steps", postgresql.JSONB(), nullable=False, server_default="[]"))
    op.add_column("tasks", sa.Column("prd_note_id", sa.String(36), sa.ForeignKey("notes.id"), nullable=True))
    op.add_column("tasks", sa.Column("success_criteria", postgresql.JSONB(), nullable=False, server_default="[]"))
    op.add_column("tasks", sa.Column("verify", postgresql.JSONB(), nullable=True))
    op.add_column("tasks", sa.Column("turn_cap", sa.Integer(), nullable=False, server_default="15"))
    # execution_targets arrives in 0028; the foreign key is added there.
    op.add_column("tasks", sa.Column("execution_target_id", sa.String(36), nullable=True))
    op.add_column("tasks", sa.Column("plan_revision", sa.Integer(), nullable=False, server_default="0"))
    op.create_index("ix_tasks_prd_note_id", "tasks", ["prd_note_id"])


def downgrade():
    op.drop_index("ix_tasks_prd_note_id", table_name="tasks")
    for column in ("plan_revision", "execution_target_id", "turn_cap", "verify", "success_criteria", "prd_note_id", "steps"):
        op.drop_column("tasks", column)
