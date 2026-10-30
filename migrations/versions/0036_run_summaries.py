"""Run summaries: a one-line plain-language note of what a run's agent is doing now (run_summaries.py)."""
import sqlalchemy as sa
from alembic import op

revision = "0036_run_summaries"
down_revision = "0035_action_receipts"
branch_labels = depends_on = None


def upgrade():
    op.add_column("agent_runs", sa.Column("summary", sa.Text(), nullable=False, server_default=""))
    op.add_column("agent_runs", sa.Column("summary_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("agent_runs", sa.Column("summary_seq", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("agent_runs", sa.Column("activity_seq", sa.Integer(), nullable=False, server_default="0"))


def downgrade():
    for column in ("activity_seq", "summary_seq", "summary_at", "summary"):
        op.drop_column("agent_runs", column)
