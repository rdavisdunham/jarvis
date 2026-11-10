"""Restart run: a finished run continues as a new run linked to it (run_restart.py, docs/AGENT_CONTROL.md).

`agent_runs.restarted_from_run_id`: the terminal run a restart continues. The old run stays terminal; the link lets
each run's page say "Restarted from" / "Restarted as". Nullable, so the previous release's inserts still work."""
import sqlalchemy as sa
from alembic import op

revision = "0062_run_restart"
down_revision = "0061_session_recall"
branch_labels = depends_on = None


def upgrade():
    op.add_column("agent_runs", sa.Column("restarted_from_run_id", sa.String(36), nullable=True))
    op.create_foreign_key("agent_runs_restarted_from_run_id_fkey", "agent_runs", "agent_runs",
                          ["restarted_from_run_id"], ["id"])
    op.create_index("ix_agent_runs_restarted_from_run_id", "agent_runs", ["restarted_from_run_id"])


def downgrade():
    op.drop_index("ix_agent_runs_restarted_from_run_id", table_name="agent_runs")
    op.drop_constraint("agent_runs_restarted_from_run_id_fkey", "agent_runs", type_="foreignkey")
    op.drop_column("agent_runs", "restarted_from_run_id")
