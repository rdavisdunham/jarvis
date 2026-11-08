"""Runs can be dismissed from the remotea page, and a run records how it started.

`agent_runs.dismissed_at`: the owner hid the run from the remotea page (an active run is cancelled first). The task's
own run history still lists it. `agent_runs.origin`: "task" (Run on a task) or "session" (New session on the remotea
page: a freestyle run that talks to the agent directly until the owner asks for a plan)."""
import sqlalchemy as sa
from alembic import op

revision = "0059_run_dismiss_origin"
down_revision = "0058_in_review_status"
branch_labels = depends_on = None


def upgrade():
    op.add_column("agent_runs", sa.Column("dismissed_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("agent_runs", sa.Column("origin", sa.String(20), nullable=False, server_default="task"))


def downgrade():
    op.drop_column("agent_runs", "origin")
    op.drop_column("agent_runs", "dismissed_at")
