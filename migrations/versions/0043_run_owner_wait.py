"""The wall clock counts the agent's working time: agent_runs.owner_wait_since and owner_wait_total_s.

Backfill for unsettled runs: a run with an open interview for its current generation has waited since that interview
was asked; a run in needs_you or waiting_answer has waited since it entered that state. Nothing slides here: the
run token's expiry moves forward when the wait ends (run_coordinator.end_wait).
"""
import sqlalchemy as sa
from alembic import op

revision = "0043_run_owner_wait"
down_revision = "0042_bug_reports"
branch_labels = depends_on = None


# Open interview first (the wait began when it was asked), then needs_you / waiting_answer since the state began.
BACKFILL = (
    """UPDATE agent_runs AS r SET owner_wait_since = i.asked
       FROM (SELECT run_id, generation, min(created_at) AS asked FROM agent_interviews
             WHERE status = 'open' GROUP BY run_id, generation) AS i
       WHERE i.run_id = r.id AND i.generation = r.generation
         AND r.state NOT IN ('completed', 'stopped', 'capped', 'failed', 'stopping', 'plan_reviewing')""",
    """UPDATE agent_runs SET owner_wait_since = state_since
       WHERE owner_wait_since IS NULL AND state IN ('needs_you', 'waiting_answer')""",
)


def upgrade():
    op.add_column("agent_runs", sa.Column("owner_wait_since", sa.DateTime(timezone=True), nullable=True))
    op.add_column("agent_runs", sa.Column("owner_wait_total_s", sa.Integer(), nullable=False, server_default="0"))
    for statement in BACKFILL:
        op.execute(statement)


def downgrade():
    op.drop_column("agent_runs", "owner_wait_total_s")
    op.drop_column("agent_runs", "owner_wait_since")
