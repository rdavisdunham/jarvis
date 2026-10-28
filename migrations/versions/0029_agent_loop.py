"""Planning and the judge loop: plan-pending flag and bound turn on runs, attempt deadlines, frozen child ids."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0029_agent_loop"
down_revision = "0028_agent_runs"
branch_labels = depends_on = None


def upgrade():
    # The root turn the run is on (the host's turn seq), and a plan submitted but not yet executing.
    op.add_column("agent_runs", sa.Column("turn_seq", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("agent_runs", sa.Column("plan_pending", sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column("agent_attempts", sa.Column("deadline_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("agent_attempts", sa.Column("retries", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("agent_attempts", sa.Column("verify_id", sa.String(64), nullable=False, server_default=""))
    op.add_column("agent_attempts", sa.Column("initiator", sa.String(20), nullable=False, server_default="eridani"))
    op.create_index("ix_agent_attempts_run_id_kind_status", "agent_attempts", ["run_id", "kind", "status"])
    op.add_column("agent_run_plans", sa.Column("required_children", postgresql.JSONB(), nullable=False,
                                               server_default="[]"))
    op.add_column("agent_interviews", sa.Column("generation", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("agent_interviews", sa.Column("answered_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("agent_interviews", sa.Column("answered_via", sa.String(20), nullable=False, server_default=""))


def downgrade():
    for column in ("answered_via", "answered_at", "generation"):
        op.drop_column("agent_interviews", column)
    op.drop_column("agent_run_plans", "required_children")
    op.drop_index("ix_agent_attempts_run_id_kind_status", table_name="agent_attempts")
    for column in ("initiator", "verify_id", "retries", "deadline_at"):
        op.drop_column("agent_attempts", column)
    for column in ("plan_pending", "turn_seq"):
        op.drop_column("agent_runs", column)
