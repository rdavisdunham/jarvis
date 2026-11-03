"""Durable human decisions, outbox identity, role options and explicit pause."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0045_durable_runs"
down_revision = "0044_task_archive"
branch_labels = depends_on = None


def upgrade():
    op.alter_column("tasks", "turn_cap", server_default="0")
    op.add_column("tasks", sa.Column("completion_policy_explicit", sa.Boolean(), nullable=False, server_default="false"))
    op.execute("UPDATE tasks SET completion_policy_explicit=true")
    op.alter_column("agent_runs", "turn_cap", server_default="0")
    op.alter_column("agent_runs", "wall_clock_limit", server_default="0")
    for table in ("agent_questions", "agent_interviews"):
        op.add_column(table, sa.Column("plan_revision", sa.Integer(), nullable=False, server_default="0"))
        op.execute(f"UPDATE {table} q SET plan_revision=r.plan_revision FROM agent_runs r WHERE q.run_id=r.id")
    op.add_column("agent_questions", sa.Column("durable", sa.Boolean(), nullable=False, server_default="false"))
    op.add_column("agent_questions", sa.Column("recovery", sa.String(30), nullable=False, server_default=""))
    op.create_index("uq_agent_question_durable", "agent_questions", ["host_id", "question_id"], unique=True,
                    postgresql_where=sa.text("durable = true"))
    op.add_column("agent_questions", sa.Column("action", postgresql.JSONB(), nullable=True))
    op.add_column("agent_questions", sa.Column("request_digest", sa.String(64), nullable=True))
    op.add_column("agent_runs", sa.Column("paused", sa.Boolean(), nullable=False, server_default="false"))
    for name in ("effort", "permission_profile"):
        op.add_column("agent_runs", sa.Column(name, sa.String(40), nullable=True))
    op.add_column("agent_runs", sa.Column("activity", postgresql.JSONB(), nullable=True))
    op.add_column("agent_commands", sa.Column("action_id", sa.String(36), nullable=True))
    op.create_index("ix_agent_commands_action_id", "agent_commands", ["action_id"])
    op.add_column("agent_commands", sa.Column("plan_revision", sa.Integer(), nullable=False, server_default="0"))
    op.alter_column("agent_commands", "expires_at", nullable=True)
    op.alter_column("agent_pending_controls", "expires_at", nullable=True)
    op.execute("UPDATE agent_pending_controls SET expires_at=NULL WHERE status='pending'")
    op.add_column("agent_run_plans", sa.Column("approval_status", sa.String(20), nullable=False, server_default="pending"))
    op.add_column("agent_run_plans", sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("agent_interviews", sa.Column("answer_op_id", sa.String(100), nullable=True))
    # Historical executing plans were already accepted under the previous policy; do not stall them retroactively.
    op.execute("UPDATE agent_run_plans p SET approval_status='approved' FROM agent_runs r WHERE r.id=p.run_id AND (NOT r.plan_pending OR r.state IN ('executing','verifying','judging','completed','stopped','capped','failed'))")


def downgrade():
    op.alter_column("tasks", "turn_cap", server_default="15")
    op.drop_column("tasks", "completion_policy_explicit")
    op.alter_column("agent_runs", "turn_cap", server_default="15")
    op.alter_column("agent_runs", "wall_clock_limit", server_default="21600")
    for table in ("agent_questions", "agent_interviews"):
        op.drop_column(table, "plan_revision")
    op.drop_index("uq_agent_question_durable", table_name="agent_questions")
    op.drop_column("agent_questions", "durable")
    op.drop_column("agent_questions", "recovery")
    op.drop_column("agent_questions", "request_digest")
    op.drop_column("agent_questions", "action")
    op.execute("UPDATE agent_pending_controls SET expires_at=now()+interval '100 years' WHERE expires_at IS NULL")
    op.alter_column("agent_pending_controls", "expires_at", nullable=False)
    op.execute("UPDATE agent_commands SET expires_at=now()+interval '100 years' WHERE expires_at IS NULL")
    op.alter_column("agent_commands", "expires_at", nullable=False)
    for table, names in (("agent_runs", ("paused", "effort", "permission_profile", "activity")),
                         ("agent_commands", ("action_id", "plan_revision")),
                         ("agent_run_plans", ("approval_status", "approved_at")),
                         ("agent_interviews", ("answer_op_id",))):
        for name in names:
            op.drop_column(table, name)
