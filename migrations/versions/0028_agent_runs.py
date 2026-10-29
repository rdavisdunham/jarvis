"""Agent runs: execution targets, runs with frozen plans, attempts, interviews, autolink tombstones."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0028_agent_runs"
down_revision = "0027_agent_link"
branch_labels = depends_on = None

TERMINAL = "state NOT IN ('completed', 'stopped', 'capped', 'failed')"


def stamps():
    return [sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False)]


def upgrade():
    op.create_table(
        "execution_targets",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("owner_id", sa.String(100), nullable=False),
        sa.Column("host_id", sa.String(36), sa.ForeignKey("agent_hosts.id"), nullable=True),
        sa.Column("record_id", sa.String(36), nullable=True),
        sa.Column("host_target", sa.String(64), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("repo", sa.Text(), nullable=False),
        sa.Column("worktree_root", sa.Text(), nullable=False),
        sa.Column("launch", postgresql.JSONB(), nullable=False),
        sa.Column("verify", postgresql.JSONB(), nullable=False),
        *stamps(),
    )
    op.create_index("ix_execution_targets_owner_id", "execution_targets", ["owner_id"])
    op.create_index("ix_execution_targets_host_id", "execution_targets", ["host_id"])
    op.create_index("ix_execution_targets_record_id", "execution_targets", ["record_id"])
    op.create_index("uq_execution_target_record", "execution_targets", ["owner_id", "record_id"], unique=True,
                    postgresql_where=sa.text("record_id IS NOT NULL"))
    # 0026 added the column; the table it points at exists now.
    op.create_foreign_key("tasks_execution_target_id_fkey", "tasks", "execution_targets",
                          ["execution_target_id"], ["id"])

    op.create_table(
        "agent_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("owner_id", sa.String(100), nullable=False),
        sa.Column("account_id", sa.String(100), sa.ForeignKey("user_accounts.id"), nullable=True),
        sa.Column("task_id", sa.String(36), sa.ForeignKey("tasks.id"), nullable=False),
        sa.Column("host_id", sa.String(36), sa.ForeignKey("agent_hosts.id"), nullable=False),
        sa.Column("session_pk", sa.String(36), nullable=True),
        sa.Column("execution_target_id", sa.String(36), sa.ForeignKey("execution_targets.id"), nullable=True),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("state", sa.String(30), nullable=False),
        sa.Column("state_revision", sa.Integer(), nullable=False),
        sa.Column("state_reason", sa.String(300), nullable=False),
        sa.Column("state_since", sa.DateTime(timezone=True), nullable=False),
        sa.Column("plan_revision", sa.Integer(), nullable=False),
        sa.Column("worktree", sa.Text(), nullable=False),
        sa.Column("branch", sa.String(200), nullable=False),
        sa.Column("base_branch", sa.String(200), nullable=False),
        sa.Column("tool", sa.String(30), nullable=False),
        sa.Column("model", sa.String(80), nullable=False),
        sa.Column("effective_permission_mode", sa.String(40), nullable=False),
        sa.Column("surf", sa.String(120), nullable=False),
        sa.Column("incarnation", sa.Integer(), nullable=False),
        sa.Column("turn_count", sa.Integer(), nullable=False),
        sa.Column("turn_cap", sa.Integer(), nullable=False),
        sa.Column("wall_clock_limit", sa.Integer(), nullable=False),
        sa.Column("no_progress_limit", sa.Integer(), nullable=False),
        sa.Column("last_progress_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("stop_requested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("stop_mode", sa.String(20), nullable=False),
        sa.Column("stop_reason", sa.String(60), nullable=False),
        sa.Column("stop_final", sa.String(20), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completion_policy", sa.String(20), nullable=False),
        sa.Column("sync_state", postgresql.JSONB(), nullable=False),
        sa.Column("run_token_hash", sa.String(64), nullable=True, unique=True),
        sa.Column("run_token_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("history", postgresql.JSONB(), nullable=False),
        *stamps(),
    )
    op.create_index("ix_agent_runs_owner_id", "agent_runs", ["owner_id"])
    op.create_index("ix_agent_runs_task_id", "agent_runs", ["task_id"])
    op.create_index("ix_agent_runs_host_id", "agent_runs", ["host_id"])
    op.create_index("ix_agent_runs_session_pk", "agent_runs", ["session_pk"])
    op.create_index("ix_agent_runs_execution_target_id", "agent_runs", ["execution_target_id"])
    op.create_index("ix_agent_runs_owner_id_state", "agent_runs", ["owner_id", "state"])
    op.create_index("uq_agent_run_active_task", "agent_runs", ["task_id"], unique=True,
                    postgresql_where=sa.text(TERMINAL))

    op.create_table(
        "agent_run_plans",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("owner_id", sa.String(100), nullable=False),
        sa.Column("run_id", sa.String(36), sa.ForeignKey("agent_runs.id"), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("task_plan_revision", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("prd_text", sa.Text(), nullable=False),
        sa.Column("steps", postgresql.JSONB(), nullable=False),
        sa.Column("criteria", postgresql.JSONB(), nullable=False),
        sa.Column("verify", postgresql.JSONB(), nullable=True),
        sa.Column("frozen_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("run_id", "revision", name="uq_agent_run_plan"),
    )
    op.create_index("ix_agent_run_plans_owner_id", "agent_run_plans", ["owner_id"])
    op.create_index("ix_agent_run_plans_run_id", "agent_run_plans", ["run_id"])

    op.create_table(
        "agent_attempts",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("owner_id", sa.String(100), nullable=False),
        sa.Column("run_id", sa.String(36), sa.ForeignKey("agent_runs.id"), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(20), nullable=False),
        sa.Column("turn_seq", sa.Integer(), nullable=False),
        sa.Column("plan_revision", sa.Integer(), nullable=False),
        sa.Column("workspace_fingerprint", sa.String(200), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("result", postgresql.JSONB(), nullable=False),
        sa.Column("model", sa.String(100), nullable=False),
        sa.Column("prompt_version", sa.String(40), nullable=False),
        sa.Column("cost", sa.Float(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        *stamps(),
    )
    op.create_index("ix_agent_attempts_owner_id", "agent_attempts", ["owner_id"])
    op.create_index("ix_agent_attempts_run_id", "agent_attempts", ["run_id"])

    op.create_table(
        "agent_interviews",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("owner_id", sa.String(100), nullable=False),
        sa.Column("run_id", sa.String(36), sa.ForeignKey("agent_runs.id"), nullable=False),
        sa.Column("questions", postgresql.JSONB(), nullable=False),
        sa.Column("answers", postgresql.JSONB(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        *stamps(),
    )
    op.create_index("ix_agent_interviews_owner_id", "agent_interviews", ["owner_id"])
    op.create_index("ix_agent_interviews_run_id", "agent_interviews", ["run_id"])

    op.create_table(
        "autolink_rejections",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("owner_id", sa.String(100), nullable=False),
        sa.Column("session_pk", sa.String(36), sa.ForeignKey("agent_sessions.id"), nullable=False),
        sa.Column("task_id", sa.String(36), sa.ForeignKey("tasks.id"), nullable=False),
        sa.Column("evidence_revision", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("session_pk", "task_id", name="uq_autolink_rejection"),
    )
    op.create_index("ix_autolink_rejections_owner_id", "autolink_rejections", ["owner_id"])
    op.create_index("ix_autolink_rejections_session_pk", "autolink_rejections", ["session_pk"])
    op.create_index("ix_autolink_rejections_task_id", "autolink_rejections", ["task_id"])

    op.add_column("agent_sessions", sa.Column("run_id", sa.String(36), sa.ForeignKey("agent_runs.id"), nullable=True))
    op.create_index("ix_agent_sessions_run_id", "agent_sessions", ["run_id"])
    for table in ("agent_commands", "agent_turns", "agent_questions"):
        op.create_foreign_key(f"{table}_run_id_fkey", table, "agent_runs", ["run_id"], ["id"])
        op.create_index(f"ix_{table}_run_id", table, ["run_id"])

    for column in (
        sa.Column("surf", sa.String(100), nullable=False, server_default=""),
        sa.Column("incarnation", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("provider_turn_id", sa.String(200), nullable=False, server_default=""),
        sa.Column("hold_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("remote_answerable", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("answer_op_id", sa.String(100), nullable=True),
        sa.Column("detail", sa.String(300), nullable=False, server_default=""),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
    ):
        op.add_column("agent_questions", column)


def downgrade():
    for column in ("resolved_at", "detail", "answer_op_id", "remote_answerable", "hold_until",
                   "provider_turn_id", "incarnation", "surf"):
        op.drop_column("agent_questions", column)
    for table in ("agent_questions", "agent_turns", "agent_commands"):
        op.drop_index(f"ix_{table}_run_id", table_name=table)
        op.drop_constraint(f"{table}_run_id_fkey", table, type_="foreignkey")
    op.drop_index("ix_agent_sessions_run_id", table_name="agent_sessions")
    op.drop_column("agent_sessions", "run_id")
    op.drop_table("autolink_rejections")
    op.drop_table("agent_interviews")
    op.drop_table("agent_attempts")
    op.drop_table("agent_run_plans")
    op.drop_table("agent_runs")
    op.drop_constraint("tasks_execution_target_id_fkey", "tasks", type_="foreignkey")
    op.drop_table("execution_targets")
